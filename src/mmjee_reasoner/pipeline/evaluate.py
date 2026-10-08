"""Evaluation on the 2025 held-out set: SOP Table I and all analyses.

Runs fully offline from the candidate files written by the generation stages.
Every correctness value comes from the upstream scorer (stored in the
candidate records at generation time).

Row kinds (``eval.table1`` in ``configs/base.yaml``):

* ``mean``       -- Pass@1 = mean correctness over the N samples (one "run" per
  sample index, as the base paper averages k runs).
* ``sc_matched`` -- self-consistency (majority vote) over the first n CoT samples,
  n chosen so that CoT tokens match the full framework's generated tokens.
* ``verifier``   -- best-of-N with the learned verifier over the pool
  {original PoT samples} U {their corrected versions}.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mmjee_reasoner.config import artifacts_dir, load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.pipeline.common import candidates_path, load_candidates
from mmjee_reasoner.scoring.metrics import (
    bootstrap_ci,
    breakdown,
    mcnemar,
    paired_bootstrap,
    pass_at_1,
    per_question_mean,
)
from mmjee_reasoner.verifier.model import (
    SELECTION_METHODS,
    build_features,
    classification_metrics,
    selection_accuracy,
)

log = logging.getLogger(__name__)


def eval_dir(cfg: dict) -> Path:
    return artifacts_dir(cfg, "eval", "x").parent


class Evaluator:
    def __init__(self, cfg: dict, suffix: str = ""):
        self.cfg = cfg
        self.suffix = suffix
        self.questions = load_questions(cfg, "test")
        self.clusters = self.questions.set_index("uid")["twin_key"]  # EN/HI pairs
        self._cands: dict[str, pd.DataFrame] = {}
        self._verifiers: dict | None = None

    # ------------------------------------------------------------------ data
    def has(self, exp: str) -> bool:
        return candidates_path(self.cfg, exp, self.suffix).exists()

    def cands(self, exp: str) -> pd.DataFrame:
        if exp not in self._cands:
            df = load_candidates(self.cfg, exp, self.suffix)
            df = df[df["uid"].isin(set(self.questions["uid"]))].reset_index(drop=True)
            self._cands[exp] = df
        return self._cands[exp]

    def pool(self, exp: str) -> pd.DataFrame:
        """Verifier pool: corrected finals + the original candidates they came from."""
        e = load_experiment(exp)
        df = self.cands(exp)
        if e.get("kind") == "correct":
            src = self.cands(e["source"]).copy()
            df = pd.concat([src, df], ignore_index=True)
        return df.reset_index(drop=True)

    def tag_tokens_per_question(self) -> float:
        from mmjee_reasoner.pipeline.common import read_jsonl
        from mmjee_reasoner.rag.tagger import tags_path

        p = tags_path(self.cfg)
        if not p.exists():
            return 0.0
        tags = [r for r in read_jsonl(p) if r["uid"] in set(self.questions["uid"])]
        return float(np.mean([r["completion_tokens"] for r in tags])) if tags else 0.0

    # --------------------------------------------------------------- compute
    def cost(self, exp: str, kind: str) -> dict:
        """Generated tokens and model calls per question for a configuration."""
        df = self.cands(exp)
        e = load_experiment(exp)
        per_q = df.groupby("uid")[["completion_tokens", "calls"]]
        if kind == "mean":  # one trajectory
            tok, calls = per_q.mean().mean()
        else:  # all trajectories of the pool are generated
            tok, calls = per_q.sum().mean()
        rag = e.get("rag") or (e.get("kind") == "correct" and
                               load_experiment(e["source"]).get("rag"))
        if rag and kind != "mean":
            tok += self.tag_tokens_per_question()
            calls += 1
        return {"tokens": float(tok), "calls": float(calls)}

    # ------------------------------------------------------------- verifier
    def verifiers(self) -> dict:
        if self._verifiers is None:
            from mmjee_reasoner.verifier.train import load_verifier, primary_selection, verifier_dir

            report = json.loads((verifier_dir(self.cfg) / "report.json").read_text())
            vs = {}
            for c in report["configs"]:
                vs[(c["model"], c["feature_set"])] = load_verifier(
                    self.cfg, c["model"], c["feature_set"])
            self._verifiers = {"all": vs, "primary": primary_selection(self.cfg)}
        return self._verifiers

    def verifier_scores(self, df: pd.DataFrame, which=None) -> np.ndarray:
        from mmjee_reasoner.verifier.train import verifier_dir

        v = self.verifiers()
        prim = v["primary"]
        key = which or (prim["model"], prim["feature_set"])
        # Agreement features are computed within each source's sub-pool (8 originals,
        # 8 corrected) so they match the 8-sample pools the verifier was trained on.
        fm = build_features(df, self.cfg, cache_dir=verifier_dir(self.cfg) / "emb_cache",
                            pool_cols=("uid", "source"))
        return v["all"][key].predict(fm)

    # ------------------------------------------------------------------ rows
    def row_scores(self, spec: dict, sc_n: int | None = None) -> tuple[pd.Series, dict]:
        kind, exp = spec["kind"], spec["exp"]
        if kind == "mean":
            df = self.cands(exp)
            return per_question_mean(df), {"pass_at_1": pass_at_1(df)}
        if kind == "sc_matched":
            df = self.cands(exp)
            avail = int(df.groupby("uid")["sample"].nunique().min())
            n = min(sc_n or avail, avail)
            sub = df[df["sample"] < n]
            _, ser = selection_accuracy(sub, None, "majority")
            return ser, {"n_samples": n, "n_needed": sc_n, "available": avail,
                         "insufficient": bool(sc_n and sc_n > avail)}
        if kind == "verifier":
            pool = self.pool(exp)
            scores = self.verifier_scores(pool)
            method = self.verifiers()["primary"]["selection"]
            _, ser = selection_accuracy(pool, scores, method)
            extra = {"selection": method, "pool_size": int(pool.groupby("uid").size().mean())}
            for m in SELECTION_METHODS:
                extra[f"sel_{m}"] = selection_accuracy(pool, scores, m)[0] * 100
            return ser, extra
        raise ValueError(kind)

    def matched_samples(self, match_exp: str) -> int | None:
        if not (self.has(match_exp) and self.has("cot")):
            return None
        full_tok = self.cost(match_exp, "verifier")["tokens"]
        cot_tok = self.cands("cot")["completion_tokens"].mean()
        return int(math.ceil(full_tok / cot_tok))

    # ------------------------------------------------------------- analyses
    def correction_analysis(self, exp: str) -> dict:
        df = self.cands(exp)
        wrong, right = df[~df["orig_correct"]], df[df["orig_correct"]]
        flagged = df["corr_critic_flagged_first"].astype(bool)
        return {
            "n_chains": len(df),
            "acc_before": float(df["orig_correct"].mean() * 100),
            "acc_after": float(df["correct"].mean() * 100),
            # error presence (detection) on originally-wrong solutions, cf. paper EP
            "detect_rate": float(flagged[~df["orig_correct"]].mean() * 100) if len(wrong) else 0,
            "false_alarm_rate": float(flagged[df["orig_correct"]].mean() * 100) if len(right)
            else 0,
            # correction success: wrong -> right among detected (cf. paper EP->EC)
            "fix_rate_of_detected": float(
                wrong[flagged[~df["orig_correct"]]]["correct"].mean() * 100)
            if flagged[~df["orig_correct"]].any() else 0.0,
            "wrong_to_right": int((~df["orig_correct"] & df["correct"]).sum()),
            "right_to_wrong": int((df["orig_correct"] & ~df["correct"]).sum()),
            "answer_changed": float(df["corr_answer_changed"].mean() * 100),
            "mean_rounds": float(df["corr_rounds"].mean()),
            "stop_reasons": df["corr_stop_reason"].value_counts().to_dict(),
            "critic_unparsed": int(df["corr_critic_unparsed"].sum()),
        }

    def verifier_test_quality(self, exp: str) -> dict:
        """AUC/F1 of every verifier variant on the 2025 pool (report only, no selection)."""
        pool = self.pool(exp)
        y = pool["correct"].astype(int).to_numpy()
        out = {}
        for (model, fs), tv in self.verifiers()["all"].items():
            p = self.verifier_scores(pool, (model, fs))
            m = classification_metrics(y, p, tv.threshold)
            for meth in ("max_prob", "weighted_vote"):
                m[f"sel_{meth}"] = selection_accuracy(pool, p, meth)[0] * 100
            out[f"{model}/{fs}"] = m
        out["majority"] = selection_accuracy(pool, None, "majority")[0] * 100
        out["oracle"] = selection_accuracy(pool, None, "oracle")[0] * 100
        return out

    def sc_curve(self, exp: str = "cot") -> dict:
        df = self.cands(exp)
        n_max = int(df.groupby("uid")["sample"].nunique().min())
        return {n: selection_accuracy(df[df["sample"] < n], None, "majority")[0] * 100
                for n in range(1, n_max + 1)}

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg
        rows, scores = [], {}
        table = cfg["eval"]["table1"]
        match_specs = [s for s in table if s["kind"] == "sc_matched"]
        sc_n = self.matched_samples(match_specs[0]["match"]) if match_specs else None
        baseline = None
        for spec in table:
            exp = spec["exp"]
            if not self.has(exp) or (spec["kind"] == "verifier" and not self._verifier_ready()):
                rows.append({"row": spec["row"], "available": False})
                continue
            if spec["kind"] == "verifier" and load_experiment(exp).get("kind") == "correct":
                if not self.has(load_experiment(exp)["source"]):
                    rows.append({"row": spec["row"], "available": False})
                    continue
            ser, extra = self.row_scores(spec, sc_n)
            scores[spec["row"]] = ser
            qnum = self.questions.set_index("uid").loc[ser.index, "question_type"] == "Numerical"
            row = {
                "row": spec["row"], "available": True, "kind": spec["kind"], "exp": exp,
                "overall": float(ser.mean() * 100),
                "numerical": float(ser[qnum.to_numpy()].mean() * 100),
                "ci95": bootstrap_ci(ser, cfg["eval"]["bootstrap"], cfg["seed"], self.clusters),
                "n_questions": len(ser),
                "cost": self.cost(exp, "mean" if spec["kind"] == "mean" else "verifier")
                if spec["kind"] != "sc_matched" else None,
                "breakdown": breakdown(ser, self.questions),
                **extra,
            }
            if spec["kind"] == "sc_matched":
                cot_tok = self.cands(exp)["completion_tokens"].mean()
                row["cost"] = {"tokens": float(cot_tok * extra["n_samples"]),
                               "calls": float(extra["n_samples"])}
            if baseline is None:
                baseline = ser
            else:
                row["vs_baseline"] = paired_bootstrap(ser, baseline, cfg["eval"]["bootstrap"],
                                                      cfg["seed"], self.clusters)
            rows.append(row)

        results = {"table1": rows, "n_test_questions": len(self.questions)}
        sc_row = next((r for r in rows if r.get("kind") == "sc_matched"), None)
        full_row = next((r for r in rows if r["row"] == "Full framework" and r["available"]),
                        None)
        if sc_row and full_row:
            results["full_vs_sc_matched"] = paired_bootstrap(
                scores["Full framework"], scores[sc_row["row"]], cfg["eval"]["bootstrap"],
                cfg["seed"], self.clusters)
            results["full_vs_sc_matched_mcnemar"] = mcnemar(
                scores["Full framework"], scores[sc_row["row"]])
        if self.has("cot"):
            results["sc_curve"] = self.sc_curve("cot")
        results["correction"] = {e: self.correction_analysis(e)
                                 for e in ("pot_corr_same", "pot_corr_cross", "full")
                                 if self.has(e)}
        if self._verifier_ready():
            results["verifier_test"] = {e: self.verifier_test_quality(e)
                                        for e in ("pot", "pot_corr_cross", "full")
                                        if self.has(e) and self._pool_ready(e)}
        extra_means = {}
        for e in ("pot_rag",):
            if self.has(e):
                ser = per_question_mean(self.cands(e))
                extra_means[e] = {"overall": float(ser.mean() * 100),
                                  "pass_at_1": pass_at_1(self.cands(e))}
        results["extra"] = extra_means
        if self.has("cot_train") and self.has("cot"):  # contamination check (paper Table 7)
            from mmjee_reasoner.pipeline.common import load_candidates as _lc

            tr = _lc(self.cfg, "cot_train", self.suffix)
            te = self.cands("cot")
            te0 = te[te["sample"] == 0]          # same protocol: 1 sample per question
            results["contamination"] = {
                "train_by_year": {str(y): float(g["correct"].mean() * 100)
                                  for y, g in tr.groupby("year")},
                "train_2019_2024": float(tr["correct"].mean() * 100),
                "test_2025_sample0": float(te0["correct"].mean() * 100),
                "delta_2025_minus_train": float((te0["correct"].mean()
                                                 - tr["correct"].mean()) * 100),
                "n_train": len(tr), "n_test": len(te0),
            }
        if self.has("repro_qwen25"):  # reproduction of a base-paper number
            exp = load_experiment("repro_qwen25")
            df = self.cands("repro_qwen25")
            ser = per_question_mean(df)
            results["reproduction"] = {
                "model": self.cfg["models"][exp["model"]]["hf_id"],
                "pass_at_1": pass_at_1(df), "n_questions": int(len(ser)),
                "ci95": bootstrap_ci(ser, cfg["eval"]["bootstrap"], cfg["seed"], self.clusters),
                "paper_pass1": exp["paper_pass1"], "paper_std": exp["paper_std"],
                "breakdown": breakdown(ser, self.questions),
            }
        per_q = pd.DataFrame(scores)
        per_q.index.name = "uid"
        out_dir = eval_dir(cfg)
        per_q.to_csv(out_dir / f"per_question{self.suffix}.csv")
        (out_dir / f"results{self.suffix}.json").write_text(json.dumps(results, indent=2,
                                                                       default=_json_default))
        log.info("wrote %s", out_dir / f"results{self.suffix}.json")
        return results

    def _verifier_ready(self) -> bool:
        from mmjee_reasoner.verifier.train import verifier_dir

        return (verifier_dir(self.cfg) / "report.json").exists()

    def _pool_ready(self, exp: str) -> bool:
        e = load_experiment(exp)
        return e.get("kind") != "correct" or self.has(e["source"])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def run_evaluate_stage(cfg: dict, limit: int | None = None) -> dict:
    from mmjee_reasoner.pipeline.common import set_subset, subset_suffix

    if limit is not None:
        set_subset(cfg, limit=limit)
    return Evaluator(cfg, subset_suffix(cfg)).run()
