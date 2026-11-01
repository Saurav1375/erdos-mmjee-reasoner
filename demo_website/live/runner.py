"""Run ONE question through the frozen final framework, reporting each stage as it happens.

Every step calls the project's own code (nothing in src/ is changed):
  transcribe / translate / tag   the same requests as rag/tagger.py (greedy, seed 0)
  retrieval                      rag/retrieve.py search + query_sims on the kb_v3 knowledge base
  solve                          pipeline/solve.py solve_pot (code sandbox, adaptive budget, CoT fallback)
  critique + correct             agents/loop.py critic_phase / corrector_phase, round by round
  verify                         the frozen verifier (verifier/v3.py predict_pool), select, gate_choices
The only differences from the batch pipeline: one question at a time, the two models are
separate servers (no model swap), outputs stay in memory, and the demo has its own caches
(live/cache.sqlite) so nothing is written into the project.

``salt=None`` uses the frozen seeds (exactly the saved run's requests: test_parity.py replays
them from the saved cache); a salt gives a fresh, different set of samples.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

LIVE = Path(__file__).resolve().parent
SITE = LIVE.parent
PROJ = SITE.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(SITE / "_build"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MMJEE_EMBED_DEVICE", "cpu")
os.chdir(PROJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mmjee_reasoner.pipeline.solve as solve_mod  # noqa: E402
from mmjee_reasoner.agents.loop import Chain, chain_summary, critic_phase, corrector_phase, new_solution  # noqa: E402
from mmjee_reasoner.config import load_config  # noqa: E402
from mmjee_reasoner.data.adapter import adapter_columns  # noqa: E402
from mmjee_reasoner.data.dataset import load_questions  # noqa: E402
from mmjee_reasoner.embed import embed_texts  # noqa: E402
from mmjee_reasoner.llm import GenRequest, user_message  # noqa: E402
from mmjee_reasoner.llm.cache import GenerationCache  # noqa: E402
from mmjee_reasoner.llm.engine import CachedEngine  # noqa: E402
from mmjee_reasoner.pipeline.common import load_candidates, relative_image  # noqa: E402
from mmjee_reasoner.prompts.templates import (  # noqa: E402
    TAGGER_PROMPT, TRANSCRIBE_PROMPT, TRANSLATE_PROMPT, format_exemplars)
from mmjee_reasoner.rag.kb import kb_embeddings, load_kb, load_problem_vecs  # noqa: E402
from mmjee_reasoner.rag.retrieve import hybrid_weight, query_sims, search  # noqa: E402
from mmjee_reasoner.rag.tagger import parse_tag  # noqa: E402
from mmjee_reasoner.sandbox.executor import ExecCache  # noqa: E402
from mmjee_reasoner.scoring import is_correct  # noqa: E402
from mmjee_reasoner.verifier.train import load_verifier, primary_selection  # noqa: E402
from mmjee_reasoner.verifier.v3 import gate_choices, prepare_pool  # noqa: E402

from records import question_record, short_id  # noqa: E402

log = logging.getLogger("live")
CONFIGS = ("configs/v3.yaml", "configs/v3_test2026.yaml", "configs/v3_final.yaml")
EXEC_CACHE = LIVE / "exec_cache.sqlite"
UPLOADS = LIVE / "uploads"
MODES = {"full": {"n_samples": 8, "max_rounds": None},    # None: the frozen max_rounds (3)
         "quick": {"n_samples": 4, "max_rounds": 1}}

# solve_pot opens the code-execution cache inside the project's artifacts; use the demo's own.
solve_mod.ExecCache = lambda _path: ExecCache(str(EXEC_CACHE))


def _jsonable(records: list[dict]) -> list[dict]:
    """Same round trip as writing and re-reading candidates.jsonl."""
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(type(x))
    return [json.loads(json.dumps(r, ensure_ascii=False, default=conv)) for r in records]


class Framework:
    """Everything that is loaded once: config, questions, knowledge base, verifier, baselines."""

    def __init__(self):
        self.cfg = load_config(*CONFIGS)
        self.art = Path(self.cfg["paths"]["artifacts"])
        self.questions = load_questions(self.cfg, "test").set_index("uid", drop=False)
        self.entries, index, self.kb_meta = load_kb(self.cfg)
        self.kb_vecs = kb_embeddings(index)
        self.kb_problem = load_problem_vecs(self.cfg) if hybrid_weight(self.cfg) < 1.0 else None
        self.kb = {e["kb_id"]: e for e in self.entries}
        self.verifier = load_verifier(self.cfg)
        self.selection = primary_selection(self.cfg)["selection"]
        self.cot = load_candidates(self.cfg, "cot_ext")
        self.sc_n = 18
        self.lock = threading.Lock()   # one live run at a time

    # ------------------------------------------------------------------ questions
    def exam_question(self, uid: str) -> dict:
        return self.questions.loc[uid].to_dict()

    def upload_question(self, image_png: bytes, subject: str, qtype: str, language: str,
                        diagram: bool) -> dict:
        """A new question from an uploaded image (no answer key)."""
        sha = hashlib.sha256(image_png).hexdigest()[:12]
        UPLOADS.mkdir(exist_ok=True)
        path = UPLOADS / f"{sha}.png"
        if not path.exists():
            path.write_bytes(image_png)
        art_abs = self.art.resolve()
        rel = os.path.relpath(path, art_abs)
        uid = f"upload_{sha}"
        placeholder = {"Numerical": "0", "MCQ-Multiple": "A"}.get(qtype, "A")
        return {"uid": uid, "question_id": uid, "twin_key": uid, "year": 0, "paper": "new",
                "language": language, "subject": subject, "question_type": qtype,
                "requires_image": bool(diagram), "answer": placeholder, "split": "test",
                "image_path": str(self.art / rel), "question_text": None,
                **adapter_columns(qtype, placeholder), "_upload": True}

    # ------------------------------------------------------------------ helpers
    def _gen_one(self, engine, prompt: str, sampling: dict, image: str | None, kind: str,
                 uid: str) -> str:
        """The request rag/tagger.py makes for one question (greedy, seed 0)."""
        req = GenRequest(messages=[user_message(prompt, image)],
                         sampling={**sampling, "seed": 0}, meta={"uid": uid, "kind": kind})
        return engine.generate([req])[0].text.strip()

    def retrieve(self, cfg: dict, q: dict, query: str, transcript: str | None) -> dict:
        """rag/retrieve.py exemplars_for_questions, for one question (no log file)."""
        rag = cfg["rag"]
        qv = embed_texts([query], rag["embedding_model"])[0]
        qx = None
        if hybrid_weight(cfg) < 1.0 and transcript is not None:
            qx = embed_texts([transcript], rag["embedding_model"])[0]
        sims = (query_sims(cfg, qv, qx, self.kb_vecs, self.kb_problem)
                if self.kb_problem is not None else None)
        hits = search(qv, self.kb_vecs, self.entries, q["subject"] if rag["same_subject_only"]
                      else None, q["twin_key"], rag["top_k"], self.kb_meta["threshold"]["value"],
                      sims=sims)
        examples = [{"problem": self.entries[i]["problem"], "solution": self.entries[i]["solution"]}
                    for i, _ in hits]
        return {"text": format_exemplars(examples),
                "kb_ids": [self.entries[i]["kb_id"] for i, _ in hits],
                "sims": [s for _, s in hits]}

    def finalize(self, cfg: dict, chains: list[Chain], src: list[dict], q: dict) -> list[dict]:
        """pipeline/correct.py finalize, in memory (one corrected candidate per chain)."""
        base_of = {r["sample"]: r for r in src}
        records = []
        for c in chains:
            sample = int(c.cid.rsplit("|s", 1)[1])
            base = base_of[sample]
            final = c.solutions[-1]
            summ = chain_summary(c)
            from_original = all(s.get("correction_failed") or s["text"] == c.solutions[0]["text"]
                                for s in c.solutions[1:]) and final["text"] == c.solutions[0]["text"]
            own_tokens = int(base["completion_tokens"]) if from_original else int(
                final["stats"].get("completion_tokens", base["completion_tokens"]))
            own_finish = base.get("finish") if from_original else final["stats"].get("finish")
            rec = {k: base[k] for k in ("question_id", "twin_key", "year", "paper", "language",
                                        "subject", "question_type", "requires_image", "answer")}
            rec.update({
                "uid": c.uid, "exp": "v3_full", "sample": sample, "source": "corrected",
                "text": final["text"], "pred": final["answer"],
                "correct": is_correct(final["answer"], q),
                "orig_pred": c.solutions[0]["answer"], "orig_correct": bool(base["correct"]),
                "n_steps": final["n_steps"],
                "n_code_blocks": final["stats"].get("n_code_blocks", base.get("n_code_blocks", 0)),
                "n_code_ok": final["stats"].get("n_code_ok", base.get("n_code_ok", 0)),
                "n_code_err": final["stats"].get("n_code_err", base.get("n_code_err", 0)),
                "used_fallback": bool(base.get("used_fallback", False)),
                "correction_failed_last": bool(final.get("correction_failed")),
                "calls": int(base["calls"]) + summ["extra_calls"],
                "prompt_tokens": int(base["prompt_tokens"]) + sum(
                    x["prompt_tokens"] for x in c.critiques) + sum(
                    s["attempt"]["prompt_tokens"] for s in c.solutions[1:]),
                "completion_tokens": int(base["completion_tokens"]) + summ["extra_completion_tokens"],
                "sol_tokens": own_tokens, "finish": own_finish,
                "critic": "critic_cross",
                "exemplars": base.get("exemplars", []),
                **{f"corr_{k}": v for k, v in summ.items()},
            })
            records.append(rec)
        return records

    # ------------------------------------------------------------------ the run
    def run(self, q: dict, solver, critic, emit: Callable[[dict], None], mode: str = "full",
            salt: str | None = None) -> dict:
        """One question through every stage; returns the page record (data.js shape)."""
        t0 = time.time()
        cfg = copy.deepcopy(self.cfg)
        if salt is not None:
            cfg["seed"] = f"{cfg['seed']}-{salt}"
        m = MODES[mode]
        if m["max_rounds"] is not None:
            cfg["correction"]["max_rounds"] = m["max_rounds"]
        uid, upload = q["uid"], bool(q.get("_upload"))
        image = relative_image(cfg, q["image_path"])
        stage = lambda i, status, **kw: emit({"type": "stage", "i": i, "status": status,  # noqa: E731
                                              "t": round(time.time() - t0, 1), **kw})

        # 1. read: transcription (+ English translation of a Hindi transcription)
        stage(0, "start")
        rag = cfg["rag"]
        tx = self._gen_one(solver, TRANSCRIBE_PROMPT, rag["transcribe_sampling"], image,
                           "transcribe", uid)
        tx_en = tx
        if q["language"] == "Hindi":
            tx_en = self._gen_one(solver, TRANSLATE_PROMPT.format(text=tx),
                                  rag["transcribe_sampling"], None, "translate", uid)
        stage(0, "done", transcript=tx_en)

        # 2. look up: concept tag + retrieval
        stage(1, "start")
        tag = parse_tag(self._gen_one(solver, TAGGER_PROMPT, rag["tag_sampling"], image, "tag",
                                      uid))
        ex = self.retrieve(cfg, q, tag["query"], tx_en)
        stage(1, "done", tag={k: tag.get(k) for k in ("subject", "topic", "method", "query")},
              retrieval=[{"id": k, "sim": round(float(s), 4)} for k, s in zip(ex["kb_ids"], ex["sims"])])

        # 3. solve: n samples with the code sandbox
        stage(2, "start", n=m["n_samples"])
        df = pd.DataFrame([q])
        src = _jsonable(solve_mod.solve_pot(cfg, {"name": "v3_pot", "n_samples": m["n_samples"]},
                                            solver, df, {uid: ex}, {uid: tx_en}))
        stage(2, "done", samples=[{"s": r["sample"], "pred": r["pred"],
                                   "ok": None if upload else bool(r["correct"]),
                                   "blocks": r["n_code_blocks"], "tokens": r["completion_tokens"]}
                                  for r in src])

        # 4. critique and correct, round by round (critic and solver on separate servers)
        stage(3, "start")
        chains = []
        for r in src:
            ch = Chain(cid=f"{uid}|s{r['sample']}", uid=uid, language=r["language"],
                       question_type=r["question_type"], image=image,
                       seed_base=f"{cfg['seed']}|v3_full|{uid}|s{r['sample']}",
                       exemplars=r.get("exemplar_text") if isinstance(r.get("exemplar_text"), str)
                       else None)
            ch.solutions.append(new_solution(r["text"], r["question_type"],
                                             {k: r[k] for k in ("calls", "prompt_tokens",
                                                                "completion_tokens")}))
            chains.append(ch)
        exec_cache = ExecCache(str(EXEC_CACHE))
        for rnd in range(cfg["correction"]["max_rounds"]):
            critic_phase(chains, critic, cfg, rnd)
            emit({"type": "critic", "round": rnd,
                  "chains": [{"s": int(c.cid.rsplit("|s", 1)[1]),
                              "error": c.critiques[-1]["has_error"], "step": c.critiques[-1]["step"]}
                             for c in chains if len(c.critiques) == rnd + 1]})
            corrector_phase(chains, solver, cfg, rnd, exec_cache)
            emit({"type": "corrector", "round": rnd,
                  "chains": [{"s": int(c.cid.rsplit("|s", 1)[1]), "answer": c.current["answer"]}
                             for c in chains if len(c.solutions) == rnd + 2]})
            if all(c.done for c in chains):
                break
        for c in chains:
            if not c.done:
                c.done, c.stop_reason = True, "max_rounds"
        corr = _jsonable(self.finalize(cfg, chains, src, q))
        stage(3, "done")

        # 5. verify and pick over the pool (originals + corrected, as in the evaluation)
        stage(4, "start")
        pool = pd.DataFrame(src + corr)
        scores = self.verifier.predict_pool(pool, cfg, LIVE / "emb_cache")
        gate = gate_choices(prepare_pool(pool), scores, float(self.verifier.delta)).set_index(
            ["uid", "sample"])
        pool = pool.assign(score=scores)
        pot = pd.DataFrame(src)
        cot = None if upload else self.cot[self.cot["uid"] == uid]
        rid = f"NEW-{uid[-6:]}" if upload else short_id(uid)
        image_url = (f"live/uploads/{Path(q['image_path']).name}" if upload
                     else f"img/{short_id(uid)}.webp")
        rec, _ = question_record(q, rid, image_url, pool.reset_index(drop=True), pot,
                                 {c.cid: asdict(c) for c in chains}, gate, tx_en, tag, ex,
                                 self.kb, self.selection, cot=cot, sc_n=self.sc_n,
                                 has_key=not upload)
        rec["live"] = {"mode": mode, "seconds": round(time.time() - t0, 1),
                       "salt": salt, "n_samples": m["n_samples"],
                       "max_rounds": cfg["correction"]["max_rounds"],
                       "tokens": int(pool["completion_tokens"].sum()) if "completion_tokens" in pool else None,
                       "recorded": time.strftime("%Y-%m-%d %H:%M")}
        stage(4, "done")
        stage(5, "done")
        return rec


def cached_engine(name: str, cache_path: Path, factory=None, offline: bool = False) -> CachedEngine:
    return CachedEngine(name, GenerationCache(cache_path), factory=factory, offline=offline)
