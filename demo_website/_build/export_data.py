"""Export the real 2026 runs of the frozen final framework for the demo website.

Reads only saved outputs (no model calls, CPU only) and writes:
  ../data.js        window.DEMO = {...}: every 2026 test question with its transcript, concept
                    tag, retrieval, one solver trajectory, its correction chain, the 16-answer
                    pool with the frozen verifier's scores, and what each method answers;
                    plus the headline tables (2025 + 2026) and the Step-5 diagnostics.
  ../img/*.webp     the question images (compressed copies).

Every number is computed by the project's own code (Evaluator, verifier, select, gate_choices),
and the script checks that it reproduces the published 2026 Table I rows before writing.

usage (from the repo root):
  .venv/bin/python demo_website/_build/export_data.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
PROJ = SITE.parent
sys.path.insert(0, str(PROJ / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.chdir(PROJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from mmjee_reasoner.config import load_config  # noqa: E402
from mmjee_reasoner.pipeline.common import read_jsonl  # noqa: E402
from mmjee_reasoner.pipeline.evaluate import Evaluator  # noqa: E402
from mmjee_reasoner.verifier.v3 import gate_choices, prepare_pool  # noqa: E402

sys.path.insert(0, str(HERE))
from records import question_record, short_id  # noqa: E402

logging.disable(logging.WARNING)


def pyval(x):
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main() -> None:
    cfg = load_config("configs/v3.yaml", "configs/v3_test2026.yaml",
                      "configs/v3_final.yaml", "configs/v3_eval.yaml")
    art = Path(cfg["paths"]["artifacts"])
    ev = Evaluator(cfg)
    qs = ev.questions.set_index("uid")
    res26 = json.loads((art / "eval" / "results.y2026.v3.json").read_text())
    res25 = json.loads((art / "eval" / "results.v3.json").read_text())
    res_p2 = json.loads((PROJ / "artifacts" / "eval" / "results.json").read_text())
    v4b = json.loads((PROJ / "artifacts_v4b" / "analysis" / "v4b.json").read_text())
    kb_meta = json.loads((art / "rag" / "kb_v3" / "meta.json").read_text())
    tau = kb_meta["threshold"]["value"]

    # ---------------------------------------------------------------- pool + verifier
    pool = ev.pool("v3_full")
    scores = ev.verifier_scores(pool)
    prim = ev.verifiers()["primary"]
    model = ev.verifiers()["all"][(prim["model"], prim["feature_set"])]
    delta = float(model.delta)
    gate = gate_choices(prepare_pool(pool), scores, delta).set_index(["uid", "sample"])
    pool = pool.assign(score=scores)

    pot = ev.cands("v3_pot")
    cot = ev.cands("cot_ext")
    sc_n = next(r["n_samples"] for r in res26["table1"] if r["kind"] == "sc_matched")
    chains = {c["cid"]: c for c in json.loads(
        (art / "generations" / "v3_full" / "chains.y2026.json").read_text())}
    perq = pd.read_csv(art / "eval" / "per_question.y2026.v3.csv", index_col="uid")

    tx = {r["uid"]: r["text"] for r in read_jsonl(art / "rag" / "transcripts.jsonl")}
    tx.update({r["uid"]: r["text"] for r in read_jsonl(art / "rag" / "transcripts_en.jsonl")})
    tags = {r["uid"]: r for r in read_jsonl(art / "rag" / "tags.jsonl")}
    ret = {r["uid"]: r for r in read_jsonl(art / "rag" / "retrieval_test.y2026_kb_v3.jsonl")}
    kb = {r["kb_id"]: r for r in read_jsonl(art / "rag" / "kb_v3" / "kb.jsonl")}

    (SITE / "img").mkdir(exist_ok=True)
    out_q, check = [], {"final": [], "sc": [], "base": []}
    for uid in qs.index:
        m = qs.loc[uid]
        img = f"img/{short_id(uid)}.webp"
        src_img = art / "data" / "images" / f"{uid}.png"
        if src_img.exists() and not (SITE / img).exists():
            im = Image.open(src_img).convert("RGB")
            if im.width > 1000:
                im = im.resize((1000, round(im.height * 1000 / im.width)), Image.LANCZOS)
            im.save(SITE / img, "WEBP", quality=72, method=6)
        rec, chk = question_record(
            {**m.to_dict(), "uid": uid}, short_id(uid), img,
            pool[pool["uid"] == uid].reset_index(drop=True), pot[pot["uid"] == uid], chains, gate,
            tx.get(uid), tags.get(uid, {}), ret.get(uid, {}), kb, prim["selection"],
            cot=cot[cot["uid"] == uid], sc_n=sc_n)
        out_q.append(rec)
        for k in check:
            check[k].append(chk[k])

    # ---------------------------------------------------------------- reproduce Table I
    row = {r["row"]: r for r in res26["table1"]}
    got = {"final": 100 * np.mean(check["final"]), "sc": 100 * np.mean(check["sc"]),
           "base": 100 * np.mean(check["base"])}
    want = {"final": row["v3 framework (verifier v3)"]["overall"],
            "sc": row["Self-consistency (matched compute)"]["overall"],
            "base": row["Baseline (single-pass CoT, adaptive budget)"]["overall"]}
    for k in got:
        assert abs(got[k] - want[k]) < 1e-6, (k, got[k], want[k])
        assert abs(perq[row["v3 framework (verifier v3)"]["row"]].mean() * 100 - want["final"]) < 1e-6
    print("reproduced 2026 Table I:", {k: round(v, 2) for k, v in got.items()})

    def t1(res):
        return [{"row": r["row"], "acc": round(r["overall"], 2),
                 "ci": [round(x, 2) for x in r["ci95"]] if r.get("ci95") else None,
                 "tokens": round(r["cost"]["tokens"]) if r.get("cost") else None,
                 "calls": round(r["cost"]["calls"], 2) if r.get("cost") else None,
                 "n": r.get("n_samples"), "breakdown": r.get("breakdown")}
                for r in res["table1"]]

    data = {
        "questions": out_q,
        "tau": round(tau, 4), "delta": delta, "sc_n": sc_n,
        "verifier": prim,
        "kb": {"entries": kb_meta["entries"], "by_source": kb_meta["entries_by_source"],
               "coverage": kb_meta["threshold"]["coverage"],
               "precision": kb_meta["threshold"]["proxy_precision"]},
        "table": {"y2026": t1(res26), "y2025": t1(res25), "phase2": t1(res_p2)},
        "vs_sc": {"y2026": res26["full_vs_sc_matched"], "mcnemar": res26["full_vs_sc_matched_mcnemar"],
                  "y2025": res25.get("full_vs_sc_matched")},
        "sc_curve": res26["sc_curve"],
        "correction": res26["correction"]["v3_full"],
        "decomposition": v4b["decomposition"],
        "discrimination": v4b["discrimination"],
        "errors": v4b["errors_2026"],
    }
    js = "window.DEMO = " + json.dumps(data, ensure_ascii=False, default=pyval) + ";\n"
    js = js.replace("\ufffd", "")  # broken characters in two saved model outputs
    (SITE / "data.js").write_text(js, encoding="utf-8")
    print(f"wrote data.js ({len(js) / 1e6:.2f} MB), {len(out_q)} questions")


if __name__ == "__main__":
    main()
