"""CLI walk-through of the full framework on one 2025 test question (CPU only, no model calls).

Reads the saved Phase-2 full-run outputs (artifacts/) and prints, for one question:
retrieval -> one solver trajectory (code + outputs) -> its correction chain -> the 16-answer
pool with verifier scores -> final pick vs. majority vote and the single-answer baseline.

usage:
  .venv/bin/python scripts/demo.py                  # a question the framework fixes (vs baseline)
  .venv/bin/python scripts/demo.py --pick loss      # one it gets wrong
  .venv/bin/python scripts/demo.py --pick random --type Numerical --lang Hindi
  .venv/bin/python scripts/demo.py --uid <uid> --sample 3 --full-text
  .venv/bin/python scripts/demo.py --table          # SOP Table I only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from mmjee_reasoner.config import load_config  # noqa: E402
from mmjee_reasoner.pipeline.common import read_jsonl  # noqa: E402
from mmjee_reasoner.verifier.features import answer_key  # noqa: E402

W = min(shutil.get_terminal_size((110, 40)).columns, 110)
TTY = sys.stdout.isatty()
B, D, G, R, Y, X = (("\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m") if TTY
                    else ("",) * 6)


def head(title: str) -> None:
    print(f"\n{B}{'─' * 3} {title} {'─' * max(3, W - len(title) - 5)}{X}")


def mark(ok) -> str:
    return f"{G}✓ right{X}" if bool(ok) else f"{R}✗ wrong{X}"


def wrap(text: str, indent: str = "  ", limit: int | None = 1200) -> str:
    text = text.strip()
    if limit and len(text) > limit:
        text = text[: limit // 2] + f"\n{D}  [... {len(text) - limit} characters cut ...]{X}\n" + text[-limit // 2:]
    out = []
    for line in text.splitlines():
        out += textwrap.wrap(line, W - len(indent), subsequent_indent="  ") or [""]
    return "\n".join(indent + ln for ln in out)


def table1(results: dict) -> None:
    head("SOP Table I  (JEE Advanced 2025, 190 questions)")
    print(f"  {'configuration':48s} {'overall':>8s} {'numerical':>10s} {'tokens/q':>9s}")
    for r in results["table1"]:
        if r.get("available"):
            print(f"  {r['row']:48s} {r['overall']:8.1f} {r['numerical']:10.1f} "
                  f"{r['cost']['tokens'] / 1000:8.1f}k")
    d = results["full_vs_sc_matched"]
    print(f"\n  full framework vs matched self-consistency: {d['diff']:+.1f} points "
          f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}], McNemar p = "
          f"{results['full_vs_sc_matched_mcnemar']['p']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid")
    ap.add_argument("--pick", choices=["win", "loss", "random"], default="win",
                    help="win: framework right, baseline wrong; loss: framework wrong")
    ap.add_argument("--type", choices=["Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"])
    ap.add_argument("--lang", choices=["English", "Hindi"])
    ap.add_argument("--sample", type=int, default=None, help="solver sample to show (0-7)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--full-text", action="store_true", help="do not shorten long solutions")
    ap.add_argument("--table", action="store_true", help="only print Table I")
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    cfg = load_config()                       # Phase-2 configuration, artifacts/
    art = Path(cfg["paths"]["artifacts"])
    results = json.loads((art / "eval" / "results.json").read_text())
    if args.table:
        table1(results)
        return

    import pandas as pd

    from mmjee_reasoner.pipeline.evaluate import Evaluator
    from mmjee_reasoner.verifier.train import primary_selection

    ev = Evaluator(cfg)
    q = ev.questions.set_index("uid")
    perq = pd.read_csv(art / "eval" / "per_question.csv", index_col="uid")
    full_col, base_col = "Full framework", "Baseline (single-pass CoT)"

    # ------------------------------------------------------------ choose the question
    if args.uid:
        uid = args.uid
    else:
        cand = q.index.to_series()
        if args.type:
            cand = cand[q["question_type"] == args.type]
        if args.lang:
            cand = cand[q["language"] == args.lang]
        if args.pick == "win":
            cand = cand[(perq.loc[cand, full_col] == 1) & (perq.loc[cand, base_col] < 0.5)]
        elif args.pick == "loss":
            cand = cand[perq.loc[cand, full_col] == 0]
        if cand.empty:
            sys.exit("no question matches these filters")
        uid = random.Random(args.seed).choice(sorted(cand))
    if uid not in q.index:
        sys.exit(f"unknown 2025 test uid: {uid}")
    meta = q.loc[uid]
    limit = None if args.full_text else 900

    head("1. Question")
    print(f"  {B}{uid}{X}")
    print(f"  {meta['subject']} · {meta['question_type']} · {meta['language']} · "
          f"{'needs the diagram' if meta['requires_image'] else 'text only'}")
    print(f"  correct answer (gold): {B}{meta['answer']}{X}")
    print(f"  image: {art / 'data' / 'images' / (uid + '.png')}")

    # ------------------------------------------------------------ retrieval
    head("2. Find similar solved problems (RAG)")
    ret = {r["uid"]: r for r in read_jsonl(art / "rag" / "retrieval_test.jsonl")}.get(uid, {})
    tau = json.loads((art / "rag" / "kb" / "meta.json").read_text())["threshold"]["value"]
    print(f"  concept tag (search key): {ret.get('query', '?')}")
    if ret.get("kb_ids"):
        for k, s in zip(ret["kb_ids"], ret["sims"]):
            print(f"  → {k}   similarity {s:.3f}  (threshold {tau:.3f})")
    else:
        print(f"  → nothing above the similarity threshold {tau:.3f}: solved without examples")

    # ------------------------------------------------------------ one solver trajectory
    pot = ev.cands("pot_rag")
    pot = pot[pot["uid"] == uid].set_index("sample").sort_index()
    s = args.sample if args.sample is not None else int(
        (pot["correct"].astype(bool) == False).idxmax() if (~pot["correct"].astype(bool)).any()  # noqa: E712
        else pot.index[0])
    t = pot.loc[s]
    head(f"3. Solver + code sandbox  (sample {s} of {len(pot)})")
    n_ok, n_blocks = int(t["n_code_ok"]), int(t["n_code_blocks"])
    print(f"  code blocks run: {n_ok}/{n_blocks}"
          f"{'   (fell back to plain reasoning)' if t['used_fallback'] else ''}"
          f"   tokens: {int(t['completion_tokens'])}   finish: {t['finish']}")
    text = re.sub(r"\n{3,}", "\n\n", str(t["text"]))
    print(wrap(text, limit=limit))
    print(f"  answer: {B}{t['pred'] or '(none)'}{X}   {mark(t['correct'])}")

    # ------------------------------------------------------------ correction chain
    head(f"4. Agentic correction of sample {s}  (critic: InternVL3.5-8B)")
    chains = json.loads((art / "generations" / "full" / "chains.json").read_text())
    ch = next((c for c in chains if c["uid"] == uid and c["cid"].endswith(f"|s{s}")), None)
    if ch is None:
        print("  (no chain found)")
    else:
        for r, crit in enumerate(ch["critiques"]):
            verdict = "ERROR at step " + str(crit["step"]) if crit["has_error"] else "CORRECT"
            forced = " (verdict forced)" if crit.get("forced_verdict") else ""
            print(f"  round {r + 1}: critic says {B}{verdict}{X}{forced}")
            if crit["has_error"] and crit.get("explanation"):
                print(wrap("why: " + crit["explanation"], "     ", limit=400))
            if r + 1 < len(ch["solutions"]):
                sol = ch["solutions"][r + 1]
                fail = " (fix failed, previous answer kept)" if sol.get("correction_failed") else ""
                print(f"     corrector's new answer: {B}{sol['answer'] or '(none)'}{X}{fail}")
        final = ch["solutions"][-1]["answer"]
        okf = ev.cands("full").set_index(["uid", "sample"]).loc[(uid, s), "correct"]
        print(f"  stopped: {ch['stop_reason']}   final answer of this chain: {B}{final}{X}  {mark(okf)}")

    # ------------------------------------------------------------ verifier over the pool
    head("5. Learned verifier over the 16-answer pool")
    pool = ev.pool("full")
    pool = pool[pool["uid"] == uid].reset_index(drop=True)
    scores = ev.verifier_scores(pool)
    prim = primary_selection(cfg)
    pick = int(scores.argmax())
    print(f"  verifier: {prim['model']} / {prim['feature_set']} features, rule {prim['selection']}")
    print(f"  {'':2s}{'origin':10s}{'sample':>7s}  {'answer':12s}{'P(correct)':>11s}")
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    for i in order:
        r = pool.loc[i]
        src = "fixed" if r["source"] == "corrected" else "original"
        star = f"{Y}▶{X} " if i == pick else "  "
        print(f"  {star}{src:10s}{int(r['sample']):7d}  {str(r['pred'] or '-'):12s}{scores[i]:11.3f}  "
              f"{mark(r['correct'])}")

    # ------------------------------------------------------------ summary
    head("6. What each method answers")
    cot = ev.cands("cot")
    cot = cot[cot["uid"] == uid].sort_values("sample")
    keys = [answer_key(p, meta["question_type"]) for p in cot["pred"].fillna("").astype(str)]
    n_sc = next((r["n_samples"] for r in results["table1"] if r.get("kind") == "sc_matched"), 20)
    vote = Counter(keys[:n_sc]).most_common(1)[0][0]
    vote_ok = cot.iloc[[keys.index(vote)]]["correct"].iloc[0]
    pool_keys = [answer_key(p, meta["question_type"]) for p in pool["pred"].fillna("").astype(str)]
    pv = Counter(pool_keys).most_common(1)[0][0]
    pv_ok = pool.loc[pool_keys.index(pv), "correct"]
    rows = [
        ("Single CoT answer (baseline, sample 0)", cot.iloc[0]["pred"], cot.iloc[0]["correct"]),
        (f"Majority vote of {n_sc} CoT answers (same compute)", vote, vote_ok),
        ("Majority vote of the 16-answer pool", pv, pv_ok),
        ("Full framework (verifier pick)", pool.loc[pick, "pred"], pool.loc[pick, "correct"]),
    ]
    for name, ans, ok in rows:
        print(f"  {name:52s} {str(ans or '(none)'):10s} {mark(ok)}")
    print(f"  {'correct answer':52s} {B}{meta['answer']}{X}")
    print(f"\n  {D}PoT samples right: {int(pot['correct'].sum())}/{len(pot)} · "
          f"pool answers right: {int(pool['correct'].sum())}/{len(pool)} · "
          f"CoT samples right: {int(cot['correct'].sum())}/{len(cot)}{X}")


if __name__ == "__main__":
    main()
