"""Proves the live runner IS the frozen pipeline, without a GPU.

With the frozen seeds (salt=None) the runner makes exactly the requests of the saved 2026 run,
so a read-only copy of the saved generation cache answers all of them (offline: any request
that differs is a cache miss and fails the test). The resulting page records must equal the
saved ones in data.js: transcript, tag, retrieval, the shown solution, critic rounds, the
16-answer pool with the verifier's scores, the pick and the outcome.

usage (from the repo root):
  .venv/bin/python demo_website/live/test_parity.py [CACHE_COPY] [--exec-cache]
CACHE_COPY: a copy of artifacts_v3/cache/generations.sqlite (+ -wal/-shm).
--all: every 2026 question instead of the six showcase cases.
--exec-cache: reuse the saved code-execution results too (default: re-run all code locally).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import runner  # noqa: E402  (sets up paths)

CASES = ["P1-EN-Phys-Num-q2", "P1-EN-Chem-Num-q1", "P2-HI-Phys-MCQ-q1", "P2-HI-Chem-Num-q7",
         "P2-HI-Chem-MCQ-q4", "P1-HI-Phys-Num-q1"]


def pyval(x):   # the conversion export_data.py applies when writing data.js
    import numpy as np
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def norm(x):
    return json.dumps(json.loads(json.dumps(x, ensure_ascii=False, default=pyval).replace("\ufffd", "")),
                      sort_keys=True)


KEYS = ["transcript", "tag", "retrieval", "shown", "chain", "pool", "pick", "methods",
        "pot_rate", "outcome"]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache = Path(args[0]) if args else runner.PROJ / "artifacts_v3/cache/generations.sqlite"
    if "--exec-cache" in sys.argv:
        runner.EXEC_CACHE = cache
    saved = json.loads((runner.SITE / "data.js").read_text(encoding="utf-8")
                       .removeprefix("window.DEMO = ").rstrip().rstrip(";"))
    by_id = {q["id"]: q for q in saved["questions"]}
    fw = runner.Framework()
    solver = runner.cached_engine("qwen3vl8b-awq", cache, offline=True)
    critic = runner.cached_engine("internvl3.5-8b-awq", cache, offline=True)
    bad = 0
    cases = list(by_id) if "--all" in sys.argv else CASES
    for cid in cases:
        want = by_id[cid]
        t0 = time.time()
        try:
            got = fw.run(fw.exam_question(want["uid"]), solver, critic, lambda e: None)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {cid}: {type(e).__name__}: {e}")
            bad += 1
            continue
        diff = [k for k in KEYS if norm(got[k]) != norm(want[k])]
        print(f"{'OK  ' if not diff else 'DIFF'} {cid}  ({time.time() - t0:.0f}s)"
              + (f"  differs in: {', '.join(diff)}" if diff else ""))
        bad += bool(diff)
    print("live runner reproduces the frozen run" if not bad else f"{bad} case(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
