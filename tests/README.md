# Tests (`tests/`)

73 tests. They run on CPU in about 2 minutes, with a fake VLM: no GPU and no model downloads,
apart from the small embedding model.

```bash
pytest                    # whole suite
pytest tests/test_agents.py -q
```

| File | Covers |
|---|---|
| `conftest.py` | Fixtures: a test config writing to a temp dir, a synthetic dataset, and the real dataset (skipped if `mmjee prepare` has not run) |
| `test_scoring.py` | **Byte-identity** of `scoring/upstream.py` with the notebooks in `third_party/mmJEE-Eval`; extraction edge cases; adapter ranges and alternatives; every gold scores correct (real dataset) |
| `test_data.py` | Twin keys, metadata fixes, split-leak detection, stratified and paired pilot subset |
| `test_sandbox.py` | Executor (timeout, memory cap, blocked imports and calls, output cap, prelude replay, scipy) and the PoT loop (fence handling, fallback, max blocks) |
| `test_agents.py` | Step splitting, critic parsing, verdict forcing, correction stopping rules |
| `test_verifier.py` | Answer keys, pool features, selection rules, the verifier learning a separable signal (AUC > 0.9), GroupKFold OOF indexing |
| `test_rag.py` | Tag parsing, topic matching, thresholded and subject-filtered retrieval with twin exclusion |
| `test_pipeline_e2e.py` | Full pipeline on the fake VLM: solve → correct → verifier → evaluate. Checks that SC over 1 sample equals Pass@1, and that cached re-runs make no model calls. |

Tests marked `dataset` need the real dataset in `artifacts/data` (run `mmjee prepare` first);
otherwise they are skipped.
