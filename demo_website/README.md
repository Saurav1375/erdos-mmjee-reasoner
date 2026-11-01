# Demo website

Interactive replay of the frozen final framework's real runs on JEE Advanced 2026. No model is called;
everything comes from the saved runs in `../artifacts_v3/` (and the Step-5 analysis).

- **Open offline:** double-click `index.html` (works without internet; fonts fall back to Georgia/system).
- **Live mode (real model on the lab GPUs, plus code sandbox, verifier what-if and search):** see `live/README.md`; start with `bash demo_website/live/start.sh`.

Rebuild (only needed if the saved runs change), from the repo root:
```
.venv/bin/python demo_website/_build/export_data.py   # data.js + img/ (checks it reproduces Table I)
python3 demo_website/_build/make_index.py                             # _build/page.html -> index.html
```

| File | What it is |
|---|---|
| `index.html` | the page (built from `_build/page.html`) |
| `data.js` | all 204 questions: transcript, tag, retrieval, one solution, critic rounds, 16-answer pool with verifier scores |
| `img/` | the question images |
| `vendor/katex/` | maths rendering, kept locally so it works offline |
| `live/` | the live demo server, lab scripts and the parity test |
| `_build/records.py` | the per-question record, shared by the export and the live runner |
