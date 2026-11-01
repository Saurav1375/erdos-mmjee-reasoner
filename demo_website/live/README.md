# Live demo

Runs the frozen final framework on a question **on demand** and shows every stage on the demo page as it happens. Nothing in `src/` is changed; this folder only imports the package.

```
laptop                                                    lab (ssh tunnels)
 browser ─ http://localhost:8765 ─ live/server.py ─┬─ :8001 ─► ml-lab2 : vllm serve Qwen3-VL-8B    (solver, corrector, tagger)
                                  (CPU: sandbox,    └─ :8002 ─► ml-lab  : vllm serve InternVL3.5-8B (critic)
                                   retrieval, verifier)
```

The two models stay loaded on separate PCs, so the batch pipeline's model swaps (2–3.5 min each) are gone.

**Requirement: both GPUs must be free.** Check with `bash demo_website/live/start.sh --check`. serve.sh refuses to start next to someone else's job, and other users' jobs are never touched. If a GPU is busy, use `start.sh --local`: the replay, sandbox, verifier what-if, retrieval search and recorded runs all work without the lab.

## For the viva

1. **About 10 minutes before**, run (from the repo root):
   ```
   bash demo_website/live/start.sh
   ```
   - It checks that both GPUs are free, copies `lab/serve.sh` to both PCs, starts the two model servers and opens the ssh tunnels.
   - It waits until both models are loaded (about 3–4 min; they load in parallel), then starts the demo server and opens http://localhost:8765.
   - The top bar shows **Live · lab connected**.
2. **Rehearse.** Run the questions you plan to show once. Each finished run is saved in `live/runs/` (and every model call in `live/cache.sqlite`), so the same question in the same mode later shows instantly. It also still shows if the lab fails during the viva; the badge says *recorded run*.
3. **In the viva**, use the "Run it live" section:
   - **Run live:** the transcript, the tag and retrieval, attempt 1 being written token by token with its Python running in the sandbox, the eight attempts' answers, the critic's verdicts per round, and the verifier's pick. For exam questions the saved frozen answer is shown next to it.
   - **New samples:** the same question with new random samples.
   - **New question:** drop, paste or choose an image of any question; pick its subject, type and language. There is no answer key, so nothing is marked.
   - **Step through this run:** opens the finished run in the replay viewer.
   - **■ Stop** (next to the timer): ends the run at once. The requests on the lab GPUs are cancelled, and the GPUs are free for the next run within a second. The run gets no final answer and is not saved as a recorded run, because the verifier needs the full pool. What it produced so far stays on screen. Model calls that had already finished stay in `cache.sqlite`, so **Run it again** resumes close to where it stopped (tested: stopped after 40 s, the re-run finished in 51 s, correct).
4. **Interactive parts.** These work whenever the demo server runs, even without the lab:
   - **Solve stage:** "Edit and run ▸" on any code block runs it in the real sandbox. The "try to break it" buttons show the safety layers: the syntax-tree check blocks `import os`, `open` and `__import__`; the time limit stops an endless loop; the memory limit stops a memory bomb.
   - **Verify stage:** click a row to see the five features that moved its score most. Double-click an answer to change it, and the real verifier re-scores all 16 answers.
   - **Look-up stage:** search the knowledge base with any text.
5. **Afterwards:**
   ```
   bash demo_website/live/stop.sh
   ```
   This stops the demo server, the tunnels and both model servers, so nothing is left running on the lab PCs.

## If something fails

| Situation | What the page does |
|---|---|
| Lab not reachable | Badge "lab offline". Rehearsed questions show their recorded run. The replay and every interactive part still work. |
| Lab lost during a run | "Lab connection lost", plus a button to replay the saved run of that question |
| Page reloaded, or another question started, during a run | Only one run goes at a time. The page reconnects to the run in progress and shows it from its start; the server keeps every event of the run. |
| No demo server (double-clicked `index.html`, or the online copy) | Replay only; the live parts stay hidden |

Run only the laptop side: `bash demo_website/live/start.sh --local`.

## Settings

| Mode | Attempts | Correction rounds | Note |
|---|---|---|---|
| Full | 8 | up to 3 | the frozen settings |
| Quick | 4 | 1 | faster; the pool is smaller than the verifier was trained on |

Measured on 2026-10-31 (both models loaded, nothing cached):

| Run | Total | Transcribe + tag + look-up | Solve (attempts in parallel) | Critic + corrector | Verify |
|---|---|---|---|---|---|
| Full, P2-EN-Phys-MCQ-q1 | 4.3 min, correct (C) | 5 s | 3.5 min | 38 s | 3 s |
| Quick, P1-EN-Chem-Num-q1 | 3.9 min, correct (9.8) | | | | |
| Quick, upload of P1-EN-Math-Num-q2 | 3.0 min, answered 6 (key 5) | | | | |

- Starting both models takes about 3.5 min (they load in parallel).
- The solve stage takes most of the time. Its attempts run side by side, so it lasts as long as the slowest attempt, and Quick saves little. **Use Full**: it is the frozen setting and costs under a minute more.
- Some questions take much longer. On one uploaded physics question the model repeated the same sentences until each attempt hit its 8,192-token cap, so the critique and correction stage alone took 8.5 min. The page keeps streaming the repeated text meanwhile; this is the model, not a hang. Rehearse the questions you will show.
- vLLM 0.30 bug, fixed in `lab/serve.sh` (2026-10-31): when a request with an image was rejected for being longer than 12,288 tokens, later requests with that image failed with 500 "multi-modal cache drift" and the run stopped. The servers now start with `--mm-processor-cache-gb 0`, and `http_engine.py` retries a 5xx twice.
- A question that was run before shows instantly (recorded run). So does any part already in `cache.sqlite`, for example a Full run after a Quick run of the same question.

**Other layouts tried on 2026-10-31.**
- **Critic on ml-lab beside another user's job (2.7 GB):** at a 0.70 or 0.75 GPU share there is not enough attention cache (0.53 GB where a 12,288-token request needs 0.84 GB), so it does not start.
- **Both models on ml-lab2 with vLLM sleep mode:** the GPU part works, since a sleeping model leaves 0.8 GB on the GPU. But the parked model's 6.5 GB of weights in CPU memory, plus loading the second model, ran the 15 GB-RAM PC out of memory, and it stopped responding for some minutes. This layout was therefore removed.

## Checks

- **`test_parity.py`.** This is the proof that the live runner is the frozen pipeline. With the frozen seeds, it answers every model call from a copy of the saved cache, re-runs all code in the sandbox, and requires the page record to equal the saved one: transcript, tag, retrieval, solution, critic rounds, all 16 verifier scores, pick and outcome.
  ```
  cp artifacts_v3/cache/generations.sqlite /tmp/gen.sqlite      # the main file only, see below
  .venv/bin/python demo_website/live/test_parity.py /tmp/gen.sqlite [--all]
  ```
  Result (2026-10-31):
  - With `--all`, 202 of the 204 questions match exactly while all code is re-run on this laptop.
  - In the other 2 (P1-EN-Phys-MCQ-q3, P2-HI-Math-Num-q7), the model's own Python prints something different here than on the lab PC, so the next request differs.
  - With `--exec-cache` (the saved code outputs), both match too.
- **Test mode, with no lab.** Every model call is answered from that cache copy, so the whole live UI can be tried end to end. Exam questions reproduce the saved runs; new questions need the lab.
  ```
  LIVE_REPLAY_CACHE=/tmp/gen.sqlite .venv/bin/python demo_website/live/server.py
  ```

**Note on the laptop's `artifacts_v3/cache/`.** Next to `generations.sqlite` (Oct 18) sits a stale `generations.sqlite-wal` from Oct 8. Opening the database read-write with that file present corrupts it. Copy only the main file, as above. The lab's copy has no such file.

## Files

| File | What it does |
|---|---|
| `runner.py` | One question through every stage, using the package's own functions |
| `http_engine.py` | Sends the package's model requests to a `vllm serve` server, and reports progress and streamed text to the page |
| `server.py` | The page, live runs (Server-Sent Events), sandbox, verifier what-if, retrieval search. Standard library only. |
| `test_parity.py` | The parity check above |
| `lab/serve.sh` | Starts or stops one model server on a lab PC (flags mirror `configs/models.yaml`) |
| `start.sh`, `stop.sh` | Start and stop everything from the laptop |
| `runs/`, `cache.sqlite`, `exec_cache.sqlite`, `uploads/`, `emb_cache/` | Created at run time |
