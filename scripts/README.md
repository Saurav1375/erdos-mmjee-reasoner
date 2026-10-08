# Scripts (`scripts/`)

| Script | Runs on | Purpose |
|---|---|---|
| `vendor_upstream.py` | either | Regenerates `src/mmjee_reasoner/scoring/upstream.py` byte for byte from the base-paper notebooks in `third_party/mmJEE-Eval` (located with `ast`). Run it once; `tests/test_scoring.py` re-checks identity. |
| `phase2_report.py` | CPU | Builds the Phase-2 progress report: `report/phase2/` (figures, generated LaTeX tables, `phase2_report.tex/.pdf`). It covers baselines, the pilot and the v1→v2 fixes only, with no full-run results. |
| `final_report.py` | CPU | Builds the final-report draft: `report/final/` (figures, tables, `final_report.tex/.pdf`) from the full-run artifacts. |
| `lab/env.sh` | — | Shared settings: `LAB_HOST` (ssh alias, default `ml-lab`), `LAB_ROOT` (default `~/saurav_mmjee`), project paths. Override them with environment variables. |
| `lab/setup_lab.sh` | local → lab | One-time setup. Syncs the code, creates `$LAB_ROOT/.venv` with vLLM, and installs the project in editable mode. No sudo. |
| `lab/sync_to_lab.sh` | local → lab | rsyncs the code (not the artifacts) to the lab PC |
| `lab/sync_from_lab.sh` | lab → local | rsyncs the generated artifacts back (never deletes local files) |
| `lab/launch.sh` | local → lab | Starts a detached, supervised chain of stages (`nohup setsid`). It refuses to start if the GPU is busy. |
| `lab/supervise.sh` | lab | Runs `mmjee` stages one after another and retries a dead stage up to 3 times. Stages resume through the caches. Writes progress to `artifacts/logs/<name>.supervisor.out`, which ends in `ALL_DONE` or `FAILED`. |

## Typical lab session

```bash
scripts/lab/setup_lab.sh                                   # once
scripts/lab/launch.sh smoke "prepare" "solve --exp pot --limit 10"
scripts/lab/launch.sh pilot "solve --exp pot --pilot" "correct --exp pot_corr_cross --pilot"
scripts/lab/launch.sh full  "solve --exp cot" "solve --exp pot" "correct --exp pot_corr_cross"
ssh ml-lab 'tail -3 ~/saurav_mmjee/mmjee-reasoner/artifacts/logs/full.supervisor.out'
scripts/lab/sync_from_lab.sh
```

> **Why not tmux or `systemd-run`?** On the shared lab PC, systemd-oomd applies a
> memory-pressure kill policy to everything under `user@1000.service`. Model swaps (6–7 GB of
> weights) caused short pressure spikes, and oomd killed the job twice, wasting 10.5 h.
> Processes started with `nohup setsid` from an ssh session fall outside that policy.
