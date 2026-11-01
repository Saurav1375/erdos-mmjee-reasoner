"""Live demo server: the demo page plus live runs and the interactive (CPU) features.

Runs on the presenter's laptop. The two models run on the lab PCs as `vllm serve` servers,
reached through ssh tunnels (see start.sh): solver on localhost:8001, critic on localhost:8002.
Everything else (code sandbox, retrieval, verifier) runs here on the CPU, so those features
work even when the lab is offline. Only the Python standard library is used for HTTP.

usage (from the repo root):
  .venv/bin/python demo_website/live/server.py [--port 8765]
then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import logging
import mimetypes
import os
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import runner  # sets up paths and imports the project  # noqa: I001
from http_engine import Cancelled, HTTPEngine, LabUnavailable, ObservedEngine
from records import cut

import numpy as np
import pandas as pd
from PIL import Image

from mmjee_reasoner.embed import embed_texts
from mmjee_reasoner.rag.retrieve import search
from mmjee_reasoner.sandbox.executor import run_code
from mmjee_reasoner.scoring import is_correct
from mmjee_reasoner.verifier.model import select
from mmjee_reasoner.verifier.v3 import prepare_pool, v3_features

log = logging.getLogger("live.server")
SOLVER_URL = os.environ.get("LIVE_SOLVER_URL", "http://127.0.0.1:8001")
CRITIC_URL = os.environ.get("LIVE_CRITIC_URL", "http://127.0.0.1:8002")
RUNS = runner.LIVE / "runs"            # finished live runs (shown again when the lab is offline)
LIVE_CACHE = runner.LIVE / "cache.sqlite"
STATIC_OK = {".html", ".js", ".css", ".webp", ".png", ".jpg", ".woff2", ".svg", ".json", ".ico"}
MAX_PIXELS = 1_700_000                 # configs/models.yaml mm_processor_kwargs.max_pixels

FEATURE_LABELS = {
    "agree_frac": "agreement with its sub-pool", "is_majority": "majority answer of its sub-pool",
    "vote_rank": "vote rank", "distinct_frac": "share of distinct answers",
    "full_agree": "agreement with all 16", "full_is_majority": "majority of all 16",
    "other_half_agree": "agreement with the other half", "n_code_ok": "code blocks that ran",
    "code_ok_frac": "share of code that ran", "n_code_err": "code errors",
    "truncated": "cut off", "used_fallback": "fell back to plain reasoning",
    "valid_format": "valid answer format", "corr_critic_flagged_first": "critic flagged it",
    "corr_n_flags": "critic flags", "corr_answer_changed": "correction changed the answer",
    "log_tokens": "length (tokens)", "log_chars": "length (characters)", "n_steps": "steps",
    "hedges_per_kchar": "hedging words", "is_corrected": "corrected version",
    "n_letters": "letters in the answer", "n_boxed": "boxed answers", "is_number": "answer is a number",
    "log_abs_value": "size of the number", "is_integer": "answer is an integer",
    "is_negative": "answer is negative", "n_decimals": "decimal places", "has_chain": "was checked",
    "corr_rounds": "correction rounds", "correction_failed_last": "last correction failed",
    "type_MCQ-Single": "question type", "type_MCQ-Multiple": "question type",
    "type_Numerical": "question type", "type_Matching": "question type",
    "lang_hindi": "Hindi question", "requires_image": "needs the diagram",
}


class Run:
    def __init__(self, rid: str):
        self.id, self.events, self.done = rid, [], False
        self.cond = threading.Condition()
        self.stop = threading.Event()

    def emit(self, ev: dict) -> None:
        with self.cond:
            self.events.append(ev)
            if ev["type"] in ("done", "error", "stopped"):
                self.done = True
            self.cond.notify_all()


class App:
    def __init__(self):
        log.info("loading the framework (config, KB, verifier)...")
        self.fw = runner.Framework()
        art = self.fw.art
        self.http = {"solver": HTTPEngine(SOLVER_URL, "qwen3vl8b-awq", art),
                     "critic": HTTPEngine(CRITIC_URL, "internvl3.5-8b-awq", art)}
        self.engines = {
            "solver": runner.cached_engine("qwen3vl8b-awq", LIVE_CACHE, lambda: self.http["solver"]),
            "critic": runner.cached_engine("internvl3.5-8b-awq", LIVE_CACHE,
                                           lambda: self.http["critic"]),
        }
        # Test mode: answer every model call from a saved cache with the frozen seeds (no lab).
        self.replay = os.environ.get("LIVE_REPLAY_CACHE")
        if self.replay:
            self.engines = {"solver": runner.cached_engine("qwen3vl8b-awq", Path(self.replay), offline=True),
                            "critic": runner.cached_engine("internvl3.5-8b-awq", Path(self.replay),
                                                           offline=True)}
            log.info("TEST MODE: model calls answered from %s", self.replay)
        self.runs: dict[str, Run] = {}
        self.ids = itertools.count(1)
        self.busy = threading.Lock()
        self.active: dict | None = None    # the run in progress, so a reloaded page can reattach to it
        self._pool = None
        RUNS.mkdir(exist_ok=True)
        log.info("ready")

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        recorded = sorted(p.stem for p in RUNS.glob("*.json"))
        if self.replay:
            return {"server": True, "solver": True, "critic": True, "busy": self.busy.locked(),
                    "active": self.active, "recorded": recorded, "test_mode": True}
        return {"server": True, "solver": self.http["solver"].healthy(),
                "critic": self.http["critic"].healthy(), "busy": self.busy.locked(),
                "active": self.active, "recorded": recorded}

    # ------------------------------------------------------------------ live runs
    @staticmethod
    def _featured(r) -> bool:
        key = r.meta.get("rid") or r.meta.get("cid") or ""
        return key.endswith("|s0") or "|s0|corr" in key

    def start_run(self, body: dict) -> dict:
        mode = body.get("mode", "full")
        salt = str(body.get("salt") or "live-1")
        if body.get("image"):
            raw = base64.b64decode(body["image"].split(",", 1)[-1])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if im.width * im.height > MAX_PIXELS:
                f = (MAX_PIXELS / (im.width * im.height)) ** 0.5
                im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG")
            q = self.fw.upload_question(buf.getvalue(), body["subject"], body["type"],
                                        body["language"], bool(body.get("diagram")))
        else:
            q = self.fw.exam_question(body["uid"])
        key = f"{q['uid']}__{mode}__{salt}"
        run = Run(str(next(self.ids)))
        self.runs[run.id] = run
        saved = RUNS / f"{key}.json"
        lab = bool(self.replay) or (self.http["solver"].healthy() and self.http["critic"].healthy())
        if self.replay:
            salt, key = None, f"{q['uid']}__{mode}__test"
            saved = RUNS / "test" / f"{key}.json"
            saved.parent.mkdir(exist_ok=True)
        if saved.exists() and (body.get("recorded") or not lab):
            rec = json.loads(saved.read_text(encoding="utf-8"))
            run.emit({"type": "recorded", "at": rec.get("live", {}).get("recorded")})
            run.emit({"type": "done", "record": rec})
            return {"run": run.id, "recorded": True}
        if not lab:
            run.emit({"type": "error", "message": "The lab GPUs are not reachable, and this "
                      "question has no recorded live run. Use the replay below."})
            return {"run": run.id}
        if not self.busy.acquire(blocking=False):
            # One run at a time: hand back the run in progress so the page can show it.
            del self.runs[run.id]
            active = self.active
            if active is None:
                return {"error": "Another live run is still going."}
            return {"run": active["run"], "attached": True, "active": active}
        self.active = {"run": run.id, "title": str(body.get("title") or q["uid"]), "exam": body.get("exam"),
                       "mode": mode, "n": runner.MODES[mode]["n_samples"], "t0": time.time()}
        threading.Thread(target=self._work, args=(run, q, mode, salt, saved), daemon=True).start()
        return {"run": run.id}

    def _work(self, run: Run, q: dict, mode: str, salt: str, saved: Path) -> None:
        t0 = time.time()

        def emit(ev: dict) -> None:
            run.emit(ev)
            if run.stop.is_set() and ev["type"] == "stage":   # CPU stages stop at the next boundary
                raise Cancelled("stopped")
        try:
            for name, h in self.http.items():
                h.cancel.clear()
                h.stream_to = (lambda r, n=name: (lambda d: run.emit(
                    {"type": "delta", "rid": r.meta.get("rid") or r.meta.get("cid"), "text": d}))
                    if self._featured(r) else None)
            solver = ObservedEngine(self.engines["solver"], "solver", run.emit, self._featured)
            critic = ObservedEngine(self.engines["critic"], "critic", run.emit, self._featured)
            rec = self.fw.run(q, solver, critic, emit, mode=mode, salt=salt)
            saved.write_text(json.dumps(rec, ensure_ascii=False, default=_py), encoding="utf-8")
            run.emit({"type": "done", "record": json.loads(json.dumps(rec, default=_py))})
        except Cancelled:
            # Nothing is scored or saved as a recorded run (the verifier needs the full pool).
            # Model calls that finished before the stop stay in the generation cache, so running
            # the question again resumes quickly.
            run.emit({"type": "stopped", "seconds": round(time.time() - t0, 1)})
        except LabUnavailable as e:
            run.emit({"type": "error", "message": f"Lab connection lost ({e}). The replay below "
                      "still works."})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            run.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self.active = None
            self.busy.release()

    def stop_run(self, body: dict) -> dict:
        run = self.runs.get(str(body.get("run")))
        if run is None or run.done:
            return {"stopped": False}
        run.stop.set()
        for h in self.http.values():
            h.abort()
        return {"stopped": True}

    # ------------------------------------------------------------------ CPU features
    def sandbox(self, body: dict) -> dict:
        sb = self.fw.cfg["sandbox"]
        t0 = time.time()
        res = run_code(body.get("code", ""), timeout_s=sb["timeout_s"], memory_mb=sb["memory_mb"],
                       max_output_chars=sb["max_output_chars"],
                       allowed_imports=tuple(sb["allowed_imports"]))
        return {"ok": res.ok, "output": res.output, "error": res.error,
                "timed_out": res.timed_out, "seconds": round(time.time() - t0, 2)}

    def saved_pool(self) -> pd.DataFrame:
        if self._pool is None:
            from mmjee_reasoner.pipeline.evaluate import Evaluator

            ev = Evaluator(self.fw.cfg)
            self._pool = ev.pool("v3_full")
        return self._pool

    def verify(self, body: dict) -> dict:
        """Re-score one question's 16-answer pool after the viewer's edits."""
        uid = body["uid"]
        pool = self.saved_pool()
        g = pool[pool["uid"] == uid].reset_index(drop=True).copy()
        q = self.fw.exam_question(uid)
        for e in body.get("edits", []):
            i = int(e["i"])
            if "pred" in e:
                g.at[i, "pred"] = str(e["pred"])
                g.at[i, "correct"] = is_correct(str(e["pred"]), q)
            if "code_ok" in e:
                nb = max(1, int(g.at[i, "n_code_blocks"] or 0))
                g.at[i, "n_code_blocks"] = nb
                g.at[i, "n_code_ok"] = nb if e["code_ok"] else 0
                g.at[i, "n_code_err"] = 0 if e["code_ok"] else nb
            if "cut_off" in e:
                g.at[i, "finish"] = "length" if e["cut_off"] else "stop"
        v = self.fw.verifier
        fm = v3_features(prepare_pool(g), self.fw.cfg, runner.LIVE / "emb_cache")
        X, _, _ = fm.select(v.feature_set)
        scores = 1.0 / (1.0 + np.exp(-v.pipeline.predict(X)))
        import xgboost as xgb

        Xt = v.pipeline.named_steps["pre"].transform(X)
        contrib = v.pipeline.named_steps["clf"].get_booster().predict(
            xgb.DMatrix(Xt), pred_contribs=True)
        names = list(fm.hand.columns)
        pick = int(np.argmax(scores))
        fin = select(g, scores, self.fw.selection)
        why = []
        for i in range(len(g)):
            c = contrib[i][:-1]
            top = np.argsort(-np.abs(c))[:5]
            why.append([{"f": FEATURE_LABELS.get(names[j], names[j].replace("_", " ")),
                         "v": round(float(c[j]), 3)} for j in top])
        return {"scores": [round(float(s), 4) for s in scores], "pick": pick,
                "pred": fin["pred"], "ok": bool(fin["correct"]),
                "rows": [{"pred": p if isinstance(p, str) else None, "ok": bool(o)}
                         for p, o in zip(g["pred"], g["correct"])], "why": why}

    def retrieve(self, body: dict) -> dict:
        cfg = self.fw.cfg
        rag = cfg["rag"]
        query = (body.get("query") or "").strip()
        if not query:
            return {"hits": []}
        qv = embed_texts([query], rag["embedding_model"])[0]
        subject = body.get("subject") or None
        hits = search(qv, self.fw.kb_vecs, self.fw.entries, subject, body.get("twin"), 5, -1.0)
        tau = self.fw.kb_meta["threshold"]["value"]
        e = self.fw.entries
        return {"tau": tau, "note": "tag similarity only (the live pipeline also mixes in 10% "
                "problem-text similarity)",
                "hits": [{"id": e[i]["kb_id"], "sim": round(float(s), 4), "above": bool(s >= tau),
                          "subj": e[i]["subject"], "year": e[i]["year"], "topic": e[i].get("topic"),
                          "problem": cut(e[i]["problem"], 500)}
                         for i, s in hits]}


def _py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


APP: App | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=_py).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._json(APP.health())
        if path.startswith("/api/events/"):
            return self._events(path.rsplit("/", 1)[1])
        return self._static(path)

    def do_POST(self):  # noqa: N802
        routes = {"/api/run": APP.start_run, "/api/stop": APP.stop_run, "/api/sandbox": APP.sandbox,
                  "/api/verify": APP.verify, "/api/retrieve": APP.retrieve}
        fn = routes.get(self.path)
        if fn is None:
            return self._json({"error": "not found"}, 404)
        try:
            return self._json(fn(self._body()))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (runner.SITE / rel).resolve()
        if (runner.SITE not in f.parents and f != runner.SITE) or f.suffix not in STATIC_OK \
                or not f.is_file() or "_build" in f.parts:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _events(self, rid: str) -> None:
        run = APP.runs.get(rid)
        if run is None:
            return self._json({"error": "unknown run"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sent = 0
        try:
            while True:
                with run.cond:
                    while sent == len(run.events) and not run.done:
                        run.cond.wait(timeout=15)
                        if sent == len(run.events) and not run.done:
                            break                      # keep-alive below
                    new = run.events[sent:]
                    done = run.done
                if not new:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                for ev in new:
                    self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False,
                                                            default=_py).encode() + b"\n\n")
                self.wfile.flush()
                sent += len(new)
                if done and sent == len(run.events):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    global APP
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "sentence_transformers", "mmjee_reasoner"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    APP = App()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    log.info("demo at http://localhost:%d  (solver %s, critic %s)", args.port, SOLVER_URL,
             CRITIC_URL)
    srv.serve_forever()


if __name__ == "__main__":
    main()
