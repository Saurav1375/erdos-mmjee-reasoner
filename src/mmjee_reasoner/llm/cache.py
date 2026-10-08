"""Persistent generation cache (SQLite).

Every model call is stored under ``sha256(model, messages, sampling)``. Stages
are therefore resumable: re-running a stage after a crash only generates the
missing completions, and the evaluation/report stages can run fully offline.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from mmjee_reasoner.llm.types import GenResult


class GenerationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS gen (key TEXT PRIMARY KEY, model TEXT, result TEXT)"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, GenResult]:
        out: dict[str, GenResult] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = f"SELECT key, result FROM gen WHERE key IN ({','.join('?' * len(chunk))})"
                for key, blob in self._conn.execute(q, chunk):
                    out[key] = GenResult.from_json(blob)
        return out

    def put_many(self, model: str, items: list[tuple[str, GenResult]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO gen (key, model, result) VALUES (?, ?, ?)",
                [(k, model, r.to_json()) for k, r in items],
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM gen").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
