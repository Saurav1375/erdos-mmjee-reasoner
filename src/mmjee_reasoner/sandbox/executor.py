"""Sandboxed execution of model-written Python code (Module A).

Defence in depth for untrusted, model-generated code:

1. Static check (``ast``): only whitelisted top-level modules may be imported;
   calls to ``open/exec/eval/compile/__import__/input/globals/...`` and any
   dunder attribute access are rejected.
2. The code runs in a fresh ``python -I -S``-style subprocess (isolated mode,
   no user site, empty environment) in a temporary working directory.
3. POSIX resource limits: CPU time, address space, no core files, limited
   file size (stdout/stderr are files, so output size is capped too).
4. Wall-clock timeout; the whole process group is killed on expiry.

This is a guard against accidents and casual misuse by the model, not a
security boundary against a determined attacker (no network namespace).
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview",
}
# Attributes that evaluate strings as code or touch files (sympy/numpy/operator escapes).
FORBIDDEN_ATTRS = {
    "sympify", "parse_expr", "lambdify", "S", "attrgetter", "methodcaller", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "load", "save", "savez", "savez_compressed",
    "savetxt", "memmap", "DataSource", "fromregex", "system", "popen", "preview",
}

DEFAULT_ALLOWED = (
    "math", "cmath", "fractions", "decimal", "itertools", "functools", "statistics",
    "sympy", "numpy", "scipy", "collections", "operator",
)
# Submodules of allowed packages that read/write files or parse strings into code.
FORBIDDEN_MODULES = (
    "scipy.io", "scipy.datasets", "scipy.misc", "numpy.lib.npyio", "numpy.f2py",
    "numpy.distutils", "numpy.testing", "numpy.ctypeslib", "sympy.parsing", "sympy.utilities",
    "sympy.printing.preview",
)


def _module_forbidden(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES)


@dataclass
class ExecResult:
    ok: bool
    output: str          # stdout (and stderr tail on error), truncated
    error: str | None    # short error category / message, None if ok
    timed_out: bool = False


class UnsafeCodeError(ValueError):
    pass


def check_code(code: str, allowed_imports: tuple[str, ...] | list[str] = DEFAULT_ALLOWED) -> None:
    """Raise ``UnsafeCodeError`` / ``SyntaxError`` if the code is not allowed."""
    tree = ast.parse(code)
    allowed = set(allowed_imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed or _module_forbidden(alias.name):
                    raise UnsafeCodeError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in allowed or _module_forbidden(mod) or any(
                    _module_forbidden(f"{mod}.{a.name}") for a in node.names):
                raise UnsafeCodeError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise UnsafeCodeError(f"call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS | {"__builtins__"}:
            raise UnsafeCodeError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise UnsafeCodeError(f"'{node.attr}' is not allowed")
        elif isinstance(node, ast.alias) and node.name in FORBIDDEN_ATTRS:
            raise UnsafeCodeError(f"import of '{node.name}' is not allowed")


# Child-side runner: apply resource limits, then run the (already checked) code
# read from stdin. Using a runner instead of ``preexec_fn`` keeps this safe to
# call from many threads at once.
_RUNNER = """
import resource, sys
cpu, mem = int(sys.argv[1]), int(sys.argv[2]) * 1024 * 1024
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
import contextlib, io, json
job = json.loads(sys.stdin.read())
del resource, sys, cpu, mem
g = {"__name__": "__main__"}
# Earlier code blocks of the same solution are replayed silently so that variables
# defined there exist (the model writes blocks that build on each other).
for i, block in enumerate(job["prelude"]):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(block, "<earlier block %d>" % (i + 1), "exec"), g)
    except BaseException:  # incl. SystemExit: a replayed block must never end the run
        pass
del job["prelude"]
exec(compile(job["code"], "<solution>", "exec"), g)
"""


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def safe_prelude(blocks: list[str], allowed_imports=DEFAULT_ALLOWED) -> list[str]:
    """Earlier blocks that pass the safety check (unsafe/unparseable ones are dropped)."""
    out = []
    for b in blocks:
        try:
            check_code(b, allowed_imports)
        except Exception:
            continue
        out.append(b)
    return out


def run_code(code: str, timeout_s: float = 10, memory_mb: int = 1024,
             max_output_chars: int = 2000,
             allowed_imports: tuple[str, ...] | list[str] = DEFAULT_ALLOWED,
             prelude: list[str] | tuple[str, ...] = ()) -> ExecResult:
    """Check and execute ``code`` (after silently replaying ``prelude`` blocks)."""
    import json

    prelude = safe_prelude(list(prelude), allowed_imports)
    try:
        check_code(code, allowed_imports)
    except SyntaxError as e:
        return ExecResult(False, f"SyntaxError: {e.msg} (line {e.lineno})", "syntax")
    except UnsafeCodeError as e:
        return ExecResult(False, f"Blocked: {e}", "blocked")
    except Exception as e:  # e.g. ValueError for null bytes on Python 3.11
        return ExecResult(False, f"SyntaxError: {e}", "syntax")

    with tempfile.TemporaryDirectory(prefix="mmjee_sbx_") as tmp:
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
        # stdout/stderr go to files so RLIMIT_FSIZE bounds them (pipes are unbounded).
        out_path, err_path = os.path.join(tmp, "out.txt"), os.path.join(tmp, "err.txt")
        with open(out_path, "w") as fo, open(err_path, "w") as fe:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-c", _RUNNER, str(int(timeout_s) + 1),
                     str(memory_mb)],
                    cwd=tmp, env=env, stdin=subprocess.PIPE, stdout=fo, stderr=fe, text=True,
                    start_new_session=True,
                )
            except OSError as e:  # pragma: no cover
                return ExecResult(False, f"ExecutionError: {e}", "spawn")
            try:
                proc.communicate(input=json.dumps({"prelude": prelude, "code": code}),
                                 timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:  # pragma: no cover
                    pass
                proc.wait()
                return ExecResult(False, f"TimeoutError: code ran longer than {timeout_s} s",
                                  "timeout", timed_out=True)
        with open(out_path, errors="replace") as f:
            out = f.read(max_output_chars * 4)
        with open(err_path, errors="replace") as f:
            err = f.read()[-4000:]
        if proc.returncode == -signal.SIGXFSZ:
            return ExecResult(False, _truncate(out, max_output_chars) + "\nOutputTooLarge",
                              "output_too_large")
    if proc.returncode != 0:
        last = err.strip().splitlines()[-1] if err.strip() else f"exit code {proc.returncode}"
        text = _truncate((out + "\n" + last).strip(), max_output_chars)
        return ExecResult(False, text, last.split(":")[0][:60])
    if not out.strip():
        return ExecResult(True, "(no output - use print() to show results)", None)
    return ExecResult(True, _truncate(out, max_output_chars), None)


class ExecCache:
    """Persistent code-execution cache (SQLite table next to the generation cache).

    Program output feeds back into the next generation request, so a re-run
    must see *exactly* the same output to hit the generation cache (a timeout
    under different machine load would otherwise change the trajectory).
    """

    def __init__(self, path: str):
        import sqlite3
        import threading

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS exec (key TEXT PRIMARY KEY, ok INTEGER, output TEXT, "
            "error TEXT, timed_out INTEGER)"
        )
        self._conn.commit()

    @staticmethod
    def key(code: str, settings: dict) -> str:
        import hashlib
        import json

        blob = json.dumps({"code": code, **settings}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> ExecResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT ok, output, error, timed_out FROM exec WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return ExecResult(bool(row[0]), row[1], row[2], bool(row[3]))

    def put(self, key: str, r: ExecResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO exec VALUES (?, ?, ?, ?, ?)",
                (key, int(r.ok), r.output, r.error, int(r.timed_out)),
            )
            self._conn.commit()


def run_many(codes: list[str], workers: int = 8, cache: ExecCache | None = None,
             preludes: list[list[str]] | None = None, **kwargs) -> list[ExecResult]:
    """Execute several programs in parallel (each in its own subprocess)."""
    if not codes:
        return []
    preludes = preludes or [[] for _ in codes]
    settings = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}

    def safe_run(code: str, prelude: list[str]) -> ExecResult:
        try:
            return run_code(code, prelude=prelude, **kwargs)
        except Exception as e:  # never let one program kill a whole stage
            return ExecResult(False, f"ExecutionError: {type(e).__name__}: {e}", "internal")

    def one(job: tuple[str, list[str]]) -> ExecResult:
        code, prelude = job
        if cache is None:
            return safe_run(code, prelude)
        key = ExecCache.key(code, {**settings, "prelude": prelude} if prelude else settings)
        hit = cache.get(key)
        if hit is not None:
            return hit
        res = safe_run(code, prelude)
        cache.put(key, res)
        return res

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, list(zip(codes, preludes))))
