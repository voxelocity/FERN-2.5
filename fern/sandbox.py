"""(Phase -1) A small, safe-enough sandbox for executing generated code.

Runs a Python program in a FRESH, ISOLATED interpreter as a separate process
with a wall-clock timeout (and Unix resource caps where available), capturing
the exit code and output. Exit code 0 == the program (and its embedded unit
tests) passed.

This is load-bearing infrastructure: the eval harness uses it for pass@k, and
Phase 2's RL-from-execution-feedback will use the SAME runner to score rollouts
(reward = tests passed).

SECURITY NOTE: this isolates by process + timeout + a fresh interpreter, which
is appropriate for code WE generate on a dev box. It is NOT a hardened sandbox
for adversarial untrusted code — for that, run inside a container / gVisor and
treat reward-hacking (ROADMAP risk register) with held-out tests.
"""

import os
import subprocess
import sys
import tempfile

# Unix-only hard resource caps (CPU seconds + address space). No-op on Windows.
try:
    import resource  # type: ignore

    def _limits(cpu_s: int, mem_mb: int):
        def _set():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            if mem_mb:
                b = mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        return _set
except ImportError:  # Windows
    def _limits(cpu_s: int, mem_mb: int):
        return None


def run_program(source: str, timeout: float = 8.0, mem_mb: int = 1024) -> dict:
    """Execute `source` as a standalone Python program. Returns a dict:
    {passed, returncode, timeout, stdout, stderr}. `passed` is True iff the
    process exited 0 within the time/resource budget."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-S", path],   # -I isolated, -S no site
                capture_output=True, text=True, cwd=d, timeout=timeout,
                preexec_fn=_limits(int(timeout) + 1, mem_mb) if os.name != "nt" else None,
            )
            return {
                "passed": r.returncode == 0,
                "returncode": r.returncode,
                "timeout": False,
                "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-4000:],
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "returncode": None, "timeout": True,
                    "stdout": "", "stderr": f"timeout after {timeout}s"}
        except Exception as e:  # pragma: no cover — harness robustness
            return {"passed": False, "returncode": None, "timeout": False,
                    "stdout": "", "stderr": f"sandbox error: {e!r}"}


def check_humaneval(prompt: str, completion: str, test: str, entry_point: str,
                    timeout: float = 8.0) -> dict:
    """Assemble a HumanEval-style program (prompt + completion + test +
    `check(entry_point)`) and run it. The official `test` defines `check`."""
    program = f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"
    return run_program(program, timeout=timeout)


def check_with_asserts(program_body: str, tests: list[str],
                       timeout: float = 8.0) -> dict:
    """For MBPP / built-in tasks: a code body plus a list of assert statements.
    All asserts run in one process; any failure -> non-zero exit -> not passed."""
    program = program_body + "\n\n" + "\n".join(tests) + "\n"
    return run_program(program, timeout=timeout)
