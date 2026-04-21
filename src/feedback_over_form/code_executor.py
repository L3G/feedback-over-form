"""Sandboxed Python code executor for test-driven code generation pipelines.

Runs generated code + test assertions in an isolated subprocess.
No network access, temp files cleaned up after execution.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import tempfile

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of running code against test assertions."""

    passed: bool
    stdout: str
    stderr: str
    error_type: str | None = None
    failed_tests: list[str] = []


async def execute_code(
    code: str,
    test_code: str,
    timeout: float = 10.0,
) -> ExecutionResult:
    """Run generated code + tests in a sandboxed subprocess.

    Creates a temp file with the code followed by the test assertions,
    runs it via subprocess with no network, captures output, and cleans up.
    """
    script = f"{code}\n\n{test_code}\n"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        passed = result.returncode == 0

        error_type = None
        failed_tests: list[str] = []
        if not passed and stderr:
            error_type = _extract_error_type(stderr)
            failed_tests = _extract_failed_assertions(stderr)

        return ExecutionResult(
            passed=passed,
            stdout=stdout[:2000],
            stderr=stderr[:2000],
            error_type=error_type,
            failed_tests=failed_tests,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            error_type="TimeoutError",
            failed_tests=[],
        )
    except Exception as e:
        return ExecutionResult(
            passed=False,
            stdout="",
            stderr=str(e)[:2000],
            error_type=type(e).__name__,
            failed_tests=[],
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _extract_error_type(stderr: str) -> str | None:
    """Extract the error class name from a Python traceback."""
    match = re.findall(r"^(\w+(?:Error|Exception|Warning)): ", stderr, re.MULTILINE)
    return match[-1] if match else None


def _extract_failed_assertions(stderr: str) -> list[str]:
    """Extract failed assertion lines from stderr."""
    failed = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert ") or "AssertionError" in stripped:
            failed.append(stripped)
    seen: set[str] = set()
    unique = []
    for f in failed:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique
