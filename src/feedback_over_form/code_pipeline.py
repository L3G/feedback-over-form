"""Code generation pipeline with execution feedback loop.

Flow: generator -> executor -> [analyzer] -> refiner -> executor -> ... (up to max_iterations)
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel

from .code_executor import ExecutionResult, execute_code
from .code_neat_genome import CodeNeatGenome
from .inference_client import InferenceClient


class IterationRecord(BaseModel):
    """Record of one iteration through the generate/refine loop."""

    iteration: int
    code: str
    passed: bool
    error_type: str | None = None
    stderr_snippet: str = ""
    analysis: str | None = None


class CodePipelineResult(BaseModel):
    """Final result of running a code genome on a problem."""

    final_code: str
    passed: bool
    iterations_used: int
    iteration_history: list[IterationRecord]
    total_latency_ms: float


def _extract_code(text: str, entry_point: str | None = None) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if stripped.startswith(("def ", "class ", "import ", "from ")):
        return stripped

    match = re.search(r"(def \w+.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return stripped


def _build_feedback_prompt(
    problem_prompt: str,
    failed_code: str,
    result: ExecutionResult,
) -> str:
    """Build CYCLE-style feedback prompt for the refiner."""
    stderr_snippet = result.stderr[:1000] if result.stderr else "No error output"
    failed_str = "\n".join(result.failed_tests) if result.failed_tests else "See traceback above"

    return (
        f"Problem:\n{problem_prompt}\n\n"
        f"Previous code:\n```python\n{failed_code}\n```\n\n"
        f"Execution result:\n{stderr_snippet}\n\n"
        f"Failed tests:\n{failed_str}\n\n"
        f"Write the corrected function. Output ONLY the function code."
    )
