#!/usr/bin/env python3
"""Error taxonomy analysis: what can self-refinement fix, what can't it?

Runs instrumented pipeline execution to capture error types at each iteration,
then analyzes:
- Fix rates by error type (what refinement can repair)
- Unfixable problems (failed across all configs)
- Regression patterns (solo passes but pipeline fails)

Usage:
    python scripts/run_error_analysis.py
    python scripts/run_error_analysis.py --benchmark mbpp
    python scripts/run_error_analysis.py --configs custom --benchmark humaneval
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys

# ── Path setup ─────────────────────────────────────────────────────────
# Add src/ so we can import the feedback_over_form package directly.
# Add scripts/ so we can import sibling scripts (run_code_neat, run_revalidation).

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from feedback_over_form.code_executor import execute_code, ExecutionResult
from feedback_over_form.code_neat_genome import (
    CodeNeatGenome,
    LLMNodeConfig,
    RefinementStage,
)
from feedback_over_form.code_pipeline import _extract_code
from feedback_over_form.inference_client import InferenceClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Benchmark loader ───────────────────────────────────────────────────
# Try to import from sibling scripts first; fall back to inline loaders.

def _try_import_loaders():
    """Attempt to import load_humaneval / load_problems from run_code_neat."""
    try:
        from run_code_neat import load_humaneval, load_problems
        return load_humaneval, load_problems
    except ImportError:
        pass
    return None, None


_imported_load_humaneval, _imported_load_problems = _try_import_loaders()


def load_humaneval(variant: str = "original", hard_only: bool = False) -> list[dict]:
    """Load HumanEval problems from HuggingFace datasets."""
    if _imported_load_humaneval is not None:
        return _imported_load_humaneval(variant=variant, hard_only=hard_only)

    from datasets import load_dataset

    if variant == "plus":
        ds = load_dataset("evalplus/humanevalplus", split="test")
    else:
        ds = load_dataset("openai/openai_humaneval", split="test")

    problems = []
    for row in ds:
        entry_point = row["entry_point"]
        test_code = row["test"] + f"\ncheck({entry_point})\n"
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test_code": test_code,
            "entry_point": entry_point,
            "canonical_solution": row.get("canonical_solution", ""),
        })
    return problems


def load_mbpp() -> list[dict]:
    """Load MBPP sanitized problems from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for row in ds:
        test_code = "\n".join(row["test_list"])
        problems.append({
            "task_id": f"Mbpp/{row['task_id']}",
            "prompt": row["prompt"],
            "test_code": test_code,
            "entry_point": None,
        })
    return problems


def load_problems(benchmark: str = "humaneval", variant: str = "original") -> list[dict]:
    """Unified dispatcher: load HumanEval, HumanEval+, or MBPP sanitized."""
    if _imported_load_problems is not None:
        return _imported_load_problems(benchmark=benchmark, variant=variant)
    if benchmark == "mbpp":
        return load_mbpp()
    return load_humaneval(variant=variant)


# ── Config presets ─────────────────────────────────────────────────────
# Try to import from run_revalidation (sibling script) for config definitions.

def _try_import_configs():
    """Attempt to import config factories from run_revalidation."""
    try:
        from run_revalidation import (
            qwen_exec_llama_config,
            llama_exec_llama_config,
            neat_v3_champion_config,
            coder_self_refine_config,
            coder_self_refine_iters1_config,
        )
        return {
            "qwen_exec_llama": qwen_exec_llama_config,
            "llama_exec_llama": llama_exec_llama_config,
            "neat_v3_champion": neat_v3_champion_config,
            "coder_self_refine": coder_self_refine_config,
            "coder_self_refine_iters1": coder_self_refine_iters1_config,
        }
    except ImportError:
        return {}


_config_factories = _try_import_configs()


def qwen_exec_llama_config() -> tuple[str, CodeNeatGenome]:
    """Manual best from cross-model benchmark: qwen generates, llama refines."""
    if "qwen_exec_llama" in _config_factories:
        return _config_factories["qwen_exec_llama"]()
    genome = CodeNeatGenome(
        generator=LLMNodeConfig(
            model="qwen2.5:1.5b",
            system_prompt=(
                "You are an expert Python programmer. Given a problem description, "
                "write a correct Python function. Output ONLY the function code, "
                "no explanations, no tests, no markdown."
            ),
            temperature=0.3,
        ),
        stages=[
            RefinementStage(
                innovation=1,
                max_iterations=3,
                analyzer=None,
                refiner=LLMNodeConfig(
                    model="llama3.2:3b",
                    system_prompt=(
                        "You are a Python debugger. Given a failed function, the error output, "
                        "and optionally an error analysis, write a corrected version of the "
                        "function. Output ONLY the corrected function code, no explanations."
                    ),
                    temperature=0.2,
                ),
            ),
        ],
    )
    return ("qwen->exec->llama", genome)


def llama_exec_llama_config() -> tuple[str, CodeNeatGenome]:
    """llama3.2:3b self-refinement."""
    if "llama_exec_llama" in _config_factories:
        return _config_factories["llama_exec_llama"]()
    genome = CodeNeatGenome(
        generator=LLMNodeConfig(
            model="llama3.2:3b",
            system_prompt=(
                "You are an expert Python programmer. Given a problem description, "
                "write a correct Python function. Output ONLY the function code, "
                "no explanations, no tests, no markdown."
            ),
            temperature=0.3,
        ),
        stages=[
            RefinementStage(
                innovation=1,
                max_iterations=3,
                analyzer=None,
                refiner=LLMNodeConfig(
                    model="llama3.2:3b",
                    system_prompt=(
                        "You are a Python debugger. Given a failed function, the error output, "
                        "and optionally an error analysis, write a corrected version of the "
                        "function. Output ONLY the corrected function code, no explanations."
                    ),
                    temperature=0.2,
                ),
            ),
        ],
    )
    return ("llama->exec->llama", genome)


def neat_v3_champion_config() -> tuple[str, CodeNeatGenome]:
    """Load the NEAT v3 champion from results/neat_runs/v3_champion.json."""
    if "neat_v3_champion" in _config_factories:
        return _config_factories["neat_v3_champion"]()
    champ_path = RESULTS_DIR / "neat_runs" / "v3_champion.json"
    with open(champ_path) as f:
        data = json.load(f)
    g = data["genome"]

    stages = []
    for s in g["stages"]:
        stages.append(RefinementStage(
            innovation=s["innovation"],
            max_iterations=s["max_iterations"],
            analyzer=LLMNodeConfig(
                model=s["analyzer"]["model"],
                system_prompt=s["analyzer"]["system_prompt"],
                temperature=s["analyzer"]["temperature"],
            ) if s.get("analyzer") else None,
            refiner=LLMNodeConfig(
                model=s["refiner"]["model"],
                system_prompt=s["refiner"]["system_prompt"],
                temperature=s["refiner"]["temperature"],
            ),
        ))

    genome = CodeNeatGenome(
        generator=LLMNodeConfig(
            model=g["generator"]["model"],
            system_prompt=g["generator"]["system_prompt"],
            temperature=g["generator"]["temperature"],
        ),
        stages=stages,
    )
    return ("NEAT v3 champion", genome)


def coder_self_refine_config() -> tuple[str, CodeNeatGenome]:
    """qwen2.5-coder:3b self-refinement."""
    if "coder_self_refine" in _config_factories:
        return _config_factories["coder_self_refine"]()
    genome = CodeNeatGenome(
        generator=LLMNodeConfig(
            model="qwen2.5-coder:3b",
            system_prompt=(
                "You are an expert Python programmer. Given a problem description, "
                "write a correct Python function. Output ONLY the function code, "
                "no explanations, no tests, no markdown."
            ),
            temperature=0.3,
        ),
        stages=[
            RefinementStage(
                innovation=1,
                max_iterations=3,
                analyzer=None,
                refiner=LLMNodeConfig(
                    model="qwen2.5-coder:3b",
                    system_prompt=(
                        "You are a Python debugger. Given a failed function, the error output, "
                        "write a corrected version. Output ONLY the corrected function code."
                    ),
                    temperature=0.2,
                ),
            ),
        ],
    )
    return ("qwen2.5-coder:3b self-refine", genome)


def coder_self_refine_iters1_config() -> tuple[str, CodeNeatGenome]:
    """qwen2.5-coder:3b self-refinement with iters=1 (cost-optimal)."""
    if "coder_self_refine_iters1" in _config_factories:
        return _config_factories["coder_self_refine_iters1"]()
    _, base = coder_self_refine_config()
    base.stages[0].max_iterations = 1
    return ("qwen2.5-coder:3b self-refine (iters=1)", base)


# ── Instrumented pipeline execution ───────────────────────────────────

def _build_feedback(
    prompt: str, code: str, stderr: str, failed_tests: list[str],
) -> str:
    """Build a feedback prompt combining the problem, failed code, and errors."""
    stderr_snippet = stderr[:1000] if stderr else "No error output"
    failed_str = "\n".join(failed_tests) if failed_tests else "See traceback above"
    return (
        f"Problem:\n{prompt}\n\n"
        f"Previous code:\n```python\n{code}\n```\n\n"
        f"Execution result:\n{stderr_snippet}\n\n"
        f"Failed tests:\n{failed_str}\n\n"
        f"Write the corrected function. Output ONLY the function code."
    )


async def execute_genome_instrumented(
    genome: CodeNeatGenome,
    problem: dict,
    client: InferenceClient,
    max_tokens: int = 512,
) -> dict:
    """Run a genome and return detailed iteration history with error types.

    Unlike a normal execution that stops on first pass, this captures the
    full error trajectory so we can analyze what refinement actually fixes.

    Returns:
        dict with keys: passed, final_error_type, initial_error_type, trajectory
        where trajectory is a list of {iteration, passed, error_type} dicts.
    """
    prompt = problem["prompt"]
    test_code = problem["test_code"]
    entry_point = problem.get("entry_point")

    trajectory: list[dict] = []

    # Step 1: Generate initial code
    try:
        raw = await client.generate(
            model=genome.generator.model,
            prompt=prompt,
            system_prompt=genome.generator.system_prompt,
            temperature=genome.generator.temperature,
            max_tokens=max_tokens,
        )
        current_code = _extract_code(raw, entry_point)
    except Exception as e:
        return {
            "passed": False,
            "final_error_type": type(e).__name__,
            "trajectory": [],
            "initial_error_type": "GenerationFailure",
        }

    # Step 2: Iterate through refinement stages
    iter_num = 0
    for stage in genome.stages:
        for _ in range(stage.max_iterations):
            iter_num += 1

            # Execute current code against tests
            try:
                result = await execute_code(current_code, test_code)
            except Exception as e:
                trajectory.append({
                    "iteration": iter_num,
                    "passed": False,
                    "error_type": type(e).__name__,
                })
                break

            trajectory.append({
                "iteration": iter_num,
                "passed": result.passed,
                "error_type": result.error_type if not result.passed else None,
            })

            # Early exit on success
            if result.passed:
                return {
                    "passed": True,
                    "trajectory": trajectory,
                    "initial_error_type": (
                        trajectory[0]["error_type"]
                        if len(trajectory) > 1
                        else None
                    ),
                    "final_error_type": None,
                }

            # Build feedback for refinement
            feedback = _build_feedback(
                prompt, current_code, result.stderr, result.failed_tests,
            )

            # Optional analyzer pass
            if stage.analyzer:
                try:
                    analysis = await client.generate(
                        model=stage.analyzer.model,
                        prompt=feedback,
                        system_prompt=stage.analyzer.system_prompt,
                        temperature=stage.analyzer.temperature,
                        max_tokens=max_tokens,
                    )
                    feedback += f"\n\nError analysis:\n{analysis}"
                except Exception:
                    pass  # Analyzer failure is non-fatal

            # Refiner produces corrected code
            try:
                raw_refined = await client.generate(
                    model=stage.refiner.model,
                    prompt=feedback,
                    system_prompt=stage.refiner.system_prompt,
                    temperature=stage.refiner.temperature,
                    max_tokens=max_tokens,
                )
                current_code = _extract_code(raw_refined, entry_point)
            except Exception:
                break  # Refinement generation failed, stop this stage

    # Step 3: Final execution after all stages
    iter_num += 1
    try:
        final_result = await execute_code(current_code, test_code)
        trajectory.append({
            "iteration": iter_num,
            "passed": final_result.passed,
            "error_type": final_result.error_type if not final_result.passed else None,
        })
        return {
            "passed": final_result.passed,
            "trajectory": trajectory,
            "initial_error_type": trajectory[0]["error_type"] if trajectory else None,
            "final_error_type": (
                final_result.error_type if not final_result.passed else None
            ),
        }
    except Exception as e:
        return {
            "passed": False,
            "trajectory": trajectory,
            "initial_error_type": trajectory[0]["error_type"] if trajectory else None,
            "final_error_type": type(e).__name__,
        }


async def run_solo_instrumented(
    client: InferenceClient,
    model: str,
    problem: dict,
    max_tokens: int = 512,
) -> dict:
    """Run a solo model once, returning pass/fail + error type.

    Solo baselines have no refinement -- just generate and test once.
    Used as the control group for regression analysis.
    """
    try:
        raw = await client.generate(
            model=model,
            prompt=problem["prompt"],
            system_prompt=(
                "You are an expert Python programmer. Given a problem description, "
                "write a correct Python function. Output ONLY the function code, "
                "no explanations, no tests, no markdown."
            ),
            temperature=0.3,
            max_tokens=max_tokens,
        )
        code = _extract_code(raw, problem.get("entry_point"))
        result = await execute_code(code, problem["test_code"])
        return {
            "passed": result.passed,
            "final_error_type": result.error_type if not result.passed else None,
            "initial_error_type": result.error_type if not result.passed else None,
            "trajectory": [{
                "iteration": 1,
                "passed": result.passed,
                "error_type": result.error_type if not result.passed else None,
            }],
        }
    except Exception as e:
        return {
            "passed": False,
            "final_error_type": type(e).__name__,
            "initial_error_type": type(e).__name__,
            "trajectory": [],
        }


# ── Config runner ──────────────────────────────────────────────────────

async def run_config(
    client: InferenceClient,
    config_name: str,
    problems: list[dict],
    genome: CodeNeatGenome | None = None,
    solo_model: str | None = None,
    max_concurrent: int = 5,
) -> list[dict]:
    """Run one config on all problems with error instrumentation.

    For pipeline configs (genome is not None), uses execute_genome_instrumented.
    For solo configs (solo_model is not None), uses run_solo_instrumented.

    Returns list of per-problem result dicts.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(p: dict) -> dict:
        async with semaphore:
            try:
                if genome:
                    result = await asyncio.wait_for(
                        execute_genome_instrumented(genome, p, client),
                        timeout=120.0,
                    )
                else:
                    result = await asyncio.wait_for(
                        run_solo_instrumented(client, solo_model, p),
                        timeout=120.0,
                    )
                result["task_id"] = p["task_id"]
                return result
            except asyncio.TimeoutError:
                return {
                    "task_id": p["task_id"],
                    "passed": False,
                    "initial_error_type": "TimeoutError",
                    "final_error_type": "TimeoutError",
                    "trajectory": [],
                }

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(p) for p in problems])
    elapsed = time.perf_counter() - t0
    passed = sum(1 for r in results if r["passed"])
    print(
        f"  {config_name}: {passed}/{len(problems)} "
        f"({passed / len(problems) * 100:.1f}%) in {elapsed:.0f}s"
    )
    return results


# ── Error categorization ──────────────────────────────────────────────

def categorize_error(error_type: str | None) -> str:
    """Normalize error types into coarse categories for analysis."""
    if error_type is None:
        return "None"
    if error_type in ("SyntaxError", "IndentationError"):
        return "SyntaxError"
    if error_type == "NameError":
        return "NameError"
    if error_type == "TypeError":
        return "TypeError"
    if error_type == "AssertionError":
        return "AssertionError"
    if error_type in ("IndexError", "KeyError"):
        return "IndexError/KeyError"
    if error_type == "TimeoutError":
        return "TimeoutError"
    if error_type == "ValueError":
        return "ValueError"
    if error_type == "AttributeError":
        return "AttributeError"
    return "Other"


# ── Analysis functions ────────────────────────────────────────────────

def compute_fix_rates(results: list[dict]) -> dict[str, dict]:
    """Compute fix rate by initial error type.

    A 'fix' means the first iteration had error X, but the final result passed.
    This tells us which error types are amenable to self-refinement.

    Returns:
        dict mapping error category -> {count, fixed, fix_rate}
    """
    stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "fixed": 0})

    for r in results:
        if not r["trajectory"]:
            continue
        # First iteration error (what the refiner saw first)
        first_iter = r["trajectory"][0]
        if first_iter["passed"]:
            continue  # Passed immediately, nothing to fix
        initial_error = categorize_error(first_iter.get("error_type"))
        stats[initial_error]["count"] += 1
        if r["passed"]:
            stats[initial_error]["fixed"] += 1

    # Add fix_rate
    for err, s in stats.items():
        s["fix_rate"] = s["fixed"] / s["count"] if s["count"] > 0 else 0

    return dict(stats)


def find_unfixable(all_config_results: dict[str, list[dict]]) -> dict:
    """Find problems that failed in EVERY config.

    These are problems that no pipeline topology or model combination could solve.

    Returns:
        dict mapping task_id -> {error_types: [...], dominant_error: str}
    """
    problem_results: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for config_name, results in all_config_results.items():
        for r in results:
            problem_results[r["task_id"]].append(
                (config_name, r.get("final_error_type"))
            )

    unfixable: dict[str, dict] = {}
    for task_id, entries in problem_results.items():
        if all(e[1] is not None for e in entries):  # Failed in every config
            error_types = [categorize_error(e[1]) for e in entries]
            unfixable[task_id] = {
                "error_types": error_types,
                "dominant_error": Counter(error_types).most_common(1)[0][0],
            }
    return unfixable


def find_regressions(
    solo_results: list[dict],
    pipeline_results: list[dict],
) -> list[dict]:
    """Find problems where solo passed but pipeline failed.

    These regressions indicate that the refinement loop is actually *hurting*
    on certain problems -- the generator got it right but refinement broke it.

    Returns:
        list of {task_id, final_error_type, trajectory_length} dicts
    """
    solo_map = {r["task_id"]: r for r in solo_results}
    regressions = []
    for pr in pipeline_results:
        sr = solo_map.get(pr["task_id"])
        if sr and sr["passed"] and not pr["passed"]:
            regressions.append({
                "task_id": pr["task_id"],
                "final_error_type": categorize_error(pr.get("final_error_type")),
                "trajectory_length": len(pr["trajectory"]),
            })
    return regressions


# ── Pretty printing ───────────────────────────────────────────────────

def print_fix_rates(
    all_results: dict[str, list[dict]],
    pipeline_configs: list[str],
) -> None:
    """Print Part A: fix rates by initial error type."""
    print("  === A. Fix rates by initial error type (what refinement can fix) ===\n")
    for name in pipeline_configs:
        if name not in all_results:
            continue
        stats = compute_fix_rates(all_results[name])
        print(f"  {name}:")
        print(f"    {'Error Type':<25} {'Count':>7} {'Fixed':>7} {'Rate':>8}")
        print(f"    {'-' * 25} {'-' * 7} {'-' * 7} {'-' * 8}")
        for err, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
            print(
                f"    {err:<25} {s['count']:>7} {s['fixed']:>7} "
                f"{s['fix_rate'] * 100:>7.1f}%"
            )
        print()


def print_unfixable(
    all_results: dict[str, list[dict]],
    n_problems: int,
) -> None:
    """Print Part B: unfixable problems."""
    print("  === B. Unfixable problems (failed in every config) ===\n")
    unfixable = find_unfixable(all_results)
    print(f"  Total: {len(unfixable)}/{n_problems} problems unfixable by any config\n")

    error_counter: Counter = Counter()
    for task_id, info in unfixable.items():
        error_counter[info["dominant_error"]] += 1

    print(f"    {'Error Type':<25} {'Count':>7}")
    print(f"    {'-' * 25} {'-' * 7}")
    for err, count in error_counter.most_common():
        print(f"    {err:<25} {count:>7}")
    print()
    return unfixable, error_counter


def print_regressions(
    all_results: dict[str, list[dict]],
    configs: list[tuple[str, CodeNeatGenome | None, str | None]],
) -> None:
    """Print Part C: regression patterns."""
    print("  === C. Regression patterns (solo passes, pipeline fails) ===\n")

    pipeline_configs = [name for name, genome, _ in configs if genome is not None]
    llama_solo = "llama3.2:3b solo"
    coder_solo = "qwen2.5-coder:3b solo"
    solo_pairs = []

    for pname in pipeline_configs:
        if "coder" in pname.lower() and coder_solo in all_results:
            solo_pairs.append((coder_solo, pname))
        elif llama_solo in all_results:
            solo_pairs.append((llama_solo, pname))

    for solo_name, pipeline_name in solo_pairs:
        if solo_name not in all_results or pipeline_name not in all_results:
            continue
        regressions = find_regressions(
            all_results[solo_name], all_results[pipeline_name],
        )
        print(f"  {pipeline_name} vs {solo_name}:")
        print(f"    {len(regressions)} regressions (solo passed, pipeline failed)")
        if regressions:
            err_dist = Counter(r["final_error_type"] for r in regressions)
            for err, count in err_dist.most_common():
                print(f"      {err:<25} {count}")
            avg_iters = (
                sum(r["trajectory_length"] for r in regressions) / len(regressions)
            )
            print(f"    Avg iterations before final fail: {avg_iters:.1f}")
        print()


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Error taxonomy analysis: what can self-refinement fix?",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="Which benchmark to analyze (default: humaneval)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="default",
        help=(
            "'default' (HumanEval 6 configs), "
            "'mbpp' (5 configs with coder iters=1), "
            "or comma-separated config names"
        ),
    )
    args = parser.parse_args()

    # ── Load benchmark ──
    benchmark_label = "MBPP sanitized" if args.benchmark == "mbpp" else "HumanEval"
    print(f"\n  Loading {benchmark_label} problems...")
    problems = load_problems(benchmark=args.benchmark)
    print(f"  Loaded {len(problems)} problems\n")

    # ── Build inference client ──
    client = InferenceClient(backend="ollama")

    # ── Define config presets ──
    # Each entry: (display_name, genome_or_None, solo_model_or_None)
    presets: dict[str, list[tuple[str, CodeNeatGenome | None, str | None]]] = {
        "default": [
            ("llama3.2:3b solo", None, "llama3.2:3b"),
            (qwen_exec_llama_config()[0], qwen_exec_llama_config()[1], None),
            (llama_exec_llama_config()[0], llama_exec_llama_config()[1], None),
            (neat_v3_champion_config()[0], neat_v3_champion_config()[1], None),
            ("qwen2.5-coder:3b solo", None, "qwen2.5-coder:3b"),
            (coder_self_refine_config()[0], coder_self_refine_config()[1], None),
        ],
        "mbpp": [
            ("llama3.2:3b solo", None, "llama3.2:3b"),
            (llama_exec_llama_config()[0], llama_exec_llama_config()[1], None),
            (qwen_exec_llama_config()[0], qwen_exec_llama_config()[1], None),
            ("qwen2.5-coder:3b solo", None, "qwen2.5-coder:3b"),
            (
                coder_self_refine_iters1_config()[0],
                coder_self_refine_iters1_config()[1],
                None,
            ),
        ],
    }
    configs = presets.get(args.configs, presets["default"])

    # ── Run all configs with error instrumentation ──
    all_results: dict[str, list[dict]] = {}
    print("  === Running all configs with error instrumentation ===\n")

    try:
        for name, genome, solo_model in configs:
            results = await run_config(client, name, problems, genome, solo_model)
            all_results[name] = results
    finally:
        await client.close()

    # ── Analysis ──
    print(f"\n  {'=' * 85}")
    print(f"  ERROR TAXONOMY ANALYSIS")
    print(f"  {'=' * 85}\n")

    # Part A: Fix rates by error type (for pipeline configs only)
    pipeline_config_names = [name for name, genome, _ in configs if genome is not None]
    print_fix_rates(all_results, pipeline_config_names)

    # Part B: Unfixable errors
    unfixable, error_counter = print_unfixable(all_results, len(problems))

    # Part C: Regressions (solo passes, pipeline fails)
    print_regressions(all_results, configs)

    # ── Save full analysis ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "mbpp" if args.benchmark == "mbpp" else "humaneval"
    output_file = RESULTS_DIR / f"error_taxonomy_{suffix}_{timestamp}.json"

    analysis = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "total_problems": len(problems),
        "configs": {
            name: {
                "passed": sum(1 for r in results if r["passed"]),
                "total": len(results),
                "fix_rates": compute_fix_rates(results) if genome else None,
                "per_problem": [
                    {
                        "task_id": r["task_id"],
                        "passed": r["passed"],
                        "initial_error_type": r.get("initial_error_type"),
                        "final_error_type": r.get("final_error_type"),
                        "trajectory_length": len(r.get("trajectory", [])),
                    }
                    for r in results
                ],
            }
            for (name, genome, _), results in zip(
                configs, [all_results[c[0]] for c in configs]
            )
        },
        "unfixable": unfixable,
        "unfixable_error_distribution": dict(error_counter),
    }

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
