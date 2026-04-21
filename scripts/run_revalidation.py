#!/usr/bin/env python3
"""Re-validate pipeline configurations with repeated runs.

Runs each config N times on all 164 HumanEval problems to measure
mean/std/variance — separating real signal from LLM sampling noise.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feedback_over_form.code_executor import execute_code
from feedback_over_form.code_neat_genome import (
    CodeNeatGenome, LLMNodeConfig, RefinementStage,
)
from feedback_over_form.code_pipeline import _extract_code
from feedback_over_form.inference_client import InferenceClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_RUNS = 5


# ── Benchmark loaders ──────────────────────────────────────────────────

def load_humaneval(variant: str = "original", hard_only: bool = False) -> list[dict]:
    """Load HumanEval or HumanEval+ (much stricter test cases).

    Args:
        variant: "original" for OpenAI HumanEval, "plus" for EvalPlus HumanEval+
        hard_only: If True, filter to only problems where coder:3b self-refine fails
            (using the latest diagnostic file as the filter source).
    """
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
        })

    if hard_only:
        diag_files = sorted(
            RESULTS_DIR.glob("hard_problem_diagnostic_*.json"),
            reverse=True,
        )
        if not diag_files:
            raise FileNotFoundError(
                "No hard_problem_diagnostic_*.json found. Run diagnostics first."
            )
        with open(diag_files[0]) as f:
            diag = json.load(f)
        hard_ids = set(diag["hard_problem_ids"])
        problems = [p for p in problems if p["task_id"] in hard_ids]

    return problems


def _extract_function_name(assert_line: str) -> str:
    """Extract the function name from an MBPP assert line."""
    match = re.search(r"assert\s+(\w+)\s*\(", assert_line)
    return match.group(1) if match else "solution"


def load_mbpp() -> list[dict]:
    """Load MBPP sanitized — 427 human-reviewed problems across 4 splits."""
    from datasets import load_dataset
    ds_dict = load_dataset("google-research-datasets/mbpp", "sanitized")

    problems: list[dict] = []
    for split_name in ["train", "test", "validation", "prompt"]:
        if split_name not in ds_dict:
            continue
        for row in ds_dict[split_name]:
            task_id = row["task_id"]
            description = row["prompt"]
            test_list: list[str] = row["test_list"]
            test_imports: list[str] = row.get("test_imports") or []

            entry_point = _extract_function_name(test_list[0])

            tests_preview = "\n".join(test_list)
            llm_prompt = (
                f"Task: {description}\n\n"
                f"Your function should pass these tests:\n{tests_preview}\n\n"
                f"Write the Python function. Output ONLY the function code, "
                f"no explanations, no tests, no markdown."
            )

            imports_block = "\n".join(test_imports)
            test_code = (imports_block + "\n\n" if imports_block else "") + "\n".join(test_list) + "\n"

            problems.append({
                "task_id": f"MBPP/{task_id}",
                "prompt": llm_prompt,
                "test_code": test_code,
                "entry_point": entry_point,
            })
    return problems


def load_problems(benchmark: str = "humaneval", variant: str = "original") -> list[dict]:
    """Unified dispatcher: load HumanEval, HumanEval+, or MBPP sanitized."""
    if benchmark == "mbpp":
        return load_mbpp()
    return load_humaneval(variant=variant)


# ── Pipeline execution ─────────────────────────────────────────────────

def _build_feedback(prompt: str, code: str, stderr: str, failed_tests: list[str]) -> str:
    stderr_snippet = stderr[:1000] if stderr else "No error output"
    failed_str = "\n".join(failed_tests) if failed_tests else "See traceback above"
    return (
        f"Problem:\n{prompt}\n\n"
        f"Previous code:\n```python\n{code}\n```\n\n"
        f"Execution result:\n{stderr_snippet}\n\n"
        f"Failed tests:\n{failed_str}\n\n"
        f"Write the corrected function. Output ONLY the function code."
    )


async def execute_genome(
    genome: CodeNeatGenome,
    problem: dict,
    client: InferenceClient,
    max_tokens: int = 512,
    eval_temp: float | None = None,
) -> tuple[bool, int]:
    """Execute a code NEAT genome on a problem.

    Args:
        eval_temp: If set, overrides all node temperatures during this evaluation.
            The genome's stored temperatures are ignored for inference but remain
            in the genotype for mutation/crossover.

    Returns (passed, total_iterations_used).
    """
    prompt = problem["prompt"]
    test_code = problem["test_code"]
    entry_point = problem.get("entry_point")
    total_iters = 0

    def _temp(node_temp: float) -> float:
        return eval_temp if eval_temp is not None else node_temp

    # Step 1: Generator produces initial code
    raw = await client.generate(
        model=genome.generator.model,
        prompt=prompt,
        system_prompt=genome.generator.system_prompt,
        temperature=_temp(genome.generator.temperature),
        max_tokens=max_tokens,
    )
    current_code = _extract_code(raw, entry_point)

    # Step 2: Run through refinement stages
    for stage in genome.stages:
        for iteration in range(stage.max_iterations):
            total_iters += 1
            result = await execute_code(current_code, test_code)

            if result.passed:
                return True, total_iters

            # Build feedback
            feedback = _build_feedback(
                prompt, current_code, result.stderr, result.failed_tests,
            )

            # Optional analyzer
            if stage.analyzer:
                analysis = await client.generate(
                    model=stage.analyzer.model,
                    prompt=feedback,
                    system_prompt=stage.analyzer.system_prompt,
                    temperature=_temp(stage.analyzer.temperature),
                    max_tokens=max_tokens,
                )
                feedback += f"\n\nError analysis:\n{analysis}"

            # Refiner
            raw_refined = await client.generate(
                model=stage.refiner.model,
                prompt=feedback,
                system_prompt=stage.refiner.system_prompt,
                temperature=_temp(stage.refiner.temperature),
                max_tokens=max_tokens,
            )
            current_code = _extract_code(raw_refined, entry_point)

    # Final check after all stages
    total_iters += 1
    final_result = await execute_code(current_code, test_code)
    return final_result.passed, total_iters


# ── Config definitions ──────────────────────────────────────────────────

def llama_solo_config() -> tuple[str, CodeNeatGenome | None]:
    return ("llama3.2:3b solo", None)  # Solo runs handled specially


def qwen_exec_llama_config() -> tuple[str, CodeNeatGenome]:
    """Manual best from cross-model benchmark."""
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
    """The NEAT v3 champion -- the result we want to re-validate."""
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


def coder_solo_config() -> tuple[str, CodeNeatGenome | None]:
    return ("qwen2.5-coder:3b solo", None)


def coder_self_refine_config() -> tuple[str, CodeNeatGenome]:
    """qwen2.5-coder:3b self-refinement."""
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
    """qwen2.5-coder:3b self-refinement with iters=1.

    Based on the HumanEval iteration analysis finding: coder self-refine
    produces 0 new fixes at iter 2 and iter 3. Dropping to iters=1 cuts
    inference cost by ~2x with no measured performance loss.
    """
    _, base = coder_self_refine_config()
    base.stages[0].max_iterations = 1
    return ("qwen2.5-coder:3b self-refine (iters=1)", base)


# ── Runners ──────────────────────────────────────────────────────────

async def run_solo_once(
    client: InferenceClient, model: str, problems: list[dict],
    max_concurrent: int = 5, max_tokens: int = 512,
) -> list[bool]:
    """Run solo model on all problems, return pass/fail list."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_one(p: dict) -> bool:
        async with semaphore:
            try:
                raw = await client.generate(
                    model=model,
                    prompt=p["prompt"],
                    system_prompt=(
                        "You are an expert Python programmer. Given a problem description, "
                        "write a correct Python function. Output ONLY the function code, "
                        "no explanations, no tests, no markdown."
                    ),
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                code = _extract_code(raw, p.get("entry_point"))
                result = await execute_code(code, p["test_code"])
                return result.passed
            except Exception:
                return False

    return await asyncio.gather(*[eval_one(p) for p in problems])


async def run_genome_once(
    client: InferenceClient, genome: CodeNeatGenome, problems: list[dict],
    max_concurrent: int = 5, max_tokens: int = 512,
) -> list[bool]:
    """Run a genome pipeline on all problems, return pass/fail list."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_one(p: dict) -> bool:
        async with semaphore:
            try:
                ok, _ = await asyncio.wait_for(
                    execute_genome(genome, p, client, max_tokens),
                    timeout=120.0,
                )
                return ok
            except Exception:
                return False

    return await asyncio.gather(*[eval_one(p) for p in problems])


async def run_config_n_times(
    name: str,
    runner_fn,
    n_runs: int,
    problems: list[dict],
) -> dict:
    """Run a config n times, return pass counts and per-problem data."""
    print(f"\n  === {name} ===")
    pass_counts: list[int] = []
    per_problem_passes: list[list[bool]] = []  # [run][problem]

    for run_idx in range(n_runs):
        t0 = time.perf_counter()
        results = await runner_fn()
        elapsed = time.perf_counter() - t0
        count = sum(results)
        pass_counts.append(count)
        per_problem_passes.append(results)
        print(f"  Run {run_idx+1}/{n_runs}: {count}/{len(problems)} "
              f"({count/len(problems)*100:.1f}%) in {elapsed:.0f}s")

    return {
        "name": name,
        "pass_counts": pass_counts,
        "per_problem_passes": per_problem_passes,
        "mean": statistics.mean(pass_counts),
        "stdev": statistics.stdev(pass_counts) if len(pass_counts) > 1 else 0.0,
        "min": min(pass_counts),
        "max": max(pass_counts),
    }


def compute_consistency(per_problem_passes: list[list[bool]]) -> dict:
    """Compute per-problem consistency across runs.

    For each problem, count how many runs it passed in (0 to N).
    """
    n_runs = len(per_problem_passes)
    n_problems = len(per_problem_passes[0])
    pass_distribution = {i: 0 for i in range(n_runs + 1)}
    for prob_idx in range(n_problems):
        passes = sum(1 for run in per_problem_passes if run[prob_idx])
        pass_distribution[passes] += 1
    return pass_distribution


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Re-validate pipeline configs")
    parser.add_argument("--benchmark", type=str, default="humaneval",
                       choices=["humaneval", "mbpp"],
                       help="Which benchmark to run against")
    parser.add_argument("--configs", type=str, default="default",
                       help="'default' (6 HumanEval configs) or 'mbpp' (5 MBPP configs with coder iters=1) or comma-separated names")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                       help="Number of repeated runs per config")
    args = parser.parse_args()

    benchmark_label = "MBPP sanitized" if args.benchmark == "mbpp" else "HumanEval"
    print(f"\n  Loading {benchmark_label} problems...")
    problems = load_problems(benchmark=args.benchmark)
    print(f"  Loaded {len(problems)} problems")
    print(f"  Runs per config: {args.runs}")

    client = InferenceClient(backend="ollama")

    # Config presets
    presets = {
        "default": [
            llama_solo_config(),
            qwen_exec_llama_config(),
            llama_exec_llama_config(),
            neat_v3_champion_config(),
            coder_solo_config(),
            coder_self_refine_config(),
        ],
        "mbpp": [
            llama_solo_config(),
            llama_exec_llama_config(),
            qwen_exec_llama_config(),
            coder_solo_config(),
            coder_self_refine_iters1_config(),
        ],
    }
    configs = presets.get(args.configs, presets["default"])
    print(f"  Total evaluations: {len(problems) * args.runs * len(configs)}")

    all_results: list[dict] = []

    try:
        for name, genome in configs:
            if genome is None:
                # Solo run
                model = "llama3.2:3b" if "llama" in name else "qwen2.5-coder:3b"
                async def runner(m=model):
                    return await run_solo_once(client, m, problems)
            else:
                async def runner(g=genome):
                    return await run_genome_once(client, g, problems)

            result = await run_config_n_times(name, runner, args.runs, problems)
            all_results.append(result)

    finally:
        await client.close()

    # ────────── Print results table ──────────
    print(f"\n  {'=' * 85}")
    print(f"  RE-VALIDATION RESULTS -- {args.runs} runs x {len(problems)} problems ({benchmark_label})")
    print(f"  {'=' * 85}")

    header = f"  {'Config':<36}"
    for i in range(args.runs):
        header += f" {'Run'+str(i+1):>6}"
    header += f" {'Mean':>10} {'Std':>6} {'Min':>5} {'Max':>5}"
    print(f"\n{header}")
    print("  " + "-" * (36 + args.runs * 7 + 28))

    for r in all_results:
        line = f"  {r['name']:<36}"
        for count in r['pass_counts']:
            line += f" {count:>6}"
        line += f" {r['mean']:>6.1f}/{len(problems):<3} {r['stdev']:>6.2f} {r['min']:>5} {r['max']:>5}"
        print(line)

    # ────────── Per-problem consistency ──────────
    print(f"\n  === Per-problem consistency ===")
    print(f"  How many runs each problem passes in (0=always fails, {args.runs}=always passes):")
    print(f"\n  {'Config':<36}", end="")
    for i in range(args.runs + 1):
        print(f" {i:>5}", end="")
    print(f" {'Noisy':>8}")
    print("  " + "-" * (36 + (args.runs + 1) * 6 + 9))

    for r in all_results:
        dist = compute_consistency(r["per_problem_passes"])
        line = f"  {r['name']:<36}"
        for i in range(args.runs + 1):
            line += f" {dist[i]:>5}"
        noisy = sum(dist[i] for i in range(1, args.runs))
        line += f" {noisy:>8}"
        print(line)

    # ────────── NEAT v3 champion verdict (HumanEval only) ──────────
    neat_result = next((r for r in all_results if "NEAT v3" in r["name"]), None)
    if neat_result and args.benchmark == "humaneval":
        print(f"\n  === NEAT v3 Champion Verdict ===")
        print(f"  Original reported result: 106/164")
        print(f"  Re-validation mean: {neat_result['mean']:.1f}/{len(problems)}")
        print(f"  Re-validation range: {neat_result['min']}-{neat_result['max']}")
        print(f"  Re-validation std: {neat_result['stdev']:.2f}")
        # Is 106 within 1 std of the mean?
        in_range = abs(106 - neat_result['mean']) <= neat_result['stdev'] * 2
        print(f"  106 within 2-sigma of re-validation mean: {in_range}")
        if neat_result['min'] <= 106 <= neat_result['max']:
            print(f"  106 is within the observed range -- consistent with noise")
        elif 106 > neat_result['max']:
            print(f"  106 exceeds all re-validation runs (max={neat_result['max']}) -- original was a lucky outlier")
        else:
            print(f"  106 is below all re-validation runs -- original was unlucky")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "mbpp" if args.benchmark == "mbpp" else "humaneval"
    output_file = RESULTS_DIR / f"revalidation_{suffix}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": args.benchmark,
            "n_runs": args.runs,
            "total_problems": len(problems),
            "configs": [
                {
                    "name": r["name"],
                    "pass_counts": r["pass_counts"],
                    "mean": r["mean"],
                    "stdev": r["stdev"],
                    "min": r["min"],
                    "max": r["max"],
                    "consistency": compute_consistency(r["per_problem_passes"]),
                }
                for r in all_results
            ],
        }, f, indent=2)
    print(f"\n  Saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
