#!/usr/bin/env python3
"""Benchmark code generation: pipeline with execution feedback vs solo model.

Uses HumanEval (164 problems) to measure pass@1 for:
- Solo model: single generation, no feedback
- Pipeline: generate -> execute -> refine loop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feedback_over_form.code_executor import execute_code
from feedback_over_form.code_neat_genome import (
    CodeNeatGenome, LLMNodeConfig, RefinementStage,
)
from feedback_over_form.code_pipeline import _extract_code, _build_feedback_prompt
from feedback_over_form.inference_client import InferenceClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_humaneval() -> list[dict]:
    """Load HumanEval problems from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for row in ds:
        # HumanEval test code calls check(candidate_fn), so we need to
        # wire up the entry point
        entry_point = row["entry_point"]
        test_code = row["test"] + f"\ncheck({entry_point})\n"

        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test_code": test_code,
            "entry_point": entry_point,
            "canonical_solution": row["canonical_solution"],
        })
    return problems


async def run_solo(
    client: InferenceClient,
    model: str,
    problem: dict,
    max_tokens: int = 512,
) -> dict:
    """Run a single model on a problem with no refinement."""
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]

    t0 = time.perf_counter()
    raw = await client.generate(
        model=model,
        prompt=prompt,
        system_prompt=(
            "You are an expert Python programmer. Given a problem description, "
            "write a correct Python function. Output ONLY the function code, "
            "no explanations, no tests, no markdown."
        ),
        temperature=0.3,
        max_tokens=max_tokens,
    )
    latency = (time.perf_counter() - t0) * 1000

    code = _extract_code(raw, entry_point)
    exec_result = await execute_code(code, problem["test_code"])

    return {
        "task_id": problem["task_id"],
        "passed": exec_result.passed,
        "code": code,
        "error_type": exec_result.error_type,
        "stderr_snippet": exec_result.stderr[:300],
        "latency_ms": round(latency),
    }


async def run_pipeline(
    client: InferenceClient,
    genome: CodeNeatGenome,
    problem: dict,
    max_tokens: int = 512,
) -> dict:
    """Run a CodeNeatGenome pipeline on a problem with execution feedback."""
    prompt = problem["prompt"]
    test_code = problem["test_code"]
    entry_point = problem.get("entry_point")

    t0 = time.perf_counter()
    iteration_history = []
    total_iters = 0

    # Step 1: Generator produces initial code
    raw = await client.generate(
        model=genome.generator.model,
        prompt=prompt,
        system_prompt=genome.generator.system_prompt,
        temperature=genome.generator.temperature,
        max_tokens=max_tokens,
    )
    current_code = _extract_code(raw, entry_point)

    # Step 2: Run through refinement stages
    for stage in genome.stages:
        for iteration in range(stage.max_iterations):
            total_iters += 1
            exec_result = await execute_code(current_code, test_code)

            iteration_history.append({
                "iteration": total_iters,
                "passed": exec_result.passed,
                "error_type": exec_result.error_type,
            })

            if exec_result.passed:
                latency = (time.perf_counter() - t0) * 1000
                return {
                    "task_id": problem["task_id"],
                    "passed": True,
                    "code": current_code,
                    "iterations_used": total_iters,
                    "iteration_history": iteration_history,
                    "latency_ms": round(latency),
                }

            # Build feedback and refine
            feedback = _build_feedback_prompt(prompt, current_code, exec_result)

            # Optional analyzer
            if stage.analyzer:
                analysis = await client.generate(
                    model=stage.analyzer.model,
                    prompt=feedback,
                    system_prompt=stage.analyzer.system_prompt,
                    temperature=stage.analyzer.temperature,
                    max_tokens=max_tokens,
                )
                feedback += f"\n\nError analysis:\n{analysis}"

            # Refiner
            raw_refined = await client.generate(
                model=stage.refiner.model,
                prompt=feedback,
                system_prompt=stage.refiner.system_prompt,
                temperature=stage.refiner.temperature,
                max_tokens=max_tokens,
            )
            current_code = _extract_code(raw_refined, entry_point)

    # Final check after all stages
    total_iters += 1
    final_result = await execute_code(current_code, test_code)
    iteration_history.append({
        "iteration": total_iters,
        "passed": final_result.passed,
        "error_type": final_result.error_type,
    })

    latency = (time.perf_counter() - t0) * 1000
    return {
        "task_id": problem["task_id"],
        "passed": final_result.passed,
        "code": current_code,
        "iterations_used": total_iters,
        "iteration_history": iteration_history,
        "latency_ms": round(latency),
    }


def make_default_genome(model: str) -> CodeNeatGenome:
    """Create a simple gen->exec->refine genome using CodeNeatGenome.

    Uses a single RefinementStage with 3 iterations -- the simplest
    pipeline topology that still demonstrates execution feedback.
    """
    return CodeNeatGenome(
        generator=LLMNodeConfig(
            model=model,
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
                    model=model,
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


def print_results(
    solo_results: list[dict],
    pipeline_results: list[dict],
    solo_model: str,
    genome: CodeNeatGenome,
) -> None:
    """Print pass@1, comparison, and details."""
    n = len(solo_results)
    solo_pass = sum(1 for r in solo_results if r["passed"])
    pipe_pass = sum(1 for r in pipeline_results if r["passed"])

    # Problems fixed by pipeline that solo got wrong
    solo_ids = {r["task_id"] for r in solo_results if r["passed"]}
    pipe_ids = {r["task_id"] for r in pipeline_results if r["passed"]}
    fixed = pipe_ids - solo_ids
    regressed = solo_ids - pipe_ids

    # Avg iterations for pipeline
    avg_iters = sum(r["iterations_used"] for r in pipeline_results) / n if n else 0

    # Topology string: gen_model -> exec -> refiner_model
    stage = genome.stages[0] if genome.stages else None
    topology = f"{genome.generator.model}"
    if stage:
        topology += f" -> exec -> {stage.refiner.model}"

    print(f"\n  {'='*60}")
    print(f"  CODE BENCHMARK RESULTS")
    print(f"  {'='*60}")
    print(f"  Problems: {n}")
    print(f"  Solo model: {solo_model}")
    print(f"  Pipeline: {topology}")
    print(f"\n  {'Metric':<30} {'Solo':>10} {'Pipeline':>10}")
    print(f"  {'-'*30} {'-'*10:>10} {'-'*10:>10}")
    print(f"  {'pass@1':<30} {solo_pass:>10} ({solo_pass/n*100:.1f}%) {pipe_pass:>10} ({pipe_pass/n*100:.1f}%)")
    print(f"  {'avg latency (ms)':<30} {sum(r['latency_ms'] for r in solo_results)/n:>10.0f} {sum(r['latency_ms'] for r in pipeline_results)/n:>10.0f}")
    print(f"  {'avg iterations':<30} {'1':>10} {avg_iters:>10.1f}")

    if fixed:
        print(f"\n  Pipeline FIXED ({len(fixed)} problems solo got wrong):")
        for tid in sorted(fixed):
            r = next(r for r in pipeline_results if r["task_id"] == tid)
            print(f"    {tid} (solved in {r['iterations_used']} iterations)")

    if regressed:
        print(f"\n  Pipeline REGRESSED ({len(regressed)} problems solo got right):")
        for tid in sorted(regressed):
            r = next(r for r in pipeline_results if r["task_id"] == tid)
            print(f"    {tid} (failed after {r['iterations_used']} iterations)")

    # Error type distribution for failures
    solo_errors = [r["error_type"] for r in solo_results if not r["passed"] and r.get("error_type")]
    pipe_final_errors = []
    for r in pipeline_results:
        if not r["passed"] and r.get("iteration_history"):
            last = r["iteration_history"][-1]
            if last.get("error_type"):
                pipe_final_errors.append(last["error_type"])

    if solo_errors:
        print(f"\n  Solo error types: {dict(Counter(solo_errors).most_common(5))}")
    if pipe_final_errors:
        print(f"  Pipeline final error types: {dict(Counter(pipe_final_errors).most_common(5))}")

    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Code generation benchmark")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Solo baseline model")
    parser.add_argument("--limit", type=int, default=None, help="Test on first N problems")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pipeline", action="store_true",
                       help="Run pipeline mode (solo + pipeline). Without this flag, only solo is run.")
    args = parser.parse_args()

    print(f"\n  Loading HumanEval problems...")
    problems = load_humaneval()
    if args.limit:
        problems = problems[:args.limit]
    print(f"  Loaded {len(problems)} problems")

    client = InferenceClient(backend=args.backend)

    # Build default genome for pipeline mode
    genome = make_default_genome(args.model)

    # Topology string for display
    stage = genome.stages[0] if genome.stages else None
    topology = f"{genome.generator.model}"
    if stage:
        topology += f" -> exec -> {stage.refiner.model}"

    print(f"  Solo model: {args.model}")
    if args.pipeline:
        print(f"  Pipeline: {topology}")
    print(f"  Backend: {args.backend}")
    print()

    solo_results = []
    pipeline_results = []

    try:
        for i, problem in enumerate(problems):
            print(f"  [{i+1:>3}/{len(problems)}] {problem['task_id']:<20}", end="", flush=True)

            # Solo
            try:
                sr = await run_solo(client, args.model, problem, args.max_tokens)
            except Exception as e:
                sr = {
                    "task_id": problem["task_id"], "passed": False,
                    "code": "", "error_type": str(e)[:100],
                    "stderr_snippet": "", "latency_ms": 0,
                }
            solo_results.append(sr)

            solo_mark = "PASS" if sr["passed"] else "FAIL"

            if args.pipeline:
                # Pipeline
                try:
                    pr = await run_pipeline(client, genome, problem, args.max_tokens)
                except Exception as e:
                    pr = {
                        "task_id": problem["task_id"], "passed": False,
                        "code": "", "iterations_used": 0,
                        "iteration_history": [], "latency_ms": 0,
                    }
                pipeline_results.append(pr)

                pipe_mark = "PASS" if pr["passed"] else "FAIL"
                iters = pr.get("iterations_used", 0)
                print(f"  solo={solo_mark}  pipe={pipe_mark} ({iters} iters)")
            else:
                print(f"  solo={solo_mark}")

    finally:
        await client.close()

    if args.pipeline and pipeline_results:
        print_results(solo_results, pipeline_results, args.model, genome)
    else:
        # Solo-only summary
        n = len(solo_results)
        passed = sum(1 for r in solo_results if r["passed"])
        print(f"\n  {'='*40}")
        print(f"  SOLO RESULTS: {args.model}")
        print(f"  {'='*40}")
        print(f"  pass@1: {passed}/{n} ({passed/n*100:.1f}%)")
        avg_lat = sum(r["latency_ms"] for r in solo_results) / n if n else 0
        print(f"  avg latency: {avg_lat:.0f}ms")
        solo_errors = [r["error_type"] for r in solo_results if not r["passed"] and r.get("error_type")]
        if solo_errors:
            print(f"  error types: {dict(Counter(solo_errors).most_common(5))}")
        print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"code_benchmark_{timestamp}.json"

    save_data = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "solo_model": args.model,
        "total_problems": len(problems),
        "solo_results": solo_results,
    }
    if args.pipeline and pipeline_results:
        save_data["pipeline_topology"] = topology
        save_data["pipeline_results"] = pipeline_results

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
