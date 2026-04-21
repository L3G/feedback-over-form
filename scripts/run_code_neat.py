#!/usr/bin/env python3
"""NEAT evolution for code generation pipelines on HumanEval."""

from __future__ import annotations

import argparse
import asyncio
import json
import random as _random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feedback_over_form.code_neat_genome import (
    CodeNeatGenome,
    LLMNodeConfig,
    RefinementStage,
    compatibility_distance,
    create_code_neat_genome,
    crossover,
    mutate,
    set_innovation_counter,
    get_innovation_counter,
)
from feedback_over_form.code_executor import execute_code
from feedback_over_form.inference_client import InferenceClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Local code extraction ────────────────────────────────────────────

def _extract_code(text: str, entry_point: str | None = None) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    # Try to extract from markdown code block first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If the response looks like raw code (starts with def/class/import), use as-is
    stripped = text.strip()
    if stripped.startswith(("def ", "class ", "import ", "from ")):
        return stripped

    # Try to find a function definition
    match = re.search(r"(def \w+.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return the whole thing
    return stripped


# ── Load HumanEval ────────────────────────────────────────────────────

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
                "No hard_problem_diagnostic_*.json found. Run diagnose_hard_problems.py first."
            )
        with open(diag_files[0]) as f:
            diag = json.load(f)
        hard_ids = set(diag["hard_problem_ids"])
        problems = [p for p in problems if p["task_id"] in hard_ids]

    return problems


def _extract_function_name(assert_line: str) -> str:
    """Extract the function name from an MBPP assert line.

    Example: 'assert first_repeated_char("abcabc") == "a"' -> 'first_repeated_char'
    """
    import re
    match = re.search(r"assert\s+(\w+)\s*\(", assert_line)
    return match.group(1) if match else "solution"


def load_mbpp() -> list[dict]:
    """Load MBPP sanitized — 427 human-reviewed problems across 4 splits.

    MBPP gives a natural language description + a list of assert statements.
    We build a HumanEval-compatible problem dict by:
    - Combining the description and test examples into a prompt
    - Wiring test_list + test_imports into a runnable test_code
    - Extracting the function name from the first assert for entry_point
    """
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


# ── Pipeline execution for CodeNeatGenome ─────────────────────────────

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


# ── Speciation ────────────────────────────────────────────────────────

@dataclass
class Species:
    species_id: int
    representative: CodeNeatGenome
    members: list[CodeNeatGenome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0

    def is_compatible(self, genome: CodeNeatGenome, threshold: float) -> bool:
        return compatibility_distance(self.representative, genome) < threshold

    def update_best(self) -> None:
        if not self.members:
            return
        current_best = max(m.fitness for m in self.members)
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation = 0
        else:
            self.stagnation += 1

    def adjusted_fitness_sum(self) -> float:
        if not self.members:
            return 0.0
        return sum(m.fitness for m in self.members) / len(self.members)


def speciate(
    population: list[CodeNeatGenome],
    existing: list[Species],
    threshold: float,
    next_id: int,
) -> tuple[list[Species], int]:
    for s in existing:
        s.members.clear()

    new_species: list[Species] = []

    for genome in population:
        placed = False
        for s in existing:
            if s.is_compatible(genome, threshold):
                s.members.append(genome)
                genome.species_id = s.species_id
                placed = True
                break
        if not placed:
            for s in new_species:
                if s.is_compatible(genome, threshold):
                    s.members.append(genome)
                    genome.species_id = s.species_id
                    placed = True
                    break
        if not placed:
            s = Species(species_id=next_id, representative=genome, members=[genome])
            genome.species_id = next_id
            new_species.append(s)
            next_id += 1

    all_species = [s for s in existing if s.members] + new_species
    for s in all_species:
        s.representative = _random.choice(s.members)
        s.update_best()

    return all_species, next_id


# ── Evolution ─────────────────────────────────────────────────────────

class CodeNeatEvolution:
    def __init__(
        self,
        population_size: int,
        model_pool: list[str],
        problems: list[dict],
        tasks_per_eval: int,
        client: InferenceClient,
        max_concurrent: int = 1,
        max_tokens: int = 512,
        compatibility_threshold: float = 2.0,
        eval_temp: float | None = None,
    ):
        self.population_size = population_size
        self.model_pool = model_pool
        self.problems = problems
        self.tasks_per_eval = tasks_per_eval
        self.client = client
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.compatibility_threshold = compatibility_threshold
        self.eval_temp = eval_temp

        # Live progress tracking (set via run_dir in run())
        self.run_dir: Path | None = None
        self._progress_lock = asyncio.Lock()
        self._gen_progress: dict | None = None

        self.generation = 0
        self.species: list[Species] = []
        self.next_species_id = 1
        self.best_fitness_ever = 0.0
        self.best_genome_ever: CodeNeatGenome | None = None
        self.log: list[dict] = []

        self.population = [
            create_code_neat_genome(model_pool) for _ in range(population_size)
        ]

        # Stratified rotation: rank problems by difficulty using prior
        # benchmark data so every generation sees a balanced mix.
        self._stratified_order = self._build_stratified_order()

    def _build_stratified_order(self) -> list[int]:
        """Rank problems by difficulty, then interleave into stratified groups.

        Uses the cross-model benchmark to compute difficulty (pass count
        across runs). Falls back to original order if unavailable.
        """
        difficulty: dict[str, int] = {}
        bench_files = sorted(
            RESULTS_DIR.glob("cross_model_benchmark_*.json"),
            reverse=True,
        )
        if bench_files:
            with open(bench_files[0]) as f:
                data = json.load(f)
            for r in data.get("solo_results", []):
                difficulty[r["task_id"]] = difficulty.get(r["task_id"], 0) + (1 if r["passed"] else 0)
            for cfg in data.get("configs", {}).values():
                for r in cfg.get("results", []):
                    difficulty[r["task_id"]] = difficulty.get(r["task_id"], 0) + (1 if r["passed"] else 0)

        # Sort problem indices: easiest (highest pass count) first.
        # Problems with no data get pass_count = -1 (treated as hardest).
        indexed = list(range(len(self.problems)))
        indexed.sort(
            key=lambda i: difficulty.get(self.problems[i]["task_id"], -1),
            reverse=True,
        )
        return indexed

    def _get_gen_problems(self) -> list[dict]:
        """Stratified subset: each generation sees a balanced difficulty mix.

        Conceptually: the difficulty-ranked problems form a matrix with
        tasks_per_eval rows and ~(N/tasks_per_eval) columns. Each column
        is one generation's problems, containing one problem from each
        difficulty rank.
        """
        ranked = self._stratified_order
        n_per_gen = min(self.tasks_per_eval, len(ranked))
        cols = max(1, len(ranked) // n_per_gen)
        col = (self.generation - 1) % cols

        subset: list[dict] = []
        for row in range(n_per_gen):
            idx = row * cols + col
            if idx < len(ranked):
                subset.append(self.problems[ranked[idx]])
        return subset

    async def evaluate_fitness(
        self, genome: CodeNeatGenome, gen_problems: list[dict],
    ) -> None:
        """Evaluate genome on the generation's fixed problem subset."""
        passed = 0
        real_iters = 0
        real_runs = 0
        failed_runs = 0

        for problem in gen_problems:
            try:
                ok, iters = await asyncio.wait_for(
                    execute_genome(genome, problem, self.client, self.max_tokens,
                                  eval_temp=self.eval_temp),
                    timeout=120.0,
                )
                if ok:
                    passed += 1
                real_iters += iters
                real_runs += 1
            except (asyncio.TimeoutError, Exception):
                failed_runs += 1

        raw_fitness = passed / len(gen_problems) if gen_problems else 0.0
        # Parsimony pressure: penalize complexity (0.02 per node)
        genome.fitness = max(0.0, raw_fitness - 0.02 * genome.node_count)
        genome.raw_pass_count = passed
        # Average iterations over successful runs only — avoids phantom
        # iterations from timeouts/errors polluting the metric
        genome.total_iterations = real_iters / real_runs if real_runs else 0.0
        genome.failed_runs = failed_runs

    async def _write_progress(self) -> None:
        """Atomically write current_gen.json so live viewers see fresh state.

        Uses a temp-file-plus-rename so readers never see half-written JSON.
        """
        if self.run_dir is None or self._gen_progress is None:
            return
        async with self._progress_lock:
            tmp = self.run_dir / "current_gen.tmp.json"
            final = self.run_dir / "current_gen.json"
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w") as f:
                    json.dump(self._gen_progress, f)
                tmp.replace(final)  # atomic on POSIX
            except OSError:
                pass  # filesystem hiccups shouldn't crash evolution

    async def evolve_generation(self) -> dict:
        self.generation += 1
        gen_start = time.perf_counter()
        gen_problems = self._get_gen_problems()

        # Initialize live progress tracking for this gen
        self._gen_progress = {
            "generation": self.generation,
            "phase": "evaluating",
            "population_size": len(self.population),
            "tasks_per_genome": len(gen_problems),
            "started_at": time.time(),
            "genomes": [
                {
                    "idx": i,
                    "topology": g.topology_str,
                    "models": g.all_models(),
                    "node_count": g.node_count,
                    "status": "waiting",
                    "started_at": None,
                    "completed_at": None,
                    "raw_pass": None,
                    "fitness": None,
                    "total_iterations": None,
                }
                for i, g in enumerate(self.population)
            ],
        }
        await self._write_progress()

        # 1. Evaluate
        eval_failures = 0
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def eval_one(g: CodeNeatGenome, idx: int) -> bool:
            async with semaphore:
                # Mark evaluating
                if self._gen_progress is not None:
                    self._gen_progress["genomes"][idx]["status"] = "evaluating"
                    self._gen_progress["genomes"][idx]["started_at"] = time.time()
                await self._write_progress()
                try:
                    await self.evaluate_fitness(g, gen_problems)
                    if self._gen_progress is not None:
                        gp = self._gen_progress["genomes"][idx]
                        gp["status"] = "complete"
                        gp["completed_at"] = time.time()
                        gp["raw_pass"] = g.raw_pass_count
                        gp["fitness"] = g.fitness
                        gp["total_iterations"] = g.total_iterations
                    await self._write_progress()
                    return True
                except Exception as e:
                    print(f"    EVAL FAIL [{idx}]: {e}")
                    g.fitness = 0.0
                    g.total_iterations = 99.0
                    if self._gen_progress is not None:
                        gp = self._gen_progress["genomes"][idx]
                        gp["status"] = "failed"
                        gp["completed_at"] = time.time()
                        gp["fitness"] = 0.0
                    await self._write_progress()
                    return False

        results = await asyncio.gather(
            *[eval_one(g, i) for i, g in enumerate(self.population)]
        )
        eval_failures = sum(1 for ok in results if not ok)

        # Transition to reproduction phase
        if self._gen_progress is not None:
            self._gen_progress["phase"] = "reproducing"
        await self._write_progress()

        # 2. Speciate
        self.species, self.next_species_id = speciate(
            self.population, self.species,
            self.compatibility_threshold, self.next_species_id,
        )

        if len(self.species) <= 1 and self.generation >= 3:
            self.compatibility_threshold *= 0.6
            self.species, self.next_species_id = speciate(
                self.population, [], self.compatibility_threshold, self.next_species_id,
            )

        # 3. Remove stagnant species
        self.species = [
            s for s in self.species if s.stagnation < 8 or len(self.species) <= 2
        ]

        # 4. Offspring allocation
        total_adj = sum(s.adjusted_fitness_sum() for s in self.species)
        offspring_counts: list[int] = []
        for s in self.species:
            share = s.adjusted_fitness_sum() / total_adj if total_adj > 0 else 1.0 / len(self.species)
            offspring_counts.append(max(1, int(share * self.population_size)))

        while sum(offspring_counts) > self.population_size:
            idx = offspring_counts.index(max(offspring_counts))
            offspring_counts[idx] -= 1
        while sum(offspring_counts) < self.population_size:
            idx = _random.randrange(len(offspring_counts))
            offspring_counts[idx] += 1

        # 5. Reproduce with top-2 global elitism
        new_population: list[CodeNeatGenome] = []
        elite_count = min(2, len(self.population))
        # Sort by fitness (primary), then by fewer iterations (secondary)
        elites = sorted(
            self.population,
            key=lambda g: (g.fitness, -g.total_iterations),
            reverse=True,
        )[:elite_count]
        for e in elites:
            new_population.append(e.copy())

        remaining = self.population_size - elite_count
        total_offspring = sum(offspring_counts)
        if total_offspring > 0:
            offspring_counts = [
                max(1, int(c / total_offspring * remaining))
                for c in offspring_counts
            ]
        while sum(offspring_counts) > remaining:
            idx = offspring_counts.index(max(offspring_counts))
            offspring_counts[idx] -= 1
        while sum(offspring_counts) < remaining:
            idx = _random.randrange(len(offspring_counts))
            offspring_counts[idx] += 1

        for species_obj, count in zip(self.species, offspring_counts):
            members = sorted(
                species_obj.members,
                key=lambda g: (g.fitness, -g.total_iterations),
                reverse=True,
            )

            for _ in range(count):
                if len(members) >= 2 and _random.random() < 0.75:
                    p1 = _random.choice(members[:max(1, len(members) // 2)])
                    p2 = _random.choice(members)
                    if p2.fitness > p1.fitness or (
                        p2.fitness == p1.fitness and p2.total_iterations < p1.total_iterations
                    ):
                        p1, p2 = p2, p1
                    child = crossover(p1, p2)
                else:
                    child = _random.choice(members).copy()

                mutate(child, self.model_pool)
                new_population.append(child)

        self.population = new_population

        # Track champion
        best = max(
            self.population,
            key=lambda g: (g.fitness, -g.total_iterations),
        )
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        new_champion = False
        if (best.fitness > self.best_fitness_ever or
            (best.fitness == self.best_fitness_ever and
             self.best_genome_ever and
             best.total_iterations < self.best_genome_ever.total_iterations)):
            self.best_fitness_ever = best.fitness
            self.best_genome_ever = best.copy()
            new_champion = True

        # Stats
        gen_time = time.perf_counter() - gen_start
        n_problems = len(gen_problems)
        best_passed = best.raw_pass_count

        total_failed_runs = sum(g.failed_runs for g in self.population)

        gen_log = {
            "generation": self.generation,
            "best_fitness": best.fitness,
            "best_passed": f"{best_passed}/{n_problems}",
            "avg_fitness": avg_fitness,
            "species_count": len(self.species),
            "avg_stages": sum(len(g.stages) for g in self.population) / len(self.population),
            "avg_analyzers": sum(
                sum(1 for s in g.stages if s.analyzer) for g in self.population
            ) / len(self.population),
            "best_topology": best.topology_str,
            "best_avg_iters": round(best.total_iterations, 1),
            "total_failed_runs": total_failed_runs,
            "new_champion": new_champion,
            "eval_failures": eval_failures,
            "gen_time_s": round(gen_time, 1),
        }
        self.log.append(gen_log)

        # Mark phase complete so the dashboard can show the summary view
        if self._gen_progress is not None:
            self._gen_progress["phase"] = "complete"
            self._gen_progress["gen_log"] = gen_log
        await self._write_progress()

        return gen_log

    async def run(self, generations: int, run_dir: Path | None = None) -> CodeNeatGenome:
        """Run evolution, optionally saving incrementally after each generation.

        If run_dir is provided, every completed generation writes its JSON,
        plus a running evolution_summary.json and champion.json update.
        This makes partial progress visible to external viewers.
        """
        self.run_dir = run_dir  # Enable live progress writes
        # Write a config file so the dashboard knows the total target gens
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "run_config.json"
            if not config_path.exists():
                with open(config_path, "w") as f:
                    json.dump({
                        "total_generations": generations,
                        "population_size": self.population_size,
                        "tasks_per_eval": self.tasks_per_eval,
                        "model_pool": self.model_pool,
                        "eval_temp": self.eval_temp,
                        "started_at": time.time(),
                    }, f)
        for _ in range(generations):
            gl = await self.evolve_generation()

            print(
                f"  Gen {gl['generation']:>3} | "
                f"Best: {gl['best_fitness']:.2f} ({gl['best_passed']}) | "
                f"Mean: {gl['avg_fitness']:.2f} | "
                f"Species: {gl['species_count']} | "
                f"Stages: {gl['avg_stages']:.1f} | "
                f"Analyzers: {gl['avg_analyzers']:.1f} | "
                f"Iters: {gl['best_avg_iters']} | "
                f"Failed: {gl['total_failed_runs']} | "
                f"{gl['gen_time_s']}s"
            )

            if gl["new_champion"]:
                print(f"    NEW CHAMPION: {gl['best_topology']}")

            # Incremental save so viewers can watch progress in real time
            if run_dir is not None:
                try:
                    self.save_state(run_dir)
                except Exception as e:
                    print(f"    [warn] Could not save incremental state: {e}")

        return max(
            self.population,
            key=lambda g: (g.fitness, -g.total_iterations),
        )

    def save_state(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

        for gl in self.log:
            gen_path = run_dir / f"generation_{gl['generation']}.json"
            if not gen_path.exists():
                with open(gen_path, "w") as f:
                    json.dump(gl, f, indent=2)

        if self.best_genome_ever:
            with open(run_dir / "champion.json", "w") as f:
                json.dump({
                    "fitness": self.best_fitness_ever,
                    "genome": self.best_genome_ever.serialize(),
                }, f, indent=2)

        with open(run_dir / "evolution_summary.json", "w") as f:
            json.dump({
                "generations": len(self.log),
                "best_fitness_ever": self.best_fitness_ever,
                "innovation_counter": get_innovation_counter(),
                "log": self.log,
            }, f, indent=2)


# ── Validation: run champion on all 164 problems ─────────────────────

async def validate_champion(
    genome: CodeNeatGenome,
    problems: list[dict],
    client: InferenceClient,
    max_concurrent: int = 5,
    max_tokens: int = 512,
    eval_temp: float | None = None,
) -> tuple[int, int]:
    """Run genome on all problems. Returns (passed, total)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def eval_one(problem: dict) -> bool:
        async with semaphore:
            try:
                ok, _ = await asyncio.wait_for(
                    execute_genome(genome, problem, client, max_tokens,
                                  eval_temp=eval_temp),
                    timeout=120.0,
                )
                return ok
            except Exception:
                return False

    results = await asyncio.gather(*[eval_one(p) for p in problems])
    passed = sum(1 for r in results if r)
    return passed, len(problems)


# ── CLI ──────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="NEAT evolution for code pipelines")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--tasks-per-eval", type=int, default=25)
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "vllm"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated model pool (default: gemma3:1b,qwen2.5:1.5b,llama3.2:3b)")
    parser.add_argument("--benchmark", type=str, default="original",
                       choices=["original", "plus"],
                       help="HumanEval variant: 'original' (164 problems) or 'plus' (80x more tests)")
    parser.add_argument("--hard-only", action="store_true",
                       help="Filter problem pool to only the hard problems from the latest diagnostic (where coder:3b self-refine fails)")
    parser.add_argument("--eval-temperature", type=float, default=None,
                       help="Override all node temperatures to this value during eval (e.g., 0 for deterministic)")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else _random.randint(0, 2**32 - 1)
    _random.seed(seed)

    benchmark_name = "HumanEval+" if args.benchmark == "plus" else "HumanEval"
    if args.hard_only:
        benchmark_name += " (hard subset)"
    print(f"\n  Loading {benchmark_name} problems...")
    problems = load_humaneval(variant=args.benchmark, hard_only=args.hard_only)
    print(f"  Loaded {len(problems)} problems")

    if args.models:
        model_pool = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_pool = ["gemma3:1b", "qwen2.5:1.5b", "llama3.2:3b"]
    raw_name = args.run_name or f"code_neat_{seed}"
    run_name = re.sub(r'[^a-zA-Z0-9_-]', '', raw_name) or f"code_neat_{seed}"

    print(f"\n  === Code NEAT Evolution ===")
    print(f"  Run: {run_name}")
    print(f"  Seed: {seed}")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Models: {model_pool}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Tasks per eval: {args.tasks_per_eval}/{len(problems)}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Backend: {args.backend}")
    if args.eval_temperature is not None:
        print(f"  Eval temperature: {args.eval_temperature} (deterministic override)")
    print()

    client = InferenceClient(backend=args.backend)

    evolution = CodeNeatEvolution(
        population_size=args.population,
        model_pool=model_pool,
        problems=problems,
        tasks_per_eval=args.tasks_per_eval,
        client=client,
        max_concurrent=args.parallel,
        max_tokens=args.max_tokens,
        eval_temp=args.eval_temperature,
    )

    run_dir = RESULTS_DIR / run_name
    interrupted = False
    try:
        best = await evolution.run(generations=args.generations, run_dir=run_dir)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  Interrupted at generation {evolution.generation}. Saving...")
        best = max(
            evolution.population,
            key=lambda g: (g.fitness, -g.total_iterations),
        )
    finally:
        evolution.save_state(run_dir)

    status = f"INTERRUPTED at gen {evolution.generation}" if interrupted else "COMPLETE"
    print(f"\n  {'='*60}")
    print(f"  EVOLUTION {status} — {run_name}")
    print(f"  {'='*60}")
    print(f"  Seed: {seed}")
    print(f"  Best fitness: {evolution.best_fitness_ever:.2f}")
    if evolution.best_genome_ever:
        print(f"  Topology: {evolution.best_genome_ever.topology_str}")
        print(f"  Models: {evolution.best_genome_ever.all_models()}")

    # Validation: run champion on problem set (+ full set if hard-only)
    if not interrupted and not args.skip_validation and evolution.best_genome_ever:
        validation_data: dict = {
            "champion_topology": evolution.best_genome_ever.topology_str,
        }

        if args.hard_only:
            # Hard subset: how many of the "unfixable" problems did we fix?
            print(f"\n  Running champion on {len(problems)} hard problems...")
            hard_passed, hard_total = await validate_champion(
                evolution.best_genome_ever, problems, client,
                max_concurrent=args.parallel, max_tokens=args.max_tokens,
            )
            print(f"  Hard subset: {hard_passed}/{hard_total} ({hard_passed/hard_total*100:.1f}%)")
            print(f"    Baseline (coder self-refine on hard): 0/{hard_total} by definition")
            print(f"    Upper bound (union of 3 configs):      12/35 (34.3%)")
            validation_data["hard_subset"] = {
                "passed": hard_passed,
                "total": hard_total,
                "pass_rate": hard_passed / hard_total,
            }

            # Full 164: how does the champion do overall?
            print(f"\n  Running champion on all 164 HumanEval+ problems...")
            full_problems = load_humaneval(variant=args.benchmark, hard_only=False)
            full_passed, full_total = await validate_champion(
                evolution.best_genome_ever, full_problems, client,
                max_concurrent=args.parallel, max_tokens=args.max_tokens,
            )
            print(f"  Full HumanEval+: {full_passed}/{full_total} ({full_passed/full_total*100:.1f}%)")
            print(f"    coder:3b self-refine:          131/164 (79.9%)")
            print(f"    coder:3b solo:                 123/164 (75.0%)")
            validation_data["full_set"] = {
                "passed": full_passed,
                "total": full_total,
                "pass_rate": full_passed / full_total,
            }
        elif args.eval_temperature is not None:
            # Deterministic evaluation mode: validate at temp=0 and at evolved temps
            print(f"\n  Running champion on all {len(problems)} problems at temp=0 (deterministic)...")
            passed_t0, total = await validate_champion(
                evolution.best_genome_ever, problems, client,
                max_concurrent=args.parallel, max_tokens=args.max_tokens,
                eval_temp=args.eval_temperature,
            )
            print(f"  Champion (temp=0): {passed_t0}/{total} ({passed_t0/total*100:.1f}%)")

            # Also run 5 times at the genome's evolved temperatures
            print(f"\n  Running champion 5x at evolved temperatures...")
            evolved_runs = []
            for run_i in range(5):
                passed_ev, _ = await validate_champion(
                    evolution.best_genome_ever, problems, client,
                    max_concurrent=args.parallel, max_tokens=args.max_tokens,
                    eval_temp=None,
                )
                evolved_runs.append(passed_ev)
                print(f"    Run {run_i+1}/5: {passed_ev}/{total}")

            import statistics
            ev_mean = statistics.mean(evolved_runs)
            ev_std = statistics.stdev(evolved_runs)
            print(f"  Champion (evolved temps): {ev_mean:.1f} ± {ev_std:.1f}")

            print(f"\n  Reference baselines (5-run validated means on HumanEval):")
            print(f"    NEAT v3 champion:       98.2 ± 3.4")
            print(f"    llama→exec→llama:       94.0 ± 2.7")
            print(f"    llama solo:             76.6 ± 3.7")

            validation_data["temp0_pass"] = passed_t0
            validation_data["temp0_total"] = total
            validation_data["evolved_temp_runs"] = evolved_runs
            validation_data["evolved_temp_mean"] = ev_mean
            validation_data["evolved_temp_std"] = ev_std
        else:
            print(f"\n  Running champion on all {len(problems)} HumanEval problems...")
            passed, total = await validate_champion(
                evolution.best_genome_ever, problems, client,
                max_concurrent=args.parallel, max_tokens=args.max_tokens,
            )
            print(f"  Champion: {passed}/{total} ({passed/total*100:.1f}%)")
            if args.benchmark == "original":
                print(f"  Baselines (HumanEval original):")
                print(f"    qwen2.5-coder:3b self-refining: 136/164 (82.9%)")
                print(f"    qwen2.5-coder:3b solo:          135/164 (82.3%)")
                print(f"    NEAT v3 champion:               106/164 (64.6%)")
                print(f"    llama solo:                      82/164 (50.0%)")
            else:
                print(f"  Baselines (HumanEval+):")
                print(f"    qwen2.5-coder:3b self-refining: 131/164 (79.9%)")
                print(f"    qwen2.5-coder:3b solo:          123/164 (75.0%)")
            validation_data["champion_pass_at_1"] = f"{passed}/{total}"
            validation_data["champion_pass_rate"] = passed / total

        # Save validation result
        val_path = run_dir / "validation.json"
        with open(val_path, "w") as f:
            json.dump(validation_data, f, indent=2)

    await client.close()
    print(f"\n  Saved to: {run_dir}\n")


if __name__ == "__main__":
    asyncio.run(main())
