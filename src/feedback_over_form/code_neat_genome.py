"""NEAT genome for code generation pipelines with execution feedback.

Topology: generator -> [stage1: exec->(analyzer?)->refiner] -> [stage2: ...] -> ...

NEAT evolves the number of refinement stages, whether analyzers exist,
model assignment, prompts, temperatures, and executor iterations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .prompts import ANALYZER_PROMPTS, GENERATOR_PROMPTS, REFINER_PROMPTS


# --- Node configs ---

@dataclass
class LLMNodeConfig:
    """Configuration for an LLM node in the pipeline."""

    model: str
    system_prompt: str
    temperature: float

    def copy(self) -> LLMNodeConfig:
        return LLMNodeConfig(
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )


@dataclass
class RefinementStage:
    """One refinement loop: executor -> (optional analyzer) -> refiner.

    The executor runs code, and if it fails, sends to the analyzer (if present)
    then to the refiner, looping up to max_iterations times.
    """

    innovation: int
    max_iterations: int  # 1-4
    analyzer: LLMNodeConfig | None
    refiner: LLMNodeConfig

    def copy(self) -> RefinementStage:
        return RefinementStage(
            innovation=self.innovation,
            max_iterations=self.max_iterations,
            analyzer=self.analyzer.copy() if self.analyzer else None,
            refiner=self.refiner.copy(),
        )


@dataclass
class CodeNeatGenome:
    """NEAT genome for code pipelines."""

    generator: LLMNodeConfig
    stages: list[RefinementStage]  # at least 1
    fitness: float = 0.0
    raw_pass_count: int = 0
    total_iterations: float = 0.0
    failed_runs: int = 0
    species_id: int = 0

    @property
    def topology_str(self) -> str:
        gen_m = self.generator.model.split(":")[0]
        parts = [f"gen({gen_m})"]
        for s in self.stages:
            parts.append(f"exec(iters={s.max_iterations})")
            if s.analyzer:
                parts.append(f"analyze({s.analyzer.model.split(':')[0]})")
            parts.append(f"refine({s.refiner.model.split(':')[0]})")
        return " -> ".join(parts)

    @property
    def node_count(self) -> int:
        count = 1  # generator
        for s in self.stages:
            count += 1  # executor
            if s.analyzer:
                count += 1
            count += 1  # refiner
        return count

    def copy(self) -> CodeNeatGenome:
        return CodeNeatGenome(
            generator=self.generator.copy(),
            stages=[s.copy() for s in self.stages],
            fitness=self.fitness,
            raw_pass_count=self.raw_pass_count,
            total_iterations=self.total_iterations,
            failed_runs=self.failed_runs,
            species_id=self.species_id,
        )

    def all_models(self) -> list[str]:
        models = [self.generator.model]
        for s in self.stages:
            if s.analyzer:
                models.append(s.analyzer.model)
            models.append(s.refiner.model)
        return models

    def serialize(self) -> dict:
        return {
            "fitness": self.fitness,
            "total_iterations": self.total_iterations,
            "topology": self.topology_str,
            "generator": _serialize_node(self.generator),
            "stages": [
                {
                    "innovation": s.innovation,
                    "max_iterations": s.max_iterations,
                    "analyzer": _serialize_node(s.analyzer) if s.analyzer else None,
                    "refiner": _serialize_node(s.refiner),
                }
                for s in self.stages
            ],
        }


def _serialize_node(node: LLMNodeConfig) -> dict:
    return {
        "model": node.model,
        "system_prompt": node.system_prompt,
        "temperature": node.temperature,
    }


# --- Global innovation counter ---

_next_innovation = 3  # 0=generator, 1=first_executor, 2=first_refiner


def next_innovation() -> int:
    global _next_innovation
    val = _next_innovation
    _next_innovation += 1
    return val


def set_innovation_counter(val: int) -> None:
    global _next_innovation
    _next_innovation = val


def get_innovation_counter() -> int:
    return _next_innovation


# --- Creation ---

def create_code_neat_genome(model_pool: list[str]) -> CodeNeatGenome:
    """Create a minimal genome: generator -> exec -> refiner."""
    return CodeNeatGenome(
        generator=LLMNodeConfig(
            model=random.choice(model_pool),
            system_prompt=random.choice(GENERATOR_PROMPTS),
            temperature=round(random.uniform(0.2, 0.6), 2),
        ),
        stages=[
            RefinementStage(
                innovation=1,
                max_iterations=random.choice([1, 2, 2]),
                analyzer=None,
                refiner=LLMNodeConfig(
                    model=random.choice(model_pool),
                    system_prompt=random.choice(REFINER_PROMPTS),
                    temperature=round(random.uniform(0.1, 0.4), 2),
                ),
            ),
        ],
    )


# --- Mutations ---

MAX_NODES = 8
MAX_STAGES = 3


def mutate(
    genome: CodeNeatGenome,
    model_pool: list[str],
    p_add_stage: float = 0.04,
    p_add_analyzer: float = 0.05,
    p_remove_node: float = 0.18,
    p_model: float = 0.25,
    p_prompt: float = 0.30,
    p_temperature: float = 0.20,
    p_iterations: float = 0.10,
) -> None:
    """Mutate a code NEAT genome in-place."""

    # --- Structural mutations ---

    # Add refinement stage
    if (random.random() < p_add_stage
            and len(genome.stages) < MAX_STAGES
            and genome.node_count < MAX_NODES - 1):
        new_stage = RefinementStage(
            innovation=next_innovation(),
            max_iterations=random.choice([1, 2]),
            analyzer=None,
            refiner=LLMNodeConfig(
                model=random.choice(model_pool),
                system_prompt=random.choice(REFINER_PROMPTS),
                temperature=round(random.uniform(0.1, 0.4), 2),
            ),
        )
        genome.stages.append(new_stage)

    # Add analyzer to a stage that doesn't have one
    if random.random() < p_add_analyzer and genome.node_count < MAX_NODES:
        stages_without = [s for s in genome.stages if s.analyzer is None]
        if stages_without:
            stage = random.choice(stages_without)
            stage.analyzer = LLMNodeConfig(
                model=random.choice(model_pool),
                system_prompt=random.choice(ANALYZER_PROMPTS),
                temperature=round(random.uniform(0.2, 0.5), 2),
            )

    # Remove node (analyzer or entire extra stage)
    if random.random() < p_remove_node:
        # Prefer removing analyzers first
        stages_with_analyzer = [s for s in genome.stages if s.analyzer is not None]
        if stages_with_analyzer and random.random() < 0.6:
            stage = random.choice(stages_with_analyzer)
            stage.analyzer = None
        elif len(genome.stages) > 1:
            # Remove last stage (keep at least 1)
            genome.stages.pop()

    # --- Config mutations ---

    # Pick one random LLM node
    node, prompts = _pick_random_node(genome)

    if random.random() < p_model:
        node.model = random.choice(model_pool)

    if random.random() < p_prompt:
        node.system_prompt = random.choice(prompts)

    if random.random() < p_temperature:
        node.temperature = round(
            max(0.05, min(1.2, node.temperature + random.gauss(0, 0.08))), 2,
        )

    # Mutate iterations on a random stage (biased toward decrease)
    if random.random() < p_iterations and genome.stages:
        stage = random.choice(genome.stages)
        delta = -1 if random.random() < 0.6 else 1
        stage.max_iterations = max(1, min(3, stage.max_iterations + delta))


def _pick_random_node(genome: CodeNeatGenome) -> tuple[LLMNodeConfig, list[str]]:
    """Pick a random LLM node and its prompt pool."""
    candidates: list[tuple[LLMNodeConfig, list[str]]] = [
        (genome.generator, GENERATOR_PROMPTS),
    ]
    for s in genome.stages:
        if s.analyzer:
            candidates.append((s.analyzer, ANALYZER_PROMPTS))
        candidates.append((s.refiner, REFINER_PROMPTS))
    return random.choice(candidates)


# --- Crossover ---

def crossover(parent_a: CodeNeatGenome, parent_b: CodeNeatGenome) -> CodeNeatGenome:
    """Crossover two genomes. parent_a should have higher fitness.

    Aligns refinement stages by innovation number.
    """
    child = parent_a.copy()

    # Generator: randomly from either parent
    if random.random() < 0.5:
        child.generator = parent_b.generator.copy()

    # Align stages by innovation number
    b_stages = {s.innovation: s for s in parent_b.stages}
    new_stages = []

    for stage in parent_a.stages:
        if stage.innovation in b_stages and random.random() < 0.5:
            new_stages.append(b_stages[stage.innovation].copy())
        else:
            new_stages.append(stage.copy())

    # Extra stages from parent_b (not in parent_a) -- include with 30% chance
    a_innovations = {s.innovation for s in parent_a.stages}
    for innov, stage in b_stages.items():
        if innov not in a_innovations and random.random() < 0.3:
            new_stages.append(stage.copy())

    # Sort by innovation to maintain order
    new_stages.sort(key=lambda s: s.innovation)

    # Enforce max stages
    child.stages = new_stages[:MAX_STAGES] if new_stages else parent_a.stages[:1]

    child.fitness = 0.0
    child.total_iterations = 0.0
    return child


# --- Compatibility distance ---

def compatibility_distance(a: CodeNeatGenome, b: CodeNeatGenome) -> float:
    """Distance for speciation."""
    dist = 0.0

    # Topology: stage count difference
    dist += abs(len(a.stages) - len(b.stages)) * 1.0

    # Analyzer presence difference
    a_analyzers = sum(1 for s in a.stages if s.analyzer is not None)
    b_analyzers = sum(1 for s in b.stages if s.analyzer is not None)
    dist += abs(a_analyzers - b_analyzers) * 0.5

    # Model differences
    if a.generator.model != b.generator.model:
        dist += 0.4
    min_stages = min(len(a.stages), len(b.stages))
    for i in range(min_stages):
        if a.stages[i].refiner.model != b.stages[i].refiner.model:
            dist += 0.4

    # Prompt differences
    if a.generator.system_prompt != b.generator.system_prompt:
        dist += 0.2
    for i in range(min_stages):
        if a.stages[i].refiner.system_prompt != b.stages[i].refiner.system_prompt:
            dist += 0.2

    # Config differences (temperature, iterations)
    dist += abs(a.generator.temperature - b.generator.temperature) * 0.1
    for i in range(min_stages):
        dist += abs(a.stages[i].refiner.temperature - b.stages[i].refiner.temperature) * 0.1
        dist += abs(a.stages[i].max_iterations - b.stages[i].max_iterations) * 0.1

    return dist
