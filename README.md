# Feedback Over Form

**Why Execution Feedback Matters More Than Pipeline Topology in 1-3B Code Generation**

We use NEAT (NeuroEvolution of Augmenting Topologies) to evolve multi-model code generation pipelines on HumanEval, searching over topology (number of refinement stages, error analyzers), model assignment (gemma3:1b, qwen2.5:1.5b, llama3.2:3b), prompts, temperatures, and iteration counts. Across 8 independent evolution runs, every champion converged to the same flat generator-executor-refiner topology as hand-designed baselines. The execution feedback loop — not the pipeline structure — drives the 17-23 percentage point improvement over solo models.

## Key Findings

- **Execution feedback is the dominant factor.** Adding a generate-execute-refine loop improves pass@1 by 17-23 points across all model configurations, regardless of topology.
- **NEAT rediscovers manual baselines.** 7 of 8 evolved champions converged to the same flat topology as hand-designed pipelines; the one structural outlier (with an error analyzer node) underperformed.
- **Cross-model pipelines match same-model ones.** qwen2.5:1.5b generating + llama3.2:3b refining performs comparably to llama3.2:3b doing both (93.6 vs 94.0 mean pass@1).
- **Extra iterations have diminishing returns.** Iteration analysis shows >90% of fixes occur on the first refinement attempt; iterations 2-3 rarely contribute new fixes.
- **Error type determines fixability.** Self-refinement fixes 40-60% of assertion errors but <5% of deep logic errors, independent of pipeline complexity.
- **Evaluation noise is real.** 5-run validation shows standard deviations of 2.5-3.7 points; single-run comparisons can be misleading.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### Install

```bash
pip install -e .
```

### Pull models

```bash
ollama pull gemma3:1b && ollama pull qwen2.5:1.5b && ollama pull llama3.2:3b
```

### Run baseline benchmark

```bash
# Solo model (no refinement)
python scripts/run_code_benchmark.py --model llama3.2:3b --limit 10

# Pipeline with execution feedback
python scripts/run_code_benchmark.py --model llama3.2:3b --pipeline --limit 10
```

### Run NEAT evolution

```bash
python scripts/run_code_neat.py --population 10 --generations 5 --seed 42 --tasks-per-eval 20
```

## Reproducing Paper Results

### Full benchmark (solo vs pipeline on all 164 HumanEval problems)

```bash
python scripts/run_code_benchmark.py --model llama3.2:3b
```

### 5-run validation (measures mean and variance)

```bash
python scripts/run_revalidation.py
```

### Error taxonomy analysis

```bash
python scripts/run_error_analysis.py
```

## Pre-computed Results

The `results/` directory contains validated results from the paper:

- `main_results.json` — 5-run validation for all 6 configurations
- `neat_runs/v2_champion.json` — NEAT v2 champion genome (flat topology)
- `neat_runs/v3_champion.json` — NEAT v3 champion genome (with error analyzer)
- `neat_runs/temp0_champion.json` — Deterministic-eval champion genome
- `error_taxonomy.json` — Fix rates by error type
- `iteration_analysis.json` — Per-iteration marginal fix analysis

## Hardware

All experiments were run on a 128GB M4 Max MacBook Pro with Ollama. No cloud compute required.

## Citation

```bibtex
@article{feedbackoverform2026,
  title={Feedback Over Form: Why Execution Feedback Matters More Than
         Pipeline Topology in 1-3B Code Generation},
  author={TODO},
  year={2026},
  note={Code and data: https://github.com/L3G/feedback-over-form}
}
```

## License

MIT
