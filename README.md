# MedUMM

MedUMM is an open platform for inference, evaluation, reporting, and
post-training of medical unified multimodal models. Its stable interfaces let
models, datasets, benchmarks, and training methods evolve independently.

> Research use only. MedUMM is not a medical device and must not be used for
> diagnosis, treatment, or other clinical decisions.

## v0.2: stable platform interfaces

Version 0.2 establishes one four-layer platform architecture:

1. **Application and API:** schema-versioned YAML, one CLI, typed Python API,
   run manifests, reports, and leaderboards.
2. **Task and execution:** separate understanding, generation, and editing
   pipelines plus benchmark-neutral cross-task evaluation.
3. **Core functionality:** lazy registries for model, dataset, benchmark, and
   post-training plugins; typed requests and results; declared model
   capabilities.
4. **Infrastructure:** abstract interfaces, runtime/distributed context,
   metrics and atomic I/O, and adapters over model backbones.

The architecture and extension contracts are specified in
[docs/architecture.md](docs/architecture.md). v0.2 validates the entire
platform with dependency-light synthetic data; it does not claim clinical
model quality.

## Install

MedUMM requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,baseline]"
```

Install the `medical` extra only for heavyweight Transformers backbones:

```bash
pip install -e ".[medical]"
```

## Unified CLI

```bash
# Inspect plugins without loading model weights
medumm catalog

# Post-train, infer, and evaluate
medumm post-train --config configs/post_training/medical_sft_smoke.yaml
medumm infer --config configs/inference/medical_reference_workflow.yaml
medumm evaluate --config configs/evaluation/medical_vqa_linear_smoke.yaml

# Aggregate one or more score reports
medumm report \
  --scores outputs/evaluation/medical_vqa_linear/score.json \
  --output-directory outputs/reports/smoke
```

All execution commands accept repeatable dotted overrides such as
`--set evaluation.data.max_samples=2`.

## Unified YAML

Every new config declares `schema_version: "1.0"` and contains exactly one
execution block:

```yaml
schema_version: "1.0"
runtime:
  seed: 42
  device: auto
evaluation:
  benchmark: medical_vqa
  data:
    adapter: medical_vqa_jsonl
    path: examples/medical/tiny_eval.jsonl
    image_root: examples/medical/images
  model:
    backbone: medical_reference
    parameters:
      fixed_answer: A
  mode: full       # generate, score, or full
  batch_size: 2
  resume: true
  output_directory: outputs/evaluation/example
```

Flat v0.1 evaluation configs remain readable. New files should use the unified
envelope above.

## Python API

Use the high-level functions for config-driven work:

```python
from medumm import evaluate, infer, post_train

results = infer({
    "schema_version": "1.0",
    "inference": {
        "backbone": "medical_reference",
        "requests": [{
            "task": "understanding",
            "prompt": "Describe the supplied image.",
            "images": ["examples/medical/images/synthetic_scan.pgm"],
        }],
    },
})
print(results[0].text)
```

Use `InferencePipeline` when managing a loaded model directly:

```python
from medumm import InferencePipeline, InferenceRequest

with InferencePipeline("medical_reference", {}) as pipeline:
    result = pipeline.run(InferenceRequest(
        request_id="case-001",
        task="generation",
        prompt="Synthetic medical phantom for software testing",
    ))
```

`InferenceResult`, `EvaluationResult`, and `TrainingResult` have stable
`to_dict()` representations with schema version `1.0`.

## Evaluation and reproducibility

Evaluation supports three explicit states:

- `generate`: write fingerprinted `predictions.jsonl` only;
- `score`: score an existing compatible prediction file without loading a
  model;
- `full`: generate missing predictions, then score them.

Scoring writes `results.jsonl`, `score.json`, and `metrics.csv`. CLI runs also
write `run_manifest.json` containing the resolved config, component identity,
run ID, software environment, Git commit, and result. The `cross_task`
benchmark composes any registered benchmarks behind the same result contract.

## Built-in v0.2 plugins

| Kind | Name | Purpose |
|---|---|---|
| Model | `medical_reference` | Deterministic understanding/generation/editing reference |
| Model | `medical_linear` | Reloadable trainable VQA smoke baseline |
| Model | `medgemma` | Optional medical image-text understanding adapter |
| Dataset | `medical_vqa_jsonl` | Normalized local JSON/JSONL medical VQA data |
| Benchmark | `medical_vqa` | Generate/score medical VQA with grouped metrics |
| Benchmark | `cross_task` | Compose registered benchmark runs |
| Trainer | `medical_sft` | Dependency-light supervised training smoke path |

The reference and linear plugins prove platform behavior; they are not medical
foundation models. Model and dataset expansion follows
[docs/roadmap.md](docs/roadmap.md).

## Run the smoke workflow

```bash
bash scripts/run_reference_workflow.sh
```

For Slurm:

```bash
sbatch scripts/slurm_reference_workflow.sh
```

Outputs are written below `outputs/` and ignored by Git.

## Repository layout

```text
src/medumm/
  api.py            high-level Python API
  cli/              unified command line
  core/             contracts, registries, config, runtime, I/O
  inference/        understanding, generation, editing execution
  backbones/        model adapters
  medical/          medical schemas, datasets, metrics
  evaluation/       benchmark and cross-task runners
  post_training/    registered training methods
  reporting/        reports and leaderboards
configs/            schema-versioned execution recipes
examples/medical/   synthetic, non-clinical smoke data
scripts/            local and Slurm workflows
tests/              contract and end-to-end tests
```

## License

Apache-2.0. Model weights and datasets retain their own licenses and terms.
