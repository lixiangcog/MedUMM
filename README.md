# MedUMM

MedUMM is an open platform for inference, evaluation, reporting, and
post-training of medical unified multimodal models. Its stable interfaces let
models, datasets, benchmarks, and training methods evolve independently.

> Research use only. MedUMM is not a medical device and must not be used for
> diagnosis, treatment, or other clinical decisions.

## v0.9: architecture-diverse runtime slices

Version 0.9 turns the first items in the v0.8 validation queue into reproducible
vertical slices:

- a native `lingshu_7b` Qwen2.5-VL adapter with multi-image input, immutable
  revision enforcement, official chat templating, and CUDA/Slurm evidence;
- source-pinned SLAKE and PathVQA exporters into the common medical VQA schema;
- a real PubMedCLIP contrastive path with per-sample candidate ranking;
- a fixed MedMNIST v2 PneumoniaMNIST exporter and zero-shot classification
  evaluation through the same benchmark/report contract;
- CPU asset-preparation and offline A800 acceptance jobs for both model families.

Both open slices passed a pinned A800 Slurm acceptance run and are marked
`runtime_validated`. Gated MedSigLIP remains interface-validated until its
upstream terms are accepted and weights are available. See
[docs/runtime-slices-v0.9.md](docs/runtime-slices-v0.9.md) and the committed
[machine-readable evidence](docs/results/v0.9-runtime-slices.json).

## v0.8: scale catalog for medical models and evaluation data

Version 0.8 registers 32 medical multimodal model releases and 34 evaluation
datasets behind the stable platform interfaces. Every resource has an audited
primary source, paper/code links, license and access level, revision policy,
medical domains, tasks/modalities, an executor or normalized-dataset family,
and an explicit validation status.

The catalog distinguishes `interface_validated` from `runtime_validated`.
Only real pinned server runs receive the latter; a model name or downloadable
weight is not counted as execution evidence. Gated and credentialed resources
require explicit acceptance/access flags, and remote runs require immutable
revisions. See [docs/resource-catalog-v0.8.md](docs/resource-catalog-v0.8.md).
The v0.8 A800 acceptance evidence is stored in
[docs/results/v0.8-scale-catalog.json](docs/results/v0.8-scale-catalog.json).

## v0.7: advanced post-training and research methods

Version 0.7 adds a real parameter-efficient alignment layer rather than another
classification-style training alias:

- one `medical_alignment` trainer for causal-LM SFT, DPO, SimPO, ORPO, and
  clinical-relevance-weighted DPO;
- LoRA and optional 4-bit QLoRA loading with a self-describing PEFT checkpoint;
- provenance-aware supervised/preference records with rationale, annotation
  source, safety category, specialty, task, and clinical-relevance fields;
- deterministic weighted multi-dataset mixtures with source-namespaced sample
  identifiers and content-stable fingerprints;
- data gates for license, de-identification, preference provenance, rationale,
  non-expert disclosure, and invalid relevance weights;
- token-level completion masking, frozen-reference DPO without a second model
  copy, gradient/history evidence, and independent adapter reload verification.

The real acceptance recipe performs LoRA-DPO on a pinned eight-pair
UltraMedical-Preference slice. It uses a small Apache-2.0 Pythia-14M research
model so the training system is cheap to reproduce; it is not a medical model
quality claim. See [docs/advanced-post-training-v0.7.md](docs/advanced-post-training-v0.7.md).
The verified A800 evidence is stored in
[docs/results/v0.7-advanced-post-training.json](docs/results/v0.7-advanced-post-training.json).

## v0.6: task-aware medicine, not natural-image classification

Version 0.6 adds a medical semantic layer across perception, reasoning, and
long-form generation:

- eight stable intents for finding assessment, clinical description, anatomy
  localization, quantitative assessment, imaging context, diagnostic reasoning,
  report generation, and patient communication;
- a `medical_tasks_jsonl` contract with task, concept, evidence, case/turn, and
  reference-provenance fields;
- a `medical_tasks` benchmark and `medical_task_core` suite with task-specific
  success, concept/evidence coverage, negation-aware extra concepts, strict
  diagnosis, and uncertainty intervals;
- audit gates that distinguish expert/native task labels from transparent
  heuristic mappings;
- a balanced 24-sample VQA-RAD + LLaVA-Med A800 acceptance recipe covering six
  tasks supported by real source questions, without fabricating report or
  patient-communication labels.

The task hierarchy is informed by the open diagnosis, clinical explanation,
and interaction goals described in
[VisionUnite](https://arxiv.org/pdf/2408.02865). MedUMM does not redistribute its
weights or treat its pretrained model license as part of this implementation.
See [docs/medical-tasks-v0.6.md](docs/medical-tasks-v0.6.md) for the schema,
metrics, provenance rules, and server recipe.
The verified A800 evidence is stored in
[docs/results/v0.6-medical-tasks.json](docs/results/v0.6-medical-tasks.json).

## v0.4: medical evaluation base

Version 0.4 turns the first real-model slice into a reusable medical evaluation
foundation:

- a versioned evaluation protocol and independently registered metric suite;
- preflight dataset quality/governance audits with content and provenance hashes;
- closed/open VQA metrics, configurable subgroup reports, abstention, and seeded
  bootstrap confidence intervals;
- per-batch atomic prediction checkpoints, fingerprint-safe resume, automatic
  Slurm/torchrun sharding, and strict deterministic shard merge;
- protocol-aware score files, CSV exports, leaderboards, and real A800 evidence.

See [docs/medical-evaluation-v0.4.md](docs/medical-evaluation-v0.4.md) for the
contract, CLI examples, artifacts, and acceptance criteria.

## v0.3: first real medical model

Version 0.3 connects the stable platform to a real biomedical vision-language
model and public medical benchmark:

- `llava_med`: Microsoft LLaVA-Med v1.5 Mistral-7B through its official model
  implementation;
- `flaviagiammarino/vqa-rad`: a revision-pinned VQA-RAD test slice normalized
  into the MedUMM dataset contract;
- real CUDA inference and `generate → score → report` evaluation on one A800;
- run evidence containing exact revisions, device/dtype, latency, peak allocated
  GPU memory, predictions, and grouped medical-VQA metrics.

This first vertical slice supports medical image understanding only. Generation,
editing, broader datasets, and post-training remain platform capabilities or
future real-model slices; v0.3 does not claim clinical model quality.

See [docs/real-model-v0.3.md](docs/real-model-v0.3.md) for the reproducible
server recipe, pinned revisions, and license boundaries.

## Stable platform interfaces

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

LLaVA-Med uses the upstream implementation's older Transformers compatibility
window. Keep it in a separate environment; the supplied setup script creates
one without modifying the parent Conda environment:

```bash
bash scripts/setup_llava_med_env.sh
```

## Unified CLI

```bash
# Inspect plugins without loading model weights
medumm catalog

# Inspect and validate the medical resource catalog
medumm resources list --kind model
medumm resources list --kind dataset
medumm resources validate
medumm resources template vqa_rad --kind dataset

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

Evaluation supports four explicit states:

- `audit`: validate dataset quality and governance without loading a model;
- `generate`: write fingerprinted `predictions.jsonl` only;
- `score`: score an existing compatible prediction file without loading a
  model;
- `full`: generate missing predictions, then score them.

Scoring writes `results.jsonl`, `score.json`, and `metrics.csv`. CLI runs also
write `run_manifest.json` containing the resolved config, component identity,
run ID, software environment, Git commit, and result. The `cross_task`
benchmark composes any registered benchmarks behind the same result contract.

Distributed workers automatically emit rank-local artifacts. Merge them only
after all ranks finish:

```bash
medumm merge-predictions \
  --shards outputs/evaluation/run/predictions.rank-*.jsonl \
  --output outputs/evaluation/run/predictions.jsonl \
  --expected-count 500
```

## Built-in plugins

| Kind | Name | Purpose |
|---|---|---|
| Model | `medical_reference` | Deterministic understanding/generation/editing reference |
| Model | `medical_linear` | Reloadable trainable VQA smoke baseline |
| Model | `medgemma` | Optional medical image-text understanding adapter |
| Model | `llava_med` | Real LLaVA-Med v1.5 biomedical understanding adapter |
| Model | `lingshu_7b` | Native Lingshu medical Qwen2.5-VL understanding adapter |
| Dataset | `medical_vqa_jsonl` | Normalized local JSON/JSONL medical VQA data |
| Dataset | `medical_tasks_jsonl` | Task-aware perception, reasoning, report, and communication data |
| Benchmark | `medical_vqa` | Generate/score medical VQA with grouped metrics |
| Benchmark | `medical_tasks` | Task-specific medical generation and scoring benchmark |
| Benchmark | `cross_task` | Compose registered benchmark runs |
| Trainer | `medical_sft` | Dependency-light supervised training smoke path |
| Trainer | `medical_alignment` | LoRA/QLoRA SFT and offline medical preference optimization |

The v0.8 resource catalog adds 32 individually registered model resources and
34 individually registered dataset resources. Run `medumm resources list` for
the current machine-readable inventory instead of maintaining a duplicate
table here.

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

The real-model workflow separates network-bound asset preparation from GPU
execution:

```bash
sbatch scripts/slurm_prepare_llava_med_assets.sh
sbatch scripts/slurm_llava_med_vqa_rad.sh
sbatch scripts/slurm_medical_tasks_v0.6.sh
sbatch scripts/slurm_prepare_runtime_slices_v0.9.sh
sbatch scripts/slurm_runtime_slices_v0.9.sh
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
