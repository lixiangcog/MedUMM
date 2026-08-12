# Medical evaluation base — v0.4

MedUMM v0.4 establishes a benchmark-neutral execution base and a first
versioned medical VQA protocol. It follows the same four-layer architecture as
the platform: YAML/CLI and reports at the application layer, evaluation state
machines at the task layer, model/dataset/benchmark plugins at the core layer,
and protocols, metric suites, sharding, auditing, and atomic I/O at the
infrastructure layer.

This release is for reproducible research evaluation. It does not validate a
model for diagnosis or clinical deployment.

## Stable protocol surface

The optional `evaluation.protocol` block resolves to a self-contained object:

```yaml
protocol:
  name: vqa_rad_closed
  version: "1.0"
  metric_suite: medical_vqa_core
  group_by: [modality, category, answer_type, language]
  bootstrap_samples: 2000
  confidence_level: 0.95
  seed: 42
  require_provenance: true
  require_deidentified: true
  minimum_samples: 32
```

The resolved protocol, metric-suite version, dataset fingerprint, model
configuration, and prompt template are hashed together. Cached predictions
from a different protocol cannot be scored silently.

`medical_vqa_core` 1.0 reports exact match, token F1, closed-answer accuracy,
abstention rate, configurable subgroup breakdowns, and deterministic percentile
bootstrap intervals. These are engineering benchmark metrics, not clinical
performance or calibration claims.

## Dataset audit

Audit-only mode does not load model weights:

```bash
medumm evaluate --config configs/evaluation/vqa_rad_audit.yaml
```

`dataset_audit.json` records the manifest SHA-256, dataset fingerprint, sample,
reference and image counts, missing-image checks, task-format distribution,
subgroup distributions, unknown metadata, de-identification declaration,
source, license, revision, provenance SHA-256, warnings, and hard errors.
Required governance gates fail before GPU allocation or model loading.

## Failure recovery and distributed execution

Predictions are written atomically after each configurable checkpoint interval.
With `resume: true`, only rows with the exact current run fingerprint are
reused. An interrupted run therefore resumes from its last complete checkpoint.

Under Slurm or torchrun, rank and world size are inferred from the environment.
Each worker evaluates a deterministic stride shard and writes paths such as:

```text
predictions.rank-00000-of-00002.jsonl
score.rank-00000-of-00002.json
metrics.rank-00000-of-00002.csv
```

After every rank finishes, merge predictions strictly:

```bash
medumm merge-predictions \
  --shards outputs/evaluation/run/predictions.rank-*.jsonl \
  --output outputs/evaluation/run/predictions.jsonl \
  --expected-count 500
```

The merge fails on a missing/duplicate rank, duplicate sample ID, mixed
fingerprint, or unexpected total. A successful merge writes
`merge_manifest.json`.

## Real acceptance run

The v0.4 acceptance recipe uses the same revision-pinned real assets as v0.3:

- model: `microsoft/llava-med-v1.5-mistral-7b` at
  `91bb16c122001ddc9cf1fd36ce1dae09448943a2`;
- dataset: `flaviagiammarino/vqa-rad` test split at
  `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`;
- selection: first 32 usable closed yes/no samples under deterministic source
  order;
- execution: one NVIDIA A800 through Slurm, CUDA float16, offline assets;
- acceptance: all unit tests pass; audit has no hard error; all predictions are
  non-empty; model device is CUDA; latency and peak memory exist; metric and
  protocol versions are pinned; uncertainty intervals are present.

Prepare or refresh the expanded dataset slice and submit evaluation:

```bash
sbatch scripts/slurm_prepare_llava_med_assets.sh
sbatch scripts/slurm_medical_evaluation_v0.4.sh
```

The committed evidence is written to
`docs/results/v0.4-medical-evaluation-base.json` after the server run. The
32-sample score is a pipeline acceptance result, not a paper-quality benchmark.

## License boundary

MedUMM code is Apache-2.0. LLaVA-Med source/weights and VQA-RAD retain their own
upstream terms. The asset provenance files and v0.3 model guide remain the
authority for the pinned source links and restrictions; no model weight or
medical image is redistributed in this repository.
