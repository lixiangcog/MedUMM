# Architecture

MedUMM separates orchestration from model-specific code. Configuration files
select a registered component; the surrounding pipeline stays unchanged.

```text
CLI / Python API
       |
       v
configuration loader
       |
       v
inference pipeline ---- model adapter
       |
       +---- evaluation runner ---- metrics and reports
       |
       +---- post-training runner - checkpoint and manifest
```

## Model adapter contract

An adapter has a `load(config)` method and may implement one or more task
methods: `understanding`, `generation`, and `editing`. It declares supported
tasks explicitly. Unsupported tasks fail early with a clear error.

Heavy libraries are imported inside an adapter's `load` method. Listing or
using a lightweight adapter therefore does not require every model dependency.

## Evaluation contract

Evaluation is split into generation and scoring. Predictions are stored in
JSONL and can be resumed; reports are emitted as JSON plus a flat metrics CSV.
The initial benchmark plugin is medical VQA with exact-match, token F1, and
abstention metrics grouped by modality, category, answer type, and language.

## Post-training contract

A trainer receives a configuration, writes a self-describing model directory,
and returns a JSON-serializable run summary. The initial trainer is a small
softmax baseline used to validate data, checkpoint, inference, and evaluation
plumbing before expensive GPU training is introduced.

## Extension rule

New models, datasets, metrics, and trainers are accepted only with:

1. a registry entry and stable interface implementation;
2. a minimal configuration;
3. a dependency-safe smoke test;
4. documented licenses and intended research use.
