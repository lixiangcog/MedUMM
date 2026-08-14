# MedUMM v1.6 independent medical benchmarks

Version 1.6 fixes a counting and implementation gap: a dataset resource is not
an executable benchmark. MedUMM still catalogs 34 medical evaluation datasets,
but now exposes 13 specialized benchmark plugins, two generic plugins
(`medical_vqa` and `medical_tasks`), and one composite runner (`cross_task`).
Therefore the truthful platform count is 15 independent executable benchmark
adapters, not 34.

This is a software research release. The included synthetic matrix tests the
platform contracts and cannot establish medical model quality, safety,
fairness, or clinical validity.

## What makes one benchmark independent

Each specialized benchmark owns all of the following instead of only pointing
at a dataset name:

1. a versioned benchmark specification;
2. compatible normalized dataset families;
3. required annotations, choices, candidate scores, groups, or pairs;
4. a task-specific medical prompt template;
5. a fixed versioned metric suite that cannot be replaced in YAML;
6. data provenance, de-identification, media, and annotation audits;
7. inference-item construction and candidate propagation;
8. a dedicated scorer and aggregate report.

All 13 use the common execution state machine:

```text
normalized data → benchmark audit → model inference → dedicated scorer
                → per-sample results → aggregate JSON/CSV report
```

The stable runner, runtime context, registries, and model interfaces remain
shared infrastructure. Medical task semantics and scoring stay inside the
benchmark/core-functionality layer.

## Specialized matrix

| Benchmark | Dataset families | Required input | Dedicated aggregates |
|---|---|---|---|
| `pathology_vqa` | VQA | pathology references and answer type | yes/no, free-form, overall and macro answer-type accuracy |
| `medical_mcqa` | VQA, medical task | labelled choices | strict choice accuracy and invalid-response rate |
| `medical_image_classification` | classification | labelled choices; optional class scores | accuracy, balanced accuracy, macro F1/AUC and confusion matrix |
| `medical_multilabel_findings` | classification, report | `annotations.multilabel` | micro precision/recall/F1, sample macro F1 and exact match |
| `radiology_report_generation` | report | `annotations.report` | finding factuality, contradiction, critical recall and section completeness |
| `medical_grounding` | detection/measurement, report | `annotations.grounding` | normalized box IoU, IoU@0.5, point distance and pointing accuracy |
| `medical_measurement` | detection/measurement | `annotations.measurements` | unit-aware MAE/MRE, tolerance accuracy and unit errors |
| `medical_temporal_reasoning` | video | `annotations.temporal` | sequence exact match, edit similarity, phase accuracy and transition F1 |
| `medical_retrieval` | retrieval, report | candidates, positives and model scores | Recall@1/5/10 and mean reciprocal rank |
| `medical_calibration` | VQA, medical task, classification | choices and full candidate scores | ECE, Brier, NLL and selective coverage/accuracy |
| `medical_fairness` | classification, medical task | choices and `annotations.fairness` | group accuracy, worst group, demographic-parity and equal-opportunity gaps |
| `medical_safety` | medical task | `annotations.safety` | safe completion, expected refusal, over-refusal and unsafe compliance |
| `medical_robustness` | VQA, medical task, classification | complete `annotations.robustness` pairs | baseline/perturbed accuracy, accuracy drop and prediction consistency |

The metric-suite name and benchmark version are part of the run fingerprint.
For example, `medical_grounding` rejects a configuration that tries to replace
its metric suite with a generic VQA score.

## Dataset routing

`src/medumm/resources/catalog/datasets.yaml` assigns every resource a primary
benchmark family. This routing supports discovery and compatibility checks; it
does not claim that every resource has already been downloaded, normalized, or
run. Catalog statuses continue to distinguish `interface_validated` from
`runtime_validated`.

Examples of the routing policy:

- PathVQA routes to `pathology_vqa`;
- PMC-VQA, OmniMedVQA and MedXpertQA-MM route to `medical_mcqa`;
- CheXpert and ChestX-ray14 route to `medical_multilabel_findings`;
- MIMIC-CXR and IU X-Ray route to `radiology_report_generation`;
- GEMeX routes to `medical_grounding`;
- Surg-MLLM Bench routes to `medical_temporal_reasoning`;
- FairVLMed and Camelyon17 route to `medical_fairness`;
- CARES routes to `medical_safety`;
- MediConfusion and MedBLINK route to `medical_robustness`.

Generic VQA and task datasets may remain assigned to `medical_vqa` or
`medical_tasks` until a source-specific specialized protocol is appropriate.

## CLI and configuration

Inspect the matrix without loading heavyweight model libraries:

```bash
medumm benchmarks list
medumm benchmarks list --json
medumm benchmarks show medical_retrieval
medumm benchmarks audit
medumm benchmarks template medical_safety
```

`medumm benchmarks audit` checks registrations, metric versions, dataset
routing, and the dataset/benchmark counts. A healthy v1.6 tree reports:

```text
registered benchmark plugins: 16
independent benchmark plugins: 15
specialized medical benchmarks: 13
composite benchmarks: 1
dataset resources: 34
```

Run one complete software-contract example:

```bash
medumm evaluate \
  --config configs/evaluation/benchmarks_v1.6/medical_grounding.yaml
```

Every specialized configuration writes:

- `dataset_audit.json` — governance, media, annotation, choice, and pair/group checks;
- `predictions.jsonl` — resumable model outputs and optional candidate scores;
- `results.jsonl` — task-specific per-sample scoring fields;
- `score.json` — resolved benchmark/protocol/model identity and aggregate metrics;
- `metrics.csv` — flattened metric export.

The complete deterministic matrix is submitted with:

```bash
sbatch scripts/slurm_specialized_benchmarks_v1.6_cpu.sh
sbatch scripts/slurm_specialized_benchmarks_v1.6.sh
```

The CPU job matches the deterministic reference adapter's requirements. The
A800 variant additionally requires a CUDA-visible allocation and checks CUDA
visibility, but does not turn a reference-adapter run into real-model evidence.
Both run the benchmark-focused test suite, execute all 13 public CLI
configurations, and produce
`outputs/verification/specialized_benchmarks_v1.6.json`.

The accepted matrix ran as Slurm job `437789` on `node15` and completed with
exit code `0:0`. It verified 16 registered plugins, 15 independent benchmark
plugins, all 13 specialized contracts, 34 dataset resources, and 21 aligned
fixture predictions. The exact resolved protocols and aggregate outputs are in
[`docs/results/v1.6-specialized-benchmarks.json`](results/v1.6-specialized-benchmarks.json).

## Validation levels and remaining work

The repository fixture uses two small synthetic PGM images and the deterministic
`medical_reference` adapter. It verifies annotation parsing, audit failures,
candidate-score transport, model invocation, scorer behavior, and artifact
generation. It is deliberately labelled `Not a medical model quality estimate`.

The next validation queue is source-specific:

1. reuse the pinned PathVQA + Lingshu run for pathology VQA;
2. normalize a pinned public PMC-VQA/OmniMedVQA slice for MCQA;
3. reuse PneumoniaMNIST + PubMedCLIP for single-label classification and add a
   source-supported score calibration protocol;
4. add expert/source annotations for report, grounding, measurement, temporal,
   retrieval, fairness, safety, and perturbation robustness;
5. only promote a benchmark to real runtime evidence after its model weights,
   dataset revision, license/access state, predictions, scores, and Slurm
   provenance are committed together.

Small acceptance slices remain pipeline evidence, not statistically meaningful
leaderboard results. Full-dataset evaluation and clinical review are separate
release gates.
