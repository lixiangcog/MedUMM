# Roadmap

MedUMM grows through validated medical vertical slices. Platform breadth and
medical scale are tracked separately so a long model list cannot hide an
incomplete evaluation or data-governance path.

## v0.1 — executable foundation

- [x] installable package and unified CLI
- [x] local reference model and trainable smoke baseline
- [x] medical VQA data, metrics, reports, and resumable predictions
- [x] train → infer → evaluate workflow on Slurm

## v0.2 — stable platform interfaces

- [x] four typed plugin contracts and lazy registries
- [x] model capability declaration and task/modality validation
- [x] understanding, generation, and editing execution pipelines
- [x] unified YAML schema with v0.1 evaluation compatibility
- [x] stable request/result objects and high-level Python API
- [x] generate/score/full evaluation state machine
- [x] cross-task benchmark composition
- [x] runtime manifests, dataset/model fingerprints, and leaderboards
- [x] Slurm-aware distributed context and deterministic sharding primitive

## v0.3 — first real medical vertical slices

- [x] real LLaVA-Med v1.5 Mistral-7B understanding adapter
- [x] revision-pinned VQA-RAD public benchmark slice and provenance
- [x] A800 Slurm inference/evaluation job with verifiable runtime evidence
- [ ] production MedGemma inference recipe and checkpoint/license guide
- [ ] a second real medical VLM adapter from a different architecture family
- [ ] pathology VQA benchmark adapter with public-data provenance
- [ ] medical text-to-image model and generation benchmark
- [ ] medical image editing model and editing benchmark
- [ ] GPU Slurm smoke jobs for every advertised capability

## v0.4 — medical evaluation base

- [x] versioned medical evaluation protocol and metric-suite contract
- [x] preflight dataset quality, provenance, license, and de-identification audit
- [x] exact match, token F1, closed accuracy, abstention, and subgroup metrics
- [x] deterministic bootstrap confidence intervals
- [x] batch-level atomic checkpoints and fingerprint-safe recovery
- [x] automatic Slurm/torchrun sharding and strict deterministic prediction merge
- [x] protocol-aware score, CSV, leaderboard, and reproducibility artifacts
- [x] expanded real LLaVA-Med + VQA-RAD A800 acceptance run

## v0.5 — post-training, distributed scale, and evaluation breadth

- [ ] Transformers SFT plus LoRA/QLoRA trainer plugin
- [ ] multi-dataset mixture schema with sampling weights and provenance
- [ ] initialized multi-node process groups and experiment-tracker hooks
- [ ] preference-training interface and first medical safety preference recipe
- [ ] localization and grounding metrics
- [ ] report generation and longitudinal reasoning benchmarks
- [ ] calibration, uncertainty, selective prediction, and abstention
- [ ] hallucination, demographic robustness, and shortcut audits
- [ ] cross-modal consistency evaluation across text, image, and editing outputs
- [ ] signed public leaderboard manifests and reproducibility bundles

## Scale target

After the platform and governance gates hold across the first real slices,
expand across radiology, pathology, ophthalmology, dermatology, endoscopy,
ultrasound, medical video, documents, and longitudinal multimodal records. Each
new entry must ship as a model + dataset + benchmark + reproducible recipe slice
where applicable; raw counts are not accepted as evidence of platform support.
