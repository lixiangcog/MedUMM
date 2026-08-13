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

## v0.6 — task-aware medical semantics

- [x] eight-intent medical task taxonomy across perception, reasoning, and generation
- [x] structured concepts, evidence, case/turn, and reference-provenance schema
- [x] independently registered task-aware dataset, benchmark, and metric suite
- [x] task-specific success instead of a universal natural-image top-1 score
- [x] concept/evidence coverage and negation-aware extra affirmed concepts
- [x] strict diagnostic accuracy plus Wilson and bootstrap confidence intervals
- [x] task coverage, target completeness, mapping, and governance audit gates
- [x] balanced six-task real VQA-RAD exporter with explicit heuristic-label disclosure
- [x] LLaVA-Med A800 acceptance and machine-readable evidence recipe
- [ ] expert-labelled report-generation benchmark slice
- [ ] expert-labelled multi-turn patient-communication benchmark slice

## v0.7 — advanced post-training and research methods

- [x] Transformers causal-LM post-training plugin with PEFT checkpoint output
- [x] LoRA plus optional 4-bit QLoRA loading path
- [x] completion-only SFT, DPO, SimPO, and ORPO objectives
- [x] clinical-relevance-weighted DPO objective
- [x] frozen-reference DPO without a duplicate base-model allocation
- [x] weighted multi-dataset mixture schema and deterministic epoch sampling
- [x] preference rationale, annotation-source, safety, and provenance contract
- [x] license, de-identification, expert-status, and invalid-weight audit gates
- [x] non-zero gradient, before/after metric, PEFT save/reload, and CUDA evidence gates
- [ ] clinician-annotated medical safety preference slice
- [ ] real multimodal MMedPO-style lesion-grounded preference run
- [ ] multi-node FSDP/DeepSpeed alignment recipe

## v0.8 — medical resource scale catalog

- [x] 32 individually registered medical multimodal model resources
- [x] 34 individually registered medical evaluation dataset resources
- [x] typed source, paper/code, license, access, revision, task, modality, and domain specs
- [x] Transformers image-text, Transformers contrastive, OpenCLIP, and official-runtime families
- [x] normalized source-pinned dataset adapter with gated/credentialed access checks
- [x] resource list/show/template/validate CLI and Python catalog API
- [x] explicit catalog/interface/runtime validation levels
- [x] 188/188 populated source, paper, and official-code URLs reachable at release audit
- [x] catalog-alias LLaVA-Med + official OSF VQA-RAD A800 acceptance job and evidence
- [ ] runtime validation for one Qwen-derived medical generative VLM
- [ ] runtime validation for one medical contrastive encoder
- [ ] raw-source exporters and acceptance runs for SLAKE and PathVQA
- [ ] source-specific report, grounding, measurement, fairness, 3D, and video scorers

## v0.9 — architecture-diverse runtime slices

- [x] native Lingshu-7B Qwen2.5-VL adapter with immutable revision and run metadata
- [x] SLAKE and PathVQA raw-source exporters with image/provenance normalization
- [x] sample-level candidate propagation for contrastive zero-shot evaluation
- [x] fixed MedMNIST v2 PneumoniaMNIST exporter
- [x] pinned PubMedCLIP asset recipe and shared benchmark/report path
- [x] passed A800 acceptance for Lingshu-7B + SLAKE
- [x] passed A800 acceptance for PubMedCLIP + PneumoniaMNIST
- [ ] gated MedSigLIP acceptance after source terms and weight access
- [x] PathVQA balanced real acceptance on A800
- [x] PathVQA pathology-specific scoring protocol

## v1.0 — medical-specific evaluation contracts

- [x] independently registered pathology VQA answer-type protocol
- [x] auditable report fact, negation, contradiction, critical-finding, and section scoring
- [x] normalized box-IoU and point-grounding metrics
- [x] unit-aware measurement error and per-reference tolerance scoring
- [x] ECE, Brier, NLL, confidence, and selective prediction from preserved model scores
- [x] minimum-size-gated subgroup and max-min disparity summaries
- [x] pinned balanced PathVQA + Lingshu A800 acceptance
- [ ] expert-labelled report-generation runtime slice
- [ ] public grounding/measurement model + dataset runtime slice

## v1.2 — optimized inference engines

- [x] stable native/vLLM/SGLang backend and scheduler configuration
- [x] OpenAI-compatible vLLM and SGLang continuous-batching client
- [x] TP/PP/DP server launch planning with immutable model revisions
- [x] native Emu3.5 vLLM 0.11.0 adapter with cond/uncond CFG scheduling
- [x] strict rejection of unsupported Emu3.5 CFG over SGLang/HTTP
- [x] latency, TTFT, queue, request-throughput, and token-throughput reports
- [x] sequential-versus-concurrent two-GPU Slurm acceptance recipe
- [ ] full pinned Emu3.5-Image generation evidence after 72 GB asset retrieval
- [ ] multi-node backend acceptance and production load/stability benchmark

## Scale target

After the platform and governance gates hold across the first real slices,
expand across radiology, pathology, ophthalmology, dermatology, endoscopy,
ultrasound, medical video, documents, and longitudinal multimodal records. Each
new entry must ship as a model + dataset + benchmark + reproducible recipe slice
where applicable; raw counts are not accepted as evidence of platform support.
