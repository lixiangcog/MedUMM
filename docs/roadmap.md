# Roadmap

MedUMM grows by validated vertical slices instead of importing a large set of
partially working integrations.

## Phase 1 — workflow foundation

- [x] package and CLI
- [x] model registry and common adapter
- [x] configuration-driven inference
- [x] medical VQA evaluation and reports
- [x] supervised post-training checkpoint loop
- [x] local and Slurm smoke workflows

## Phase 2 — real medical backbones and datasets

- [ ] production MedGemma recipe and licensed checkpoint documentation
- [ ] Qwen medical vision-language adapter
- [ ] radiology and pathology dataset adapters
- [ ] distributed inference and sharded evaluation
- [ ] LoRA supervised fine-tuning

## Phase 3 — scale and safety

- [ ] multi-dataset mixture and provenance registry
- [ ] calibration, uncertainty, and abstention benchmarks
- [ ] medical hallucination and localization evaluation
- [ ] distributed post-training and experiment tracking
- [ ] reproducible public leaderboard artifacts

Dataset adapters will store provenance and split metadata. Clinical data is out
of scope unless its governance, de-identification, and access rules are explicit.
