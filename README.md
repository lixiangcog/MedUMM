# MedUMM

MedUMM is an open toolkit for building, evaluating, and post-training medical
multimodal models behind one stable interface.

> Status: early research preview. MedUMM is not a medical device and must not be
> used for diagnosis, treatment, or other clinical decisions.

## What works in v0.1

- One configuration-driven CLI: `medumm`
- Medical image understanding through a common model adapter
- Medical VQA evaluation with resumable prediction files and JSON/CSV reports
- A small supervised post-training path that produces a reusable checkpoint
- A local reference adapter, a trainable NumPy baseline, and an optional
  MedGemma adapter
- Slurm scripts and a complete train → infer → evaluate smoke workflow

The first release deliberately keeps the surface small. Models and datasets are
added one at a time only after they fit the same interfaces and pass the common
workflow.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,medical]"
```

For the lightweight smoke workflow, only PyYAML, NumPy, and Pillow are needed:

```bash
pip install -e ".[test,baseline]"
```

## Run the complete workflow

```bash
bash scripts/run_reference_workflow.sh
```

Or run each stage separately:

```bash
medumm post-train --config configs/post_training/medical_sft_smoke.yaml
medumm infer --config configs/inference/medical_linear_understanding.yaml
medumm evaluate --config configs/evaluation/medical_vqa_linear_smoke.yaml
```

Outputs are written under `outputs/` and are ignored by Git.

## Interface

Every model adapter implements the same lifecycle:

```python
from medumm.inference import InferencePipeline

with InferencePipeline(
    backbone_name="medical_reference",
    backbone_config={},
) as pipeline:
    result = pipeline.run({
        "task": "understanding",
        "prompt": "Describe this image.",
        "images": ["examples/medical/images/synthetic_scan.pgm"],
    })
```

The CLI accepts dotted configuration overrides:

```bash
medumm evaluate \
  --config configs/evaluation/medical_vqa_linear_smoke.yaml \
  --set data.max_samples=2
```

## Repository layout

```text
src/medumm/         stable Python API and CLI
configs/            inference, evaluation, and post-training recipes
examples/medical/   synthetic, non-clinical smoke data
scripts/            local and Slurm workflows
tests/              unit and end-to-end tests
docs/               architecture and expansion roadmap
```

See [docs/architecture.md](docs/architecture.md) for the component contract and
[docs/roadmap.md](docs/roadmap.md) for the gradual expansion plan.

## License

Apache-2.0. Model weights and datasets keep their own licenses and terms.
