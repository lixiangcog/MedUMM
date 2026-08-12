# v0.3 real medical model slice

MedUMM v0.3 runs Microsoft LLaVA-Med v1.5 Mistral-7B through the same model,
inference, dataset, evaluation, reporting, and runtime contracts established in
v0.2. It is the first real-model proof for the understanding path.

## Pinned components

| Component | Identifier | Revision |
|---|---|---|
| Model weights | `microsoft/llava-med-v1.5-mistral-7b` | `91bb16c122001ddc9cf1fd36ce1dae09448943a2` |
| Official model source | `microsoft/LLaVA-Med` | `30697ca50b5c29a8e955c99330b259776aef27b9` |
| Vision tower | `openai/clip-vit-large-patch14-336` | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| Evaluation data | `flaviagiammarino/vqa-rad` | `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` |

The smoke evaluation selects four closed yes/no records from the public test
split. Export preserves source indices and stable identifiers. Images and model
assets are runtime files under ignored directories and are never committed.

## Slurm recipe

Asset preparation is a CPU job. The script creates `.venv-llava-med`, obtains
the pinned source/model/vision/dataset files, verifies expected weight sizes and
SHA-256 digests, rewrites only the local vision-tower path, and writes
provenance manifests.

```bash
HF_ENDPOINT=https://hf-mirror.com \
MEDUMM_DYNAMIC_PROXY_TARGET=<login-node-ssh-target> \
sbatch --export=ALL scripts/slurm_prepare_llava_med_assets.sh
```

After it completes, submit the GPU job:

```bash
sbatch scripts/slurm_llava_med_vqa_rad.sh
```

The GPU job requests one A800, runs the complete test suite, performs one direct
image inference, evaluates the four-record VQA-RAD slice, and rejects the run if
answers are empty or CUDA, latency, or peak-memory evidence is absent.

Primary outputs are:

```text
outputs/inference/llava_med_vqa_rad.json
outputs/evaluation/vqa_rad_llava_med_smoke/predictions.jsonl
outputs/evaluation/vqa_rad_llava_med_smoke/score.json
outputs/verification/llava_med_v0.3.json
```

## Compatibility boundary

The official LLaVA-Med implementation is pinned to Transformers 4.36.2. The
recipe deliberately isolates it from newer MedGemma/vLLM environments. Model
loading stays lazy: `medumm catalog` and non-LLaVA workflows do not import its
heavy dependencies.

## Verified result

The v0.3 acceptance run completed on 2026-08-12 using one NVIDIA A800-SXM4
80 GB device inside an existing Slurm allocation step (a separately submitted
job was held by the account's per-user GPU quota). The server test suite passed
37 tests. Direct VQA returned `Yes` in 411.47 ms. The four-sample closed
VQA-RAD smoke slice reached 50.0% exact match and token F1, with 143.62 ms mean
generation time and 15,159.58 MiB maximum allocated GPU memory.

These four records are an execution proof, not a statistically meaningful
quality benchmark. The machine-readable evidence is committed at
`docs/results/v0.3-llava-med-vqa-rad.json`.

## Intended use and licenses

This integration is for research evaluation only. It is not for diagnosis,
treatment, clinical decisions, or deployed use. The upstream model card states
that the checkpoint is intended for research and reproducibility and describes
it as unsuitable for clinical use. The source checkout includes Microsoft
Research License terms with non-commercial and redistribution restrictions;
the model card also carries its own license metadata. VQA-RAD's dataset card
declares CC0-1.0. Always review the upstream model, source, base-model, and
dataset terms before use; MedUMM's Apache-2.0 license does not replace them.
