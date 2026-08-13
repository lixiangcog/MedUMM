# v0.9 architecture-diverse runtime slices

MedUMM v0.9 validates the platform with two model families and two medical
datasets rather than adding more interface-only names. This is software and
workflow validation for research use; it is not a clinical-performance claim.

## Slice A: Lingshu-7B and SLAKE

`lingshu_7b` is a native adapter over the upstream Qwen2.5-VL implementation.
It uses `Qwen2_5_VLForConditionalGeneration`, the processor chat template, and
`qwen_vl_utils.process_vision_info`. The adapter supports one to four images,
requires a full immutable source revision, trims prompt tokens before decoding,
and records model identity, dtype/device, latency, generated-token count, peak
allocated CUDA memory, hostname, and Slurm identifiers.

Accepted source identities:

- model: `lingshu-medical-mllm/Lingshu-7B` at
  `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`;
- dataset: `BoKelvin/SLAKE` at
  `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`.

The SLAKE exporter preserves language, modality, anatomy/location, content
type, source question/image identifiers, and source-image SHA-256. It copies
images into a self-contained MedUMM manifest and can select English, Chinese,
or both languages plus closed-only subsets.

## Slice B: PubMedCLIP and PneumoniaMNIST

PubMedCLIP uses the shared Transformers contrastive executor. Each normalized
sample supplies its own candidate labels; the benchmark passes these labels to
the model request and reports the selected label, probability map, latency,
device/dtype, peak CUDA memory, source revision, and Slurm provenance.

Accepted source identities:

- model: `flaviagiammarino/pubmed-clip-vit-base-patch32` at
  `26c0c67f6da303ad2a38909130bd35744ea93517`;
- dataset: official MedMNIST PneumoniaMNIST fixed release `v2`, with archive MD5
  `28209eda62fecd6e6a2d98b1501bb15f`.

The exporter uses the official train/validation/test arrays, verifies the
archive checksum, preserves source indices and labels, and emits the medical
labels `normal chest x-ray` and `pneumonia chest x-ray`.

## Reproduction

Prepare all pinned assets on a CPU node:

```bash
sbatch scripts/slurm_prepare_runtime_slices_v0.9.sh
```

After preparation succeeds, run both slices offline on one A800:

```bash
sbatch scripts/slurm_runtime_slices_v0.9.sh
```

The acceptance job first runs the full test suite, evaluates both slices, and
then writes `outputs/verification/runtime_slices_v0.9.json`. A release may mark
resources `runtime_validated` only when that evidence reports `status: passed`.

The accepted run completed as Slurm job `437260` on one NVIDIA A800-SXM4-80GB
with Python 3.10.20, PyTorch 2.7.1+cu126, and CUDA 12.6. Lingshu-7B evaluated
four SLAKE samples at 341.67 ms mean generation time and 15,908.22 MiB peak
allocated memory. PubMedCLIP evaluated 32 PneumoniaMNIST samples at 27.75 ms
mean ranking time and 588.12 MiB peak allocated memory. The small-slice scores
(75.0% and 28.12% exact match respectively) validate execution and are not
model-quality estimates. Machine-readable evidence is committed at
[`docs/results/v0.9-runtime-slices.json`](results/v0.9-runtime-slices.json).

## Boundaries

- Four SLAKE questions and 32 PneumoniaMNIST images are wiring slices, not
  statistically meaningful model comparisons.
- PathVQA now has a raw-source exporter but is not part of this GPU acceptance.
- MedSigLIP is gated and is not treated as runtime validated without explicit
  access to its upstream terms and weights.
- All outputs require independent review and are prohibited from clinical use.
