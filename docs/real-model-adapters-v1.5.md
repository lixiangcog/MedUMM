# Wider real-model runtime coverage (v1.5)

## Outcome

MedUMM v1.5 moves four more catalog entries from interface-defined to
runtime-validated. Every accepted path loaded an immutable local snapshot,
executed a real image request through the public `medumm infer` command on a
Slurm-allocated GPU, returned non-empty text, and recorded its executor,
revision, device, scheduler identity, latency, peak allocated memory, and
isolated environment.

The total is now 11 runtime-validated models out of 32. This evidence validates
software integration and data flow; it does not establish clinical quality,
benchmark accuracy, safety, or support for every modality advertised by each
upstream release.

## Accepted models

Slurm job `437697` completed on `gpu01` with exit code 0 using one NVIDIA
A800-SXM4-80GB. Timings measure one generation after model loading and are not
throughput benchmarks.

| Model | Explicit executor | Immutable revision | Transformers | Time | Peak GPU memory |
|---|---|---|---:|---:|---:|
| MedMO-4B | `qwen3_vl_chat` | `0e220705d851598b37725326aadb852aa8b37f43` | 4.57.1 | 938.11 ms | 8524.51 MiB |
| MedMO-8B | `qwen3_vl_chat` | `8eafab80545fb4d60b0fb126a097e972e6475851` | 4.57.1 | 1002.42 ms | 17246.90 MiB |
| Lingshu-I-8B | `internvl_transformers` | `b004bfc0554d90bd44baedf4de08c361e71ef017` | 4.52.4 | 4260.66 ms | 15846.52 MiB |
| Fleming-VL-8B | `internvl_chat` | `801e2bef9645bca0646d55837a6630fb468e2901` | 4.46.0 | 4298.79 ms | 15286.65 MiB |

MedMO-4B, MedMO-8B, and Lingshu-I-8B ran with Torch 2.7.1 and CUDA 12.6.
Fleming's upstream-compatible environment ran with Torch 2.3.0 and CUDA 12.1.
All four used Python 3.10.20 and MedUMM 1.5.0 in separate environment
directories.

## Adapter details

MedMO uses the Qwen3-VL processor chat template and the concrete
`Qwen3VLForConditionalGeneration` implementation. Lingshu-I is not routed
through a generic image-to-text pipeline: MedUMM loads the native
`InternVLForConditionalGeneration` class and renders the upstream InternVL 2.5
MPT conversation, including `<IMG_CONTEXT>` and `<|im_start|>` boundaries.
Floating image tensors are explicitly cast to the model's precision while
integer token IDs retain their type. Fleming uses the model release's pinned
remote InternVL implementation and official `.chat()` interface with dynamic
image tiling.

## Defects found by real execution

The first full attempt, Slurm job `437692`, passed both MedMO models but exposed
a Lingshu-I vision-input mismatch: the processor produced `float32` pixels
while the loaded vision tower used `bfloat16`. The executor now aligns only
floating inputs to the model precision before generation. Job `437697` then
passed all four models.

Environment construction also exposed unreliable transfer of Fleming's 781 MB
Torch wheel. The environment bootstrap now supports a pinned local pip wheel
and pip's resume retries, while preserving the existing immutable dependency
hash check.

## Access boundaries

The asset preparation step made authenticated configuration probes for three
high-priority restricted models. On the current server all remain blocked by
upstream terms or an authorized token and therefore remain non-runtime claims:

- MedSigLIP (`google/medsiglip-448`)
- MedGemma 1.5 4B IT (`google/medgemma-1.5-4b-it`)
- MAIRA-2 (`microsoft/maira-2`)

MedUMM does not bypass these controls or treat a failed access probe as model
validation.

## Reproduction

The cluster compute nodes are offline, so immutable snapshots and isolated
environments are prepared on the login/build node before submission:

```bash
bash scripts/prepare_real_model_assets_v1.5.sh
bash scripts/prepare_model_envs_v1.5.sh
sbatch scripts/slurm_real_model_adapters_v1.5.sh
```

The Slurm acceptance disables external model and dataset access, performs a
CUDA preflight, runs focused catalog/environment tests, invokes the four public
inference configs, and runs a strict verifier. The verifier rejects a wrong
model identity, revision, executor, device, Slurm job ID, MedUMM version,
Transformers version, empty response, or missing latency.

## Evidence and next queue

The complete record is
[`docs/results/v1.5-real-model-adapters.json`](results/v1.5-real-model-adapters.json).
It includes responses, environment fingerprints, package/CUDA versions, gated
access outcomes, timing and memory data.

Twenty-one catalog releases still lack committed GPU execution. The next work
should prioritize a new runtime family rather than another catalog alias:
CheXagent, M3D-LaMed, a LLaVA-Qwen derivative, then the source-specific
Med-Flamingo/RadFM/VILA/XrayGPT stacks. Restricted models enter the runtime
queue only after their upstream terms and access requirements are satisfied.
