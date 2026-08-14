# Explicit real-model adapters (v1.4)

## Outcome

MedUMM v1.4 separates three claims that were previously easy to confuse:

1. **Cataloged** means the model has source, license, access, revision, task and
   environment metadata.
2. **Adapter-defined** means MedUMM fixes a model-specific class, processor,
   prompt protocol and executor instead of guessing from a broad family tag.
3. **Runtime-validated** means pinned real weights completed the public MedUMM
   interface on an allocated GPU and committed machine-readable evidence.

All 32 models are cataloged and adapter-defined. Eighteen select MedUMM's
Transformers/OpenCLIP-style built-in executors; fourteen retain pinned
official-source runtime recipes because their upstream stacks are
repository-specific. Seven of 32 now have committed GPU evidence. The other 25
remain `interface_validated` and must not be reported as successful model runs.

## Stable adapter surface

The recipe catalog records, for every model release:

- exact executor and upstream `model_type`;
- concrete model and processor classes where the runtime is library-native;
- prompt/chat or contrastive-ranking protocol;
- official source entry point and source-checkout requirement where applicable;
- maximum supported image count and model-specific notes.

The public CLI exposes the same information:

```bash
medumm models list
medumm models show medvlm_r1
medumm models audit
medumm models preflight medvlm_r1 \
  --model-path /models/medvlm-r1 \
  --revision d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9
```

Preflight fails closed on a mutable or mismatched revision, missing local
snapshot, unaccepted restricted terms, missing pinned source checkout, source
commit drift, or missing imports. The former arbitrary `module:Class` bridge
placeholder is no longer part of resource templates.

## New A800-validated models

Slurm job `437526` completed on `gpu01`, one NVIDIA A800-SXM4-80GB, with exit
code 0. Every model ran in its own Python 3.10 environment with Torch
2.7.1+cu126. Times below are single-request model-forward/generation times after
process-level model loading; they are acceptance measurements, not throughput
benchmarks.

| Model | Executor | Immutable revision | Time | Peak allocated GPU memory |
|---|---|---|---:|---:|
| PLIP | `transformers_contrastive` | `67ade53ddd32195868f422585f72698ef5d15094` | 280.73 ms | 588.12 MiB |
| QuiltNet | `transformers_contrastive` | `8ce77289ce35a90b2f1db1137dfa4bc2df175e33` | 279.15 ms | 588.10 MiB |
| MedVLM-R1 | `qwen2_vl_chat` | `d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9` | 1702.38 ms | 4337.41 MiB |
| BiomedCLIP | `open_clip_hf_hub` | `9f341de24bfb00180f1b847274256e9b65a3a32e` | 230.59 ms | 777.61 MiB |

The smoke input is a real PathVQA image already prepared by MedUMM. Candidate
scores and generated text prove data flow and model execution only; their
clinical or benchmark quality is not inferred from one sample.

## Defects found by real execution

Real validation exposed three issues that catalog registration and import tests
did not catch:

- QuiltNet's pinned release is a native Transformers `CLIPModel`, so its runtime
  contract was corrected from OpenCLIP to Transformers contrastive.
- BiomedCLIP's OpenCLIP configuration instantiates a Hugging Face text encoder;
  `transformers` and `tokenizers` are now explicit locked dependencies, and the
  auxiliary BiomedBERT config/tokenizer is pinned locally.
- MedVLM-R1 stores `use_cache: null` in its generation configuration, while its
  official demo explicitly invokes `generate(..., use_cache=True)`. The first
  A800 attempt (`437506`) failed with a Qwen2-VL attention-mask length mismatch;
  the explicit cache setting fixed the second job.

## Reproduction on the current cluster

The compute nodes have no external DNS. Asset and environment preparation must
therefore run on a networked login/build node, while GPU acceptance is entirely
offline:

```bash
python scripts/prepare_real_model_adapters_v1_4.py \
  --asset-root /data/user/hd66945/MedUMM-assets/v1.4 \
  --models plip quiltnet medvlm_r1 biomedclip

for model in plip quiltnet medvlm_r1 biomedclip; do
  bash scripts/setup_model_env.sh "$model"
done

sbatch scripts/slurm_real_model_adapters_v1.4.sh
```

The final Slurm script sets all Hugging Face libraries offline, verifies CUDA,
runs focused contract tests, executes the four public inference configs, and
then rejects any result missing the expected model identity, revision, executor,
CUDA device, matching Slurm job ID, non-empty output, or measured latency.

## Evidence and remaining gap

The committed record is
[`docs/results/v1.4-real-model-adapters.json`](results/v1.4-real-model-adapters.json).
It includes environment fingerprints, exact versions, GPU identity, outputs,
candidate scores, latency and peak memory.

The largest remaining gap is still real-model breadth: 25 model releases lack a
committed GPU task run. Priority should now follow distinct runtime families,
not catalog order: CheXagent, InternVL/Fleming, MedGemma, M3D-LaMed, one
LLaVA-Qwen model, then the repository-specific Flamingo/RadFM/VILA/XrayGPT
stacks and multi-GPU 27B–34B releases.
