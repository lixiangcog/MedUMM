# MedUMM v0.7: advanced post-training and research methods

Version 0.7 turns post-training into a first-class research layer. The previous
`medical_sft` plugin remains a dependency-light linear workflow test. The new
`medical_alignment` plugin performs real causal-language-model optimization and
writes loadable PEFT adapters.

## Method contract

One trainer exposes five objectives:

| Objective | Data | Reference policy | Sequence score | Primary use |
|---|---|---|---|---|
| `sft` | prompt + chosen | no | completion NLL | supervised adaptation |
| `dpo` | prompt + chosen/rejected | frozen base | summed completion log-probability | standard offline preference alignment |
| `simpo` | prompt + chosen/rejected | no | mean completion log-probability plus target margin | length-normalized reference-free alignment |
| `orpo` | prompt + chosen/rejected | no | completion NLL plus odds-ratio penalty | monolithic supervised/preference training |
| `clinical_dpo` | prompt + chosen/rejected + relevance | frozen base | DPO weighted by positive clinical relevance | auditable medical priority weighting |

LoRA is the default adapter. `model.quantization: 4bit` activates the optional
QLoRA path with NF4, double quantization, and a configurable compute dtype. The
implementation accepts `target_modules: all-linear` or an architecture-specific
module list.

For DPO, MedUMM does not allocate a second base model. The policy forward uses
the trainable LoRA adapter; the reference forward runs without gradients while
the adapter is disabled. Therefore the reference is exactly the pinned frozen
base policy and memory remains suitable for larger models.

Only completion tokens contribute to sequence scores. Prompt and padding tokens
are masked once in the shared objective layer. SFT, DPO, SimPO, ORPO, and
clinical DPO therefore use the same causal shift and token-accounting contract.

## Alignment data and mixtures

Local JSON/JSONL rows accept:

```json
{
  "id": "case-001",
  "prompt": "Explain this benchmark result safely.",
  "chosen": "Preferred response.",
  "rejected": "Dispreferred response.",
  "medical_task": "patient_communication",
  "specialty": "medication_safety",
  "safety_categories": ["unsafe_dosing", "escalation"],
  "preference_rationale": "Why the chosen output is preferred.",
  "label_source": "clinician",
  "clinical_relevance": 1.5,
  "images": ["optional-study-image.png"],
  "modality": "radiograph",
  "anatomy": "chest",
  "preference_provenance": {
    "kind": "paired_expert_review",
    "review_protocol": "protocol-v1"
  }
}
```

The common alignment record preserves optional medical image paths, modality,
and anatomy. The v0.7 `medical_alignment` implementation deliberately rejects
records with images because its accepted backbone is a causal text LM; it never
silently drops multimodal inputs. A processor-backed VLM alignment plugin is a
separate roadmap item.

`data.mixtures` composes multiple sources. Every source has its own name,
sampling weight, content hash, license, revision, provenance document, and
de-identification declaration. Sample identifiers are namespaced by source.
Epoch sampling is deterministic for a seed and epoch; source weights affect
sampling probability without changing the original records.

The audit rejects missing licenses, invalid or non-finite relevance weights,
undeclared de-identification, and required provenance/rationale failures. It
separately reports clinician/expert versus AI/synthetic labels. A field named
`human` in an upstream dataset is never promoted to clinician annotation unless
the preference record explicitly declares that evidence level.

## Real acceptance recipe

The v0.7 acceptance slice uses:

- `EleutherAI/pythia-14m` at revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`, Apache-2.0;
- `TsinghuaC3I/UltraMedical-Preference` test data at revision
  `761eb7935310ba662a96d93c5af342e5269d5759`, MIT;
- eight deterministic, short MedMCQA pairs after an exam-source allowlist and
  automated direct-identifier scan;
- LoRA rank 8, DPO beta 0.1, three epochs, and one A800 GPU.

The upstream preference judgments are model-based, not clinician annotations.
The small Pythia model is a pretrained causal LM selected to make a real
parameter-update test inexpensive. It is not a medical foundation model and
the run does not demonstrate clinical alignment quality.

Prepare assets outside the compute job, then submit:

```bash
bash scripts/setup_alignment_env.sh
python scripts/prepare_alignment_assets.py \
  --output-directory /path/to/MedUMM-assets/pythia-14m-v0.7
MEDUMM_ASSET_ROOT=/path/to/MedUMM-assets \
  sbatch scripts/slurm_alignment_v0.7.sh
```

For gateways that cannot stream a large Hugging Face object reliably, the
repository also provides `scripts/download_verified_ranges.py`. It accepts an
exact URL, byte size, and SHA-256, assembles the ranges atomically, and removes
temporary parts after verification; it does not weaken revision pinning.

The job reruns all tests, reconstructs the pinned preference slice, trains the
adapter, loads it into a fresh base-model instance, and fails unless the
verification bundle proves:

- exact model/dataset revisions and licenses;
- eight aligned preference pairs with complete provenance and rationale;
- no mislabeling of AI judgments as clinician/expert annotations;
- CUDA and Slurm execution with peak memory evidence;
- a non-empty PEFT safetensors checkpoint and non-zero gradients;
- before/after loss, preference accuracy, and reward-margin measurements;
- independently reloadable adapter weights.

The accepted run completed as Slurm step `436330.21` on one
NVIDIA A800-SXM4-80GB. It executed six DPO optimizer steps, saved 98,304
trainable LoRA parameters, and reloaded the saved adapter. On this eight-pair
software acceptance slice, preference accuracy changed from 0.0 to 1.0 and the
reward margin from 0.0 to 1.661. These measurements are pipeline evidence, not
an estimate of generalization or medical quality. The complete machine-readable
record is in
[`docs/results/v0.7-advanced-post-training.json`](results/v0.7-advanced-post-training.json).

## Research boundaries

DPO follows the [direct preference objective introduced by Rafailov et
al.](https://arxiv.org/abs/2305.18290). [SimPO](https://arxiv.org/abs/2405.14734)
uses a length-normalized, reference-free reward with a target margin.
[ORPO](https://arxiv.org/abs/2403.07691) combines chosen-response likelihood
with an odds-ratio preference term.
The multimodal data fields and relevance weighting make room for lesion-grounded
medical VLM alignment, but v0.7 does not claim an
[MMedPO](https://arxiv.org/abs/2412.06141) reproduction: no
lesion-noised images or clinician-relevance scores are present in the accepted
text-only slice.

All artifacts are research-only. Preference optimization can amplify annotation
errors and does not establish correctness, safety, calibration, fairness, or
clinical utility without independent evaluation and qualified review.
