# MedUMM v0.6: task-aware medical evaluation

Version 0.6 adds a medical task semantic layer above the stable model and
execution interfaces. It does not treat every image question as natural-image
classification, and it does not report a single top-1 score as medical ability.

The design is informed by the clinical hierarchy used in
[VisionUnite](https://arxiv.org/pdf/2408.02865): perception, disease/condition
reasoning, evidence-backed explanation, and interactive clinical language are
different outputs and need different validation. MedUMM implements the general
task and evaluation contracts only. It does not bundle VisionUnite weights;
the [upstream repository](https://github.com/HUANGLIZI/VisionUnite) states
additional restrictions for its pretrained models.

## Stable task taxonomy

`MedicalTaskType` v1.0 contains eight mutually exclusive intents:

| Family | Task | Expected output | Primary success rule |
|---|---|---|---|
| perception | `finding_assessment` | presence/absence of a finding | exact closed answer |
| perception | `clinical_description` | visible medical observations | reference concept coverage |
| perception | `anatomy_localization` | anatomical site or side | closed exact or open concept coverage |
| perception | `quantitative_assessment` | count, size, or measurement | closed exact or open concept coverage |
| perception | `image_context` | modality, view, plane, or acquisition context | closed exact or open concept coverage |
| reasoning | `diagnostic_reasoning` | diagnosis plus supporting evidence | all diagnosis concepts, no extra affirmed concepts, required evidence |
| generation | `report_generation` | findings/impression-style text | structured concept coverage |
| generation | `patient_communication` | plain-language, uncertainty-aware explanation | communication concept coverage |

Clinical explanation is represented by the structured `evidence` target inside
`diagnostic_reasoning`. This keeps the top-level tasks mutually exclusive while
allowing diagnosis and explanation to be scored separately.

`InferenceRequest` exposes this as an optional `medical_task` field alongside
the existing execution `task`. For example, `task: understanding` plus
`medical_task: report_generation` means medical image/text to report text;
generic `task: generation` remains the text-to-image/content pipeline.

## Dataset contract

The `medical_tasks_jsonl` adapter accepts local JSON or JSONL. Every item has:

```json
{
  "id": "case-001-turn-1",
  "task": "diagnostic_reasoning",
  "prompt": "What is the likely diagnosis and supporting evidence?",
  "image": "case-001.png",
  "references": ["pneumonia supported by airspace opacity"],
  "concepts": ["pneumonia"],
  "evidence": ["airspace opacity"],
  "specialty": "radiology",
  "modality": "xray",
  "anatomy": "chest",
  "reference_provenance": {
    "kind": "expert_annotation"
  }
}
```

`case_id` and `turn_index` support multi-turn cases. `choices`, `answer_type`,
and `language` support closed and multilingual slices. Task labels and reference
answers have separate provenance: an answer can come from the source dataset
while the task mapping is a transparent non-expert heuristic.

The preflight audit checks:

- required tasks and minimum sample count per task;
- reference and image integrity;
- reference provenance and dataset provenance;
- de-identification and research-only declarations;
- missing structured concepts/evidence for reasoning and long-form tasks;
- the number of heuristic versus native/expert task labels.

## Task-aware metrics

The registered `medical_task_core` v1.0 suite reports:

- task-specific success and macro task success;
- exact match and token F1 as supporting, not universal, measures;
- concept precision, recall, and F1;
- evidence coverage;
- negation-aware extra affirmed concept rate;
- strict diagnostic accuracy;
- abstention rate;
- Wilson confidence intervals for binary task/diagnostic success;
- seeded bootstrap intervals for continuous language metrics.

Concept vocabularies are isolated per task so a valid report term cannot create
a false hallucination penalty in an unrelated task. The lightweight negation
guard distinguishes an affirmed concept from statements such as “no pleural
effusion”. It is an auditable deterministic baseline, not a replacement for
expert review or a clinical terminology engine.

## Real acceptance slice

The v0.6 server recipe uses the pinned VQA-RAD test parquet already established
in v0.4 and exports 24 samples: four each for finding assessment, clinical
description, anatomy localization, quantitative assessment, image context, and
diagnostic reasoning.

VQA-RAD does not provide this eight-task taxonomy. The exporter therefore:

1. applies versioned, deterministic question-pattern rules;
2. saves the rule and source index on every sample;
3. marks every mapping `expert_validated: false`;
4. audits and reports the full source and selected task distributions;
5. makes no claim that VQA-RAD validates report generation or patient
   communication.

The last two tasks are covered by schema, software workflow, and synthetic
contract tests in v0.6. They require appropriate structured real datasets in a
future vertical slice; labels are never fabricated from classification data.

Prepare and run locally with a compatible environment:

```bash
python scripts/prepare_vqa_rad_tasks.py \
  --parquet-path /path/to/vqa-rad-test.parquet \
  --output-directory data/vqa_rad_tasks_v0.6 \
  --samples-per-task 4

medumm evaluate \
  --config configs/evaluation/vqa_rad_medical_tasks_v0.6.yaml
```

Submit the complete server workflow:

```bash
sbatch scripts/slurm_medical_tasks_v0.6.sh
```

The job runs all tests, performs real LLaVA-Med CUDA inference, writes the
standard evaluation artifacts, and fails unless the verification bundle proves
task balance, mapping disclosure, reference provenance, task metrics,
uncertainty methods, CUDA execution, and peak GPU memory evidence.

The LLaVA-Med adapter follows the upstream `model_vqa` generation behavior by
default: standard EOS stopping is enabled and keyword stopping is opt-in. Run
metadata includes `generated_tokens` and `keyword_stopping`, so unintended
one-token truncation cannot be hidden by non-empty-output checks.

The accepted run is recorded in
[`docs/results/v0.6-medical-tasks.json`](results/v0.6-medical-tasks.json): 24
predictions from 20 cases on one NVIDIA A800, 36.0 mean generated tokens,
15,213.8 MiB peak allocated memory, 20.83% task success, and 75.0% strict
diagnostic accuracy on four heuristic-mapped diagnosis questions. These
small-slice values prove the workflow and expose model behavior; they are not
claims of clinical validity.
VQA-RAD diagnosis answers contain no reference evidence, so this slice reports
strict diagnostic accuracy but deliberately cannot award diagnostic-reasoning
task success.

## Scope and safety

All v0.6 outputs are for research evaluation. MedUMM does not diagnose patients,
does not convert a benchmark label into a clinical explanation, and does not
claim clinical validity from a software acceptance run.
