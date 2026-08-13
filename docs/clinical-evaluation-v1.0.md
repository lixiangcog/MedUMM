# MedUMM v1.0 clinical evaluation contracts

Version 1.0 closes the largest evaluation-interface gap left after the model
and dataset catalog expansion. Five independently registered metric suites now
run through the same benchmark-neutral `generate → score → report` state
machine. This release validates software and benchmark wiring for research; it
does not establish clinical safety or model quality.

## Versioned metric suites

| Suite | Required annotations or outputs | Main aggregate metrics |
|---|---|---|
| `pathology_vqa` 1.0 | PathVQA-style references and answer type | yes/no, free-form, overall, and macro answer-type exact accuracy |
| `medical_report_factuality` 1.0 | positive, negative, critical, vocabulary, and optional required-section annotations | fact precision/recall/F1, contradictions, negative assertion accuracy, critical recall, section completeness |
| `medical_grounding` 1.0 | reference boxes or points; image size for pixel coordinates | normalized mean IoU, IoU@0.5 recall, normalized point distance, pointing accuracy |
| `medical_measurement` 1.0 | values, units, and per-reference tolerances | unit-aware MAE, mean relative error, tolerance accuracy, unit errors |
| `medical_calibration` 1.0 | all candidate probabilities plus the accepted answer | ECE, Brier score, NLL, confidence, coverage/accuracy at declared thresholds |

Each specialized scorer returns an explicit availability flag. If the source
does not provide the required structured reference, the metric remains
unavailable instead of treating missing annotation as a wrong prediction.
Report factuality is therefore not a generic text-overlap score. Grounding
coordinates may be normalized or pixel-space, but pixel coordinates require
image dimensions. Length measurements convert between `mm`, `cm`, and `m`;
other unit families must already match.

The evaluation runner now preserves `InferenceResult.scores` as
`model_scores` in the per-sample result. This is the stable bridge from
contrastive/candidate-scoring models to calibration. A model that only emits a
chosen label cannot claim calibration metrics.

## Shared protocol fields

The existing `group_by` protocol field applies the selected scorer to each
declared subgroup. `minimum_group_samples` gates worst/best group comparisons,
and the report records the maximum-minus-minimum gap rather than presenting it
as a causal fairness result. Calibration additionally accepts
`calibration_bins` and `selective_thresholds`; all are serialized into the
resolved protocol and therefore enter the run fingerprint.

Task-aware JSON/JSONL records may carry scorer input under `annotations`:

```json
{
  "annotations": {
    "report": {
      "positive_findings": ["right lower lobe opacity"],
      "negative_findings": ["pleural effusion"],
      "critical_findings": ["pneumothorax"],
      "finding_vocabulary": ["right lower lobe opacity", "pleural effusion", "pneumothorax"],
      "required_sections": ["findings", "impression"]
    },
    "grounding": {
      "boxes": [[120, 80, 330, 290]],
      "image_size": [512, 512]
    },
    "measurements": [
      {"name": "lesion diameter", "value": 12, "unit": "mm", "absolute_tolerance": 1}
    ]
  }
}
```

## PathVQA acceptance slice

The first real domain-specific acceptance uses the pinned PathVQA Hugging Face
revision `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` and the existing pinned
Lingshu adapter. The exporter selects four yes/no and four free-form test
questions so both official answer families are exercised. This is an
eight-sample wiring slice, not a statistically meaningful quality result.

Prepare data and submit the A800 acceptance independently. On clusters where
compute nodes cannot resolve the login alias, set `MEDUMM_DYNAMIC_PROXY_TARGET`
to the login node's reachable private address:

```bash
bash scripts/setup_medical_cuda126_env.sh
MEDUMM_DYNAMIC_PROXY_TARGET=REACHABLE_LOGIN_HOST sbatch scripts/slurm_prepare_pathvqa_v1.0.sh
sbatch scripts/slurm_clinical_evaluation_v1.0.sh
```

The GPU job runs the entire test suite before inference and verifies pinned
dataset/model revisions, answer-type balance, CUDA/Slurm metadata, non-empty
predictions, latency, peak memory, and all pathology-specific aggregate keys.
Its machine-readable evidence is copied to
`docs/results/v1.0-clinical-evaluation.json` after acceptance.

The accepted A800 job was `437271` on `gpu01` with PyTorch 2.8.0 + CUDA 12.6.
All server tests passed before eight real inferences. Peak allocated GPU memory
was 16,020.72 MiB and mean latency was 576.01 ms. The balanced slice scored
75.0% yes/no exact accuracy, 0.0% free-form exact accuracy, and 37.5% overall.
The free-form token F1 was 7.14%. These eight samples validate wiring and expose
an open-answer weakness; they are not a model-quality estimate. Exact evidence
is committed in [`docs/results/v1.0-clinical-evaluation.json`](results/v1.0-clinical-evaluation.json).

## Remaining limits

- The report scorer needs source-provided structured clinical facts; full
  expert-labelled IU X-Ray or MIMIC-CXR acceptance remains future work.
- The grounding and measurement suites have deterministic contract tests but
  still need a pinned public dataset/model runtime slice.
- ECE is sample-size and binning sensitive. Selective prediction metrics must
  be compared only under the same protocol and candidate set.
- Subgroup gaps describe benchmark strata and do not by themselves demonstrate
  demographic fairness or clinical equity.

MedUMM remains research software and is not a medical device.
