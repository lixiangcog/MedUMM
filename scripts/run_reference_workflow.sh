#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDUMM_PYTHON="${MEDUMM_PYTHON:-python}"
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${MEDUMM_ROOT}"

"${MEDUMM_PYTHON}" -m medumm post-train \
  --config configs/post_training/medical_sft_smoke.yaml
"${MEDUMM_PYTHON}" -m medumm infer \
  --config configs/inference/medical_reference_workflow.yaml
"${MEDUMM_PYTHON}" -m medumm infer \
  --config configs/inference/medical_linear_understanding.yaml
"${MEDUMM_PYTHON}" -m medumm evaluate \
  --config configs/evaluation/medical_vqa_linear_smoke.yaml

echo "[MedUMM] workflow completed"
