#!/usr/bin/env bash
#SBATCH --job-name=medumm-v10-eval
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_V09_ASSETS="${MEDUMM_V09_ASSETS:-${MEDUMM_ROOT}-assets/v0.9}"
MEDUMM_V10_DATA="${MEDUMM_V10_DATA:-${MEDUMM_ROOT}/data/v1.0}"

export LINGSHU_MODEL_PATH="${MEDUMM_V09_ASSETS}/lingshu-7b"
export MEDUMM_PATHVQA_MANIFEST="${MEDUMM_V10_DATA}/pathvqa/samples.jsonl"
export MEDUMM_PATHVQA_IMAGE_ROOT="${MEDUMM_V10_DATA}/pathvqa/images"
export MEDUMM_PATHVQA_PROVENANCE="${MEDUMM_V10_DATA}/pathvqa/provenance.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_RUNTIME_ENV}/bin/python"
test -f "${LINGSHU_MODEL_PATH}/model.safetensors.index.json"
test -f "${MEDUMM_PATHVQA_MANIFEST}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The allocated Slurm GPU is unavailable to PyTorch.")
torch.zeros(1, device="cuda")
print(f"[MedUMM] CUDA preflight passed: {torch.cuda.get_device_name(0)}")
PY
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pytest -q tests
"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/pathvqa_lingshu_v1.0.yaml
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_clinical_evaluation_v1_0.py \
  --predictions outputs/evaluation/pathvqa_lingshu_v1.0/predictions.jsonl \
  --score outputs/evaluation/pathvqa_lingshu_v1.0/score.json \
  --audit outputs/evaluation/pathvqa_lingshu_v1.0/dataset_audit.json \
  --provenance "${MEDUMM_PATHVQA_PROVENANCE}" \
  --output outputs/verification/clinical_evaluation_v1.0.json

echo "[MedUMM] v1.0 clinical evaluation acceptance completed"
