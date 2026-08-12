#!/usr/bin/env bash
#SBATCH --job-name=medumm-v04-eval
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-${MEDUMM_ROOT}/vendor/llava-med-v1.5}"
MEDUMM_LLAVA_ENV="${MEDUMM_LLAVA_ENV:-${MEDUMM_ROOT}/.venv-llava-med}"

export LLAVA_MED_MODEL_PATH="${MEDUMM_ASSET_ROOT}/llava-med-v1.5-mistral-7b"
export LLAVA_MED_SOURCE_PATH="${MEDUMM_ASSET_ROOT}/LLaVA-Med"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_LLAVA_ENV}/bin/python"
test -f "${LLAVA_MED_MODEL_PATH}/model.safetensors.index.json"
test -f "${LLAVA_MED_SOURCE_PATH}/llava/__init__.py"
test -f data/vqa_rad_eval/samples.jsonl

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_LLAVA_ENV}/bin/python" -m pytest -q
"${MEDUMM_LLAVA_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/vqa_rad_llava_med_v0.4.yaml
"${MEDUMM_LLAVA_ENV}/bin/python" scripts/verify_evaluation_base.py \
  --predictions outputs/evaluation/vqa_rad_llava_med_v0.4/predictions.jsonl \
  --score outputs/evaluation/vqa_rad_llava_med_v0.4/score.json \
  --audit outputs/evaluation/vqa_rad_llava_med_v0.4/dataset_audit.json \
  --asset-provenance "${MEDUMM_ASSET_ROOT}/provenance.json" \
  --dataset-provenance data/vqa_rad_eval/provenance.json \
  --output outputs/verification/medical_evaluation_v0.4.json

echo "[MedUMM] v0.4 medical evaluation base completed"
