#!/usr/bin/env bash
#SBATCH --job-name=medumm-v08-catalog
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ROOT="${MEDUMM_RUNTIME_ROOT:-${MEDUMM_ROOT}}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-${MEDUMM_RUNTIME_ROOT}-assets/llava-med-v1.5}"
MEDUMM_LLAVA_ENV="${MEDUMM_LLAVA_ENV:-${MEDUMM_RUNTIME_ROOT}/.venv-llava-med}"
MEDUMM_VQA_ROOT="${MEDUMM_VQA_ROOT:-${MEDUMM_RUNTIME_ROOT}/data/vqa_rad_smoke}"

export LLAVA_MED_MODEL_PATH="${MEDUMM_ASSET_ROOT}/llava-med-v1.5-mistral-7b"
export LLAVA_MED_SOURCE_PATH="${MEDUMM_ASSET_ROOT}/LLaVA-Med"
export MEDUMM_VQA_MANIFEST="${MEDUMM_VQA_ROOT}/samples.jsonl"
export MEDUMM_VQA_IMAGE_ROOT="${MEDUMM_VQA_ROOT}/images"
export MEDUMM_VQA_PROVENANCE="${MEDUMM_VQA_ROOT}/provenance.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_LLAVA_ENV}/bin/python"
test -f "${LLAVA_MED_MODEL_PATH}/model.safetensors.index.json"
test -f "${LLAVA_MED_SOURCE_PATH}/llava/__init__.py"
test -f "${MEDUMM_VQA_MANIFEST}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_LLAVA_ENV}/bin/python" -m pytest -q tests
"${MEDUMM_LLAVA_ENV}/bin/python" -m medumm resources validate \
  --output outputs/verification/resource_catalog_v0.8.json
"${MEDUMM_LLAVA_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/vqa_rad_catalog_v0.8.yaml
"${MEDUMM_LLAVA_ENV}/bin/python" scripts/verify_scale_catalog_v0_8.py \
  --catalog-validation outputs/verification/resource_catalog_v0.8.json \
  --predictions outputs/evaluation/vqa_rad_catalog_v0.8/predictions.jsonl \
  --results outputs/evaluation/vqa_rad_catalog_v0.8/results.jsonl \
  --score outputs/evaluation/vqa_rad_catalog_v0.8/score.json \
  --audit outputs/evaluation/vqa_rad_catalog_v0.8/dataset_audit.json \
  --output outputs/verification/scale_catalog_v0.8.json

echo "[MedUMM] v0.8 scale catalog acceptance completed"
