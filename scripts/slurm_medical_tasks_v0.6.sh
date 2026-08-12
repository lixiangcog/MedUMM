#!/usr/bin/env bash
#SBATCH --job-name=medumm-v06-tasks
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
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-${MEDUMM_ROOT}-assets/llava-med-v1.5}"
MEDUMM_LLAVA_ENV="${MEDUMM_LLAVA_ENV:-${MEDUMM_ROOT}/.venv-llava-med}"
MEDUMM_VQA_PARQUET="${MEDUMM_VQA_PARQUET:-${MEDUMM_ASSET_ROOT}/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet}"

export LLAVA_MED_MODEL_PATH="${MEDUMM_ASSET_ROOT}/llava-med-v1.5-mistral-7b"
export LLAVA_MED_SOURCE_PATH="${MEDUMM_ASSET_ROOT}/LLaVA-Med"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_LLAVA_ENV}/bin/python"
test -f "${LLAVA_MED_MODEL_PATH}/model.safetensors.index.json"
test -f "${LLAVA_MED_SOURCE_PATH}/llava/__init__.py"
test -f "${MEDUMM_VQA_PARQUET}"

"${MEDUMM_LLAVA_ENV}/bin/python" scripts/prepare_vqa_rad_tasks.py \
  --dataset flaviagiammarino/vqa-rad \
  --revision bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9 \
  --split test \
  --parquet-path "${MEDUMM_VQA_PARQUET}" \
  --output-directory data/vqa_rad_tasks_v0.6 \
  --samples-per-task 4

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_LLAVA_ENV}/bin/python" -m pytest -q tests
"${MEDUMM_LLAVA_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/vqa_rad_medical_tasks_v0.6.yaml
"${MEDUMM_LLAVA_ENV}/bin/python" scripts/verify_medical_tasks_v0_6.py \
  --predictions outputs/evaluation/vqa_rad_medical_tasks_v0.6/predictions.jsonl \
  --score outputs/evaluation/vqa_rad_medical_tasks_v0.6/score.json \
  --audit outputs/evaluation/vqa_rad_medical_tasks_v0.6/dataset_audit.json \
  --asset-provenance "${MEDUMM_ASSET_ROOT}/provenance.json" \
  --dataset-provenance data/vqa_rad_tasks_v0.6/provenance.json \
  --output outputs/verification/medical_tasks_v0.6.json

echo "[MedUMM] v0.6 task-aware medical evaluation completed"
