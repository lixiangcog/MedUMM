#!/usr/bin/env bash
#SBATCH --job-name=medumm-v14-models
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/MedUMM/.venv-alignment}"
MEDUMM_ENV_ROOT="${MEDUMM_ENV_ROOT:-/data/user/hd66945/MedUMM-model-envs-v1.4/models}"
MEDUMM_V14_ASSETS="${MEDUMM_V14_ASSETS:-/data/user/hd66945/MedUMM-assets/v1.4}"
export PLIP_MODEL_PATH="${MEDUMM_V14_ASSETS}/plip"
export QUILTNET_MODEL_PATH="${MEDUMM_V14_ASSETS}/quiltnet"
export MEDVLM_R1_MODEL_PATH="${MEDUMM_V14_ASSETS}/medvlm-r1"
export BIOMEDCLIP_MODEL_PATH="${MEDUMM_V14_ASSETS}/biomedclip"
export BIOMEDCLIP_TEXT_MODEL_PATH="${MEDUMM_V14_ASSETS}/biomedclip-text-model"
export MEDUMM_ADAPTER_SMOKE_IMAGE="${MEDUMM_ADAPTER_SMOKE_IMAGE:-${MEDUMM_ROOT}/data/v1.0/pathvqa/images/smoke.png}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -f "${MEDUMM_V14_ASSETS}/provenance.json"
test -f "${MEDUMM_ADAPTER_SMOKE_IMAGE}"
for model in plip quiltnet medvlm_r1 biomedclip; do
  test -x "${MEDUMM_ENV_ROOT}/${model}/bin/python"
done

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The allocated Slurm GPU is unavailable to PyTorch.")
torch.zeros(1, device="cuda")
print(f"[MedUMM] CUDA preflight passed: {torch.cuda.get_device_name(0)}")
PY
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pytest -q \
  tests/test_model_adapter_recipes.py tests/test_resource_catalog.py tests/test_model_environments.py
"${MEDUMM_ENV_ROOT}/plip/bin/python" -m medumm infer --config configs/inference/plip_v1.4.yaml
"${MEDUMM_ENV_ROOT}/quiltnet/bin/python" -m medumm infer --config configs/inference/quiltnet_v1.4.yaml
"${MEDUMM_ENV_ROOT}/medvlm_r1/bin/python" -m medumm infer --config configs/inference/medvlm_r1_v1.4.yaml
"${MEDUMM_ENV_ROOT}/biomedclip/bin/python" -m medumm infer --config configs/inference/biomedclip_v1.4.yaml
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_real_model_adapters_v1_4.py \
  --plip outputs/inference/plip_v1.4.json \
  --quiltnet outputs/inference/quiltnet_v1.4.json \
  --medvlm-r1 outputs/inference/medvlm_r1_v1.4.json \
  --biomedclip outputs/inference/biomedclip_v1.4.json \
  --provenance "${MEDUMM_V14_ASSETS}/provenance.json" \
  --environment-root "${MEDUMM_ENV_ROOT}" \
  --output outputs/verification/real_model_adapters_v1.4.json

echo "[MedUMM] v1.4 real model adapter acceptance completed"
