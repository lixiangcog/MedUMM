#!/usr/bin/env bash
#SBATCH --job-name=medumm-v15-models
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_ENV_ROOT="${MEDUMM_ENV_ROOT:-/data/user/hd66945/MedUMM-model-envs-v1.5/models}"
MEDUMM_V15_ASSETS="${MEDUMM_V15_ASSETS:-/data/user/hd66945/MedUMM-assets/v1.5}"
export MEDMO_4B_MODEL_PATH="${MEDUMM_V15_ASSETS}/medmo-4b"
export MEDMO_8B_MODEL_PATH="${MEDUMM_V15_ASSETS}/medmo-8b"
export LINGSHU_I_8B_MODEL_PATH="${MEDUMM_V15_ASSETS}/lingshu-i-8b"
export FLEMING_VL_8B_MODEL_PATH="${MEDUMM_V15_ASSETS}/fleming-vl-8b"
export MEDUMM_ADAPTER_SMOKE_IMAGE="${MEDUMM_ADAPTER_SMOKE_IMAGE:-${MEDUMM_ROOT}/data/v1.0/pathvqa/images/smoke.png}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -f "${MEDUMM_V15_ASSETS}/provenance.json"
test -f "${MEDUMM_ADAPTER_SMOKE_IMAGE}"
for model in medmo_4b medmo_8b lingshu_i_8b fleming_vl_8b; do
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
"${MEDUMM_ENV_ROOT}/medmo_4b/bin/python" -m medumm infer --config configs/inference/medmo_4b_v1.5.yaml
"${MEDUMM_ENV_ROOT}/medmo_8b/bin/python" -m medumm infer --config configs/inference/medmo_8b_v1.5.yaml
"${MEDUMM_ENV_ROOT}/lingshu_i_8b/bin/python" -m medumm infer --config configs/inference/lingshu_i_8b_v1.5.yaml
"${MEDUMM_ENV_ROOT}/fleming_vl_8b/bin/python" -m medumm infer --config configs/inference/fleming_vl_8b_v1.5.yaml
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_real_model_adapters_v1_5.py \
  --medmo-4b outputs/inference/medmo_4b_v1.5.json \
  --medmo-8b outputs/inference/medmo_8b_v1.5.json \
  --lingshu-i-8b outputs/inference/lingshu_i_8b_v1.5.json \
  --fleming-vl-8b outputs/inference/fleming_vl_8b_v1.5.json \
  --provenance "${MEDUMM_V15_ASSETS}/provenance.json" \
  --environment-root "${MEDUMM_ENV_ROOT}" \
  --output outputs/verification/real_model_adapters_v1.5.json

echo "[MedUMM] v1.5 real model adapter acceptance completed"
