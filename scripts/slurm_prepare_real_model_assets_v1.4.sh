#!/usr/bin/env bash
#SBATCH --job-name=medumm-v14-assets
#SBATCH --partition=Intel-8358-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/MedUMM/.venv-alignment}"
MEDUMM_V14_ASSETS="${MEDUMM_V14_ASSETS:-/data/user/hd66945/MedUMM-assets/v1.4}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/user/hd66945/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_RUNTIME_ENV}/bin/python"
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/prepare_real_model_adapters_v1_4.py \
  --asset-root "${MEDUMM_V14_ASSETS}" \
  --models plip quiltnet medvlm_r1 biomedclip

echo "[MedUMM] v1.4 immutable model assets prepared"
