#!/usr/bin/env bash
#SBATCH --job-name=medumm-v14-envs
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
MEDUMM_BOOTSTRAP_ENV="${MEDUMM_BOOTSTRAP_ENV:-/data/user/hd66945/MedUMM-model-envs-v1.4/bootstrap-py310}"
export MEDUMM_ENV_ROOT="${MEDUMM_ENV_ROOT:-/data/user/hd66945/MedUMM-model-envs-v1.4/models}"
export MEDUMM_CATALOG_PYTHON="${MEDUMM_CATALOG_PYTHON:-/data/user/hd66945/MedUMM/.venv-alignment/bin/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/user/hd66945/.cache/pip}"
export PYTHONUNBUFFERED=1

cd "${MEDUMM_ROOT}"
if [[ ! -x "${MEDUMM_BOOTSTRAP_ENV}/bin/python" ]]; then
  conda create --prefix "${MEDUMM_BOOTSTRAP_ENV}" --yes python=3.10 pyyaml=6.0.2
fi
export MEDUMM_PYTHON_COMMAND="${MEDUMM_BOOTSTRAP_ENV}/bin/python"

for model in plip quiltnet medvlm_r1 biomedclip; do
  bash scripts/setup_model_env.sh "${model}"
done

echo "[MedUMM] v1.4 isolated model environments prepared"
