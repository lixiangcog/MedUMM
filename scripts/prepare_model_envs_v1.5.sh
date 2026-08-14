#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MEDUMM_BUILD_PROXY:-}" ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  export ALL_PROXY="${MEDUMM_BUILD_PROXY}"
  export all_proxy="${MEDUMM_BUILD_PROXY}"
fi

MEDUMM_ROOT="${MEDUMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDUMM_BOOTSTRAP_ENV="${MEDUMM_BOOTSTRAP_ENV:-/data/user/hd66945/MedUMM-model-envs-v1.5/bootstrap-py310}"
export MEDUMM_ENV_ROOT="${MEDUMM_ENV_ROOT:-/data/user/hd66945/MedUMM-model-envs-v1.5/models}"
export MEDUMM_CATALOG_PYTHON="${MEDUMM_CATALOG_PYTHON:-${MEDUMM_ROOT}/.venv-alignment/bin/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/user/hd66945/.cache/pip}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-20}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-20}"
export MEDUMM_SKIP_PIP_UPGRADE=1
export MEDUMM_PIP_WHEEL="${MEDUMM_PIP_WHEEL:-/data/user/hd66945/MedUMM-model-envs-v1.5/wheelhouse/pip-25.1.1-py3-none-any.whl}"
export MEDUMM_SOCKS_WHEEL="${MEDUMM_SOCKS_WHEEL:-/data/user/hd66945/MedUMM-model-envs-v1.5/wheelhouse/PySocks-1.7.1-py3-none-any.whl}"
export PYTHONUNBUFFERED=1

cd "${MEDUMM_ROOT}"
if [[ ! -x "${MEDUMM_BOOTSTRAP_ENV}/bin/python" ]]; then
  conda create --prefix "${MEDUMM_BOOTSTRAP_ENV}" --yes python=3.10 pyyaml=6.0.2
fi
export MEDUMM_PYTHON_COMMAND="${MEDUMM_BOOTSTRAP_ENV}/bin/python"

for model in medmo_4b medmo_8b lingshu_i_8b fleming_vl_8b; do
  bash scripts/setup_model_env.sh "${model}"
done

echo "[MedUMM] v1.5 isolated model environments prepared"
