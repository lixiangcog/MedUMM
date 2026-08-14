#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MEDUMM_BUILD_PROXY:-}" ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  export ALL_PROXY="${MEDUMM_BUILD_PROXY}"
  export all_proxy="${MEDUMM_BUILD_PROXY}"
fi

MEDUMM_ROOT="${MEDUMM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-/data/user/hd66945/MedUMM-assets/v1.5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

cd "${MEDUMM_ROOT}"
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/prepare_real_model_adapters_v1_5.py \
  --asset-root "${MEDUMM_ASSET_ROOT}"
