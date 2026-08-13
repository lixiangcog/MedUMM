#!/usr/bin/env bash

set -euo pipefail

MEDUMM_INFERENCE_BACKEND="${MEDUMM_INFERENCE_BACKEND:-vllm}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-/data/user/hd66945/MedUMM-assets/v1.2}"
EMU3_5_SOURCE_REVISION="${EMU3_5_SOURCE_REVISION:-8fcf1da48f9c5252fe0a0b8dc842e07b6efcd745}"

if [[ "${MEDUMM_INFERENCE_BACKEND}" == "vllm" ]]; then
  MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/envs/medumm-vllm-011}"
  python3 -m venv "${MEDUMM_RUNTIME_ENV}"
  "${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install "vllm==0.11.0" \
    "transformers==4.56.1" imageio==2.37.0 imageio-ffmpeg==0.6.0 \
    omegaconf==2.3.0
  CUDA_HOME="${CUDA_HOME:-/data/apps/cuda/12.9}" \
    MAX_JOBS="${MAX_JOBS:-16}" \
    "${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install \
    --no-build-isolation "flash-attn==2.8.3"
  if [[ ! -d "${MEDUMM_ASSET_ROOT}/Emu3.5/.git" ]]; then
    git clone https://github.com/baaivision/Emu3.5.git \
      "${MEDUMM_ASSET_ROOT}/Emu3.5"
  fi
  git -C "${MEDUMM_ASSET_ROOT}/Emu3.5" checkout --detach \
    "${EMU3_5_SOURCE_REVISION}"
  (
    cd "${MEDUMM_ASSET_ROOT}/Emu3.5"
    "${MEDUMM_RUNTIME_ENV}/bin/python" src/patch/apply.py \
      --patch-dir third_party/vllm
  )
elif [[ "${MEDUMM_INFERENCE_BACKEND}" == "sglang" ]]; then
  MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/envs/medumm-sglang-054}"
  python3 -m venv "${MEDUMM_RUNTIME_ENV}"
  "${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install "sglang[srt]==0.5.4.post3"
else
  echo "MEDUMM_INFERENCE_BACKEND must be vllm or sglang" >&2
  exit 2
fi

"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
import json

values = {
    "vllm": version("vllm") if find_spec("vllm") else None,
    "sglang": version("sglang") if find_spec("sglang") else None,
    "emu3_5_cfg_scheduler": bool(
        find_spec("vllm.v1.core.sched.batch_scheduler")
    ) if find_spec("vllm") else False,
}
print(json.dumps(values, indent=2))
PY
