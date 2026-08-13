#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-medical-cu126}"
MEDUMM_BASE_PYTHON="${MEDUMM_BASE_PYTHON:-python3}"

if [[ ! -x "${MEDUMM_RUNTIME_ENV}/bin/python" ]]; then
  "${MEDUMM_BASE_PYTHON}" -m venv "${MEDUMM_RUNTIME_ENV}"
fi

"${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install \
  "torch==2.8.0" "torchvision==0.23.0" \
  --index-url https://download.pytorch.org/whl/cu126
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install \
  -e "${MEDUMM_ROOT}[medical,data,test]"

"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
import torch
import torchvision
import transformers

if torch.version.cuda != "12.6":
    raise RuntimeError(f"Expected a CUDA 12.6 PyTorch build, found {torch.version.cuda}.")
print(
    f"[MedUMM] medical CUDA environment: torch={torch.__version__} "
    f"torchvision={torchvision.__version__} transformers={transformers.__version__}"
)
PY
