#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDUMM_LLAVA_ENV="${MEDUMM_LLAVA_ENV:-${MEDUMM_ROOT}/.venv-llava-med}"
MEDUMM_BASE_PYTHON="${MEDUMM_BASE_PYTHON:-python}"

if [[ ! -x "${MEDUMM_LLAVA_ENV}/bin/python" ]]; then
  "${MEDUMM_BASE_PYTHON}" -m venv --system-site-packages "${MEDUMM_LLAVA_ENV}"
fi

"${MEDUMM_LLAVA_ENV}/bin/python" -m pip install \
  "transformers==4.36.2" \
  "tokenizers==0.15.2" \
  "sentencepiece==0.1.99" \
  "einops==0.6.1" \
  "accelerate==0.21.0" \
  "pytest>=8,<9"

"${MEDUMM_LLAVA_ENV}/bin/python" - <<'PY'
import torch
import transformers
import pytest

assert transformers.__version__ == "4.36.2", transformers.__version__
print(
    f"[MedUMM] LLaVA-Med environment: torch={torch.__version__} "
    f"transformers={transformers.__version__} pytest={pytest.__version__}"
)
PY
