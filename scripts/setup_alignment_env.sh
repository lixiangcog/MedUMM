#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDUMM_ALIGNMENT_ENV="${MEDUMM_ALIGNMENT_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_BASE_PYTHON="${MEDUMM_BASE_PYTHON:-python}"

if [[ ! -x "${MEDUMM_ALIGNMENT_ENV}/bin/python" ]]; then
  "${MEDUMM_BASE_PYTHON}" -m venv --system-site-packages "${MEDUMM_ALIGNMENT_ENV}"
fi

"${MEDUMM_ALIGNMENT_ENV}/bin/python" -m pip install \
  "transformers>=4.50,<5" \
  "peft>=0.14,<1" \
  "accelerate>=1.0,<2" \
  "pytest>=8,<9"

"${MEDUMM_ALIGNMENT_ENV}/bin/python" - <<'PY'
import accelerate
import peft
import torch
import transformers

print(
    f"[MedUMM] alignment environment: torch={torch.__version__} "
    f"transformers={transformers.__version__} peft={peft.__version__} "
    f"accelerate={accelerate.__version__}"
)
PY
