#!/usr/bin/env bash
#SBATCH --job-name=medumm-v16-bench
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_RUNTIME_ENV}/bin/python"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The allocated Slurm GPU is unavailable to PyTorch.")
torch.zeros(1, device="cuda")
print(f"[MedUMM] CUDA preflight passed: {torch.cuda.get_device_name(0)}")
PY

"${MEDUMM_RUNTIME_ENV}/bin/python" -m pytest -q \
  tests/test_specialized_benchmarks_v1_6.py \
  tests/test_clinical_metrics_v1.py \
  tests/test_resource_catalog.py

for config in configs/evaluation/benchmarks_v1.6/*.yaml; do
  "${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm evaluate --config "${config}"
done

"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_specialized_benchmarks_v1_6.py \
  --output-root outputs/evaluation/benchmarks_v1.6 \
  --output outputs/verification/specialized_benchmarks_v1.6.json \
  --require-cuda

echo "[MedUMM] v1.6 specialized benchmark acceptance completed"
