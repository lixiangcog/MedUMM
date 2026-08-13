#!/usr/bin/env bash
#SBATCH --job-name=medumm-v09-runtime
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_V09_ASSETS="${MEDUMM_V09_ASSETS:-${MEDUMM_ROOT}-assets/v0.9}"
MEDUMM_V09_DATA="${MEDUMM_V09_DATA:-${MEDUMM_ROOT}/data/v0.9}"

export LINGSHU_MODEL_PATH="${MEDUMM_V09_ASSETS}/lingshu-7b"
export PUBMEDCLIP_MODEL_PATH="${MEDUMM_V09_ASSETS}/pubmedclip"
export MEDUMM_SLAKE_MANIFEST="${MEDUMM_V09_DATA}/slake/samples.jsonl"
export MEDUMM_SLAKE_IMAGE_ROOT="${MEDUMM_V09_DATA}/slake/images"
export MEDUMM_SLAKE_PROVENANCE="${MEDUMM_V09_DATA}/slake/provenance.json"
export MEDUMM_PNEUMONIAMNIST_MANIFEST="${MEDUMM_V09_DATA}/pneumoniamnist/samples.jsonl"
export MEDUMM_PNEUMONIAMNIST_IMAGE_ROOT="${MEDUMM_V09_DATA}/pneumoniamnist/images"
export MEDUMM_PNEUMONIAMNIST_PROVENANCE="${MEDUMM_V09_DATA}/pneumoniamnist/provenance.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_RUNTIME_ENV}/bin/python"
test -f "${LINGSHU_MODEL_PATH}/model.safetensors.index.json"
test -f "${PUBMEDCLIP_MODEL_PATH}/pytorch_model.bin"
test -f "${MEDUMM_SLAKE_MANIFEST}"
test -f "${MEDUMM_PNEUMONIAMNIST_MANIFEST}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_RUNTIME_ENV}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The allocated Slurm GPU is unavailable to PyTorch.")
torch.zeros(1, device="cuda")
print(f"[MedUMM] CUDA preflight passed: {torch.cuda.get_device_name(0)}")
PY
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pytest -q tests
"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/slake_lingshu_v0.9.yaml
"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm evaluate \
  --config configs/evaluation/pneumoniamnist_pubmedclip_v0.9.yaml
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_runtime_slices_v0_9.py \
  --lingshu-predictions outputs/evaluation/slake_lingshu_v0.9/predictions.jsonl \
  --lingshu-score outputs/evaluation/slake_lingshu_v0.9/score.json \
  --lingshu-audit outputs/evaluation/slake_lingshu_v0.9/dataset_audit.json \
  --slake-provenance "${MEDUMM_SLAKE_PROVENANCE}" \
  --pubmedclip-predictions outputs/evaluation/pneumoniamnist_pubmedclip_v0.9/predictions.jsonl \
  --pubmedclip-score outputs/evaluation/pneumoniamnist_pubmedclip_v0.9/score.json \
  --pubmedclip-audit outputs/evaluation/pneumoniamnist_pubmedclip_v0.9/dataset_audit.json \
  --pneumoniamnist-provenance "${MEDUMM_PNEUMONIAMNIST_PROVENANCE}" \
  --output outputs/verification/runtime_slices_v0.9.json

echo "[MedUMM] v0.9 runtime slices acceptance completed"
