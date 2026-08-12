#!/usr/bin/env bash
#SBATCH --job-name=medumm-v04-assets
#SBATCH --partition=Intel-8358
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-${MEDUMM_ROOT}/vendor/llava-med-v1.5}"
MEDUMM_CONDA_ENV="${MEDUMM_CONDA_ENV:-med-tiv-eye}"
MEDUMM_CONDA_INIT="${MEDUMM_CONDA_INIT:-/data/anaconda3/etc/profile.d/conda.sh}"
MEDUMM_PROXY_SSH_TARGET="${MEDUMM_PROXY_SSH_TARGET:-}"
MEDUMM_DYNAMIC_PROXY_TARGET="${MEDUMM_DYNAMIC_PROXY_TARGET:-}"
proxy_pid=""

cleanup() {
  if [[ -n "${proxy_pid}" ]]; then
    kill "${proxy_pid}" 2>/dev/null || true
    wait "${proxy_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -n "${MEDUMM_PROXY_SSH_TARGET}" ]]; then
  proxy_port=$((21000 + SLURM_JOB_ID % 5000))
  ssh -N -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -L "127.0.0.1:${proxy_port}:127.0.0.1:7890" \
    "${MEDUMM_PROXY_SSH_TARGET}" &
  proxy_pid=$!
  sleep 3
  kill -0 "${proxy_pid}"
  export HTTP_PROXY="http://127.0.0.1:${proxy_port}"
  export HTTPS_PROXY="${HTTP_PROXY}"
  export ALL_PROXY="socks5://127.0.0.1:${proxy_port}"
  export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" all_proxy="${ALL_PROXY}"
  export NO_PROXY="127.0.0.1,localhost,::1" no_proxy="${NO_PROXY}"
elif [[ -n "${MEDUMM_DYNAMIC_PROXY_TARGET}" ]]; then
  proxy_port=$((21000 + SLURM_JOB_ID % 5000))
  ssh -N -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -D "127.0.0.1:${proxy_port}" \
    "${MEDUMM_DYNAMIC_PROXY_TARGET}" &
  proxy_pid=$!
  sleep 3
  kill -0 "${proxy_pid}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export MEDUMM_DOWNLOAD_PROXY="socks5h://127.0.0.1:${proxy_port}"
else
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

module load anaconda3
source "${MEDUMM_CONDA_INIT}"
conda activate "${MEDUMM_CONDA_ENV}"
cd "${MEDUMM_ROOT}"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

if [[ -x "${MEDUMM_ROOT}/.venv-llava-med/bin/python" ]] && \
  "${MEDUMM_ROOT}/.venv-llava-med/bin/python" - <<'PY'
import pytest
import sentencepiece
import torch
import transformers

assert transformers.__version__ == "4.36.2"
print(
    f"[MedUMM] reusing LLaVA-Med environment: torch={torch.__version__} "
    f"transformers={transformers.__version__} pytest={pytest.__version__}"
)
PY
then
  :
else
  bash scripts/setup_llava_med_env.sh
fi
"${MEDUMM_ROOT}/.venv-llava-med/bin/python" scripts/prepare_llava_med_assets.py \
  --destination "${MEDUMM_ASSET_ROOT}"
"${MEDUMM_ROOT}/.venv-llava-med/bin/python" scripts/prepare_vqa_rad.py \
  --dataset flaviagiammarino/vqa-rad \
  --revision bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9 \
  --split test \
  --parquet-path "${MEDUMM_ASSET_ROOT}/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet" \
  --output-directory data/vqa_rad_smoke \
  --max-samples 4 \
  --closed-only
"${MEDUMM_ROOT}/.venv-llava-med/bin/python" scripts/prepare_vqa_rad.py \
  --dataset flaviagiammarino/vqa-rad \
  --revision bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9 \
  --split test \
  --parquet-path "${MEDUMM_ASSET_ROOT}/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet" \
  --output-directory data/vqa_rad_eval \
  --max-samples "${MEDUMM_VQA_RAD_SAMPLES:-32}" \
  --closed-only

"${MEDUMM_ROOT}/.venv-llava-med/bin/python" - <<PY
import json
from pathlib import Path

asset = json.loads(Path("${MEDUMM_ASSET_ROOT}/provenance.json").read_text())
smoke = json.loads(Path("data/vqa_rad_smoke/provenance.json").read_text())
dataset = json.loads(Path("data/vqa_rad_eval/provenance.json").read_text())
assert Path(asset["model_path"], "model.safetensors.index.json").is_file()
assert Path(asset["vision_model_path"], "config.json").is_file()
assert Path(asset["source_path"], "llava", "__init__.py").is_file()
assert smoke["sample_count"] == 4
assert dataset["sample_count"] == int("${MEDUMM_VQA_RAD_SAMPLES:-32}")
print("[MedUMM] pinned LLaVA-Med and VQA-RAD assets are ready")
PY
