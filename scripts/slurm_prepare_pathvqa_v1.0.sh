#!/usr/bin/env bash
#SBATCH --job-name=medumm-v10-data
#SBATCH --partition=Intel-8358
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-${MEDUMM_ROOT}/.venv-alignment}"
MEDUMM_V10_DATA="${MEDUMM_V10_DATA:-${MEDUMM_ROOT}/data/v1.0}"
MEDUMM_DYNAMIC_PROXY_TARGET="${MEDUMM_DYNAMIC_PROXY_TARGET:-}"
proxy_pid=""

cleanup() {
  if [[ -n "${proxy_pid}" ]]; then
    kill "${proxy_pid}" 2>/dev/null || true
    wait "${proxy_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ -n "${MEDUMM_DYNAMIC_PROXY_TARGET}" ]]; then
  proxy_port=$((22000 + SLURM_JOB_ID % 4000))
  ssh -N -o BatchMode=yes -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -D "127.0.0.1:${proxy_port}" "${MEDUMM_DYNAMIC_PROXY_TARGET}" &
  proxy_pid=$!
  sleep 3
  kill -0 "${proxy_pid}"
  export ALL_PROXY="socks5h://127.0.0.1:${proxy_port}"
  export all_proxy="${ALL_PROXY}"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MEDUMM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MEDUMM_ROOT}"
test -x "${MEDUMM_RUNTIME_ENV}/bin/python"
"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/prepare_pathvqa_v1_0.py \
  --output-directory "${MEDUMM_V10_DATA}/pathvqa" \
  --closed-samples "${MEDUMM_PATHVQA_CLOSED_SAMPLES:-4}" \
  --open-samples "${MEDUMM_PATHVQA_OPEN_SAMPLES:-4}"

echo "[MedUMM] v1.0 pinned PathVQA data are ready"
