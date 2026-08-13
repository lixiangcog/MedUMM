#!/usr/bin/env bash
#SBATCH --job-name=medumm-v12-infer
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/v1.2-inference-%j.out
#SBATCH --error=outputs/slurm/v1.2-inference-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_INFERENCE_BACKEND="${MEDUMM_INFERENCE_BACKEND:-vllm}"
MEDUMM_SERVER_MODEL_PATH="${MEDUMM_SERVER_MODEL_PATH:-/data/user/hd66945/models/Qwen2.5-VL-3B-Instruct}"
MEDUMM_SERVER_MODEL_REVISION="${MEDUMM_SERVER_MODEL_REVISION:-66285546d2b821cf421d4f5eb2576359d3770cd3}"
if [[ "${MEDUMM_INFERENCE_BACKEND}" == "vllm" ]]; then
  MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/envs/medumm-vllm-011}"
  SERVER_CONFIG="configs/inference/serve_vllm.yaml"
  BENCHMARK_CONFIG="configs/inference/benchmark_openai_vllm_v1.2.yaml"
  SERVER_PORT=8000
elif [[ "${MEDUMM_INFERENCE_BACKEND}" == "sglang" ]]; then
  MEDUMM_RUNTIME_ENV="${MEDUMM_RUNTIME_ENV:-/data/user/hd66945/envs/medumm-sglang-054}"
  SERVER_CONFIG="configs/inference/serve_sglang.yaml"
  BENCHMARK_CONFIG="configs/inference/benchmark_openai_sglang_v1.2.yaml"
  SERVER_PORT=30000
else
  echo "MEDUMM_INFERENCE_BACKEND must be vllm or sglang" >&2
  exit 2
fi

export MEDUMM_SERVER_MODEL_PATH MEDUMM_SERVER_MODEL_REVISION
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
RUN_ROOT="${MEDUMM_ROOT}/outputs/inference-optimization/v1.2-${MEDUMM_INFERENCE_BACKEND}-${SLURM_JOB_ID:-local}"
mkdir -p "${RUN_ROOT}" "${MEDUMM_ROOT}/outputs/slurm"
cd "${MEDUMM_ROOT}"

test -x "${MEDUMM_RUNTIME_ENV}/bin/python"
test -f "${MEDUMM_SERVER_MODEL_PATH}/config.json"
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pip install --no-deps --no-build-isolation \
  -e "${MEDUMM_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
"${MEDUMM_RUNTIME_ENV}/bin/python" -m pytest -q tests/test_inference_optimization.py
"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm backends --json > "${RUN_ROOT}/backend-catalog.json"

setsid "${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm serve \
  --config "${SERVER_CONFIG}" --set server.execution=launch \
  --set server.output_directory="${RUN_ROOT}/server" \
  > "${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 120 "${RUN_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  echo "Inference server did not become healthy" >&2
  exit 1
fi

"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm benchmark-inference \
  --config "${BENCHMARK_CONFIG}" \
  --set benchmark.batch_size=1 \
  --set benchmark.output_directory="${RUN_ROOT}/sequential" \
  > "${RUN_ROOT}/sequential-cli.json"
"${MEDUMM_RUNTIME_ENV}/bin/python" -m medumm benchmark-inference \
  --config "${BENCHMARK_CONFIG}" \
  --set benchmark.batch_size=8 \
  --set benchmark.output_directory="${RUN_ROOT}/concurrent" \
  > "${RUN_ROOT}/concurrent-cli.json"

"${MEDUMM_RUNTIME_ENV}/bin/python" scripts/verify_inference_optimization_v1_2.py \
  --backend "${MEDUMM_INFERENCE_BACKEND}" \
  --sequential "${RUN_ROOT}/sequential/benchmark.json" \
  --concurrent "${RUN_ROOT}/concurrent/benchmark.json" \
  --server-plan "${RUN_ROOT}/server/server_plan.json" \
  --output "${RUN_ROOT}/verification.json"

echo "[MedUMM] v1.2 ${MEDUMM_INFERENCE_BACKEND} acceptance passed"
