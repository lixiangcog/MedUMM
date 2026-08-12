#!/usr/bin/env bash
#SBATCH --job-name=medumm-v07-dpo
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/medumm-v07-dpo-%j.out
#SBATCH --error=outputs/logs/medumm-v07-dpo-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_PYTHON="${MEDUMM_PYTHON:-${MEDUMM_ROOT}/.venv-alignment/bin/python}"
MEDUMM_ASSET_ROOT="${MEDUMM_ASSET_ROOT:-${MEDUMM_ROOT}/assets}"
RAW_PREFERENCES="${RAW_PREFERENCES:-${MEDUMM_ASSET_ROOT}/ultramedical-preference/test-761eb793.json}"
MODEL_ASSETS="${MODEL_ASSETS:-${MEDUMM_ASSET_ROOT}/pythia-14m-v0.7}"
RUN_DIR="${MEDUMM_ROOT}/outputs/post_training/ultramedical_dpo_v0.7"

cd "${MEDUMM_ROOT}"
mkdir -p outputs/logs
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MEDUMM_SOURCE_COMMIT="${MEDUMM_SOURCE_COMMIT:-$(git rev-parse HEAD 2>/dev/null || true)}"
export MEDUMM_SOURCE_COMMIT MODEL_ASSETS RUN_DIR

"${MEDUMM_PYTHON}" -m pytest -q
"${MEDUMM_PYTHON}" scripts/prepare_ultramedical_preferences.py \
  --raw-path "${RAW_PREFERENCES}" \
  --output-directory data/ultramedical_preferences_v0.7 \
  --samples 8
"${MEDUMM_PYTHON}" -m medumm post-train \
  --config configs/post_training/ultramedical_dpo_v0.7.yaml \
  --set post_training.model.name_or_path="${MODEL_ASSETS}/model"
"${MEDUMM_PYTHON}" - <<'PY'
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM

base = os.path.join(os.environ["MODEL_ASSETS"], "model")
adapter = os.path.join(os.environ["RUN_DIR"], "adapter")
model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True)
PeftModel.from_pretrained(model, adapter, local_files_only=True)
print("[MedUMM] PEFT adapter reload passed")
PY
"${MEDUMM_PYTHON}" scripts/verify_alignment_v0_7.py \
  --result "${RUN_DIR}/result.json" \
  --checkpoint "${RUN_DIR}/checkpoint_manifest.json" \
  --audit "${RUN_DIR}/data_audit.json" \
  --history "${RUN_DIR}/history.jsonl" \
  --adapter "${RUN_DIR}/adapter" \
  --model-provenance "${MODEL_ASSETS}/provenance.json" \
  --dataset-provenance data/ultramedical_preferences_v0.7/provenance.json \
  --preferences data/ultramedical_preferences_v0.7/preferences.jsonl \
  --output "${RUN_DIR}/real_run_evidence.json"
