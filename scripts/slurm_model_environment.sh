#!/usr/bin/env bash
#SBATCH --job-name=medumm-env
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-/data/user/hd66945/MedUMM}"
MODEL_NAME="${MODEL_NAME:?submit with sbatch --export=ALL,MODEL_NAME=<catalog-name>}"
cd "${MEDUMM_ROOT}"

if [[ "${MEDUMM_BUILD_CONTAINER:-0}" == "1" ]]; then
  bash scripts/build_model_container.sh "${MODEL_NAME}"
else
  bash scripts/setup_model_env.sh "${MODEL_NAME}" ${MEDUMM_ACCEPT_TERMS:+--accept-terms}
fi
