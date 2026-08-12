#!/usr/bin/env bash
#SBATCH --job-name=medumm-v02
#SBATCH --partition=Intel-8358
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

MEDUMM_ROOT="${MEDUMM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MEDUMM_PYTHON="${MEDUMM_PYTHON:-python}"
export MEDUMM_ROOT MEDUMM_PYTHON

cd "${MEDUMM_ROOT}"
export PYTHONPATH="${MEDUMM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${MEDUMM_PYTHON}" -m pytest
bash scripts/run_reference_workflow.sh
