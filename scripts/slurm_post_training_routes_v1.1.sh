#!/usr/bin/env bash
#SBATCH --job-name=medumm-v11-routes
#SBATCH --partition=Intel-8358
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/v1.1-routes-%j.out
#SBATCH --error=outputs/slurm/v1.1-routes-%j.err

set -euo pipefail

project_root="${MEDUMM_ROOT:-/data/user/hd66945/MedUMM}"
python_bin="${MEDUMM_PYTHON:-$project_root/.venv-alignment/bin/python}"
cd "$project_root"
mkdir -p outputs/slurm

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -m pytest tests/test_research_post_training.py -q
"$python_bin" -c 'from medumm.cli.main import main; raise SystemExit(main(["post-train", "--list-methods", "--json"]))' \
  > outputs/post-training-methods-v1.1.json
"$python_bin" -m medumm.post_training.acceptance \
  --output-directory "outputs/post_training/v1.1-sequential-${SLURM_JOB_ID:-local}"
