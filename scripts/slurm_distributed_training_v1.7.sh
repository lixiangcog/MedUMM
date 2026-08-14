#!/usr/bin/env bash
#SBATCH --job-name=medumm-v17-dist
#SBATCH --partition=A800-N
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/v1.7-distributed-%j.out
#SBATCH --error=outputs/slurm/v1.7-distributed-%j.err

set -euo pipefail

project_root="${MEDUMM_ROOT:-/data/user/hd66945/MedUMM}"
python_bin="${MEDUMM_PYTHON:-$project_root/.venv-distributed/bin/python}"
strategy="${MEDUMM_STRATEGY:-fsdp}"
nproc_per_node="${MEDUMM_NPROC_PER_NODE:-1}"
device="${MEDUMM_DEVICE:-}"
backend="${MEDUMM_BACKEND:-}"
fsdp_sync_module_states="${MEDUMM_FSDP_SYNC_MODULE_STATES:-}"
job_key="${SLURM_JOB_ID:-local}"
output_directory="${MEDUMM_OUTPUT_DIRECTORY:-$project_root/outputs/post_training/v1.7-${strategy}-${job_key}}"
evidence_path="${MEDUMM_EVIDENCE_PATH:-$project_root/outputs/post_training/v1.7-${strategy}-${job_key}-evidence.json}"
config_path="$project_root/configs/post_training/distributed_reference_${strategy}.yaml"

cd "$project_root"
mkdir -p outputs/slurm "$output_directory"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MEDUMM_PYTHON="$python_bin"
export MEDUMM_CONFIG="$config_path"
export MEDUMM_OUTPUT_DIRECTORY="$output_directory"
export MEDUMM_NPROC_PER_NODE="$nproc_per_node"
export MEDUMM_DEVICE="$device"
export MEDUMM_BACKEND="$backend"
export MEDUMM_FSDP_SYNC_MODULE_STATES="$fsdp_sync_module_states"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + job_key % 20000))"
# Some Slurm installations expose both variables with partition-specific values.
# srun rejects the launch when they disagree, so keep the allocation's canonical
# SLURM_CPUS_PER_TASK value and remove the redundant inherited TRES spelling.
unset SLURM_TRES_PER_TASK

launch() {
  export MEDUMM_PHASE="$1"
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -lc '
      args=(
        -m torch.distributed.run
        --nnodes="$SLURM_NNODES"
        --nproc-per-node="$MEDUMM_NPROC_PER_NODE"
        --node-rank="$SLURM_NODEID"
        --master-addr="$MASTER_ADDR"
        --master-port="$MASTER_PORT"
        -m medumm post-train
        --config "$MEDUMM_CONFIG"
        --set "post_training.output_directory=$MEDUMM_OUTPUT_DIRECTORY"
      )
      if [[ "$MEDUMM_PHASE" == "interrupt" ]]; then
        args+=(--set post_training.training.max_optimizer_steps=2)
      else
        args+=(--set post_training.resume_from=auto)
      fi
      if [[ -n "$MEDUMM_DEVICE" ]]; then
        args+=(--set "runtime.device=$MEDUMM_DEVICE")
      fi
      if [[ -n "$MEDUMM_BACKEND" ]]; then
        args+=(--set "post_training.distributed.backend=$MEDUMM_BACKEND")
      fi
      if [[ -n "$MEDUMM_FSDP_SYNC_MODULE_STATES" ]]; then
        args+=(--set "post_training.distributed.fsdp_sync_module_states=$MEDUMM_FSDP_SYNC_MODULE_STATES")
      fi
      exec "$MEDUMM_PYTHON" "${args[@]}"
    '
}

launch interrupt
launch resume

world_size="$((SLURM_NNODES * nproc_per_node))"
"$python_bin" scripts/verify_distributed_training_v1_7.py \
  --output-directory "$output_directory" \
  --expected-strategy "$strategy" \
  --expected-world-size "$world_size" \
  --evidence "$evidence_path"
