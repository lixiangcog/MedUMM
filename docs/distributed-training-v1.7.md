# Distributed post-training substrate (v1.7)

MedUMM v1.7 provides reusable distributed mechanics for model-specific
post-trainers. It does not treat a generated `torchrun` command as distributed
training evidence: acceptance must initialize every process, optimize a real
PyTorch model, write model and optimizer shards, stop, recover, and continue.

## Public API

The stable package is `medumm.training`:

| API | Responsibility |
| --- | --- |
| `DistributedTrainingConfig` | validated single/DDP/FSDP, precision, EMA, activation and checkpoint contract |
| `DistributedSession` | Slurm/torchrun environment normalization, process-group lifecycle, rank device and collectives |
| `wrap_model` | DDP/FSDP wrapping, mixed-precision policy, auto-wrap and activation checkpointing |
| `DistributedTrainingEngine` | accumulation, `no_sync`, AMP, clipping, optimizer/scheduler steps, history and restart position |
| `ExponentialMovingAverage` | EMA over trainable parameters or local FSDP parameter shards |
| `DistributedCheckpointManager` | distributed model/optimizer shards, rank sidecars, completion markers, retention and recovery |
| `create_dataloader` | deterministic `DistributedSampler` construction and epoch reseeding |

A model adapter owns its forward/loss function and optimizer choice. The engine
owns the mechanics that must behave identically across training methods.

## Strategies and precision

`strategy` is one of `single`, `ddp`, or `fsdp`. With more than one process,
`single` fails closed. `backend: auto` selects NCCL for CUDA and Gloo for CPU.
Each CUDA process is bound to `LOCAL_RANK` before the model is wrapped.

FSDP exposes full, gradient-only, hybrid, and no-shard policies, size-based
auto-wrapping, optional CPU parameter offload, `use_orig_params`, and synchronized
module initialization. FP16 uses autocast plus dynamic gradient scaling; BF16 uses
autocast without a scaler. Gradients are accumulated under `no_sync` until the
optimizer boundary and are unscaled before clipping.

Activation checkpointing uses PyTorch's non-reentrant wrapper. A configuration
must name module classes or set a direct-parameter threshold; enabling the feature
without a selection policy is rejected.

## Checkpoint contract

Each completed checkpoint has this layout:

```text
checkpoints/step-00000008/
├── shards/                 # torch.distributed.checkpoint model + optimizer shards
├── rank-00000.pt           # scheduler, scaler, EMA, RNG and training cursor
├── rank-00001.pt
├── manifest.json
└── COMPLETED
```

`latest.json` only points at a directory containing `COMPLETED`. A crash while
writing a new step therefore leaves the preceding checkpoint recoverable. The
training cursor contains the epoch, next micro-batch, optimizer step and global
sample count, so recovery does not repeat an already applied optimizer update.

PyTorch Distributed Checkpoint can reshard model and optimizer state when the
world size changes. MedUMM's EMA is intentionally rank-local to avoid gathering a
second full model; resuming EMA with a different world size currently fails
closed. Set `checkpoint_strict_topology: true` when every sidecar must retain the
original rank topology.

## Local DDP acceptance

Install PyTorch and run:

```bash
pip install -e ".[test,distributed]"
python -m pytest tests/test_distributed_training.py -q

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m medumm post-train \
  --config configs/post_training/distributed_reference_ddp.yaml
```

The automated test performs two separate launches. The first stops after two
optimizer steps; the second uses `resume_from: auto` and must finish with two rank
sidecars, distributed model/optimizer shards, restored EMA, and a non-null
`resumed_from` field.

## Slurm multi-node FSDP acceptance

The supplied batch script starts one torchrun agent per node through `srun` and
derives `MASTER_ADDR`, `MASTER_PORT`, node rank, process rank and world size from
the allocation:

```bash
sbatch --export=ALL,MEDUMM_STRATEGY=fsdp \
  scripts/slurm_distributed_training_v1.7.sh
```

Use `MEDUMM_NPROC_PER_NODE` together with a matching GPU allocation to increase
local workers. `MEDUMM_PYTHON`, `MEDUMM_ROOT`, `MEDUMM_OUTPUT_DIRECTORY`, and
`MEDUMM_EVIDENCE_PATH` can point at an isolated cluster environment and evidence
location.

The job executes an interrupted phase and a recovery phase, then runs
`scripts/verify_distributed_training_v1_7.py`. Passing requires:

1. the expected strategy and total world size;
2. a completed recovery with a non-empty `resumed_from` value;
3. distributed model and optimizer shard files;
4. one scheduler/scaler/EMA/RNG sidecar per rank;
5. EMA updates after recovery;
6. activation checkpointing for the FSDP recipe.

This reference job validates the shared systems layer only. A real medical UMM
training run remains separately responsible for licensed, de-identified data,
model-specific loss fidelity, checkpoint licensing, and medical evaluation.
