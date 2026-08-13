# Per-model environment isolation (v1.3)

MedUMM has 32 model resources with mutually incompatible dependency histories:
modern Gemma/Qwen runtimes, InternVL/LLaVA variants, legacy OpenFlamingo and
MedCLIP stacks, native 3D libraries, and contrastive encoders. A single
`.[medical]` installation cannot reproduce all of them safely.

## Contract and generated artifacts

The source of truth is
`src/medumm/environments/catalog/models.yaml`. There is exactly one record per
model catalog entry. Each record contains:

- exact Python minor, CUDA line and digest-pinned base image;
- exact direct dependencies and immutable Git commits;
- immutable model artifact revision;
- upstream access level and an explicit restricted-model gate;
- minimum GPU memory, suggested GPU count and import probes;
- a validation level with committed evidence when runtime validation exists.

The following files are generated for every model and must not be edited by
hand:

```text
environments/models/<model>/
├── requirements.txt
├── lock.txt
├── sources.lock
├── Dockerfile
└── apptainer.def
```

Resolve, regenerate and verify all 160 files with:

```bash
python scripts/generate_model_environments.py
python scripts/generate_model_environments.py --check
python scripts/resolve_model_environments.py
```

CI runs the check form, resolves all 32 Linux dependency graphs, validates all
contracts, rejects mutable package/Git or base-image references, checks shell
syntax and confirms 1:1 coverage against the model resource catalog.

## Local or HPC virtual environment

```bash
# Creates .venv-models/lingshu_7b
bash scripts/setup_model_env.sh lingshu_7b

# Re-check an existing environment without installing
bash scripts/setup_model_env.sh lingshu_7b --check-only

# Restricted weights: only after accepting upstream terms
bash scripts/setup_model_env.sh medgemma_1_5_4b_it --accept-terms
```

The script selects the contract's Python minor, installs only that model's
dependencies, installs MedUMM itself without resolving the global heavyweight
extras, runs isolated import probes and records the contract fingerprint in the
environment directory.

On Slurm:

```bash
sbatch --export=ALL,MODEL_NAME=lingshu_7b scripts/slurm_model_environment.sh
```

## Docker and Apptainer/Singularity

```bash
# Docker/Podman where available
bash scripts/build_model_container.sh lingshu_7b

# HPC auto-detects Apptainer or Singularity and writes .containers/lingshu_7b.sif
MEDUMM_CONTAINER_ENGINE=singularity \
MEDUMM_APPTAINER_BUILD_MODE=fakeroot \
  bash scripts/build_model_container.sh lingshu_7b
```

Apptainer/Singularity definition builds require one facility supplied by the
cluster administrator: subordinate-ID `fakeroot`, a configured remote builder,
or approved `sudo`. Select it with `MEDUMM_APPTAINER_BUILD_MODE` (`fakeroot`,
`remote`, or `sudo`). Pulling and running prebuilt SIF/OCI images does not need
these build privileges. The current acceptance cluster can pull/run OCI images,
but does not grant this account fakeroot mappings or a remote build endpoint.

Weights, datasets and secrets are not baked into images. Mount them at runtime
under `/models`, `/cache/huggingface` and `/outputs`. The base image uses a
registry digest, while `sources.lock` records official upstream source commits.

## Modal

MedUMM mirrors TorchUMM's useful cloud-isolation principle without maintaining
a second dependency catalog. `modal/images.py:image_for()` creates one image
per model directly from the same environment contract:

```bash
pip install modal==1.1.4
modal setup
python modal/run.py --model lingshu_7b --json
```

Attach Hugging Face credentials as a Modal secret; do not commit credentials or
place them in image build arguments.

## Validation truthfulness

Environment status is deliberately separate from model integration status:

1. `contract_validated`: schema, immutable pins and generated artifacts pass.
2. `lock_resolved`: package resolver succeeds for the target Linux/Python pair.
3. `container_built`: the complete image builds successfully.
4. `import_validated`: imports succeed inside that isolated image/environment.
5. `runtime_validated`: pinned model weights complete a real task and evidence
   is committed.

All 32 models now meet level 1. Existing evidence preserves level 5 only for
LLaVA-Med v1.5, Lingshu-7B and PubMedCLIP. The other 29 must not be described as
runtime validated until their gated assets, image builds and model smoke runs
have succeeded on target GPU hardware.

## Adding a model

Adding a model resource now requires an environment contract in the same
change. Run the generator, add all four generated artifacts, and add runtime
evidence only after a real model invocation. Sharing a compatibility profile is
allowed; sharing a mutable environment directory between model names is not.
