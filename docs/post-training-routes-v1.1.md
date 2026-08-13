# Research post-training routes (v1.1)

MedUMM v1.1 exposes seven heterogeneous post-training methods through one
stable command while keeping each method's actual stage graph and data needs
visible. The route layer launches pinned upstream runtimes; it does not copy
their source code into MedUMM and it does not relabel generic medical SFT as a
paper reproduction.

> Research use only. Every medical dataset must be de-identified and have a
> local provenance record and explicit license before `execution: launch` is
> accepted. Passing the contract smoke test is not evidence of model quality,
> clinical safety, or faithful reproduction of a paper result.

## CLI

Discover methods and stages without importing any model runtime:

```bash
medumm post-train --list-methods
medumm post-train --list-methods --json
medumm post-train --template reca > reca-medical.yaml
```

Generate a preflight artifact without launching a training process:

```bash
medumm post-train --config reca-medical.yaml --plan
```

Launch only after replacing every template value with a pinned checkout,
audited data manifest, provenance file, official entrypoint, and declared
checkpoint path:

```bash
medumm post-train --config reca-medical.yaml
```

Concrete pinned profiles for all seven routes are under
`configs/post_training/profiles/`. They retain native script/module names,
distributed settings, flag conventions, stage-specific objectives, and the
important paper defaults. They intentionally remain `execution: plan` with
`deidentified: false` until the operator supplies and audits medical assets.
Runtimes that expose a Python function instead of a standalone CLI (currently
the integrated UniGame implementation) use an isolated callable worker; its
config payload is redacted from the persisted command plan.

`preflight.json` records the paper/code source, revision, data audit, resolved
working directory, redacted command, stage dependencies, and expected
checkpoint. A non-zero child process or a missing declared checkpoint fails the
command and therefore fails a Slurm job. With `strict_git_revision: true`, a
revision mismatch, unavailable Git status, or tracked local source modification
also fails closed. Paper-runtime launches cannot disable this check, and license
placeholders are not accepted as verified licenses.

## Route contracts

| Method | Stable method | Ordered stages | Medical contract fields beyond `id` |
|---|---|---|---|
| BAGEL SFT | `bagel_sft` | `joint_sft` | `task`, `prompt`, `response` |
| Reconstruction Alignment | `reca` | `reconstruction_alignment` | `image`, `reconstruction_target` |
| Uni-CoT | `unicot` | `hierarchical_cot_sft` | `prompt`, `trajectory`, `response` |
| IRG | `irg` | `think_generate` → `reflect_refine` | reasoning/initial image, then reflection/refined image |
| UniGame | `unigame` | `self_adversarial` | `image`, `question`, `answer` |
| UniPath | `unipath` | four executor stages → `planner` | path/trajectory/image fields; planner features and outcomes |
| LatentUMM | `latentumm` | `dual_latent_alignment` → `latent_dynamics` | paired embeddings, then preferred/rejected latent trajectories |

All stages additionally require `data.deidentified: true`, `data.license`,
`data.provenance`, and `data.contract_manifest`. These are platform gates; the
external runtime can keep its native dataset format, with the contract JSONL
serving as the auditable index connecting patient-safe records to runtime
inputs.

Later stages may consume `{{stages.<name>.checkpoint}}` in launcher arguments.
MedUMM resolves it only after the earlier stage produced the declared artifact.
A later stage may also run by itself when
`inputs.<name>_checkpoint` names an existing external checkpoint.

## Method fidelity and sources

- `bagel_sft`: bridge to the official BAGEL runtime. BAGEL uses a Mixture-of-
  Transformers unified model and documents multimodal supervised fine-tuning.
  Source: [paper](https://arxiv.org/abs/2505.14683),
  [code](https://github.com/ByteDance-Seed/Bagel).
- `reca`: uses frozen understanding-encoder representations as dense
  conditioning and reconstructs the input image. This is self-supervised
  reconstruction alignment, not caption SFT. Source:
  [paper](https://arxiv.org/abs/2509.07295),
  [code](https://github.com/HorizonWind2004/reconstruction-alignment).
- `unicot`: hierarchical macro/micro multimodal reasoning SFT. The paper uses
  interleaved CE/MSE supervision and auxiliary transition objectives. Source:
  [paper](https://arxiv.org/abs/2508.05606),
  [code](https://github.com/Fr0zenCrane/UniCoT).
- `irg`: first strengthens think-and-generate, then trains reflection and
  faithful refinement using full interleaved trajectories. Source:
  [paper](https://arxiv.org/abs/2509.06945),
  [code](https://github.com/Osilly/Interleaving-Reasoning-Generation).
- `unigame`: alternates a decoder-constrained latent challenge update with an
  understanding update over clean and hard examples. Source:
  [paper](https://arxiv.org/abs/2511.19413),
  [current reference code](https://github.com/AIFrontierLab/TorchUMM/tree/main/src/umm/post_training/unigame).
  Its [standalone repository](https://github.com/AIFrontierLab/UniGame) is
  archived and directs users to the integrated implementation.
- `unipath`: trains a path-conditioned executor and then a lightweight planner
  over `direct`, `l0`, `l1`, `l2`, and `l3`. The reference implementation uses
  four staged LoRA executor phases before planner training. Source:
  [paper](https://arxiv.org/abs/2605.11400),
  [reference code](https://github.com/AIFrontierLab/TorchUMM/tree/main/src/umm/post_training/unipath).
- `latentumm`: first aligns cross-modal and bidirectional capacity mappings,
  then stabilizes latent dynamics using stochastic rollouts and preferences.
  Source: [paper](https://arxiv.org/abs/2605.17766),
  [reference code](https://github.com/AIFrontierLab/TorchUMM/tree/main/src/umm/post_training/LatentUMM).

The BAGEL SFT, RecA, Uni-CoT, and IRG routes are marked
`official_runtime_bridge`; UniGame, UniPath, and LatentUMM are marked
`reference_runtime_bridge` because their maintained/reference implementations
live inside the larger toolkit. Users remain
responsible for checking the license of the exact pinned revision and all
transitive model/data assets.

## Acceptance

Acceptance is deliberately recorded per route instead of treating one method
as representative of the others. Run the sequential acceptance suite with:

```bash
python -m medumm.post_training.acceptance \
  --output-directory outputs/post_training/v1.1-sequential-acceptance
```

For each of BAGEL SFT, RecA, Uni-CoT, IRG, UniGame, UniPath, and LatentUMM, the
suite first sends the checked-in paper-runtime profile through the real
`medumm post-train --plan` path and verifies the method-specific stage graph.
It then starts a separate `medumm post-train` process in contract-smoke mode,
which launches every stage as a child process and performs a small
gradient-style optimization. The suite verifies the method and stage identity,
process exit status, log, checkpoint, ordering, and consumption of dependency
checkpoints. It writes a per-method config, paper-profile CLI log,
contract-smoke CLI log, preflights, result, stage logs, and checkpoints, plus a
machine-readable top-level `summary.json`.

Unit tests exercise the same seven-method acceptance path and additionally
validate data schemas, source/runtime isolation, command construction, failure
propagation, and missing artifacts. Every smoke checkpoint includes
`paper_fidelity_claim: false` by design.
The smoke config must set `acceptance_mode: contract_smoke`; that mode is
hard-limited to MedUMM's smoke worker. Normal profiles default to
`paper_runtime` and require the route's declared upstream repository.

The Slurm acceptance entrypoint is
`scripts/slurm_post_training_routes_v1.1.sh`; it runs the route-specific tests,
writes the machine-readable method catalog, then prints a visible PASS/FAIL
line for each of the seven independent CLI executions. The exact job evidence
and per-route result paths are recorded in
`docs/results/v1.1-post-training-routes.json`.

There are three distinct validation levels in that evidence:

1. `paper_profile_plan`: the native launcher contract and paper-specific stage
   graph parsed and produced a preflight; it is intentionally not launch-ready
   while audited assets are absent;
2. `contract_execution`: the MedUMM CLI/process/dependency/artifact contract ran
   successfully for that method;
3. `paper_runtime`: a full upstream model-training execution, which this
   dependency-light acceptance does not run or claim.

For a real medical acceptance run, success additionally requires:

1. immutable source and model revisions;
2. an upstream-compatible, de-identified medical dataset with provenance;
3. the official method runtime and objective, not the smoke worker;
4. a completed GPU job with logs, environment snapshot, checkpoints, and
   downstream medical evaluation;
5. an explicit distinction between engineering success and scientific/clinical
   performance.
