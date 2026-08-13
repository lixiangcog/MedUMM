# Inference optimization (v1.2)

MedUMM v1.2 makes inference engines a first-class platform contract. It adds
vLLM and SGLang serving, continuous request admission, tensor/pipeline/data
parallel configuration, a native patched-vLLM Emu3.5 path, and reproducible
latency/throughput reports.

> Research use only. Generated images and text are not clinical evidence, and
> an engine acceptance run is not a medical quality or safety evaluation.

## Capability boundary

| Path | Tasks | CFG | Continuous batching | Parallel controls |
|---|---|---:|---:|---|
| native model adapter | adapter-specific | adapter-specific | no | adapter-specific |
| vLLM OpenAI server | understanding | no Emu3.5 native CFG | yes | TP, PP, DP |
| SGLang OpenAI server | understanding | no Emu3.5 native CFG | yes | TP, PP, DP |
| Emu3.5 patched vLLM | understanding, generation, X2I editing | yes | yes | TP |

This distinction is enforced. A standard OpenAI-compatible request cannot carry
Emu3.5's `uncond_prompt_token_ids`, and SGLang does not declare BAAI's custom
cond/uncond scheduler. MedUMM therefore rejects Emu3.5 CFG on both HTTP routes
instead of silently running a different algorithm. SGLang remains available for
other medical vision-language models through the same HTTP adapter.

## Stable configuration

`BackendConfig` contains four stable groups:

- backend name and execution mode (`native`, `vllm`, or `sglang`;
  `in_process` or `openai_http`);
- `ParallelConfig` with tensor, pipeline, and data parallel sizes plus the
  calculated world size;
- `SchedulerConfig` with continuous batching, maximum in-flight sequences,
  token budget, and queue limit;
- endpoint, timeout, and optional API-key environment variable.

Model capabilities separately declare compatible backends, CFG support,
continuous batching, and parallelism. Inspect the active environment with:

```bash
medumm backends --json
```

The capability report declares Emu3.5 CFG only when both vLLM `0.11.0` and the
patched `vllm.v1.core.sched.batch_scheduler` module are present.

## Emu3.5

The `emu3_5` adapter uses the immutable BAAI source checkout and builds the
native `Emu3_5ForCausalLM` vLLM implementation. It submits every request as a
conditional/unconditional pair in one engine batch, passes CFG and differential
text/image sampling controls through `SamplingParams.extra_args`, and verifies
that the patched architecture and scheduler are active after engine creation.

Required versions are Python 3.12, vLLM 0.11.0, Transformers 4.56.1, and
FlashAttention 2.8.3. The source, model, and vision-tokenizer revisions must all
be immutable. MedUMM additionally checks the source checkout's actual Git HEAD.

The setup helper creates an isolated runtime and applies the 20 patches from
the pinned upstream checkout:

```bash
MEDUMM_INFERENCE_BACKEND=vllm \
  scripts/setup_inference_optimization_v1.2.sh
```

Example T2I inference and its batch benchmark use
`configs/inference/emu3_5_vllm.yaml` and
`configs/inference/benchmark_emu3_5_vllm.yaml`. Both default to two-way tensor
parallelism, two simultaneously scheduled sequences, and a 26,000-token engine
budget, matching the upstream vLLM route while keeping the limits configurable.

## Serving and performance benchmark

Planning validates the pinned revision, world size, visible GPU count, package
availability, redacted configuration, and exact launch command:

```bash
medumm serve --config configs/inference/serve_vllm.yaml --plan
medumm serve --config configs/inference/serve_sglang.yaml --plan
```

Launching uses `server.execution=launch`. vLLM maps TP/PP/DP and
`max_num_seqs`/`max_num_batched_tokens`; SGLang maps TP/PP/DP,
`max_running_requests`, prefill-token, and queue limits. Both expose an
OpenAI-compatible endpoint consumed by `openai_http`.

`benchmark-inference` performs warm-up and measured iterations, preserves a
complete runtime/environment snapshot, and reports:

- request and output-token throughput;
- mean, p50, p95, and p99 end-to-end latency;
- engine TTFT, queue, and engine latency when the backend exposes them;
- per-iteration wall time and generated tokens.

The A800 acceptance script starts a two-GPU server, runs 24 sequential and 24
eight-way concurrent requests against the same pinned Qwen2.5-VL-3B model, and
verifies the server plan and both reports:

```bash
MEDUMM_INFERENCE_BACKEND=vllm sbatch scripts/slurm_inference_optimization_v1.2.sh
MEDUMM_INFERENCE_BACKEND=sglang sbatch scripts/slurm_inference_optimization_v1.2.sh
```

The sequential/concurrent comparison proves that the request path admits a
real concurrent workload; it does not promise that every short-prompt workload
will obtain a speedup. Hardware, prompt lengths, output lengths, cache state,
and scheduler settings must be retained with any performance claim.

## Validation levels

1. `interface`: schemas, capability boundaries, launch commands, batching,
   result ordering, metrics, and fail-closed CFG behavior pass unit tests.
2. `runtime_preflight`: exact packages, Emu3.5 patches, immutable revisions,
   model assets, and CUDA imports are present.
3. `backend_runtime`: a pinned model completes sequential and concurrent
   inference on allocated GPUs and writes benchmark evidence.
4. `emu3_5_runtime`: pinned Emu3.5 and vision-tokenizer weights complete real
   CFG generation on the patched two-GPU engine.

Evidence must state the highest level actually reached. Missing gated weights,
downloads, or GPU allocation are never rewritten as a successful run.
