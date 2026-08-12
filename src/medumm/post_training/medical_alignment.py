from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

from medumm.core.interfaces import PostTrainer
from medumm.core.io import ensure_directory, write_json, write_jsonl
from medumm.core.results import Artifact, TrainingResult
from medumm.core.runtime import RuntimeContext, environment_snapshot
from medumm.medical.alignment import (
    AlignmentObjective,
    MedicalAlignmentSample,
    deterministic_epoch_samples,
    load_alignment_data,
)
from medumm.post_training.objectives import alignment_loss, sequence_log_probabilities


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _dtype(torch: Any, value: str) -> Any:
    name = value.casefold()
    if name in {"auto", "bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported training dtype: {value}")


def _device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for alignment training but is unavailable.")
    return requested


def _expanded(value: Any) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


class MedicalAlignmentTrainer(PostTrainer):
    """Parameter-efficient medical SFT and offline preference optimization."""

    name = "medical_alignment"

    def fit(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        runtime: RuntimeContext,
    ) -> TrainingResult:
        if runtime.world_size != 1:
            raise ValueError(
                "medical_alignment v1 uses one process; launch one Slurm task per run."
            )
        try:
            import torch
            import transformers
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "Medical alignment requires torch, transformers, accelerate, and peft."
            ) from error

        objective = AlignmentObjective(str(config.get("objective", "dpo")).casefold())
        data_config = config.get("data")
        model_config = config.get("model")
        if not isinstance(data_config, dict) or not isinstance(model_config, dict):
            raise ValueError("Medical alignment requires data and model mappings.")
        bundle = load_alignment_data(
            data_config, project_root=runtime.project_root, objective=objective
        )
        if bundle.audit["status"] == "failed":
            raise ValueError("Alignment data audit failed: " + "; ".join(bundle.audit["errors"]))
        image_samples = [sample.sample_id for sample in bundle.samples if sample.image_paths]
        if image_samples:
            raise ValueError(
                "medical_alignment v0.7 uses a causal text LM and cannot consume image "
                f"inputs; found {len(image_samples)} multimodal sample(s). The alignment "
                "data contract preserves images for a future processor-backed trainer."
            )

        model_path = _expanded(model_config.get("name_or_path", "")).strip()
        revision = str(model_config.get("revision", "")).strip()
        model_license = str(model_config.get("license", "")).strip()
        if not model_path or not revision or not model_license:
            raise ValueError("Alignment model requires name_or_path, revision, and license.")
        output_directory = Path(
            config.get("output_directory", "outputs/post_training/medical_alignment")
        )
        output_directory = (
            output_directory
            if output_directory.is_absolute()
            else runtime.project_root / output_directory
        )
        ensure_directory(output_directory)
        audit_path = write_json(output_directory / "data_audit.json", bundle.audit)

        seed = int(config.get("seed", runtime.seed))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        requested_device = str(model_config.get("device", runtime.device))
        device = _device(torch, requested_device)
        dtype = _dtype(torch, str(model_config.get("dtype", runtime.dtype)))
        quantization = str(model_config.get("quantization", "none")).casefold()
        load_options: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
            "local_files_only": bool(model_config.get("local_files_only", False)),
        }
        if model_config.get("cache_dir"):
            load_options["cache_dir"] = _expanded(model_config["cache_dir"])
        if quantization == "4bit":
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(model_config.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(
                    model_config.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=dtype,
            )
            load_options["device_map"] = {"": device}
        elif quantization == "none":
            transformers_version = tuple(
                int(part) for part in transformers.__version__.split(".")[:2]
            )
            load_options[
                "dtype" if transformers_version >= (4, 56) else "torch_dtype"
            ] = dtype
        else:
            raise ValueError("model.quantization must be none or 4bit.")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            **{
                key: load_options[key]
                for key in (
                    "revision",
                    "trust_remote_code",
                    "local_files_only",
                    "cache_dir",
                )
                if key in load_options
            },
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_options)
        if quantization == "none":
            model.to(device)
        if bool(config.get("gradient_checkpointing", False)):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        if quantization == "4bit":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            )

        adapter_config = config.get("adapter", {})
        if not isinstance(adapter_config, dict):
            raise ValueError("Alignment adapter must be a mapping.")
        adapter_type = str(adapter_config.get("type", "lora")).casefold()
        if adapter_type != "lora":
            raise ValueError("medical_alignment v1 currently supports adapter.type=lora.")
        raw_targets = adapter_config.get("target_modules", "all-linear")
        target_modules = (
            [str(value) for value in raw_targets]
            if isinstance(raw_targets, list)
            else str(raw_targets)
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(adapter_config.get("rank", 8)),
                lora_alpha=int(adapter_config.get("alpha", 16)),
                lora_dropout=float(adapter_config.get("dropout", 0.05)),
                bias=str(adapter_config.get("bias", "none")),
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
        if bool(config.get("gradient_checkpointing", False)) and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if trainable_parameters <= 0:
            raise RuntimeError("LoRA produced no trainable parameters.")

        batch_size = int(config.get("per_device_batch_size", 1))
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        epochs = int(config.get("epochs", 1))
        epoch_size = int(config.get("samples_per_epoch", 0) or len(bundle.samples))
        max_steps = int(config.get("max_steps", 0) or 0)
        if min(batch_size, accumulation, epochs, epoch_size) < 1:
            raise ValueError("Batch, accumulation, epoch, and epoch-size values must be positive.")
        beta = float(config.get("beta", 0.1))
        margin = float(config.get("margin", 0.0))
        max_length = int(config.get("max_length", 512))
        if max_length < 8:
            raise ValueError("Alignment max_length must be at least 8 tokens.")
        learning_rate = float(config.get("learning_rate", 2e-4))
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        steps_per_epoch = math.ceil(epoch_size / batch_size / accumulation)
        total_steps = max_steps or steps_per_epoch * epochs
        warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.0)))

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-8, step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        system_prompt = str(
            config.get(
                "system_prompt",
                "You are a research medical assistant. Be accurate, state uncertainty, "
                "and do not replace professional clinical judgment.",
            )
        )
        relevance_mean = sum(
            sample.clinical_relevance for sample in bundle.samples
        ) / len(bundle.samples)

        def prompt_text(sample: MedicalAlignmentSample) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": sample.prompt})
            if getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return "\n".join(
                f"{message['role'].title()}: {message['content']}" for message in messages
            ) + "\nAssistant: "

        def encoded(samples: list[MedicalAlignmentSample], response: str) -> dict[str, Any]:
            prompt_ids: list[list[int]] = []
            full_ids: list[list[int]] = []
            for sample in samples:
                prefix = prompt_text(sample)
                completion = sample.chosen if response == "chosen" else sample.rejected
                if completion is None:
                    raise ValueError(f"Sample {sample.sample_id} has no rejected response.")
                prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                prefix_tokens = prefix_tokens[-(max_length - 2) :]
                reserve_eos = int(tokenizer.eos_token_id is not None)
                completion_budget = max_length - len(prefix_tokens) - reserve_eos
                completion_tokens = tokenizer(
                    completion,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=completion_budget,
                )["input_ids"]
                if tokenizer.eos_token_id is not None:
                    completion_tokens = completion_tokens + [tokenizer.eos_token_id]
                complete_tokens = (prefix_tokens + completion_tokens)[:max_length]
                if not completion_tokens or len(complete_tokens) <= len(prefix_tokens):
                    raise ValueError(
                        f"Sample {sample.sample_id} has no completion tokens within max_length."
                    )
                prompt_ids.append(prefix_tokens)
                full_ids.append(complete_tokens)
            width = max(len(value) for value in full_ids)
            input_ids = []
            labels = []
            attention = []
            for prefix, value in zip(prompt_ids, full_ids):
                padding = width - len(value)
                input_ids.append(value + [tokenizer.pad_token_id] * padding)
                labels.append([-100] * len(prefix) + value[len(prefix) :] + [-100] * padding)
                attention.append([1] * len(value) + [0] * padding)
            return {
                "input_ids": torch.tensor(input_ids, device=device),
                "attention_mask": torch.tensor(attention, device=device),
                "labels": torch.tensor(labels, device=device),
            }

        def scores(samples: list[MedicalAlignmentSample], response: str) -> tuple[Any, Any, Any]:
            batch = encoded(samples, response)
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            return sequence_log_probabilities(outputs.logits, batch["labels"])

        def batch_loss(samples: list[MedicalAlignmentSample]) -> tuple[Any, dict[str, Any]]:
            chosen_sum, chosen_mean, chosen_nll = scores(samples, "chosen")
            rejected_sum = rejected_mean = None
            if objective.requires_rejected:
                rejected_sum, rejected_mean, _ = scores(samples, "rejected")
            reference_chosen = reference_rejected = None
            if objective.uses_reference_policy:
                was_training = model.training
                model.eval()
                reference_context = (
                    model.disable_adapter()
                    if hasattr(model, "disable_adapter")
                    else nullcontext()
                )
                try:
                    with torch.no_grad(), reference_context:
                        reference_chosen, _, _ = scores(samples, "chosen")
                        reference_rejected, _, _ = scores(samples, "rejected")
                finally:
                    if was_training:
                        model.train()
            weights = torch.tensor(
                [sample.clinical_relevance / relevance_mean for sample in samples],
                device=device,
            )
            return alignment_loss(
                objective,
                chosen_logps=chosen_sum,
                rejected_logps=rejected_sum,
                chosen_mean_logps=chosen_mean,
                rejected_mean_logps=rejected_mean,
                chosen_nll=chosen_nll,
                reference_chosen_logps=reference_chosen,
                reference_rejected_logps=reference_rejected,
                beta=beta,
                margin=margin,
                weights=weights,
            )

        def evaluate() -> dict[str, float]:
            model.eval()
            totals: dict[str, float] = {}
            count = 0
            limit = int(config.get("evaluation_samples", len(bundle.samples)))
            with torch.no_grad():
                for batch in _chunks(bundle.samples[:limit], batch_size):
                    _, diagnostics = batch_loss(batch)
                    count += len(batch)
                    for key, value in diagnostics.items():
                        totals[key] = totals.get(key, 0.0) + float(value) * len(batch)
            return {key: value / count for key, value in totals.items()}

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        before = evaluate()
        history: list[dict[str, Any]] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        stop = False
        for epoch in range(epochs):
            values = deterministic_epoch_samples(
                bundle.samples, seed=seed, epoch=epoch, epoch_size=epoch_size
            )
            model.train()
            micro_batches = list(_chunks(values, batch_size))
            partial_group = len(micro_batches) % accumulation
            for micro_step, batch in enumerate(micro_batches, 1):
                loss, diagnostics = batch_loss(batch)
                divisor = (
                    partial_group
                    if partial_group and micro_step > len(micro_batches) - partial_group
                    else accumulation
                )
                (loss / divisor).backward()
                should_step = micro_step % accumulation == 0 or micro_step == len(
                    micro_batches
                )
                if not should_step:
                    continue
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(config.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                history.append(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(diagnostics["loss"]),
                        "chosen_nll": float(diagnostics["chosen_nll"]),
                        "reward_margin": float(diagnostics["reward_margin"]),
                        "preference_accuracy": float(
                            diagnostics["preference_accuracy"]
                        ),
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
                if max_steps and global_step >= max_steps:
                    stop = True
                    break
            if stop:
                break
        duration_seconds = time.perf_counter() - started
        after = evaluate()

        adapter_directory = ensure_directory(output_directory / "adapter")
        model.save_pretrained(adapter_directory, safe_serialization=True)
        tokenizer.save_pretrained(adapter_directory)
        history_path = write_jsonl(output_directory / "history.jsonl", history)
        runtime_evidence = environment_snapshot(runtime)
        peak_memory_mb = (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if device.startswith("cuda")
            else None
        )
        checkpoint = write_json(
            output_directory / "checkpoint_manifest.json",
            {
                "schema_version": "1.0",
                "format": "peft_adapter",
                "method": self.name,
                "objective": objective.value,
                "base_model": model_path,
                "base_model_revision": revision,
                "base_model_license": model_license,
                "adapter_path": "adapter",
                "adapter_type": adapter_type,
                "quantization": quantization,
                "dataset_fingerprint": bundle.fingerprint,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "clinical_use": False,
            },
        )
        metrics = {
            "initial_loss": before["loss"],
            "final_loss": after["loss"],
            "initial_preference_accuracy": before["preference_accuracy"],
            "final_preference_accuracy": after["preference_accuracy"],
            "initial_reward_margin": before["reward_margin"],
            "final_reward_margin": after["reward_margin"],
        }
        result = TrainingResult(
            method=self.name,
            status="completed",
            output_directory=str(output_directory),
            checkpoint_path=str(checkpoint),
            metrics=metrics,
            artifacts=[
                Artifact("checkpoint_manifest", str(checkpoint), "application/json"),
                Artifact("adapter", str(adapter_directory), "application/vnd.peft.adapter"),
                Artifact("training_history", str(history_path), "application/x-ndjson"),
                Artifact("data_audit", str(audit_path), "application/json"),
            ],
            metadata={
                "objective": objective.value,
                "samples": len(bundle.samples),
                "sources": len(bundle.sources),
                "dataset_fingerprint": bundle.fingerprint,
                "global_steps": global_step,
                "epochs": epochs,
                "duration_seconds": round(duration_seconds, 3),
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "quantization": quantization,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_parameter_percent": round(
                    100 * trainable_parameters / total_parameters, 6
                ),
                "peak_gpu_memory_mb": peak_memory_mb,
                "hostname": platform.node(),
                "scheduler": {
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                },
                "environment": runtime_evidence,
                "clinical_use": False,
            },
        )
        write_json(output_directory / "result.json", result.to_dict())
        return result
