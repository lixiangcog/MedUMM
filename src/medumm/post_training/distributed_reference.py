from __future__ import annotations

from pathlib import Path
from typing import Any

from medumm.core.interfaces import PostTrainer
from medumm.core.io import ensure_directory, write_json
from medumm.core.results import Artifact, TrainingResult
from medumm.core.runtime import RuntimeContext, environment_snapshot
from medumm.training import (
    DistributedSession,
    DistributedTrainingConfig,
    DistributedTrainingEngine,
    create_dataloader,
)


class DistributedReferenceTrainer(PostTrainer):
    """Small real trainer used to validate the shared DDP/FSDP substrate."""

    name = "distributed_reference"

    def fit(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None,
        runtime: RuntimeContext,
    ) -> TrainingResult:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "distributed_reference requires PyTorch; install MedUMM with the "
                "distributed extra."
            ) from error
        data_config = config.get("data")
        training_config = config.get("training")
        distributed_config = config.get("distributed")
        if not isinstance(data_config, dict) or data_config.get("synthetic") is not True:
            raise ValueError(
                "distributed_reference only accepts data.synthetic=true; it is a systems "
                "acceptance trainer, not a medical quality experiment."
            )
        if not isinstance(training_config, dict):
            raise ValueError("distributed_reference requires a training mapping.")
        if distributed_config is not None and not isinstance(distributed_config, dict):
            raise ValueError("distributed must be a mapping.")
        distributed = DistributedTrainingConfig.from_mapping(
            distributed_config,
            world_size=runtime.world_size,
        )
        output_directory = Path(
            config.get("output_directory", "outputs/post_training/distributed_reference")
        )
        if not output_directory.is_absolute():
            output_directory = runtime.project_root / output_directory
        ensure_directory(output_directory)
        seed = int(config.get("seed", runtime.seed))
        input_dimensions = int(data_config.get("input_dimensions", 16))
        samples = int(data_config.get("samples", 64))
        if input_dimensions < 2 or samples < 8:
            raise ValueError("Reference data needs at least 2 dimensions and 8 samples.")

        with DistributedSession(distributed, requested_device=runtime.device) as session:
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            features = torch.randn(samples, input_dimensions, generator=generator)
            target_weights = torch.linspace(-0.75, 0.75, input_dimensions)
            targets = features @ target_weights + 0.125
            dataset = torch.utils.data.TensorDataset(features, targets.unsqueeze(-1))
            hidden_dimensions = int(training_config.get("hidden_dimensions", 32))
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dimensions, hidden_dimensions),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dimensions, 1),
            )
            dataloader = create_dataloader(
                dataset,
                session=session,
                batch_size=int(training_config.get("batch_size", 8)),
                shuffle=True,
                seed=seed,
                num_workers=int(training_config.get("num_workers", 0)),
            )
            learning_rate = float(training_config.get("learning_rate", 0.02))
            weight_decay = float(training_config.get("weight_decay", 0.0))
            engine = DistributedTrainingEngine(
                model,
                session=session,
                config=distributed,
                output_directory=output_directory,
                optimizer_factory=lambda parameters: torch.optim.AdamW(
                    parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                ),
                scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=float(training_config.get("lr_gamma", 0.98)),
                ),
            )

            def training_step(active_model: Any, batch: Any) -> Any:
                inputs, labels = batch
                predictions = active_model(inputs)
                return torch.nn.functional.mse_loss(predictions.float(), labels.float())

            max_steps = training_config.get("max_optimizer_steps")
            report = engine.fit(
                dataloader,
                step_function=training_step,
                epochs=int(training_config.get("epochs", 4)),
                resume_from=config.get("resume_from"),
                max_optimizer_steps=None if max_steps is None else int(max_steps),
                checkpoint_metadata={
                    "method": self.name,
                    "seed": seed,
                    "synthetic": True,
                    "clinical_use": False,
                },
            )
            checkpoint = Path(report["checkpoint"])
            if session.is_main_process:
                write_json(output_directory / "distributed_report.json", report)
                write_json(
                    output_directory / "environment.json",
                    environment_snapshot(runtime),
                )
            session.barrier()

        artifacts = [
            Artifact(
                "sharded_checkpoint",
                str(checkpoint),
                "application/vnd.pytorch.distributed-checkpoint",
            ),
            Artifact(
                "distributed_report",
                str(output_directory / "distributed_report.json"),
                "application/json",
            ),
        ]
        if (output_directory / "history.jsonl").is_file():
            artifacts.append(
                Artifact(
                    "training_history",
                    str(output_directory / "history.jsonl"),
                    "application/x-ndjson",
                )
            )
        result = TrainingResult(
            method=self.name,
            status=str(report["status"]),
            output_directory=str(output_directory),
            checkpoint_path=str(checkpoint),
            metrics={"mean_loss": float(report["mean_loss"])},
            artifacts=artifacts,
            metadata={
                **report,
                "synthetic": True,
                "clinical_use": False,
                "validation_scope": (
                    "Distributed systems acceptance only; no medical model quality claim."
                ),
            },
        )
        if runtime.rank == 0:
            write_json(output_directory / "result.json", result.to_dict())
        return result
