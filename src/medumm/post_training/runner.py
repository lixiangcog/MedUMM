from __future__ import annotations

from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.registry import registry
from medumm.core.results import TrainingResult
from medumm.core.runtime import RuntimeContext


class PostTrainingRunner:
    """Dispatch post-training methods through the trainer registry."""

    def __init__(self, runtime: RuntimeContext | None = None) -> None:
        register_builtins()
        self.runtime = runtime

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
    ) -> TrainingResult:
        method = str(config.get("method", "")).strip().lower()
        if not method:
            raise ValueError("Post-training config requires a method.")
        context = self.runtime or RuntimeContext.create(
            command="post_training",
            config_path=config_path,
            output_directory=config.get("output_directory"),
            runtime_config=config.get("runtime"),
        )
        trainer = registry.trainers.create(method)
        return trainer.fit(config, config_path=config_path, runtime=context)
