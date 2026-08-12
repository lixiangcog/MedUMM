from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from medumm.core.registry import registry


def register_builtin_trainers() -> None:
    if not registry.contains("trainer", "medical_sft"):
        registry.add(
            "trainer",
            "medical_sft",
            lambda: getattr(import_module("medumm.post_training.medical_sft"), "MedicalSFTTrainer")(),
        )


class PostTrainingRunner:
    def __init__(self) -> None:
        register_builtin_trainers()

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        method = str(config.get("method", "")).strip()
        if not method:
            raise ValueError("Post-training config requires a method.")
        return registry.get("trainer", method)().fit(config, config_path)
