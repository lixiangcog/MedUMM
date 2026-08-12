from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ModelAdapter(Protocol):
    name: str
    supported_tasks: frozenset[str]

    def load(self, config: dict[str, Any]) -> None: ...

    def understanding(
        self,
        prompt: str | None,
        images: list[str],
        videos: list[str],
        parameters: dict[str, Any],
    ) -> Any: ...


class Trainer(Protocol):
    name: str

    def fit(
        self,
        config: dict[str, Any],
        config_path: str | Path | None = None,
    ) -> dict[str, Any]: ...
