from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

from medumm.core.contracts import ModelCapabilities
from medumm.core.exceptions import UnsupportedTaskError
from medumm.core.results import EvaluationResult, InferenceResult, TrainingResult

if TYPE_CHECKING:
    from medumm.inference.request import InferenceRequest
    from medumm.core.runtime import RuntimeContext


class ModelAdapter(ABC):
    """Standard interface between task pipelines and model implementations."""

    name: str
    capabilities: ModelCapabilities

    @abstractmethod
    def load(self, config: dict[str, Any], runtime: "RuntimeContext") -> None:
        """Load weights and model-specific processors."""

    def understand_batch(self, requests: list["InferenceRequest"]) -> list[InferenceResult]:
        raise UnsupportedTaskError(f"{self.name} does not implement understanding.")

    def generate_batch(self, requests: list["InferenceRequest"]) -> list[InferenceResult]:
        raise UnsupportedTaskError(f"{self.name} does not implement generation.")

    def edit_batch(self, requests: list["InferenceRequest"]) -> list[InferenceResult]:
        raise UnsupportedTaskError(f"{self.name} does not implement editing.")

    def close(self) -> None:
        """Release model resources when an adapter needs explicit cleanup."""

    def runtime_info(self) -> dict[str, Any]:
        """Return non-secret, machine-readable details about the loaded runtime."""

        return {}


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, config: dict[str, Any], project_root: Path) -> list[Any]:
        """Load and normalize a dataset into stable sample objects."""

    @abstractmethod
    def fingerprint(self, config: dict[str, Any], project_root: Path) -> str:
        """Return a stable identity for the exact configured dataset."""


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path,
        runtime: "RuntimeContext",
    ) -> EvaluationResult:
        """Execute generation, scoring, or both and return EvaluationResult."""


class MetricSuite(ABC):
    """Versioned, model-independent scoring contract for a benchmark family."""

    name: str
    version: str

    @abstractmethod
    def score(self, prediction: str, content: dict[str, Any]) -> dict[str, Any]:
        """Score one normalized prediction against one sample payload."""

    @abstractmethod
    def summarize(
        self,
        rows: list[dict[str, Any]],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate item scores under an explicit evaluation protocol."""


class PostTrainer(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None,
        runtime: "RuntimeContext",
    ) -> TrainingResult:
        """Train a model and return a self-describing checkpoint result."""
