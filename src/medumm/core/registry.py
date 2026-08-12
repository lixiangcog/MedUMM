from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from medumm.core.contracts import ComponentDescriptor
from medumm.core.exceptions import ComponentNotFoundError, DuplicateComponentError


Component = TypeVar("Component")
Factory = Callable[[], Component]


@dataclass(frozen=True, slots=True)
class Registration(Generic[Component]):
    descriptor: ComponentDescriptor
    factory: Factory[Component]


class TypedRegistry(Generic[Component]):
    """One plugin registry with deterministic duplicate and lookup behavior."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Registration[Component]] = {}

    def register(
        self,
        name: str,
        factory: Factory[Component],
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Component name cannot be empty.")
        if normalized in self._items and not replace:
            raise DuplicateComponentError(f"{self.kind} {normalized!r} is already registered.")
        self._items[normalized] = Registration(
            descriptor=ComponentDescriptor(
                kind=self.kind,
                name=normalized,
                description=description,
                metadata=dict(metadata or {}),
            ),
            factory=factory,
        )

    def decorator(
        self,
        name: str,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[Factory[Component]], Factory[Component]]:
        def register_factory(factory: Factory[Component]) -> Factory[Component]:
            self.register(name, factory, description=description, metadata=metadata)
            return factory

        return register_factory

    def create(self, name: str) -> Component:
        normalized = name.strip().lower()
        try:
            registration = self._items[normalized]
        except KeyError as error:
            available = ", ".join(self.names()) or "<none>"
            raise ComponentNotFoundError(
                f"Unknown {self.kind} {name!r}; available: {available}."
            ) from error
        return registration.factory()

    def contains(self, name: str) -> bool:
        return name.strip().lower() in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def descriptors(self) -> list[ComponentDescriptor]:
        return [self._items[name].descriptor for name in self.names()]


class ComponentHub:
    """Four registries matching the core functionality layer."""

    def __init__(self) -> None:
        self.models: TypedRegistry[Any] = TypedRegistry("model")
        self.datasets: TypedRegistry[Any] = TypedRegistry("dataset")
        self.benchmarks: TypedRegistry[Any] = TypedRegistry("benchmark")
        self.trainers: TypedRegistry[Any] = TypedRegistry("trainer")

    def _registry(self, kind: str) -> TypedRegistry[Any]:
        aliases = {
            "model": self.models,
            "models": self.models,
            "backbone": self.models,
            "dataset": self.datasets,
            "datasets": self.datasets,
            "benchmark": self.benchmarks,
            "benchmarks": self.benchmarks,
            "evaluator": self.benchmarks,
            "trainer": self.trainers,
            "trainers": self.trainers,
            "post_trainer": self.trainers,
        }
        try:
            return aliases[kind.strip().lower()]
        except KeyError as error:
            raise ValueError(f"Unknown component kind: {kind!r}.") from error

    def register(
        self,
        kind: str,
        name: str,
        factory: Factory[Any],
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        self._registry(kind).register(
            name,
            factory,
            description=description,
            metadata=metadata,
            replace=replace,
        )

    def create(self, kind: str, name: str) -> Any:
        return self._registry(kind).create(name)

    def names(self, kind: str) -> list[str]:
        return self._registry(kind).names()

    def descriptors(self, kind: str) -> list[ComponentDescriptor]:
        return self._registry(kind).descriptors()

    def catalog(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [item.to_dict() for item in self.models.descriptors()],
            "datasets": [item.to_dict() for item in self.datasets.descriptors()],
            "benchmarks": [item.to_dict() for item in self.benchmarks.descriptors()],
            "trainers": [item.to_dict() for item in self.trainers.descriptors()],
        }


registry = ComponentHub()
