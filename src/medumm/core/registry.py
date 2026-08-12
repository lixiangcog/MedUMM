from __future__ import annotations

from collections.abc import Callable
from typing import Any


Factory = Callable[[], Any]


class ComponentRegistry:
    """Small registry shared by built-in and third-party components."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Factory]] = {
            "backbone": {},
            "evaluator": {},
            "trainer": {},
        }

    def add(self, kind: str, name: str, factory: Factory) -> None:
        if kind not in self._items:
            raise KeyError(f"Unknown component kind: {kind}")
        self._items[kind][name] = factory

    def get(self, kind: str, name: str) -> Factory:
        try:
            return self._items[kind][name]
        except KeyError as error:
            available = ", ".join(self.names(kind)) or "<none>"
            raise KeyError(f"Unknown {kind} {name!r}; available: {available}") from error

    def names(self, kind: str) -> list[str]:
        if kind not in self._items:
            raise KeyError(f"Unknown component kind: {kind}")
        return sorted(self._items[kind])

    def contains(self, kind: str, name: str) -> bool:
        return name in self._items.get(kind, {})


registry = ComponentRegistry()
