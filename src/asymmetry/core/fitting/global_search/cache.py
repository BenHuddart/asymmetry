"""Cache helpers for staged global-search evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from asymmetry.core.fitting.parameters import ParameterSet


@dataclass
class ExactStructureCache:
    """Cache of exact candidate evaluations keyed by structure signature."""

    entries: dict[tuple[object, ...], object] = field(default_factory=dict)

    def get(self, key: tuple[object, ...]) -> object | None:
        return self.entries.get(key)

    def put(self, key: tuple[object, ...], value: object) -> None:
        self.entries[key] = value


@dataclass
class ApproximateScoreCache:
    """Cache of approximate candidate scores keyed by structure signature."""

    entries: dict[tuple[object, ...], float] = field(default_factory=dict)

    def get(self, key: tuple[object, ...]) -> float | None:
        return self.entries.get(key)

    def put(self, key: tuple[object, ...], value: float) -> None:
        self.entries[key] = float(value)


@dataclass
class WarmStartStore:
    """Cache of parameter seeds for nearby structures."""

    entries: dict[tuple[object, ...], dict[int, ParameterSet]] = field(default_factory=dict)

    def get(self, key: tuple[object, ...]) -> dict[int, ParameterSet] | None:
        return self.entries.get(key)

    def put(self, key: tuple[object, ...], value: dict[int, ParameterSet]) -> None:
        self.entries[key] = value

