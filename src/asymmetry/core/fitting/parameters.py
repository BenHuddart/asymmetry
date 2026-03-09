"""Parameter objects with bounds, constraints, and linking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Parameter:
    """A single fit parameter."""

    name: str
    value: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    fixed: bool = False
    expr: str | None = None  # Expression constraint (e.g. tie to another param)

    @property
    def is_constrained(self) -> bool:
        return self.fixed or self.expr is not None


class ParameterSet:
    """Ordered collection of :class:`Parameter` objects."""

    def __init__(self, params: list[Parameter] | None = None) -> None:
        self._params: dict[str, Parameter] = {}
        for p in params or []:
            self.add(p)

    def add(self, param: Parameter) -> None:
        self._params[param.name] = param

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    @property
    def free_parameters(self) -> list[Parameter]:
        return [p for p in self if not p.is_constrained]

    @property
    def names(self) -> list[str]:
        return list(self._params)

    def values_array(self) -> list[float]:
        return [p.value for p in self]

    def update_values(self, values: dict[str, float]) -> None:
        for name, val in values.items():
            if name in self._params:
                self._params[name].value = val
