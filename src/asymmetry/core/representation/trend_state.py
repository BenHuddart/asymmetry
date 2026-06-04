"""Structured, per-representation trend state.

Each :class:`~asymmetry.core.representation.base.Representation` owns a
``trend_state`` dict.  Historically this was opaque; :class:`TrendState`
formalises it so the *identity* of a trend is ``(representation, quantity)`` —
there is no global trend namespace, and a quantity named ``lambda`` in the time
representation cannot collide with one in the frequency representation because
they live in different representations' trend states.

The wire format stays a plain ``dict`` (see :meth:`Representation.to_dict`).
:meth:`TrendState.to_dict` omits default/empty fields so an unused trend state
serialises as ``{}`` and round-trips losslessly.  Any unrecognised keys are
preserved under ``legacy`` rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Top-level keys owned by :class:`TrendState` (everything else → ``legacy``).
_KNOWN_KEYS = frozenset(
    {"x_key", "selected_quantities", "derived_params", "model_fits", "axes_state"}
)


@dataclass
class TrendState:
    """The trend configuration for one representation.

    Attributes
    ----------
    x_key
        The trend abscissa: ``"field"``, ``"temperature"``, ``"run"`` or
        ``None`` (auto).
    selected_quantities
        Parameter / derived-quantity names currently plotted.
    derived_params
        Composite (derived) parameter definitions for this representation.
    model_fits
        Per-quantity trend-fit results and annotations.
    axes_state
        Persisted axis/limits/scale state for the trend plot.
    legacy
        Any unrecognised keys, preserved verbatim for forward/backward safety.
    """

    x_key: str | None = None
    selected_quantities: list[str] = field(default_factory=list)
    derived_params: list[dict] = field(default_factory=list)
    model_fits: dict[str, Any] = field(default_factory=dict)
    axes_state: dict[str, Any] = field(default_factory=dict)
    legacy: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return ``True`` when no trend configuration has been stored."""
        return not (
            self.x_key is not None
            or self.selected_quantities
            or self.derived_params
            or self.model_fits
            or self.axes_state
            or self.legacy
        )

    @classmethod
    def from_dict(cls, data: dict | None) -> TrendState:
        """Reconstruct from a stored trend-state dict (unknown keys → legacy)."""
        if not isinstance(data, dict):
            return cls()
        legacy: dict[str, Any] = dict(data.get("legacy") or {})
        for key, value in data.items():
            if key not in _KNOWN_KEYS and key != "legacy":
                legacy[key] = value
        x_key = data.get("x_key")
        raw_quantities = data.get("selected_quantities")
        raw_derived = data.get("derived_params")
        model_fits = data.get("model_fits")
        axes_state = data.get("axes_state")
        return cls(
            x_key=str(x_key) if isinstance(x_key, str) else None,
            selected_quantities=(
                [str(q) for q in raw_quantities] if isinstance(raw_quantities, list) else []
            ),
            derived_params=(
                [dict(p) for p in raw_derived if isinstance(p, dict)]
                if isinstance(raw_derived, list)
                else []
            ),
            model_fits=dict(model_fits) if isinstance(model_fits, dict) else {},
            axes_state=dict(axes_state) if isinstance(axes_state, dict) else {},
            legacy=legacy,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a compact dict, omitting default/empty fields."""
        out: dict[str, Any] = {}
        if self.x_key is not None:
            out["x_key"] = self.x_key
        if self.selected_quantities:
            out["selected_quantities"] = list(self.selected_quantities)
        if self.derived_params:
            out["derived_params"] = [dict(p) for p in self.derived_params]
        if self.model_fits:
            out["model_fits"] = dict(self.model_fits)
        if self.axes_state:
            out["axes_state"] = dict(self.axes_state)
        if self.legacy:
            out["legacy"] = dict(self.legacy)
        return out
