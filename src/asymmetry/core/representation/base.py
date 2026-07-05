"""Core representation abstraction for the Domain → Representation model.

A :class:`Representation` is a *recipe over a* :class:`~asymmetry.core.data.dataset.Run`
that yields one or more plottable :class:`~asymmetry.core.data.dataset.MuonDataset`
curves, plus the single stored fit and trend state for that view of the data.

Persistence is **recipe-only**: ``to_dict``/``from_dict`` serialise the
generation recipe, the fit slot, and the trend state — never the computed
arrays.  The arrays live in the transient ``_datasets`` cache, repopulated by
:meth:`Representation.compute` on demand (e.g. after loading a project).

Each dataset owns up to four representations, one per
:class:`RepresentationType`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from asymmetry.core.data.dataset import MuonDataset, Run


class RepresentationType(str, Enum):
    """The kinds of analysis representation a dataset can hold."""

    TIME_FB_ASYMMETRY = "time_fb_asymmetry"
    TIME_GROUPS = "time_groups"
    TIME_MAXENT_RECON = "time_maxent_recon"
    FREQ_FFT = "freq_fft"
    FREQ_MAXENT = "freq_maxent"

    @property
    def domain(self) -> str:
        """Return the analysis domain (``"time"`` or ``"frequency"``)."""
        return DOMAIN_OF[self]


#: Map each representation type to its analysis domain.
DOMAIN_OF: dict[RepresentationType, str] = {
    RepresentationType.TIME_FB_ASYMMETRY: "time",
    RepresentationType.TIME_GROUPS: "time",
    RepresentationType.TIME_MAXENT_RECON: "time",
    RepresentationType.FREQ_FFT: "frequency",
    RepresentationType.FREQ_MAXENT: "frequency",
}

#: Allowed fit provenance markers.
FIT_PROVENANCE = ("none", "single", "batch", "global", "wizard")


@dataclass
class FitSlot:
    """The single stored fit for one ``(dataset, representation)`` pair.

    ``model`` is a :meth:`CompositeModel.to_dict` payload (or ``None`` for an
    empty slot); ``result`` is a JSON-serialisable fit-result summary.  A fit
    produced as a member of a
    :class:`~asymmetry.core.representation.series.FitSeries` records the series
    id (``batch_id``) so trending and divergence can find its series.
    """

    model: dict | None = None
    parameters: list[dict] = field(default_factory=list)
    result: dict | None = None
    provenance: str = "none"
    batch_id: str | None = None
    diverged: bool = False
    include_in_trend: bool = True
    #: The fit panel's single-fit *form* payload (composite_model, parameters,
    #: result_html, wizard_state) for restoring the editor when this slot is
    #: re-selected.  It carries the GUI-only extras (result HTML, wizard cache)
    #: that ``model``/``parameters`` do not, so a per-projection single fit can
    #: be restored verbatim.  Empty for slots produced outside the single-fit
    #: GUI path (batch/global members) and for pre-this-change projects.
    ui_state: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return ``True`` when no model or result has been stored."""
        return self.model is None and self.result is None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of the slot."""
        payload = {
            "model": None if self.model is None else dict(self.model),
            "parameters": [dict(p) for p in self.parameters],
            "result": None if self.result is None else dict(self.result),
            "provenance": self.provenance,
            "batch_id": self.batch_id,
            "diverged": bool(self.diverged),
            "include_in_trend": bool(self.include_in_trend),
        }
        # Only persist ``ui_state`` when populated — batch/global members and
        # pre-this-change slots carry none, and an empty dict would bloat every
        # saved slot for no gain.
        if self.ui_state:
            payload["ui_state"] = dict(self.ui_state)
        return payload

    @classmethod
    def from_dict(cls, data: dict | None) -> FitSlot:
        """Reconstruct a :class:`FitSlot` from serialised data."""
        if not isinstance(data, dict):
            return cls()
        provenance = str(data.get("provenance", "none"))
        if provenance not in FIT_PROVENANCE:
            provenance = "none"
        raw_params = data.get("parameters")
        parameters = (
            [dict(p) for p in raw_params if isinstance(p, dict)]
            if (isinstance(raw_params, list))
            else []
        )
        model = data.get("model")
        # Migrate legacy ``fraction_<k>`` parameter entries to the n-1 free-fraction
        # scheme so pre-rework projects load, display, and refit. Guarded: a
        # missing/malformed model payload simply skips migration. Only pay for the
        # model rebuild when a legacy ``fraction_<k>`` name is actually present —
        # every migratable key has this prefix, so its absence guarantees a no-op.
        has_legacy_fraction_name = any(
            isinstance(p.get("name"), str) and p["name"].startswith("fraction_") for p in parameters
        )
        if isinstance(model, dict) and parameters and has_legacy_fraction_name:
            from asymmetry.core.fitting.composite import (
                CompositeModel,
                migrate_legacy_fraction_parameter_entries,
            )

            try:
                composite = CompositeModel.from_dict(model, allow_missing=True)
            except (ValueError, KeyError, TypeError):
                composite = None
            if composite is not None:
                parameters = migrate_legacy_fraction_parameter_entries(composite, parameters)
        result = data.get("result")
        raw_ui_state = data.get("ui_state")
        return cls(
            model=dict(model) if isinstance(model, dict) else None,
            parameters=parameters,
            result=dict(result) if isinstance(result, dict) else None,
            provenance=provenance,
            batch_id=(str(data["batch_id"]) if data.get("batch_id") is not None else None),
            diverged=bool(data.get("diverged", False)),
            include_in_trend=bool(data.get("include_in_trend", True)),
            ui_state=dict(raw_ui_state) if isinstance(raw_ui_state, dict) else {},
        )


class Representation(ABC):
    """Base class for a recipe-driven view of a run's data.

    Subclasses set :attr:`rep_type` and implement :meth:`compute`.
    """

    #: Set by each concrete subclass.
    rep_type: ClassVar[RepresentationType]

    #: Whether :meth:`ProjectModel.recompute_all` should rebuild this
    #: representation when a project loads.  Expensive iterative
    #: representations (MaxEnt) opt out and are recomputed on demand instead.
    recompute_on_load: ClassVar[bool] = True

    def __init__(
        self,
        recipe: dict | None = None,
        fit: FitSlot | None = None,
        trend_state: dict | None = None,
        result_metadata: dict | None = None,
        projection_fits: dict[str, FitSlot] | None = None,
    ) -> None:
        self.recipe: dict[str, Any] = dict(recipe or {})
        # ``fit`` is the default (non-projection / single-pair) slot — a plain
        # attribute, so every existing ``representation.fit`` read/write/in-place
        # mutation is unchanged. ``projection_fits`` holds the per-projection
        # slots (P_x/P_y/P_z, transverse-field labels, …), letting each
        # projection of a vector grouping remember its own fit.
        self.fit: FitSlot = fit if isinstance(fit, FitSlot) else FitSlot()
        self.projection_fits: dict[str, FitSlot] = {}
        if isinstance(projection_fits, dict):
            for key, slot in projection_fits.items():
                if key and isinstance(slot, FitSlot):
                    self.projection_fits[str(key)] = slot
        self.trend_state: dict[str, Any] = dict(trend_state or {})
        self.result_metadata: dict[str, Any] = dict(result_metadata or {})
        self._datasets: list[MuonDataset] | None = None

    # ── fit slots (per projection) ─────────────────────────────────────────

    @staticmethod
    def _fit_key(projection: str | None) -> str | None:
        """Normalise a projection label to a projection-fit key.

        Falsy labels and the ``"ALL"`` aggregate sentinel (which is not a
        physical projection and is never fit) map to ``None`` — the default
        ``fit`` slot — so they never create a phantom projection entry.
        """
        if not projection or projection == "ALL":
            return None
        return str(projection)

    def fit_for(self, projection: str | None) -> FitSlot:
        """Return the fit slot for *projection* (a fresh empty slot if unfit).

        This is a pure read — it never inserts, so inspecting an unfit
        projection does not leak an empty slot into the saved project.
        """
        key = self._fit_key(projection)
        if key is None:
            return self.fit
        return self.projection_fits.get(key, FitSlot())

    def set_fit_for(self, projection: str | None, slot: FitSlot) -> None:
        """Store *slot* as the fit for *projection*."""
        resolved = slot if isinstance(slot, FitSlot) else FitSlot()
        key = self._fit_key(projection)
        if key is None:
            self.fit = resolved
        else:
            self.projection_fits[key] = resolved

    def iter_fit_slots(self) -> list[tuple[str | None, FitSlot]]:
        """Return ``(projection_key, slot)`` for every stored fit (default + projections)."""
        return [(None, self.fit), *self.projection_fits.items()]

    # ── identity ───────────────────────────────────────────────────────────

    @property
    def domain(self) -> str:
        """Return the analysis domain of this representation."""
        return self.rep_type.domain

    # ── computation (transient arrays) ─────────────────────────────────────

    @abstractmethod
    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        """Build the representation's plottable curves from *run*.

        Returns one or more :class:`MuonDataset` curves.  F-B asymmetry yields a
        single-element list; grouped and frequency representations yield one
        entry per detector group.  The result is **not** persisted.
        """
        raise NotImplementedError

    def ensure_computed(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        """Return the cached curves, computing them once if needed."""
        if self._datasets is None:
            self._datasets = self.compute(run, context=context)
        return self._datasets

    def invalidate(self) -> None:
        """Drop the transient computed arrays (e.g. after a recipe change).

        ``result_metadata`` is deliberately left intact: it is persisted state
        (saved diagnostics survive a failed recompute) and every successful
        :meth:`compute` path overwrites it.  Callers that discard a result
        outright (e.g. MaxEnt restart) clear it explicitly.
        """
        self._datasets = None

    def cache_datasets(self, datasets: list[MuonDataset]) -> None:
        """Store externally-computed curves as the transient cache.

        Used when a freshly generated result is already in hand (e.g. the GUI
        just computed it) to avoid recomputing immediately.
        """
        self._datasets = list(datasets)

    def datasets(self) -> list[MuonDataset]:
        """Return the currently cached curves (empty if not yet computed)."""
        return list(self._datasets) if self._datasets is not None else []

    @property
    def primary(self) -> MuonDataset | None:
        """Return the first cached curve, or ``None`` if not computed."""
        return self._datasets[0] if self._datasets else None

    # ── persistence (recipe + fit + trend only) ────────────────────────────

    def to_dict(self) -> dict:
        """Return the serialisable recipe/fit/trend state (no arrays)."""
        payload = {
            "rep_type": self.rep_type.value,
            "recipe": dict(self.recipe),
            "fit": self.fit.to_dict(),
            "trend_state": dict(self.trend_state),
            "result_metadata": dict(self.result_metadata),
        }
        # Only persist projections that actually carry a fit — never empty slots.
        projection_fits = {
            key: slot.to_dict() for key, slot in self.projection_fits.items() if not slot.is_empty()
        }
        if projection_fits:
            payload["projection_fits"] = projection_fits
        return payload

    def __repr__(self) -> str:
        computed = "uncomputed" if self._datasets is None else f"{len(self._datasets)} curve(s)"
        return f"{type(self).__name__}(recipe_keys={sorted(self.recipe)}, {computed})"
