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

    def is_empty(self) -> bool:
        """Return ``True`` when no model or result has been stored."""
        return self.model is None and self.result is None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of the slot."""
        return {
            "model": None if self.model is None else dict(self.model),
            "parameters": [dict(p) for p in self.parameters],
            "result": None if self.result is None else dict(self.result),
            "provenance": self.provenance,
            "batch_id": self.batch_id,
            "diverged": bool(self.diverged),
            "include_in_trend": bool(self.include_in_trend),
        }

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
        result = data.get("result")
        return cls(
            model=dict(model) if isinstance(model, dict) else None,
            parameters=parameters,
            result=dict(result) if isinstance(result, dict) else None,
            provenance=provenance,
            batch_id=(str(data["batch_id"]) if data.get("batch_id") is not None else None),
            diverged=bool(data.get("diverged", False)),
            include_in_trend=bool(data.get("include_in_trend", True)),
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
        # Fits keyed by projection label; the ``None`` key is the default
        # (non-projection / single-pair) slot, exposed as the ``fit`` property so
        # every existing ``representation.fit`` read/write keeps operating on it.
        # Per-projection slots (P_x/P_y/P_z, transverse-field labels, …) let each
        # projection of a vector grouping remember its own fit.
        self._fits: dict[str | None, FitSlot] = {
            None: fit if isinstance(fit, FitSlot) else FitSlot()
        }
        if isinstance(projection_fits, dict):
            for key, slot in projection_fits.items():
                if key is not None and isinstance(slot, FitSlot):
                    self._fits[str(key)] = slot
        self.trend_state: dict[str, Any] = dict(trend_state or {})
        self.result_metadata: dict[str, Any] = dict(result_metadata or {})
        self._datasets: list[MuonDataset] | None = None

    # ── fit slots (per projection) ─────────────────────────────────────────

    @property
    def fit(self) -> FitSlot:
        """The default (non-projection) fit slot."""
        return self._fits[None]

    @fit.setter
    def fit(self, slot: FitSlot) -> None:
        self._fits[None] = slot if isinstance(slot, FitSlot) else FitSlot()

    @staticmethod
    def _fit_key(projection: str | None) -> str | None:
        """Normalise a projection label to a fit-slot key (empty → default)."""
        return projection if projection else None

    def fit_for(self, projection: str | None) -> FitSlot:
        """Return the fit slot for *projection*, creating an empty one if absent."""
        return self._fits.setdefault(self._fit_key(projection), FitSlot())

    def set_fit_for(self, projection: str | None, slot: FitSlot) -> None:
        """Store *slot* as the fit for *projection*."""
        self._fits[self._fit_key(projection)] = slot if isinstance(slot, FitSlot) else FitSlot()

    @property
    def projection_fits(self) -> dict[str, FitSlot]:
        """Return the per-projection (non-default) fit slots, keyed by label."""
        return {key: slot for key, slot in self._fits.items() if key is not None}

    def iter_fit_slots(self) -> list[tuple[str | None, FitSlot]]:
        """Return ``(projection_key, slot)`` for every stored fit (default + projections)."""
        return list(self._fits.items())

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
        projection_fits = self.projection_fits
        if projection_fits:
            payload["projection_fits"] = {
                key: slot.to_dict() for key, slot in projection_fits.items()
            }
        return payload

    def __repr__(self) -> str:
        computed = "uncomputed" if self._datasets is None else f"{len(self._datasets)} curve(s)"
        return f"{type(self).__name__}(recipe_keys={sorted(self.recipe)}, {computed})"
