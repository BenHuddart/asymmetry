"""Core representation abstraction for the Domain → Representation model.

A :class:`Representation` is a *recipe over a* :class:`~asymmetry.core.data.dataset.Run`
that yields one or more plottable :class:`~asymmetry.core.data.dataset.MuonDataset`
curves, plus the saved single fits and trend state for that view of the data.

Persistence is **recipe-only**: ``to_dict``/``from_dict`` serialise the
generation recipe, the saved single fits, and the trend state — never the
computed arrays.  The arrays live in the transient ``_datasets`` cache, repopulated by
:meth:`Representation.compute` on demand (e.g. after loading a project).

Each dataset owns up to four representations, one per
:class:`RepresentationType`.
"""

from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
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

#: Allowed fit provenance markers. Only ``"none"``, ``"single"`` and
#: ``"wizard"`` are ever written now that a slot holds the Single tab's fit
#: alone (D4); ``"batch"``/``"global"`` stay in the tuple so a pre-v20 file
#: still parses on its way through the migration.
FIT_PROVENANCE = ("none", "single", "batch", "global", "wizard")


@dataclass
class FitSlot:
    """One saved Single-tab fit on a ``(dataset, representation, projection)``.

    Per-run fit state is single fits only (series D4): a batch, global, grouped
    or scan run records its results on its
    :class:`~asymmetry.core.representation.series.FitSeries`, never on its
    members' slots. A slot lives in a :class:`SingleFitSet`, beside the other
    fits saved on the same data (docs/plans/single-fit-compare.md D4).

    ``model`` is a :meth:`CompositeModel.to_dict` payload (or ``None`` for an
    empty slot); ``result`` is a JSON-serialisable fit-result summary.
    ``fit_id`` is unique across the project (the plot keys curves by it);
    ``label`` is a user rename or a disambiguating suffix, ``None`` while the
    default name applies; ``fit_range`` is the window fitted, ``{"min", "max"}``
    in the domain's unit with ``None`` for an open side (a frequency fit over
    the full spectrum), and ``None`` as a whole for a fit saved before v25.
    """

    model: dict | None = None
    parameters: list[dict] = field(default_factory=list)
    result: dict | None = None
    provenance: str = "none"
    #: The fit panel's single-fit *form* payload (composite_model, parameters,
    #: result_html, wizard_state) for restoring the editor when this slot is
    #: re-selected.  It carries the GUI-only extras (result HTML, wizard cache)
    #: that ``model``/``parameters`` do not, so a per-projection single fit can
    #: be restored verbatim.  Empty for pre-this-change projects.
    ui_state: dict = field(default_factory=dict)
    fit_id: str = ""
    label: str | None = None
    fit_range: dict | None = None

    def is_empty(self) -> bool:
        """Return ``True`` when no model or result has been stored."""
        return self.model is None and self.result is None

    def identity(self) -> str:
        """What analysis this fit is, as an opaque comparable token (plan D3).

        Two fits are the same analysis exactly when their tokens are equal:
        the normalised model, the window, and each parameter's fixed flag,
        fixed value, link group and tie. Seeds and bounds are left out, so
        iterating on starting guesses re-fits the same analysis.
        """
        model = self.model
        if model is not None:
            from asymmetry.core.fitting.composite import CompositeModel

            model = CompositeModel.from_dict(model, allow_missing=True).to_dict()
        structure = sorted(
            (
                str(entry.get("name", "")),
                bool(entry.get("fixed", False)),
                float(entry.get("value", 0.0)) if entry.get("fixed", False) else None,
                json.dumps(entry.get("link_group"), sort_keys=True),
                json.dumps(entry.get("tie"), sort_keys=True),
            )
            for entry in self.parameters
        )
        return json.dumps(
            {"model": model, "parameters": structure, "fit_range": self.fit_range},
            sort_keys=True,
            default=str,
        )

    def fitted_values(self) -> dict[str, float]:
        """The values the fit ended on: its result's, else its parameter table's.

        A fit saved before results were structured (a v5-era slot holding a
        model and a result-HTML read-out, or a model alone) carries no
        ``result["parameters"]``; its form's table holds the values it showed.
        """
        result = self.result or {}
        if isinstance(result.get("parameters"), dict):
            return {str(name): float(value) for name, value in result["parameters"].items()}
        return {str(entry["name"]): float(entry["value"]) for entry in self.parameters}

    def reduced_chi_squared(self) -> float:
        """The fit's χ²ᵣ, NaN for a pre-structured-result slot that recorded none."""
        value = (self.result or {}).get("reduced_chi_squared")
        return math.nan if value is None else float(value)

    def data_key(self) -> tuple[float | None, float | None, int] | None:
        """``(min, max, n)`` of the data this fit saw, or ``None`` when unknown (plan D7).

        Two fits with equal keys were fitted to the same points, so an
        information criterion ranks them; ``n`` is ``ndof + npar``.
        """
        result = self.result or {}
        if self.fit_range is None or result.get("npar") is None or result.get("ndof") is None:
            return None
        return (
            self.fit_range["min"],
            self.fit_range["max"],
            int(result["ndof"]) + int(result["npar"]),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of the slot."""
        payload = {
            "model": None if self.model is None else dict(self.model),
            "parameters": [dict(p) for p in self.parameters],
            "result": None if self.result is None else dict(self.result),
            "provenance": self.provenance,
            "fit_id": self.fit_id,
            "label": self.label,
            "fit_range": None if self.fit_range is None else dict(self.fit_range),
        }
        # Only persist ``ui_state`` when populated — pre-this-change slots carry
        # none, and an empty dict would bloat every saved slot for no gain.
        if self.ui_state:
            payload["ui_state"] = dict(self.ui_state)
        return payload

    @classmethod
    def from_dict(cls, data: dict | None) -> FitSlot:
        """Reconstruct a :class:`FitSlot` from serialised data.

        ``batch_id``, ``diverged`` and ``include_in_trend`` written by a pre-v20
        project are ignored: the v19->v20 migration moves what they meant onto
        the series, and a file hand-edited back to the old shape must not
        resurrect them.
        """
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
        # Migrate pre-rework parameter entries — legacy ``fraction_<k>`` names to
        # the n-1 free-fraction scheme, legacy per-factor amplitudes onto the
        # one-scale-per-product policy — so old projects load, display, and refit.
        # Guarded: a missing/malformed model payload simply skips migration. Both
        # migrations carry their own cheap precondition and are no-ops for data
        # saved under the current schemes.
        if isinstance(model, dict) and parameters:
            from asymmetry.core.fitting.composite import (
                CompositeModel,
                migrate_legacy_fraction_parameter_entries,
            )
            from asymmetry.core.fitting.legacy_product_amplitudes import (
                fold_legacy_product_amplitude_entries,
            )

            try:
                composite = CompositeModel.from_dict(model, allow_missing=True)
            except (ValueError, KeyError, TypeError):
                composite = None
            if composite is not None:
                parameters = migrate_legacy_fraction_parameter_entries(composite, parameters)
                parameters = fold_legacy_product_amplitude_entries(composite, parameters)
        result = data.get("result")
        raw_ui_state = data.get("ui_state")
        fit_range = data.get("fit_range")
        return cls(
            model=dict(model) if isinstance(model, dict) else None,
            parameters=parameters,
            result=dict(result) if isinstance(result, dict) else None,
            provenance=provenance,
            ui_state=dict(raw_ui_state) if isinstance(raw_ui_state, dict) else {},
            fit_id=str(data.get("fit_id") or new_fit_id()),
            label=str(data["label"]) if data.get("label") else None,
            fit_range=(
                {side: _optional_float(fit_range.get(side)) for side in ("min", "max")}
                if isinstance(fit_range, dict)
                else None
            ),
        )


def _optional_float(value: object) -> float | None:
    """*value* as a float, or ``None`` for an open window side."""
    return None if value is None else float(value)


def new_fit_id() -> str:
    """A fresh project-unique single-fit id."""
    return f"fit-{uuid.uuid4().hex[:12]}"


@dataclass
class SingleFitSet:
    """The single fits saved on one ``(representation, projection)``, and the open one.

    ``open_id`` names one of ``fits`` exactly when ``fits`` is non-empty: every
    mutator below keeps that true, so :meth:`open_fit` never searches in vain.
    """

    fits: list[FitSlot] = field(default_factory=list)
    open_id: str | None = None

    def open_fit(self) -> FitSlot:
        """The open fit, or an empty slot when nothing is saved."""
        return self.get(self.open_id) if self.open_id is not None else FitSlot()

    def get(self, fit_id: str) -> FitSlot:
        """The saved fit *fit_id*; ``KeyError`` when it is not in this set."""
        for slot in self.fits:
            if slot.fit_id == fit_id:
                return slot
        raise KeyError(f"no saved fit {fit_id!r} in this set")

    def record(
        self, slot: FitSlot, *, detached: bool, default_name: Callable[[FitSlot], str]
    ) -> FitSlot:
        """Save a completed fit by the matching rule of plan D1, open it and return it.

        Not *detached*: a fit with the same :meth:`FitSlot.identity` as the open
        fit replaces it in place, else the newest fit sharing it is replaced.
        Otherwise (or when *detached*) the fit is added under a fresh id, with a
        label only when its default name would read like another fit's.
        """
        identity = slot.identity()
        matches = [] if detached else [s for s in self.fits if s.identity() == identity]
        target = next((s for s in matches if s.fit_id == self.open_id), None)
        if target is None and matches:
            target = matches[-1]
        if target is not None:
            slot.fit_id, slot.label = target.fit_id, target.label
            self.fits[self.fits.index(target)] = slot
        else:
            from asymmetry.core.representation.naming import disambiguate_series_label

            slot.fit_id = new_fit_id()
            default = default_name(slot)
            shown = [s.label or default_name(s) for s in self.fits]
            distinct = disambiguate_series_label(default, shown)
            slot.label = None if distinct == default else distinct
            self.fits.append(slot)
        self.open_id = slot.fit_id
        return slot

    def replace_open(self, slot: FitSlot) -> None:
        """Store *slot* as the open fit's content, keeping its id and label.

        With nothing saved yet, *slot* becomes the set's only fit.
        """
        if self.open_id is None:
            slot.fit_id = slot.fit_id or new_fit_id()
            self.fits.append(slot)
        else:
            current = self.open_fit()
            slot.fit_id, slot.label = current.fit_id, current.label
            self.fits[self.fits.index(current)] = slot
        self.open_id = slot.fit_id

    def open(self, fit_id: str) -> None:
        """Make the saved fit *fit_id* the open one."""
        self.open_id = self.get(fit_id).fit_id

    def rename(self, fit_id: str, label: str) -> None:
        """Name *fit_id* ``label``; a blank label restores the default name."""
        self.get(fit_id).label = label.strip() or None

    def delete(self, fit_id: str) -> None:
        """Drop *fit_id*; deleting the open fit opens the newest remaining one."""
        self.fits.remove(self.get(fit_id))
        if self.open_id == fit_id:
            self.open_id = self.fits[-1].fit_id if self.fits else None

    def to_dict(self) -> dict:
        return {"open_id": self.open_id, "fits": [slot.to_dict() for slot in self.fits]}

    @classmethod
    def from_dict(cls, data: dict) -> SingleFitSet:
        """Read a saved set; an ``open_id`` naming no fit opens the newest one."""
        fits = [FitSlot.from_dict(entry) for entry in data.get("fits") or []]
        fits = [slot for slot in fits if not slot.is_empty()]
        ids = [slot.fit_id for slot in fits]
        open_id = data.get("open_id")
        if open_id not in ids:
            open_id = ids[-1] if ids else None
        return cls(fits=fits, open_id=open_id)


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
        trend_state: dict | None = None,
        result_metadata: dict | None = None,
        single_fits: dict[str | None, SingleFitSet] | None = None,
    ) -> None:
        self.recipe: dict[str, Any] = dict(recipe or {})
        #: The saved single fits per projection: ``None`` keys the default
        #: (non-projection) set, a label (P_x/P_y/P_z, transverse-field labels,
        #: …) a vector grouping's projection.
        self.single_fits: dict[str | None, SingleFitSet] = {
            self._fit_key(key): fit_set for key, fit_set in (single_fits or {}).items()
        }
        self.trend_state: dict[str, Any] = dict(trend_state or {})
        self.result_metadata: dict[str, Any] = dict(result_metadata or {})
        self._datasets: list[MuonDataset] | None = None

    # ── saved single fits (per projection) ─────────────────────────────────

    @staticmethod
    def _fit_key(projection: str | None) -> str | None:
        """Normalise a projection label to a single-fit set key.

        Falsy labels and the ``"ALL"`` aggregate sentinel (which is not a
        physical projection and is never fit) map to ``None`` — the default
        set — so they never create a phantom projection entry.
        """
        if not projection or projection == "ALL":
            return None
        return str(projection)

    def fit_set(self, projection: str | None) -> SingleFitSet:
        """The saved fits on *projection*; an empty set, not stored, when it has none."""
        return self.single_fits.get(self._fit_key(projection), SingleFitSet())

    def _stored_fit_set(self, projection: str | None) -> SingleFitSet:
        """The saved fits on *projection*, stored on first use so a write lands."""
        return self.single_fits.setdefault(self._fit_key(projection), SingleFitSet())

    def fit_for(self, projection: str | None) -> FitSlot:
        """Return the open fit on *projection* (a fresh empty slot if unfit).

        This is a pure read — it never inserts, so inspecting an unfit
        projection does not leak an empty set into the saved project.
        """
        fit_set = self.single_fits.get(self._fit_key(projection))
        return fit_set.open_fit() if fit_set is not None else FitSlot()

    def set_fit_for(self, projection: str | None, slot: FitSlot) -> None:
        """Store *slot* as the open fit on *projection*; an empty slot clears its fits."""
        key = self._fit_key(projection)
        if slot.is_empty():
            self.single_fits.pop(key, None)
        else:
            self._stored_fit_set(key).replace_open(slot)

    @property
    def fit(self) -> FitSlot:
        """The open fit on the default (non-projection) set."""
        return self.fit_for(None)

    @fit.setter
    def fit(self, slot: FitSlot) -> None:
        self.set_fit_for(None, slot)

    def record_single_fit(
        self, projection: str | None, slot: FitSlot, *, detached: bool
    ) -> FitSlot:
        """Save a completed single fit on *projection* (plan D1) and return it, opened."""
        return self._stored_fit_set(projection).record(
            slot, detached=detached, default_name=self.default_fit_name
        )

    def default_fit_name(self, slot: FitSlot) -> str:
        """``"<model> · <window>"``, the name a fit reads under until renamed (plan D5)."""
        from asymmetry.core.representation.naming import default_single_fit_label

        return default_single_fit_label(slot, self.domain)

    def fit_name(self, slot: FitSlot) -> str:
        """The name *slot* reads under: its label, else its default name."""
        return slot.label or self.default_fit_name(slot)

    def has_projection_fits(self) -> bool:
        """Whether any genuine projection holds a saved fit."""
        return any(key is not None and fit_set.fits for key, fit_set in self.single_fits.items())

    def iter_fit_slots(self) -> list[tuple[str | None, FitSlot]]:
        """Return ``(projection_key, slot)`` for every saved fit, default set first."""
        keys = sorted(self.single_fits, key=lambda key: (key is not None, key or ""))
        return [(key, slot) for key in keys for slot in self.single_fits[key].fits]

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
        return {
            "rep_type": self.rep_type.value,
            "recipe": dict(self.recipe),
            "trend_state": dict(self.trend_state),
            "result_metadata": dict(self.result_metadata),
            # JSON keys are strings: the default set is written under "".
            "single_fits": {
                key or "": fit_set.to_dict()
                for key, fit_set in self.single_fits.items()
                if fit_set.fits
            },
        }

    def __repr__(self) -> str:
        computed = "uncomputed" if self._datasets is None else f"{len(self._datasets)} curve(s)"
        return f"{type(self).__name__}(recipe_keys={sorted(self.recipe)}, {computed})"
