"""Session handles for cached fit-wizard recommendations.

Why this module exists
----------------------

A fit-wizard recommendation is a *large* analysis result: for every candidate
it assessed it holds a dense fitted curve, its component curves and the fit
residuals at the resolution of the analysed record. On a 90k-bin run a single
recommendation is ~225 MB once serialised to JSON.

The fit tabs' ``get_state()``/``restore_state()`` pair is a **session**
snapshot mechanism, not only a persistence one: the fit panel calls it on
every run switch (saving the leaving run's form, restoring the arriving run's)
and ``copy.deepcopy``s the resulting dicts several times per switch. Embedding
a *serialised* recommendation in that dict made each switch pay two
serialisations, a deserialisation and a couple of deep copies of a
hundreds-of-MB payload — 3–5 s per switch, plus the GC pauses its garbage
triggered.

So the session form carries the recommendation **by reference**, in the
immutable handles below: ``FitWizardRecommendation`` /
``GlobalFitWizardRecommendation`` and everything they contain are frozen
dataclasses (plus read-only numpy arrays in practice — nothing mutates them
after the analysis worker returns), so sharing one between several state
snapshots is safe, and ``__deepcopy__``/``__copy__`` returning ``self`` keeps
the panel's existing ``copy.deepcopy(state)`` calls O(form size).

Serialisation happens only at persistence boundaries — a project file or a
:class:`~asymmetry.core.representation.base.FitSlot`'s ``ui_state`` — through
:meth:`WizardCacheEntry.to_persisted`, which uses the *compact* form (curves
strided to a bounded point count, residual series dropped; see
``core.fitting.fit_wizard.PERSISTED_CURVE_MAX_POINTS``).

Both directions accept either shape, so a project file written before this
change (a full-resolution dict) still restores, and a state dict that has
already been through a persistence boundary can be restored again unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TypeVar

from asymmetry.core.fitting.fit_wizard import (
    FitWizardRecommendation,
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalFitWizardRecommendation,
    deserialize_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)

#: A form state passes through unchanged when it is not a dict (``None`` for an
#: unset slot), so the helpers are typed as identity on the input type.
_StateT = TypeVar("_StateT")

__all__ = [
    "GlobalWizardCacheEntry",
    "WizardCacheEntry",
    "global_wizard_cache_entry",
    "persisted_global_fit_form_state",
    "persisted_global_wizard_state",
    "persisted_single_fit_form_state",
    "persisted_wizard_state",
    "wizard_cache_entry",
]


# ``eq=False`` (identity equality): the generated ``__eq__`` would compare the
# recommendations field-by-field, and those carry numpy arrays — a comparison
# of two different caches would raise "truth value of an array is ambiguous"
# in whatever code happened to compare two state dicts. Identity is also the
# only equality this handle means: one cached analysis, shared.
@dataclass(frozen=True, eq=False)
class WizardCacheEntry:
    """A single-fit wizard recommendation held by reference in session state.

    ``signature`` is copied on construction and handed back copied
    (:meth:`signature_copy`) so callers can neither see nor cause mutation
    through the shared handle; ``recommendation`` is a frozen dataclass tree
    and is shared as-is.
    """

    recommendation: FitWizardRecommendation
    signature: dict[str, object] = field(default_factory=dict)
    log_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", copy.deepcopy(dict(self.signature)))
        object.__setattr__(self, "log_text", str(self.log_text))

    # Immutable: copying the handle is copying the reference. This is what
    # keeps FitPanel's per-switch ``copy.deepcopy(state)`` calls cheap.
    def __deepcopy__(self, memo: dict) -> WizardCacheEntry:
        return self

    def __copy__(self) -> WizardCacheEntry:
        return self

    def signature_copy(self) -> dict[str, object]:
        """Return a private copy of the signature this cache was built for."""
        return copy.deepcopy(self.signature)

    def to_persisted(self) -> dict[str, object]:
        """Return the JSON-safe, compact payload for a project file / slot."""
        return {
            "signature": copy.deepcopy(self.signature),
            "recommendation": serialize_fit_wizard_recommendation(
                self.recommendation, compact=True
            ),
            "log_text": self.log_text,
        }

    @classmethod
    def from_persisted(cls, payload: object) -> WizardCacheEntry | None:
        """Rebuild an entry from a persisted ``wizard_state`` dict (or ``None``)."""
        if not isinstance(payload, dict):
            return None
        recommendation = deserialize_fit_wizard_recommendation(payload.get("recommendation"))
        signature = payload.get("signature")
        if recommendation is None or not isinstance(signature, dict):
            return None
        return cls(
            recommendation=recommendation,
            signature=signature,
            log_text=str(payload.get("log_text", "")),
        )


@dataclass(frozen=True, eq=False)
class GlobalWizardCacheEntry:
    """A global-fit wizard recommendation held by reference in session state.

    ``run_numbers`` is the run set the cache is keyed under in the global tab's
    per-run-set store; it is empty for the tab's single "active" cache block.
    """

    recommendation: GlobalFitWizardRecommendation
    signature: dict[str, object] = field(default_factory=dict)
    log_text: str = ""
    run_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", copy.deepcopy(dict(self.signature)))
        object.__setattr__(self, "log_text", str(self.log_text))
        object.__setattr__(self, "run_numbers", tuple(int(run) for run in self.run_numbers))

    def __deepcopy__(self, memo: dict) -> GlobalWizardCacheEntry:
        return self

    def __copy__(self) -> GlobalWizardCacheEntry:
        return self

    def signature_copy(self) -> dict[str, object]:
        """Return a private copy of the signature this cache was built for."""
        return copy.deepcopy(self.signature)

    def to_persisted(self) -> dict[str, object]:
        """Return the JSON-safe, compact payload for a project file / slot."""
        payload: dict[str, object] = {
            "signature": copy.deepcopy(self.signature),
            "recommendation": serialize_global_fit_wizard_recommendation(
                self.recommendation, compact=True
            ),
            "log_text": self.log_text,
        }
        if self.run_numbers:
            payload["run_numbers"] = list(self.run_numbers)
        return payload

    @classmethod
    def from_persisted(cls, payload: object) -> GlobalWizardCacheEntry | None:
        """Rebuild an entry from a persisted global ``wizard_state`` dict."""
        if not isinstance(payload, dict):
            return None
        recommendation = deserialize_global_fit_wizard_recommendation(payload.get("recommendation"))
        signature = payload.get("signature")
        if recommendation is None or not isinstance(signature, dict):
            return None
        raw_runs = payload.get("run_numbers")
        if not isinstance(raw_runs, tuple | list):
            raw_runs = signature.get("run_numbers")
        run_numbers: tuple[int, ...] = ()
        if isinstance(raw_runs, tuple | list):
            coerced: list[int] = []
            for run in raw_runs:
                try:
                    coerced.append(int(run))
                except (TypeError, ValueError):
                    continue
            run_numbers = tuple(coerced)
        return cls(
            recommendation=recommendation,
            signature=signature,
            log_text=str(payload.get("log_text", "")),
            run_numbers=run_numbers,
        )


def wizard_cache_entry(value: object) -> WizardCacheEntry | None:
    """Coerce a session handle *or* a persisted dict to a :class:`WizardCacheEntry`."""
    if isinstance(value, WizardCacheEntry):
        return value
    return WizardCacheEntry.from_persisted(value)


def global_wizard_cache_entry(value: object) -> GlobalWizardCacheEntry | None:
    """Coerce a session handle *or* a persisted dict to a global cache entry."""
    if isinstance(value, GlobalWizardCacheEntry):
        return value
    return GlobalWizardCacheEntry.from_persisted(value)


def persisted_wizard_state(value: object) -> dict[str, object] | None:
    """Return the persistable form of a single-fit ``wizard_state`` block.

    Accepts a session handle (serialised compactly here) or an
    already-persisted dict (returned unchanged); anything else yields ``None``,
    so a caller can simply drop the key.
    """
    if isinstance(value, WizardCacheEntry):
        return value.to_persisted()
    if isinstance(value, dict):
        return value
    return None


def persisted_global_wizard_state(value: object) -> dict[str, object] | None:
    """Return the persistable form of a global ``wizard_state`` block."""
    if isinstance(value, GlobalWizardCacheEntry):
        return value.to_persisted()
    if isinstance(value, dict):
        return value
    return None


def persisted_single_fit_form_state(state: _StateT) -> _StateT:
    """Return an independent, JSON-safe copy of a single-fit form state.

    The copy is cheap even when the form carries a wizard cache: deep-copying
    the handle returns the handle, and only the final conversion serialises
    (compactly). Use it wherever a form payload leaves the session — a fit
    slot's ``ui_state``, project state — never on the run-switch path.
    """
    if not isinstance(state, dict):
        return state
    persisted = copy.deepcopy(state)
    if "wizard_state" in persisted:
        wizard_state = persisted_wizard_state(persisted["wizard_state"])
        if wizard_state is None:
            persisted.pop("wizard_state", None)
        else:
            persisted["wizard_state"] = wizard_state
    return persisted


def persisted_global_fit_form_state(state: _StateT) -> _StateT:
    """Return an independent, JSON-safe copy of a global/grouped form state.

    The global tab keeps both a single active ``wizard_state`` block and a
    per-run-set ``wizard_state_by_run_set`` list; both hold session handles and
    are converted here.
    """
    if not isinstance(state, dict):
        return state
    persisted = copy.deepcopy(state)
    if "wizard_state" in persisted:
        wizard_state = persisted_global_wizard_state(persisted["wizard_state"])
        if wizard_state is None:
            persisted.pop("wizard_state", None)
        else:
            persisted["wizard_state"] = wizard_state
    raw_store = persisted.get("wizard_state_by_run_set")
    if isinstance(raw_store, list):
        persisted["wizard_state_by_run_set"] = [
            entry for raw in raw_store if (entry := persisted_global_wizard_state(raw)) is not None
        ]
    return persisted
