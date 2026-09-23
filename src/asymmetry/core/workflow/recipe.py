"""The fit recipe: the only contract between screening and fitting.

A recipe is a small JSON document naming a model and everything a fit needs to
start from it — the component expression, the serialised
:class:`~asymmetry.core.fitting.composite.CompositeModel`, one entry per
parameter (start value, bounds, fixed flag), the time window, a rebin factor,
and where the recipe came from.

``wizard`` writes one, ``fit`` and ``fit-series`` consume one, and a caller that
wants to change something edits the recipe rather than retyping a parameter
table. That is the whole point of the format: a scripted analysis never has to
express a model as anything but this document.

Two ways in: :meth:`FitRecipe.from_assessment`, which turns a fit-wizard
candidate into a recipe whose starting values are that candidate's *fitted*
values, and :meth:`FitRecipe.from_expression`, which seeds a fresh expression
through :func:`~asymmetry.core.fitting.seeding.seed_parameters` — the same one
seeding function every fit surface asks.

A recipe is portable across a scan
----------------------------------

Whichever way it was built, a recipe's **bounds are the model's static
defaults** — what :func:`~asymmetry.core.fitting.seeding.seed_parameters`
returns for a bare :class:`~asymmetry.core.fitting.seeding.SeedContext`, and so
exactly what a user gets typing the same expression into the desktop
application. The fit wizard's own bounds are deliberately *not* carried: every
one of them (``bounds_overrides`` in
:mod:`~asymmetry.core.fitting.fit_wizard` — the window around a detected
spectral line, 0.5–2× a seeded ``Delta``/``A_hf``/``r_muF``, the Nyquist and
duration caps) describes *the one run the wizard screened*. Carrying a
frequency window measured at 300 K into a fit at 380 K clamps the fit at a
bound instead of letting it follow the physics, which is the opposite of what a
recipe is for.

Values are a different matter: a wizard fit's numbers are the best available
starting point, so they *are* carried. The values that describe a run rather
than the physics — an applied field, a spectral peak — are re-seeded per run by
:func:`~asymmetry.core.workflow.series.fit_series`; see that module.

``rebin`` is a **fit** setting: the factor the recipe's consumers merge value
bins by before fitting (1 = the reduced record as stored). It is unrelated to
the wizard's own internal analysis rebinning, which is recorded separately in
the screening payload.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclasses_field
from typing import Any

from asymmetry import __version__
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.seeding import SeedContext, seed_parameters

#: Schema version stamped into every recipe document.
SCHEMA = 1


def _bound_to_json(value: float) -> float | None:
    """Render a bound for JSON: an infinite bound is ``null``, not ``Infinity``.

    ``json.dumps`` happily writes the non-standard ``Infinity`` token, which
    Python reads back but every other JSON consumer rejects. A recipe is meant
    to be read (and edited) by tools that are not this one.
    """
    return None if math.isinf(value) else float(value)


@dataclass(frozen=True)
class RecipeParameter:
    """One parameter's starting value, bounds and fixed flag."""

    name: str
    value: float
    min: float = -math.inf
    max: float = math.inf
    fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        return {
            "name": self.name,
            "value": float(self.value),
            "min": _bound_to_json(self.min),
            "max": _bound_to_json(self.max),
            "fixed": bool(self.fixed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipeParameter:
        """Reconstruct a parameter from :meth:`to_dict` output."""
        return cls(
            name=str(data["name"]),
            value=float(data["value"]),
            min=-math.inf if data["min"] is None else float(data["min"]),
            max=math.inf if data["max"] is None else float(data["max"]),
            fixed=bool(data["fixed"]),
        )

    def to_parameter(self) -> Parameter:
        """The :class:`Parameter` a fit starts from."""
        return Parameter(
            name=self.name,
            value=float(self.value),
            min=float(self.min),
            max=float(self.max),
            fixed=bool(self.fixed),
        )


@dataclass(frozen=True)
class FitRecipe:
    """A model plus its starting parameters, window and provenance."""

    expression: str
    #: :meth:`CompositeModel.to_dict` output — the model's serialised form, which
    #: is what a hand-edited recipe carries, so it is the source of truth and
    #: :meth:`model` rebuilds from it.
    model_payload: dict[str, Any]
    parameters: tuple[RecipeParameter, ...]
    t_min: float | None = None
    t_max: float | None = None
    rebin: int = 1
    #: Where this recipe came from: ``{"wizard_run": N, "template_key": "..."}``
    #: for a screened one, ``{"user": True}`` for one built from an expression.
    source: dict[str, Any] = dataclasses_field(default_factory=lambda: {"user": True})
    #: Parameters a *person* pinned — ``--fix``, ``--global``, or an edit to this
    #: file. A series fit re-seeds run-bound parameters (an applied field, a
    #: spectral peak) from each run's own record; one named here is never moved,
    #: because the value was chosen rather than measured. Distinct from
    #: :attr:`RecipeParameter.fixed`, which also covers a parameter the model or
    #: the wizard holds by default and which *is* re-seeded per run.
    pinned: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.rebin < 1:
            raise ValueError(f"Rebin factor must be at least 1, got {self.rebin}.")
        if self.t_min is not None and self.t_max is not None and self.t_min >= self.t_max:
            raise ValueError(f"Time window t_min={self.t_min} is not below t_max={self.t_max}.")

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        return {
            "schema": SCHEMA,
            "asymmetry_version": __version__,
            "expression": self.expression,
            "model": dict(self.model_payload),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "t_min": None if self.t_min is None else float(self.t_min),
            "t_max": None if self.t_max is None else float(self.t_max),
            "rebin": int(self.rebin),
            "source": dict(self.source),
            "pinned": list(self.pinned),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitRecipe:
        """Reconstruct a recipe from :meth:`to_dict` output."""
        return cls(
            expression=str(data["expression"]),
            model_payload=dict(data["model"]),
            parameters=tuple(RecipeParameter.from_dict(entry) for entry in data["parameters"]),
            t_min=None if data["t_min"] is None else float(data["t_min"]),
            t_max=None if data["t_max"] is None else float(data["t_max"]),
            rebin=int(data["rebin"]),
            source=dict(data["source"]),
            pinned=tuple(str(name) for name in data["pinned"]),
        )

    # -- what a fit needs ---------------------------------------------------

    def model(self) -> CompositeModel:
        """The composite model this recipe fits."""
        return CompositeModel.from_dict(self.model_payload)

    def parameter_set(self) -> ParameterSet:
        """A fresh :class:`ParameterSet` seeded from this recipe.

        A new set every call: a fit mutates the parameters it is given, and a
        series fits one run per set.
        """
        return ParameterSet([parameter.to_parameter() for parameter in self.parameters])

    @property
    def parameter_names(self) -> list[str]:
        """Every parameter name, in model order."""
        return [parameter.name for parameter in self.parameters]

    def free_parameter_names(self) -> list[str]:
        """The parameters a fit from this recipe would vary, in model order."""
        return [parameter.name for parameter in self.parameters if not parameter.fixed]

    # -- editing ------------------------------------------------------------

    def with_overrides(
        self,
        *,
        fix: Mapping[str, float] | None = None,
        free: Iterable[str] | None = None,
        initial: Mapping[str, float] | None = None,
    ) -> FitRecipe:
        """A copy with parameters pinned at a value, released, or restarted.

        ``fix`` maps a parameter name to the value it is held at; ``free``
        names parameters to release; ``initial`` moves a starting value
        without holding it. A name neither the model nor the recipe carries
        raises :class:`KeyError` naming it — a typo must not silently do
        nothing to the fit.

        A fixed name is recorded in :attr:`pinned`: the value came from a
        person, so a series fit must never re-seed it from a run's own record.
        Releasing a parameter un-pins it.
        """
        fix = dict(fix or {})
        free = list(free or [])
        initial = dict(initial or {})
        known = set(self.parameter_names)
        unknown = sorted((set(fix) | set(free) | set(initial)) - known)
        if unknown:
            raise KeyError(
                f"{', '.join(unknown)} is not a parameter of {self.expression!r} "
                f"(it has {', '.join(self.parameter_names)})."
            )

        rebuilt: list[RecipeParameter] = []
        for parameter in self.parameters:
            if parameter.name in initial:
                parameter = replace(parameter, value=float(initial[parameter.name]))
            if parameter.name in fix:
                parameter = replace(parameter, value=float(fix[parameter.name]), fixed=True)
            if parameter.name in free:
                parameter = replace(parameter, fixed=False)
            rebuilt.append(parameter)
        pinned = (set(self.pinned) | set(fix)) - set(free)
        return replace(
            self,
            parameters=tuple(rebuilt),
            pinned=tuple(name for name in self.parameter_names if name in pinned),
        )

    def with_window(self, *, t_min: float | None, t_max: float | None) -> FitRecipe:
        """A copy fitted over a different time window."""
        return replace(self, t_min=t_min, t_max=t_max)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_assessment(cls, assessment: Any, *, run_number: int) -> FitRecipe:
        """Build a recipe from a fit-wizard :class:`CandidateAssessment`.

        The candidate's *fitted* values become the starting values and its
        fixed flags carry over, so a fit from this recipe restarts where the
        wizard's own fit of that template finished. Its **bounds do not**: they
        are the model's static defaults instead (see "A recipe is portable
        across a scan" at the top of this module). Every bound the wizard sets
        is measured from the single run it screened — a window around that
        run's detected line, a multiple of that run's seeded width — so
        carrying one would pin the fit of every *other* run in a scan at a
        bound rather than letting it follow the physics.
        """
        model = assessment.template.model
        defaults = seed_parameters(model, SeedContext())
        fitted = assessment.fit_result.parameters
        return cls(
            expression=model.component_expression_string(),
            model_payload=model.to_dict(),
            parameters=tuple(
                RecipeParameter(
                    name=name,
                    value=float(fitted[name].value),
                    min=defaults[name].min,
                    max=defaults[name].max,
                    fixed=bool(fitted[name].fixed),
                )
                for name in model.param_names
            ),
            source={
                "wizard_run": int(run_number),
                "template_key": str(assessment.template.key),
            },
        )

    @classmethod
    def from_expression(
        cls,
        expression: str,
        *,
        dataset: MuonDataset | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        rebin: int = 1,
    ) -> FitRecipe:
        """Build a recipe for a component expression, bypassing the wizard.

        Starting values, bounds and fixed flags come from
        :func:`~asymmetry.core.fitting.seeding.seed_parameters`, the one
        function every fit surface seeds through. Given a *dataset* the seeds
        additionally read that record's own amplitude/background scale and its
        applied field; without one they are the components' static defaults.
        Only the *values* differ between those two cases — the seeding layers
        above the first set values, never bounds — so a recipe's bounds are the
        model's static defaults either way.
        """
        model = CompositeModel.from_expression(expression)
        context = SeedContext(
            dataset=dataset,
            field_gauss=None if dataset is None else dataset.field,
        )
        seeds = seed_parameters(model, context)
        return cls(
            expression=model.component_expression_string(),
            model_payload=model.to_dict(),
            parameters=tuple(
                RecipeParameter(
                    name=name,
                    value=seeds[name].value,
                    min=seeds[name].min,
                    max=seeds[name].max,
                    fixed=seeds[name].fixed,
                )
                for name in model.param_names
            ),
            t_min=t_min,
            t_max=t_max,
            rebin=rebin,
            source={"user": True},
        )


__all__ = ["SCHEMA", "FitRecipe", "RecipeParameter"]
