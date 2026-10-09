"""Whether a longitudinal field has decoupled the zero-field relaxation.

A static field distribution of width Δ (µs⁻¹) relaxes the muon in zero field
and is decoupled by a longitudinal field B_L once γ_μB_L ≫ Δ: the polarisation
then stays along the field and the relaxation vanishes. A series fitted in a
field the survey found no Larmor line at (longitudinal, whatever the file's
stamp) whose fitted curves stay flat, beside a zero-field series whose curves
relax, is that experiment — and its result is that the zero-field fields are
static on the muon time scale, which an analyst should say rather than file the
pinned or unconstrained rates as a failed fit.

The reading compares the fitted curves, not a rate parameter, so it holds
whatever model each side was fitted with: a Kubo–Toyabe fitted in the field has
an unconstrained width, not a small one, when the field has decoupled it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

import numpy as np

from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
from asymmetry.core.workflow.recipe import FitRecipe

#: γ_μ in µs⁻¹ per gauss (angular), so Δ/γ_μ is a field in gauss.
_GAMMA_MU_PER_US_G = 2.0 * math.pi * field_gauss_to_frequency_mhz(1.0)

#: The time / µs both sides' fitted curves are compared at: inside every
#: source's window, and late enough for a nuclear width (0.1–0.5 µs⁻¹) to relax.
_HORIZON_US = 8.0

#: The zero-field curves relax when they lose at least this fraction of their
#: asymmetry by the horizon.
_RELAXES = 0.1

#: A field-run curve is flat when it loses less than this fraction of what the
#: zero-field curves lose.
_GONE = 0.1


#: Zero-field runs are compared only within the field series' temperature span,
#: widened by this fraction: hopping can narrow the zero-field relaxation at
#: temperatures the field series never visited.
_SPAN = 0.1


def _losses(series: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[float]:
    """Each row's fitted curve: the fraction of its t = 0 asymmetry lost by the horizon."""
    model = FitRecipe.from_dict(series["recipe"]).model()
    losses = []
    for row in rows:
        start, end = model.function(np.array([0.0, _HORIZON_US]), **row["parameters"])
        losses.append(float((start - end) / abs(start)))
    return losses


def _kinds(series: Mapping[str, Any]) -> dict[str, ParameterKind]:
    """What each parameter of a stored series' model measures."""
    return CompositeModel.from_expression(series["expression"]).parameter_kinds()


def _width(series: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> float | None:
    """The rows' largest median rate or width, when every row measures it."""
    measures = _kinds(series)
    widths = [
        statistics.median(row["parameters"][name] for row in rows)
        for name in series["free_params"]
        if measures.get(name) in (ParameterKind.RATE, ParameterKind.STATIC_WIDTH)
        and all(
            row["parameters"][name] > 3.0 * row["uncertainties"].get(name, math.inf) for row in rows
        )
    ]
    return max(widths) if widths else None


def decoupling_note(
    name: str, stored: Mapping[str, Mapping[str, Any]], survey_runs: Mapping[int, Mapping]
) -> str | None:
    """The note for fit series *name* when it is one side of a decoupled pair.

    *stored* maps every stored fit series (``WorkDir.fit_series``) by name, and
    *survey_runs* maps run numbers to the survey's per-run entries.
    """

    # A run's field is None when its file records none: neither side then.
    def longitudinal(series: Mapping[str, Any]) -> bool:
        return all(
            entry["field"] is not None and entry["field"] > 0.0 and entry["precession"] == "none"
            for entry in (survey_runs[row["run"]] for row in series["results"])
        )

    def zero_field(series: Mapping[str, Any]) -> bool:
        return all(survey_runs[row["run"]]["field"] == 0.0 for row in series["results"])

    def converged(series: Mapping[str, Any], low=-math.inf, high=math.inf):
        return [
            row
            for row in series["results"]
            if row["success"] and low <= survey_runs[row["run"]]["temperature"] <= high
        ]

    this = stored[name]
    if longitudinal(this):
        pairs = [(name, other) for other in stored if zero_field(stored[other])]
    elif zero_field(this):
        pairs = [(other, name) for other in stored if longitudinal(stored[other])]
    else:
        return None
    for lf_name, zf_name in pairs:
        lf, zf = stored[lf_name], stored[zf_name]
        lf_rows = converged(lf)
        temperatures = [survey_runs[row["run"]]["temperature"] for row in lf["results"]]
        low, high = min(temperatures), max(temperatures)
        zf_rows = converged(zf, (1 - _SPAN) * low, (1 + _SPAN) * high)
        if not lf_rows or not zf_rows:
            continue
        relaxed = statistics.median(_losses(zf, zf_rows))
        flat = {
            row["run"]: loss
            for row, loss in zip(lf_rows, _losses(lf, lf_rows), strict=True)
            if abs(loss) < _GONE * relaxed
        }
        if relaxed < _RELAXES or 2 * len(flat) <= len(lf_rows):
            continue
        field = min(float(survey_runs[run]["field"]) for run in flat)
        span = f"{low:g} K" if low == high else f"{low:g}-{high:g} K"
        width = _width(zf, zf_rows)
        scale = (
            ""
            if width is None
            else f" (its width, {width:.3g} µs⁻¹, is Δ/γ_μ = {width / _GAMMA_MU_PER_US_G:.2g} G)"
        )
        return (
            f"NOTE: at {span}, in a longitudinal field of {field:g} G and more (no line at its "
            f"Larmor frequency), the relaxation is gone: by {_HORIZON_US:g} µs the fitted curves "
            f"of {lf_name} lose at most {max(map(abs, flat.values())):.1%} of their asymmetry "
            f"on {len(flat)} of {len(lf_rows)} converged runs, against {relaxed:.1%} (median) "
            f"in the zero-field series {zf_name} at those temperatures{scale}. A longitudinal "
            f"field well above a static distribution's width decouples it: at these "
            f"temperatures the zero-field relaxation is from fields static on the muon time "
            f"scale, decoupled as a static distribution is. A rate pinned at zero or a width "
            f"left unconstrained here — whatever the model — is that result, not a failed fit: "
            f"say so."
            + (
                ""
                if ParameterKind.STATIC_WIDTH in _kinds(zf).values()
                else f" {zf_name} ({zf['expression']}) has no static-width term to measure "
                f"those fields with: refit the zero-field runs with a static Kubo-Toyabe "
                f"(--expression 'StaticGKT_ZF * Exponential + Constant') and report its "
                f"width, Delta, against temperature."
            )
        )
    return None


__all__ = ["decoupling_note"]
