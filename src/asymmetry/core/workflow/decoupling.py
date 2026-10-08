"""Whether a longitudinal field has decoupled the zero-field relaxation.

A static field distribution of width Δ (µs⁻¹) relaxes the muon in zero field
and is decoupled by a longitudinal field B_L once γ_μB_L ≫ Δ: the polarisation
then stays along the field and the relaxation vanishes. A series fitted in a
field the survey found no Larmor line at (longitudinal, whatever the file's
stamp) whose rates have collapsed, beside a zero-field series that relaxes, is
that experiment — and its result is that the zero-field fields are static on
the muon time scale, which an analyst should say rather than file the pinned
rates as a failed fit.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz

#: γ_μ in µs⁻¹ per gauss (angular), so Δ/γ_μ is a field in gauss.
_GAMMA_MU_PER_US_G = 2.0 * math.pi * field_gauss_to_frequency_mhz(1.0)

#: A longitudinal field this many times Δ/γ_μ decouples a static distribution
#: (the Kubo–Toyabe dip and tail are gone well before ten).
_DECOUPLING_FIELDS = 10.0

#: A field-run rate counts as gone when, with two errors added, it stays below
#: this fraction of the zero-field width.
_GONE = 0.1


def _rates(series: Mapping[str, Any]) -> list[str]:
    kinds = CompositeModel.from_expression(series["expression"]).parameter_kinds()
    return [
        name
        for name in series["free_params"]
        if kinds.get(name) in (ParameterKind.RATE, ParameterKind.STATIC_WIDTH)
    ]


def _converged(series: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in series["results"] if row["success"]]


def _zero_field_width(series: Mapping[str, Any]) -> tuple[str, float] | None:
    """The zero-field series' largest median rate or width, when it is measured."""
    rows = _converged(series)
    widths = [
        (name, statistics.median(row["parameters"][name] for row in rows))
        for name in _rates(series)
        if rows
        and all(
            row["parameters"][name] > 3.0 * row["uncertainties"].get(name, math.inf) for row in rows
        )
    ]
    return max(widths, key=lambda entry: entry[1]) if widths else None


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

    this = stored[name]
    if longitudinal(this):
        pairs = [(name, other) for other in stored if zero_field(stored[other])]
    elif zero_field(this):
        pairs = [(other, name) for other in stored if longitudinal(stored[other])]
    else:
        return None
    for lf_name, zf_name in pairs:
        width = _zero_field_width(stored[zf_name])
        if width is None:
            continue
        zf_param, delta = width
        lf = stored[lf_name]
        rows = _converged(lf)
        rates = _rates(lf)
        gone = [
            row
            for row in rows
            if all(
                row["parameters"][rate] + 2.0 * row["uncertainties"].get(rate, math.inf)
                < _GONE * delta
                for rate in rates
            )
        ]
        field = min(float(survey_runs[row["run"]]["field"]) for row in lf["results"])
        decoupling_field = _DECOUPLING_FIELDS * delta / _GAMMA_MU_PER_US_G
        if not rates or 2 * len(gone) <= len(rows) or field < decoupling_field:
            continue
        largest = max(row["parameters"][rate] for row in gone for rate in rates)
        return (
            f"NOTE: in a longitudinal field of {field:g} G and more (no line at its Larmor "
            f"frequency) the relaxation is gone: {', '.join(rates)} of {lf_name} stays at or "
            f"below {largest:.2g} µs⁻¹ on {len(gone)} of {len(rows)} converged runs, against "
            f"{zf_param} {delta:.3g} µs⁻¹ (median) in the zero-field series {zf_name}. A "
            f"static field distribution of that width is decoupled once the field is well "
            f"above Δ/γ_μ = {delta / _GAMMA_MU_PER_US_G:.2g} G ({_DECOUPLING_FIELDS:g}× is "
            f"{decoupling_field:.2g} G): the zero-field relaxation is from fields static on "
            f"the muon time scale, decoupled as a static distribution is. A rate pinned at "
            f"zero here is that result, not a failed fit: say so."
        )
    return None


__all__ = ["decoupling_note"]
