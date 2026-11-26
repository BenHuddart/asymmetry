"""Fit statistics and parameter rows used by the fit results displays."""

from __future__ import annotations

import math
from collections.abc import Mapping

from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.result_summary import fit_result_summary, fit_statistic_labels
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.fitting.spectral_weights import spectral_weighting_detail
from asymmetry.gui.fit_settings import fit_quality_confidence
from asymmetry.gui.styles.widgets import FIT_VERDICT_CHIP_COLOURS, NEUTRAL_CHIP_COLOURS
from asymmetry.gui.utils.formatting import format_fit_statistic, format_param_label
from asymmetry.gui.windows.fit_results_window import FitParameterRow, FitRangeResults, FitResults


def fit_statistic_text(
    summary: Mapping, precision: str, *, short: bool = False, include_raw: bool = False
) -> str:
    raw_label, label = fit_statistic_labels(summary)
    if label is None:
        return f"{raw_label} {format(summary['chi_squared'], precision)}"
    reduced = summary.get("reduced_chi_squared")
    label = "χ²ᵣ" if short and label == "χ²/ν" else label
    text = f"{label} {format_fit_statistic(reduced, precision)}"
    if include_raw and reduced is None and summary.get("chi_squared") is not None:
        text += f" · {raw_label} {summary['chi_squared']:.6g}"
    return text


def fit_results_snapshot(
    result: FitResult,
    *,
    title: str,
    model: str,
    fit_range: str,
    runs: str,
) -> FitResults:
    """Freeze fitted values and likelihoods; launch strings supply their provenance."""
    rows = []
    free_names = {param.name for param in result.parameters.free_parameters}
    for param in result.parameters:
        info = get_param_info(param.name)
        rows.append(
            FitParameterRow(
                name=param.name,
                symbol=info.unicode_label(include_unit=False),
                unit=info.unit or "",
                value=float(param.value),
                error=result.uncertainties.get(param.name),
                fixed=param.name not in free_names,
            )
        )

    summary = fit_result_summary(result, confidence=fit_quality_confidence())
    quality = summary["quality"]
    if quality is None:
        verdict, colours = summary["quality_reason"] or "no verdict", NEUTRAL_CHIP_COLOURS
        raw_label, reduced_label = fit_statistic_labels(summary)
        if reduced_label is not None and summary["reduced_chi_squared"] is None:
            verdict = f"{verdict} {raw_label} = {result.chi_squared:.6g}."
    else:
        band = f"{quality['band_low']:.2g}–{quality['band_high']:.2g}"
        verdict = f"{quality['verdict']} fit (band {band} at {quality['confidence'] * 100:g} %)"
        colours = FIT_VERDICT_CHIP_COLOURS[quality["verdict"]]

    return FitResults(
        title=title,
        # A run's asymmetry fit has neither a trended parameter nor an x axis of
        # its own: it is one run against the fitted range, named by `runs`.
        parameter_name="",
        x_label="",
        runs=runs,
        ranges=(
            FitRangeResults(
                model=model,
                chi_squared=fit_statistic_text(summary, ".3g", short=True),
                verdict=verdict,
                colours=colours,
                bounds=fit_range,
                error_mode=spectral_weighting_detail(result.input_attribution) or "",
                # Free parameters first ("sorted" is stable, as in the trend
                # panel's snapshot).
                parameters=tuple(sorted(rows, key=lambda row: row.fixed)),
            ),
        ),
    )


def fit_readout_html(record: Mapping, source: str) -> str:
    """Render values and their available statistical interpretation after fitting or reopening."""
    if record.get("success") is False:
        return f"<b>{source} failed:</b> {record.get('message', 'Fit failed')}"
    raw_label, reduced_label = fit_statistic_labels(record)
    lines = [f"<b>{source}</b>"]
    if (raw := record.get("chi_squared")) is not None:
        lines.append(f"<b>{raw_label} = {raw:.4f}</b>")
    if reduced_label is not None and "reduced_chi_squared" in record:
        reduced = record.get("reduced_chi_squared")
        label = "χ²ᵣ" if reduced_label == "χ²/ν" else reduced_label
        lines.append(f"<b>{label} = {format_fit_statistic(reduced, '.4f')}</b>")
    if weighting := spectral_weighting_detail(record.get("input_attribution")):
        lines.append(weighting)
    if reason := record.get("quality_reason"):
        lines.append(str(reason))
    lines.append("<br><b>Parameters:</b>")
    uncertainties = record.get("uncertainties") or {}
    for name, value in (record.get("parameters") or {}).items():
        error = uncertainties.get(name)
        suffix = (
            f" ± {float(error):.6f}" if error is not None and math.isfinite(float(error)) else ""
        )
        lines.append(f"  {format_param_label(name)} = {float(value):.6f}{suffix}")
    return "<br>".join(lines)
