"""Human-readable fit-record formatting (the persistent fit log).

The durable fit record in Asymmetry is structured and lives **in the ``.asymp``
project**: the latest fit per ``(dataset, representation)`` on ``FitSlot.result`` and
the latest batch on ``FitSeries.results_by_run`` — the structured equivalents of
WiMDA's overwrite ``.fit``/``.bfit`` snapshots (which are likewise latest-only, not
append logs). This module does **not** introduce a parallel append-to-file log; it
is a Qt-free *formatter* that turns one such record (an enriched
:func:`~asymmetry.core.fitting.result_summary.fit_result_summary` dict) into a
readable provenance block. The same formatter feeds the in-app activity log entry
and the on-demand "Export fit report" action.

The formatter is deliberately tolerant: it renders whatever keys are present, so a
legacy record (without the additive ``quality`` / ``uncertainties_asymmetric`` /
provenance keys) still produces a sensible block.

Provenance that the core cannot source without a clock or GUI context — the
timestamp, model name, fit range, and provenance label — is injected by the caller
via :func:`enrich_summary_provenance`, keeping this module clock-free and testable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

#: Provenance keys this module reads/writes on a fit-summary dict. Additive — they
#: ride alongside the canonical summary fields and round-trip through ``.asymp``.
PROVENANCE_KEYS = ("model_name", "fit_range", "timestamp", "provenance", "npar", "ndof")


def enrich_summary_provenance(
    summary: dict,
    *,
    model_name: str | None = None,
    fit_range: str | None = None,
    timestamp: str | None = None,
    provenance: str | None = None,
    npar: int | None = None,
    ndof: int | None = None,
) -> dict:
    """Return ``summary`` with the additive provenance keys set (in place).

    Only non-``None`` values are written, so calling this twice never erases an
    earlier value. The timestamp is supplied by the caller (the core stays
    clock-free); pass an ISO-8601 string.
    """
    for key, value in (
        ("model_name", model_name),
        ("fit_range", fit_range),
        ("timestamp", timestamp),
        ("provenance", provenance),
        ("npar", npar),
        ("ndof", ndof),
    ):
        if value is not None:
            summary[key] = value
    return summary


def _fmt(value: float, spec: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return format(float(value), spec)


class FitLog:
    """Formats fit-result records into human-readable provenance blocks.

    Stateless apart from formatting preferences; one instance can format many
    records. ``value_format`` controls the numeric format of values and errors.
    """

    def __init__(self, *, value_format: str = ".6g") -> None:
        self._value_format = value_format

    # -- one record ---------------------------------------------------------

    def format_record(self, record: Mapping[str, Any], *, title: str | None = None) -> str:
        """Format one fit-summary record as a titled block.

        Recognised keys (all optional except for a graceful empty block):
        ``parameters``, ``uncertainties``, ``uncertainties_asymmetric``, ``quality``,
        ``chi_squared``, ``reduced_chi_squared``, ``success`` plus the provenance keys
        from :func:`enrich_summary_provenance`.
        """
        lines: list[str] = []
        heading = title or self._default_title(record)
        lines.append(f"=== {heading} ===")

        model_name = record.get("model_name")
        if model_name:
            lines.append(f"Model:  {model_name}")
        fit_range = record.get("fit_range")
        if fit_range:
            lines.append(f"Range:  {fit_range}")
        provenance = record.get("provenance")
        if provenance:
            lines.append(f"Source: {provenance}")
        if record.get("success") is False:
            lines.append("Status: fit did not converge")

        chi_line = self._chi_line(record)
        if chi_line:
            lines.append(chi_line)

        param_lines = self._param_lines(record)
        if param_lines:
            lines.append("")
            lines.extend(param_lines)
        return "\n".join(lines)

    # -- many records (export report) ---------------------------------------

    def format_report(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        header: str | None = None,
        titles: Iterable[str] | None = None,
    ) -> str:
        """Format several records into one report, blank-line separated."""
        title_list = list(titles) if titles is not None else None
        blocks: list[str] = []
        if header:
            blocks.append(header)
        for index, record in enumerate(records):
            title = title_list[index] if title_list and index < len(title_list) else None
            blocks.append(self.format_record(record, title=title))
        return "\n\n".join(blocks) + "\n"

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _default_title(record: Mapping[str, Any]) -> str:
        ts = record.get("timestamp")
        return f"Fit {ts}" if ts else "Fit"

    def _chi_line(self, record: Mapping[str, Any]) -> str:
        reduced = record.get("reduced_chi_squared")
        if reduced is None:
            return ""
        quality = record.get("quality") or {}
        dof = quality.get("dof") if isinstance(quality, Mapping) else None
        if dof is None:
            dof = record.get("ndof")
        line = f"chi^2/nu = {_fmt(reduced, '.4f')}"
        if dof is not None:
            line += f"  (nu={int(dof)})"
        verdict = quality.get("verdict") if isinstance(quality, Mapping) else None
        if verdict:
            low = quality.get("band_low")
            high = quality.get("band_high")
            confidence = quality.get("confidence")
            band = ""
            if low is not None and high is not None and confidence is not None:
                band = f"; target {_fmt(low, '.3f')}-{_fmt(high, '.3f')} at {int(round(confidence * 100))}%"
            line += f"  [{verdict}{band}]"
        return line

    def _param_lines(self, record: Mapping[str, Any]) -> list[str]:
        parameters = record.get("parameters") or {}
        if not parameters:
            return []
        uncertainties = record.get("uncertainties") or {}
        asymmetric = record.get("uncertainties_asymmetric") or {}
        width = max((len(str(name)) for name in parameters), default=0)
        out: list[str] = []
        for name, value in parameters.items():
            text = f"  {str(name):<{width}} = {_fmt(value, self._value_format)}"
            sigma = uncertainties.get(name)
            if sigma is not None:
                text += f" +/- {_fmt(sigma, self._value_format)}"
            interval = asymmetric.get(name)
            if interval is not None and len(interval) == 2:
                lo, hi = interval
                text += f"  (+{_fmt(hi, self._value_format)} / {_fmt(lo, self._value_format)})"
            out.append(text)
        return out
