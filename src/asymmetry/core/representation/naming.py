"""Unified default labelling for fit/trend series.

One naming scheme for every :class:`~asymmetry.core.representation.series.FitSeries`
chip, replacing the four divergent conventions the D4/D8 audit found ("Model ·
2923–2960", "B = 60 G", "GaussianPeak + ConstantBackground · 2952–29…",
"Series N"). The label produced here is a *default* (a display fallback): a user
rename stored on :attr:`FitSeries.label` always wins via
:meth:`FitSeries.display_name`.

The scheme is ``"<model> · <member-range>[ · <group>]"``:

* ``<model>`` — the composite-model expression (e.g. ``"Exponential + Constant"``);
  omitted for model-less (computed) series.
* ``<member-range>`` — the source-run span (``"2923"`` / ``"2923–2960"``), with a
  ``"groups "`` prefix for detector-group series so a grouped fit reads distinctly
  from a run batch over the same runs.
* ``<group>`` — an optional :class:`DataGroup`-name suffix (e.g. ``"B = 60 G"``)
  when the batch's members coincide with a browser data group. It is a *suffix*,
  not a replacement, so the group hint survives without colliding with the model.
"""

from __future__ import annotations

from asymmetry.core.representation.series import FitSeries


def composite_model_label(composite: object) -> str | None:
    """Human-readable model expression from a serialised composite model.

    e.g. ``{"component_names": ["Exponential", "Constant"], "operators": ["+"]}``
    -> ``"Exponential + Constant"``. Returns ``None`` when the structure is
    missing or empty.
    """
    if not isinstance(composite, dict):
        return None
    names = composite.get("component_names") or []
    operators = composite.get("operators") or []
    if not names:
        return None
    parts = [str(names[0])]
    for op, name in zip(operators, names[1:]):
        parts.append(str(op))
        parts.append(str(name))
    return " ".join(parts)


def _source_runs(series: FitSeries) -> list[int]:
    """Ordered, de-duplicated physical source runs backing *series*' members."""
    if series.member_kind == "groups":
        runs = {series.source_run_for(key) for key in series.member_run_numbers}
    else:
        runs = {int(r) for r in series.member_run_numbers}
    return sorted(runs)


def member_range(series: FitSeries) -> str:
    """Return the compact member-range string for *series*.

    ``""`` when it has no members, ``"2960"`` for a single run, ``"2923–2960"``
    for a span; detector-group series gain a ``"groups "`` prefix.
    """
    runs = _source_runs(series)
    if not runs:
        return ""
    span = f"{runs[0]}" if len(runs) == 1 else f"{runs[0]}–{runs[-1]}"
    return f"groups {span}" if series.member_kind == "groups" else span


def default_series_label(series: FitSeries, *, group_name: str | None = None) -> str:
    """Return the default (fallback) label for *series*.

    ``"<model> · <member-range>[ · <group>]"``. *group_name*, when supplied, is
    the browser :class:`DataGroup` name shared by every member; it is appended as
    a suffix. A user rename on :attr:`FitSeries.label` takes precedence — this is
    only the fallback rendered when no label is set.
    """
    model = composite_model_label(series.canonical_model)
    rng = member_range(series)
    parts = [part for part in (model, rng) if part]
    base = " · ".join(parts) if parts else "Series"
    suffix = (group_name or "").strip()
    return f"{base} · {suffix}" if suffix else base
