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


def format_run_range(runs) -> str:
    """Return the compact run-range string for an arbitrary run collection.

    ``""`` for no runs, ``"2960"`` for a single run, ``"2923–2960"`` for a span
    (first–last of the sorted, de-duplicated numbers). The single formatter both
    :func:`member_range` and the auto-group namer share so a group minted from a
    batch ("Runs 1001–1010") reads identically to the batch's own member range.
    """
    nums = sorted({int(r) for r in (runs or [])})
    if not nums:
        return ""
    return f"{nums[0]}" if len(nums) == 1 else f"{nums[0]}–{nums[-1]}"


def member_range(series: FitSeries) -> str:
    """Return the compact member-range string for *series*.

    ``""`` when it has no members, ``"2960"`` for a single run, ``"2923–2960"``
    for a span; detector-group series gain a ``"groups "`` prefix.
    """
    span = format_run_range(series.source_runs())
    if not span:
        return ""
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
