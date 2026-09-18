"""Unified default labelling for fit/trend series.

One naming scheme for every :class:`~asymmetry.core.representation.series.FitSeries`
chip, replacing the four divergent conventions the D4/D8 audit found ("Model ·
2923–2960", "B = 60 G", "GaussianPeak + ConstantBackground · 2952–29…",
"Series N"). The label produced here is a *default* (a display fallback): a user
rename stored on :attr:`FitSeries.label` always wins via
:meth:`FitSeries.display_name`.

The scheme is ``"<model> · <fit-range>[ · <group>]"`` (D10):

* ``<model>`` — the composite-model expression (e.g. ``"Exponential + Constant"``);
  omitted for model-less (computed) series.
* ``<fit-range>`` — the series recipe's fit window in its domain's unit
  (``"0–6 µs"`` in the time domain, ``"0–20 MHz"`` in the frequency domain);
  omitted when the recipe leaves the window unbounded. The window is what
  distinguishes two otherwise identical runs of one analysis (D3), so it is the
  part of the label that tells them apart.
* ``<group>`` — an optional :class:`DataGroup`-name suffix (e.g. ``"B = 60 G"``)
  when the batch's members coincide with a browser data group. It is a *suffix*,
  not a replacement, so the group hint survives without colliding with the model.

Two series in one group can still land on the same default label (same model,
same window); :func:`disambiguate_series_label` appends ``" (2)"``, ``" (3)"``…
at record time so the chips stay distinguishable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from asymmetry.core.representation.series import FitSeries

#: Axis unit rendered in a default label, per representation domain.
_DOMAIN_UNITS = {"time": "µs", "frequency": "MHz"}


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


def _format_bound(value: float | None) -> str:
    """Render one fit-range bound compactly (``6.0`` -> ``"6"``, ``None`` -> ``""``)."""
    if value is None:
        return ""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text or "0"


def fit_range_label(series: FitSeries) -> str:
    """Return the series' fit window as a label fragment, or ``""``.

    ``"0–6 µs"`` for a time-domain series, ``"0–20 MHz"`` for a frequency-domain
    one. An unbounded side renders as nothing (``"–6 µs"`` reads "up to 6 µs");
    a window unbounded on *both* sides has nothing to say and returns ``""``.
    """
    fit_range = series.recipe["fit_range"]
    low, high = fit_range["min"], fit_range["max"]
    if low is None and high is None:
        return ""
    unit = _DOMAIN_UNITS[series.rep_type.domain]
    return f"{_format_bound(low)}–{_format_bound(high)} {unit}"


def default_series_label(series: FitSeries, *, group_name: str | None = None) -> str:
    """Return the default (fallback) label for *series* (D10).

    ``"<model> · <fit-range>[ · <group>]"``. *group_name*, when supplied, is
    the browser :class:`DataGroup` name shared by every member; it is appended as
    a suffix. A user rename on :attr:`FitSeries.label` takes precedence — this is
    only the fallback rendered when no label is set.
    """
    model = composite_model_label(series.canonical_model)
    rng = fit_range_label(series)
    parts = [part for part in (model, rng) if part]
    base = " · ".join(parts) if parts else "Series"
    suffix = (group_name or "").strip()
    return f"{base} · {suffix}" if suffix else base


def default_joint_fit_label(series_labels: Sequence[str]) -> str:
    """Return the default label for a joint fit: ``"Joint: <A> + <B>"``.

    Same "fallback, user rename wins" contract as :func:`default_series_label`
    — :meth:`JointFit.display_name` only reaches this when no label has been
    set. *series_labels* is the member series' own display names, in the
    joint fit's member order, so relabeling one member does not reshuffle a
    joint fit's default label out from under a user who has not renamed it.
    """
    return "Joint: " + " + ".join(str(label) for label in series_labels)


def disambiguate_series_label(label: str, existing_labels: Iterable[str]) -> str:
    """Return *label*, suffixed ``" (2)"``, ``" (3)"``… until it is unused.

    Called at record time with the labels already on show, so two series that
    share a model, a window and a group still read as distinct chips (D10).
    """
    taken = {str(existing) for existing in existing_labels}
    if label not in taken:
        return label
    ordinal = 2
    while f"{label} ({ordinal})" in taken:
        ordinal += 1
    return f"{label} ({ordinal})"
