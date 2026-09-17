"""Default series labelling (D10).

``<model> · <fit-range>[ · <group>]``: the model expression, the recipe's fit
window in its domain's unit, and an optional owning-group suffix. Two series
that still collide are separated by :func:`disambiguate_series_label`.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import (
    FitSeries,
    RepresentationType,
    default_series_label,
    disambiguate_series_label,
    member_range,
)

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _model(names=("Exponential", "Constant")) -> dict:
    return CompositeModel(list(names), operators=["+"] * (len(names) - 1)).to_dict()


def _series(
    batch_id: str = "b",
    *,
    rep_type=_FB,
    model=None,
    runs=(1, 2),
    member_kind="runs",
    member_source_run=None,
    fit_range=(0.0, 6.0),
) -> FitSeries:
    return FitSeries(
        batch_id,
        rep_type,
        member_kind=member_kind,
        member_run_numbers=list(runs),
        member_source_run=member_source_run,
        canonical_model=model if model is not None else _model(),
        param_roles={"A": "local"},
        recipe={"fit_range": {"min": fit_range[0], "max": fit_range[1]}},
    )


# ── member range (still the auto-group / browser-highlight formatter) ─────────


def test_member_range_run_series():
    assert member_range(_series(runs=(2923, 2960, 2941))) == "2923–2960"
    assert member_range(_series(runs=(2960,))) == "2960"


def test_member_range_group_series_prefix():
    series = _series(
        member_kind="groups",
        runs=(-2961001, -2967001),
        member_source_run={-2961001: 2961, -2967001: 2967},
    )
    assert member_range(series) == "groups 2961–2967"


# ── default label ────────────────────────────────────────────────────────────


def test_default_series_label_model_and_time_range():
    assert default_series_label(_series()) == "Exponential + Constant · 0–6 µs"


def test_default_series_label_uses_the_frequency_unit():
    series = _series(rep_type=RepresentationType.FREQ_FFT, fit_range=(0.0, 20.0))
    assert default_series_label(series) == "Exponential + Constant · 0–20 MHz"


def test_default_series_label_keeps_fractional_bounds():
    assert default_series_label(_series(fit_range=(0.15, 8.0))).endswith("0.15–8 µs")


def test_default_series_label_omits_an_unbounded_window():
    assert default_series_label(_series(fit_range=(None, None))) == "Exponential + Constant"


def test_default_series_label_distinguishes_two_windows_of_one_model():
    """The window is what tells two runs of the same analysis apart (D3/D10)."""
    assert default_series_label(_series(fit_range=(0.0, 6.0))) != default_series_label(
        _series(fit_range=(0.0, 8.0))
    )


def test_default_series_label_appends_group_suffix():
    label = default_series_label(_series(), group_name="B = 60 G")
    assert label == "Exponential + Constant · 0–6 µs · B = 60 G"


def test_default_series_label_computed_series():
    """A model-less series still yields a sane default from its window alone."""
    scan = FitSeries(
        "s",
        _FB,
        member_run_numbers=[1, 2],
        canonical_model=None,
        recipe={"fit_range": {"min": 0.0, "max": 10.0}},
    )
    assert default_series_label(scan) == "0–10 µs"
    # Nothing to say at all still reads as something.
    assert default_series_label(FitSeries("s", _FB)) == "Series"


# ── disambiguation ───────────────────────────────────────────────────────────


def test_disambiguate_leaves_an_unused_label_alone():
    assert disambiguate_series_label("Exp · 0–6 µs", ["Other"]) == "Exp · 0–6 µs"
    assert disambiguate_series_label("Exp · 0–6 µs", []) == "Exp · 0–6 µs"


def test_disambiguate_counts_up_past_every_taken_label():
    taken = ["Exp", "Exp (2)", "Exp (3)"]
    assert disambiguate_series_label("Exp", taken) == "Exp (4)"


def test_disambiguate_skips_only_the_taken_suffixes():
    assert disambiguate_series_label("Exp", ["Exp", "Exp (3)"]) == "Exp (2)"
