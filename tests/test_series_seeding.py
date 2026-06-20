"""Robustness heuristics for batch (series) fits.

The near-transition oscillatory bistability (EuO ZF approaching ``T_C``) makes a
per-run composite fit land either on the real low frequency or on a spurious
high-frequency branch with the amplitude collapsed to ~0. These tests pin the pure
detection / seed-suggestion heuristics that spot that signature and compute the
descending-frequency warm-start that fixes it.
"""

from __future__ import annotations

from asymmetry.core.fitting.series_seeding import (
    SeriesPoint,
    detect_amplitude_collapse,
    detect_frequency_outliers,
    diagnose_series,
    is_frequency_parameter,
    recommend_series_seeding,
    resolve_series_params,
    suggest_series_seeds,
)

# --- parameter-role resolution ---------------------------------------------


def test_resolve_series_params_picks_leading_amplitude_and_frequency():
    names = ["A_1", "frequency", "phase", "lambda", "A_bg"]
    amp, freq = resolve_series_params(names)
    assert amp == "A_1"
    assert freq == "frequency"


def test_resolve_series_params_handles_missing_frequency():
    amp, freq = resolve_series_params(["A", "lambda", "A_bg"])
    assert amp == "A"
    assert freq is None


def test_is_frequency_parameter_variants():
    assert is_frequency_parameter("frequency")
    assert is_frequency_parameter("frequency_2")
    assert is_frequency_parameter("field")
    assert is_frequency_parameter("nu")
    assert not is_frequency_parameter("lambda")
    assert not is_frequency_parameter("A_1")


# --- amplitude collapse -----------------------------------------------------


def test_amplitude_collapse_flags_near_zero_against_series_scale():
    amps = [25.0, 24.0, 22.0, 0.1, 18.0, 0.05, 15.0]
    assert detect_amplitude_collapse(amps) == [3, 5]


def test_amplitude_collapse_ignores_smooth_series():
    assert detect_amplitude_collapse([25.0, 24.0, 22.0, 20.0, 18.0]) == []


def test_amplitude_collapse_needs_minimum_points():
    assert detect_amplitude_collapse([25.0, 0.01]) == []


def test_amplitude_collapse_handles_none_values():
    amps = [25.0, None, 22.0, 0.1, 18.0]
    assert detect_amplitude_collapse(amps) == [3]


# --- frequency outliers -----------------------------------------------------


def test_frequency_outlier_flags_isolated_jump_only():
    orders = [10, 20, 30, 40, 50, 60, 70, 80]
    freqs = [30, 27, 24, 21, 29.5, 15, 12, 9]  # index 4 jumped to the spurious branch
    # Only the genuine outlier is flagged — its neighbours, whose leave-one-out
    # prediction it corrupts, are cleared by the second pass.
    assert detect_frequency_outliers(orders, freqs) == [4]


def test_frequency_outlier_clean_order_parameter_curve_has_none():
    orders = [5, 15, 25, 35, 45, 55, 60, 63, 66, 69]
    freqs = [30.1, 29.8, 29.0, 27.5, 25.0, 21.0, 18.0, 14.5, 10.0, 4.5]
    assert detect_frequency_outliers(orders, freqs) == []


def test_frequency_outlier_tolerates_noise():
    orders = [10, 20, 30, 40, 50, 60, 70, 80]
    freqs = [30.2, 26.8, 24.1, 21.0, 17.9, 15.1, 12.0, 9.1]
    assert detect_frequency_outliers(orders, freqs) == []


def test_frequency_outlier_needs_minimum_points():
    assert detect_frequency_outliers([1, 2, 3], [10, 30, 12]) == []


# --- descending-frequency seed suggestion -----------------------------------


def test_suggest_seeds_interpolates_from_good_runs():
    points = [
        SeriesPoint(run=1, order=10.0, amplitude=25.0, frequency=30.0),
        SeriesPoint(run=2, order=20.0, amplitude=24.0, frequency=26.0),
        SeriesPoint(run=3, order=30.0, amplitude=0.1, frequency=30.0),  # spurious
        SeriesPoint(run=4, order=40.0, amplitude=20.0, frequency=18.0),
    ]
    seeds = suggest_series_seeds(points, [3], amplitude_param="A_1", frequency_param="frequency")
    assert 3 in seeds
    # Linear interpolation between run 2 (26 MHz @ 20) and run 4 (18 MHz @ 40) at
    # order 30 → 22 MHz, with the amplitude restored toward the good-run median.
    assert seeds[3]["frequency"] == 22.0
    assert seeds[3]["A_1"] > 1.0


def test_suggest_seeds_empty_when_nothing_flagged():
    points = [SeriesPoint(run=1, order=10.0, amplitude=25.0, frequency=30.0)]
    assert (
        suggest_series_seeds(points, [], amplitude_param="A_1", frequency_param="frequency") == {}
    )


# --- combined diagnosis -----------------------------------------------------


def test_diagnose_series_combines_collapse_outlier_and_failure():
    points = [
        SeriesPoint(run=2960, order=10.0, amplitude=25.0, frequency=30.0, success=True),
        SeriesPoint(run=2955, order=30.0, amplitude=24.0, frequency=26.0, success=True),
        SeriesPoint(run=2950, order=50.0, amplitude=22.0, frequency=18.0, success=True),
        SeriesPoint(run=2945, order=60.0, amplitude=20.0, frequency=12.0, success=True),
        SeriesPoint(run=2944, order=63.0, amplitude=0.1, frequency=30.5, success=True),  # collapse
        SeriesPoint(run=2943, order=65.5, amplitude=18.0, frequency=10.5, success=True),
        SeriesPoint(run=2941, order=68.3, amplitude=15.0, frequency=8.0, success=False),  # failed
    ]
    diag = diagnose_series(points, amplitude_param="A_1", frequency_param="frequency")
    assert diag.has_issues
    assert 2944 in diag.collapsed_runs
    assert 2941 in diag.failed_runs
    assert set(diag.flagged_runs) >= {2944, 2941}
    # Each flagged run gets a descending warm-start seed.
    for run in diag.flagged_runs:
        assert run in diag.suggested_seeds
        assert "frequency" in diag.suggested_seeds[run]
    assert "amplitude collapsed" in diag.reason


def test_diagnose_series_clean_batch_has_no_issues():
    points = [
        SeriesPoint(
            run=i, order=float(i), amplitude=25.0 - i, frequency=30.0 - 2.0 * i, success=True
        )
        for i in range(6)
    ]
    diag = diagnose_series(points, amplitude_param="A_1", frequency_param="frequency")
    assert not diag.has_issues
    assert diag.suggested_seeds == {}


def test_diagnose_series_without_frequency_param_only_collapse():
    points = [
        SeriesPoint(run=1, order=10.0, amplitude=25.0, frequency=None, success=True),
        SeriesPoint(run=2, order=20.0, amplitude=24.0, frequency=None, success=True),
        SeriesPoint(run=3, order=30.0, amplitude=0.05, frequency=None, success=True),
        SeriesPoint(run=4, order=40.0, amplitude=22.0, frequency=None, success=True),
    ]
    diag = diagnose_series(points, amplitude_param="A", frequency_param=None)
    assert diag.collapsed_runs == (3,)
    assert diag.outlier_runs == ()


# --- seeding-mode recommendation (shared Auto policy) -----------------------


def test_recommend_series_chains_ordered_scan():
    rec = recommend_series_seeding([10, 11, 12], {10: 5.0, 11: 10.0, 12: 15.0})
    assert rec.mode == "chain"
    assert "chain" in rec.reason.lower()


def test_recommend_series_skips_too_few_members():
    assert recommend_series_seeding([10, 11], {10: 5.0, 11: 10.0}).mode == "as_provided"


def test_recommend_series_skips_without_order_key():
    assert recommend_series_seeding([10, 11, 12], None).mode == "as_provided"


def test_recommend_series_skips_constant_order_key():
    rec = recommend_series_seeding([10, 11, 12], {10: 5.0, 11: 5.0, 12: 5.0})
    assert rec.mode == "as_provided"
