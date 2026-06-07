"""Tests for the time-integral asymmetry / field-scan API.

Covers ``asymmetry.core.transform.integral``: the per-run reduction
(``integrate_asymmetry`` / ``integrate_curve`` / ``integrate_run``, both
``"integral"`` and ``"differential"`` methods), the field-scan assembly
(``build_field_scan``), and the WiMDA-style ``differentiate_scan`` derivative.

Synthetic runs are constructed with known forward/backward counts so the
integrals are exact and hand-checkable, and the error is asserted against
``compute_asymmetry`` to prove the integral and time-domain observables share a
single error model (the porting-study requirement).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform import (
    FieldScan,
    build_field_scan,
    compute_asymmetry,
    differentiate_scan,
    integrate_asymmetry,
    integrate_curve,
    integrate_run,
)

# --- fixtures ----------------------------------------------------------------


def _run(
    forward_counts,
    backward_counts,
    *,
    bin_width: float = 0.1,
    t0: int = 0,
    first_good: int = 0,
    last_good: int | None = None,
    alpha: float = 1.0,
    run_number: int = 1,
    field: float | None = 100.0,
    temperature: float | None = 10.0,
) -> Run:
    """A two-detector run: detector 1 = forward, detector 2 = backward."""
    f = np.asarray(forward_counts, dtype=np.float64)
    b = np.asarray(backward_counts, dtype=np.float64)
    n = len(f)
    histograms = [
        Histogram(counts=f, bin_width=bin_width, t0_bin=t0),
        Histogram(counts=b, bin_width=bin_width, t0_bin=t0),
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": alpha,
        "first_good_bin": first_good,
        "last_good_bin": n - 1 if last_good is None else last_good,
    }
    metadata: dict = {"run_number": run_number}
    if field is not None:
        metadata["field"] = field
    if temperature is not None:
        metadata["temperature"] = temperature
    return Run(
        run_number=run_number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="synthetic.nxs",
    )


# --- integrate_asymmetry: integral method ------------------------------------


def test_integral_matches_wimda_count_asymmetry_at_alpha_one():
    # WiMDA ALC: A = (F_int - B_int) / (F_int + B_int), alpha = 1.
    forward = np.full(5, 100.0)
    backward = np.full(5, 50.0)
    value, _ = integrate_asymmetry(forward, backward, alpha=1.0, method="integral")
    assert value == pytest.approx((500.0 - 250.0) / (500.0 + 250.0))  # 1/3


def test_integral_shares_compute_asymmetry_error_model():
    forward = np.array([90.0, 100.0, 110.0])
    backward = np.array([50.0, 50.0, 50.0])
    alpha = 1.3
    value, error = integrate_asymmetry(forward, backward, alpha=alpha, method="integral")

    f_int = np.array([forward.sum()])
    b_int = np.array([backward.sum()])
    exp_value, exp_error = compute_asymmetry(f_int, b_int, alpha)
    assert value == pytest.approx(float(exp_value[0]))
    assert error == pytest.approx(float(exp_error[0]))


def test_integral_applies_alpha():
    forward = np.full(4, 100.0)
    backward = np.full(4, 100.0)
    # alpha != 1 breaks the F == B balance.
    value, _ = integrate_asymmetry(forward, backward, alpha=2.0, method="integral")
    assert value == pytest.approx((400.0 - 2.0 * 400.0) / (400.0 + 2.0 * 400.0))


# --- integral vs differential ------------------------------------------------


def test_methods_agree_for_flat_asymmetry():
    forward = np.full(6, 120.0)
    backward = np.full(6, 60.0)
    integral, _ = integrate_asymmetry(forward, backward, method="integral")
    differential, _ = integrate_asymmetry(forward, backward, method="differential")
    assert integral == pytest.approx(differential)


def test_methods_differ_for_non_flat_asymmetry():
    forward = np.array([90.0, 100.0, 110.0])
    backward = np.array([50.0, 50.0, 50.0])
    integral, _ = integrate_asymmetry(forward, backward, method="integral")
    differential, _ = integrate_asymmetry(forward, backward, method="differential")
    # Differential is the mean of per-bin asymmetries; integral sums counts first.
    per_bin, _ = compute_asymmetry(forward, backward, 1.0)
    assert differential == pytest.approx(float(np.mean(per_bin)))
    assert integral != pytest.approx(differential)


def test_differential_excludes_zero_denominator_bins():
    # The middle bin has F + alpha*B == 0; compute_asymmetry returns the
    # (0, 1.0) sentinel there. It must not pollute the differential mean/error.
    forward = np.array([100.0, 0.0, 110.0])
    backward = np.array([50.0, 0.0, 55.0])
    value, error = integrate_asymmetry(forward, backward, method="differential")
    per_bin, errs = compute_asymmetry(forward, backward, 1.0)
    assert value == pytest.approx((per_bin[0] + per_bin[2]) / 2.0)
    assert error == pytest.approx(np.hypot(errs[0], errs[2]) / 2.0)


def test_differential_all_zero_counts_rejected():
    with pytest.raises(ValueError, match="non-zero counts"):
        integrate_asymmetry(np.zeros(3), np.zeros(3), method="differential")


# --- window selection --------------------------------------------------------


def test_window_restricts_integration():
    forward = np.array([1000.0, 100.0, 100.0, 100.0, 1000.0])
    backward = np.array([1000.0, 50.0, 50.0, 50.0, 1000.0])
    time = np.arange(5) * 0.1  # 0.0 .. 0.4
    # Window 0.1 .. 0.3 selects the three central, balanced bins.
    value, _ = integrate_asymmetry(
        forward, backward, time=time, t_min=0.1, t_max=0.3, method="integral"
    )
    assert value == pytest.approx((300.0 - 150.0) / (300.0 + 150.0))  # 1/3


def test_window_requires_time_axis():
    with pytest.raises(ValueError, match="time axis is required"):
        integrate_asymmetry(np.ones(3), np.ones(3), t_min=0.1, method="integral")


# --- integrate_curve ---------------------------------------------------------


def test_integrate_curve_mean_and_error_of_mean():
    time = np.arange(4) * 0.1
    asym = np.array([0.2, 0.3, 0.4, 0.5])
    err = np.array([0.1, 0.1, 0.1, 0.1])
    value, error = integrate_curve(time, asym, err)
    assert value == pytest.approx(np.mean(asym))
    assert error == pytest.approx(np.sqrt(np.sum(err**2)) / err.size)


# --- integrate_run -----------------------------------------------------------


def test_integrate_run_reduces_groups():
    run = _run(np.full(5, 100.0), np.full(5, 50.0))
    value, _ = integrate_run(run)
    assert value == pytest.approx(1.0 / 3.0)


def test_integrate_run_accepts_dataset():
    run = _run(np.full(5, 100.0), np.full(5, 50.0))
    dataset = MuonDataset(time=np.array([]), asymmetry=np.array([]), error=np.array([]), run=run)
    value, _ = integrate_run(dataset)
    assert value == pytest.approx(1.0 / 3.0)


def test_integrate_run_defaults_to_good_bin_window():
    # Wild counts outside the good-bin window must be excluded by default.
    forward = np.array([1000.0, 100.0, 100.0, 100.0, 1000.0])
    backward = np.array([1000.0, 50.0, 50.0, 50.0, 1000.0])
    run = _run(forward, backward, first_good=1, last_good=3, bin_width=0.1)

    default_value, _ = integrate_run(run)
    assert default_value == pytest.approx(1.0 / 3.0)

    # Forcing the full range pulls in the balanced edge bins and changes it.
    full_value, _ = integrate_run(run, t_min=-1.0, t_max=100.0)
    assert full_value != pytest.approx(default_value)


def test_integrate_run_alpha_override():
    run = _run(np.full(4, 100.0), np.full(4, 100.0), alpha=1.0)
    value, _ = integrate_run(run, alpha=2.0)
    assert value == pytest.approx((400.0 - 800.0) / (400.0 + 800.0))


def test_integrate_run_requires_groups():
    run = Run(run_number=1, histograms=[Histogram(counts=np.ones(4), bin_width=0.1)])
    with pytest.raises(ValueError, match="grouping definition"):
        integrate_run(run)


def test_integrate_run_dataset_without_run():
    dataset = MuonDataset(time=np.array([]), asymmetry=np.array([]), error=np.array([]))
    with pytest.raises(ValueError, match="no source run"):
        integrate_run(dataset)


# --- validation --------------------------------------------------------------


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        integrate_asymmetry(np.ones(3), np.ones(3), method="bogus")


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0, float("nan")])
def test_non_positive_alpha_rejected(bad_alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        integrate_asymmetry(np.ones(3), np.ones(3), alpha=bad_alpha)


def test_inverted_window_rejected():
    time = np.arange(3) * 0.1
    with pytest.raises(ValueError, match="t_min must be"):
        integrate_asymmetry(np.ones(3), np.ones(3), time=time, t_min=0.2, t_max=0.1)


def test_equal_window_selects_single_bin():
    # t_min == t_max is allowed and selects the single matching bin.
    forward = np.array([100.0, 200.0, 100.0])
    backward = np.array([50.0, 50.0, 50.0])
    time = np.array([0.0, 0.1, 0.2])
    value, _ = integrate_asymmetry(
        forward, backward, time=time, t_min=0.1, t_max=0.1, method="integral"
    )
    assert value == pytest.approx((200.0 - 50.0) / (200.0 + 50.0))


def test_integrate_run_single_good_bin():
    # A run whose good-bin range is a single bin must integrate, not crash.
    run = _run(np.full(5, 100.0), np.full(5, 50.0), first_good=2, last_good=2)
    value, _ = integrate_run(run)
    assert value == pytest.approx(1.0 / 3.0)


def test_empty_counts_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        integrate_asymmetry(np.array([]), np.array([]))


def test_passing_period_list_is_typeerror():
    with pytest.raises(TypeError, match="select one period"):
        integrate_run([_run(np.ones(3), np.ones(3))])  # type: ignore[arg-type]


# --- build_field_scan --------------------------------------------------------


def _scan_run(run_number, field, level):
    """A run whose integral asymmetry is a known function of ``level``."""
    forward = np.full(5, 100.0 + level)
    backward = np.full(5, 100.0 - level)
    return _run(forward, backward, run_number=run_number, field=field)


def test_build_field_scan_orders_by_field():
    runs = [
        _scan_run(3, field=300.0, level=30.0),
        _scan_run(1, field=100.0, level=10.0),
        _scan_run(2, field=200.0, level=20.0),
    ]
    scan = build_field_scan(runs, order_key="field")
    assert isinstance(scan, FieldScan)
    assert scan.run_numbers == [1, 2, 3]
    assert list(scan.x) == [100.0, 200.0, 300.0]
    # value increases with level: A = level / 100.
    assert scan.value == pytest.approx([0.1, 0.2, 0.3])
    assert scan.x_label == "B (G)"
    assert scan.excluded == []


def test_build_field_scan_excludes_runs_missing_log():
    runs = [
        _scan_run(1, field=100.0, level=10.0),
        _run(np.full(5, 100.0), np.full(5, 50.0), run_number=2, field=None),
    ]
    scan = build_field_scan(runs, order_key="field")
    assert scan.run_numbers == [1]
    assert scan.excluded == [(2, "no field value")]


def test_build_field_scan_excludes_non_finite_log():
    # A NaN field log must be treated as missing, not sorted into the scan.
    runs = [
        _scan_run(1, field=100.0, level=10.0),
        _run(np.full(5, 100.0), np.full(5, 50.0), run_number=2, field=float("nan")),
    ]
    scan = build_field_scan(runs, order_key="field")
    assert scan.run_numbers == [1]
    assert scan.excluded == [(2, "no field value")]


def test_build_field_scan_excludes_unresolvable_item_without_aborting():
    # A MuonDataset with no source run must be excluded, not crash the scan.
    good = _scan_run(1, field=100.0, level=10.0)
    orphan = MuonDataset(
        time=np.array([]),
        asymmetry=np.array([]),
        error=np.array([]),
        metadata={"run_number": 9},
    )
    scan = build_field_scan([good, orphan], order_key="field")
    assert scan.run_numbers == [1]
    assert len(scan.excluded) == 1
    assert scan.excluded[0][0] == 9


def test_build_field_scan_keeps_zero_field_point():
    # A TF run at 0 G is a legitimate scan point, not a missing value.
    runs = [
        _scan_run(1, field=0.0, level=10.0),
        _scan_run(2, field=50.0, level=20.0),
    ]
    scan = build_field_scan(runs, order_key="field")
    assert scan.run_numbers == [1, 2]
    assert list(scan.x) == [0.0, 50.0]


def test_build_field_scan_order_by_run_and_temperature():
    runs = [
        _run(np.full(5, 110.0), np.full(5, 90.0), run_number=5, temperature=20.0),
        _run(np.full(5, 120.0), np.full(5, 80.0), run_number=2, temperature=5.0),
    ]
    by_run = build_field_scan(runs, order_key="run")
    assert by_run.run_numbers == [2, 5]
    assert by_run.x_label == "Run"

    by_temp = build_field_scan(runs, order_key="temperature")
    assert by_temp.run_numbers == [2, 5]
    assert list(by_temp.x) == [5.0, 20.0]


def test_build_field_scan_invalid_order_key():
    with pytest.raises(ValueError, match="order_key must be one of"):
        build_field_scan([], order_key="pressure")


# --- differentiate_scan ------------------------------------------------------


def _manual_scan() -> FieldScan:
    return FieldScan(
        x=np.array([0.0, 100.0, 300.0]),
        value=np.array([0.1, 0.2, 0.5]),
        error=np.array([0.01, 0.02, 0.05]),
        run_numbers=[1, 2, 3],
        order_key="field",
        method="integral",
        x_label="B (G)",
    )


def test_differentiate_scan_forward_difference():
    deriv = differentiate_scan(_manual_scan())
    assert deriv.derivative is True
    assert list(deriv.x) == [50.0, 200.0]  # midpoints
    assert deriv.value == pytest.approx([(0.2 - 0.1) / 100.0, (0.5 - 0.2) / 200.0])
    assert deriv.error[0] == pytest.approx(np.hypot(0.01, 0.02) / 100.0)


def test_differentiate_scan_max_gap_skips_wide_pairs():
    deriv = differentiate_scan(_manual_scan(), max_gap=150.0)
    # Second pair has dx = 200 > 150 and is dropped.
    assert deriv.n_points == 1
    assert deriv.x[0] == pytest.approx(50.0)


def test_differentiate_scan_rejects_derivative_input():
    deriv = differentiate_scan(_manual_scan())
    with pytest.raises(ValueError, match="not a derivative"):
        differentiate_scan(deriv)
