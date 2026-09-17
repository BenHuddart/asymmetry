"""Variable and constant-error binning modes (WiMDA Group.pas:1411–1418).

The WiMDA formula oracle: width(t) = bin0·exp(λ_µ·t·0.22·ln(bin10/bin0)),
whose rounded 0.22 makes it ≈ bin0·(bin10/bin0)^(t/10 µs); Asymmetry
implements the exact law (study divergence D8, < 0.2 % difference).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.representation.time import TimeFBAsymmetry
from asymmetry.core.transform import (
    binned_fb_asymmetry,
    binning_slice_edges,
    resolve_binning_mode,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US

BIN_WIDTH_US = 0.016


def test_resolve_binning_mode_defaults_and_validation():
    assert resolve_binning_mode({}) == ("fixed", 0.08, 0.25)
    assert resolve_binning_mode(None) == ("fixed", 0.08, 0.25)
    mode, bin0, bin10 = resolve_binning_mode(
        {"binning_mode": "variable", "bin0_us": 0.1, "bin10_us": 0.5}
    )
    assert (mode, bin0, bin10) == ("variable", 0.1, 0.5)
    assert resolve_binning_mode({"binning_mode": "nonsense"})[0] == "fixed"
    assert resolve_binning_mode({"binning_mode": "variable", "bin0_us": -1.0})[1] == 0.08


def test_variable_widths_follow_exact_law_and_wimda_formula():
    bin0, bin10 = 0.08, 0.25
    edges = binning_slice_edges(
        100_000,
        mode="variable",
        bin_width_us=BIN_WIDTH_US,
        t_start_us=0.0,
        bin0_us=bin0,
        bin10_us=bin10,
    )
    starts_us = edges[:-1] * BIN_WIDTH_US
    widths_us = np.diff(edges) * BIN_WIDTH_US
    exact = bin0 * (bin10 / bin0) ** (starts_us / 10.0)
    # Snapped to whole raw bins: within one raw bin of the exact law.
    assert np.all(np.abs(widths_us[:-1] - exact[:-1]) <= BIN_WIDTH_US + 1e-12)
    # WiMDA's rounded-constant formula (0.22 ≈ τ_µ/10) drifts from the exact
    # law as exp(0.0014·(t/10)·ln(bin10/bin0)) − 1: < 0.2 % at 10 µs and
    # < 0.6 % over a realistic 32 µs window (study divergence D8).
    in_window = starts_us <= 32.0
    wimda = bin0 * np.exp(
        (1.0 / MUON_LIFETIME_US) * starts_us[in_window] * 0.22 * np.log(bin10 / bin0)
    )
    relative = np.abs(wimda - exact[in_window]) / exact[in_window]
    assert np.all(relative < 0.006)
    assert np.all(relative[starts_us[in_window] <= 10.0] < 0.002)
    # Width at 10 µs equals bin10 (within a raw bin).
    at_10 = int(np.searchsorted(starts_us, 10.0))
    assert widths_us[at_10] == pytest.approx(bin10, abs=BIN_WIDTH_US)


def test_constant_error_widths_grow_with_muon_lifetime():
    bin0 = 0.08
    edges = binning_slice_edges(
        200_000,
        mode="constant_error",
        bin_width_us=BIN_WIDTH_US,
        t_start_us=0.0,
        bin0_us=bin0,
    )
    starts_us = edges[:-1] * BIN_WIDTH_US
    widths_us = np.diff(edges) * BIN_WIDTH_US
    exact = bin0 * np.exp(starts_us / MUON_LIFETIME_US)
    assert np.all(np.abs(widths_us[:-1] - exact[:-1]) <= BIN_WIDTH_US + 1e-12)


def test_constant_error_mode_gives_flat_errors():
    """The defining property: ~equal statistics per output bin."""
    rng = np.random.default_rng(1)
    n, t0 = 2000, 10
    t = (np.arange(n) - t0) * BIN_WIDTH_US
    decay = np.exp(-np.clip(t, 0.0, None) / MUON_LIFETIME_US) * (t >= 0)
    forward = rng.poisson(np.clip(8000.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)
    backward = rng.poisson(np.clip(6000.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)
    _, _, err = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"binning_mode": "constant_error", "bin0_us": 0.5},
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.0,
        first_good_bin=t0,
        last_good_bin=n - 1,
    )
    # All complete bins (the final bin is truncated by the window) hold the
    # same statistics, so the asymmetry error stays flat.
    complete = err[:-1]
    assert complete.size >= 4
    assert complete.max() / complete.min() < 2.5
    # Contrast: raw binning errors grow by orders of magnitude over 14 τ.


def test_counts_then_ratio_handles_empty_late_bins():
    """Late raw bins hold zero counts; the binned asymmetry must stay finite."""
    rng = np.random.default_rng(2)
    n, t0 = 2000, 5
    t = (np.arange(n) - t0) * BIN_WIDTH_US
    decay = np.exp(-np.clip(t, 0.0, None) / MUON_LIFETIME_US) * (t >= 0)
    forward = rng.poisson(np.clip(500.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)
    backward = rng.poisson(np.clip(400.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)
    time, asym, err = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"binning_mode": "constant_error", "bin0_us": 1.0},
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.0,
        first_good_bin=t0,
        last_good_bin=n - 1,
    )
    assert np.all(np.isfinite(asym))
    assert np.all(np.isfinite(err))
    assert time.size == asym.size == err.size


def test_fixed_mode_is_rejected_by_the_variable_path():
    with pytest.raises(ValueError, match="non-fixed"):
        binning_slice_edges(100, mode="fixed", bin_width_us=BIN_WIDTH_US, t_start_us=0.0)


def test_time_representation_honours_binning_mode():
    """Provenance invariant: raw histograms unchanged; only the reduced
    representation differs between modes."""
    rng = np.random.default_rng(3)
    n, t0 = 1500, 5
    t = (np.arange(n) - t0) * BIN_WIDTH_US
    decay = np.exp(-np.clip(t, 0.0, None) / MUON_LIFETIME_US) * (t >= 0)
    counts_f = rng.poisson(np.clip(4000.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)
    counts_b = rng.poisson(np.clip(3000.0 * decay * BIN_WIDTH_US, 0, None)).astype(float)

    def make_run(binning: dict) -> Run:
        return Run(
            run_number=7001,
            histograms=[
                Histogram(counts=counts_f.copy(), bin_width=BIN_WIDTH_US, t0_bin=t0),
                Histogram(counts=counts_b.copy(), bin_width=BIN_WIDTH_US, t0_bin=t0),
            ],
            metadata={"run_number": 7001},
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": t0,
                "last_good_bin": n - 1,
            }
            | binning,
        )

    rep = TimeFBAsymmetry()
    fixed_run = make_run({})
    fixed = rep.compute(fixed_run)[0]
    binned_run = make_run({"binning_mode": "constant_error", "bin0_us": 0.5})
    binned = rep.compute(binned_run)[0]

    assert binned.time.size < fixed.time.size
    assert np.all(np.diff(binned.time) > 0)
    # Raw histograms untouched by either reduction.
    np.testing.assert_array_equal(binned_run.histograms[0].counts, counts_f)
    np.testing.assert_array_equal(fixed_run.histograms[0].counts, counts_f)
    # Fixed path unchanged: matches the plain reduction bit-for-bit.
    assert fixed.time.size == n - t0


def test_binned_time_convention_matches_fixed_mode():
    """Review fix: the time stamps of binned output are the mean of the
    merged raw bins' (k − t0)·w stamps — identical to the fixed-mode path,
    so switching modes never shifts the time axis."""
    n, t0, w = 400, 10, BIN_WIDTH_US
    forward = np.full(n, 1000.0)
    backward = np.full(n, 1000.0)
    # bin0 below the raw width keeps the early output bins single raw bins
    # (the target width stays < w for ~τ·ln2): their stamps must equal the
    # fixed-mode (k − t0)·w exactly.
    time, _, _ = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"binning_mode": "constant_error", "bin0_us": 0.5 * w},
        common_t0=t0,
        bin_width_us=w,
        alpha=1.0,
        first_good_bin=t0,
        last_good_bin=n - 1,
    )
    fixed_stamps = (np.arange(t0, n) - t0) * w
    np.testing.assert_allclose(time[:20], fixed_stamps[:20])


def test_fixed_mode_bins_counts_before_ratio():
    """Fixed bunching sums counts per block, then forms one asymmetry —
    one-sided raw bins (F·B = 0) dissolve into the block instead of
    injecting sigma = 1 sentinels into a value-domain quadrature (the
    counts-then-ratio order WiMDA/musrfit/Mantid use)."""
    from asymmetry.core.transform import compute_asymmetry, rebin

    rng = np.random.default_rng(11)
    n, t0, factor = 1200, 0, 8
    t = np.arange(n) * BIN_WIDTH_US
    decay = np.exp(-t / MUON_LIFETIME_US)
    forward = rng.poisson(np.clip(3.0 * decay, 0, None)).astype(float)
    backward = rng.poisson(np.clip(2.5 * decay, 0, None)).astype(float)
    assert np.count_nonzero(forward * backward == 0) > 50  # sparse data

    time, asym, err = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"bunching_factor": factor},
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.0,
        first_good_bin=0,
        last_good_bin=n - 1,
    )

    trimmed = (n // factor) * factor
    f_packed = forward[:trimmed].reshape(-1, factor).sum(axis=1)
    b_packed = backward[:trimmed].reshape(-1, factor).sum(axis=1)
    asym_ref, err_ref = compute_asymmetry(f_packed, b_packed, alpha=1.0)
    np.testing.assert_allclose(asym, asym_ref)
    np.testing.assert_allclose(err, err_ref)

    # The old value-domain path (per-raw-bin asymmetry, then rebin) is
    # strictly noisier on this data: sentinel bins inflate its errors.
    asym_raw, err_raw = compute_asymmetry(forward, backward, alpha=1.0)
    _, _, err_old = rebin(t, asym_raw, err_raw, factor)
    populated = (f_packed > 0) & (b_packed > 0)
    assert np.median(err_old[populated] / err[populated]) > 1.05


def test_fixed_mode_factor_one_matches_plain_reduction():
    """bunching_factor 1 reproduces the per-raw-bin reduction exactly."""
    from asymmetry.core.transform import compute_asymmetry

    rng = np.random.default_rng(12)
    n, t0 = 600, 7
    forward = rng.poisson(400.0, n).astype(float)
    backward = rng.poisson(350.0, n).astype(float)
    time, asym, err = binned_fb_asymmetry(
        forward,
        backward,
        grouping={},
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.1,
        first_good_bin=t0,
        last_good_bin=n - 1,
    )
    asym_ref, err_ref = compute_asymmetry(forward[t0:], backward[t0:], alpha=1.1)
    np.testing.assert_allclose(asym, asym_ref)
    np.testing.assert_allclose(err, err_ref)
    np.testing.assert_allclose(time, (np.arange(t0, n) - t0) * BIN_WIDTH_US)


def test_fixed_mode_time_stamps_match_value_domain_rebin():
    """Switching a reduction to counts-first must not move the time axis:
    the stamps equal rebin()'s mean of the (k - t0)*w left-edge stamps."""
    from asymmetry.core.transform import rebin

    n, t0, factor = 500, 4, 10
    forward = np.full(n, 100.0)
    backward = np.full(n, 100.0)
    raw_time = (np.arange(t0, n) - t0) * BIN_WIDTH_US
    expected_time, _, _ = rebin(raw_time, raw_time * 0, raw_time * 0 + 1, factor)
    time, _, _ = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"bunching_factor": factor},
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.0,
        first_good_bin=t0,
        last_good_bin=n - 1,
    )
    np.testing.assert_allclose(time, expected_time)


def test_fixed_mode_bunching_reaches_time_representation():
    """TimeFBAsymmetry applies fixed bunching counts-first (fit input)."""
    from asymmetry.core.transform import compute_asymmetry

    rng = np.random.default_rng(13)
    n, t0, factor = 900, 5, 6
    counts_f = rng.poisson(5.0, n).astype(float)
    counts_b = rng.poisson(4.0, n).astype(float)
    run = Run(
        run_number=7002,
        histograms=[
            Histogram(counts=counts_f.copy(), bin_width=BIN_WIDTH_US, t0_bin=t0),
            Histogram(counts=counts_b.copy(), bin_width=BIN_WIDTH_US, t0_bin=t0),
        ],
        metadata={"run_number": 7002},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": t0,
            "last_good_bin": n - 1,
            "bunching_factor": factor,
        },
    )
    dataset = TimeFBAsymmetry().compute(run)[0]
    window = n - t0
    trimmed = (window // factor) * factor
    f_packed = counts_f[t0 : t0 + trimmed].reshape(-1, factor).sum(axis=1)
    b_packed = counts_b[t0 : t0 + trimmed].reshape(-1, factor).sum(axis=1)
    asym_ref, err_ref = compute_asymmetry(f_packed, b_packed, alpha=1.0)
    np.testing.assert_allclose(dataset.asymmetry, asym_ref)
    np.testing.assert_allclose(dataset.error, err_ref)


# --------------------------------------------------------------------------- #
# Exact-t0 bin-centre stamps (D4)
# --------------------------------------------------------------------------- #


def _stamp_case(grouping: dict, *, t0_time_us: float | None):
    n, t0 = 300, 10
    rng = np.random.default_rng(31)
    forward = rng.poisson(500.0, n).astype(float)
    backward = rng.poisson(450.0, n).astype(float)
    return binned_fb_asymmetry(
        forward,
        backward,
        grouping=grouping,
        common_t0=t0,
        bin_width_us=BIN_WIDTH_US,
        alpha=1.2,
        first_good_bin=t0 + 3,
        last_good_bin=n - 1,
        beta=1.0,
        t0_time_us=t0_time_us,
    )


@pytest.mark.parametrize(
    "grouping",
    [
        {},
        {"bunching_factor": 5},
        {"binning_mode": "constant_error", "bin0_us": 0.5},
    ],
    ids=["fixed_1", "fixed_5", "constant_error"],
)
def test_exact_t0_moves_every_stamp_by_the_sub_bin_amount(grouping):
    """An exact t0 0.2 bins *before* the bin centre moves every stamp +0.2·w.

    ``t0_time_us = (t0 + 0.3)·w`` against the bin-centre default ``(t0 + 0.5)·w``
    means ``t_k = (k + 0.5)·w − t0_time_us`` grows by exactly ``0.2·w`` — for
    merged bins too, since each is the mean of its members' stamps. The
    good-bin window and the asymmetry values themselves must not move.
    """
    t0 = 10
    base_time, base_asym, base_err = _stamp_case(grouping, t0_time_us=None)
    time, asym, err = _stamp_case(grouping, t0_time_us=(t0 + 0.3) * BIN_WIDTH_US)

    assert time.size == base_time.size
    np.testing.assert_allclose(time - base_time, 0.2 * BIN_WIDTH_US, rtol=0, atol=1e-15)
    # Only the axis moves: the same raw bins are selected and merged.
    np.testing.assert_array_equal(asym, base_asym)
    np.testing.assert_array_equal(err, base_err)


def test_bin_centre_default_reproduces_the_integer_bin_axis_bit_for_bit():
    """The acceptance gate: passing the fallback explicitly changes nothing."""
    t0 = 10
    base_time, _, _ = _stamp_case({"bunching_factor": 4}, t0_time_us=None)
    explicit, _, _ = _stamp_case({"bunching_factor": 4}, t0_time_us=(t0 + 0.5) * BIN_WIDTH_US)
    np.testing.assert_array_equal(explicit, base_time)


@pytest.mark.parametrize("packing", [1, 5])
def test_fixed_bunching_stamps_match_the_musrfit_packing_oracle(packing):
    """musrfit ``PRunSingleHisto.cpp:1245-1275`` packing time, transcribed.

    ``t_k = dt·((fgb − 0.5) + p/2 − t0) + k·p·dt`` for an integer ``t0``. This
    pins the bin-centre convention against the reference program: our stamp is
    the mean of the merged raw bins' centres, which is the same number.
    """
    n, t0, fgb = 400, 12, 20
    dt = BIN_WIDTH_US
    forward = np.full(n, 900.0)
    backward = np.full(n, 800.0)
    time, _, _ = binned_fb_asymmetry(
        forward,
        backward,
        grouping={"bunching_factor": packing},
        common_t0=t0,
        bin_width_us=dt,
        alpha=1.0,
        first_good_bin=fgb,
        last_good_bin=n - 1,
    )
    k = np.arange(time.size, dtype=np.float64)
    musrfit = dt * ((fgb - 0.5) + packing / 2.0 - t0) + k * packing * dt
    np.testing.assert_allclose(time, musrfit, rtol=0, atol=1e-12)


def _exact_t0_run(*, hist_t0_us: float | None, grouping_extra: dict | None = None) -> Run:
    n, t0 = 300, 8
    rng = np.random.default_rng(32)
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": t0,
        "first_good_bin": t0,
        "last_good_bin": n - 1,
    }
    grouping.update(grouping_extra or {})
    return Run(
        run_number=7100,
        histograms=[
            Histogram(
                counts=rng.poisson(500.0, n).astype(float),
                bin_width=BIN_WIDTH_US,
                t0_bin=t0,
                t0_time_us=hist_t0_us,
            ),
            Histogram(
                counts=rng.poisson(450.0, n).astype(float),
                bin_width=BIN_WIDTH_US,
                t0_bin=t0,
                t0_time_us=hist_t0_us,
            ),
        ],
        metadata={"run_number": 7100},
        grouping=grouping,
    )


@pytest.mark.parametrize("via", ["histograms", "grouping"])
def test_reduction_axis_follows_the_runs_exact_t0(via):
    """The exact t0 reaches the reduced dataset from either source.

    Detector ``t0_time_us`` (a loader fact) and ``grouping["t0_time_us"]`` (the
    same fact after a T0Policy shift) both resolve through
    ``common_t0_time_us``; the grouping wins when both are set.
    """
    exact = (8 + 0.3) * BIN_WIDTH_US
    base = TimeFBAsymmetry().compute(_exact_t0_run(hist_t0_us=None))[0]
    if via == "histograms":
        shifted = TimeFBAsymmetry().compute(_exact_t0_run(hist_t0_us=exact))[0]
    else:
        shifted = TimeFBAsymmetry().compute(
            _exact_t0_run(hist_t0_us=None, grouping_extra={"t0_time_us": exact})
        )[0]
    assert base.time[0] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(shifted.time - base.time, 0.2 * BIN_WIDTH_US, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(shifted.asymmetry, base.asymmetry)
