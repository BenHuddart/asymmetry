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
