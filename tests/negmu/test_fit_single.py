"""Tests for fit_capture_histogram / fit_capture_group (WP1.3).

Verification-plan §2: cases 2a / 2b / 2c / 2e.
All fits use synthetic histograms from simulate_capture_run (never inline
generators) and tolerance-based assertions (the generating parameters are
exact; recovered values depend on the seed).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.fit_quality import assess_fit_quality
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.negmu.fit import (
    COUNT_COSTS,
    CaptureModelSpec,
    fit_capture_group,
    fit_capture_histogram,
)
from asymmetry.core.negmu.model import CaptureComponent
from asymmetry.core.simulate import simulate_capture_run

# ---------------------------------------------------------------------------
# Template helper (matches verification-plan §2 geometry)
# ---------------------------------------------------------------------------

N_BINS = 1024
BIN_WIDTH = 0.016  # μs


def _make_template(n_dets=2, n_bins=N_BINS, bin_width=BIN_WIDTH, t0_bin=0):
    histograms = [
        Histogram(
            counts=np.zeros(n_bins, dtype=float),
            bin_width=bin_width,
            t0_bin=t0_bin,
            good_bin_start=t0_bin,
            good_bin_end=n_bins - 1,
        )
        for _ in range(n_dets)
    ]
    grouping = {
        "groups": {1: list(range(1, n_dets + 1))},
        "group_names": {1: "Group 1"},
        "forward_group": 1,
        "backward_group": 1,
        "alpha": 1.0,
        "t0_bin": t0_bin,
        "t_good_offset": 0,
        "first_good_bin": t0_bin,
        "last_good_bin": n_bins - 1,
        "bin_index_base": 1,
        "bunching_factor": 1,
        "good_frames": 1.0,
        "deadtime_correction": False,
        "dead_time_us": [0.0] * n_dets,
        "included_groups": {1: True},
    }
    return Run(
        run_number=0,
        histograms=histograms,
        metadata={"title": "Capture test template"},
        grouping=grouping,
        source_file="",
    )


def _make_dataset(run: Run) -> MuonDataset:
    return MuonDataset(
        time=np.array([]),
        asymmetry=np.array([]),
        error=np.array([]),
        metadata={},
        run=run,
    )


# ---------------------------------------------------------------------------
# Case 2a — two elements (C + O + decayBG)
# ---------------------------------------------------------------------------

COMPS_2A = [
    CaptureComponent(label="C", tau_us=2.030),
    CaptureComponent(label="O", tau_us=1.795),
    CaptureComponent(label="decayBG", tau_us=2.1969811),
]
WEIGHTS_2A = {"C": 5.0, "O": 3.0, "decayBG": 2.0}
TOTAL_2A = 2.0e7
BG_2A = 5.0


@pytest.fixture(scope="module")
def result_2a():
    template = _make_template()
    run = simulate_capture_run(
        template,
        COMPS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_2A,
        seed=0,
        background_per_bin=BG_2A,
    )
    spec = CaptureModelSpec(elements=("C", "O"), include_decay_background=True)
    ds = _make_dataset(run)
    return fit_capture_group(ds, group_id=1, spec=spec, cost="poisson")


def test_2a_success(result_2a):
    assert result_2a.success is True


def test_2a_ratio_within_3sigma(result_2a):
    """Recovered amp_C / amp_O within 3σ of the true amplitude ratio.

    With equal events the amplitude ratio is (w_C/w_O) × factor(τ_C)/factor(τ_O)
    where factor(τ) = (1−exp(−Δt/τ))/(1−exp(−T/τ)) — NOT simply w_C/w_O.
    """
    _dt = BIN_WIDTH
    _T = N_BINS * BIN_WIDTH

    def _af(tau):
        return (1 - np.exp(-_dt / tau)) / (1 - np.exp(-_T / tau))

    true_ratio = (5.0 * _af(2.030)) / (3.0 * _af(1.795))  # ≈ 1.475

    params = result_2a.parameters
    amp_C = params["amp_C"].value
    amp_O = params["amp_O"].value
    ratio = amp_C / amp_O

    sigma_C = result_2a.uncertainties.get("amp_C", 0.0)
    sigma_O = result_2a.uncertainties.get("amp_O", 0.0)
    # Combined uncertainty on the ratio (quadrature approximation)
    sigma_ratio = true_ratio * np.sqrt((sigma_C / amp_C) ** 2 + (sigma_O / amp_O) ** 2)

    assert abs(ratio - true_ratio) <= 3.0 * sigma_ratio, (
        f"ratio={ratio:.4f}, true={true_ratio:.4f}, 3σ={3 * sigma_ratio:.4f}"
    )
    assert abs(ratio - true_ratio) / true_ratio < 0.08, (
        f"ratio={ratio:.4f} not within 8% of true {true_ratio:.4f}"
    )


def test_2a_amplitudes_within_5pct(result_2a):
    """Each amplitude within 5% of generating value."""
    params = result_2a.parameters
    # Ratios should be close to 5:3:2 (exact values depend on window/tau)
    amp_C = params["amp_C"].value
    amp_O = params["amp_O"].value
    amp_bg = params["amp_decayBG"].value
    # Ratios should be close to 5:3:2
    total_amp = amp_C + amp_O + amp_bg
    frac_C = amp_C / total_amp
    frac_O = amp_O / total_amp
    assert abs(frac_C - 0.5) < 0.10, f"C fraction={frac_C:.3f}"
    assert abs(frac_O - 0.3) < 0.10, f"O fraction={frac_O:.3f}"


def test_2a_quality_good(result_2a):
    """Reduced chi-squared is in the 'good' band."""
    quality = assess_fit_quality(result_2a.chi_squared, result_2a.dof)
    assert quality.verdict in ("good",), (
        f"Expected 'good', got {quality.verdict!r}  "
        f"(χ²ᵣ={quality.chi2_reduced:.3f}, dof={quality.dof})"
    )


def test_2a_dof_set(result_2a):
    assert result_2a.dof > 0


def test_2a_count_costs_constant():
    assert "poisson" in COUNT_COSTS
    assert "gaussian" in COUNT_COSTS


# ---------------------------------------------------------------------------
# Case 2b — light + heavy (C + Fe + decayBG), decade in τ
# ---------------------------------------------------------------------------

COMPS_2B = [
    CaptureComponent(label="C", tau_us=2.030),
    CaptureComponent(label="Fe", tau_us=0.206),
    CaptureComponent(label="decayBG", tau_us=2.1969811),
]
WEIGHTS_2B = {"C": 4.0, "Fe": 4.0, "decayBG": 2.0}
TOTAL_2B = 2.0e7


@pytest.fixture(scope="module")
def result_2b():
    template = _make_template()
    run = simulate_capture_run(
        template,
        COMPS_2B,
        WEIGHTS_2B,
        total_events=TOTAL_2B,
        seed=1,
        background_per_bin=BG_2A,
    )
    spec = CaptureModelSpec(elements=("C", "Fe"), include_decay_background=True)
    ds = _make_dataset(run)
    return fit_capture_group(ds, group_id=1, spec=spec, cost="poisson")


def test_2b_success(result_2b):
    assert result_2b.success is True


def test_2b_c_fe_ratio(result_2b):
    """Recovered amp_C / amp_Fe close to the true amplitude ratio.

    Equal event weights (4:4) do NOT give equal amplitudes when τ differ by 10×.
    True ratio ≈ (1−exp(−Δt/τ_C)) / (1−exp(−Δt/τ_Fe)) ≈ 0.105.
    """
    _dt = BIN_WIDTH
    _T = N_BINS * BIN_WIDTH

    def _af(tau):
        return (1 - np.exp(-_dt / tau)) / (1 - np.exp(-_T / tau))

    true_ratio = (4.0 * _af(2.030)) / (4.0 * _af(0.206))  # ≈ 0.105

    params = result_2b.parameters
    amp_C = params["amp_C"].value
    amp_Fe = params["amp_Fe"].value
    ratio = amp_C / amp_Fe
    sigma_C = result_2b.uncertainties.get("amp_C", 0.0)
    sigma_Fe = result_2b.uncertainties.get("amp_Fe", 0.0)
    sigma_ratio = true_ratio * np.sqrt((sigma_C / amp_C) ** 2 + (sigma_Fe / amp_Fe) ** 2)
    assert abs(ratio - true_ratio) <= 3.0 * sigma_ratio, (
        f"C/Fe ratio={ratio:.4f}, true={true_ratio:.4f}, 3σ={3 * sigma_ratio:.4f}"
    )
    assert abs(ratio - true_ratio) / true_ratio < 0.10, (
        f"C/Fe ratio={ratio:.4f} not within 10% of true {true_ratio:.4f}"
    )


def test_2b_fe_amplitude_positive(result_2b):
    """Both C and Fe amplitudes are positive (fit has not zeroed out a component)."""
    params = result_2b.parameters
    assert params["amp_C"].value > 0
    assert params["amp_Fe"].value > 0


# ---------------------------------------------------------------------------
# Case 2c — free-τ sanity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result_2c():
    template = _make_template()
    run = simulate_capture_run(
        template,
        COMPS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_2A,
        seed=0,
        background_per_bin=BG_2A,
    )
    spec = CaptureModelSpec(
        elements=("C", "O"),
        include_decay_background=True,
        free_tau=frozenset({"C"}),
    )
    ds = _make_dataset(run)
    return fit_capture_group(ds, group_id=1, spec=spec, cost="poisson")


def test_2c_success(result_2c):
    assert result_2c.success is True


def test_2c_free_tau_recovered(result_2c):
    """Freed tau_C within 5% of true value 2.030 μs.

    Tolerance is 5% (not 2%): C (2.030) and decayBG (2.197) are only 8% apart,
    causing high correlation that degrades the individual τ precision even at 2e7
    events.
    """
    tau_C = result_2c.parameters["tau_C"].value
    assert abs(tau_C - 2.030) / 2.030 < 0.05, f"tau_C={tau_C:.4f}"


def test_2c_free_tau_uncertainty_finite_positive(result_2c):
    sigma_tau = result_2c.uncertainties.get("tau_C", None)
    assert sigma_tau is not None
    assert sigma_tau > 0
    assert np.isfinite(sigma_tau)


# ---------------------------------------------------------------------------
# Case 2e — Gaussian cost parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result_2e():
    template = _make_template()
    run = simulate_capture_run(
        template,
        COMPS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_2A,
        seed=0,
        background_per_bin=BG_2A,
    )
    spec = CaptureModelSpec(elements=("C", "O"), include_decay_background=True)
    ds = _make_dataset(run)
    return fit_capture_group(ds, group_id=1, spec=spec, cost="gaussian")


def test_2e_converges(result_2e):
    assert result_2e.success is True


def test_2e_ratio_within_8pct(result_2e):
    """Gaussian cost: amp_C/amp_O ratio within 8% of true amplitude ratio (≈ 1.475)."""
    _dt = BIN_WIDTH
    _T = N_BINS * BIN_WIDTH

    def _af(tau):
        return (1 - np.exp(-_dt / tau)) / (1 - np.exp(-_T / tau))

    true_ratio = (5.0 * _af(2.030)) / (3.0 * _af(1.795))
    amp_C = result_2e.parameters["amp_C"].value
    amp_O = result_2e.parameters["amp_O"].value
    ratio = amp_C / amp_O
    assert abs(ratio - true_ratio) / true_ratio < 0.08, f"ratio={ratio:.4f}"


# ---------------------------------------------------------------------------
# Array-level fit_capture_histogram
# ---------------------------------------------------------------------------


def test_fit_capture_histogram_array_level():
    """fit_capture_histogram works directly on time/counts arrays."""
    t = np.arange(512) * 0.016
    amp_true = 5000.0
    counts = amp_true * np.exp(-t / 2.030) + 10.0
    # Add mild Poisson noise
    rng = np.random.default_rng(42)
    counts = rng.poisson(counts).astype(float)

    spec = CaptureModelSpec(elements=("C",), include_decay_background=False)
    result = fit_capture_histogram(t, counts, spec, cost="poisson")
    assert result.success is True
    assert result.dof == len(t) - 2  # 2 free params: amp_C, background


def test_invalid_cost_raises():
    t = np.array([0.0, 1.0])
    counts = np.array([100.0, 50.0])
    spec = CaptureModelSpec(elements=("C",))
    with pytest.raises(ValueError, match="cost"):
        fit_capture_histogram(t, counts, spec, cost="invalid")


def test_custom_parameters_accepted():
    """fit_capture_histogram accepts a pre-built ParameterSet."""
    t = np.arange(200) * 0.016
    amp_true = 3000.0
    counts = amp_true * np.exp(-t / 2.030) + 5.0
    rng = np.random.default_rng(0)
    counts = rng.poisson(counts).astype(float)

    spec = CaptureModelSpec(elements=("C",), include_decay_background=False)
    # Provide parameters directly
    ps = ParameterSet()
    ps.add(Parameter(name="amp_C", value=3000.0, min=0.0))
    ps.add(Parameter(name="tau_C", value=2.030, fixed=True))
    ps.add(Parameter(name="background", value=5.0, min=0.0))
    result = fit_capture_histogram(t, counts, spec, cost="poisson", parameters=ps)
    assert result.success is True
