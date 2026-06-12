"""Tests for fit_capture_fb_alpha (WP2.1).

Verification-plan §2 case 2d: simultaneous F+B fit with shared amplitudes and
free α. Synthesise using simulate_capture_run (once per group), combine the
per-group histograms into a single Run, and assert α recovery within 3σ.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.fit_quality import assess_fit_quality
from asymmetry.core.negmu.fit import CaptureModelSpec, fit_capture_fb_alpha
from asymmetry.core.negmu.model import CaptureComponent
from asymmetry.core.simulate import simulate_capture_run

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

N_BINS = 1024
BIN_WIDTH = 0.016  # μs


def _make_fb_template() -> Run:
    """Two-detector template: group 1 = det 1 (forward), group 2 = det 2 (backward)."""
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS, dtype=float),
            bin_width=BIN_WIDTH,
            t0_bin=0,
            good_bin_start=0,
            good_bin_end=N_BINS - 1,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "t_good_offset": 0,
        "first_good_bin": 0,
        "last_good_bin": N_BINS - 1,
        "bin_index_base": 1,
        "bunching_factor": 1,
        "good_frames": 1.0,
        "deadtime_correction": False,
        "dead_time_us": [0.0, 0.0],
        "included_groups": {1: True, 2: True},
    }
    return Run(
        run_number=0,
        histograms=histograms,
        metadata={"title": "Capture F+B test template"},
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


def _combine_groups(run_f: Run, run_b: Run, template: Run) -> Run:
    """Merge per-group simulation runs into one run.

    run_f carries the forward signal in histogram 0 (group 1).
    run_b carries the backward signal in histogram 1 (group 2).
    """
    h_fwd = run_f.histograms[0]
    h_bwd = run_b.histograms[1]
    combined = [
        Histogram(
            counts=h_fwd.counts.copy(),
            bin_width=h_fwd.bin_width,
            t0_bin=h_fwd.t0_bin,
            good_bin_start=h_fwd.good_bin_start,
            good_bin_end=h_fwd.good_bin_end,
        ),
        Histogram(
            counts=h_bwd.counts.copy(),
            bin_width=h_bwd.bin_width,
            t0_bin=h_bwd.t0_bin,
            good_bin_start=h_bwd.good_bin_start,
            good_bin_end=h_bwd.good_bin_end,
        ),
    ]
    return Run(
        run_number=0,
        histograms=combined,
        metadata={"title": "Capture F+B combined"},
        grouping=template.grouping,
        source_file="",
    )


# ---------------------------------------------------------------------------
# Case 2d fixture — C + O + decayBG, α = 1.25
# ---------------------------------------------------------------------------

COMPS_2D = [
    CaptureComponent(label="C", tau_us=2.030),
    CaptureComponent(label="O", tau_us=1.795),
    CaptureComponent(label="decayBG", tau_us=2.1969811),
]
WEIGHTS_2D = {"C": 5.0, "O": 3.0, "decayBG": 2.0}
TOTAL_2D = 2.0e7
ALPHA_TRUE = 1.25


@pytest.fixture(scope="module")
def result_2d():
    template = _make_fb_template()
    # Simulate F group (group 1): events scaled by sqrt(alpha) — more events forward.
    run_f = simulate_capture_run(
        template,
        COMPS_2D,
        WEIGHTS_2D,
        total_events=TOTAL_2D * np.sqrt(ALPHA_TRUE),
        group_id=1,
        seed=10,
        background_per_bin=0.0,
    )
    # Simulate B group (group 2): events scaled by 1/sqrt(alpha).
    run_b = simulate_capture_run(
        template,
        COMPS_2D,
        WEIGHTS_2D,
        total_events=TOTAL_2D / np.sqrt(ALPHA_TRUE),
        group_id=2,
        seed=11,
        background_per_bin=0.0,
    )
    combined_run = _combine_groups(run_f, run_b, template)
    ds = _make_dataset(combined_run)
    spec = CaptureModelSpec(elements=("C", "O"), include_decay_background=True)
    return fit_capture_fb_alpha(
        ds,
        forward_group=1,
        backward_group=2,
        spec=spec,
        alpha_seed=1.0,
        cost="poisson",
    )


# ---------------------------------------------------------------------------
# Case 2d assertions
# ---------------------------------------------------------------------------


def test_2d_success(result_2d):
    assert result_2d.success is True


def test_2d_has_both_groups(result_2d):
    assert 1 in result_2d.group_results
    assert 2 in result_2d.group_results


def test_2d_alpha_recovered(result_2d):
    """Recovered α within 3σ of 1.25 and |Δ| < 0.05."""
    shared = result_2d.shared_parameters
    result_any = result_2d.group_results[1]
    alpha = shared["alpha"].value
    sigma_alpha = result_any.uncertainties.get("alpha", 0.0)

    assert abs(alpha - ALPHA_TRUE) <= 3.0 * sigma_alpha, (
        f"alpha={alpha:.4f}, true={ALPHA_TRUE}, 3σ={3 * sigma_alpha:.4f}"
    )
    assert abs(alpha - ALPHA_TRUE) < 0.05, (
        f"alpha={alpha:.4f} deviates from true {ALPHA_TRUE} by more than 0.05"
    )


def test_2d_alpha_in_shared(result_2d):
    assert "alpha" in result_2d.shared_parameters


def test_2d_alpha_not_in_shared_as_bg(result_2d):
    assert "bg_F" not in result_2d.shared_parameters
    assert "bg_B" not in result_2d.shared_parameters


def test_2d_shared_amplitude_ratio(result_2d):
    """Shared amp_C/amp_O within tolerance — same physics as case 2a."""
    _dt = BIN_WIDTH
    _t_window = N_BINS * BIN_WIDTH

    def _af(tau: float) -> float:
        return (1 - np.exp(-_dt / tau)) / (1 - np.exp(-_t_window / tau))

    true_ratio = (5.0 * _af(2.030)) / (3.0 * _af(1.795))

    shared = result_2d.shared_parameters
    amp_c = shared["amp_C"].value
    amp_o = shared["amp_O"].value
    ratio = amp_c / amp_o

    result_any = result_2d.group_results[1]
    sigma_c = result_any.uncertainties.get("amp_C", 0.0)
    sigma_o = result_any.uncertainties.get("amp_O", 0.0)
    sigma_ratio = true_ratio * np.sqrt((sigma_c / amp_c) ** 2 + (sigma_o / amp_o) ** 2)

    assert abs(ratio - true_ratio) <= 3.0 * sigma_ratio, (
        f"ratio={ratio:.4f}, true={true_ratio:.4f}, 3σ={3 * sigma_ratio:.4f}"
    )
    assert abs(ratio - true_ratio) / true_ratio < 0.10, (
        f"ratio={ratio:.4f} not within 10% of true {true_ratio:.4f}"
    )


def test_2d_quality_not_poor(result_2d):
    """Combined fit converged; per-side chi-squared is not 'poor'."""
    for gid, res in result_2d.group_results.items():
        quality = assess_fit_quality(res.chi_squared, res.dof)
        assert quality.verdict != "poor", (
            f"Group {gid}: unexpected 'poor' quality "
            f"(χ²ᵣ={quality.chi2_reduced:.3f}, dof={quality.dof})"
        )


def test_2d_per_group_residuals(result_2d):
    """Each group result has per-side residuals."""
    for gid, res in result_2d.group_results.items():
        assert res.residuals is not None
        assert len(res.residuals) == N_BINS, f"Group {gid}: wrong residual length"


def test_2d_distinct_groups_raise():
    """fit_capture_fb_alpha rejects forward_group == backward_group."""
    from asymmetry.core.data.dataset import MuonDataset

    dummy_ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=None
    )
    spec = CaptureModelSpec(elements=("C",))
    with pytest.raises(ValueError, match="distinct"):
        fit_capture_fb_alpha(dummy_ds, forward_group=1, backward_group=1, spec=spec)


def test_2d_invalid_cost_raises():
    from asymmetry.core.data.dataset import MuonDataset

    dummy_ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=None
    )
    spec = CaptureModelSpec(elements=("C",))
    with pytest.raises(ValueError, match="cost"):
        fit_capture_fb_alpha(dummy_ds, forward_group=1, backward_group=2, spec=spec, cost="bad")
