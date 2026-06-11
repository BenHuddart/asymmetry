"""Count-domain fit modes: single-histogram and forward/backward (alpha-free).

Verification under the study's transcribe + synthetic + cross-check oracle
(``docs/porting/count-domain-fit-modes/verification-plan.md``). WiMDA is a
source-only oracle: the count models are checked against formulas transcribed
from ``AsymFitFunction.pas``, and parameter recovery is checked against
``core/simulate`` ground truth.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import (
    COUNT_COSTS,
    _apply_deadtime,
    _percent_to_fraction,
    _raw_model,
    fb_overlay_curves,
    fit_fb_alpha,
    fit_single_histogram,
    single_histogram_overlay,
)
from asymmetry.core.fitting.grouped_time_domain import (
    build_count_group,
    build_fb_count_model,
    build_grouped_count_model,
    build_grouped_time_domain_groups,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.simulate import (
    build_builtin_template,
    simulate_double_pulse_run,
    simulate_run,
)
from asymmetry.core.transform.deadtime import promote_deadtime_to_grouping
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _tf(t, A=20.0, f=1.5, phi=0.0):  # noqa: N803 (A is the conventional asymmetry symbol)
    """Transverse-field precession asymmetry in percent."""
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


def _pulsed_tf_run(*, alpha=1.25, seed=1, total_events=40e6, background_per_bin=0.0):
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.5, "phi": 0.3},
        total_events=total_events,
        alpha=alpha,
        background_per_bin=background_per_bin,
        seed=seed,
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def _continuous_run(*, seed=2):
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6,
        alpha=1.0,
        background_per_bin=10.0,
        seed=seed,
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


# --- model transcription / identity ----------------------------------------


def test_raw_model_is_lifetime_corrected_times_decay():
    """N_raw(t) = exp(-t/tau) * [lifetime-corrected model]."""
    grouped = build_grouped_count_model(_percent_to_fraction(_tf))
    raw = _raw_model(grouped)
    t = np.linspace(0.05, 10.0, 64)
    kw = dict(N0=1.0e5, background=12.0, amplitude=1.0, relative_phase=0.0, A=20.0, f=1.5, phi=0.3)
    corrected = grouped(t, **kw)
    np.testing.assert_allclose(raw(t, **kw), np.exp(-t / MUON_LIFETIME_US) * corrected, rtol=1e-12)


def test_fb_count_model_matches_wimda_transcription():
    """fgFB: forward = N0*sqrt(alpha)*(1+a), backward = N0/sqrt(alpha)*(1-a), + bg*exp(t/tau)."""
    fb = build_fb_count_model(_percent_to_fraction(_tf))
    t = np.linspace(0.05, 8.0, 40)
    alpha, n0 = 1.3, 5.0e4
    bg_f, bg_b = 7.0, 9.0
    a = 0.01 * _tf(t, A=20.0, f=1.5, phi=0.3)
    env = np.exp(t / MUON_LIFETIME_US)
    expect_f = n0 * np.sqrt(alpha) * (1.0 + a) + bg_f * env
    expect_b = n0 / np.sqrt(alpha) * (1.0 - a) + bg_b * env
    got_f = fb(t, alpha=alpha, N0=n0, background=bg_f, sign=+1.0, A=20.0, f=1.5, phi=0.3)
    got_b = fb(t, alpha=alpha, N0=n0, background=bg_b, sign=-1.0, A=20.0, f=1.5, phi=0.3)
    np.testing.assert_allclose(got_f, expect_f, rtol=1e-12)
    np.testing.assert_allclose(got_b, expect_b, rtol=1e-12)


def test_raw_count_groups_are_corrected_divided_by_decay():
    """The lifetime_corrected=False group is the corrected counts times exp(-t/tau)."""
    ds = _pulsed_tf_run()
    corrected = build_count_group(ds, 1, lifetime_corrected=True)
    raw = build_count_group(ds, 1, lifetime_corrected=False)
    np.testing.assert_allclose(
        raw.counts, corrected.counts * np.exp(-corrected.time / MUON_LIFETIME_US), rtol=1e-9
    )


# --- forward/backward alpha fit ---------------------------------------------


@pytest.mark.parametrize("cost", COUNT_COSTS)
@pytest.mark.parametrize("alpha_true", [0.8, 1.0, 1.3])
def test_fb_alpha_recovers_known_alpha(cost, alpha_true):
    ds = _pulsed_tf_run(alpha=alpha_true, seed=11)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost=cost)
    assert result.success
    fwd = result.group_results[1]
    alpha_fit = fwd.parameters["alpha"].value
    alpha_err = fwd.uncertainties["alpha"]
    assert abs(alpha_fit - alpha_true) < 5.0 * alpha_err
    assert abs(alpha_fit - alpha_true) < 0.02
    # alpha = N0_F / N0_B by construction.
    n0 = result.shared_parameters["N0"].value
    n0_f = n0 * np.sqrt(alpha_fit)
    n0_b = n0 / np.sqrt(alpha_fit)
    assert n0_f / n0_b == pytest.approx(alpha_fit, rel=1e-9)
    # amplitude recovered (true 20%).
    assert fwd.parameters["A"].value == pytest.approx(20.0, abs=0.3)


def test_fb_alpha_reports_amplitude_correlation():
    ds = _pulsed_tf_run(alpha=1.25, seed=5)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    fwd = result.group_results[1]
    assert fwd.covariance is not None
    names = fwd.covariance_parameters
    assert "alpha" in names and "A" in names
    cov = fwd.covariance
    i, j = names.index("alpha"), names.index("A")
    rho = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    assert np.isfinite(rho) and abs(rho) <= 1.0


def test_fb_requires_background_b():
    ds = _pulsed_tf_run()
    params = ParameterSet(
        [Parameter("alpha", 1.0), Parameter("N0", 1.0e5), Parameter("background", 0.0)]
    )
    with pytest.raises(ValueError, match="background_b"):
        fit_fb_alpha(ds, 1, 2, _tf, params)


# --- single histogram -------------------------------------------------------


@pytest.mark.parametrize("cost", COUNT_COSTS)
def test_single_histogram_recovers_envelope(cost):
    ds = _continuous_run()
    params = ParameterSet(
        [
            Parameter("N0", 4.0e3, min=0.0),
            Parameter("background", 9.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
        ]
    )
    result = fit_single_histogram(ds, 1, _tf, params, side="forward", cost=cost)
    assert result.success
    assert result.parameters["background"].value == pytest.approx(10.0, abs=1.5)
    assert result.parameters["A"].value == pytest.approx(20.0, abs=0.5)
    assert result.parameters["f"].value == pytest.approx(1.0, abs=0.01)


def test_single_histogram_backward_sign():
    """Backward side flips the asymmetry sign: forward+backward asymmetry cancels."""
    ds = _pulsed_tf_run(alpha=1.0, seed=7)
    base = dict(min=0.0)
    common = [Parameter("A", 18.0, min=0.0, max=50.0), Parameter("f", 1.5), Parameter("phi", 0.3)]
    pf = ParameterSet([Parameter("N0", 1.5e5, **base), Parameter("background", 0.0), *common])
    rf = fit_single_histogram(ds, 1, _tf, pf, side="forward", cost="gaussian")
    pb = ParameterSet([Parameter("N0", 1.5e5, **base), Parameter("background", 0.0), *common])
    rb = fit_single_histogram(ds, 2, _tf, pb, side="backward", cost="gaussian")
    assert rf.success and rb.success
    # Both sides see the same physics amplitude (sign handled by the model).
    assert rf.parameters["A"].value == pytest.approx(20.0, abs=0.4)
    assert rb.parameters["A"].value == pytest.approx(20.0, abs=0.4)


def test_poisson_recovers_from_rough_start_on_low_counts():
    """From a deliberately rough start on a low-count run, Poisson recovers truth.

    Demonstrates the count-domain motivation: the Poisson likelihood surface is
    more forgiving than sqrt(N) least squares where late-time bins are sparse.
    """
    ds = _continuous_run(seed=3)
    rough = lambda: ParameterSet(  # noqa: E731
        [
            Parameter("N0", 1.0e3, min=0.0),
            Parameter("background", 5.0, min=0.0),
            Parameter("A", 14.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
        ]
    )
    poisson = fit_single_histogram(ds, 1, _tf, rough(), side="forward", cost="poisson")
    gaussian = fit_single_histogram(ds, 1, _tf, rough(), side="forward", cost="gaussian")
    assert poisson.success
    assert poisson.parameters["A"].value == pytest.approx(20.0, abs=0.6)
    # Poisson lands at least as close to the true amplitude as Gaussian.
    poisson_err = abs(poisson.parameters["A"].value - 20.0)
    gaussian_err = abs(gaussian.parameters["A"].value - 20.0)
    assert poisson_err <= gaussian_err + 1e-6


# --- guards -----------------------------------------------------------------


def test_unknown_cost_raises():
    ds = _pulsed_tf_run()
    params = ParameterSet([Parameter("N0", 1.0e5), Parameter("background", 0.0)])
    with pytest.raises(ValueError, match="Unknown count-fit cost"):
        fit_single_histogram(ds, 1, _tf, params, cost="median")


def test_single_group_builder_bypasses_two_group_guard():
    """build_count_group serves one group even though grouped fitting needs two."""
    ds = _pulsed_tf_run()
    group = build_count_group(ds, 1, lifetime_corrected=False)
    assert group.counts.size > 0
    # The multi-group builder still requires two included groups.
    groups = build_grouped_time_domain_groups(ds, lifetime_corrected=False)
    assert len(groups) == 2


# --- Phase 2: exclude range, fittable t0, baseline drift --------------------


def _single_seed():
    return ParameterSet(
        [
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0, min=0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )


def _inject_forward_artefact(run, *, lo_offset=300, hi_offset=330, spike=5e4):
    """Add a big count spike to forward detectors over an interior bin window."""
    for hist in run.histograms[:32]:
        counts = np.asarray(hist.counts, dtype=float)
        lo = int(hist.t0_bin) + lo_offset
        hi = int(hist.t0_bin) + hi_offset
        counts[lo:hi] += spike
        hist.counts = counts
    bin_width = float(run.histograms[0].bin_width)
    return lo_offset * bin_width, (hi_offset + 1) * bin_width


def test_exclude_range_recovers_clean_fit_under_artefact():
    clean = fit_single_histogram(
        _pulsed_tf_run(alpha=1.0, seed=1), 1, _tf, _single_seed(), cost="gaussian"
    )

    run = _pulsed_tf_run(alpha=1.0, seed=1).run
    ex0, ex1 = _inject_forward_artefact(run)
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    pulled = fit_single_histogram(ds, 1, _tf, _single_seed(), cost="gaussian")
    masked = fit_single_histogram(ds, 1, _tf, _single_seed(), cost="gaussian", exclude=(ex0, ex1))

    clean_a = clean.parameters["A"].value
    # The artefact pulls the un-masked fit; excluding it recovers the clean value.
    assert abs(pulled.parameters["A"].value - clean_a) > 0.5
    assert masked.parameters["A"].value == pytest.approx(clean_a, abs=0.1)


def test_exclude_drops_interior_bins():
    ds = _pulsed_tf_run()
    full = build_count_group(ds, 1, lifetime_corrected=False)
    ex0, ex1 = 2.0, 4.0
    masked = build_count_group(ds, 1, lifetime_corrected=False, exclude=(ex0, ex1))
    assert masked.time.size < full.time.size
    assert not np.any((masked.time >= ex0) & (masked.time <= ex1))
    # Bins outside the window are untouched.
    assert masked.time.min() == pytest.approx(full.time.min())


def test_fittable_t0_off_state_is_noop():
    ds = _pulsed_tf_run(alpha=1.0, seed=1)
    base = fit_single_histogram(ds, 1, _tf, _single_seed(), cost="gaussian")
    params = _single_seed()
    params.add(Parameter("t0", 0.0, fixed=True))
    fixed_t0 = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert fixed_t0.parameters["A"].value == pytest.approx(base.parameters["A"].value, abs=1e-6)


def test_fittable_t0_is_unbiased_on_clean_data():
    ds = _pulsed_tf_run(alpha=1.0, seed=1)
    params = _single_seed()
    params.add(Parameter("t0", 0.0, min=-0.05, max=0.05))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["t0"].value == pytest.approx(0.0, abs=0.02)


def test_baseline_drift_recovered():
    def _tf_drift(t, A=20.0, f=1.5, phi=0.0, lam0=0.3, beta0=1.0):  # noqa: N803
        env = np.exp(-((lam0 * np.asarray(t, dtype=float)) ** beta0))
        return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi) * env

    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template,
        _tf_drift,
        {"A": 20.0, "f": 1.5, "phi": 0.3, "lam0": 0.3, "beta0": 1.0},
        total_events=40e6,
        alpha=1.0,
        seed=4,
    )
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    params = _single_seed()
    params.add(Parameter("lambda_base", 0.1, min=0.0, max=5.0))
    params.add(Parameter("beta_base", 1.0, min=0.2, max=3.0, fixed=True))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["lambda_base"].value == pytest.approx(0.3, abs=0.03)
    assert result.parameters["A"].value == pytest.approx(20.0, abs=0.5)


def test_baseline_drift_off_state_is_noop():
    ds = _pulsed_tf_run(alpha=1.0, seed=1)
    base = fit_single_histogram(ds, 1, _tf, _single_seed(), cost="gaussian")
    params = _single_seed()
    params.add(Parameter("lambda_base", 0.0, fixed=True))
    params.add(Parameter("beta_base", 1.0, fixed=True))
    with_terms = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert with_terms.parameters["A"].value == pytest.approx(base.parameters["A"].value, abs=1e-6)


def test_fb_alpha_accepts_exclude_and_t0():
    ds = _pulsed_tf_run(alpha=1.2, seed=8)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
            Parameter("t0", 0.0, min=-0.05, max=0.05),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian", exclude=(2.0, 3.0))
    assert result.success
    assert result.group_results[1].parameters["alpha"].value == pytest.approx(1.2, abs=0.02)


# --- Phase 3: count loss (deadtime) + promote + double pulse -----------------


def _continuous_single(*, seed=2, n0=4.5e3):
    return ParameterSet(
        [
            Parameter("N0", n0, min=0.0),
            Parameter("background", 10.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
        ]
    )


def test_deadtime_dt0_recovered():
    """Inject a known non-paralyzable deadtime, recover DT0 from the fit."""
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6,
        alpha=1.0,
        background_per_bin=10.0,
        seed=2,
    )
    good_frames = 1.0e6
    run.grouping["good_frames"] = good_frames
    bin_width = float(run.histograms[0].bin_width)
    dt0_true = 0.01
    hist = run.histograms[0]
    n_true = np.asarray(hist.counts, dtype=float)
    # Single-detector group, so frame_norm = bin_width * good_frames.
    hist.counts = n_true * (1.0 - n_true * dt0_true / (bin_width * good_frames))
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    params = _continuous_single()
    params.add(Parameter("DT0", 0.005, min=0.0, max=0.1))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["DT0"].value == pytest.approx(dt0_true, abs=0.0015)


def _retau_forward_histogram(run, tau_inject):
    """Rescale the forward detector's decay envelope to a chosen (non-physical) tau.

    The run is generated at the physical lifetime with no flat background, so the
    counts are ``N0·exp(-t/τ_phys)·(1+P)``; multiplying bin ``i ≥ t0`` by
    ``exp(t_i·(1/τ_phys − 1/τ_inject))`` converts the envelope to ``τ_inject``
    while leaving the polarization untouched.
    """
    tau_phys = float(MUON_LIFETIME_US)
    hist = run.histograms[0]
    bin_width = float(hist.bin_width)
    t0 = max(0, int(hist.t0_bin))
    counts = np.asarray(hist.counts, dtype=float).copy()
    idx = np.arange(counts.size, dtype=float)
    t = np.maximum(idx - t0, 0.0) * bin_width
    counts *= np.exp(t * (1.0 / tau_phys - 1.0 / tau_inject))
    hist.counts = counts


def test_free_lifetime_recovers_injected_tau():
    """A free tau recovers an injected non-physical muon lifetime (musrfit-style)."""
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=40e6, alpha=1.0, background_per_bin=0.0, seed=2,
    )  # fmt: skip
    tau_inject = 2.5  # μs, well away from the physical 2.197 μs
    _retau_forward_histogram(run, tau_inject)
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    params = ParameterSet(
        [
            Parameter("N0", 4.5e3, min=0.0),
            Parameter("background", 0.0, fixed=True),  # no flat background in this run
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
            Parameter("tau", float(MUON_LIFETIME_US), min=1.0, max=4.0),
        ]
    )
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["tau"].value == pytest.approx(tau_inject, abs=0.03)
    assert result.parameters["A"].value == pytest.approx(20.0, abs=0.6)


def test_free_lifetime_off_state_equivalent_to_fixed_physical():
    """tau fixed at the physical value reproduces the no-tau fit (raw-model branch)."""
    ds = _continuous_run()
    base = fit_single_histogram(ds, 1, _tf, _continuous_single(), cost="gaussian")
    params = _continuous_single()
    params.add(Parameter("tau", float(MUON_LIFETIME_US), fixed=True))
    fixed_tau = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert fixed_tau.parameters["A"].value == pytest.approx(base.parameters["A"].value, rel=1e-6)
    assert fixed_tau.parameters["background"].value == pytest.approx(
        base.parameters["background"].value, rel=1e-6
    )


def test_fb_free_lifetime_reported_and_recovered():
    """The F+B fit accepts a free tau, recovers it, and reports it as shared."""
    ds = _pulsed_tf_run(alpha=1.2, seed=11, background_per_bin=0.0)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0, fixed=True),
            Parameter("background_b", 0.0, fixed=True),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
            Parameter("tau", 2.0, min=1.0, max=4.0),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    assert result.success
    # tau is a shared parameter (excluded from the per-side background slots).
    assert result.shared_parameters["tau"].value == pytest.approx(float(MUON_LIFETIME_US), abs=0.02)
    assert result.group_results[1].parameters["alpha"].value == pytest.approx(1.2, abs=0.03)


def test_free_lifetime_rejected_with_double_pulse():
    ds = _continuous_run()
    params = _continuous_single()
    params.add(Parameter("tau", float(MUON_LIFETIME_US), min=1.0, max=4.0))
    params.add(Parameter("dpsep", 0.324, fixed=True))
    with pytest.raises(ValueError, match="Free muon lifetime is not supported"):
        fit_single_histogram(ds, 1, _tf, params)


def test_deadtime_off_state_is_noop():
    ds = _continuous_run()
    base = fit_single_histogram(ds, 1, _tf, _continuous_single(), cost="gaussian")
    params = _continuous_single()
    params.add(Parameter("DT0", 0.0, fixed=True))
    with_dt = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert with_dt.parameters["A"].value == pytest.approx(base.parameters["A"].value, abs=1e-6)


# --- deadtime loss-form transcription (AsymFitFunction.pas:280-314) ----------


def test_deadtime_simple_form_transcription():
    counts = np.array([1.0e4, 5.0e3, 2.0e3])
    frame_norm, dt0 = 1.0e5, 0.02
    qq = counts / frame_norm
    expected = counts * (1.0 - dt0 * qq)
    terms = (dt0, 0.0, 0.0, 0.0, 0.0)
    out = _apply_deadtime(
        counts, terms, frame_norm=frame_norm, time=counts, evfr=1.0, model="simple"
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_deadtime_linear_form_uses_event_fraction():
    counts = np.array([1.0e4, 5.0e3, 2.0e3])
    frame_norm, dt0, dt1, evfr = 1.0e5, 0.02, 0.5, 0.4
    qq = counts / frame_norm
    expected = counts * (1.0 - (dt0 + dt1 * evfr) * qq)
    terms = (dt0, dt1, 0.0, 0.0, 0.0)
    out = _apply_deadtime(
        counts, terms, frame_norm=frame_norm, time=counts, evfr=evfr, model="linear"
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_deadtime_polynomial_form_carries_wimda_decade_scalings():
    counts = np.array([1.0e4, 5.0e3, 2.0e3])
    frame_norm = 1.0e6  # keeps qq small so every loss term stays < 1 (no clip)
    dt0, c2, c3, c4 = 0.02, 0.003, 0.001, 0.0005
    qq = counts / frame_norm
    loss = dt0 * qq + c2 * 1e3 * qq**2 + c3 * 1e6 * qq**3 + c4 * 1e9 * qq**4
    assert np.all(loss < 1.0)
    expected = counts * (1.0 - loss)
    terms = (dt0, 0.0, c2, c3, c4)
    out = _apply_deadtime(
        counts, terms, frame_norm=frame_norm, time=counts, evfr=1.0, model="polynomial"
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_deadtime_power_form_transcription():
    counts = np.array([1.0e4, 5.0e3, 2.0e3])
    time = np.array([0.1, 1.0, 5.0])
    dt0, c2, c3, c4, evfr = 0.02, 1.5, 2.0, 0.3, 0.4
    tau = float(MUON_LIFETIME_US)
    loss = (evfr * dt0) ** c2 * np.exp(-((c4 * time / tau) ** c3))
    expected = counts * (1.0 - loss)
    terms = (dt0, 0.0, c2, c3, c4)
    out = _apply_deadtime(counts, terms, frame_norm=1.0e5, time=time, evfr=evfr, model="power")
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_deadtime_factor_clipped_nonnegative():
    """A runaway loss > 1 cannot drive the corrected counts negative."""
    counts = np.array([1.0e4])
    terms = (1.0e3, 0.0, 0.0, 0.0, 0.0)  # absurd DT0 -> loss >> 1
    out = _apply_deadtime(counts, terms, frame_norm=1.0, time=counts, evfr=1.0, model="simple")
    assert np.all(out >= 0.0)


def test_deadtime_all_zero_is_exact_noop():
    counts = np.array([1.0e4, 5.0e3])
    terms = (0.0, 0.0, 0.0, 0.0, 0.0)
    for model in ("simple", "linear", "polynomial", "power"):
        out = _apply_deadtime(counts, terms, frame_norm=1e5, time=counts, evfr=0.5, model=model)
        assert out is counts  # returned unchanged, not a copy


def test_unknown_deadtime_model_raises():
    ds = _continuous_run()
    with pytest.raises(ValueError, match="Unknown deadtime model"):
        fit_single_histogram(ds, 1, _tf, _continuous_single(), deadtime_model="quintic")


def test_deadtime_dt1_recovered_with_event_fraction():
    """With evfr metadata present, the linear form recovers an injected DT1."""
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6, alpha=1.0, background_per_bin=10.0, seed=2,
    )  # fmt: skip
    good_frames, evfr = 1.0e6, 0.5
    run.grouping["good_frames"] = good_frames
    run.grouping["event_fraction"] = evfr
    bin_width = float(run.histograms[0].bin_width)
    dt0_true, dt1_true = 0.004, 0.012
    hist = run.histograms[0]
    n_true = np.asarray(hist.counts, dtype=float)
    qq = n_true / (bin_width * good_frames)  # single-detector group
    hist.counts = n_true * (1.0 - (dt0_true + dt1_true * evfr) * qq)
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    params = _continuous_single()
    params.add(Parameter("DT0", dt0_true, fixed=True))  # pin DT0 to isolate DT1
    params.add(Parameter("DT1", 0.005, min=0.0, max=0.1))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian", deadtime_model="linear")
    assert result.success
    assert result.parameters["DT1"].value == pytest.approx(dt1_true, abs=0.003)


def test_promote_deadtime_replace_and_additive():
    grouping = {"dead_time_us": [0.0, 0.0]}
    out = promote_deadtime_to_grouping(
        grouping, 0.012, n_histograms=2, detector_indices=[0], additive=False
    )
    assert out["before"] == {0: 0.0}
    assert out["after"] == {0: 0.012}
    assert grouping["dead_time_us"] == [0.012, 0.0]
    assert grouping["deadtime_correction"] is True
    assert grouping["deadtime_method"] == "value"

    out2 = promote_deadtime_to_grouping(
        grouping, 0.003, n_histograms=2, detector_indices=[0], additive=True
    )
    assert out2["after"][0] == pytest.approx(0.015)


def test_promote_deadtime_defaults_to_all_detectors():
    grouping = {}
    out = promote_deadtime_to_grouping(grouping, 0.02, n_histograms=3)
    assert out["after"] == {0: 0.02, 1: 0.02, 2: 0.02}


def test_promote_deadtime_per_detector_list_and_method():
    """A per-detector sequence aligns by index; ``method`` records provenance."""
    grouping = {}
    out = promote_deadtime_to_grouping(
        grouping, [0.012, 0.018], n_histograms=2, method="maxent_fit"
    )
    assert out["before"] == {0: 0.0, 1: 0.0}
    assert out["after"] == {0: 0.012, 1: 0.018}
    assert grouping["dead_time_us"] == [0.012, 0.018]
    assert grouping["deadtime_method"] == "maxent_fit"
    assert grouping["deadtime_correction"] is True


def test_promote_deadtime_short_list_pads_with_zero():
    """A sequence shorter than the detector count leaves the tail untouched."""
    grouping = {}
    out = promote_deadtime_to_grouping(grouping, [0.02], n_histograms=3)
    assert out["after"] == {0: 0.02, 1: 0.0, 2: 0.0}


def test_promote_deadtime_carries_polynomial_terms():
    """A polynomial/power-law promotion records the model + higher-order terms."""
    grouping = {"dead_time_us": [0.0, 0.0]}
    promote_deadtime_to_grouping(
        grouping,
        0.012,
        n_histograms=2,
        detector_indices=[0],
        model="polynomial",
        extra_terms={"C2": 0.003, "C3": 0.001, "DT1": 0.0},  # DT1 zero is dropped
    )
    # DT0 still drives the per-detector deadtime the reduction applies.
    assert grouping["dead_time_us"][0] == pytest.approx(0.012)
    assert grouping["deadtime_model"] == "polynomial"
    assert grouping["deadtime_model_terms"] == {"C2": 0.003, "C3": 0.001}

    # Additive accumulates the higher-order terms too.
    promote_deadtime_to_grouping(
        grouping,
        0.001,
        n_histograms=2,
        detector_indices=[0],
        additive=True,
        model="polynomial",
        extra_terms={"C2": 0.002},
    )
    assert grouping["deadtime_model_terms"]["C2"] == pytest.approx(0.005)
    assert grouping["dead_time_us"][0] == pytest.approx(0.013)


def _double_pulse_dataset(*, dpsep_us=0.324, seed=3):
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_double_pulse_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6,
        dpsep_us=dpsep_us,
        alpha=1.0,
        background_per_bin=10.0,
        seed=seed,
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def test_double_pulse_fixed_dpsep_matches():
    """With dpsep fixed at the instrument value the two-pulse model fits cleanly."""
    ds = _double_pulse_dataset(dpsep_us=0.324)
    params = _continuous_single()
    params.add(Parameter("dpsep", 0.324, fixed=True))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.reduced_chi_squared < 1.2
    assert result.parameters["A"].value == pytest.approx(20.0, abs=0.5)


def test_double_pulse_wrong_dpsep_is_worse():
    """A mis-set dpsep degrades the fit — the pulse weighting genuinely matters."""
    ds = _double_pulse_dataset(dpsep_us=0.324)
    good = _continuous_single()
    good.add(Parameter("dpsep", 0.324, fixed=True))
    bad = _continuous_single()
    bad.add(Parameter("dpsep", 0.20, fixed=True))
    r_good = fit_single_histogram(ds, 1, _tf, good, cost="gaussian")
    r_bad = fit_single_histogram(ds, 1, _tf, bad, cost="gaussian")
    assert r_good.reduced_chi_squared < r_bad.reduced_chi_squared


def test_double_pulse_free_dpsep_refines_from_seed():
    """A free dpsep refines to the true separation from a near-truth seed."""
    ds = _double_pulse_dataset(dpsep_us=0.324)
    params = _continuous_single()
    params.add(Parameter("dpsep", 0.33, min=0.1, max=0.5))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["dpsep"].value == pytest.approx(0.324, abs=0.01)


@pytest.mark.parametrize("seed_start", [0.12, 0.30, 0.48])
def test_double_pulse_free_dpsep_recovers_from_arbitrary_start(seed_start):
    """The coarse->fine scan finds dpsep from any in-range start (gate is non-smooth)."""
    ds = _double_pulse_dataset(dpsep_us=0.324)
    params = _continuous_single()
    # The seed is deliberately far from the truth; the scan spans [min, max] and
    # does not rely on starting near the minimum.
    params.add(Parameter("dpsep", seed_start, min=0.1, max=0.5))
    result = fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
    assert result.success
    assert result.parameters["dpsep"].value == pytest.approx(0.324, abs=0.01)
    assert result.reduced_chi_squared < 1.2


def test_fb_double_pulse_free_dpsep_recovers_from_arbitrary_start():
    """The F+B free-dpsep scan recovers both dpsep and α from a far start."""
    ds = _fb_double_pulse_dataset(dpsep_us=0.324, alpha=1.25, seed=4)
    params = _fb_double_pulse_params()
    params.add(Parameter("dpsep", 0.46, min=0.1, max=0.5))  # far from truth
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    assert result.success
    # dpsep is a shared (fixed-at-scan-value) parameter; α is per-side.
    assert result.shared_parameters["dpsep"].value == pytest.approx(0.324, abs=0.01)
    assert result.group_results[1].parameters["alpha"].value == pytest.approx(1.25, abs=0.03)


def _fb_double_pulse_dataset(*, dpsep_us=0.324, alpha=1.25, seed=4):
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_double_pulse_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6,
        dpsep_us=dpsep_us,
        alpha=alpha,
        background_per_bin=10.0,
        seed=seed,
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def _fb_double_pulse_params(alpha_seed=1.0):
    return ParameterSet(
        [
            Parameter("alpha", alpha_seed, min=0.1, max=5.0),
            Parameter("N0", 4.0e3, min=0.0),
            Parameter("background", 9.0, min=0.0),
            Parameter("background_b", 9.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
        ]
    )


@pytest.mark.parametrize("alpha_true", [1.0, 1.25])
def test_fb_double_pulse_recovers_alpha(alpha_true):
    """F+B double-pulse round-trip: α recovered with dpsep fixed at the instrument value."""
    ds = _fb_double_pulse_dataset(dpsep_us=0.324, alpha=alpha_true, seed=4)
    params = _fb_double_pulse_params()
    params.add(Parameter("dpsep", 0.324, fixed=True))
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    assert result.success
    fwd = result.group_results[1]
    assert fwd.parameters["alpha"].value == pytest.approx(alpha_true, abs=0.03)
    assert fwd.parameters["A"].value == pytest.approx(20.0, abs=0.6)
    assert fwd.reduced_chi_squared < 1.3


def test_fb_double_pulse_single_pulse_limit():
    """As dpsep → 0 the F+B double-pulse model reduces to the single-pulse fit."""
    ds = _fb_double_pulse_dataset(dpsep_us=0.0, alpha=1.25, seed=6)
    dp = _fb_double_pulse_params()
    dp.add(Parameter("dpsep", 1e-9, fixed=True))
    single = _fb_double_pulse_params()
    r_dp = fit_fb_alpha(ds, 1, 2, _tf, dp, cost="gaussian")
    r_single = fit_fb_alpha(ds, 1, 2, _tf, single, cost="gaussian")
    assert r_dp.success and r_single.success
    a_dp = r_dp.group_results[1].parameters["alpha"].value
    a_single = r_single.group_results[1].parameters["alpha"].value
    assert a_dp == pytest.approx(a_single, abs=1e-3)


def test_double_pulse_tolerates_model_nonfinite_below_zero():
    """The gated second pulse must not poison early bins for a t<0-undefined model."""

    def _nan_below_zero(t, A=20.0, f=1.0, phi=0.0):  # noqa: N803
        t = np.asarray(t, dtype=float)
        out = A * np.cos(2.0 * np.pi * f * t + phi)
        return np.where(t >= 0.0, out, np.nan)

    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_double_pulse_run(
        template,
        _tf,
        {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6,
        dpsep_us=0.324,
        alpha=1.0,
        background_per_bin=10.0,
        seed=3,
    )
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    params = _continuous_single()
    params.add(Parameter("dpsep", 0.324, fixed=True))
    result = fit_single_histogram(ds, 1, _nan_below_zero, params, cost="gaussian")
    assert result.success
    assert np.all(np.isfinite(result.residuals))


# --- review fixes: F/B chi2, distinct groups, alpha sign, amplitude unit -----


def test_fb_per_side_reduced_chi2_not_doubled():
    """Each F/B side reports its OWN reduced chi2 (~1), not the joint cost over one side."""
    ds = _pulsed_tf_run(alpha=1.0, seed=11)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    fwd = result.group_results[1]
    bwd = result.group_results[2]
    assert fwd.reduced_chi_squared == pytest.approx(1.0, abs=0.2)
    assert bwd.reduced_chi_squared == pytest.approx(1.0, abs=0.2)
    # Per-side cost, so the two chi2 values differ (joint would make them identical).
    assert fwd.chi_squared != bwd.chi_squared


def test_fb_requires_distinct_groups():
    ds = _pulsed_tf_run()
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1),
            Parameter("N0", 1.0e5),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
        ]
    )
    with pytest.raises(ValueError, match="two distinct groups"):
        fit_fb_alpha(ds, 1, 1, _tf, params)


def test_fb_negative_alpha_seed_is_clamped_positive():
    """A negative/unbounded alpha is floored to a positive balance, not reported negative."""
    ds = _pulsed_tf_run(alpha=1.25, seed=11)
    params = ParameterSet(
        [
            Parameter("alpha", -1.0),  # negative seed, default min=-inf
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")
    alpha_fit = result.group_results[1].parameters["alpha"].value
    assert alpha_fit > 0.0
    assert alpha_fit == pytest.approx(1.25, abs=0.02)


@pytest.mark.parametrize("bad_name", ["t0", "dpsep", "DT0", "N0", "background", "alpha", "tau"])
def test_model_param_colliding_with_reserved_name_raises(bad_name):
    """A model parameter named like a count-fit nuisance/structural slot fails loudly."""
    ds = _continuous_run()

    # A model whose declared parameter literally shadows a reserved count-fit name.
    def colliding_model(t, A, **_kwargs):  # noqa: N803 (A is the conventional amplitude symbol)
        return A * np.cos(2.0 * np.pi * np.asarray(t, dtype=float))

    pk = inspect.Parameter
    colliding_model.__signature__ = inspect.Signature(
        parameters=[
            pk("t", pk.POSITIONAL_OR_KEYWORD),
            pk("A", pk.POSITIONAL_OR_KEYWORD),
            pk(bad_name, pk.KEYWORD_ONLY, default=0.0),
        ]
    )
    params = ParameterSet([Parameter("N0", 4.0e3, min=0.0), Parameter("background", 9.0)])
    with pytest.raises(ValueError, match="collide with reserved count-fit names"):
        fit_single_histogram(ds, 1, colliding_model, params)


def test_linked_follower_reports_fitted_main_value():
    """A tied (link-group) parameter is reported at the fitted main value, not its seed."""

    def tied_model(t, A, f, f_tied, phi):  # noqa: N803  (f_tied is declared but mirrors f)
        return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)

    ds = _continuous_run()
    params = ParameterSet(
        [
            Parameter("N0", 4.0e3, min=0.0),
            Parameter("background", 9.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0, link_group=1),
            Parameter("f_tied", 5.0, min=0.0, link_group=1),  # seed deliberately wrong
            Parameter("phi", 0.0),
        ]
    )
    result = fit_single_histogram(ds, 1, tied_model, params, cost="gaussian")
    assert result.success
    f_main = result.parameters["f"].value
    # The follower must mirror the fitted main (~1.0), not its 5.0 seed.
    assert result.parameters["f_tied"].value == pytest.approx(f_main)
    assert f_main == pytest.approx(1.0, abs=0.02)


def test_power_deadtime_stays_finite_for_degenerate_exponent():
    """0**negative in the power form must not crash or poison the cost (-> finite)."""
    counts = np.array([1.0e4, 5.0e3, 2.0e3])
    time = np.array([0.0, 1.0, 5.0])
    # base = evfr*DT0 = 0 with a negative C2 would be 0**negative (ZeroDivisionError
    # on a Python float / inf in numpy); the guard maps it to a full, finite loss.
    # C2 != 0 keeps the loss factor active (not short-circuited to a no-op).
    terms = (0.0, 0.0, -1.5, 2.0, 0.3)  # (DT0, DT1, C2, C3, C4)
    out = _apply_deadtime(counts, terms, frame_norm=1e5, time=time, evfr=0.4, model="power")
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_non_colliding_model_is_accepted():
    """A model whose declared parameters avoid the reserved set fits normally."""

    def clean_model(t, A, freq, phase):  # noqa: N803
        return A * np.cos(2.0 * np.pi * freq * np.asarray(t, dtype=float) + phase)

    ds = _continuous_run()
    params = ParameterSet(
        [
            Parameter("N0", 4.0e3, min=0.0),
            Parameter("background", 9.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("freq", 1.0, min=0.0),
            Parameter("phase", 0.0),
        ]
    )
    result = fit_single_histogram(ds, 1, clean_model, params, cost="gaussian")
    assert result.success


def test_amplitude_identified_by_percent_unit_not_rate():
    """The seeding contract: the asymmetry amplitude carries unit '%'; rates do not."""
    from asymmetry.core.fitting.parameters import get_param_info

    assert get_param_info("A0").unit == "%"
    assert get_param_info("A_1").unit == "%"
    assert get_param_info("a_L").unit != "%"  # a Lorentzian rate, not an amplitude
    assert get_param_info("Lambda").unit != "%"


# --- plot overlay -----------------------------------------------------------


def _single_overlay_seed():
    return ParameterSet(
        [
            Parameter("N0", 4.0e3, min=0.0),
            Parameter("background", 9.0, min=0.0),
            Parameter("A", 19.0, min=0.0, max=50.0),
            Parameter("f", 1.0, min=0.0),
            Parameter("phi", 0.0),
        ]
    )


def test_single_overlay_is_corrected_model_at_fit_points():
    """The overlay curve = raw model * exp(t/tau), recovered exactly from residuals."""
    ds = _continuous_run()
    result = fit_single_histogram(ds, 1, _tf, _single_overlay_seed(), side="forward")
    assert result.success

    overlay = single_histogram_overlay(ds, 1, result)
    assert set(overlay) == {1}
    time, corrected = overlay[1]

    # Rebuild the same raw trace the fit used; model_raw = counts - residuals; the
    # overlay must be that on the lifetime-corrected (displayed) scale.
    group = build_count_group(ds, 1, lifetime_corrected=False)
    np.testing.assert_allclose(time, group.time, rtol=1e-12)
    model_raw = np.asarray(group.counts, float) - np.asarray(result.residuals, float)
    expected = model_raw * np.exp(np.asarray(group.time, float) / MUON_LIFETIME_US)
    np.testing.assert_allclose(corrected, expected, rtol=1e-12)
    assert np.all(np.isfinite(corrected))


def test_single_overlay_respects_fit_window():
    """An overlay built with the fit's window matches the fit's point count."""
    ds = _continuous_run()
    t_min, t_max = 0.5, 6.0
    result = fit_single_histogram(
        ds, 1, _tf, _single_overlay_seed(), side="forward", t_min=t_min, t_max=t_max
    )
    overlay = single_histogram_overlay(ds, 1, result, t_min=t_min, t_max=t_max)
    time, corrected = overlay[1]
    assert time.size == np.asarray(result.residuals).size
    assert float(np.min(time)) >= t_min - 1e-9
    assert float(np.max(time)) <= t_max + 1e-9


def test_fb_overlay_keys_both_banks():
    """The F+B overlay returns one corrected curve per fitted bank, keyed by group."""
    ds = _pulsed_tf_run(alpha=1.25, seed=11)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A", 18.0, min=0.0, max=50.0),
            Parameter("f", 1.5, min=0.0),
            Parameter("phi", 0.2),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, _tf, params)
    assert result.success

    overlay = fb_overlay_curves(ds, 1, 2, result)
    assert set(overlay) == {1, 2}
    for gid in (1, 2):
        time, corrected = overlay[gid]
        side = result.group_results[gid]
        # Rebuilt in one shared context, matching fit_fb_alpha's time axis exactly.
        assert time.size == np.asarray(side.residuals).size
        assert np.all(np.isfinite(corrected))
        # Corrected counts are positive (raw counts modulated by the decay envelope).
        assert np.all(corrected > 0.0)
