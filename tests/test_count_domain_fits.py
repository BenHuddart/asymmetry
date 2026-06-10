"""Count-domain fit modes: single-histogram and forward/backward (alpha-free).

Verification under the study's transcribe + synthetic + cross-check oracle
(``docs/porting/count-domain-fit-modes/verification-plan.md``). WiMDA is a
source-only oracle: the count models are checked against formulas transcribed
from ``AsymFitFunction.pas``, and parameter recovery is checked against
``core/simulate`` ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import (
    COUNT_COSTS,
    _percent_to_fraction,
    _raw_model,
    fit_fb_alpha,
    fit_single_histogram,
)
from asymmetry.core.fitting.grouped_time_domain import (
    build_count_group,
    build_fb_count_model,
    build_grouped_count_model,
    build_grouped_time_domain_groups,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.simulate import build_builtin_template, simulate_run
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
