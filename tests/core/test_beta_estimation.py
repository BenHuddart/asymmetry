"""Backward-amplitude balance (β) estimation from a weak-TF calibration run.

Covers both measurement protocols in
:mod:`asymmetry.core.fitting.beta_calibration`:

- ``"count_fit"`` (Protocol A) — the simultaneous forward/backward count fit with
  a shared free ``beta`` on the backward polarization amplitude;
- ``"single_histogram"`` (Protocol B) — paired independent single-histogram fits
  with β̂ = Â₀,b/Â₀,f and α̂ = N̂₀,f/N̂₀,b by ratio.

Synthetic data injects a known β by scaling the backward-group polarization
amplitude (the forward group carries ``+A·P(t)``, the backward ``−β·A·P(t)``),
which is the exact generating model the estimator inverts. Convention check: for
this generating model a single fit recovers ``A0_f = A``, ``A0_b = β·A``,
``N0_f = N0·√α`` and ``N0_b = N0/√α``; hence β = A0_b/A0_f (backward/forward) but
α = N0_f/N0_b (forward/backward) — the two ratios run in opposite directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import beta_calibration
from asymmetry.core.fitting.beta_calibration import (
    BETA_ESTIMATION_METHODS,
    BetaEstimate,
    _n0_seed,
    estimate_beta_detailed,
)
from asymmetry.core.fitting.count_domain import fit_fb_alpha, fit_single_histogram
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_fb_count_model,
)
from asymmetry.core.fitting.models import oscillatory, stretched_exponential
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.simulate import (
    _fb_group_signals_and_weights,
    build_builtin_template,
    simulate_run_from_group_signals,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def _field_gauss_for_frequency(freq_mhz: float) -> float:
    """Applied field (Gauss) whose muon Larmor frequency is ``freq_mhz``."""
    return float(freq_mhz / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA))


def _fb_beta_run(
    *,
    alpha=1.15,
    beta=0.85,
    A=20.0,  # noqa: N803 (A is the conventional asymmetry-amplitude symbol)
    f=0.5,
    phi=0.3,
    lam=0.1,
    seed=1,
    total_events=40e6,
    field_gauss=None,
    background_per_bin=0.0,
) -> MuonDataset:
    """Synthetic forward/backward run with a known α and a known backward-β.

    The forward group sees ``+A·cos(2πft+φ)·e^(−λt)`` (percent), the backward
    group ``−β·`` that; the α split fixes the forward/backward event totals.
    """
    template = build_builtin_template("ideal_pulsed_fb")
    grouping = template.grouping
    fwd_gid = int(grouping["forward_group"])
    bwd_gid = int(grouping["backward_group"])

    def polarization_percent(t):
        t = np.asarray(t, dtype=float)
        return A * np.cos(2.0 * np.pi * f * t + phi) * np.exp(-lam * np.abs(t))

    def forward_signal(t):
        return polarization_percent(t) / 100.0

    def backward_signal(t):
        return -beta * polarization_percent(t) / 100.0

    group_signals, group_weights = _fb_group_signals_and_weights(
        grouping,
        len(template.histograms),
        forward_signal=forward_signal,
        backward_signal=backward_signal,
        forward_gid=fwd_gid,
        backward_gid=bwd_gid,
        alpha=alpha,
    )
    run = simulate_run_from_group_signals(
        template,
        group_signals,
        total_events=total_events,
        seed=seed,
        group_weights=group_weights,
        background_per_bin=background_per_bin,
    )
    if field_gauss is not None:
        run.metadata["field"] = float(field_gauss)
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def _fb_flat_run(*, A=20.0, alpha=1.1, beta=0.85, seed=4, total_events=40e6):  # noqa: N803
    """Non-precessing, non-relaxing run: a *flat* asymmetry, no field metadata.

    Without any time structure in the asymmetry the amplitude is entangled with
    the ``N0·√α`` normalization (a constant ``1 + A`` factor cannot be split from
    ``N0``), so β is genuinely degenerate — the estimator must report failure
    rather than a number. (A resolvable relaxation, by contrast, does constrain
    β and is correctly fittable, so a pure exponential is not a degenerate case.)
    """
    template = build_builtin_template("ideal_pulsed_fb")
    grouping = template.grouping
    fwd_gid = int(grouping["forward_group"])
    bwd_gid = int(grouping["backward_group"])

    def polarization_percent(t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, A)

    def forward_signal(t):
        return polarization_percent(t) / 100.0

    def backward_signal(t):
        return -beta * polarization_percent(t) / 100.0

    group_signals, group_weights = _fb_group_signals_and_weights(
        grouping,
        len(template.histograms),
        forward_signal=forward_signal,
        backward_signal=backward_signal,
        forward_gid=fwd_gid,
        backward_gid=bwd_gid,
        alpha=alpha,
    )
    run = simulate_run_from_group_signals(
        template,
        group_signals,
        total_events=total_events,
        seed=seed,
        group_weights=group_weights,
    )
    # No field metadata: a genuine ZF/relaxation run carries no transverse field.
    run.metadata.pop("field", None)
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


# --- truth recovery ---------------------------------------------------------


@pytest.mark.parametrize("method", BETA_ESTIMATION_METHODS)
def test_recovers_known_beta(method):
    """Both protocols recover β=0.85 (and α=1.15) within a few quoted σ."""
    alpha_true, beta_true, f = 1.15, 0.85, 0.5
    ds = _fb_beta_run(alpha=alpha_true, beta=beta_true, f=f, seed=11)
    est = estimate_beta_detailed(
        ds, 1, 2, method=method, field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    assert est.ok, est.message
    assert est.method == method
    assert est.beta_error is not None and est.beta_error > 0.0
    assert abs(est.beta - beta_true) < 3.0 * est.beta_error
    assert abs(est.beta - beta_true) < 0.05
    # α reported alongside as a consistency readout (forward/backward convention).
    assert abs(est.alpha - alpha_true) < 0.05
    assert est.n_bins_used > 0
    assert est.reduced_chi2 is not None and np.isfinite(est.reduced_chi2)


@pytest.mark.parametrize("method", BETA_ESTIMATION_METHODS)
def test_beta_one_is_recovered(method):
    """β=1 data (balanced pair) gives β̂ consistent with 1."""
    f = 0.5
    ds = _fb_beta_run(alpha=1.0, beta=1.0, f=f, seed=13)
    est = estimate_beta_detailed(
        ds, 1, 2, method=method, field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    assert est.ok, est.message
    assert abs(est.beta - 1.0) < 3.0 * est.beta_error
    assert abs(est.beta - 1.0) < 0.05


def test_protocols_agree_within_combined_error():
    """Protocol A and Protocol B agree on β within their combined errors."""
    f = 0.5
    ds = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=17)
    field_tesla = f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    a = estimate_beta_detailed(ds, 1, 2, method="count_fit", field_tesla=field_tesla)
    b = estimate_beta_detailed(ds, 1, 2, method="single_histogram", field_tesla=field_tesla)
    assert a.ok and b.ok
    combined = np.hypot(a.beta_error, b.beta_error)
    assert abs(a.beta - b.beta) < 3.0 * combined
    # Both report the same α convention (forward/backward), so α agrees too.
    assert abs(a.alpha - b.alpha) < 0.05


# --- correlation ------------------------------------------------------------


def test_count_fit_reports_finite_alpha_beta_correlation():
    """Protocol A exposes a finite α–β correlation in [-1, 1]."""
    f = 0.5
    ds = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=19)
    est = estimate_beta_detailed(
        ds, 1, 2, method="count_fit", field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    assert est.ok, est.message
    rho = est.alpha_beta_correlation
    assert rho is not None
    assert np.isfinite(rho) and -1.0 <= rho <= 1.0


def test_single_histogram_has_no_correlation():
    """Protocol B (independent fits) reports no α–β correlation."""
    f = 0.5
    ds = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=23)
    est = estimate_beta_detailed(
        ds, 1, 2, method="single_histogram", field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    assert est.ok, est.message
    assert est.alpha_beta_correlation is None


# --- degeneracy / failure modes ---------------------------------------------


@pytest.mark.parametrize("method", BETA_ESTIMATION_METHODS)
def test_non_precessing_reports_failure(method):
    """Non-precessing (flat asymmetry) input is degenerate → ok=False."""
    ds = _fb_flat_run(seed=29)
    est = estimate_beta_detailed(ds, 1, 2, method=method)
    assert not est.ok
    assert "degenerate" in est.message or "precession" in est.message or "converge" in est.message


def test_count_fit_matches_a_hand_seeded_n0():
    """Protocol A now leaves ``N0`` to the count-fit driver's data seed, which is
    the same first-good-bin rule it used to apply itself — so nothing moves."""
    f = 0.5
    ds = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=11)
    est = estimate_beta_detailed(
        ds, 1, 2, method="count_fit", field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    assert est.ok, est.message
    hand_seeded = fit_fb_alpha(
        ds,
        1,
        2,
        oscillatory,
        ParameterSet(
            [
                Parameter("alpha", 1.0, min=1.0e-6, max=100.0),
                Parameter("N0", _n0_seed(ds, 1), min=0.0),
                Parameter("background", 0.0, min=0.0),
                Parameter("background_b", 0.0, min=0.0),
                Parameter("A0", 20.0, min=0.0, max=100.0),
                Parameter("frequency", f, min=0.0),
                Parameter("phase", 0.0),
                Parameter("Lambda", 0.1, min=0.0),
                Parameter("baseline", 0.0, fixed=True),
            ]
        ),
        estimate_beta=True,
    )
    assert hand_seeded.success
    assert est.beta == pytest.approx(float(hand_seeded.shared_parameters["beta"].value), rel=1e-9)
    assert est.alpha == pytest.approx(float(hand_seeded.shared_parameters["alpha"].value), rel=1e-9)


def test_count_fit_passes_the_underlying_failure_message_through(monkeypatch):
    """A fit rejected for an implausible converged scale must not be flattened
    into "did not converge" — the caller needs to know the scale is the problem."""
    ds = _fb_beta_run(f=0.5, field_gauss=_field_gauss_for_frequency(0.5))
    rejected = GroupedTimeDomainFitResult(
        success=False,
        group_results={},
        shared_parameters=ParameterSet(),
        message="Forward/backward count fit converged at a spurious minimum",
    )
    monkeypatch.setattr(beta_calibration, "fit_fb_alpha", lambda *a, **k: rejected)
    est = estimate_beta_detailed(ds, 1, 2, method="count_fit")
    assert not est.ok
    assert est.message == rejected.message


def test_same_group_reports_failure():
    ds = _fb_beta_run(f=0.5, field_gauss=_field_gauss_for_frequency(0.5))
    est = estimate_beta_detailed(ds, 1, 1)
    assert not est.ok
    assert "distinct" in est.message


def test_unknown_method_raises():
    ds = _fb_beta_run(f=0.5, field_gauss=_field_gauss_for_frequency(0.5))
    with pytest.raises(ValueError, match="Unknown β-estimation method"):
        estimate_beta_detailed(ds, 1, 2, method="wishful")


# --- seeding: explicit field vs dataset metadata ----------------------------


def test_seeding_from_explicit_field_and_from_metadata_agree():
    """The frequency seed can come from field_tesla or from dataset metadata."""
    f = 0.5
    field_gauss = _field_gauss_for_frequency(f)
    # (a) explicit field_tesla, no metadata field.
    ds_explicit = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=31)
    est_explicit = estimate_beta_detailed(
        ds_explicit, 1, 2, method="count_fit", field_tesla=f / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    # (b) field read from dataset metadata (no field_tesla argument).
    ds_meta = _fb_beta_run(alpha=1.15, beta=0.85, f=f, seed=31, field_gauss=field_gauss)
    est_meta = estimate_beta_detailed(ds_meta, 1, 2, method="count_fit")
    assert est_explicit.ok and est_meta.ok
    assert abs(est_explicit.beta - 0.85) < 0.05
    assert abs(est_meta.beta - 0.85) < 0.05
    # Same seed data + same effective frequency seed → identical result.
    assert est_explicit.beta == pytest.approx(est_meta.beta, rel=1e-9)


# --- regression: default (estimate_beta=False) path is unchanged -------------


def test_estimate_beta_false_has_no_beta_in_shared_params():
    """The default fgFB path never grows a `beta` shared parameter."""
    ds = _fb_beta_run(alpha=1.15, beta=0.85, f=0.5, seed=37)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A0", 18.0, min=0.0, max=50.0),
            Parameter("frequency", 0.5, min=0.0),
            Parameter("phase", 0.2),
            Parameter("Lambda", 0.1, min=0.0),
            Parameter("baseline", 0.0, fixed=True),
        ]
    )
    result = fit_fb_alpha(ds, 1, 2, oscillatory, params)
    assert result.success
    assert "beta" not in result.shared_parameters.names
    assert "beta" not in result.group_results[1].parameters.names


def test_with_beta_false_model_matches_manual_expectation():
    """build_fb_count_model(with_beta=False) is byte-identical to the manual formula."""
    fb = build_fb_count_model(_frac(oscillatory))
    t = np.linspace(0.05, 8.0, 40)
    alpha, n0, bg_f, bg_b = 1.3, 5.0e4, 7.0, 9.0
    phys = dict(A0=20.0, frequency=1.5, phase=0.3, Lambda=0.1, baseline=0.0)
    a = 0.01 * oscillatory(t, **phys)
    env = np.exp(t / MUON_LIFETIME_US)
    expect_f = n0 * np.sqrt(alpha) * (1.0 + a) + bg_f * env
    expect_b = n0 / np.sqrt(alpha) * (1.0 - a) + bg_b * env
    got_f = fb(t, alpha=alpha, N0=n0, background=bg_f, sign=+1.0, **phys)
    got_b = fb(t, alpha=alpha, N0=n0, background=bg_b, sign=-1.0, **phys)
    np.testing.assert_array_equal(got_f, expect_f)
    np.testing.assert_array_equal(got_b, expect_b)


def test_with_beta_true_scales_backward_amplitude_only():
    """with_beta=True multiplies only the backward polarization by beta."""
    fb = build_fb_count_model(_frac(oscillatory), with_beta=True)
    t = np.linspace(0.05, 8.0, 40)
    alpha, n0, bg_f, bg_b, beta = 1.3, 5.0e4, 7.0, 9.0, 0.8
    phys = dict(A0=20.0, frequency=1.5, phase=0.3, Lambda=0.1, baseline=0.0)
    a = 0.01 * oscillatory(t, **phys)
    env = np.exp(t / MUON_LIFETIME_US)
    expect_f = n0 * np.sqrt(alpha) * (1.0 + a) + bg_f * env
    expect_b = n0 / np.sqrt(alpha) * (1.0 - beta * a) + bg_b * env
    got_f = fb(t, alpha=alpha, N0=n0, background=bg_f, sign=+1.0, beta=beta, **phys)
    got_b = fb(t, alpha=alpha, N0=n0, background=bg_b, sign=-1.0, beta=beta, **phys)
    np.testing.assert_allclose(got_f, expect_f, rtol=1e-12)
    np.testing.assert_allclose(got_b, expect_b, rtol=1e-12)
    # Forward side is independent of beta.
    got_f2 = fb(t, alpha=alpha, N0=n0, background=bg_f, sign=+1.0, beta=0.5, **phys)
    np.testing.assert_array_equal(got_f, got_f2)


# --- collision guard: `beta` reserved only on the β path ---------------------


def test_beta_named_model_rejected_on_beta_path():
    """A model with a `beta` param is rejected loudly on estimate_beta=True."""
    ds = _fb_beta_run(f=0.5, field_gauss=_field_gauss_for_frequency(0.5))
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A0", 18.0, min=0.0, max=50.0),
            Parameter("Lambda", 0.3, min=0.0),
            Parameter("beta", 1.0, min=0.2, max=3.0),
        ]
    )
    with pytest.raises(ValueError, match="beta"):
        fit_fb_alpha(ds, 1, 2, stretched_exponential, params, estimate_beta=True)


def test_beta_named_model_accepted_on_default_fb_path():
    """The same stretched-exponential model still fits on the non-β fgFB path."""
    ds = _fb_beta_run(alpha=1.1, beta=1.0, f=0.0, lam=0.4, seed=41)
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=0.1, max=5.0),
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("background_b", 0.0),
            Parameter("A0", 18.0, min=0.0, max=50.0),
            Parameter("Lambda", 0.4, min=0.0),
            Parameter("beta", 1.0, min=0.2, max=3.0),
        ]
    )
    # estimate_beta defaults to False: the model's own `beta` (stretch exponent)
    # is a legitimate physics parameter here.
    result = fit_fb_alpha(ds, 1, 2, stretched_exponential, params)
    assert result.success


def test_beta_named_model_accepted_by_single_histogram():
    """fit_single_histogram accepts a `beta`-named physics param (regression)."""
    ds = _fb_beta_run(alpha=1.0, beta=1.0, f=0.0, lam=0.4, seed=43)
    params = ParameterSet(
        [
            Parameter("N0", 1.5e5, min=0.0),
            Parameter("background", 0.0),
            Parameter("A0", 18.0, min=0.0, max=50.0),
            Parameter("Lambda", 0.4, min=0.0),
            Parameter("beta", 1.0, min=0.2, max=3.0),
        ]
    )
    result = fit_single_histogram(ds, 1, stretched_exponential, params, side="forward")
    assert result.success


# --- dataclass shape --------------------------------------------------------


def test_beta_estimate_is_frozen():
    est = BetaEstimate(
        beta=0.9,
        beta_error=0.01,
        alpha=1.1,
        alpha_error=0.02,
        alpha_beta_correlation=0.3,
        method="count_fit",
        n_bins_used=100,
        reduced_chi2=1.0,
        ok=True,
        message="ok",
    )
    with pytest.raises(AttributeError):
        est.beta = 1.0  # type: ignore[misc]


def _frac(model_fn):
    """Percent→fraction wrapper mirroring the count model's internal scaling."""

    def fraction(t, **kwargs):
        return 0.01 * np.asarray(model_fn(t, **kwargs), dtype=float)

    return fraction
