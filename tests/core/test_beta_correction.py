"""Tests for the β (intrinsic-asymmetry balance) correction.

Port of musrfit's asymmetry-fit (fit type 2) beta: applied with alpha as
``A = (F − αB)/(βF + αB)`` with the exact Poisson error
``σ = |α|(1+β)·√(FB(F+B))/(βF+αB)²``. Pins the identities from
``docs/porting/beta-correction/verification-plan.md``: β = 1 bit-identity with
the pre-port formula, numerical equivalence with musrfit's α-on-forward form,
ground-truth recovery on a synthetic two-detector model, the error closed
form, sentinel/lenient-read guards, α-estimator independence, integral parity,
and profile/payload persistence (emit-only-when-≠1).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    profile_from_payload,
    reconcile_instrument_for_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
    estimate_alpha,
)
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.integral import integrate_asymmetry
from asymmetry.core.transform.rebin import binned_fb_asymmetry

MU_TAU_US = 2.197


def _synthetic_counts(
    *,
    n0_f: float = 1000.0,
    n0_b: float = 800.0,
    a0_f: float = 0.25,
    a0_b: float = 0.20,
    n_bins: int = 512,
    bin_width_us: float = 0.016,
    omega: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Noise-free two-detector wTF model with known ground truth.

    Returns ``(t, forward, backward, alpha_true, beta_true)`` in Asymmetry's
    convention (α multiplies the backward group): ``alpha_true = N0_f/N0_b``,
    ``beta_true = A0_b/A0_f``.
    """
    t = np.arange(n_bins, dtype=float) * bin_width_us
    p = np.cos(omega * t)
    decay = np.exp(-t / MU_TAU_US)
    forward = n0_f * decay * (1.0 + a0_f * p)
    backward = n0_b * decay * (1.0 - a0_b * p)
    return t, forward, backward, n0_f / n0_b, a0_b / a0_f


class TestBetaFormula:
    def test_beta_one_is_bit_identical_to_pre_port_formula(self):
        rng = np.random.default_rng(7)
        f = rng.uniform(1.0, 1000.0, size=256)
        b = rng.uniform(1.0, 1000.0, size=256)
        for alpha in (0.5, 1.0, 1.7):
            base_a, base_e = compute_asymmetry(f, b, alpha=alpha)
            beta_a, beta_e = compute_asymmetry(f, b, alpha=alpha, beta=1.0)
            assert np.array_equal(base_a, beta_a)
            assert np.array_equal(base_e, beta_e)

    def test_beta_one_count_error_path_bit_identical(self):
        rng = np.random.default_rng(11)
        f = rng.uniform(1.0, 1000.0, size=128)
        b = rng.uniform(1.0, 1000.0, size=128)
        ef = np.sqrt(f)
        eb = np.sqrt(b)
        base_a, base_e = compute_asymmetry_with_count_errors(f, b, ef, eb, alpha=1.3)
        beta_a, beta_e = compute_asymmetry_with_count_errors(f, b, ef, eb, alpha=1.3, beta=1.0)
        assert np.array_equal(base_a, beta_a)
        assert np.array_equal(base_e, beta_e)

    def test_matches_musrfit_convention(self):
        """Ours == musrfit's (α_m·f − b)/(α_m·β·f + b) with α_m = 1/α.

        The musrfit form is transcribed from PRunAsymmetry.cpp:1412; β is
        numerically identical in both conventions (comparison.md).
        """
        rng = np.random.default_rng(13)
        f = rng.uniform(1.0, 1000.0, size=256)
        b = rng.uniform(1.0, 1000.0, size=256)
        for alpha, beta in ((0.8, 0.9), (1.4, 1.25), (1.0, 0.7)):
            ours, _ = compute_asymmetry(f, b, alpha=alpha, beta=beta)
            alpha_m = 1.0 / alpha
            musrfit = (alpha_m * f - b) / (alpha_m * beta * f + b)
            np.testing.assert_allclose(ours, musrfit, rtol=1e-12)

    def test_ground_truth_recovery(self):
        """Reducing with the true (α, β) recovers A0_f · P(t) exactly."""
        t, f, b, alpha_true, beta_true = _synthetic_counts()
        asym, _ = compute_asymmetry(f, b, alpha=alpha_true, beta=beta_true)
        expected = 0.25 * np.cos(8.0 * t)
        np.testing.assert_allclose(asym, expected, rtol=1e-12, atol=1e-14)

    def test_beta_error_closed_form(self):
        rng = np.random.default_rng(17)
        f = rng.uniform(10.0, 1000.0, size=128)
        b = rng.uniform(10.0, 1000.0, size=128)
        alpha, beta = 1.2, 0.85
        _, err = compute_asymmetry(f, b, alpha=alpha, beta=beta)
        den = beta * f + alpha * b
        expected = abs(alpha) * (1.0 + beta) * np.sqrt(f * b * (f + b)) / den**2
        np.testing.assert_allclose(err, expected, rtol=1e-12)

    def test_beta_count_error_closed_form(self):
        rng = np.random.default_rng(19)
        f = rng.uniform(10.0, 1000.0, size=64)
        b = rng.uniform(10.0, 1000.0, size=64)
        ef, eb = np.sqrt(f), np.sqrt(b)
        alpha, beta = 0.9, 1.3
        _, err = compute_asymmetry_with_count_errors(f, b, ef, eb, alpha=alpha, beta=beta)
        den = beta * f + alpha * b
        expected = abs(alpha) * (1.0 + beta) * np.sqrt((b * ef) ** 2 + (f * eb) ** 2) / den**2
        np.testing.assert_allclose(err, expected, rtol=1e-12)

    def test_sentinels_survive_beta(self):
        """Zero-denominator and one-sided bins keep the (0, 1) sentinels."""
        f = np.array([0.0, 100.0, 0.0])
        b = np.array([0.0, 0.0, 50.0])
        asym, err = compute_asymmetry(f, b, alpha=1.1, beta=0.8)
        assert asym[0] == 0.0 and err[0] == 1.0  # F = B = 0 → denominator 0
        assert err[1] == 1.0  # one-sided (B = 0)
        assert err[2] == 1.0  # one-sided (F = 0)

    def test_binned_reduction_threads_beta(self):
        t, f, b, alpha_true, beta_true = _synthetic_counts()
        _, with_beta, _ = binned_fb_asymmetry(
            f,
            b,
            grouping={"bunching_factor": 1},
            common_t0=0,
            bin_width_us=0.016,
            alpha=alpha_true,
            first_good_bin=0,
            last_good_bin=f.size - 1,
            beta=beta_true,
        )
        _, without, _ = binned_fb_asymmetry(
            f,
            b,
            grouping={"bunching_factor": 1},
            common_t0=0,
            bin_width_us=0.016,
            alpha=alpha_true,
            first_good_bin=0,
            last_good_bin=f.size - 1,
        )
        assert not np.allclose(with_beta, without)
        np.testing.assert_allclose(with_beta, 0.25 * np.cos(8.0 * t), rtol=1e-12, atol=1e-14)


class TestBetaGroupingRead:
    def _histograms(self) -> list[Histogram]:
        return [
            Histogram(
                counts=np.linspace(100.0, 50.0, 32) + 10.0 * i,
                bin_width=0.016,
                t0_bin=0,
                good_bin_start=0,
                good_bin_end=31,
            )
            for i in range(2)
        ]

    def _grouping(self, beta) -> dict:
        return {"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2, "beta": beta}

    def test_reads_valid_beta(self):
        fb = group_forward_backward(self._histograms(), self._grouping(0.9))
        assert fb.beta == pytest.approx(0.9)

    @pytest.mark.parametrize("degenerate", [float("nan"), float("inf"), 0.0, -2.0, "x", None])
    def test_degenerate_beta_falls_back_to_one(self, degenerate):
        fb = group_forward_backward(self._histograms(), self._grouping(degenerate))
        assert fb.beta == 1.0

    def test_missing_beta_defaults_to_one(self):
        grouping = self._grouping(0.9)
        del grouping["beta"]
        fb = group_forward_backward(self._histograms(), grouping)
        assert fb.beta == 1.0


class TestBetaAlphaIndependence:
    def test_alpha_estimate_is_beta_blind(self):
        """The α estimator consumes counts only — β cannot reach or bias it."""
        _, f, b, _, _ = _synthetic_counts()
        assert estimate_alpha(f, b) == pytest.approx(float(np.sum(f) / np.sum(b)))
        # β does not move the corrected asymmetry's zero: A = 0 ⇔ F = αB.
        alpha = 1.25
        f0 = np.array([125.0])
        b0 = np.array([100.0])
        for beta in (0.7, 1.0, 1.4):
            asym, _ = compute_asymmetry(f0, b0, alpha=alpha, beta=beta)
            assert asym[0] == pytest.approx(0.0, abs=1e-15)


class TestBetaIntegral:
    def test_integral_matches_summed_counts_formula(self):
        _, f, b, alpha, beta = _synthetic_counts()
        value, error = integrate_asymmetry(f, b, alpha=alpha, beta=beta)
        ref_a, ref_e = compute_asymmetry(np.array([np.sum(f)]), np.array([np.sum(b)]), alpha, beta)
        assert value == pytest.approx(float(ref_a[0]))
        assert error == pytest.approx(float(ref_e[0]))

    def test_invalid_beta_rejected(self):
        f = np.ones(8)
        b = np.ones(8)
        with pytest.raises(ValueError, match="beta"):
            integrate_asymmetry(f, b, alpha=1.0, beta=0.0)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _run(*, grouping: dict | None = None) -> Run:
    histograms = [
        Histogram(
            counts=np.arange(10 * i, 10 * i + 20, dtype=float) + 1.0,
            bin_width=0.016,
            t0_bin=5,
            good_bin_start=6,
            good_bin_end=19,
        )
        for i in range(4)
    ]
    base = {"instrument": "EMU"}
    if grouping:
        base.update(grouping)
    return Run(
        run_number=1,
        histograms=histograms,
        grouping=base,
        metadata={"instrument": "EMU"},
    )


def _profile(**kwargs) -> GroupingProfile:
    return GroupingProfile(
        name="Default (EMU)",
        fingerprint=ProfileFingerprint(instrument="EMU", histogram_count=4),
        groups={1: [1, 2], 2: [3, 4]},
        forward_group=1,
        backward_group=2,
        **kwargs,
    )


class TestBetaPersistence:
    def test_default_beta_not_emitted(self):
        """A β = 1 profile serializes byte-identically to a pre-β one."""
        assert "beta" not in _profile().to_dict()

    def test_beta_round_trips(self):
        profile = _profile(beta=0.9)
        data = profile.to_dict()
        assert data["beta"] == pytest.approx(0.9)
        assert GroupingProfile.from_dict(data).beta == pytest.approx(0.9)

    def test_from_dict_sanitizes_degenerate_beta(self):
        data = _profile(beta=0.9).to_dict()
        for bad in (float("nan"), -1.0, 0.0, "x"):
            data["beta"] = bad
            assert GroupingProfile.from_dict(data).beta == 1.0

    def test_profile_from_payload_lifts_beta(self):
        payload = {"groups": {1: [1], 2: [2]}, "beta": 1.2}
        fingerprint = ProfileFingerprint(instrument="EMU", histogram_count=4)
        assert profile_from_payload(payload, "p", fingerprint).beta == pytest.approx(1.2)
        payload_no_beta = {"groups": {1: [1], 2: [2]}}
        assert profile_from_payload(payload_no_beta, "p", fingerprint).beta == 1.0

    def test_resolution_writes_beta_only_when_active(self):
        run = _run()
        assert "beta" not in resolve_effective_grouping(_profile(), run)
        resolved = resolve_effective_grouping(_profile(beta=0.9), run)
        assert resolved["beta"] == pytest.approx(0.9)

    def test_resolution_end_to_end_changes_reduction(self):
        run = _run()
        plain = resolve_effective_grouping(_profile(), run)
        with_beta = resolve_effective_grouping(_profile(beta=0.8), run)
        fb_plain = group_forward_backward(run.histograms, plain)
        fb_beta = group_forward_backward(run.histograms, with_beta)
        a_plain, _ = compute_asymmetry(fb_plain.forward, fb_plain.backward, fb_plain.alpha)
        a_beta, _ = compute_asymmetry(
            fb_beta.forward, fb_beta.backward, fb_beta.alpha, fb_beta.beta
        )
        assert not np.allclose(a_plain, a_beta)

    def test_vector_profile_never_emits_beta(self):
        """Scalar-only: a projection-carrying profile resolves without β."""
        profile = _profile(
            beta=0.9,
            projections=[{"label": "P_z", "forward_group": 1, "backward_group": 2}],
        )
        assert "beta" not in resolve_effective_grouping(profile, _run())

    def test_instrument_heal_keeps_beta(self):
        """β is instrument-independent, so a stale-identity heal keeps it."""
        run = _run(grouping={"instrument": "FLAME"})
        payload = {"instrument": "GPS", "groups": {1: [1], 2: [2]}, "beta": 0.9}
        reconciled, note = reconcile_instrument_for_payload(run, payload)
        if note is not None:  # heal fired (detection disagreed with "GPS")
            assert reconciled.get("beta") == pytest.approx(0.9)
        else:  # detection inconclusive on the synthetic run — payload untouched
            assert reconciled is payload
