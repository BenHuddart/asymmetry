"""Tests for rotating-reference-frame demodulation (core/transform/rrf.py).

Numbered against docs/porting/rrf/verification-plan.md items 1-4.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.rrf import (
    RRFCurve,
    default_bandwidth_mhz,
    rrf_demodulate,
    rrf_demodulate_values,
)


def _signal(
    nu_mhz: float = 30.0,
    lam: float = 0.3,
    phase_rad: float = 0.0,
    a0: float = 25.0,
    # Well-sampled by default (fs = 250 MHz) so the 2ν₀ image at 60 MHz sits
    # below Nyquist and inside the filter stopband; the aliasing regime has
    # its own dedicated test.
    dt: float = 0.004,
    t_max: float = 8.0,
):
    t = np.arange(0.0, t_max, dt)
    envelope = a0 * np.exp(-lam * t)
    asym = envelope * np.cos(2.0 * np.pi * nu_mhz * t + phase_rad)
    err = np.full_like(t, 0.5)
    return t, asym, err, envelope


class TestEnvelopeExactness:
    """Plan item 1: on-resonance demodulation returns the generating envelope."""

    def test_real_part_recovers_envelope(self):
        t, asym, err, envelope = _signal()
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0)
        assert isinstance(curve, RRFCurve)
        assert curve.valid.sum() > t.size // 2
        np.testing.assert_allclose(
            curve.real[curve.valid], envelope[curve.valid], rtol=5e-4, atol=6e-3
        )
        # Quadrature carries no signal on resonance with matched phase.
        assert np.max(np.abs(curve.imag[curve.valid])) < 5e-4 * envelope.max()

    def test_phase_mismatch_moves_signal_to_quadrature(self):
        phase = np.deg2rad(90.0)
        t, asym, err, envelope = _signal(phase_rad=phase)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, phase_deg=0.0)
        # cos(ωt+90°) demodulated at φ=0 lands in −Im; magnitude is unaffected.
        np.testing.assert_allclose(
            curve.magnitude[curve.valid], envelope[curve.valid], rtol=5e-4, atol=6e-3
        )
        assert np.max(np.abs(curve.real[curve.valid])) < 5e-4 * envelope.max()

    def test_matched_phase_restores_real_part(self):
        t, asym, err, envelope = _signal(phase_rad=np.deg2rad(40.0))
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, phase_deg=40.0)
        np.testing.assert_allclose(
            curve.real[curve.valid], envelope[curve.valid], rtol=5e-4, atol=6e-3
        )


class TestBeat:
    """Plan item 2: detuning by δ leaves a δ beat under an intact envelope."""

    def test_magnitude_is_envelope_off_resonance(self):
        t, asym, err, envelope = _signal(nu_mhz=30.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=29.0)
        np.testing.assert_allclose(
            curve.magnitude[curve.valid], envelope[curve.valid], rtol=1e-3, atol=6e-3
        )

    def test_real_part_beats_at_delta(self):
        delta = 1.0
        t, asym, err, envelope = _signal(nu_mhz=30.0, lam=0.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0 - delta)
        valid = curve.valid
        # Count sign changes of Re inside the valid range: a cos(2πδt) beat
        # crosses zero 2δ times per µs.
        signs = np.sign(curve.real[valid])
        crossings = int(np.sum(signs[1:] * signs[:-1] < 0))
        span = t[valid][-1] - t[valid][0]
        expected = 2.0 * delta * span
        assert abs(crossings - expected) <= 2

    def test_beat_phase_convention(self):
        # ν > ν₀ must rotate anticlockwise: Im leads Re by a quarter beat,
        # matching Mantid RRFMuon's e^{−i(ωt+φ)} rotation sign.
        delta = 0.5
        t, asym, err, _ = _signal(nu_mhz=30.0, lam=0.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0 - delta)
        valid = np.flatnonzero(curve.valid)
        quarter = int(round(1.0 / (4.0 * delta) / (t[1] - t[0])))
        i0 = valid[len(valid) // 2]
        np.testing.assert_allclose(curve.imag[i0 + quarter], curve.real[i0], rtol=0.05, atol=0.2)


class TestImageSuppression:
    """Plan item 3: the FIR stopband kills the 2ω image; WiMDA's box leaks."""

    @staticmethod
    def _image_amplitude(t, values, image_mhz):
        # Project onto the image line frequency; subtract the mean first so
        # finite-window spectral leakage of the (large) baseband term does
        # not contaminate the small image measurement.
        centred = values - np.mean(values)
        probe = np.exp(-2j * np.pi * image_mhz * t)
        return 2.0 * np.abs(np.mean(centred * probe))

    def test_fir_suppresses_image_below_stopband(self):
        t, asym, err, envelope = _signal(nu_mhz=30.0, lam=0.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0)
        valid = curve.valid
        complex_curve = curve.real[valid] + 1j * curve.imag[valid]
        image = self._image_amplitude(t[valid], complex_curve, 60.0)
        # Blackman stopband is −74 dB ≈ 2e-4 in amplitude.
        assert image < 5e-4 * envelope.max()

    def test_wimda_default_box_nulls_image_on_field(self):
        # WiMDA's default box width (one image period) nulls the image when
        # ν₀ sits exactly on the line — its clever special case. The null
        # additionally requires the *discretized* box (trunc(box/dt) div 2)
        # to span an integer number of image periods: ν₀ = 25 MHz at
        # dt = 4 ns gives a 5-bin box exactly matching the 50 MHz image.
        t, asym, err, envelope = _signal(nu_mhz=25.0, lam=0.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=25.0, method="wimda")
        image = self._image_amplitude(t[curve.valid], curve.real[curve.valid], 50.0)
        assert image < 0.01 * envelope.max()

    def test_wimda_default_box_null_breaks_off_grid(self):
        # Same setup detuned so the discretized box no longer spans an
        # integer number of image periods: the null misses (ledger context
        # for the comparison doc — the default is only conditionally clever).
        t, asym, err, envelope = _signal(nu_mhz=30.0, lam=0.0)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, method="wimda")
        image = self._image_amplitude(t[curve.valid], curve.real[curve.valid], 60.0)
        assert image > 0.05 * envelope.max()

    def test_wimda_detuned_box_leaks_image(self):
        # Detune ν₀ by 2 MHz with the box still sized for the on-field image:
        # the sinc null misses and the image rides through; FIR does not.
        t, asym, err, envelope = _signal(nu_mhz=30.0, lam=0.0)
        nu0 = 28.0
        box = 1.0 / (2.0 * 30.0)
        wimda = rrf_demodulate(
            t, asym, err, frequency_mhz=nu0, method="wimda", wimda_box_width_us=box
        )
        fir = rrf_demodulate(t, asym, err, frequency_mhz=nu0)
        image_freq = 30.0 + nu0
        leak_wimda = self._image_amplitude(t[wimda.valid], wimda.real[wimda.valid], image_freq)
        fir_complex = fir.real[fir.valid] + 1j * fir.imag[fir.valid]
        leak_fir = self._image_amplitude(t[fir.valid], fir_complex, image_freq)
        assert leak_wimda > 10.0 * leak_fir
        assert leak_wimda > 0.01 * envelope.max()

    def test_wimda_mode_matches_pascal_loop(self):
        """Bin-for-bin fidelity against a direct transcription of Plot.pas."""
        rng = np.random.default_rng(7)
        n = 400
        dt = 0.016
        t = np.arange(n) * dt
        asym = rng.normal(0.0, 1.0, n)
        err = rng.uniform(0.1, 0.5, n)
        nu0, phase_deg, box = 12.0, 30.0, 0.08

        # Direct Pascal transcription (Plot.pas plotdata, RRF branch).
        rrffreq = 2.0 * np.pi * nu0
        rrfphase = np.deg2rad(phase_deg)
        factor = 2.0 * np.cos(t * rrffreq + rrfphase)
        dd = asym * factor
        ee = err * np.abs(factor)
        ii = int(box / dt) // 2
        fd = np.zeros(n)
        fe = np.zeros(n)
        for i in range(n):
            acc_v = acc_e = 0.0
            count = 0
            for i1 in range(i - ii, i + ii + 1):
                if 0 <= i1 < n:
                    acc_v += dd[i1]
                    acc_e += ee[i1]
                    count += 1
            fd[i] = acc_v / count if ii <= i <= n - ii - 1 else 0.0
            fe[i] = acc_e / count

        curve = rrf_demodulate(
            t,
            asym,
            err,
            frequency_mhz=nu0,
            phase_deg=phase_deg,
            method="wimda",
            wimda_box_width_us=box,
        )
        np.testing.assert_allclose(curve.real, fd, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(curve.real_error, fe, rtol=1e-12, atol=1e-12)


class TestErrorPropagation:
    """Plan item 4: pulls ~N(0,1); correlation follows the kernel."""

    def test_pull_distribution_per_quadrature(self):
        rng = np.random.default_rng(42)
        nu = 25.0
        t = np.arange(0.0, 8.0, 0.016)
        sigma = 0.8
        pulls_re = []
        pulls_im = []
        for _ in range(40):
            noise = rng.normal(0.0, sigma, t.size)
            curve = rrf_demodulate(
                t, noise, np.full_like(t, sigma), frequency_mhz=nu, bandwidth_mhz=5.0
            )
            # Truth is zero; thin by the filter support so the sampled pulls
            # are approximately independent.
            step = curve.filter_taps
            sel = np.flatnonzero(curve.valid)[::step]
            pulls_re.append(curve.real[sel] / curve.real_error[sel])
            pulls_im.append(curve.imag[sel] / curve.imag_error[sel])
        for pulls in (np.concatenate(pulls_re), np.concatenate(pulls_im)):
            assert abs(np.mean(pulls)) < 0.1
            assert abs(np.std(pulls) - 1.0) < 0.1

    def test_effective_independent_fraction_matches_box(self):
        t, asym, err, _ = _signal()
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, method="wimda")
        assert curve.effective_independent_fraction == pytest.approx(1.0 / curve.filter_taps)

    def test_neighbour_correlation_scale(self):
        rng = np.random.default_rng(3)
        t = np.arange(0.0, 8.0, 0.016)
        sigma = 1.0
        lag = None
        corrs = []
        for _ in range(30):
            noise = rng.normal(0.0, sigma, t.size)
            curve = rrf_demodulate(
                t, noise, np.full_like(t, sigma), frequency_mhz=25.0, bandwidth_mhz=5.0
            )
            r = curve.real[curve.valid]
            lag = curve.filter_taps
            r0 = r[:-lag]
            r1 = r[lag:]
            corrs.append(np.corrcoef(r0, r1)[0, 1])
        # One full support apart, correlation is gone.
        assert abs(np.mean(corrs)) < 0.1


class TestApiContract:
    def test_component_selector_and_magnitude_error(self):
        t, asym, err, _ = _signal()
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0)
        for name in ("real", "imag", "magnitude"):
            values, errors = curve.component(name)
            assert values.shape == t.shape
            assert np.all(errors >= 0.0)
        with pytest.raises(ValueError, match="Unknown RRF component"):
            curve.component("phase")

    def test_default_bandwidth(self):
        assert default_bandwidth_mhz(30.0) == pytest.approx(15.0)
        with pytest.raises(ValueError):
            default_bandwidth_mhz(0.0)

    def test_default_bandwidth_respects_aliased_image(self):
        # fs = 62.5 MHz, ν₀ = 30 MHz: the 60 MHz image folds to 2.5 MHz —
        # inside the naive ν₀/2 passband. The sampling-aware default must
        # duck below the folded image, and the envelope must still come back.
        t, asym, err, envelope = _signal(dt=0.016)
        assert default_bandwidth_mhz(30.0, sample_rate_mhz=62.5) == pytest.approx(1.75)
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0)
        assert curve.bandwidth_mhz == pytest.approx(1.75)
        np.testing.assert_allclose(
            curve.real[curve.valid], envelope[curve.valid], rtol=2e-2, atol=0.05
        )

    def test_invalid_inputs_raise(self):
        t, asym, err, _ = _signal()
        with pytest.raises(ValueError, match="frequency_mhz"):
            rrf_demodulate(t, asym, err, frequency_mhz=-1.0)
        with pytest.raises(ValueError, match="bandwidth_mhz"):
            rrf_demodulate(t, asym, err, frequency_mhz=30.0, bandwidth_mhz=0.0)
        with pytest.raises(ValueError, match="method"):
            rrf_demodulate(t, asym, err, frequency_mhz=30.0, method="iir")
        with pytest.raises(ValueError, match="matching shapes"):
            rrf_demodulate(t[:-1], asym, err, frequency_mhz=30.0)

    def test_nan_holes_do_not_poison_neighbours(self):
        t, asym, err, envelope = _signal()
        asym = asym.copy()
        asym[100:103] = np.nan
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0)
        ok = curve.valid & np.isfinite(curve.real)
        assert ok.sum() > t.size // 2
        # Away from the hole the curve is stopband-exact; within one filter
        # support of it the impaired image cancellation biases the output by
        # up to ~(missing bins / taps) × twice the local amplitude — bounded,
        # not catastrophic (see the module docstring).
        near_hole = np.zeros_like(ok)
        lo = max(0, 100 - curve.filter_taps)
        hi = min(t.size, 103 + curve.filter_taps)
        near_hole[lo:hi] = True
        far = ok & ~near_hole
        np.testing.assert_allclose(curve.real[far], envelope[far], rtol=5e-4, atol=6e-3)
        near = ok & near_hole
        bound = 2.0 * 2.0 * (3.0 / curve.filter_taps) * envelope[near]
        assert np.all(np.abs(curve.real[near] - envelope[near]) < bound + 6e-3)

    def test_values_only_helper_matches_full(self):
        t, asym, err, _ = _signal()
        full = rrf_demodulate(t, asym, np.zeros_like(t), frequency_mhz=30.0)
        light = rrf_demodulate_values(t, asym, frequency_mhz=30.0)
        np.testing.assert_allclose(light.real, full.real)
        np.testing.assert_allclose(light.imag, full.imag)
        assert np.all(light.real_error == 0.0)

    def test_frame_label(self):
        t, asym, err, _ = _signal()
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, phase_deg=15.0)
        label = curve.frame_label("magnitude")
        assert "ν₀ = 30 MHz" in label
        assert "15" in label
        assert "magnitude" in label

    def test_wide_bandwidth_degenerates_to_identity(self):
        t, asym, err, _ = _signal()
        nyquist = 0.5 / (t[1] - t[0])
        curve = rrf_demodulate(t, asym, err, frequency_mhz=30.0, bandwidth_mhz=nyquist * 1.1)
        assert curve.filter_taps == 1
        # Unfiltered demodulation is just 2·A·e^{−iθ}.
        theta = 2.0 * np.pi * 30.0 * t
        np.testing.assert_allclose(curve.real, 2.0 * asym * np.cos(theta), atol=1e-12)
