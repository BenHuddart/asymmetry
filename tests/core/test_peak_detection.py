"""Tests for the spectral peak-detection core service."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.peak_detection import (
    USER_PEAK_SNR_SENTINEL,
    DetectedPeak,
    PeakAnalysis,
    analyze_dataset_peaks,
    deserialize_detected_peak,
    deserialize_peak_analysis,
    detect_peaks_in_spectrum,
    effective_analysis_window,
    merge_user_peaks,
    serialize_detected_peak,
    serialize_peak_analysis,
)


def _dataset(time: np.ndarray, asym: np.ndarray, *, error: float = 0.05) -> MuonDataset:
    return MuonDataset(
        time=np.asarray(time, dtype=float),
        asymmetry=np.asarray(asym, dtype=float),
        error=np.full_like(np.asarray(time, dtype=float), error),
        metadata={"run_number": 1},
    )


def _damped_cosines(
    t: np.ndarray,
    components: list[tuple[float, float]],
    lam: float,
    rng: np.random.Generator,
    noise: float,
) -> np.ndarray:
    env = np.exp(-lam * t)
    signal = np.zeros_like(t)
    for amp, freq in components:
        signal += amp * np.cos(2.0 * np.pi * freq * t)
    signal *= env
    signal += rng.normal(0.0, noise, size=t.size)
    return signal


# --------------------------------------------------------------------------- #
# 1. Two damped cosines
# --------------------------------------------------------------------------- #


def test_two_damped_cosines_both_found() -> None:
    rng = np.random.default_rng(1234)
    t = np.linspace(0.0, 16.0, 4096)
    f1, f2 = 1.3, 3.7
    a1, a2 = 3.0, 2.0
    y = _damped_cosines(t, [(a1, f1), (a2, f2)], lam=0.05, rng=rng, noise=0.05)
    dataset = _dataset(t, y)

    analysis = analyze_dataset_peaks(dataset, burg_check="never")
    freqs = sorted(p.frequency_mhz for p in analysis.peaks)

    assert len(analysis.peaks) >= 2
    bin_spacing = analysis.resolution_mhz  # 1/T
    # Both target lines present within half a resolution element.
    near1 = min(abs(f - f1) for f in freqs)
    near2 = min(abs(f - f2) for f in freqs)
    assert near1 < 0.5 * bin_spacing
    assert near2 < 0.5 * bin_spacing

    # The two strongest peaks are the two injected lines (SNR-descending; SNR
    # ordering need not follow amplitude ordering — the local noise floor is
    # higher near DC where relaxation leakage lives).
    top_two = analysis.peaks[:2]
    assert top_two[0].snr >= top_two[1].snr
    top_freqs = sorted(p.frequency_mhz for p in top_two)
    assert abs(top_freqs[0] - f1) < 0.5 * bin_spacing
    assert abs(top_freqs[1] - f2) < 0.5 * bin_spacing

    for peak in analysis.peaks:
        assert peak.width_mhz > 0.0
        assert np.isfinite(peak.width_mhz)
        assert peak.source == "fft"


# --------------------------------------------------------------------------- #
# 2. Off-bin single sinusoid, truncated — no sidelobe false peaks
# --------------------------------------------------------------------------- #


def test_offbin_single_sinusoid_no_sidelobes() -> None:
    rng = np.random.default_rng(77)
    t = np.linspace(0.0, 10.0, 2048)
    f0 = 2.35  # off-bin
    y = 2.0 * np.cos(2.0 * np.pi * f0 * t) + rng.normal(0.0, 0.02, size=t.size)
    dataset = _dataset(t, y)

    analysis = analyze_dataset_peaks(dataset, burg_check="never", min_snr=2.5)
    strong = [p for p in analysis.peaks if p.snr >= 2.5]

    assert len(strong) == 1
    assert abs(strong[0].frequency_mhz - f0) < 0.5 * analysis.resolution_mhz


# --------------------------------------------------------------------------- #
# 3. Noise only
# --------------------------------------------------------------------------- #


def test_noise_only_no_peaks() -> None:
    rng = np.random.default_rng(9)
    t = np.linspace(0.0, 12.0, 2048)
    y = rng.normal(0.0, 1.0, size=t.size)
    dataset = _dataset(t, y)

    analysis = analyze_dataset_peaks(dataset, burg_check="never", min_snr=2.5)
    assert analysis.peaks == ()


# --------------------------------------------------------------------------- #
# 4. Short window, close doublet — Burg path exercised
# --------------------------------------------------------------------------- #


def test_short_window_burg_path() -> None:
    rng = np.random.default_rng(2026)
    t = np.linspace(0.0, 8.0, 128)  # resolution 1/8 = 0.125 MHz
    # Δf = 0.15 MHz < 2/T = 0.25 MHz -> close doublet triggers auto burg.
    f1, f2 = 1.5, 1.65
    y = _damped_cosines(t, [(3.0, f1), (3.0, f2)], lam=0.05, rng=rng, noise=0.02)
    dataset = _dataset(t, y)

    analysis = analyze_dataset_peaks(dataset, burg_check="auto", min_snr=2.0)

    # n_points < 512 alone forces the burg cross-check.
    assert analysis.burg_order is not None
    assert isinstance(analysis.burg_order, int)
    assert len(analysis.peaks) >= 1
    for peak in analysis.peaks:
        assert isinstance(peak.burg_confirmed, bool)


# --------------------------------------------------------------------------- #
# 5. Detrend curve path
# --------------------------------------------------------------------------- #


def test_detrend_curve_recovers_weak_line() -> None:
    rng = np.random.default_rng(4242)
    t = np.linspace(0.0, 16.0, 4096)  # resolution 1/16 = 0.0625 MHz
    lam = 2.5  # decay spectral width comparable to the weak line
    decay = 20.0 * np.exp(-lam * t)
    weak_freq = 0.4
    oscillation = 0.6 * np.cos(2.0 * np.pi * weak_freq * t)
    y = decay + oscillation + rng.normal(0.0, 0.02, size=t.size)
    dataset = _dataset(t, y)

    tail_only = analyze_dataset_peaks(dataset, burg_check="never", min_snr=2.0)
    detrended = analyze_dataset_peaks(
        dataset,
        detrend_curve=decay,
        detrend_template_key="exp_decay",
        burg_check="never",
        min_snr=2.0,
    )

    assert detrended.detrended is True
    assert detrended.detrend_template_key == "exp_decay"
    assert all(p.source == "residual_fft" for p in detrended.peaks)
    assert tail_only.detrended is False

    def _nearest_snr(analysis: PeakAnalysis) -> float:
        best_snr = 0.0
        best_dist = detrended.resolution_mhz
        for p in analysis.peaks:
            dist = abs(p.frequency_mhz - weak_freq)
            if dist <= best_dist:
                best_dist = dist
                best_snr = p.snr
        return best_snr

    detrended_snr = _nearest_snr(detrended)
    tail_snr = _nearest_snr(tail_only)

    # The detrended residual FFT resolves the 0.4 MHz line.
    assert detrended_snr >= 2.0
    # ...and it is genuinely better than the tail-subtraction-only estimate.
    assert detrended_snr > tail_snr


# --------------------------------------------------------------------------- #
# 6. merge_user_peaks
# --------------------------------------------------------------------------- #


def _analysis_with(peaks: list[DetectedPeak], resolution: float = 0.1) -> PeakAnalysis:
    return PeakAnalysis(
        peaks=tuple(peaks),
        noise_floor=1.0,
        resolution_mhz=resolution,
        nyquist_mhz=50.0,
        detrended=False,
    )


def test_merge_user_peak_replaces_within_resolution() -> None:
    detected = DetectedPeak(
        frequency_mhz=2.03,
        amplitude=5.0,
        snr=12.0,
        width_mhz=0.08,
        prominence=4.0,
        source="fft",
        burg_confirmed=True,
    )
    analysis = _analysis_with([detected], resolution=0.1)

    merged = merge_user_peaks(analysis, [2.0])  # within 1*resolution

    assert len(merged.peaks) == 1
    user = merged.peaks[0]
    assert user.source == "user"
    assert user.snr == USER_PEAK_SNR_SENTINEL
    assert user.frequency_mhz == pytest.approx(2.0)
    # Detected amplitude/width preserved.
    assert user.amplitude == pytest.approx(5.0)
    assert user.width_mhz == pytest.approx(0.08)
    assert user.burg_confirmed is None


def test_merge_user_peak_added_when_far() -> None:
    detected = DetectedPeak(
        frequency_mhz=2.0,
        amplitude=5.0,
        snr=12.0,
        width_mhz=0.08,
        prominence=4.0,
        source="fft",
    )
    analysis = _analysis_with([detected], resolution=0.1)

    merged = merge_user_peaks(analysis, [5.0])  # far away -> new peak

    assert len(merged.peaks) == 2
    # User peak sorts first (sentinel SNR).
    assert merged.peaks[0].source == "user"
    assert merged.peaks[0].frequency_mhz == pytest.approx(5.0)
    assert merged.peaks[1].source == "fft"


def test_merge_user_peaks_never_dropped_and_first() -> None:
    # An analysis that already respected max_peaks=1.
    detected = DetectedPeak(
        frequency_mhz=2.0,
        amplitude=5.0,
        snr=12.0,
        width_mhz=0.08,
        prominence=4.0,
        source="fft",
    )
    analysis = _analysis_with([detected], resolution=0.1)

    merged = merge_user_peaks(analysis, [7.0, 9.0])

    # Neither user peak dropped despite the original cap.
    assert len(merged.peaks) == 3
    user_peaks = [p for p in merged.peaks if p.source == "user"]
    assert len(user_peaks) == 2
    # Both user peaks sort ahead of the detected one.
    assert merged.peaks[0].source == "user"
    assert merged.peaks[1].source == "user"
    assert merged.peaks[2].source == "fft"


# --------------------------------------------------------------------------- #
# 7. Serializer round-trips + default tolerance
# --------------------------------------------------------------------------- #


def test_detected_peak_round_trip() -> None:
    peak = DetectedPeak(
        frequency_mhz=1.234,
        amplitude=5.6,
        snr=8.9,
        width_mhz=0.12,
        prominence=3.4,
        source="residual_fft",
        burg_confirmed=True,
    )
    restored = deserialize_detected_peak(serialize_detected_peak(peak))
    assert restored == peak


def test_peak_analysis_round_trip() -> None:
    analysis = PeakAnalysis(
        peaks=(
            DetectedPeak(1.0, 2.0, 3.0, 0.1, 0.5, "fft", None),
            DetectedPeak(4.0, 1.0, 2.0, 0.2, 0.4, "user", False),
        ),
        noise_floor=0.7,
        resolution_mhz=0.0625,
        nyquist_mhz=64.0,
        detrended=True,
        detrend_template_key="exp_decay",
        burg_order=12,
        burg_hit_boundary=True,
    )
    restored = deserialize_peak_analysis(serialize_peak_analysis(analysis))
    assert restored == analysis


def test_deserialize_tolerates_missing_optional_keys() -> None:
    peak = deserialize_detected_peak({"frequency_mhz": 2.0, "amplitude": 1.0})
    assert peak is not None
    assert peak.frequency_mhz == pytest.approx(2.0)
    assert peak.snr == pytest.approx(0.0)
    assert peak.source == "fft"
    assert peak.burg_confirmed is None

    analysis = deserialize_peak_analysis({"resolution_mhz": 0.1})
    assert analysis is not None
    assert analysis.peaks == ()
    assert analysis.resolution_mhz == pytest.approx(0.1)
    assert analysis.burg_order is None
    assert analysis.detrend_template_key is None
    assert analysis.detrended is False


def test_deserialize_rejects_non_dict() -> None:
    assert deserialize_detected_peak(None) is None
    assert deserialize_detected_peak([1, 2, 3]) is None
    assert deserialize_peak_analysis(None) is None


# --------------------------------------------------------------------------- #
# detect_peaks_in_spectrum direct: degenerate spectra
# --------------------------------------------------------------------------- #


def test_detect_peaks_handles_tiny_spectrum() -> None:
    analysis = detect_peaks_in_spectrum(
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        resolution_mhz=1.0,
    )
    assert analysis.peaks == ()


# --------------------------------------------------------------------------- #
# 8. Exploding late-time errors — SNR-truncated detection window (failure F2)
# --------------------------------------------------------------------------- #


def _exploding_error_larmor(seed: int = 12345) -> MuonDataset:
    """A clean percent-scale 0.271 MHz Larmor line with realistic μSR statistics.

    Amplitude 20 %, constant 4 % background, damped at λ = 0.1 µs⁻¹; per-point
    σ(t) = 0.7·exp(t / (2·2.2)) capped at 100 % (dying-muon statistics), with
    Gaussian noise drawn per point from σ(t).  ~1/3 of the record is pure noise
    at the 100 % cap — enough to whiten a naive full-window FFT to zero peaks.
    """
    t = np.linspace(0.15, 32.6, 2000)
    signal = 20.0 * np.exp(-0.1 * t) * np.cos(2.0 * np.pi * 0.271 * t) + 4.0
    sigma = np.minimum(0.7 * np.exp(t / (2.0 * 2.2)), 100.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma)
    return MuonDataset(
        time=t,
        asymmetry=signal + noise,
        error=sigma,
        metadata={"run_number": 1, "field": 20.0},
    )


def test_exploding_errors_larmor_line_recovered() -> None:
    # The late-time σ blow-up (capped at 100 %) dominates the full-window FFT and
    # whitens the spectrum; the SNR-truncated window recovers the 0.271 MHz line.
    dataset = _exploding_error_larmor(seed=12345)
    analysis = analyze_dataset_peaks(dataset, burg_check="never")

    assert len(analysis.peaks) >= 1
    best = max(analysis.peaks, key=lambda p: p.snr)
    # Within 5 % of the true 20 G Larmor frequency.
    assert abs(best.frequency_mhz - 0.271) <= 0.05 * 0.271
    assert best.snr >= 2.5

    # The reported resolution reflects the *effective* (truncated) window, not
    # the full record — the noise tail was discarded before the transform.
    end = effective_analysis_window(dataset.time, dataset.error)
    assert end < dataset.n_points
    effective_duration = dataset.time[end - 1] - dataset.time[0]
    assert analysis.resolution_mhz == pytest.approx(1.0 / effective_duration, rel=1e-6)
    # Nyquist is set by the sampling interval and is unaffected by truncation.
    assert analysis.nyquist_mhz == pytest.approx(30.8, rel=0.02)


def test_snr_truncation_window_is_no_op_on_flat_errors() -> None:
    # Constant-σ records (the synthetic/test convention) must keep the full
    # window so the reported resolution — and the existing suite's tolerances —
    # are unchanged.
    t = np.linspace(0.0, 16.0, 4096)
    flat = np.full_like(t, 0.05)
    assert effective_analysis_window(t, flat) == t.size

    # ...and a monotonically growing σ truncates the noisy tail.
    growing = 0.5 * np.exp(t / 4.0)
    assert effective_analysis_window(t, growing) < t.size


# --------------------------------------------------------------------------- #
# 9. Nyquist top-edge guard mirrors the DC guard
# --------------------------------------------------------------------------- #


def test_nyquist_guard_rejects_near_nyquist_artifact() -> None:
    # A flat spectrum with one artifact line hard against Nyquist and one clean
    # mid-band line: the near-Nyquist artifact is rejected (mirroring the DC
    # guard at the bottom edge) while the mid-band line survives.
    df = 0.01
    freqs = np.arange(0.0, 10.0, df)
    nyquist = float(freqs[-1])
    mags = np.ones_like(freqs)

    mid_idx = int(np.argmin(np.abs(freqs - 5.0)))
    artifact_idx = int(np.argmin(np.abs(freqs - (nyquist - 0.02))))
    mags[mid_idx] = 50.0
    mags[artifact_idx] = 50.0

    analysis = detect_peaks_in_spectrum(freqs, mags, resolution_mhz=0.05, min_snr=2.0)

    detected = sorted(p.frequency_mhz for p in analysis.peaks)
    assert len(detected) == 1
    assert abs(detected[0] - 5.0) < 0.05
    # Nothing survives in the top-edge guard band.
    assert all(p.frequency_mhz < nyquist - 0.02 for p in analysis.peaks)


# --------------------------------------------------------------------------- #
# 10. Fingerprint de-whitening on the same SNR-truncated window
# --------------------------------------------------------------------------- #


def test_fingerprint_dewhitens_exploding_error_precession() -> None:
    # ``fingerprint_spectrum`` (core, Qt-free) shares the SNR-truncated window,
    # so clean TF precession with exploding late-time errors yields a strong,
    # correctly-located dominant spectral line rather than a whitened spectrum.
    from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum

    dataset = _exploding_error_larmor(seed=12345)
    fingerprint = fingerprint_spectrum(dataset)

    assert fingerprint.dominant_fft_snr >= 3.0
    assert fingerprint.dominant_fft_cycles_in_window >= 1.5
    # Dominant line within 10 % of the 0.271 MHz Larmor frequency (the FFT bin
    # granularity of the truncated window is coarser than the peak detector's
    # zero-padded refinement, hence the looser tolerance than the peak test).
    assert abs(fingerprint.dominant_fft_frequency_mhz - 0.271) <= 0.10 * 0.271
