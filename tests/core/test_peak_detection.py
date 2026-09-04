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


# --------------------------------------------------------------------------- #
# 11. Matched-apodisation damped-line pass
# --------------------------------------------------------------------------- #

#: A heavily damped two-line record: 4.7 % at 240 MHz (λ = 44 µs⁻¹) and 1.9 %
#: at 120 MHz (λ = 22 µs⁻¹) on a slowly relaxing background, at 0.1 ns binning
#: with µSR-like errors growing as exp(t/4.4 µs) and saturating at 25 %. Every
#: number invented; the same shape the scan module's own tests use, so the two
#: layers are measured on comparable data.
_SCAN_LINES = ((4.7, 240.0, 44.0, 0.3), (1.9, 120.0, 22.0, -0.7))


def _damped_scan_record(
    *,
    seed: int = 39,
    n_points: int = 60_000,
    bin_width_us: float = 1e-4,
    sigma0_percent: float = 3.3,
    lines: tuple[tuple[float, float, float, float], ...] = _SCAN_LINES,
    relaxation: bool = True,
) -> MuonDataset:
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=float) * bin_width_us
    sigma0 = sigma0_percent * np.sqrt(1e-4 / bin_width_us)
    error = np.minimum(sigma0 * np.exp(time / 4.4), 25.0)
    signal = (2.6 * np.exp(-0.19 * time) + 4.6) if relaxation else np.zeros_like(time)
    for amplitude, frequency, rate, phase in lines:
        signal = signal + amplitude * np.exp(-rate * time) * np.cos(
            2.0 * np.pi * frequency * time + phase
        )
    return MuonDataset(
        time=time,
        asymmetry=signal + rng.normal(0.0, error),
        error=error,
        metadata={"run_number": 1},
    )


def test_damped_scan_peaks_reach_the_merged_peak_set() -> None:
    """Both damped lines arrive as ``damped_scan`` peaks carrying their fit."""
    analysis = analyze_dataset_peaks(_damped_scan_record())

    scan = [peak for peak in analysis.peaks if peak.source == "damped_scan"]
    assert len(scan) == 2
    strong, weak = scan
    assert strong.frequency_mhz == pytest.approx(240.0, rel=0.02)
    assert strong.damping_rate_per_us == pytest.approx(44.0, rel=0.4)
    assert strong.amplitude_percent == pytest.approx(4.7, rel=0.3)
    assert weak.frequency_mhz == pytest.approx(120.0, rel=0.02)
    # Δχ²-descending, and each line's width is the Lorentzian FWHM λ/π of its
    # own envelope — not a spectral resolution, which the scan never uses.
    assert strong.delta_chi_squared > weak.delta_chi_squared
    assert strong.width_mhz == pytest.approx(strong.damping_rate_per_us / np.pi)
    # No crop: the damping is the scan variable, so there is nothing to stamp.
    assert all(peak.crop_us is None for peak in scan)
    assert all(peak.burg_confirmed is None for peak in scan)
    # The threshold and the trial count live on the attached analysis, since
    # they are properties of the search rather than of any one line.
    assert analysis.damped_lines is not None
    assert analysis.damped_lines.threshold_delta_chi_squared > 0.0
    assert weak.delta_chi_squared >= analysis.damped_lines.threshold_delta_chi_squared


def test_damped_pass_can_be_switched_off() -> None:
    analysis = analyze_dataset_peaks(_damped_scan_record(), damped_pass=False)

    assert not any(peak.source == "damped_scan" for peak in analysis.peaks)
    assert analysis.damped_lines is None


def _scan_analysis(peaks, *, damped_lines=None) -> PeakAnalysis:
    return PeakAnalysis(
        peaks=tuple(peaks),
        noise_floor=0.0,
        resolution_mhz=0.1,
        nyquist_mhz=500.0,
        detrended=False,
        damped_lines=damped_lines,
    )


def _scan_peak(frequency: float, *, rate: float = 40.0, dchi2: float = 120.0) -> DetectedPeak:
    return DetectedPeak(
        frequency_mhz=frequency,
        amplitude=5.0,
        snr=12.0,
        width_mhz=rate / np.pi,
        prominence=0.0,
        source="damped_scan",
        crop_us=None,
        damping_rate_per_us=rate,
        amplitude_percent=5.0,
        phase_rad=0.3,
        delta_chi_squared=dchi2,
    )


def _hann_peak(frequency: float, *, snr: float = 10.0) -> DetectedPeak:
    return DetectedPeak(
        frequency_mhz=frequency,
        amplitude=100.0,
        snr=snr,
        width_mhz=0.05,
        prominence=10.0,
        source="fft",
    )


def test_merge_lets_the_hann_pass_keep_the_frequency_but_take_the_measurement() -> None:
    from asymmetry.core.fitting.peak_detection import merge_damped_scan_peaks

    hann = _scan_analysis([_hann_peak(240.0)])
    # 12.7 MHz linewidth, so 245 MHz is the same line.
    scan = _scan_analysis([_scan_peak(245.0, rate=40.0)])

    merged = merge_damped_scan_peaks(hann, scan)

    assert len(merged.peaks) == 1
    kept = merged.peaks[0]
    assert kept.source == "fft"
    assert kept.frequency_mhz == pytest.approx(240.0)
    assert kept.damping_rate_per_us == pytest.approx(40.0)
    assert kept.amplitude_percent == pytest.approx(5.0)
    assert kept.phase_rad == pytest.approx(0.3)


def test_merge_adds_a_line_the_hann_pass_missed_and_reserves_a_slot() -> None:
    from asymmetry.core.fitting.peak_detection import merge_damped_scan_peaks

    hann = _scan_analysis([_hann_peak(float(f), snr=20.0 - f) for f in (1.0, 2.0, 3.0, 4.0)])
    scan = _scan_analysis([_scan_peak(240.0), _scan_peak(120.0)])

    merged = merge_damped_scan_peaks(hann, scan, max_peaks=4)

    assert [peak.source for peak in merged.peaks] == ["fft", "fft", "damped_scan", "damped_scan"]
    # The weakest Hann peaks are displaced, never the strongest.
    assert [peak.frequency_mhz for peak in merged.peaks[:2]] == [1.0, 2.0]


def test_fresh_user_frequency_inherits_a_nearby_scan_measurement() -> None:
    # The match radius above is the Hann resolution (0.1 MHz), so a click 4 MHz
    # off a 12.7 MHz-wide line lands "fresh" while plainly naming that line.
    analysis = _scan_analysis([_scan_peak(240.0, rate=40.0)])

    merged = merge_user_peaks(analysis, [244.0])

    user = merged.peaks[0]
    assert user.source == "user"
    assert user.snr == USER_PEAK_SNR_SENTINEL
    assert user.frequency_mhz == pytest.approx(244.0)
    assert user.damping_rate_per_us == pytest.approx(40.0)
    assert user.amplitude_percent == pytest.approx(5.0)
    assert user.phase_rad == pytest.approx(0.3)
    # ...and it takes the line's width with them: the seeding path cuts the
    # frequency box from that width, and a click's nominal width is the Hann
    # resolution, which would bound a heavily damped line far too tightly.
    assert user.width_mhz == pytest.approx(40.0 / np.pi)


def test_user_frequency_far_from_every_scan_line_inherits_nothing() -> None:
    analysis = _scan_analysis([_scan_peak(240.0, rate=40.0)])

    merged = merge_user_peaks(analysis, [180.0])

    assert merged.peaks[0].damping_rate_per_us is None


def test_scan_lines_inside_one_linewidth_collapse_to_one_seed() -> None:
    """Two peaks a fit cannot separate must not spend two components."""
    from asymmetry.core.fitting.peak_detection import merge_damped_scan_peaks

    hann = _scan_analysis([])
    # 12.7 MHz linewidth; 4 MHz apart is one line the peeling split in two.
    scan = _scan_analysis([_scan_peak(240.0, dchi2=500.0), _scan_peak(244.0, dchi2=90.0)])
    merged = merge_damped_scan_peaks(hann, scan)

    assert [peak.frequency_mhz for peak in merged.peaks] == [240.0, 244.0], (
        "the merge itself must not dedupe — that is the scan pass's own rule"
    )


def test_analyze_damped_scan_peaks_dedupes_inside_one_linewidth() -> None:
    """The dedupe lives where the peaks are built, and keeps the stronger line."""
    from asymmetry.core.fitting.peak_detection import analyze_damped_scan_peaks

    # 30 MHz line at λ = 6 µs⁻¹ on a short, finely binned record: this is the
    # shape whose peeling produced a 30.02/30.66 pair against a 1.9 MHz width.
    time = np.arange(1, 1201, dtype=float) * (4.0 / 1200.0)
    sigma = np.minimum(0.9 * np.exp(time / (2.0 * 2.197)), 100.0)
    rng = np.random.default_rng(20260730)
    signal = (
        6.0 * np.exp(-0.25 * time)
        + 0.5
        + 24.0 * np.exp(-10.0 * time) * np.cos(2.0 * np.pi * 60.0 * time)
        + 16.0 * np.exp(-6.0 * time) * np.cos(2.0 * np.pi * 30.0 * time)
    )
    analysis = analyze_damped_scan_peaks(time, signal + rng.normal(0.0, sigma), sigma)

    frequencies = sorted(peak.frequency_mhz for peak in analysis.peaks)
    assert len(frequencies) == 2
    assert frequencies[0] == pytest.approx(30.0, rel=0.05)
    assert frequencies[1] == pytest.approx(60.0, rel=0.05)
    for peak, other in zip(analysis.peaks, analysis.peaks[1:]):
        assert abs(peak.frequency_mhz - other.frequency_mhz) >= max(peak.width_mhz, other.width_mhz)


def test_detected_peak_round_trip_carries_the_measurement() -> None:
    peak = _scan_peak(240.0, rate=42.5, dchi2=141.2)

    assert deserialize_detected_peak(serialize_detected_peak(peak)) == peak


def test_detected_peak_payloads_predating_the_measurement_still_load() -> None:
    payload = {
        "frequency_mhz": 1.4,
        "amplitude": 100.0,
        "snr": 12.0,
        "width_mhz": 0.05,
        "prominence": 3.0,
        "source": "fft",
        "burg_confirmed": True,
        "crop_us": None,
    }

    restored = deserialize_detected_peak(payload)

    assert restored is not None
    assert restored.damping_rate_per_us is None
    assert restored.amplitude_percent is None
    assert restored.phase_rad is None
    assert restored.delta_chi_squared is None


def test_peak_analysis_round_trip_carries_the_scan_result() -> None:
    analysis = analyze_dataset_peaks(_damped_scan_record())

    restored = deserialize_peak_analysis(serialize_peak_analysis(analysis))

    assert restored == analysis
    assert restored is not None and restored.damped_lines is not None
    assert restored.damped_lines.lines == analysis.damped_lines.lines


def test_peak_analysis_payloads_predating_the_scan_still_load() -> None:
    payload = serialize_peak_analysis(
        PeakAnalysis(
            peaks=(_hann_peak(1.4),),
            noise_floor=1.0,
            resolution_mhz=0.1,
            nyquist_mhz=50.0,
            detrended=False,
        )
    )
    payload.pop("damped_lines")

    restored = deserialize_peak_analysis(payload)

    assert restored is not None
    assert restored.damped_lines is None


def test_nyquist_guard_rejects_junk_on_a_finely_binned_record() -> None:
    """0.1 ns binning: the top 2 % of the band must contribute no ``fft`` peak.

    White noise on a slow relaxation at 5000 MHz Nyquist has a resolution
    element four orders of magnitude narrower than Nyquist, so the historical
    half-resolution guard leaves the aliasing/roll-off edge effectively
    unguarded and the pass returns clusters of peaks hard against it.
    """
    dataset = _damped_scan_record(lines=(), seed=7, n_points=40_000)

    analysis = analyze_dataset_peaks(dataset, damped_pass=False)

    nyquist = analysis.nyquist_mhz
    assert nyquist > 4000.0
    offenders = [
        peak.frequency_mhz
        for peak in analysis.peaks
        if peak.source == "fft" and peak.frequency_mhz > 0.98 * nyquist
    ]
    assert offenders == []


def test_nyquist_guard_leaves_a_coarse_record_alone() -> None:
    """The floor is a fraction of Nyquist, so it does not touch conventional data.

    At 6.6 ns binning over 10 µs the guard is 1.5 MHz out of 75 — wide in
    resolution elements, but nowhere near the 1.4 MHz line the pass is for.
    """
    time = np.linspace(0.05, 10.0, 1500)
    signal = 18.0 * np.exp(-0.12 * time) * np.cos(2.0 * np.pi * 1.4 * time) + 3.0
    rng = np.random.default_rng(4242)
    dataset = MuonDataset(
        time=time,
        asymmetry=signal + rng.normal(0.0, 0.6, size=time.size),
        error=np.full_like(time, 0.6),
        metadata={"run_number": 1},
    )

    analysis = analyze_dataset_peaks(dataset, damped_pass=False, burg_check="never")

    assert any(abs(peak.frequency_mhz - 1.4) < 0.1 for peak in analysis.peaks)
