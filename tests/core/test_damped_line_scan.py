"""Tests for the matched-apodisation damped-line scan.

Every record here is synthetic: µSR-like in shape (percent asymmetry, error
bars growing as ``exp(t/τ_µ)`` until they saturate) but generated from a seeded
RNG, so the acceptance numbers below are reproducible and no measured data is
involved.

Two of the records are deliberately *hard*, because the easy version of each
was what let a regression through.  The two-line record is binned and counted
like a real one, so its second line is marginal on the scan (best SNR ~7) and
only the Δχ² test can decide it; the three-line record is a cluster spanning a
factor 2.5 in frequency, whose members sit inside a few line widths of each
other and so are each other's noise floor and each other's model.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable

import numpy as np
import pytest

from asymmetry.core.fitting.damped_line_scan import (
    DampedLine,
    DampedLineAnalysis,
    cluster_scan_peaks,
    damped_line_delta_chi2,
    deserialize_damped_line,
    deserialize_damped_line_analysis,
    detect_damped_lines,
    look_elsewhere_threshold,
    matched_apodisation_scan,
    refine_line,
    serialize_damped_line,
    serialize_damped_line_analysis,
    tau_ladder,
)
from asymmetry.core.transform.rebin import rebin

#: µSR error growth: σ doubles roughly every 3 µs and saturates at 25 % once
#: the record is pure noise, which is what makes
#: ``effective_analysis_window`` truncate.
_ERROR_LIFETIME_US = 4.4
_ERROR_CAP_PERCENT = 25.0

#: Reference per-bin σ (percent asymmetry) at 0.1 ns binning.  Coarser binnings
#: scale it as ``1/√(bin width)`` — a wider bin holds proportionally more
#: counts — so the records differ in resolution, not in total statistics.
_SIGMA0_AT_0P1NS = 1.5

#: Per-bin σ of a finely binned real record: 0.1 ns bins hold few counts, and
#: 3.3 % per bin is what a 90 000-bin record actually carries at ``t = 0``.
_SIGMA0_FINE = 3.3


def _record(
    curve: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    n_points: int = 90_000,
    bin_width_us: float = 1e-4,
    sigma0_percent: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(time, asymmetry, error)`` for a synthetic µSR-like record."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=float) * bin_width_us
    if sigma0_percent is None:
        sigma0_percent = _SIGMA0_AT_0P1NS
    sigma0_percent *= np.sqrt(1e-4 / bin_width_us)
    error = np.minimum(sigma0_percent * np.exp(time / _ERROR_LIFETIME_US), _ERROR_CAP_PERCENT)
    return time, curve(time) + rng.normal(0.0, error), error


def _two_damped_lines(t: np.ndarray) -> np.ndarray:
    """4.7 % at 240 MHz (λ = 44) + 1.9 % at 120 MHz (λ = 22) on a relaxing tail."""
    return (
        4.7 * np.exp(-44.0 * t) * np.cos(2.0 * np.pi * 240.0 * t + 0.3)
        + 1.9 * np.exp(-22.0 * t) * np.cos(2.0 * np.pi * 120.0 * t - 0.7)
        + 2.6 * np.exp(-0.19 * t)
        + 4.6
    )


def _three_damped_lines(t: np.ndarray) -> np.ndarray:
    """A cluster: 45/75/115 MHz at λ = 40/65/65 µs⁻¹ and 4/6/3.5 %.

    The lines are 30 and 40 MHz apart while their own widths (``λ/π``) are 13
    and 21 MHz, so each sits within about two widths of its neighbour.
    """
    return (
        4.0 * np.exp(-40.0 * t) * np.cos(2.0 * np.pi * 45.0 * t + 0.3)
        + 6.0 * np.exp(-65.0 * t) * np.cos(2.0 * np.pi * 75.0 * t - 0.7)
        + 3.5 * np.exp(-65.0 * t) * np.cos(2.0 * np.pi * 115.0 * t + 1.1)
        + 5.0 * np.exp(-0.2 * t)
        + 0.4
    )


def _very_fast_line(t: np.ndarray) -> np.ndarray:
    """8 % at 200 MHz with λ = 100 µs⁻¹ — gone within ~30 ns."""
    return 8.0 * np.exp(-100.0 * t) * np.cos(2.0 * np.pi * 200.0 * t) + 6.0 * np.exp(-0.4 * t) + 0.2


def _slow_transverse_field_line(t: np.ndarray) -> np.ndarray:
    """A conventional 20 % TF line: 1 MHz, λ = 0.2 µs⁻¹."""
    return 20.0 * np.exp(-0.2 * t) * np.cos(2.0 * np.pi * 1.0 * t + 0.1) + 1.0


def _flat(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def _exponential(t: np.ndarray) -> np.ndarray:
    return 20.0 * np.exp(-0.7 * t) + 0.5


def _stretched_exponential(t: np.ndarray) -> np.ndarray:
    return 18.0 * np.exp(-((0.6 * t) ** 0.5)) + 0.3


def _static_gaussian_kubo_toyabe(t: np.ndarray) -> np.ndarray:
    """Zero-field static Gaussian KT: a dip and a 1/3 tail, no oscillation."""
    delta = 0.8
    relaxation = (1.0 / 3.0) + (2.0 / 3.0) * (1.0 - (delta * t) ** 2) * np.exp(
        -0.5 * (delta * t) ** 2
    )
    return 20.0 * relaxation + 0.2


def _line_at(analysis: DampedLineAnalysis, frequency_mhz: float) -> DampedLine:
    """Return the accepted line nearest ``frequency_mhz`` (asserting there is one)."""
    assert analysis.lines, "no lines detected"
    return min(analysis.lines, key=lambda line: abs(line.frequency_mhz - frequency_mhz))


def _best_scan_snr(
    time: np.ndarray,
    asymmetry: np.ndarray,
    error: np.ndarray,
    frequencies_mhz: tuple[float, ...],
) -> dict[float, float]:
    """Best scan SNR each of ``frequencies_mhz`` reaches anywhere on the ladder."""
    from asymmetry.core.fitting.peak_detection import effective_analysis_window

    end = int(effective_analysis_window(time, error))
    time, asymmetry, error = time[:end], asymmetry[:end], error[:end]
    taus = tau_ladder(float(time[1] - time[0]), float(time[-1] - time[0]))
    best = dict.fromkeys(frequencies_mhz, 0.0)
    for rung in matched_apodisation_scan(time, asymmetry, error, taus):
        for peak in rung.peaks:
            for frequency in frequencies_mhz:
                near = max(3.0 * rung.fwhm_mhz, 0.03 * frequency)
                if abs(peak.frequency_mhz - frequency) < near:
                    best[frequency] = max(best[frequency], peak.snr)
    return best


# --------------------------------------------------------------------------- #
# Ladder, threshold and scan primitives
# --------------------------------------------------------------------------- #


def test_tau_ladder_spans_twenty_bins_to_half_the_record() -> None:
    taus = tau_ladder(1e-4, 8.0)

    assert taus[0] == pytest.approx(2e-3)  # 20 bins of 0.1 ns
    assert taus[-1] == pytest.approx(4.0)
    assert np.all(np.diff(taus) > 0.0)
    # ~4 rungs per decade over 3.3 decades.
    assert 13 <= taus.size <= 16


def test_tau_ladder_never_goes_below_a_nanosecond() -> None:
    assert tau_ladder(1e-6, 8.0)[0] == pytest.approx(1e-3)


def test_tau_ladder_degenerates_to_one_rung_on_a_short_record() -> None:
    assert tau_ladder(1e-3, 0.002).size == 1
    assert tau_ladder(0.0, 0.0).size == 1


def test_look_elsewhere_threshold_grows_with_trials_and_with_strictness() -> None:
    modest = look_elsewhere_threshold(100.0)
    large = look_elsewhere_threshold(100_000.0)

    assert large > modest
    assert large - modest == pytest.approx(2.0 * np.log(1000.0))
    assert look_elsewhere_threshold(100.0, 0.001) > modest
    # 2·ln(N/α) exactly, and never below the single-trial value.
    assert modest == pytest.approx(2.0 * np.log(100.0 / 0.01))
    assert look_elsewhere_threshold(0.5) == pytest.approx(2.0 * np.log(1.0 / 0.01))
    with pytest.raises(ValueError):
        look_elsewhere_threshold(10.0, 0.0)


def test_scan_rung_records_its_cost_control_and_finds_the_line() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=3, n_points=40_000)
    taus = np.asarray([0.02, 1.0])

    rungs = matched_apodisation_scan(time, asymmetry, error, taus, sample_budget=4096)

    short, long_rung = rungs
    assert short.rebin_factor == 1  # 10·τ = 0.2 µs is only 2000 samples
    assert short.n_samples == 2000
    assert long_rung.rebin_factor > 1  # 10 µs of 0.1 ns bins must be rebinned
    assert long_rung.n_samples <= 4096
    assert long_rung.bin_width_us > short.bin_width_us
    assert short.fwhm_mhz == pytest.approx(1.0 / (np.pi * 0.02))
    assert short.band_lo_mhz == pytest.approx(3.0 / 0.02)
    assert short.n_trials > 0.0
    assert any(abs(peak.frequency_mhz - 240.0) < 5.0 for peak in short.peaks)


def test_cluster_scan_peaks_merges_rungs_and_keeps_the_matched_lifetime() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=4, n_points=40_000)
    dt = float(time[1] - time[0])
    rungs = matched_apodisation_scan(
        time, asymmetry, error, tau_ladder(dt, float(time[-1] - time[0]))
    )

    candidates = cluster_scan_peaks(rungs)

    # The gate is low enough to admit noise peaks — that is what the Δχ² stage
    # is for — but the two real lines are the two strongest candidates, each
    # seen on several rungs.
    strongest, second = candidates[0], candidates[1]
    assert {round(strongest.frequency_mhz, -1), round(second.frequency_mhz, -1)} == {240.0, 120.0}
    assert strongest.n_rungs > 1
    assert second.n_rungs > 1
    # The rung of maximum SNR is the matched one: τ* ≈ 1/λ = 23 ns.
    assert 0.005 < strongest.tau_us < 0.15
    assert candidates[0].snr >= candidates[1].snr


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_two_damped_lines_are_both_recovered() -> None:
    """Both lines, with the second one marginal on the scan.

    At this binning and counting statistics the 120 MHz line peaks at a scan
    SNR of ~7 — below the gate an earlier revision of this module used, and
    the reason it was never verified at all.  The range is asserted so the
    record stays in that regime: the scan's job here is only to shortlist it,
    and the Δχ² test's job is to decide it.
    """
    from asymmetry.core.fitting.damped_line_scan import _SCAN_MIN_SNR

    time, asymmetry, error = _record(_two_damped_lines, seed=39, sigma0_percent=_SIGMA0_FINE)

    scan_snr = _best_scan_snr(time, asymmetry, error, (240.0, 120.0))
    analysis = detect_damped_lines(time, asymmetry, error)

    assert _SCAN_MIN_SNR < scan_snr[120.0] < 8.0
    assert scan_snr[240.0] > 10.0

    assert len(analysis.lines) == 2
    strong, weak = analysis.lines
    # Δχ² ordering: the 4.7 % line outranks the 1.9 % one.
    assert strong.delta_chi_squared > weak.delta_chi_squared
    assert weak.delta_chi_squared > analysis.threshold_delta_chi_squared

    assert strong.frequency_mhz == pytest.approx(240.0, rel=0.02)
    assert strong.damping_rate_per_us == pytest.approx(44.0, rel=0.3)
    assert strong.amplitude_percent == pytest.approx(4.7, rel=0.25)

    assert weak.frequency_mhz == pytest.approx(120.0, rel=0.02)
    assert weak.damping_rate_per_us == pytest.approx(22.0, rel=0.3)
    assert weak.amplitude_percent == pytest.approx(1.9, rel=0.25)

    # The informative window truncated the noise tail, and the ladder is real.
    assert analysis.window_end_index < time.size
    assert len(analysis.taus_us) > 8
    assert analysis.n_trials > 100.0


def test_two_damped_lines_are_recovered_on_a_coarser_record() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=39, n_points=25_000, bin_width_us=4e-4)

    analysis = detect_damped_lines(time, asymmetry, error)

    assert len(analysis.lines) == 2
    strong, weak = analysis.lines
    assert strong.frequency_mhz == pytest.approx(240.0, rel=0.02)
    assert strong.damping_rate_per_us == pytest.approx(44.0, rel=0.3)
    assert weak.frequency_mhz == pytest.approx(120.0, rel=0.02)
    assert weak.damping_rate_per_us == pytest.approx(22.0, rel=0.3)


def test_a_cluster_of_three_lines_is_resolved() -> None:
    """Three lines inside a factor 2.5 in frequency, each within ~2 widths of
    its neighbour: the case a per-line running noise floor reports as empty and
    a single unbounded fit reports as one overdamped blob.
    """
    time, asymmetry, error = _record(_three_damped_lines, seed=0, sigma0_percent=1.8)

    analysis = detect_damped_lines(time, asymmetry, error)

    assert len(analysis.lines) == 3
    for truth, amplitude in ((45.0, 4.0), (75.0, 6.0), (115.0, 3.5)):
        line = _line_at(analysis, truth)
        assert line.frequency_mhz == pytest.approx(truth, rel=0.05)
        assert line.delta_chi_squared > analysis.threshold_delta_chi_squared
        # A line fitted inside an unresolved cluster is a biased measurement of
        # its own envelope — the neighbours it is not modelling push λ around —
        # so these are order-of-magnitude checks, not the 30 % of an isolated
        # line.  What must not happen is a merged blob: one envelope covering
        # two lines reports λ two to three times the truth and an amplitude
        # well over the strongest line's.
        assert 0.5 * 65.0 < line.damping_rate_per_us < 2.0 * 65.0
        assert line.amplitude_percent == pytest.approx(amplitude, rel=0.4)
    frequencies = sorted(line.frequency_mhz for line in analysis.lines)
    assert np.all(np.diff(frequencies) > 20.0)


def test_a_cluster_does_not_become_its_own_noise_floor() -> None:
    """The scan-side half of the cluster problem.

    A noise floor that follows the spectrum too closely slides from one member
    of a cluster to the next without ever seeing the white part of the band, and
    the lines end up a few MADs above a floor they lifted themselves.  Measured
    on this record, a sixteen-line-width window scores these two lines 4.9 and
    11.7; the window this module uses scores them well clear of any plausible
    shortlist gate.
    """
    time, asymmetry, error = _record(_three_damped_lines, seed=0, sigma0_percent=_SIGMA0_FINE)

    scan_snr = _best_scan_snr(time, asymmetry, error, (45.0, 75.0))

    assert scan_snr[45.0] > 8.0
    assert scan_snr[75.0] > 8.0


def test_a_very_fast_line_is_found_without_any_time_crop() -> None:
    time, asymmetry, error = _record(_very_fast_line, seed=1)

    analysis = detect_damped_lines(time, asymmetry, error)

    assert len(analysis.lines) == 1
    line = analysis.lines[0]
    assert line.frequency_mhz == pytest.approx(200.0, rel=0.02)
    assert line.damping_rate_per_us == pytest.approx(100.0, rel=0.3)
    assert line.amplitude_percent == pytest.approx(8.0, rel=0.25)
    # It lives for ~10 ns, so the matched rung is a small fraction of a µs.
    assert line.tau_us < 0.1


def test_a_conventional_slow_transverse_field_line_is_found_too() -> None:
    time, asymmetry, error = _record(
        _slow_transverse_field_line, seed=2, n_points=8_000, bin_width_us=1e-3
    )

    analysis = detect_damped_lines(time, asymmetry, error)

    line = _line_at(analysis, 1.0)
    assert line.frequency_mhz == pytest.approx(1.0, rel=0.02)
    assert line.damping_rate_per_us == pytest.approx(0.2, rel=0.5)
    assert line.amplitude_percent == pytest.approx(20.0, rel=0.25)


def test_peeling_reports_lines_in_delta_chi_squared_order() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=5, sigma0_percent=_SIGMA0_FINE)

    analysis = detect_damped_lines(time, asymmetry, error)

    deltas = [line.delta_chi_squared for line in analysis.lines]
    assert deltas == sorted(deltas, reverse=True)
    assert all(delta >= analysis.threshold_delta_chi_squared for delta in deltas)


def test_reported_delta_chi_squared_is_measured_on_the_full_informative_record() -> None:
    """Every line's Δχ² is a leave-one-out value on one common record.

    The (f, λ) search runs on a short per-candidate crop, purely for speed.  If
    the reported Δχ² came from that crop it would depend on how long the crop
    was, which depends on the line's own λ — so two lines' values would not be
    comparable, and neither would be comparable with the threshold.
    """
    time, asymmetry, error = _record(_two_damped_lines, seed=39, sigma0_percent=_SIGMA0_FINE)
    analysis = detect_damped_lines(time, asymmetry, error)
    end = analysis.window_end_index

    for line in analysis.lines:
        others = tuple(
            (other.frequency_mhz, other.damping_rate_per_us)
            for other in analysis.lines
            if other is not line
        )
        delta, amplitude, phase = damped_line_delta_chi2(
            time[:end],
            asymmetry[:end],
            error[:end],
            line.frequency_mhz,
            line.damping_rate_per_us,
            extra_lines=others,
        )
        assert delta == pytest.approx(line.delta_chi_squared, rel=1e-6)
        assert amplitude == pytest.approx(line.amplitude_percent, rel=1e-6)
        assert phase == pytest.approx(line.phase_rad, rel=1e-6)


def test_max_lines_caps_the_number_of_accepted_lines() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=39, sigma0_percent=_SIGMA0_FINE)

    analysis = detect_damped_lines(time, asymmetry, error, max_lines=1)

    assert len(analysis.lines) == 1
    assert analysis.lines[0].frequency_mhz == pytest.approx(240.0, rel=0.02)


def test_a_rebinned_copy_of_the_record_yields_the_same_lines() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=39, sigma0_percent=_SIGMA0_FINE)
    coarse = rebin(time, asymmetry, error, 2)

    original = detect_damped_lines(time, asymmetry, error)
    rebinned = detect_damped_lines(*coarse)

    assert len(rebinned.lines) == len(original.lines) == 2
    for before, after in zip(original.lines, rebinned.lines, strict=True):
        assert after.frequency_mhz == pytest.approx(before.frequency_mhz, rel=0.02)
        assert after.damping_rate_per_us == pytest.approx(before.damping_rate_per_us, rel=0.3)
        assert after.amplitude_percent == pytest.approx(before.amplitude_percent, rel=0.15)


def test_short_or_degenerate_records_return_an_empty_analysis() -> None:
    time = np.arange(16, dtype=float) * 1e-3
    empty = detect_damped_lines(time, np.zeros_like(time), np.ones_like(time))
    assert empty.lines == ()
    assert empty.taus_us == ()

    flat_time = np.zeros(256)
    degenerate = detect_damped_lines(flat_time, np.zeros_like(flat_time), np.ones_like(flat_time))
    assert degenerate.lines == ()


# --------------------------------------------------------------------------- #
# Nulls — no oscillation must ever be reported
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "curve",
    [_flat, _exponential, _stretched_exponential, _static_gaussian_kubo_toyabe],
    ids=["noise", "exponential", "stretched", "kubo-toyabe"],
)
def test_non_oscillatory_records_yield_no_lines_over_100_seeds(
    curve: Callable[[np.ndarray], np.ndarray],
) -> None:
    accepted = [
        line
        for seed in range(100)
        for line in detect_damped_lines(
            *_record(curve, seed=seed, n_points=4_000, bin_width_us=2e-3)
        ).lines
    ]

    assert accepted == []


def test_a_kubo_toyabe_dip_is_rejected_as_a_relaxation_shape() -> None:
    """The gate that the Δχ² statistic alone cannot supply.

    The dip is a real, highly significant misfit of the monotonic dictionary —
    ``damped_line_delta_chi2`` says so — and a bounded refinement no longer
    walks it off to an absurd envelope, so what rejects it is that the damped
    cosine fitting it neither oscillates (fewer than three cycles per lifetime)
    nor dies faster than the background dictionary can describe.
    """
    time, asymmetry, error = _record(
        _static_gaussian_kubo_toyabe, seed=0, n_points=4_000, bin_width_us=2e-3
    )

    delta, _amplitude, _phase = damped_line_delta_chi2(time, asymmetry, error, 3.0, 4.0)
    _refined_delta, frequency, damping = refine_line(time, asymmetry, error, 3.0, 4.0)

    assert delta > look_elsewhere_threshold(1.0)
    assert frequency < 3.0 * damping
    assert damping < 20.0
    assert detect_damped_lines(time, asymmetry, error).lines == ()


# --------------------------------------------------------------------------- #
# Δχ² primitives
# --------------------------------------------------------------------------- #


def test_delta_chi_squared_recovers_amplitude_and_phase() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=6, n_points=20_000)

    delta, amplitude, phase = damped_line_delta_chi2(time, asymmetry, error, 240.0, 44.0)

    assert delta > 100.0
    assert amplitude == pytest.approx(4.7, rel=0.2)
    assert phase == pytest.approx(0.3, abs=0.3)


def test_delta_chi_squared_is_near_zero_for_a_line_that_is_not_there() -> None:
    time, asymmetry, error = _record(_exponential, seed=7, n_points=4_000, bin_width_us=2e-3)

    delta, _amplitude, _phase = damped_line_delta_chi2(time, asymmetry, error, 33.0, 5.0)

    assert 0.0 <= delta < 15.0


def test_peeling_removes_a_line_from_the_residual() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=6, n_points=20_000)

    alone, _amplitude, _phase = damped_line_delta_chi2(time, asymmetry, error, 240.0, 44.0)
    peeled, _amplitude, _phase = damped_line_delta_chi2(
        time, asymmetry, error, 240.0, 44.0, extra_lines=((240.0, 44.0),)
    )

    assert alone > 50.0
    # Not bit-exact zero: the pair is already in the basis, so what is left is
    # the rank-truncation residue of the orthonormalisation.
    assert peeled < 1.0


def test_refine_line_improves_on_a_deliberately_offset_seed() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=6, n_points=20_000)
    seeded, _amplitude, _phase = damped_line_delta_chi2(time, asymmetry, error, 235.0, 18.0)

    delta, frequency, damping = refine_line(time, asymmetry, error, 235.0, 18.0)

    assert delta > seeded
    assert frequency == pytest.approx(240.0, rel=0.02)
    assert damping == pytest.approx(44.0, rel=0.3)


def test_refine_line_never_leaves_the_box_its_seed_defines() -> None:
    """λ is bounded, and deliberately so.

    Δχ² does not fall away on the short-envelope side the way a well-posed
    likelihood should — a shorter envelope leaves fewer points at which the
    nuisance model can be wrong — so an unbounded search drifts to several times
    the true rate, and on a cluster it merges neighbouring lines into one
    overdamped blob.  Seeded far below the truth, the refinement must stop at
    the edge of its box rather than run.
    """
    from asymmetry.core.fitting.damped_line_scan import (
        _REFINE_DAMPING_FACTOR,
        _REFINE_FREQUENCY_FWHMS,
    )

    time, asymmetry, error = _record(_two_damped_lines, seed=6, n_points=20_000)
    seed_lambda = 5.0

    _delta, frequency, damping = refine_line(time, asymmetry, error, 235.0, seed_lambda)

    assert seed_lambda / _REFINE_DAMPING_FACTOR <= damping <= seed_lambda * _REFINE_DAMPING_FACTOR
    half_width = _REFINE_FREQUENCY_FWHMS * seed_lambda / np.pi
    assert 235.0 - half_width <= frequency <= 235.0 + half_width
    # ...and a caller that knows how good its seed is can say so.
    _delta, frequency, _damping = refine_line(
        time, asymmetry, error, 235.0, seed_lambda, frequency_span_mhz=1.0
    )
    assert 234.0 <= frequency <= 236.0


# --------------------------------------------------------------------------- #
# Serialization and runtime
# --------------------------------------------------------------------------- #


def test_serialization_round_trips_an_analysis() -> None:
    time, asymmetry, error = _record(_two_damped_lines, seed=3, n_points=20_000)
    analysis = detect_damped_lines(time, asymmetry, error)
    assert analysis.lines

    payload = serialize_damped_line_analysis(analysis)
    restored = deserialize_damped_line_analysis(payload)

    assert restored == analysis
    line_payload = serialize_damped_line(analysis.lines[0])
    assert isinstance(line_payload["frequency_mhz"], float)
    assert deserialize_damped_line(line_payload) == analysis.lines[0]


def test_deserialization_tolerates_junk_and_gaps() -> None:
    assert deserialize_damped_line(None) is None
    assert deserialize_damped_line_analysis("not a payload") is None
    assert deserialize_damped_line({}) == DampedLine(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    recovered = deserialize_damped_line_analysis({"lines": [{}, "junk"]})
    assert recovered is not None
    assert len(recovered.lines) == 1
    assert recovered.taus_us == ()


def test_detection_stays_fast_on_a_hundred_thousand_point_record() -> None:
    time, asymmetry, error = _record(
        _two_damped_lines, seed=39, n_points=100_000, sigma0_percent=_SIGMA0_FINE
    )

    started = _time.perf_counter()
    analysis = detect_damped_lines(time, asymmetry, error)
    elapsed = _time.perf_counter() - started

    assert len(analysis.lines) == 2
    # Measured ~0.35 s locally, peeling rescans included; the bound is generous
    # for a loaded CI runner.
    assert elapsed < 5.0
