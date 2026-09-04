"""Acceptance tests for blind seeding of heavily damped oscillations.

The feature under test: a µSR record whose oscillation is heavily damped lives
only in the leading nanoseconds of the record, which every symmetric apodisation
deletes.  Both of the fit wizard's seeding FFTs used to be Hann-windowed, so a
damped-cosine recommendation was only reachable when the user handed the wizard
the frequencies.  These tests pin the blind path: the matched-apodisation
damped-line scan finds the line, the fingerprint carries it, the seeding path
uses its measured frequency/λ/amplitude/phase, and a damped-oscillation family
ranks top with **no** ``user_frequencies_mhz``.

The historical unwindowed early-window crop ladder
(``analyze_early_window_peaks``) is no longer wired into
``analyze_dataset_peaks``; it is still exercised directly here, because its
constants rest on a study whose conclusions the scan inherits.

Every synthetic parameter here is invented.  The records are ordered-magnet-like
zero-field signals: one or two damped cosines on a slowly relaxing tail, at fine
binning, with µSR-like errors that grow as ``exp(t / 2·τ_µ)`` and are capped at
100 %.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    TemplateSeedContext,
    _damped_envelope_rate,
    _initial_parameters_for_template,
    _multiplet_seed_peaks,
    build_fit_wizard_recommendation,
    build_oscillatory_multiplet_templates,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.parameters import split_parameter_name
from asymmetry.core.fitting.peak_detection import (
    _EARLY_MIN_POINTS,
    DetectedPeak,
    PeakAnalysis,
    analyze_dataset_peaks,
    analyze_early_window_peaks,
    deserialize_detected_peak,
    early_window_crops,
    effective_analysis_window,
    merge_early_peaks,
    merge_user_peaks,
    serialize_detected_peak,
)

pytestmark = [pytest.mark.unit]

#: Muon lifetime (µs) used to shape the invented error growth.
_TAU_MU = 2.197

#: The invented damped line: 60 MHz, envelope rate 10 µs⁻¹ (1/e time 100 ns) at
#: 24 % amplitude.  Dead inside the first ~300 ns of a 4 µs record.
_LINE_MHZ = 60.0
_LINE_RATE = 10.0
_LINE_AMPLITUDE = 24.0

#: The invented second line, at half the frequency, weaker and longer-lived.
_SECOND_LINE_MHZ = 30.0
_SECOND_LINE_RATE = 6.0
_SECOND_LINE_AMPLITUDE = 16.0

#: Deterministic seed for every record built here.
_SEED = 20260730


#: Record shape: 1200 points over 4 µs (3.3 ns binning, 150 MHz Nyquist), a
#: 6 % tail relaxing at 0.25 µs⁻¹ on a 0.5 % baseline, per-point σ starting at
#: 0.9 % and growing as exp(t / 2·τ_µ).  All invented.
_N_POINTS = 1200
_T_MAX = 4.0
_TAIL_AMPLITUDE = 6.0
_TAIL_RATE = 0.25
_BASELINE = 0.5
_SIGMA0 = 0.9


def _damped_record(
    *,
    seed: int = _SEED,
    lines: tuple[tuple[float, float, float], ...] = ((_LINE_AMPLITUDE, _LINE_MHZ, _LINE_RATE),),
) -> MuonDataset:
    """Damped precession on a slowly relaxing tail. All values invented."""
    dt = _T_MAX / _N_POINTS
    time = np.arange(dt, _T_MAX + 0.5 * dt, dt)[:_N_POINTS]
    sigma = np.minimum(_SIGMA0 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    asymmetry = _TAIL_AMPLITUDE * np.exp(-_TAIL_RATE * time) + _BASELINE
    for amplitude, frequency, rate in lines:
        asymmetry = asymmetry + amplitude * np.exp(-rate * time) * np.cos(
            2.0 * np.pi * frequency * time
        )
    rng = np.random.default_rng(seed)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry + rng.normal(0.0, sigma),
        error=sigma,
        metadata={"run_number": 1, "field": 0.0, "field_direction": "ZF"},
    )


def _two_line_record() -> MuonDataset:
    return _damped_record(
        lines=(
            (_LINE_AMPLITUDE, _LINE_MHZ, _LINE_RATE),
            (_SECOND_LINE_AMPLITUDE, _SECOND_LINE_MHZ, _SECOND_LINE_RATE),
        ),
    )


def _narrow_line_record() -> MuonDataset:
    """A conventional narrow line, alive across the whole record. Invented."""
    time = np.linspace(0.05, 10.0, 1500)
    sigma = np.minimum(0.6 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    rng = np.random.default_rng(4242)
    asymmetry = 18.0 * np.exp(-0.12 * time) * np.cos(2.0 * np.pi * 1.4 * time) + 3.0
    return MuonDataset(
        time=time,
        asymmetry=asymmetry + rng.normal(0.0, sigma),
        error=sigma,
        metadata={"run_number": 2, "field": 100.0, "field_direction": "TF"},
    )


def _pure_noise_record(seed: int) -> MuonDataset:
    time = np.linspace(_T_MAX / _N_POINTS, _T_MAX, _N_POINTS)
    sigma = np.minimum(_SIGMA0 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    rng = np.random.default_rng(seed)
    return MuonDataset(
        time=time,
        asymmetry=rng.normal(0.0, sigma),
        error=sigma,
        metadata={"run_number": 3},
    )


def _tail_only_record(seed: int) -> MuonDataset:
    return _damped_record(seed=seed, lines=())


def _centered_truncated(dataset: MuonDataset):
    """The (t, signal, σ) triple ``analyze_dataset_peaks`` transforms."""
    time = np.asarray(dataset.time, dtype=float)
    error = np.asarray(dataset.error, dtype=float)
    end = effective_analysis_window(time, error)
    y = np.asarray(dataset.asymmetry, dtype=float)
    tail = float(np.mean(y[-max(5, y.size // 5) :]))
    return time[:end], (y - tail)[:end], error[:end]


def _early_peaks(dataset: MuonDataset) -> PeakAnalysis:
    return analyze_early_window_peaks(*_centered_truncated(dataset))


# --------------------------------------------------------------------------- #
# Workstream A — the early-window pass and its study
# --------------------------------------------------------------------------- #


def test_hann_pass_is_structurally_blind_to_the_damped_line() -> None:
    """The premise: the historical Hann-only pass sees nothing at all here.

    Not "sees it weakly" — the window is zero at the first sample, so the line's
    entire support is deleted and the pass returns no peaks.
    """
    analysis = analyze_dataset_peaks(_damped_record(), damped_pass=False)

    assert analysis.peaks == ()


def test_early_pass_finds_the_damped_line_within_its_crop_resolution() -> None:
    early = _early_peaks(_damped_record())

    assert len(early.peaks) == 1
    peak = early.peaks[0]
    assert peak.source == "early_fft"
    assert peak.crop_us is not None and peak.crop_us > 0.0
    # Located within one resolution element of the crop that found it.
    assert abs(peak.frequency_mhz - _LINE_MHZ) <= 1.0 / peak.crop_us
    # ...and comfortably clear of the re-derived gate.
    assert peak.snr > 8.0
    # Burg never judges an early-pass peak.
    assert peak.burg_confirmed is None


def test_early_pass_finds_both_lines_of_the_two_line_record() -> None:
    early = _early_peaks(_two_line_record())

    assert len(early.peaks) == 2
    found = sorted(early.peaks, key=lambda peak: peak.frequency_mhz)
    for peak, target in zip(found, (_SECOND_LINE_MHZ, _LINE_MHZ)):
        assert abs(peak.frequency_mhz - target) <= 1.0 / peak.crop_us


@pytest.mark.parametrize(
    ("build", "seed_base", "draws"),
    [(_pure_noise_record, 90_000, 500), (_tail_only_record, 70_000, 250)],
    ids=["pure-noise", "relaxing-tail"],
)
def test_null_records_add_no_early_peaks(build, seed_base: int, draws: int) -> None:
    """False-seed control (study step 3): zero early-pass peaks on either null.

    ``_EARLY_MIN_SNR`` is re-derived against exactly these draws rather than
    inherited from the Hann pass, so this is the measurement it rests on.  Pure
    noise alone would be too easy a null: an unwindowed short crop of a
    *relaxing* record leaves residual trend curvature, which is what the
    low-frequency guard and the rectangular leakage profile exist to reject.
    """
    total = sum(len(_early_peaks(build(seed_base + i)).peaks) for i in range(draws))

    assert total == 0


def test_conventional_narrow_line_gains_a_measurement_but_no_extra_peak() -> None:
    """No spurious additions on a line the Hann pass can already see.

    The scan finds that line too — it is not restricted to fast ones — so the
    merge is a collision, not an addition: the Hann pass keeps the frequency it
    localised over the whole record, and picks up the λ/amplitude/phase the
    scan measured, which no Hann peak can carry.
    """
    dataset = _narrow_line_record()

    with_scan = analyze_dataset_peaks(dataset, burg_check="never")
    hann_only = analyze_dataset_peaks(dataset, burg_check="never", damped_pass=False)

    assert [peak.source for peak in with_scan.peaks] == [peak.source for peak in hann_only.peaks]
    assert not any(peak.source in ("early_fft", "damped_scan") for peak in with_scan.peaks)
    for merged, plain in zip(with_scan.peaks, hann_only.peaks):
        assert merged.frequency_mhz == plain.frequency_mhz
        assert merged.amplitude == plain.amplitude
        assert merged.snr == plain.snr
    strongest = with_scan.peaks[0]
    assert strongest.damping_rate_per_us == pytest.approx(0.12, rel=0.3)
    assert strongest.amplitude_percent == pytest.approx(18.0, rel=0.2)


def test_early_window_crops_floor_and_collapse() -> None:
    # A long record gets both rungs, in descending length.
    assert early_window_crops(7000) == (438, 109)
    # The shorter rung is floored rather than shrinking without limit.
    assert early_window_crops(4000) == (250, _EARLY_MIN_POINTS)
    # Once the floor would put the second rung on top of the first, it is
    # dropped: two transforms of nearly the same window are two looks at the
    # same noise for no extra reach.
    assert early_window_crops(1200) == (75,)
    assert early_window_crops(200) == (_EARLY_MIN_POINTS,)
    assert early_window_crops(10) == (10,)


# --------------------------------------------------------------------------- #
# Workstream A — merge policy
# --------------------------------------------------------------------------- #


def _analysis(peaks, resolution: float = 0.1) -> PeakAnalysis:
    return PeakAnalysis(
        peaks=tuple(peaks),
        noise_floor=1.0,
        resolution_mhz=resolution,
        nyquist_mhz=500.0,
        detrended=False,
    )


def _hann_peak(frequency: float, snr: float = 10.0) -> DetectedPeak:
    return DetectedPeak(
        frequency_mhz=frequency,
        amplitude=5.0,
        snr=snr,
        width_mhz=0.05,
        prominence=1.0,
        source="fft",
    )


def _scan_peak(
    frequency: float,
    *,
    rate: float = 40.0,
    amplitude_percent: float = 5.0,
    phase_rad: float = 0.3,
    delta_chi_squared: float = 120.0,
    snr: float = 12.0,
) -> DetectedPeak:
    """A damped-scan peak as ``analyze_damped_scan_peaks`` would build it."""
    return DetectedPeak(
        frequency_mhz=frequency,
        amplitude=abs(amplitude_percent),
        snr=snr,
        width_mhz=rate / np.pi,
        prominence=0.0,
        source="damped_scan",
        crop_us=None,
        damping_rate_per_us=rate,
        amplitude_percent=amplitude_percent,
        phase_rad=phase_rad,
        delta_chi_squared=delta_chi_squared,
    )


def _early_peak(frequency: float, snr: float = 8.0, crop_us: float = 0.25) -> DetectedPeak:
    return DetectedPeak(
        frequency_mhz=frequency,
        amplitude=3.0,
        snr=snr,
        width_mhz=4.0,
        prominence=1.0,
        source="early_fft",
        crop_us=crop_us,
    )


def test_merge_lets_the_hann_pass_win_a_collision() -> None:
    """Collision within the coarser resolution: the Hann estimate is kept."""
    hann = _analysis([_hann_peak(60.0)], resolution=0.25)
    # 1/0.25 µs = 4 MHz early resolution, so 62 MHz collides with 60 MHz.
    early = _analysis([_early_peak(62.0)], resolution=4.0)

    merged = merge_early_peaks(hann, early)

    assert len(merged.peaks) == 1
    assert merged.peaks[0].source == "fft"
    assert merged.peaks[0].frequency_mhz == pytest.approx(60.0)


def test_merge_adds_a_line_the_hann_pass_missed() -> None:
    hann = _analysis([_hann_peak(1.4)], resolution=0.25)
    early = _analysis([_early_peak(60.0)], resolution=4.0)

    merged = merge_early_peaks(hann, early)

    assert [peak.source for peak in merged.peaks] == ["fft", "early_fft"]
    # Per-pass order is preserved; the two SNRs are never compared.
    assert merged.peaks[0].snr == 10.0
    assert merged.peaks[1].snr == 8.0


def test_merge_reserves_slots_for_early_additions_under_the_cap() -> None:
    """A full Hann peak set cannot starve the pass that sees what Hann cannot."""
    hann = _analysis([_hann_peak(f, snr=20.0 - f) for f in (1.0, 2.0, 3.0, 4.0)], resolution=0.25)
    early = _analysis([_early_peak(60.0), _early_peak(90.0)], resolution=4.0)

    merged = merge_early_peaks(hann, early, max_peaks=4)

    assert len(merged.peaks) == 4
    sources = [peak.source for peak in merged.peaks]
    assert sources == ["fft", "fft", "early_fft", "early_fft"]
    # The weakest Hann peaks are the ones displaced, never the strongest.
    assert [peak.frequency_mhz for peak in merged.peaks[:2]] == [1.0, 2.0]


def test_merge_is_a_no_op_when_the_early_pass_found_nothing() -> None:
    hann = _analysis([_hann_peak(1.4)], resolution=0.25)

    assert merge_early_peaks(hann, _analysis([])) is hann


def test_user_frequency_collapses_onto_an_early_peak_at_its_own_resolution() -> None:
    """The match radius is the peak's own pass resolution, not the Hann pass's."""
    analysis = _analysis([_early_peak(60.0, crop_us=0.25)], resolution=0.25)

    merged = merge_user_peaks(analysis, [62.0])

    assert len(merged.peaks) == 1
    assert merged.peaks[0].source == "user"
    assert merged.peaks[0].frequency_mhz == pytest.approx(62.0)


def test_detected_peak_round_trip_carries_the_crop() -> None:
    peak = _early_peak(60.0, crop_us=0.248)

    assert deserialize_detected_peak(serialize_detected_peak(peak)) == peak


# --------------------------------------------------------------------------- #
# Workstream B — fingerprint, seeding and cost bounds
# --------------------------------------------------------------------------- #


def test_fingerprint_carries_the_damped_line_and_sets_the_oscillatory_hint() -> None:
    fingerprint = fingerprint_spectrum(_damped_record())

    assert fingerprint.has_damped_line_candidate
    assert fingerprint.damped_line_frequency_mhz == pytest.approx(_LINE_MHZ, rel=0.05)
    assert fingerprint.damped_line_snr > 8.0
    # The scan measures the envelope rather than inferring it from a crop, so
    # the rate is populated and the crop is not: there is no crop.
    assert fingerprint.damped_line_rate_per_us == pytest.approx(_LINE_RATE, rel=0.4)
    assert fingerprint.damped_line_crop_us == 0.0
    assert fingerprint.oscillatory_hint is True
    # The Hann view on its own would have gated precession off. Its "dominant
    # line" is the leftover slow tail, completing well under one cycle in the
    # window and sitting nowhere near the real oscillation.
    assert fingerprint.dominant_fft_cycles_in_window < 1.5
    assert abs(fingerprint.dominant_fft_frequency_mhz - _LINE_MHZ) > 10.0


def test_fingerprint_measures_a_conventional_record_rather_than_ignoring_it() -> None:
    """The scan is not a fast-line detector; it is a damped-line *measurement*.

    A conventional narrow line has a small λ, and the scan reports it as such.
    The crop ladder it replaced could only report a crop, so the fingerprint's
    ``damped_line_*`` fields used to stay empty here; now they carry a rate
    close to the true one and the hint fires from the Hann view *and* the scan.
    """
    fingerprint = fingerprint_spectrum(_narrow_line_record())

    assert fingerprint.has_damped_line_candidate
    assert fingerprint.damped_line_frequency_mhz == pytest.approx(1.4, rel=0.05)
    assert fingerprint.damped_line_rate_per_us == pytest.approx(0.12, rel=0.3)
    assert fingerprint.damped_line_crop_us == 0.0
    assert fingerprint.oscillatory_hint is True


def test_early_peaks_seed_the_multiplet_builder_like_user_frequencies() -> None:
    analysis = _analysis([_early_peak(60.0), _early_peak(30.0)], resolution=4.0)

    templates = build_oscillatory_multiplet_templates(analysis)

    # An early-window peak carries no measured envelope, so it builds only the
    # historical multiplet shapes — not the relaxing twins, whose extra
    # exponential needs a pinned envelope to be separable from the cosines'.
    assert [template.key for template in templates] == [
        "oscillatory2_exp_constant",
        "oscillatory2_gaussian_constant",
    ]


def test_multiplet_seeds_are_taken_in_the_order_the_passes_set() -> None:
    # The peak tuple's order is per-pass by construction (user first, then each
    # pass's block in its own SNR order); seeding takes a prefix of it rather
    # than re-ranking across passes.
    analysis = _analysis(
        [_hann_peak(1.4, snr=40.0), _early_peak(90.0, snr=12.0), _early_peak(60.0, snr=11.0)]
        + [_early_peak(30.0, snr=10.0)],
        resolution=4.0,
    )

    seeds = _multiplet_seed_peaks(analysis, 3)

    assert [peak.frequency_mhz for peak in seeds] == [1.4, 90.0, 60.0]


def test_multiplet_order_is_bounded_regardless_of_the_seed_count() -> None:
    analysis = _analysis(
        [_early_peak(float(10 * k), snr=20.0 - k) for k in range(2, 8)], resolution=4.0
    )

    templates = build_oscillatory_multiplet_templates(analysis, max_components=6)

    assert {template.key for template in templates} == {
        "oscillatory3_exp_constant",
        "oscillatory3_gaussian_constant",
    }


def test_a_weak_hann_peak_is_still_rejected_as_a_multiplet_seed() -> None:
    """The early-pass exemption must not silently relax the Hann-pass gate."""
    analysis = _analysis([_hann_peak(1.4, snr=20.0), _hann_peak(2.8, snr=1.0)])

    assert [peak.frequency_mhz for peak in _multiplet_seed_peaks(analysis, 3)] == [1.4]


def test_damped_envelope_rate_prefers_the_measured_rate_over_the_crop() -> None:
    # A measured rate wins outright, even against a crop that would imply
    # something else: one is a fit, the other a factor-of-two heuristic.
    measured = replace(_early_peak(60.0, crop_us=0.25), damping_rate_per_us=37.5)
    assert _damped_envelope_rate(measured) == pytest.approx(37.5)
    assert _damped_envelope_rate(_scan_peak(60.0, rate=42.0)) == pytest.approx(42.0)
    # ...and without one the crop heuristic still answers.
    assert _damped_envelope_rate(_early_peak(60.0, crop_us=0.25)) == pytest.approx(12.0)
    assert _damped_envelope_rate(_hann_peak(1.4)) is None
    assert _damped_envelope_rate(None) is None


def test_damped_seed_puts_the_true_envelope_inside_the_fit_bounds() -> None:
    """Without the crop-derived rate the true envelope is outside the bounds.

    ``lambda_guess`` is a slope over the leading 5 % of the record, so on a
    heavily damped line it measures the slow tail; ``_parameter_bounds`` then
    caps ``Lambda`` at 8x that guess. This is the check that the damped seed
    moves both the value and the ceiling.
    """
    dataset = _damped_record()
    fingerprint = fingerprint_spectrum(dataset)
    template = CandidateTemplate(
        key="oscillatory_exp_constant",
        title="",
        category="Oscillatory",
        rationale="",
        model=CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"]),
    )
    context = TemplateSeedContext(peak_analysis=analyze_dataset_peaks(dataset), field_gauss=None)

    seeded = _initial_parameters_for_template(dataset, fingerprint, template, seed_context=context)
    unseeded = _initial_parameters_for_template(dataset, fingerprint, template)

    assert unseeded["Lambda"].max < _LINE_RATE, (
        "precondition: the blind bounds must exclude the true envelope rate"
    )
    assert seeded["Lambda"].max > _LINE_RATE
    assert seeded["Lambda"].min <= _LINE_RATE <= seeded["Lambda"].max
    # ...and the frequency seed is the measured line, bounded around it.
    assert seeded["frequency"].value == pytest.approx(_LINE_MHZ, rel=0.05)
    assert seeded["frequency"].min < _LINE_MHZ < seeded["frequency"].max


# --------------------------------------------------------------------------- #
# Workstream C — the blind end-to-end acceptance tests
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_blind_wizard_ranks_a_damped_oscillation_top() -> None:
    """The headline: no ``user_frequencies_mhz``, damped cosine still wins.

    On ``main`` this record's recommendation was a static Kubo-Toyabe — the
    oscillation was invisible to both seeding FFTs, so no oscillatory candidate
    was ever seeded near it.

    The relaxing twin is offered here too, but this record is only 4 µs long and
    its background relaxes at ~0.25 µs⁻¹ — a 1/e time longer than the window —
    so the component-resolution rule disqualifies the extra exponential and the
    plain damped cosine wins, exactly as before.
    """
    dataset = _damped_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.recommended_key == "oscillatory_exp_constant"
    winner = recommendation.recommended_assessment
    assert winner is not None
    fitted = winner.fit_result.parameters["frequency"].value
    # Seeded, and then fitted, well inside the measured line's own width.
    assert abs(fitted - _LINE_MHZ) <= _LINE_RATE / math.pi
    assert winner.fit_result.parameters["Lambda"].value == pytest.approx(_LINE_RATE, rel=0.3)


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_blind_wizard_ranks_a_two_oscillation_family_top() -> None:
    dataset = _two_line_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.peak_analysis is not None
    scan = [peak for peak in recommendation.peak_analysis.peaks if peak.source == "damped_scan"]
    assert len(scan) == 2, "both damped lines must reach the seeding path"

    assert recommendation.recommended_key == "oscillatory2_exp_constant"
    winner = recommendation.recommended_assessment
    assert winner is not None
    fitted = sorted(
        parameter.value
        for parameter in winner.fit_result.parameters
        if parameter.name.startswith("frequency")
    )
    assert fitted[0] == pytest.approx(_SECOND_LINE_MHZ, rel=0.05)
    assert fitted[1] == pytest.approx(_LINE_MHZ, rel=0.05)


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_conventional_record_recommendation_is_unchanged() -> None:
    """Regression: a narrow line alive across the record is unaffected.

    The early pass contributes no peaks here, so the recommendation, the peak
    set and the fingerprint's damped-line fields must all read exactly as they
    did before the pass existed.
    """
    dataset = _narrow_line_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.recommended_key == "oscillatory_exp_constant"
    assert recommendation.peak_analysis is not None
    assert not any(
        peak.source in ("early_fft", "damped_scan") for peak in recommendation.peak_analysis.peaks
    )
    # The record is short enough to fit whole: no rebinning, so the information
    # criteria refer to every point of it.
    assert recommendation.rebin_factor == 1
    assert recommendation.analysed_points == dataset.n_points
    winner = recommendation.recommended_assessment
    assert winner is not None
    assert winner.fit_result.parameters["frequency"].value == pytest.approx(1.4, rel=0.02)


# --------------------------------------------------------------------------- #
# Workstream D — the matched-apodisation scan end to end
# --------------------------------------------------------------------------- #
#
# The records below are the ones the scan module itself is measured on: 0.1 ns
# binning, µSR-like errors growing as exp(t/4.4 µs) and saturating at 25 %, and
# oscillations that are gone inside the first ~50 ns. Every value invented. They
# are the regime the whole feature exists for — the historical crop ladder found
# one of the two lines at SNR 6.4 against a gate of 6.0, and no candidate in the
# portfolio could fit either of them.

#: The two invented scan-regime lines: (amplitude %, MHz, λ µs⁻¹, phase rad).
_SCAN_LINE_A = (4.7, 240.0, 44.0, 0.3)
_SCAN_LINE_B = (1.9, 120.0, 22.0, -0.7)
#: One very fast line: 8 % at 200 MHz, gone within ~30 ns.
_SCAN_FAST_LINE = (8.0, 200.0, 100.0, 0.0)

#: Per-bin σ (percent) a 0.1 ns-binned record actually carries at t = 0.
_SCAN_SIGMA0 = 3.3
#: 60 000 bins of 0.1 ns is a 6 µs record — long enough for the slow background
#: to be resolvable and short enough to keep the end-to-end fits affordable.
_SCAN_N_POINTS = 60_000


def _scan_record(
    lines: tuple[tuple[float, float, float, float], ...],
    *,
    seed: int = 39,
    n_points: int = _SCAN_N_POINTS,
    relaxation: tuple[float, float] = (2.6, 0.19),
    baseline: float = 4.6,
) -> MuonDataset:
    """Heavily damped lines over a slow relaxing background. All values invented."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=float) * 1e-4
    error = np.minimum(_SCAN_SIGMA0 * np.exp(time / 4.4), 25.0)
    tail_amplitude, tail_rate = relaxation
    signal = tail_amplitude * np.exp(-tail_rate * time) + baseline
    for amplitude, frequency, rate, phase in lines:
        signal = signal + amplitude * np.exp(-rate * time) * np.cos(
            2.0 * np.pi * frequency * time + phase
        )
    return MuonDataset(
        time=time,
        asymmetry=signal + rng.normal(0.0, error),
        error=error,
        metadata={"run_number": 11, "field": 0.0, "field_direction": "ZF"},
    )


def _fitted_pairs(assessment) -> list[tuple[float, float]]:
    """(frequency, envelope rate) of each damped cosine, frequency-ascending."""
    values = {p.name: p.value for p in assessment.fit_result.parameters}
    pairs: list[tuple[float, float]] = []
    for name, value in values.items():
        base, index = split_parameter_name(name)
        if base != "frequency":
            continue
        # The envelope sits on the component right after the oscillator. A lone
        # oscillator names its own parameters without an index (``frequency``,
        # not ``frequency_1``), and so may its envelope — but ``Lambda`` on a
        # Gaussian-envelope model is the RELAXATION term, so the indexed names
        # and ``sigma`` are tried first.
        oscillator = 1 if index is None else int(index)
        for key in (
            f"sigma_{oscillator + 1}",
            f"Lambda_{oscillator + 1}",
            "sigma",
            "Lambda",
        ):
            if key in values:
                pairs.append((float(value), float(values[key])))
                break
    return sorted(pairs)


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_blind_wizard_ranks_a_relaxing_two_line_model_on_the_scan_record() -> None:
    """The Phase-2 headline: two heavily damped lines *and* the tail they leave.

    No candidate in the portfolio had this shape before — a single-envelope
    oscillatory candidate must either fit a 24 ns line and leave the 5 µs
    relaxation in the residual, or fit the relaxation and lose the line — so the
    wizard recommended ``Exponential + Constant`` at High confidence.
    """
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B))

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key in {
        "oscillatory2_exp_relax_constant",
        "oscillatory2_gaussian_relax_constant",
    }
    winner = recommendation.recommended_assessment
    assert winner is not None
    pairs = _fitted_pairs(winner)
    assert len(pairs) == 2
    for (frequency, rate), (_amplitude, true_frequency, true_rate, _phase) in zip(
        pairs, (_SCAN_LINE_B, _SCAN_LINE_A)
    ):
        assert frequency == pytest.approx(true_frequency, rel=0.02)
        assert rate == pytest.approx(true_rate, rel=0.4)
    # Fitted on a rebinned copy — the record is 60 000 points and no model here
    # needs more than a few samples per cycle of its fastest line.
    assert recommendation.rebin_factor > 1
    assert recommendation.analysed_points == dataset.n_points // recommendation.rebin_factor


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_blind_wizard_ranks_a_one_line_relax_model_on_a_single_fast_line() -> None:
    """A lone line is enough: one measured envelope opens the ``n = 1`` shape.

    The historical multiplet builder refused below two peaks, which is right for
    peaks that carry no envelope — the plain oscillatory candidates are the same
    model. It is wrong once the line is measured *and* the record has a tail the
    line does not explain.
    """
    dataset = _scan_record((_SCAN_FAST_LINE,), seed=5, relaxation=(6.0, 0.4), baseline=0.2)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key in {
        "oscillatory1_exp_relax_constant",
        "oscillatory1_gaussian_relax_constant",
    }
    winner = recommendation.recommended_assessment
    assert winner is not None
    ((frequency, rate),) = _fitted_pairs(winner)
    assert frequency == pytest.approx(_SCAN_FAST_LINE[1], rel=0.02)
    assert rate == pytest.approx(_SCAN_FAST_LINE[2], rel=0.4)


def test_scan_peaks_seed_amplitude_phase_and_envelope_per_component() -> None:
    """Every measured quantity reaches the seed, per damped cosine."""
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B))
    fingerprint = fingerprint_spectrum(dataset)
    analysis = analyze_dataset_peaks(dataset)
    scan = [peak for peak in analysis.peaks if peak.source == "damped_scan"]
    assert len(scan) == 2
    templates = {t.key: t for t in build_oscillatory_multiplet_templates(analysis)}
    template = templates["oscillatory2_exp_relax_constant"]

    seeded = _initial_parameters_for_template(
        dataset,
        fingerprint,
        template,
        seed_context=TemplateSeedContext(peak_analysis=analysis, field_gauss=None),
    )

    for k, peak in enumerate(scan):
        osc, env = 2 * k + 1, 2 * k + 2
        assert seeded[f"A_{osc}"].value == pytest.approx(peak.amplitude_percent, rel=1e-6)
        assert seeded[f"phase_{osc}"].value == pytest.approx(peak.phase_rad, rel=1e-6)
        assert seeded[f"frequency_{osc}"].value == pytest.approx(peak.frequency_mhz)
        assert seeded[f"Lambda_{env}"].value == pytest.approx(peak.damping_rate_per_us)
        # The measured λ is a seed, not a certainty: the bounds must always
        # contain a factor of four either way (see ``_parameter_bounds``).
        assert seeded[f"Lambda_{env}"].min <= peak.damping_rate_per_us / 4.0
        assert seeded[f"Lambda_{env}"].max >= 4.0 * peak.damping_rate_per_us
        # ...and the frequency box is a few of the line's own widths, not a
        # quarter of the frequency, which would reach the neighbouring line.
        half_width = max(3.0 * peak.width_mhz, 0.05 * peak.frequency_mhz)
        assert seeded[f"frequency_{osc}"].min == pytest.approx(
            peak.frequency_mhz - half_width, rel=1e-6
        )
    # The extra relaxation term is seeded from the fingerprint's tail guesses —
    # the measurement that is useless for the lines and right for the tail.
    # (``lambda_guess`` is a slope over the leading 5 % of the record, which on
    # a record like this one still contains the tail of the damped lines, so it
    # reads a few times the true 0.19 µs⁻¹ — a seed, and one whose bounds
    # comfortably contain the truth, which is all it has to be.)
    assert seeded["A_5"].value > 0.0
    assert 0.0 < seeded["Lambda_5"].value < 20.0
    assert seeded["Lambda_5"].min <= 0.19 <= seeded["Lambda_5"].max
    assert seeded["A_bg"].value == pytest.approx(fingerprint.tail_estimate)


def test_relax_templates_need_a_measured_envelope_but_only_one() -> None:
    scan_only = _analysis([_scan_peak(240.0)], resolution=0.1)
    keys = {template.key for template in build_oscillatory_multiplet_templates(scan_only)}
    assert keys == {"oscillatory1_exp_relax_constant", "oscillatory1_gaussian_relax_constant"}

    # A Hann peak with no measured envelope contributes to the plain multiplet
    # shapes but is not counted by the relaxing ones: their extra exponential is
    # separable from a cosine's envelope only when that envelope is pinned.
    mixed = _analysis([_hann_peak(1.4, snr=40.0), _scan_peak(240.0)], resolution=0.1)
    keys = {template.key for template in build_oscillatory_multiplet_templates(mixed)}
    assert keys == {
        "oscillatory2_exp_constant",
        "oscillatory2_gaussian_constant",
        "oscillatory1_exp_relax_constant",
        "oscillatory1_gaussian_relax_constant",
    }

    # Nothing measured at all: exactly the historical behaviour.
    plain = _analysis([_hann_peak(1.4, snr=40.0)], resolution=0.1)
    assert build_oscillatory_multiplet_templates(plain) == ()


def test_relax_template_carries_the_relaxation_and_the_lines() -> None:
    templates = {
        template.key: template
        for template in build_oscillatory_multiplet_templates(
            _analysis([_scan_peak(240.0), _scan_peak(120.0)], resolution=0.1)
        )
    }
    exp_template = templates["oscillatory2_exp_relax_constant"]
    gaussian_template = templates["oscillatory2_gaussian_relax_constant"]

    # Both shapes are offered at this order: the plain multiplet and the twin
    # that carries the tail.
    assert set(templates) == {
        "oscillatory2_exp_constant",
        "oscillatory2_gaussian_constant",
        "oscillatory2_exp_relax_constant",
        "oscillatory2_gaussian_relax_constant",
    }
    assert exp_template.model.component_names == [
        "Oscillatory",
        "Exponential",
        "Oscillatory",
        "Exponential",
        "Exponential",
        "Constant",
    ]
    assert gaussian_template.model.component_names == [
        "Oscillatory",
        "Gaussian",
        "Oscillatory",
        "Gaussian",
        "Exponential",
        "Constant",
    ]
    assert "240" in exp_template.rationale and "120" in exp_template.rationale
    assert "damped cosine" in exp_template.title and "relaxation" in exp_template.title
