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

import asymmetry.core.fitting.fit_wizard as fit_wizard_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    _CALL_LIMIT_CONTINUATIONS,
    CandidateAssessment,
    CandidateTemplate,
    SelectionMetric,
    SpectrumFingerprint,
    TemplateSeedContext,
    _assess_candidate_template,
    _damped_envelope_rate,
    _fit_with_call_limit_continuations,
    _initial_parameters_for_template,
    _multiplet_model,
    _multiplet_seed_peaks,
    build_fit_wizard_recommendation,
    build_null_baseline_templates,
    build_oscillatory_multiplet_templates,
    fingerprint_spectrum,
    fit_result_is_oscillatory_admissible,
    is_oscillatory_admissible,
    oscillatory_line_amplitude_names,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, split_parameter_name
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

    The winner is the relaxing twin, which is this record's *true* shape: a
    damped cosine, a slow relaxation and a baseline.  It used to lose, not
    because the tail is not there but because the extra exponential was seeded
    at a quarter of the data span with no floor on its rate, ran away into the
    ``A·e^{-λt} + A_bg`` degeneracy and was disqualified as unresolved.  Seeded
    from the early-minus-tail step and bounded away from that degeneracy at
    ``1/T`` it recovers every generated value (below) and beats the plain damped
    cosine by ~725 AICc.
    """
    dataset = _damped_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.recommended_key == "oscillatory1_exp_relax_constant"
    winner = recommendation.recommended_assessment
    assert winner is not None
    values = {parameter.name: parameter.value for parameter in winner.fit_result.parameters}
    # Seeded, and then fitted, well inside the measured line's own width.
    assert abs(values["frequency"] - _LINE_MHZ) <= _LINE_RATE / math.pi
    assert values["Lambda_2"] == pytest.approx(_LINE_RATE, rel=0.3)
    # ...and the background the line outlives, which is what the relaxing twin
    # exists to carry.
    assert values["A_3"] == pytest.approx(_TAIL_AMPLITUDE, rel=0.15)
    assert values["Lambda_3"] == pytest.approx(_TAIL_RATE, rel=0.15)
    assert values["A_bg"] == pytest.approx(_BASELINE, abs=0.5)


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_blind_wizard_ranks_a_two_oscillation_family_top() -> None:
    """Two damped lines and the tail they outlive — the record's true shape.

    The one-line prefix of the same relaxing shape is in the portfolio too
    (every prefix is built), and is beaten here by ~3700 AICc: the second line
    is real, and the wizard is asked to choose the order rather than to inherit
    it from the detector.
    """
    dataset = _two_line_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.peak_analysis is not None
    scan = [peak for peak in recommendation.peak_analysis.peaks if peak.source == "damped_scan"]
    assert len(scan) == 2, "both damped lines must reach the seeding path"

    assert recommendation.recommended_key == "oscillatory2_exp_relax_constant"
    winner = recommendation.recommended_assessment
    assert winner is not None
    fitted = sorted(
        parameter.value
        for parameter in winner.fit_result.parameters
        if parameter.name.startswith("frequency")
    )
    assert fitted[0] == pytest.approx(_SECOND_LINE_MHZ, rel=0.05)
    assert fitted[1] == pytest.approx(_LINE_MHZ, rel=0.05)
    values = {parameter.name: parameter.value for parameter in winner.fit_result.parameters}
    assert values["A_5"] == pytest.approx(_TAIL_AMPLITUDE, rel=0.15)
    assert values["Lambda_5"] == pytest.approx(_TAIL_RATE, rel=0.15)
    by_key = {assessment.template.key: assessment for assessment in recommendation.assessments}
    one_line = by_key["oscillatory1_exp_relax_constant"]
    assert one_line.aicc is not None and winner.aicc is not None
    assert one_line.aicc > winner.aicc + 100.0


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
    # Each (Osc x Env) pair carries exactly one amplitude by construction: the
    # envelope factors (index 2, 4) never get their own A, so there is nothing
    # left to pin at 1 and fix.
    assert "A_2" not in template.model.param_names
    assert "A_4" not in template.model.param_names
    assert not any(parameter.fixed for parameter in seeded if parameter.name.startswith("A"))


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

    # Both shapes are offered at the full order — the plain multiplet and the
    # twin that carries the tail — and the relaxing twin additionally at every
    # shorter prefix, so "one line plus a tail" is a candidate the wizard can
    # choose over "two lines plus a tail" on the metric.
    assert set(templates) == {
        "oscillatory2_exp_constant",
        "oscillatory2_gaussian_constant",
        "oscillatory1_exp_relax_constant",
        "oscillatory1_gaussian_relax_constant",
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


# --------------------------------------------------------------------------- #
# Workstream E — the relaxing background of a multiplet
# --------------------------------------------------------------------------- #
#
# ``Σ(Osc × Env) + Exp + Const`` carries one term that none of the detection
# passes measures: the slow background the damped lines outlive.  Seeded and
# bounded badly it is not merely imprecise — ``A·e^{-λt} + A_bg`` with
# ``λ·T ≲ 1`` is a one-parameter family with two free scales, and migrad walks
# it until it runs out of calls.  Every value below is invented.


def _flat_background_scan_record() -> MuonDataset:
    """One damped line on a *constant* background, so ``lambda_guess`` is tiny."""
    return _scan_record((_SCAN_FAST_LINE,), seed=17, relaxation=(0.0, 0.0), baseline=4.6)


def _relax_seed(dataset: MuonDataset) -> tuple[ParameterSet, SpectrumFingerprint, str]:
    """Seed the relaxing multiplet template this record supports."""
    analysis = analyze_dataset_peaks(dataset)
    fingerprint = fingerprint_spectrum(dataset, peak_analysis=analysis)
    templates = {t.key: t for t in build_oscillatory_multiplet_templates(analysis)}
    key = max(k for k in templates if k.endswith("_exp_relax_constant"))
    seeded = _initial_parameters_for_template(
        dataset,
        fingerprint,
        templates[key],
        seed_context=TemplateSeedContext(peak_analysis=analysis, field_gauss=None),
    )
    return seeded, fingerprint, key


def test_relax_background_rate_is_bounded_away_from_the_constant() -> None:
    """A 1/e time longer than the window is not a rate, it is the baseline.

    ``1/T`` is exactly the ``slow_edge`` the component-resolution assessment
    uses, so the bound says the same thing the diagnostic does — a parameter
    that means nothing below it may not be searched below it.
    """
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B))
    duration = float(dataset.time[-1] - dataset.time[0])
    seeded, _fingerprint, key = _relax_seed(dataset)
    rate = seeded[f"Lambda_{2 * int(key[len('oscillatory')]) + 1}"]

    assert rate.min == pytest.approx(1.0 / duration)
    assert rate.value >= 2.0 / duration
    # ...and the interval is still wide enough to be a search space, not a pin.
    assert rate.max >= 4.0 * rate.value


def test_relax_background_rate_seed_clears_the_degeneracy_even_on_a_flat_tail() -> None:
    """With no relaxation to measure, the seed is two windows, not the 0.05 floor."""
    dataset = _flat_background_scan_record()
    duration = float(dataset.time[-1] - dataset.time[0])
    seeded, _fingerprint, _key = _relax_seed(dataset)

    assert seeded["Lambda_3"].value == pytest.approx(2.0 / duration)
    assert seeded["Lambda_3"].min == pytest.approx(1.0 / duration)


def test_relax_background_amplitude_is_the_early_minus_tail_step() -> None:
    """Not a quarter of the data span, which on a noisy record is the noise.

    The span of a 0.1 ns-binned record is set by its largest noise excursion —
    here ~8×  the relaxation it was standing in for — so the old floor seeded
    the background an order of magnitude high and handed migrad a start deep
    inside the degeneracy.
    """
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B))
    seeded, fingerprint, _key = _relax_seed(dataset)
    span = float(np.max(dataset.asymmetry) - np.min(dataset.asymmetry))

    assert seeded["A_5"].value == pytest.approx(abs(fingerprint.initial_amplitude_estimate))
    assert seeded["A_bg"].value == pytest.approx(fingerprint.tail_estimate)
    assert seeded["A_5"].value < 0.25 * span


class _ScriptedEngine:
    """A ``FitEngine`` stand-in that returns queued results and records its inputs."""

    def __init__(self, results: list[FitResult]) -> None:
        self._results = list(results)
        self.seen: list[ParameterSet] = []

    def fit(self, _dataset, _model_fn, parameters, **_kwargs) -> FitResult:
        self.seen.append(parameters)
        return self._results.pop(0)


def _call_limited(chi_squared: float, **values: float) -> FitResult:
    return FitResult(
        success=False,
        chi_squared=chi_squared,
        parameters=ParameterSet([Parameter(name=n, value=v) for n, v in values.items()]),
        message="Fit failed: call limit reached, invalid parameters, minimum invalid",
    )


def _relax_template() -> CandidateTemplate:
    templates = {
        t.key: t for t in build_oscillatory_multiplet_templates(_analysis([_scan_peak(240.0)]))
    }
    return templates["oscillatory1_exp_relax_constant"]


def _relax_seed_parameters(template: CandidateTemplate) -> ParameterSet:
    return ParameterSet(
        [
            Parameter(name="A_1", value=5.0, min=0.0, max=100.0),
            Parameter(name="frequency", value=240.0, min=200.0, max=280.0),
            Parameter(name="phase", value=0.0, min=-math.pi, max=math.pi),
            Parameter(name="Lambda_2", value=40.0, min=0.0, max=320.0, fixed=True),
            Parameter(name="A_3", value=2.0, min=0.0, max=100.0),
            Parameter(name="Lambda_3", value=0.33, min=0.166, max=5.0),
            Parameter(name="A_bg", value=4.6, min=-50.0, max=50.0),
        ]
    )


def test_a_call_limited_fit_is_restarted_from_the_parameters_it_returned() -> None:
    """A call limit is an unfinished fit, not a failed one."""
    template = _relax_template()
    seed = _relax_seed_parameters(template)
    stopped = _call_limited(2000.0, A_3=15.5, Lambda_3=0.019, A_bg=-8.4)
    converged = FitResult(success=True, chi_squared=900.0, parameters=ParameterSet())
    engine = _ScriptedEngine([converged])

    final = _fit_with_call_limit_continuations(
        engine,
        _scan_record((_SCAN_LINE_A,)),
        template,
        seed,
        stopped,
        cancel_callback=None,
    )

    assert final is converged
    assert len(engine.seen) == 1
    restarted = engine.seen[0]
    # Values from where migrad stopped...
    assert restarted["A_3"].value == pytest.approx(15.5)
    # ...clipped back inside the seed's bounds, which the restart keeps...
    assert restarted["Lambda_3"].value == pytest.approx(0.166)
    assert restarted["Lambda_3"].min == pytest.approx(0.166)
    assert restarted["A_bg"].value == pytest.approx(-8.4)
    # ...as it keeps the fixed flags the engine drops when it packs a result.
    assert restarted["Lambda_2"].fixed


def test_a_fit_that_stays_call_limited_gives_up_and_stays_unsuccessful() -> None:
    """Two continuations, then the verdict stands."""
    template = _relax_template()
    seed = _relax_seed_parameters(template)
    stops = [_call_limited(2000.0 - 10.0 * k, A_3=15.5, Lambda_3=0.019) for k in range(4)]
    engine = _ScriptedEngine(stops[1:])

    final = _fit_with_call_limit_continuations(
        engine,
        _scan_record((_SCAN_LINE_A,)),
        template,
        seed,
        stops[0],
        cancel_callback=None,
    )

    assert not final.success
    assert len(engine.seen) == _CALL_LIMIT_CONTINUATIONS


def test_a_diverged_fit_is_not_continued() -> None:
    """Non-finite parameters are nothing to restart from."""
    engine = _ScriptedEngine([])
    diverged = _call_limited(float("inf"), A_3=float("nan"))

    final = _fit_with_call_limit_continuations(
        engine,
        _scan_record((_SCAN_LINE_A,)),
        _relax_template(),
        _relax_seed_parameters(_relax_template()),
        diverged,
        cancel_callback=None,
    )

    assert final is diverged
    assert engine.seen == []


def test_screening_fits_are_never_continued(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Stage-1 cap is deliberate; continuing past it would undo it."""
    calls: list[int] = []
    monkeypatch.setattr(
        fit_wizard_module,
        "_fit_with_call_limit_continuations",
        lambda *args, **kwargs: (calls.append(1), args[4])[1],
    )
    dataset = _scan_record((_SCAN_LINE_A,))
    fingerprint = fingerprint_spectrum(dataset)
    template = build_null_baseline_templates()[1]

    _assess_candidate_template(
        dataset.rebin(20),
        fingerprint,
        template,
        fit_engine=FitEngine(),
        metric=SelectionMetric.AICC,
        variant_budget=1,
        migrad_ncall=3000,
    )
    assert calls == []

    _assess_candidate_template(
        dataset.rebin(20),
        fingerprint,
        template,
        fit_engine=FitEngine(),
        metric=SelectionMetric.AICC,
        variant_budget=1,
        migrad_ncall=None,
    )
    assert calls == [1]


def _ranked_assessment(key: str, aicc: float, *, success: bool) -> CandidateAssessment:
    """A candidate carrying a finite metric whose fit did or did not converge."""
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=CandidateTemplate(
            key=key,
            title=key,
            category="Oscillatory",
            rationale="",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        fit_result=FitResult(
            success=success,
            chi_squared=aicc,
            parameters=ParameterSet([Parameter(name="Lambda", value=1.0)]),
            message="" if success else "Fit failed: call limit reached",
        ),
        aic=aicc,
        aicc=aicc,
        bic=aicc,
        selected_score=aicc,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=success,
        residual_gate_reasons=() if success else ("Fit failed: call limit reached",),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
    )


def test_refinement_targets_the_metric_leader_even_when_its_fit_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The candidate a failed fit hides is exactly the one worth searching harder.

    Ranking the refinement pass on ``is_successful`` first skipped the leader
    and left the recommendation to a model it beat by 133 AICc.
    """
    leader = _ranked_assessment("oscillatory2_exp_relax_constant", 100.0, success=False)
    runner_up = _ranked_assessment("exp_constant", 233.0, success=True)
    refined = replace(
        leader,
        fit_result=FitResult(
            success=True,
            chi_squared=90.0,
            parameters=leader.fit_result.parameters,
            message="Fit successful",
        ),
    )

    requested: list[str] = []

    def _fake_run(tasks, **_kwargs):
        requested.extend(task.template.key for task in tasks)
        return (refined,)

    monkeypatch.setattr(fit_wizard_module, "_run_template_assessments", _fake_run)

    updated = fit_wizard_module._refine_top_candidates(
        dataset=_scan_record((_SCAN_LINE_A,)),
        fingerprint=fingerprint_spectrum(_scan_record((_SCAN_LINE_A,))),
        assessments=(leader, runner_up),
        metric=SelectionMetric.AICC,
        seed_context=TemplateSeedContext(),
        max_workers=1,
        cancel_callback=None,
        refine_top_candidates=1,
        progress=lambda _message: None,
    )

    assert requested == ["oscillatory2_exp_relax_constant"]
    # The refined fit converged, so it replaces the failed assessment.
    winner = next(a for a in updated if a.template.key == "oscillatory2_exp_relax_constant")
    assert winner.is_successful
    assert winner.refinement_delta_chi_squared == pytest.approx(10.0)


def test_a_refined_fit_that_still_fails_leaves_the_original_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = _ranked_assessment("oscillatory2_exp_relax_constant", 100.0, success=False)
    monkeypatch.setattr(
        fit_wizard_module,
        "_run_template_assessments",
        lambda tasks, **_kwargs: (replace(leader, aicc=95.0),),
    )

    (updated,) = fit_wizard_module._refine_top_candidates(
        dataset=_scan_record((_SCAN_LINE_A,)),
        fingerprint=fingerprint_spectrum(_scan_record((_SCAN_LINE_A,))),
        assessments=(leader,),
        metric=SelectionMetric.AICC,
        seed_context=TemplateSeedContext(),
        max_workers=1,
        cancel_callback=None,
        refine_top_candidates=1,
        progress=lambda _message: None,
    )

    assert updated is leader


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_blind_wizard_recovers_a_barely_resolvable_background_under_two_lines() -> None:
    """The 9 µs case: the tail relaxes only ~1.7 times inside its own window.

    ``λ·T ≈ 1.7`` is resolvable but close enough to the ``A·e^{-λt} + A_bg``
    degeneracy that an unfloored search runs away into it, taking the
    best-scoring model in the portfolio down with it.  Blind, on 90 000 points,
    the wizard must still recommend the relaxing two-line shape with a fit that
    converged.
    """
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B), n_points=90_000)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key in {
        "oscillatory2_exp_relax_constant",
        "oscillatory2_gaussian_relax_constant",
    }
    winner = recommendation.recommended_assessment
    assert winner is not None
    assert winner.is_successful
    pairs = _fitted_pairs(winner)
    assert len(pairs) == 2
    for (frequency, rate), (_amplitude, true_frequency, true_rate, _phase) in zip(
        pairs, (_SCAN_LINE_B, _SCAN_LINE_A)
    ):
        assert frequency == pytest.approx(true_frequency, rel=0.02)
        assert rate == pytest.approx(true_rate, rel=0.4)
    # The background is recovered too, and it is not sitting on its own floor.
    values = {parameter.name: parameter.value for parameter in winner.fit_result.parameters}
    background_rate = values.get("Lambda_5", values.get("Lambda"))
    assert background_rate == pytest.approx(0.19, rel=0.4)
    assert background_rate > 1.5 / float(dataset.time[-1] - dataset.time[0])


def test_relax_templates_are_built_for_every_prefix_of_the_measured_lines() -> None:
    """Which k the data support is the wizard's question, not the detector's.

    The seed order is Δχ²-descending, so the prefixes are nested "strongest k
    lines" hypotheses and the metric chooses between them.  The non-relaxing
    shape keeps its single full-width form: below two lines it is the plain
    oscillatory candidate the portfolio already carries.
    """
    analysis = _analysis(
        [
            _scan_peak(240.0, delta_chi_squared=300.0),
            _scan_peak(120.0, delta_chi_squared=200.0),
            _scan_peak(60.0, delta_chi_squared=100.0),
        ],
        resolution=0.1,
    )

    templates = {t.key: t for t in build_oscillatory_multiplet_templates(analysis)}

    for order in (1, 2, 3):
        assert f"oscillatory{order}_exp_relax_constant" in templates
        assert f"oscillatory{order}_gaussian_relax_constant" in templates
    assert "oscillatory4_exp_relax_constant" not in templates
    assert {"oscillatory1_exp_constant", "oscillatory2_exp_constant"}.isdisjoint(templates)
    assert "oscillatory3_exp_constant" in templates
    # The k = 2 shape carries the two strongest lines, not an arbitrary pair.
    rationale = templates["oscillatory2_exp_relax_constant"].rationale
    assert "240" in rationale and "120" in rationale


def test_a_prefix_template_is_seeded_from_the_same_prefix() -> None:
    """The builder's ``n`` and the seeder's ``n`` must select the same lines."""
    analysis = _analysis(
        [_scan_peak(240.0, delta_chi_squared=300.0), _scan_peak(120.0, delta_chi_squared=200.0)],
        resolution=0.1,
    )
    templates = {t.key: t for t in build_oscillatory_multiplet_templates(analysis)}
    dataset = _scan_record((_SCAN_LINE_A, _SCAN_LINE_B))

    seeded = _initial_parameters_for_template(
        dataset,
        fingerprint_spectrum(dataset),
        templates["oscillatory1_exp_relax_constant"],
        seed_context=TemplateSeedContext(peak_analysis=analysis, field_gauss=None),
    )

    assert seeded["frequency"].value == pytest.approx(240.0)


# --------------------------------------------------------------------------- #
# Which parameters are the lines, and when a line has vanished
# --------------------------------------------------------------------------- #


def _multiplet_template(n: int, envelope: str, *, relax: bool) -> CandidateTemplate:
    """A multiplet template of exactly one shape, built the way the wizard does."""
    return CandidateTemplate(
        key=f"oscillatory{n}_{'exp' if envelope == 'Exponential' else 'gaussian'}"
        f"{'_relax' if relax else ''}_constant",
        title="test multiplet",
        category="Oscillatory",
        rationale="test",
        model=_multiplet_model(n, envelope, relax=relax),
    )


@pytest.mark.parametrize("envelope", ["Exponential", "Gaussian"])
@pytest.mark.parametrize("relax", [False, True])
@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, ("A_1",)), (2, ("A_1", "A_3")), (3, ("A_1", "A_3", "A_5"))],
)
def test_the_line_amplitudes_are_the_products_that_hold_an_oscillation(
    n: int, expected: tuple[str, ...], envelope: str, relax: bool
) -> None:
    """One amplitude per ``Osc × Env`` product, whatever the envelope or order.

    The relaxation term is a bare leaf rather than a product, so its own
    amplitude must never be reported as a line: that is the case the structural
    derivation exists for, since by name alone it is indistinguishable.
    """
    template = _multiplet_template(n, envelope, relax=relax)

    assert oscillatory_line_amplitude_names(template) == expected


def test_the_relaxation_amplitude_is_a_parameter_but_not_a_line() -> None:
    """Pins the trap: the relaxing shape's extra ``A`` is real, and is not a line."""
    template = _multiplet_template(2, "Exponential", relax=True)

    assert "A_5" in template.model.param_names
    assert "A_5" not in oscillatory_line_amplitude_names(template)


def _amplitude_fit_result(values: dict[str, float], uncertainties: dict[str, float]) -> FitResult:
    parameters = ParameterSet([Parameter(name=name, value=value) for name, value in values.items()])
    return FitResult(success=True, parameters=parameters, uncertainties=uncertainties)


def test_one_significant_line_is_enough_to_stay_oscillatory() -> None:
    """A two-line template keeps its family while either line is measured."""
    template = _multiplet_template(2, "Exponential", relax=False)

    assert is_oscillatory_admissible(
        template,
        {"A_1": 0.001, "A_3": 0.2},
        {"A_1": 0.05, "A_3": 0.01},
    )


def test_lines_all_consistent_with_zero_are_no_longer_an_oscillation() -> None:
    """Every amplitude inside 2 sigma: the template has decayed to its envelope."""
    template = _multiplet_template(2, "Exponential", relax=False)

    assert not is_oscillatory_admissible(
        template,
        {"A_1": 0.01, "A_3": -0.02},
        {"A_1": 0.05, "A_3": 0.05},
    )


def test_the_bar_is_exactly_two_sigma() -> None:
    """``|A| > 2 sigma``, strictly — the boundary itself is not significant."""
    template = _multiplet_template(1, "Exponential", relax=False)

    assert not is_oscillatory_admissible(template, {"A_1": 0.10}, {"A_1": 0.05})
    assert is_oscillatory_admissible(template, {"A_1": 0.1001}, {"A_1": 0.05})


def test_a_line_the_fit_could_not_constrain_is_not_significant() -> None:
    """No uncertainty is not a small one: it is no measurement of the line at all."""
    template = _multiplet_template(1, "Exponential", relax=False)

    assert not is_oscillatory_admissible(template, {"A_1": 0.2}, {})
    assert not is_oscillatory_admissible(template, {"A_1": 0.2}, {"A_1": float("nan")})
    assert not is_oscillatory_admissible(template, {"A_1": 0.2}, {"A_1": 0.0})


def test_a_significant_relaxation_cannot_rescue_a_vanished_oscillation() -> None:
    """The relaxing shape survives on its ``Exp`` term while its line is gone.

    Exactly the degeneracy the rule is about: the fit is perfectly good, and it
    is a description of relaxation, so it must not count as oscillatory.
    """
    template = _multiplet_template(1, "Exponential", relax=True)

    assert not is_oscillatory_admissible(
        template,
        {"A_1": 0.001, "A_3": 0.25},
        {"A_1": 0.05, "A_3": 0.002},
    )


def test_the_rule_reads_a_fit_result_the_same_way() -> None:
    """The :class:`FitResult` wrapper and the mapping form agree."""
    template = _multiplet_template(2, "Exponential", relax=False)
    vanished = _amplitude_fit_result(
        {"A_1": 0.01, "A_3": 0.01, "A_bg": 0.0}, {"A_1": 0.05, "A_3": 0.05, "A_bg": 0.001}
    )
    alive = _amplitude_fit_result(
        {"A_1": 0.2, "A_3": 0.01, "A_bg": 0.0}, {"A_1": 0.01, "A_3": 0.05, "A_bg": 0.001}
    )

    assert not fit_result_is_oscillatory_admissible(template, vanished)
    assert fit_result_is_oscillatory_admissible(template, alive)
