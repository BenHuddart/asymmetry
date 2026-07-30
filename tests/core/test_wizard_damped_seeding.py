"""Acceptance tests for blind seeding of heavily damped oscillations.

The feature under test: a µSR record whose oscillation is heavily damped lives
only in the leading nanoseconds of the record, which every symmetric apodisation
deletes.  Both of the fit wizard's seeding FFTs used to be Hann-windowed, so a
damped-cosine recommendation was only reachable when the user handed the wizard
the frequencies.  These tests pin the blind path: the unwindowed early-window
pass finds the line, the fingerprint carries it, the seeding path uses it, and a
damped-oscillation family ranks top with **no** ``user_frequencies_mhz``.

Every synthetic parameter here is invented.  The records are ordered-magnet-like
zero-field signals: one or two damped cosines on a slowly relaxing tail, at fine
binning, with µSR-like errors that grow as ``exp(t / 2·τ_µ)`` and are capped at
100 %.
"""

from __future__ import annotations

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


def _damped_record(
    *,
    seed: int = _SEED,
    n_points: int = 1200,
    t_max: float = 4.0,
    lines: tuple[tuple[float, float, float], ...] = ((_LINE_AMPLITUDE, _LINE_MHZ, _LINE_RATE),),
    tail_amplitude: float = 6.0,
    tail_rate: float = 0.25,
    baseline: float = 0.5,
    sigma0: float = 0.9,
) -> MuonDataset:
    """Damped precession on a slowly relaxing tail. All values invented."""
    dt = t_max / n_points
    time = np.arange(dt, t_max + 0.5 * dt, dt)[:n_points]
    sigma = np.minimum(sigma0 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    asymmetry = tail_amplitude * np.exp(-tail_rate * time) + baseline
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


def _two_line_record(seed: int = _SEED) -> MuonDataset:
    return _damped_record(
        seed=seed,
        lines=(
            (_LINE_AMPLITUDE, _LINE_MHZ, _LINE_RATE),
            (_SECOND_LINE_AMPLITUDE, _SECOND_LINE_MHZ, _SECOND_LINE_RATE),
        ),
    )


def _narrow_line_record(seed: int = 4242) -> MuonDataset:
    """A conventional narrow line, alive across the whole record. Invented."""
    time = np.linspace(0.05, 10.0, 1500)
    sigma = np.minimum(0.6 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    rng = np.random.default_rng(seed)
    asymmetry = 18.0 * np.exp(-0.12 * time) * np.cos(2.0 * np.pi * 1.4 * time) + 3.0
    return MuonDataset(
        time=time,
        asymmetry=asymmetry + rng.normal(0.0, sigma),
        error=sigma,
        metadata={"run_number": 2, "field": 100.0, "field_direction": "TF"},
    )


def _pure_noise_record(seed: int) -> MuonDataset:
    time = np.linspace(4.0e-3, 4.0, 1200)
    sigma = np.minimum(0.9 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
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


def _nearest(peaks, frequency_mhz: float) -> DetectedPeak | None:
    within = [
        peak
        for peak in peaks
        if abs(peak.frequency_mhz - frequency_mhz)
        <= max(3.0 / max(peak.crop_us or 1.0, 1e-12), 0.05 * frequency_mhz)
    ]
    return max(within, key=lambda peak: peak.snr) if within else None


# --------------------------------------------------------------------------- #
# Workstream A — the early-window pass and its study
# --------------------------------------------------------------------------- #


def test_hann_pass_is_structurally_blind_to_the_damped_line() -> None:
    """The premise: the historical Hann-only pass sees nothing at all here.

    Not "sees it weakly" — the window is zero at the first sample, so the line's
    entire support is deleted and the pass returns no peaks.
    """
    analysis = analyze_dataset_peaks(_damped_record(), early_pass=False)

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
    for target in (_LINE_MHZ, _SECOND_LINE_MHZ):
        peak = _nearest(early.peaks, target)
        assert peak is not None, f"no early-pass line near {target} MHz"
        assert abs(peak.frequency_mhz - target) <= 1.0 / peak.crop_us


@pytest.mark.parametrize("draws", [500])
def test_pure_noise_adds_no_early_peaks(draws: int) -> None:
    """False-seed control (study step 3): zero early-pass peaks over 500 draws.

    The gate ``_EARLY_MIN_SNR`` is re-derived against exactly this null, not
    inherited from the Hann pass, so this is the measurement it rests on.
    """
    total = sum(len(_early_peaks(_pure_noise_record(90_000 + i)).peaks) for i in range(draws))

    assert total == 0


def test_relaxing_tail_without_oscillation_adds_no_early_peaks() -> None:
    """A record with structure but no oscillation must stay peak-free.

    Pure noise alone is too easy a null: an unwindowed short crop of a *relaxing*
    record leaves residual trend curvature, which is what the low-frequency guard
    and the rectangular leakage profile exist to reject.
    """
    total = sum(len(_early_peaks(_tail_only_record(70_000 + i)).peaks) for i in range(250))

    assert total == 0


def test_conventional_narrow_line_is_untouched_by_the_early_pass() -> None:
    """Study requirement (iii): no spurious additions, Hann peaks unchanged."""
    dataset = _narrow_line_record()

    with_early = analyze_dataset_peaks(dataset, burg_check="never")
    hann_only = analyze_dataset_peaks(dataset, burg_check="never", early_pass=False)

    assert [peak.source for peak in with_early.peaks] == [peak.source for peak in hann_only.peaks]
    assert not any(peak.source == "early_fft" for peak in with_early.peaks)
    for merged, plain in zip(with_early.peaks, hann_only.peaks):
        assert merged == plain


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
    assert fingerprint.damped_line_crop_us > 0.0
    assert fingerprint.oscillatory_hint is True
    # The Hann view on its own would have gated precession off. Its "dominant
    # line" is the leftover slow tail, completing well under one cycle in the
    # window and sitting nowhere near the real oscillation.
    assert fingerprint.dominant_fft_cycles_in_window < 1.5
    assert abs(fingerprint.dominant_fft_frequency_mhz - _LINE_MHZ) > 10.0


def test_fingerprint_leaves_a_conventional_record_alone() -> None:
    fingerprint = fingerprint_spectrum(_narrow_line_record())

    assert not fingerprint.has_damped_line_candidate
    assert fingerprint.damped_line_frequency_mhz == 0.0
    # ...and the hint still fires, from the Hann view, exactly as before.
    assert fingerprint.oscillatory_hint is True


def test_early_peaks_seed_the_multiplet_builder_like_user_frequencies() -> None:
    analysis = _analysis([_early_peak(60.0), _early_peak(30.0)], resolution=4.0)

    templates = build_oscillatory_multiplet_templates(analysis)

    assert [template.key for template in templates] == [
        "oscillatory2_exp_constant",
        "oscillatory2_gaussian_constant",
    ]


def test_multiplet_seeds_are_pruned_by_rank_within_each_pass() -> None:
    # Four early-pass lines, ranked by their own SNR; the builder takes three.
    analysis = _analysis(
        [_early_peak(90.0, snr=12.0), _early_peak(60.0, snr=11.0), _early_peak(30.0, snr=10.0)]
        + [_early_peak(20.0, snr=9.0)],
        resolution=4.0,
    )

    seeds = _multiplet_seed_peaks(analysis, 4)

    assert [peak.frequency_mhz for peak in seeds] == [90.0, 60.0, 30.0]


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


def test_damped_envelope_rate_comes_from_the_crop_and_only_from_it() -> None:
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


def _ranked(recommendation):
    return [
        assessment
        for assessment in recommendation.sorted_assessments()
        if not assessment.is_null_baseline
        and assessment.is_successful
        and not assessment.is_disqualified
    ]


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_blind_wizard_ranks_a_damped_oscillation_top() -> None:
    """The headline: no ``user_frequencies_mhz``, damped cosine still wins.

    On ``main`` this record's recommendation was a static Kubo-Toyabe — the
    oscillation was invisible to both seeding FFTs, so no oscillatory candidate
    was ever seeded near it.
    """
    dataset = _damped_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.recommended_key == "oscillatory_exp_constant"
    winner = recommendation.recommended_assessment
    assert winner is not None
    fitted = winner.fit_result.parameters["frequency"].value
    # Seeded, and then fitted, within the early pass's own resolution.
    crop_us = recommendation.fingerprint.damped_line_crop_us
    assert abs(fitted - _LINE_MHZ) <= 1.0 / crop_us
    assert winner.fit_result.parameters["Lambda"].value == pytest.approx(_LINE_RATE, rel=0.3)


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_blind_wizard_ranks_a_two_oscillation_family_top() -> None:
    dataset = _two_line_record()

    recommendation = build_fit_wizard_recommendation(
        dataset, max_workers=1, refine_top_candidates=0
    )

    assert recommendation.peak_analysis is not None
    early = [peak for peak in recommendation.peak_analysis.peaks if peak.source == "early_fft"]
    assert len(early) == 2, "both damped lines must reach the seeding path"

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
    assert not any(peak.source == "early_fft" for peak in recommendation.peak_analysis.peaks)
    assert not recommendation.fingerprint.has_damped_line_candidate
    winner = recommendation.recommended_assessment
    assert winner is not None
    assert winner.fit_result.parameters["frequency"].value == pytest.approx(1.4, rel=0.02)
