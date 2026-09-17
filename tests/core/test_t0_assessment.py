"""Verdicts on a run's time zero (:func:`assess_t0`, decision D8).

The GUI renders these messages verbatim in the grouping window's t0 row, so the
tests assert the exact text, not just the level.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.t0 import (
    EFFECTIVE_DETECTOR_T0_KEY,
    T0_TOLERANCE_BINS,
    RunT0Search,
    T0Estimate,
    assess_t0,
    tolerance_bins,
)


def _histograms(t0_bins: list[int], n_bins: int = 64) -> list[Histogram]:
    return [Histogram(counts=np.full(n_bins, 10.0), bin_width=0.016, t0_bin=t0) for t0 in t0_bins]


def _grouping(n_hist: int, **extra: object) -> dict:
    half = n_hist // 2
    return {
        "groups": {
            1: list(range(1, half + 1)),
            2: list(range(half + 1, n_hist + 1)),
        },
        "forward_group": 1,
        "backward_group": 2,
    } | extra


def _search(
    per_detector: list[int],
    *,
    strategy: str = "prompt_peak",
    peaks: list[int] | None = None,
    width: int = 0,
    ok: bool = True,
) -> RunT0Search:
    """A finished search whose detectors resolved to ``per_detector``.

    The consensus and raw spread are derived, not dictated: :func:`assess_t0`
    reads the per-detector estimates and the histograms' file t0 and works out
    the shifts itself, so a hand-written consensus could only lie. ``width`` is
    the measured width of the feature t0 was read off; the default 0 leaves the
    tolerance at its source-family floor.
    """
    peak_bins = per_detector if peaks is None else peaks
    estimates = [
        T0Estimate(t0, strategy, peak, ok=True, width_bins=width)
        for t0, peak in zip(per_detector, peak_bins, strict=True)
    ]
    return RunT0Search(
        estimates=estimates,
        consensus_t0_bin=int(round(float(np.median(per_detector)))),
        spread_bins=max(per_detector) - min(per_detector),
        strategy=strategy,
        ok=ok,
        width_bins=width,
    )


# -- tolerance lookup ------------------------------------------------------


def test_tolerance_bins_maps_each_strategy_to_its_source_family() -> None:
    assert tolerance_bins("prompt_peak") == T0_TOLERANCE_BINS["continuous"] == 2
    assert tolerance_bins("pulse_edge") == T0_TOLERANCE_BINS["pulsed"] == 3


def test_tolerance_bins_rejects_an_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        tolerance_bins("guesswork")


def test_the_source_floor_is_a_floor_not_a_ceiling() -> None:
    """A feature wider than the floor sets the tolerance; a narrower one does not.

    A bin index cannot name the centre of a prompt peak more precisely than the
    peak is wide, and how wide that is in bins depends on the binning: the same
    PSI peak is 1 bin at 1 ns and 6 at 98 ps.
    """
    assert tolerance_bins("prompt_peak", 1) == 2  # floor wins on coarse binning
    assert tolerance_bins("prompt_peak", 6) == 6  # 98 ps GPS binning
    # A 16 ns ISIS run whose pulse rises over 5 bins gets 5, not the 3-bin floor.
    assert tolerance_bins("pulse_edge", 5) == 5
    assert tolerance_bins("pulse_edge", 2) == 3


# -- error branches --------------------------------------------------------


def test_missing_header_t0_is_an_error_naming_the_detected_fallback() -> None:
    histograms = _histograms([10, 10])
    verdict = assess_t0(
        histograms,
        _grouping(2, t0_source="missing"),
        _search([10, 10]),
    )

    assert verdict.level == "error"
    assert verdict.messages == ("No time zero in the file header; using the detected value",)
    assert verdict.delta_bins is None
    assert verdict.outlier_detectors == ()


@pytest.mark.parametrize(("base", "shown"), [(0, 99), (1, 100)])
def test_t0_outside_the_histogram_range_is_an_error(base: int, shown: int) -> None:
    histograms = _histograms([10, 10], n_bins=8)
    # A stored override drives the effective common t0 past the last bin. The
    # message quotes it in the run's display base, never the internal index.
    grouping = _grouping(2, bin_index_base=base, **{EFFECTIVE_DETECTOR_T0_KEY: [10, 99]})

    verdict = assess_t0(histograms, grouping, _search([3, 3]))

    assert verdict.level == "error"
    assert verdict.messages == (f"Time zero is bin {shown}, outside the run's 8 bins",)


def test_missing_wins_over_the_range_error() -> None:
    """First error in severity order wins; the messages never accumulate."""
    histograms = _histograms([10, 10], n_bins=8)
    grouping = _grouping(2, t0_source="missing", **{EFFECTIVE_DETECTOR_T0_KEY: [10, 99]})

    verdict = assess_t0(histograms, grouping, None)

    assert verdict.level == "error"
    assert len(verdict.messages) == 1
    assert verdict.messages[0].startswith("No time zero in the file header")


# -- warning branches ------------------------------------------------------


def test_clean_run_is_ok_with_the_signed_delta() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2), _search([11, 11]))

    assert verdict.level == "ok"
    assert verdict.messages == ()
    assert verdict.delta_bins == 1
    assert verdict.outlier_detectors == ()


def test_header_conflict_warns_and_says_which_value_is_used() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2, t0_source="conflict"), _search([10, 10]))

    assert verdict.level == "warn"
    assert verdict.messages == ("Header time_zero disagrees with t0_bin; using t0_bin",)
    assert verdict.delta_bins == 0


@pytest.mark.parametrize(("base", "detected", "file_bin"), [(0, 14, 10), (1, 15, 11)])
def test_divergent_consensus_warns_naming_both_bins_and_the_tolerance(
    base: int, detected: int, file_bin: int
) -> None:
    """Both bins are quoted in the run's display base; the delta is base-free."""
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2, bin_index_base=base), _search([14, 14]))

    assert verdict.level == "warn"
    assert verdict.delta_bins == 4
    assert verdict.messages[0] == (
        f"Detected t0 is bin {detected}, file t0 is bin {file_bin} — further apart "
        f"than the 2-bin tolerance"
    )


def test_base_shifts_no_message_that_carries_no_bin_number() -> None:
    """Detector numbers and spreads are counts, not indices — the base misses them."""
    histograms = _histograms([10, 10, 10, 10])
    search = _search([10, 20, 20, 20])

    zero_based = assess_t0(histograms, _grouping(4), search)
    one_based = assess_t0(histograms, _grouping(4, bin_index_base=1), search)

    assert zero_based.delta_bins == one_based.delta_bins == 10
    assert zero_based.outlier_detectors == one_based.outlier_detectors
    # Only the "Detected t0 is bin …" message differs between the two bases.
    differing = [a for a, b in zip(zero_based.messages, one_based.messages, strict=True) if a != b]
    assert len(differing) == 1
    assert differing[0].startswith("Detected t0 is bin ")


def test_divergence_within_tolerance_does_not_warn() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2), _search([12, 12]))

    assert verdict.level == "ok"
    assert verdict.delta_bins == 2


def test_pulsed_tolerance_is_one_bin_wider_than_continuous() -> None:
    histograms = _histograms([10, 10])
    search = _search([13, 13], strategy="pulse_edge")

    verdict = assess_t0(histograms, _grouping(2), search)

    assert verdict.level == "ok"  # |3| <= 3 for a pulsed source
    assert assess_t0(histograms, _grouping(2), _search([13, 13])).level == "warn"


def test_per_detector_outliers_are_named_one_based() -> None:
    histograms = _histograms([10, 10, 10, 10])
    # Detectors 2 and 4 (1-based) sit 5 bins from their own file t0.
    search = _search([10, 15, 11, 5])

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "warn"
    assert verdict.outlier_detectors == (2, 4)
    assert (
        "Detectors 2, 4 disagree with the other detectors' t0 shift by more than 2 bins"
        in verdict.messages
    )


def test_failed_per_detector_estimates_are_skipped() -> None:
    histograms = _histograms([10, 10])
    search = RunT0Search(
        estimates=[
            T0Estimate(0, "prompt_peak", 0, ok=False, message="Histogram has no counts"),
            T0Estimate(10, "prompt_peak", 10, ok=True),
        ],
        consensus_t0_bin=10,
        spread_bins=0,
        strategy="prompt_peak",
        ok=True,
    )

    verdict = assess_t0(histograms, _grouping(2), search)

    assert verdict.outlier_detectors == ()
    assert verdict.level == "ok"


def test_wide_detector_spread_warns_about_the_strategy() -> None:
    """The spread that warns is the spread of the *shifts*, not of the estimates."""
    histograms = _histograms([10] * 4)
    search = _search([10, 10, 10, 19])  # shifts 0, 0, 0, +9 — range 9 > 4 * 2

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "warn"
    assert "Detector spread 9 bins — check the source type" in verdict.messages


def test_a_staggered_run_is_clean_when_every_detector_matches_its_own_header() -> None:
    """The PSI GPS case: detector 4 sits 170 bins early and its header says so.

    The raw estimates span 172 bins, but every detector is within 2 of its own
    file t0, so the run's shift is +2 with a spread of 2 — nothing to report. The
    pre-Phase-7 rule compared the estimates' median (52) with the group maximum
    (220) and declared a 168-bin divergence plus a 172-bin spread.
    """
    histograms = _histograms([50, 46, 50, 220], n_bins=512)
    search = _search([52, 48, 51, 222])

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "ok"
    assert verdict.messages == ()
    assert verdict.delta_bins == 2
    assert verdict.outlier_detectors == ()


def test_peak_jitter_inside_the_peak_width_is_not_a_disagreement() -> None:
    """The real 98 ps GPS run: shifts −1…6 around a median of 2, peak FWHM 6 bins.

    0.3–0.4 ns of jitter well inside the prompt peak. Against the bare 2-bin
    floor six of the fifteen detectors read as outliers; against the peak's own
    width, none does.
    """
    histograms = _histograms([50, 50, 50, 50, 50, 50], n_bins=512)
    search = _search([49, 52, 51, 56, 52, 53], width=6)

    verdict = assess_t0(histograms, _grouping(6), search)

    assert verdict.level == "ok"
    assert verdict.messages == ()
    assert verdict.delta_bins == 2
    assert verdict.outlier_detectors == ()
    # The same shifts against the 2-bin floor alone would have named detectors.
    assert assess_t0(histograms, _grouping(6), _search([49, 52, 51, 56, 52, 53])).level == "warn"


def test_a_detector_outside_the_peak_width_is_still_an_outlier() -> None:
    """The wider tolerance forgives jitter, not a genuine 10-bin disagreement."""
    histograms = _histograms([50, 50, 50, 50, 50, 50], n_bins=512)
    search = _search([49, 52, 51, 62, 52, 53], width=6)

    verdict = assess_t0(histograms, _grouping(6), search)

    assert verdict.level == "warn"
    assert verdict.outlier_detectors == (4,)
    assert (
        "Detectors 4 disagree with the other detectors' t0 shift by more than 6 bins"
        in verdict.messages
    )


def test_one_detector_disagreeing_with_the_others_about_the_shift_is_the_outlier() -> None:
    """Outliers are measured against the run's shift, not against each header."""
    histograms = _histograms([50, 46, 50, 220], n_bins=512)
    # Detector 3 moved 10 bins where the rest moved 2.
    search = _search([52, 48, 60, 222])

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "warn"
    assert verdict.outlier_detectors == (3,)
    assert (
        "Detectors 3 disagree with the other detectors' t0 shift by more than 2 bins"
        in verdict.messages
    )


# -- good window against the detected t0 -----------------------------------


def test_good_window_opening_on_the_detected_t0_warns() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2, first_good_bin=12), _search([12, 12]))

    assert verdict.level == "warn"
    assert verdict.messages == ("First good bin 12 is at or before the detected t0 (bin 12)",)


@pytest.mark.parametrize(("base", "first", "detected"), [(0, 11, 12), (1, 12, 13)])
def test_the_good_window_warning_is_written_in_the_display_base(
    base: int, first: int, detected: int
) -> None:
    histograms = _histograms([10, 10])
    grouping = _grouping(2, bin_index_base=base, first_good_bin=11)

    verdict = assess_t0(histograms, grouping, _search([12, 12]))

    assert (
        f"First good bin {first} is at or before the detected t0 (bin {detected})"
        in verdict.messages
    )


def test_a_good_window_after_the_detected_t0_says_nothing() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2, first_good_bin=13), _search([12, 12]))

    assert verdict.level == "ok"
    assert verdict.messages == ()


def test_a_pulsed_window_inside_the_muon_pulse_warns() -> None:
    """At a pulsed source the window must clear the pulse, not just its midpoint."""
    histograms = _histograms([10, 10])
    # Edge midpoint at bin 12, pulse peak four bins later.
    search = _search([12, 12], strategy="pulse_edge", peaks=[16, 16])

    verdict = assess_t0(histograms, _grouping(2, first_good_bin=14), search)

    assert verdict.level == "warn"
    assert verdict.messages == ("First good bin 14 is inside the muon pulse (peak at bin 16)",)


def test_a_pulsed_window_clear_of_the_pulse_says_nothing() -> None:
    histograms = _histograms([10, 10])
    search = _search([12, 12], strategy="pulse_edge", peaks=[16, 16])

    verdict = assess_t0(histograms, _grouping(2, first_good_bin=17), search)

    assert verdict.level == "ok"
    assert verdict.messages == ()


def test_the_pulse_check_reads_peaks_on_the_common_bin_not_each_detector_axis() -> None:
    """A detector aligned 10 bins forward carries its peak 10 bins forward too."""
    histograms = _histograms([10, 20])
    # Common t0 is 20 (the group max). Detector 1's peak at bin 16 lands on
    # 16 + (20 − 10) = 26; detector 2's peak at 26 lands on 26.
    search = _search([12, 22], strategy="pulse_edge", peaks=[16, 26])

    verdict = assess_t0(histograms, _grouping(2, first_good_bin=26), search)

    assert "First good bin 26 is inside the muon pulse (peak at bin 26)" in verdict.messages


def test_warnings_accumulate_in_check_order() -> None:
    histograms = _histograms([10, 10, 10, 10])
    search = _search([10, 20, 20, 20])

    verdict = assess_t0(histograms, _grouping(4, t0_source="conflict"), search)

    assert verdict.level == "warn"
    assert [message.split()[0] for message in verdict.messages] == [
        "Header",
        "Detected",
        "Detectors",
        "Detector",
    ]


# -- no detection yet ------------------------------------------------------


def test_pending_detection_leaves_delta_unknown_and_file_checks_live() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(histograms, _grouping(2, t0_source="conflict"), None)

    assert verdict.level == "warn"
    assert verdict.delta_bins is None
    assert verdict.messages == ("Header time_zero disagrees with t0_bin; using t0_bin",)


def test_a_failed_search_contributes_no_verdict() -> None:
    histograms = _histograms([10, 10])
    search = RunT0Search(
        estimates=[],
        consensus_t0_bin=0,
        spread_bins=0,
        strategy="prompt_peak",
        ok=False,
        message="No histogram produced a t0 estimate",
    )

    verdict = assess_t0(histograms, _grouping(2), search)

    assert verdict.level == "ok"
    assert verdict.delta_bins is None
