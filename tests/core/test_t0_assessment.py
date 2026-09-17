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
    consensus: int,
    strategy: str = "prompt_peak",
    spread: int | None = None,
    ok: bool = True,
) -> RunT0Search:
    estimates = [T0Estimate(t0, strategy, t0, ok=True) for t0 in per_detector]
    return RunT0Search(
        estimates=estimates,
        consensus_t0_bin=consensus,
        spread_bins=max(per_detector) - min(per_detector) if spread is None else spread,
        strategy=strategy,
        ok=ok,
    )


# -- tolerance lookup ------------------------------------------------------


def test_tolerance_bins_maps_each_strategy_to_its_source_family() -> None:
    assert tolerance_bins("prompt_peak") == T0_TOLERANCE_BINS["continuous"] == 2
    assert tolerance_bins("pulse_edge") == T0_TOLERANCE_BINS["pulsed"] == 3


def test_tolerance_bins_rejects_an_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        tolerance_bins("guesswork")


# -- error branches --------------------------------------------------------


def test_missing_header_t0_is_an_error_naming_the_detected_fallback() -> None:
    histograms = _histograms([10, 10])
    verdict = assess_t0(
        histograms,
        _grouping(2, t0_source="missing"),
        _search([10, 10], consensus=10),
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

    verdict = assess_t0(histograms, grouping, _search([3, 3], consensus=3))

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

    verdict = assess_t0(histograms, _grouping(2), _search([9, 11], consensus=11))

    assert verdict.level == "ok"
    assert verdict.messages == ()
    assert verdict.delta_bins == 1
    assert verdict.outlier_detectors == ()


def test_header_conflict_warns_and_says_which_value_is_used() -> None:
    histograms = _histograms([10, 10])

    verdict = assess_t0(
        histograms, _grouping(2, t0_source="conflict"), _search([10, 10], consensus=10)
    )

    assert verdict.level == "warn"
    assert verdict.messages == ("Header time_zero disagrees with t0_bin; using t0_bin",)
    assert verdict.delta_bins == 0


@pytest.mark.parametrize(("base", "detected", "file_bin"), [(0, 14, 10), (1, 15, 11)])
def test_divergent_consensus_warns_naming_both_bins_and_the_tolerance(
    base: int, detected: int, file_bin: int
) -> None:
    """Both bins are quoted in the run's display base; the delta is base-free."""
    histograms = _histograms([10, 10])

    verdict = assess_t0(
        histograms, _grouping(2, bin_index_base=base), _search([14, 14], consensus=14)
    )

    assert verdict.level == "warn"
    assert verdict.delta_bins == 4
    assert verdict.messages[0] == (
        f"Detected t0 is bin {detected}, file t0 is bin {file_bin} — further apart "
        f"than the 2-bin tolerance"
    )


def test_base_shifts_no_message_that_carries_no_bin_number() -> None:
    """Detector numbers and spreads are counts, not indices — the base misses them."""
    histograms = _histograms([10, 10, 10, 10])
    search = _search([10, 20, 20, 20], consensus=20, spread=10)

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

    verdict = assess_t0(histograms, _grouping(2), _search([12, 12], consensus=12))

    assert verdict.level == "ok"
    assert verdict.delta_bins == 2


def test_pulsed_tolerance_is_one_bin_wider_than_continuous() -> None:
    histograms = _histograms([10, 10])
    search = _search([13, 13], consensus=13, strategy="pulse_edge")

    verdict = assess_t0(histograms, _grouping(2), search)

    assert verdict.level == "ok"  # |3| <= 3 for a pulsed source
    assert assess_t0(histograms, _grouping(2), _search([13, 13], consensus=13)).level == "warn"


def test_per_detector_outliers_are_named_one_based() -> None:
    histograms = _histograms([10, 10, 10, 10])
    # Detectors 2 and 4 (1-based) sit 5 bins from their own file t0.
    search = _search([10, 15, 11, 5], consensus=10)

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "warn"
    assert verdict.outlier_detectors == (2, 4)
    assert "Detectors 2, 4 disagree with their file t0 by more than 2 bins" in verdict.messages


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
    histograms = _histograms([10] * 4)
    search = _search([10, 10, 10, 10], consensus=10, spread=9)  # > 4 * 2

    verdict = assess_t0(histograms, _grouping(4), search)

    assert verdict.level == "warn"
    assert verdict.messages == ("Detector spread 9 bins — check the source type",)


def test_warnings_accumulate_in_check_order() -> None:
    histograms = _histograms([10, 10, 10, 10])
    search = _search([10, 20, 20, 20], consensus=20, spread=10)

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
