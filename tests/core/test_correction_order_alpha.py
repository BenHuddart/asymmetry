"""Alpha is estimated on corrected (deadtime + background) counts.

Pins the physics fix from ``docs/porting/correction-order-alpha-estimation``:
the alpha estimate must consume the same deadtime-corrected, grouped,
background-subtracted forward/backward counts the reduction forms the asymmetry
from, so a calibrated alpha centres the *background-subtracted* asymmetry rather
than the raw totals.

The fixtures are deterministic (no Poisson noise) so the recovered alpha is an
exact number: with a flat pedestal whose F/B ratio differs from the detector
efficiency ratio, estimating on raw counts is biased, and estimating on
background-subtracted counts recovers the true efficiency ratio.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.transform.reduce import (
    corrected_grouped_counts,
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)

A_TRUE = 1.30  # detector efficiency ratio N0_F / N0_B
BG_F = 300.0  # flat forward pedestal
BG_B = 150.0  # flat backward pedestal (ratio 2.0 != A_TRUE, so raw counts are biased)
N_BINS = 200
TAU = 2.19703
BIN_WIDTH = 0.05


def _decay() -> np.ndarray:
    t = np.arange(N_BINS) * BIN_WIDTH
    return 2000.0 * np.exp(-t / TAU)


def _run_with_pedestal() -> Run:
    """Two detectors: F = A_TRUE·decay + BG_F, B = decay + BG_B (t0 at bin 0)."""
    decay = _decay()
    forward = A_TRUE * decay + BG_F
    backward = decay + BG_B
    histograms = [
        Histogram(
            counts=forward, bin_width=BIN_WIDTH, t0_bin=0, good_bin_start=0, good_bin_end=N_BINS - 1
        ),
        Histogram(
            counts=backward,
            bin_width=BIN_WIDTH,
            t0_bin=0,
            good_bin_start=0,
            good_bin_end=N_BINS - 1,
        ),
    ]
    return Run(run_number=1, histograms=histograms, metadata={"instrument": "EMU"})


def _base_grouping() -> dict:
    return {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 0,
        "last_good_bin": N_BINS - 1,
    }


def test_raw_counts_bias_the_estimate_but_subtraction_recovers_true_alpha():
    run = _run_with_pedestal()

    raw = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=_base_grouping(),
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    raw_alpha = estimate_alpha(
        raw.forward, raw.backward, first_good_bin=0, last_good_bin=N_BINS - 1
    )

    grouping = _base_grouping()
    grouping["background_correction"] = True
    grouping["background_mode"] = "fixed"
    grouping["background_fixed_values"] = [BG_F, BG_B]
    corrected = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    corrected_alpha = estimate_alpha(
        corrected.forward, corrected.backward, first_good_bin=0, last_good_bin=N_BINS - 1
    )

    # The flat pedestal biases the raw estimate toward 1; subtracting it recovers
    # the true efficiency ratio essentially exactly.
    assert abs(raw_alpha - A_TRUE) > 0.05
    assert corrected_alpha == np.float64(A_TRUE) or abs(corrected_alpha - A_TRUE) < 1e-9


def test_corrected_counts_match_the_reductions_pre_asymmetry_counts():
    """The estimate reads exactly what the reduction feeds ``binned_fb_asymmetry``."""
    run = _run_with_pedestal()
    grouping = _base_grouping()
    grouping["background_correction"] = True
    grouping["background_mode"] = "fixed"
    grouping["background_fixed_values"] = [BG_F, BG_B]

    corrected = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    # Background-subtracted single-detector groups equal the pure decay signal.
    np.testing.assert_allclose(corrected.forward, A_TRUE * _decay(), rtol=1e-9)
    np.testing.assert_allclose(corrected.backward, _decay(), rtol=1e-9)

    # And the reduction runs without error on the same grouping (shared pipeline).
    reduction = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=A_TRUE,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    # With alpha == A_TRUE the background-subtracted asymmetry is centred on zero.
    assert np.max(np.abs(reduction.asymmetry)) < 1e-6


def test_correction_flags_from_grouping():
    flags = correction_flags_from_grouping(
        {
            "deadtime_correction": True,
            "deadtime_mode": "LOAD",
            "background_correction": True,
            "background_mode": "fixed",
        }
    )
    assert flags.use_deadtime is True
    assert flags.deadtime_mode == "manual"  # "load" folds to "manual"
    assert flags.use_background is True

    off = correction_flags_from_grouping({"background_correction": True, "background_mode": "none"})
    assert off.use_background is False


def test_per_run_estimate_alpha_uses_background_subtracted_counts():
    """resolve_effective_grouping applies background before the per-run alpha."""
    run = _run_with_pedestal()
    run.grouping = {
        "instrument": "EMU",
        "first_good_bin": 0,
        "last_good_bin": N_BINS - 1,
        "good_frames": 1000.0,
    }
    fingerprint = ProfileFingerprint("EMU", 2)
    base = dict(
        name="Default (EMU)",
        fingerprint=fingerprint,
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
    )

    without_bg = resolve_effective_grouping(GroupingProfile(**base), run)

    with_bg = resolve_effective_grouping(
        GroupingProfile(
            **base,
            background_policy=BackgroundPolicy(
                mode="fixed", details={"background_fixed_values": [BG_F, BG_B]}
            ),
        ),
        run,
    )

    assert abs(float(without_bg["alpha"]) - A_TRUE) > 0.05
    assert abs(float(with_bg["alpha"]) - A_TRUE) < 1e-9
