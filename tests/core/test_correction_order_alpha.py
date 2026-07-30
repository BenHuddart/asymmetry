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


# --- facility defaulting reaches the reduction chain -------------------------
#
# A pre-t0 ``range`` window is trimmed to whole accelerator periods only when the
# facility is known. That used to be the caller's job, so a script driving the
# reduction directly reduced with an untrimmed window while the GUI, which passes
# the string, trimmed — the same run, two different numbers.

PSI_BIN_WIDTH = 0.005  # µs; the PSI 0.01975 µs period spans just under four bins
PSI_T0_BIN = 150


def _psi_style_run(*, facility: str = "PSI", instrument: str = "GPS") -> Run:
    """Continuous-source-shaped run with a genuine pre-t0 region."""
    counts = np.concatenate(
        [
            np.full(PSI_T0_BIN, 40.0) + np.arange(PSI_T0_BIN) * 0.5,
            2000.0 * np.exp(-np.arange(250) * PSI_BIN_WIDTH / TAU) + 40.0,
        ]
    )
    histograms = [
        Histogram(
            counts=counts * scale,
            bin_width=PSI_BIN_WIDTH,
            t0_bin=PSI_T0_BIN,
            good_bin_start=PSI_T0_BIN,
            good_bin_end=counts.size - 1,
        )
        for scale in (A_TRUE, 1.0)
    ]
    metadata: dict = {"instrument": instrument}
    if facility:
        metadata["facility"] = facility
    return Run(run_number=2, histograms=histograms, metadata=metadata)


def _psi_grouping() -> dict:
    return {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": PSI_T0_BIN,
        "last_good_bin": 399,
        "background_correction": True,
        "background_mode": "range",
    }


def _reduced_background(run: Run, **kwargs):
    corrected = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=_psi_grouping(),
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        **kwargs,
    )
    state = corrected.background_state
    assert state is not None
    return state["ranges"][0], corrected.background_level


def test_reduction_derives_the_facility_from_run_metadata():
    """The regression: without ``facility=`` the window was silently untrimmed."""
    run = _psi_style_run()
    untrimmed_range, untrimmed_level = _reduced_background(run, facility="")
    derived_range, derived_level = _reduced_background(run, metadata=run.metadata)
    explicit_range, explicit_level = _reduced_background(run, facility="PSI")

    assert derived_range == explicit_range
    assert derived_level == explicit_level
    # And the trimming is not a no-op on this window, so the assertion has teeth.
    assert derived_range != untrimmed_range
    assert derived_level != untrimmed_level


def test_reduction_facility_empty_string_is_still_the_opt_out():
    run = _psi_style_run()
    assert _reduced_background(run, facility="", metadata=run.metadata) == _reduced_background(
        run, facility=""
    )


def test_reduce_grouped_asymmetry_defaults_the_facility_too():
    """Not just the counts helper — the full reduction takes the same route."""
    run = _psi_style_run()
    kwargs = dict(
        histograms=run.histograms,
        grouping=_psi_grouping(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=A_TRUE,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    derived = reduce_grouped_asymmetry(**kwargs, metadata=run.metadata)
    explicit = reduce_grouped_asymmetry(**kwargs, facility="PSI")
    untrimmed = reduce_grouped_asymmetry(**kwargs, facility="")
    assert derived.background_state == explicit.background_state
    assert derived.background_state != untrimmed.background_state
    np.testing.assert_allclose(derived.asymmetry, explicit.asymmetry, rtol=0, atol=0)


def test_reduction_falls_back_to_the_groupings_instrument_without_metadata():
    """Last resort when a caller passes neither: the canonical instrument name the
    loaders stamp into the grouping. It resolves nothing for an instrument whose
    name is not a facility, which is the honest outcome — see the module note."""
    run = _psi_style_run(facility="", instrument="PSI")
    grouping = _psi_grouping()
    grouping["instrument"] = "PSI"
    corrected = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    state = corrected.background_state
    assert state is not None
    trimmed, _ = _reduced_background(run, facility="PSI")
    assert state["ranges"][0] == trimmed
