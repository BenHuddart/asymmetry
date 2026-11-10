"""Replay a selected run's histogram reduction with resolved reference inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

from asymmetry.core.data.dataset import Run
from asymmetry.core.io import background_reference_for_grouping
from asymmetry.core.transform.corrections import BackgroundReference, reference_observation_metadata
from asymmetry.core.transform.grouping import effective_group_indices, grouping_balances
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import (
    PeriodAsymmetryReduction,
    correction_flags_from_grouping,
    reduce_period_asymmetry,
)
from asymmetry.core.utils.constants import PeriodMode


def prepare_histogram_asymmetry(
    run: Run,
    *,
    merge: int,
    facility: str | None,
    reference: BackgroundReference | None,
    cancel_check: Callable[[], None] | None = None,
) -> PeriodAsymmetryReduction:
    """Reduce the explicitly selected histogram observation, in percent and µs.

    ``run.grouping`` must describe the requested observation: selected pair,
    period, calibration, timing, good bins, corrections, and base binning.
    ``merge`` combines complete groups of those base output bins counts-first.
    It is additional to base binning, not a replacement for it. Source arrays
    and grouping are read without modification.

    The caller resolves live reference identity, acquisition timing and scaling
    before calling; each period still adapts reference exposure through the
    established reference owner. No files, GUI state or CLI defaults are read.
    This replays the run's complete selected good-bin support; it does not infer
    the provenance or cropped membership of an existing curve. Curve-only and
    rotated observations require their own curve-domain operation.

    This operation is synchronous. Interactive callers retain their worker and
    cache ownership and may pass a cooperative ``cancel_check`` which raises
    when cancelled. A check is made before any numerical preparation.
    """
    if cancel_check is not None:
        cancel_check()
    if not run.histograms:
        raise ValueError("Histogram preparation requires raw histograms.")
    grouping = run.grouping
    forward_idx, backward_idx = (
        effective_group_indices(
            grouping, int(grouping.get(key, default)), n_histograms=len(run.histograms)
        )
        for key, default in (("forward_group", 1), ("backward_group", 2))
    )
    alpha, beta = grouping_balances(grouping)
    flags = correction_flags_from_grouping(grouping)
    return reduce_period_asymmetry(
        run.histograms,
        grouping,
        str(grouping.get("period_mode", PeriodMode.RED)),
        binning=replace(binning_spec(grouping), merge=merge),
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        alpha=alpha,
        beta=beta,
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        facility=facility,
        metadata=reference_observation_metadata(run),
        reference_resolver=partial(background_reference_for_grouping, reference),
        cancel_check=cancel_check,
    )
