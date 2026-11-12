"""Replay a selected run's histogram reduction with resolved reference inputs."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.data.ratio_validity import RATIO_OBSERVATION_POLICY, RatioValidity
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.io import background_reference_for_grouping
from asymmetry.core.io.period_counts import run_arithmetic_policy
from asymmetry.core.transform.constituent_counts import CONSTITUENT_PREPARATION_POLICY
from asymmetry.core.transform.corrections import BackgroundReference, reference_observation_metadata
from asymmetry.core.transform.grouping import effective_group_indices, grouping_balances
from asymmetry.core.transform.period_reduction import period_statistic_policy
from asymmetry.core.transform.rebin import BinningSpec, binning_spec
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
    binning: BinningSpec | None = None,
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
        binning=replace(binning_spec(grouping), merge=merge) if binning is None else binning,
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


WIZARD_PREPARATION = (
    f"counts_first_observation_{RATIO_OBSERVATION_POLICY}_{CONSTITUENT_PREPARATION_POLICY}"
)


@dataclass(frozen=True)
class HistogramObservation:
    """A resolved reduction snapshot and selections of its actual output bins.

    The snapshot owns read-only counts and variances, sharing each copied array
    within the capture, and copies settings and histogram timing. It is transient
    and removed by ``fit_record`` before process fan-out. Unknown historical
    curves have no observation and retain independent-curve semantics.
    """

    source: Run
    facility: str | None
    reference: BackgroundReference | None
    binning: BinningSpec
    identity: str
    #: Context belonging to loader-cached period curves, independent of current grouping.
    cached_periods: tuple[HistogramObservation, ...] = ()
    cached_validities: tuple[RatioValidity | None, ...] = ()
    cached_recipes: tuple[EffectiveReductionRecipe | None, ...] = ()

    def select(self, mask: np.ndarray) -> HistogramObservation:
        """Retain a contiguous centre-selected crop before any subsequent merge."""
        selected = np.flatnonzero(mask)
        first, stop = (int(selected[0]), int(selected[-1]) + 1) if selected.size else (0, 0)
        return replace(
            self,
            binning=replace(self.binning, selections=(*self.binning.selections, (first, stop, 1))),
        )

    def merge(self, size: int, factor: int) -> HistogramObservation:
        """Combine complete groups after the current recorded selections."""
        return replace(
            self,
            binning=replace(self.binning, selections=(*self.binning.selections, (0, size, factor))),
        )


def observation_histograms(run: Run) -> list[Histogram]:
    """All buffers owned by a histogram replay, including ignored acquisitions."""
    return [
        *run.histograms,
        *(h for period in (run.grouping.get("period_histograms") or []) for h in period),
        *(h for acquired in run.grouping.get("period_inputs", ()) for h in acquired.histograms),
    ]


def count_metadata_identity(metadata):
    """Condition labels do not change prepared histogram observations.

    Facility/source/timing/correction facts remain in the count identity. Wizard
    model/coordinate identity separately includes the displayed conditions.
    """
    labels = {
        "temperature",
        "field",
        "title",
        "run_label",
        "temperature_spread",
        "field_spread",
        "condition_overrides",
        "combination",
    }
    return {key: value for key, value in metadata.items() if key not in labels}


def histogram_observation(
    run: Run,
    *,
    facility: str | None = None,
    reference: BackgroundReference | None = None,
    binning: BinningSpec | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> HistogramObservation:
    """Capture an explicitly resolved histogram operation at its producing boundary.

    Call only where the supplied settings actually produced the observation;
    attaching a Run to a custom curve does not authorise this replay.
    """
    if cancel_check is not None:
        cancel_check()
    grouping = {k: v for k, v in run.grouping.items() if k != "period_reduced"}
    histograms = observation_histograms(run)
    if reference is not None:
        histograms.extend(reference.histograms)
    memo = {}
    digest = sha256(CONSTITUENT_PREPARATION_POLICY.encode())
    if run_arithmetic_policy(run.metadata):
        digest.update(pickle.dumps(run_arithmetic_policy(run.metadata)))
    if period_statistic_policy(grouping):
        digest.update(pickle.dumps(period_statistic_policy(grouping)))
    for index, hist in enumerate(histograms):
        for role, array in (("counts", hist.counts), ("variance", hist.count_variance)):
            if array is not None:
                if id(array) not in memo:
                    if cancel_check is not None:
                        cancel_check()
                    memo[id(array)] = np.array(array, copy=True, order="C")
                    memo[id(array)].setflags(write=False)
                snapshot = memo[id(array)]
                digest.update(pickle.dumps((index, role, snapshot.shape, snapshot.dtype.str)))
                data = memoryview(snapshot).cast("B")
                for first in range(0, data.nbytes, 1024 * 1024):
                    if cancel_check is not None:
                        cancel_check()
                    digest.update(data[first : first + 1024 * 1024])
    source = deepcopy(replace(run, grouping=grouping), memo)
    resolved_reference = deepcopy(reference, memo)
    # The small identity payload includes every setting and clock, but no counts.
    light_grouping = {
        k: v for k, v in source.grouping.items() if k not in {"period_histograms", "period_inputs"}
    }
    light_grouping["period_inputs"] = tuple(
        replace(acquired, histograms=[], metadata=count_metadata_identity(acquired.metadata))
        for acquired in source.grouping.get("period_inputs", ())
    )
    digest.update(
        pickle.dumps(
            (
                light_grouping,
                count_metadata_identity(source.metadata),
                facility,
                [
                    [
                        (h.bin_width, h.t0_bin, h.good_bin_start, h.good_bin_end, h.t0_time_us)
                        for h in period
                    ]
                    for period in [
                        source.histograms,
                        *(source.grouping.get("period_histograms") or []),
                        *(
                            acquired.histograms
                            for acquired in source.grouping.get("period_inputs", ())
                        ),
                        resolved_reference.histograms if resolved_reference is not None else [],
                    ]
                ],
                replace(resolved_reference, histograms=[])
                if resolved_reference is not None
                else None,
            )
        )
    )
    if cancel_check is not None:
        cancel_check()
    return HistogramObservation(
        source, facility, resolved_reference, binning or binning_spec(grouping), digest.hexdigest()
    )


def wizard_observation_identity(dataset: MuonDataset, *, policy: str = WIZARD_PREPARATION) -> str:
    """Identify the actual curve, replay context, and selected membership for reuse."""
    digest = sha256(policy.encode())
    if run_arithmetic_policy(dataset.metadata):
        digest.update(pickle.dumps(run_arithmetic_policy(dataset.metadata)))
    if dataset.run is not None and period_statistic_policy(dataset.run.grouping):
        digest.update(pickle.dumps(period_statistic_policy(dataset.run.grouping)))
    for array in (dataset.time, dataset.asymmetry, dataset.error):
        values = np.ascontiguousarray(array)
        digest.update(pickle.dumps((values.shape, values.dtype.str)))
        digest.update(values.view(np.uint8))
    digest.update(
        pickle.dumps(
            (
                "rotating_frame" in dataset.metadata,
                dataset.field,
                dataset.temperature,
                tuple(
                    dataset.metadata.get(key)
                    for key in ("field_direction", "field_state", "title", "sample")
                ),
                (dataset.run.field, dataset.run.temperature) if dataset.run is not None else None,
            )
        )
    )
    if policy == WIZARD_PREPARATION and dataset.ratio_validity is not None:
        digest.update(dataset.ratio_validity.reasons.tobytes())
    observation = dataset.histogram_observation
    if observation is not None:
        digest.update(pickle.dumps((observation.identity, observation.binning)))
    return digest.hexdigest()


def wizard_observation_matches(dataset: MuonDataset, signature: str) -> bool:
    """Whether a numerical result describes this exact prepared-observation source."""
    if WIZARD_PREPARATION not in signature:
        return False
    recorded = json.loads(signature)
    return recorded.get("wizard_preparation") == WIZARD_PREPARATION and recorded.get(
        "observation"
    ) == wizard_observation_identity(dataset)


def prepare_wizard_dataset(
    dataset: MuonDataset,
    factor: int,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> MuonDataset:
    """Merge the actual selected observation once, before making worker records."""
    if cancel_check is not None:
        cancel_check()
    BinningSpec(merge=factor)
    if factor == 1:
        return dataset
    observation = dataset.histogram_observation
    if observation is None or "rotating_frame" in dataset.metadata:
        return dataset.merge_curve_bins(factor)
    # Selection indices belong to the current output, not to new midpoint times.
    merged = observation.merge(dataset.n_points, factor)
    reduced = prepare_histogram_asymmetry(
        observation.source,
        merge=1,
        facility=observation.facility,
        reference=observation.reference,
        cancel_check=cancel_check,
        binning=merged.binning,
    )
    return MuonDataset(
        reduced.time,
        reduced.asymmetry,
        reduced.error,
        reduced.metadata_with_alignment(dataset.metadata),
        dataset.run,
        merged,
        reduced.ratio_validity,
        reduced.counting_noise,
        effective_recipe=reduced.effective_recipe,
    )


def wizard_rebuild_dataset(dataset: MuonDataset, factor: int, signature: str) -> MuonDataset:
    """Reconstruct recorded preparation without authorising reuse of historical scores."""
    recorded = json.loads(signature) if signature else {}
    policy = recorded.get("wizard_preparation")
    if policy in {WIZARD_PREPARATION, "counts_first_observation_v1"}:
        if recorded.get("observation") != wizard_observation_identity(dataset, policy=policy):
            raise ValueError(
                "Wizard observation changed; rerun the analysis before rebuilding curves."
            )
        return prepare_wizard_dataset(dataset, factor)
    return replace(dataset, ratio_validity=None).merge_curve_bins(factor)
