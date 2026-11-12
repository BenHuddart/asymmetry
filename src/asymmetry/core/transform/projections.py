"""Reduce one period of a run to every asymmetry projection its grouping declares.

A vector-polarisation grouping (EMU's P_x/P_y/P_z) declares several
forward/backward pairs; each projection is the same counts reduced through
:func:`reduce_grouped_asymmetry` with its own pair and alpha.  This is the
Qt-free path scripted analyses and the GUI's stacked
subplots share (pinned by ``tests/gui/test_core_projections.py``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.instrument import CANONICAL_VECTOR_AXES, derive_projection_pairs
from asymmetry.core.io.periods import held_period_count, period_run
from asymmetry.core.transform.constituent_counts import constituent_metadata, held_period_grids
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.counting_noise import COUNTING_NOISE_POLICY
from asymmetry.core.transform.period_reduction import period_statistic_policy
from asymmetry.core.transform.rebin import BINNING_POLICY, BinningSpec
from asymmetry.core.transform.t0 import resolve_selected_timing

if TYPE_CHECKING:
    from asymmetry.core.transform.rotating_frame import RotatingFrame

__all__ = [
    "projection_alphas",
    "reduce_run_projections",
    "reduce_selected_projections",
    "rotate_selected_projections",
    "reduction_identity",
]

#: Grouping keys of the canonical axes' alpha, written by the per-axis alpha table.
_AXIS_ALPHA_KEYS = dict(zip(CANONICAL_VECTOR_AXES, ("alpha_x", "alpha_y", "alpha_z"), strict=True))


#: Grouping keys that choose what is displayed or are read from the file, not
#: how the projections are reduced: the selected axis and its pair, the period
#: mode (a frame's period weights record it), and the per-period arrays.
_SELECTION_KEYS = frozenset(
    {
        "vector_axis",
        "forward_group",
        "backward_group",
        "period_mode",
        "period_histograms",
        "period_reduced",
        "period_inputs",
        "constituent_coefficients",
    }
)


def reduction_identity(grouping: dict[str, Any]) -> str:
    """A digest of everything in an applied grouping that shapes its projections.

    Two groupings with the same identity reduce a run's projections alike; any
    change of groups, alpha, deadtime, background, t0 or binning changes it.
    """
    shaping = {key: value for key, value in grouping.items() if key not in _SELECTION_KEYS}
    shaping.update(period_statistic_policy(grouping))
    if "period_histograms" in grouping:
        shaping["held_period_grids"] = held_period_grids(grouping)
    if "period_inputs" in grouping:
        shaping["constituents"] = constituent_metadata(grouping)
    blob = json.dumps(
        {
            "binning_policy": BINNING_POLICY,
            "counting_noise_policy": COUNTING_NOISE_POLICY,
            "grouping": computation.without_revision(shaping),
        },
        sort_keys=True,
        default=lambda value: np.asarray(value).tolist(),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def projection_alphas(grouping: dict[str, Any], *, explicit_only: bool = False) -> dict[str, float]:
    """Each declared projection's alpha.

    A canonical axis reads its ``alpha_x``/``alpha_y``/``alpha_z`` key and
    otherwise the base ``alpha``; any other projection carries its own alpha in
    the ``projections`` declaration, else the base alpha.
    """
    base = float(grouping.get("alpha", 1.0))
    declared = {
        str(p["label"]): float(p.get("alpha", base)) for p in grouping.get("projections") or []
    }
    pairs = derive_projection_pairs(
        grouping["groups"], grouping.get("group_names"), grouping.get("projections")
    )
    return {
        label: float(grouping.get(_AXIS_ALPHA_KEYS[label], base))
        if label in _AXIS_ALPHA_KEYS
        else declared.get(label, base)
        for label in pairs
        if not explicit_only or label in declared or _AXIS_ALPHA_KEYS.get(label) in grouping
    }


def reduce_run_projections(
    run: Run,
    period_index: int,
    *,
    facility: str | None = None,
    reference: BackgroundReference | None = None,
) -> dict[str, MuonDataset]:
    """Reduce period ``period_index`` (0-based) of ``run`` to ``{label: dataset}``.

    The run's applied grouping (``run.grouping``) supplies the groups, the
    corrections and the binning; each dataset's metadata records its
    ``projection`` and ``period_index``.
    """
    count = held_period_count(run)
    if not 0 <= period_index < count:
        raise ValueError(
            f"Run {run.run_number} has {count} period(s); got period index {period_index}."
        )
    source = period_run(run, period_index) if count > 1 else run
    return reduce_selected_projections(
        source, period_index=period_index, facility=facility, reference=reference
    )


def reduce_selected_projections(
    run: Run,
    *,
    facility: str | None = None,
    reference: BackgroundReference | None = None,
    period_index: int | None = None,
    binning: BinningSpec | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, MuonDataset]:
    """Reduce the selected period statistic as one counting-observation cohort.

    References must already be resolved by the file/UI boundary. Count variance
    is conditional on calibration, correction values and frame settings.
    """
    from asymmetry.core.transform.counting_noise import capture_counting_observations
    from asymmetry.core.transform.preparation import (
        histogram_observation,
        prepare_histogram_asymmetry,
    )

    source, reference = capture_counting_observations(run, reference, cancel_check=cancel_check)
    grouping = source.grouping
    pairs = derive_projection_pairs(
        grouping["groups"], grouping.get("group_names"), grouping.get("projections")
    )
    if not pairs:
        raise ValueError(f"Run {run.run_number}'s grouping declares no asymmetry projections.")
    datasets = {}
    for label, (forward, backward) in pairs.items():
        if cancel_check is not None:
            cancel_check()
        selected_grouping = dict(
            grouping, forward_group=forward, backward_group=backward, vector_axis=label
        )
        if (
            "t0_policy" in grouping
            and "period_inputs" not in grouping
            and "period_histograms" not in grouping
        ):
            policy = grouping["t0_policy"]
            resolve_selected_timing(
                source.histograms,
                selected_grouping,
                source.metadata,
                mode=policy["mode"],
                cancel_check=cancel_check,
                offset_bins=policy.get("offset_bins", 0),
            )
        selected = replace(source, grouping=selected_grouping)
        result = prepare_histogram_asymmetry(
            selected,
            merge=1,
            facility=facility,
            reference=reference,
            binning=binning,
            cancel_check=cancel_check,
        )
        if result.effective_alpha is not None:
            selected_grouping["alpha"] = result.effective_alpha
        datasets[label] = MuonDataset(
            histogram_observation=histogram_observation(
                selected,
                facility=facility,
                reference=reference,
                binning=binning,
                cancel_check=cancel_check,
            ),
            effective_recipe=result.effective_recipe,
            ratio_validity=result.ratio_validity,
            counting_noise=result.counting_noise,
            time=result.time,
            asymmetry=result.asymmetry,
            error=result.error,
            metadata=result.metadata_with_alignment(
                {**source.metadata, "projection": label, "period_index": period_index}
            ),
            run=selected,
            computation_revision=result.computation_revision,
        )
    return datasets


def rotate_selected_projections(
    run: Run,
    frame: RotatingFrame,
    *,
    weights: Sequence[float],
    facility: str | None = None,
    reference: BackgroundReference | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, MuonDataset]:
    """Prepare and rotate one selected transverse counting cohort."""
    from asymmetry.core.transform.rotating_frame import rotate_transverse

    cohort = reduce_selected_projections(
        run, facility=facility, reference=reference, cancel_check=cancel_check
    )
    if cancel_check is not None:
        cancel_check()
    pair = rotate_transverse(cohort["P_x"], cohort["P_y"], frame, weights=weights)
    if cancel_check is not None:
        cancel_check()
    return {curve.metadata["projection"]: curve for curve in pair}
