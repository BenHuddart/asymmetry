"""Reduce one period of a run to every asymmetry projection its grouping declares.

A vector-polarisation grouping (EMU's P_x/P_y/P_z) declares several
forward/backward pairs; each projection is the same counts reduced through
:func:`reduce_grouped_asymmetry` with its own pair and alpha.  This is the
Qt-free path the RF vector fit and scripted analyses use; the GUI's stacked
subplots reduce the same way (pinned by ``tests/gui/test_core_projections.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.instrument import CANONICAL_VECTOR_AXES, derive_projection_pairs
from asymmetry.core.io.periods import held_period_count, period_run
from asymmetry.core.transform.grouping import effective_group_indices, grouping_balances
from asymmetry.core.transform.rebin import BINNING_POLICY, binning_spec
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)
from asymmetry.core.transform.t0 import find_t0_for_run, resolve_selected_timing

__all__ = ["projection_alphas", "reduce_run_projections", "reduction_identity"]

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
    }
)


def reduction_identity(grouping: dict[str, Any]) -> str:
    """A digest of everything in an applied grouping that shapes its projections.

    Two groupings with the same identity reduce a run's projections alike; any
    change of groups, alpha, deadtime, background, t0 or binning changes it.
    """
    shaping = {key: value for key, value in grouping.items() if key not in _SELECTION_KEYS}
    blob = json.dumps(
        {"binning_policy": BINNING_POLICY, "grouping": shaping},
        sort_keys=True,
        default=lambda value: np.asarray(value).tolist(),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def projection_alphas(grouping: dict[str, Any]) -> dict[str, float]:
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
    }


def reduce_run_projections(run: Run, period_index: int) -> dict[str, MuonDataset]:
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
    grouping = source.grouping
    pairs = derive_projection_pairs(
        grouping["groups"], grouping.get("group_names"), grouping.get("projections")
    )
    if not pairs:
        raise ValueError(f"Run {run.run_number}'s grouping declares no asymmetry projections.")
    flags = correction_flags_from_grouping(grouping)
    n_histograms = len(source.histograms)
    datasets: dict[str, MuonDataset] = {}
    search = None

    def search_timing(histograms, metadata, **options):
        nonlocal search
        if search is None:
            search = find_t0_for_run(histograms, metadata, **options)
        return search

    for label, (forward, backward) in pairs.items():
        selected_grouping = dict(
            grouping, forward_group=forward, backward_group=backward, vector_axis=label
        )
        alpha, beta = grouping_balances(selected_grouping)
        if "t0_policy" in grouping:
            policy = grouping["t0_policy"]
            resolve_selected_timing(
                source.histograms,
                selected_grouping,
                source.metadata,
                mode=policy["mode"],
                offset_bins=policy.get("offset_bins", 0),
                search_provider=search_timing,
            )
        result = reduce_grouped_asymmetry(
            histograms=source.histograms,
            binning=binning_spec(grouping),
            grouping=selected_grouping,
            forward_idx=effective_group_indices(grouping, forward, n_histograms=n_histograms),
            backward_idx=effective_group_indices(grouping, backward, n_histograms=n_histograms),
            alpha=alpha,
            use_deadtime=flags.use_deadtime,
            deadtime_mode=flags.deadtime_mode,
            use_background=flags.use_background,
            metadata=source.metadata,
            beta=beta,
        )
        from asymmetry.core.transform.preparation import histogram_observation

        datasets[label] = MuonDataset(
            histogram_observation=histogram_observation(
                replace(source, grouping=selected_grouping)
            ),
            ratio_validity=result.ratio_validity,
            time=result.time,
            asymmetry=result.asymmetry,
            error=result.error,
            metadata={
                **source.metadata,
                "projection": label,
                "period_index": period_index,
                "grouped_alignment": result.alignment.to_dict(),
            },
            run=source,
        )
    return datasets
