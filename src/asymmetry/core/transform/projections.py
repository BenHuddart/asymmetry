"""Reduce one period of a run to every asymmetry projection its grouping declares.

A vector-polarisation grouping (EMU's P_x/P_y/P_z) declares several
forward/backward pairs; each projection is the same counts reduced through
:func:`reduce_grouped_asymmetry` with its own pair and alpha.  This is the
Qt-free path the RF vector fit and scripted analyses use; the GUI's stacked
subplots reduce the same way (pinned by ``tests/gui/test_core_projections.py``).
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.instrument import CANONICAL_VECTOR_AXES, derive_projection_pairs
from asymmetry.core.io.periods import held_period_count, period_run
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)

__all__ = ["projection_alphas", "reduce_run_projections"]

#: Grouping keys of the canonical axes' alpha, written by the per-axis alpha table.
_AXIS_ALPHA_KEYS = dict(zip(CANONICAL_VECTOR_AXES, ("alpha_x", "alpha_y", "alpha_z"), strict=True))


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
    alphas = projection_alphas(grouping)
    flags = correction_flags_from_grouping(grouping)
    n_histograms = len(source.histograms)
    datasets: dict[str, MuonDataset] = {}
    for label, (forward, backward) in pairs.items():
        result = reduce_grouped_asymmetry(
            histograms=source.histograms,
            grouping=grouping,
            forward_idx=effective_group_indices(grouping, forward, n_histograms=n_histograms),
            backward_idx=effective_group_indices(grouping, backward, n_histograms=n_histograms),
            alpha=alphas[label],
            use_deadtime=flags.use_deadtime,
            deadtime_mode=flags.deadtime_mode,
            use_background=flags.use_background,
            metadata=source.metadata,
            beta=float(grouping.get("beta", 1.0)),
        )
        datasets[label] = MuonDataset(
            time=result.time,
            asymmetry=result.asymmetry,
            error=result.error,
            metadata={**source.metadata, "projection": label, "period_index": period_index},
            run=source,
        )
    return datasets
