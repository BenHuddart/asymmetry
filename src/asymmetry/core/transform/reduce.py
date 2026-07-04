"""Grouped-histogram → asymmetry reduction (the counts-then-ratio pipeline).

This module owns the single reduction chokepoint that turns a run's detector
histograms into a forward/backward asymmetry curve, exactly the way the app does
it for real: deadtime correction, forward/backward grouping onto a common t0,
optional background subtraction, then :func:`binned_fb_asymmetry` (counts summed
onto output bins *before* the asymmetry is formed — the order WiMDA, musrfit and
Mantid all use).

It was lifted out of ``MainWindow._reduce_grouped_histograms_to_asymmetry`` so
that the GUI reduction and the grouping window's live *preview* pane share one
implementation of the numerics rather than forking them. The GUI-specific pieces
that method still owned are passed in as plain values / a callback so this
function stays Qt-free:

* ``facility`` — the string the GUI reads off the run/dataset metadata.
* ``reference_resolver`` — resolves the ``reference_run`` background mode's
  reference histograms + frame-scale (the GUI supplies the loaded-dataset
  registry; the preview supplies a no-op that skips the subtraction).

The reduction is pinned bit-identical against the original MainWindow output by
``tests/gui/test_grouping_preview_pane.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    resolve_background_mode,
)
from asymmetry.core.transform.deadtime import (
    has_file_deadtime,
    has_resolved_deadtime,
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.grouping import (
    apply_grouping_aligned,
    common_t0_for_groups,
    detector_t0_overrides,
)
from asymmetry.core.transform.rebin import binned_fb_asymmetry

#: A background ``reference_run`` resolver: given the grouping dict, return the
#: reference run's histograms and the good-frame scale, or ``None`` to skip the
#: subtraction. The GUI passes one backed by its loaded-dataset registry; the
#: preview pane passes one that always returns ``None``.
ReferenceResolver = Callable[[dict[str, Any]], "tuple[list[Histogram], float] | None"]


@dataclass(frozen=True)
class GroupedAsymmetryReduction:
    """Result of :func:`reduce_grouped_asymmetry`.

    ``time`` is in µs relative to t0; ``asymmetry`` and ``error`` are already
    scaled to percent (×100), matching the arrays the plot panel consumes.
    ``deadtime_applied`` reports whether deadtime correction actually ran, and
    ``background_state`` mirrors the per-mode dict the MainWindow records back
    onto ``run.grouping`` (``None`` when no background correction was requested).
    """

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    deadtime_applied: bool
    background_state: dict[str, object] | None


def _prepare_grouping_histograms(
    histograms: list[Histogram], grouping: dict, use_deadtime: bool
) -> tuple[list[Histogram], bool]:
    """Return histograms prepared for grouping, with optional deadtime correction."""
    if not use_deadtime:
        return list(histograms), False
    return prepare_histograms_with_deadtime(histograms, grouping, use_deadtime)


def reduce_grouped_asymmetry(
    *,
    histograms: list[Histogram],
    grouping: dict,
    forward_idx: list[int],
    backward_idx: list[int],
    alpha: float,
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    facility: str = "",
    reference_resolver: ReferenceResolver | None = None,
) -> GroupedAsymmetryReduction:
    """Reduce grouped histograms to a forward/backward asymmetry curve.

    This is the reduction body extracted verbatim from
    ``MainWindow._reduce_grouped_histograms_to_asymmetry``; see the module
    docstring for what moved to parameters. The returned arrays are final — the
    good window is applied, the binning (fixed bunching included) is applied, and
    no further slicing or bunching is left for the caller.
    """
    effective_use_deadtime = bool(use_deadtime)
    if effective_use_deadtime:
        if deadtime_mode == "file":
            effective_use_deadtime = has_file_deadtime(grouping, len(histograms))
        else:
            effective_use_deadtime = has_resolved_deadtime(grouping, len(histograms))

    working_histograms, dt_applied = _prepare_grouping_histograms(
        histograms,
        grouping,
        effective_use_deadtime,
    )

    # A *manual* T0Policy carries effective per-detector t0 bins in the grouping
    # so alignment shifts without the histograms' own t0_bin being rewritten.
    detector_t0 = detector_t0_overrides(grouping, len(working_histograms))
    common_t0 = common_t0_for_groups(
        working_histograms, forward_idx, backward_idx, detector_t0_bins=detector_t0
    )
    forward = apply_grouping_aligned(
        working_histograms,
        forward_idx,
        common_t0_bin=common_t0,
        detector_t0_bins=detector_t0,
    )
    backward = apply_grouping_aligned(
        working_histograms,
        backward_idx,
        common_t0_bin=common_t0,
        detector_t0_bins=detector_t0,
    )
    n_grouped = min(len(forward), len(backward))
    forward = forward[:n_grouped]
    backward = backward[:n_grouped]

    background_state: dict[str, object] | None = None
    bkg_result = None
    if use_background:
        bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
        reference_forward = None
        reference_backward = None
        reference_scale = None
        if resolve_background_mode(grouping) == "reference_run" and reference_resolver is not None:
            resolved = reference_resolver(grouping)
            if resolved is not None:
                reference_histograms, reference_scale = resolved
                reference_prepared, _ = _prepare_grouping_histograms(
                    reference_histograms,
                    grouping,
                    effective_use_deadtime,
                )
                reference_forward = apply_grouping_aligned(
                    reference_prepared,
                    forward_idx,
                    common_t0_bin=common_t0,
                    detector_t0_bins=detector_t0_overrides(grouping, len(reference_prepared)),
                )
                reference_backward = apply_grouping_aligned(
                    reference_prepared,
                    backward_idx,
                    common_t0_bin=common_t0,
                    detector_t0_bins=detector_t0_overrides(grouping, len(reference_prepared)),
                )
        try:
            last_good = int(grouping.get("last_good_bin", n_grouped - 1))
        except (TypeError, ValueError):
            last_good = n_grouped - 1
        bkg_result = apply_grouped_background_correction(
            forward,
            backward,
            grouping=grouping,
            t0_bin=common_t0,
            bin_width_us=bin_width,
            facility=facility,
            last_good_bin=last_good,
            reference_forward=reference_forward,
            reference_backward=reference_backward,
            reference_scale=reference_scale,
        )
        forward = bkg_result.forward
        backward = bkg_result.backward
        background_state = {"method": bkg_result.method}
        if bkg_result.applied:
            if bkg_result.values is not None:
                background_state["values"] = [
                    float(bkg_result.values[0]),
                    float(bkg_result.values[1]),
                ]
            if bkg_result.ranges is not None:
                background_state["ranges"] = [
                    [int(v) for v in bkg_result.ranges[0]],
                    [int(v) for v in bkg_result.ranges[1]],
                ]
            if bkg_result.details is not None:
                background_state["details"] = dict(bkg_result.details)

    bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
    background_errors = (
        bkg_result is not None
        and bkg_result.applied
        and bkg_result.forward_error is not None
        and bkg_result.backward_error is not None
    )

    # Every binning mode (fixed included) bins the counts before forming
    # the asymmetry — the counts-then-ratio order all reference programs
    # use. The returned arrays are final: good window applied, bunching
    # applied, no further slicing or bunching by the caller.
    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", n_grouped - 1))
    except (TypeError, ValueError):
        last_good = n_grouped - 1
    time_axis, asymmetry, error = binned_fb_asymmetry(
        forward,
        backward,
        grouping=grouping,
        common_t0=common_t0,
        bin_width_us=bin_width,
        alpha=alpha,
        first_good_bin=first_good,
        last_good_bin=last_good,
        forward_error=bkg_result.forward_error if background_errors else None,
        backward_error=bkg_result.backward_error if background_errors else None,
    )
    return GroupedAsymmetryReduction(
        time=np.asarray(time_axis, dtype=np.float64),
        asymmetry=np.asarray(asymmetry * 100.0, dtype=np.float64),
        error=np.asarray(error * 100.0, dtype=np.float64),
        deadtime_applied=dt_applied,
        background_state=background_state,
    )


__all__ = [
    "GroupedAsymmetryReduction",
    "ReferenceResolver",
    "reduce_grouped_asymmetry",
]
