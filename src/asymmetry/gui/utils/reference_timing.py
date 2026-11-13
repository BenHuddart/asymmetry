"""Prepare reference searches off-thread before synchronous final correction."""

from dataclasses import replace
from functools import partial

from PySide6.QtCore import QThread

from asymmetry.core.fourier.group_preparation import histogram_grid_identity
from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.core.transform.grouping import independent_pair_indices
from asymmetry.core.transform.reference_timing import ResolvedReferenceTiming
from asymmetry.gui.tasks import check_task_cancelled
from asymmetry.gui.utils.reduction_cache import histogram_content_key
from asymmetry.gui.utils.scientific_input import ScientificInputDialog


def prepare_reference_timing(reference, grouping, parent, *, cache=None):
    """Resolve a fresh reference snapshot for this final request, without GUI scans."""
    if reference is None:
        return None
    pair = independent_pair_indices(grouping, n_histograms=len(reference.histograms))
    if not reference.timing.search_indices(sorted(set(pair[0]) | set(pair[1]))):
        return reference
    if QThread.currentThread() != parent.thread():
        return _resolve_reference_timing(reference, pair)

    def prepare():
        dialog = ScientificInputDialog(parent, "Resolving reference timing…", "Reference timing")
        try:
            prepared = dialog.run(partial(_resolve_reference_timing, reference, pair))
        finally:
            dialog.deleteLater()
        if prepared is None:
            raise CorrectionUnavailableError("Reference timing preparation cancelled")
        return prepared.resolved_timing

    if cache is None:
        return replace(reference, resolved_timing=prepare())
    indices = sorted(set(pair[0]) | set(pair[1]))
    content = histogram_content_key(
        [reference.histograms[i] for i in reference.timing.search_indices(indices)], parent
    )
    if content is None:
        raise CorrectionUnavailableError("Reference timing input check cancelled")
    key = (
        reference.source_file,
        reference.run_number,
        reference.period_number,
        reference.timing,
        histogram_grid_identity([reference.histograms[i] for i in indices]),
        content,
        tuple(indices),
    )
    resolution = cache.get_or_compute(
        reference.histograms[indices[0]],
        "reference_timing",
        key,
        prepare,
        lambda value: len(value.bins) * 16,
    )
    return replace(reference, resolved_timing=resolution)


def _resolve_reference_timing(reference, pair, worker=None):
    """Resolve selected source anchors on the cancellable worker boundary."""
    grouping = reference.timing.resolve(
        reference.histograms,
        pair,
        cancel_check=partial(check_task_cancelled, worker) if worker is not None else None,
    )
    return replace(
        reference,
        resolved_timing=ResolvedReferenceTiming.from_grouping(reference.histograms, grouping),
    )


def load_reference_acquisition(payload, *, parent):
    """Read disk-only reference counts on the existing cancellable input worker."""
    from copy import deepcopy

    from asymmetry.core.io import load_background_run

    if QThread.currentThread() != parent.thread():
        return load_background_run(payload)
    request = deepcopy(payload)
    dialog = ScientificInputDialog(parent, "Resolving reference timing…", "Reference timing")
    try:
        result = dialog.run(
            lambda worker: load_background_run(
                request, checkpoint=partial(check_task_cancelled, worker)
            )
        )
    finally:
        dialog.deleteLater()
    if result is None:
        raise CorrectionUnavailableError("Reference input preparation cancelled")
    return result
