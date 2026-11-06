"""Helpers for WiMDA-style grouped detector Fourier inputs."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fourier.group_preparation import GroupPreparation, prepare_group_counts
from asymmetry.core.fourier.group_time import group_time_grid, time_range_mask
from asymmetry.core.transform.fb_counts import linear_combination_variance
from asymmetry.core.transform.grouping import (
    effective_group_indices,
)
from asymmetry.core.transform.rebin import binning_spec, rebin_counts
from asymmetry.core.transform.t0 import (
    require_selected_timing,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def build_group_signal_dataset(
    run: Run,
    group_id: int,
    *,
    center_signal: bool = True,
    apply_lifetime_correction: bool = True,
    use_deadtime: bool | None = None,
    reference_t0_bin: int | None = None,
    prepared: GroupPreparation | None = None,
    background_reference_cache: dict | None = None,
    time_range: tuple[float | None, float | None] | None = None,
) -> MuonDataset:
    """Build a time-domain grouped detector signal for Fourier analysis.

    This mirrors WiMDA's grouped-detector Fourier input more closely than the
    standard forward/backward asymmetry dataset used elsewhere in the GUI.
    The returned signal is centred by default so the FFT is not dominated by
    the zero-frequency component.

    WiMDA's grouped Fourier path uses lifetime-corrected detector counts as the
    FFT source. This helper mirrors that by default via ``exp(t / tau_mu)``
    before later mean subtraction and FFT filtering.
    """
    if time_range == (None, None):
        time_range = None
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        raise ValueError("Run does not define detector groups for Fourier analysis.")

    indices = effective_group_indices(grouping, group_id, n_histograms=len(run.histograms))
    histograms = list(run.histograms)
    require_selected_timing(grouping, indices, len(histograms))
    if not histograms:
        raise ValueError("Run does not contain any histograms.")

    apply_deadtime = bool(grouping.get("deadtime_correction", False))
    if use_deadtime is not None:
        apply_deadtime = bool(use_deadtime)
    if prepared is None:
        prepared = prepare_group_counts(
            run,
            [int(group_id)],
            use_deadtime=apply_deadtime,
            reference_t0_bin=reference_t0_bin,
            cache=background_reference_cache,
            time_range=time_range,
        )
    prepared.require_request(run, int(group_id), apply_deadtime, reference_t0_bin, time_range)
    layout = prepared.layouts[int(group_id)]
    alignment, window = layout.alignment, layout.window
    grouping = layout.grouping
    correction = prepared.backgrounds[int(group_id)]
    group = correction.forward
    deadtime = prepared.sample.outcome
    values = correction.outcome.effective.get("values") if correction.outcome.effective else None
    background_value = values[0] if values is not None else None

    counts = group.counts
    first_good, last_good = window.first, window.last

    trimmed_counts = np.asarray(counts[first_good : last_good + 1], dtype=np.float64)
    bunch_factor = binning_spec(grouping).factor

    time, edges = group_time_grid(histograms, grouping, indices, alignment, window)
    variance = linear_combination_variance(group, edges) if time.size else np.array([], dtype=float)
    if bunch_factor > 1:
        _, trimmed_counts = rebin_counts(
            np.arange(trimmed_counts.size), trimmed_counts, bunch_factor
        )
    mask = time_range_mask(time, time_range)
    time, trimmed_counts, variance = time[mask], trimmed_counts[mask], variance[mask]

    scale = np.ones_like(trimmed_counts, dtype=np.float64)
    if apply_lifetime_correction and trimmed_counts.size > 0:
        scale = np.exp(np.asarray(time, dtype=np.float64) / float(MUON_LIFETIME_US))

    signal = trimmed_counts * scale
    if center_signal and signal.size > 0:
        signal -= float(np.mean(signal))
    error = np.sqrt(np.clip(variance, 1.0, None)) * scale

    group_names = grouping.get("group_names") if isinstance(grouping, dict) else None
    group_name = None
    if isinstance(group_names, dict):
        group_name = group_names.get(group_id)
    if group_name is None:
        group_name = f"Group {group_id}"

    metadata = dict(run.metadata)
    metadata.update(
        {
            "run_number": run.run_number,
            "run_label": f"{run.run_number} {group_name}",
            "fourier_source": "group_signal",
            "deadtime_outcome": deadtime.to_dict(),
            "reference_deadtime_outcome": (
                prepared.reference.outcome.to_dict() if prepared.reference is not None else None
            ),
            "reference_file_identity": prepared.identity[0],
            "grouped_alignment": alignment.to_dict(),
            "fourier_lifetime_corrected": bool(apply_lifetime_correction),
            **prepared.correction_metadata(),
            "fourier_background_corrected": correction.outcome.status == "applied",
            "fourier_background_value": (
                float(background_value) if background_value is not None else None
            ),
            "group_id": int(group_id),
            "group_name": str(group_name),
        }
    )

    return MuonDataset(
        time=time,
        asymmetry=signal,
        error=error,
        metadata=metadata,
        run=run,
    )
