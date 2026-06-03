"""Averaged grouped-FFT spectrum generation (shared core).

This is the single implementation of "turn a grouped run + a Fourier
configuration into one averaged frequency spectrum".  Both the GUI Fourier
panel and the :class:`~asymmetry.core.representation.frequency.FrequencyFFT`
representation call :func:`compute_average_group_spectrum`, so a generated
spectrum and a recipe-recomputed spectrum are identical by construction.

The configuration carries **concrete** values only.  Automatic phase estimation
(which depends on transient GUI view state) is resolved by the caller into
per-group ``group_phase_degrees`` before calling here, so recompute-on-load is
deterministic and faithful to what was originally generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.fft import (
    average_fourier_display_values,
    canonical_fourier_display_mode,
    fft_complex_asymmetry,
    fourier_display_values,
    fourier_mode_uses_entropy_optimizer,
    fourier_mode_uses_phase_correction,
    optimize_phase_entropy,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import common_t0_for_groups

_YLABELS: dict[str, str] = {
    "cos": "FFT Cos (a.u.)",
    "imaginary": "FFT Imaginary (a.u.)",
    "magnitude": "FFT Magnitude (a.u.)",
    "phase_corrected": "FFT Phase-Corrected (a.u.)",
    "phase_opt_real": "FFT phaseOptReal (a.u.)",
    "phase_spectrum": "FFT Phase Spectrum (deg)",
    "power": "FFT Power (a.u.)",
    "power_sqrt": "FFT (Power)^1/2 (a.u.)",
    "real": "FFT Real (a.u.)",
    "sin": "FFT Sin (a.u.)",
}


def fourier_display_ylabel(display: str) -> str:
    """Return the y-axis label for a Fourier display mode."""
    return _YLABELS.get(canonical_fourier_display_mode(display), "FFT (a.u.)")


@dataclass
class GroupSpectrumConfig:
    """Concrete configuration for one averaged grouped-FFT spectrum."""

    display: str = "(Power)^1/2"
    window: str = "none"
    padding: int = 1
    filter_start_us: float = 0.0
    filter_time_constant_us: float = 1.5
    t0_offset_us: float = 0.0
    subtract_average_signal: bool = True
    estimate_average_error: bool = False
    t_min_us: float | None = None
    t_max_us: float | None = None
    #: Group ids to include; ``None`` means all groups in the run.
    selected_group_ids: list[int] | None = None
    #: Resolved per-group phase correction in degrees (used only for
    #: phase-correcting display modes).  Missing groups default to 0.
    group_phase_degrees: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable ``fourier_config`` recipe block."""
        return {
            "display": self.display,
            "window": self.window,
            "padding": self.padding,
            "filter_start_us": self.filter_start_us,
            "filter_time_constant_us": self.filter_time_constant_us,
            "t0_offset_us": self.t0_offset_us,
            "subtract_average_signal": self.subtract_average_signal,
            "estimate_average_error": self.estimate_average_error,
            "t_min_us": self.t_min_us,
            "t_max_us": self.t_max_us,
            "selected_group_ids": (
                None if self.selected_group_ids is None else list(self.selected_group_ids)
            ),
            "group_phase_degrees": {
                int(k): float(v) for k, v in self.group_phase_degrees.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> GroupSpectrumConfig:
        """Build a config from a serialised ``fourier_config`` recipe block."""
        data = data if isinstance(data, dict) else {}
        selected = data.get("selected_group_ids")
        phases_raw = data.get("group_phase_degrees")
        group_phase_degrees = (
            {int(k): float(v) for k, v in phases_raw.items()}
            if isinstance(phases_raw, dict)
            else {}
        )
        return cls(
            display=str(data.get("display", "(Power)^1/2")),
            window=str(data.get("window", "none")),
            padding=max(1, int(data.get("padding", 1))),
            filter_start_us=float(data.get("filter_start_us", 0.0)),
            filter_time_constant_us=float(data.get("filter_time_constant_us", 1.5)),
            t0_offset_us=float(data.get("t0_offset_us", 0.0)),
            subtract_average_signal=bool(data.get("subtract_average_signal", True)),
            estimate_average_error=bool(data.get("estimate_average_error", False)),
            t_min_us=_optional_float(data.get("t_min_us")),
            t_max_us=_optional_float(data.get("t_max_us")),
            selected_group_ids=(
                [int(g) for g in selected] if isinstance(selected, list) else None
            ),
            group_phase_degrees=group_phase_degrees,
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def precompute_group_fourier_inputs(
    run: Run,
) -> tuple[list[Histogram] | None, int | None]:
    """Prepare deadtime-corrected histograms and a shared reference t0 once.

    Mirrors the GUI's per-run precompute so every group FFT shares the same
    inputs.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    histograms = list(run.histograms)
    if not isinstance(groups, dict) or not histograms:
        return None, None

    apply_deadtime = bool(grouping.get("deadtime_correction", False))
    prepared_histograms, _ = prepare_histograms_with_deadtime(
        histograms, grouping, apply_deadtime
    )

    all_group_indices: list[list[int]] = []
    for values in groups.values():
        if not isinstance(values, list):
            continue
        normalized: list[int] = []
        for value in values:
            detector = value[0] if isinstance(value, (list, tuple)) and value else value
            try:
                normalized.append(max(0, int(detector) - 1))
            except (TypeError, ValueError):
                continue
        if normalized:
            all_group_indices.append(normalized)

    reference_t0_bin = 0
    if all_group_indices:
        reference_t0_bin = common_t0_for_groups(prepared_histograms, *all_group_indices)
    return prepared_histograms, int(reference_t0_bin)


def _group_names(run: Run) -> dict[int, str]:
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        return {}
    raw_names = grouping.get("group_names")
    names = raw_names if isinstance(raw_names, dict) else {}
    resolved: dict[int, str] = {}
    for raw_id in groups:
        try:
            gid = int(raw_id)
        except (TypeError, ValueError):
            continue
        name = names.get(gid, names.get(str(gid)))
        resolved[gid] = str(name) if name is not None else f"Group {gid}"
    return resolved


def compute_average_group_spectrum(
    run: Run,
    config: GroupSpectrumConfig,
    *,
    prepared_histograms: list[Histogram] | None = None,
    reference_t0_bin: int | None = None,
) -> MuonDataset | None:
    """Return the averaged grouped-FFT spectrum for *run*, or ``None``.

    Reproduces the GUI's averaged grouped-FFT pipeline: build each selected
    group's lifetime-corrected signal, FFT it (with the resolved per-group
    phase), and average the display channels (or, for entropy mode, average the
    complex spectra then run the entropy optimiser).
    """
    group_names = _group_names(run)
    if not group_names:
        return None

    all_ids = sorted(group_names)
    if config.selected_group_ids is None:
        selected = list(all_ids)
    else:
        wanted = {int(g) for g in config.selected_group_ids}
        selected = [gid for gid in all_ids if gid in wanted]
    if not selected:
        return None

    display = config.display
    apply_phase = fourier_mode_uses_phase_correction(display)
    is_entropy = fourier_mode_uses_entropy_optimizer(display)

    if prepared_histograms is None or reference_t0_bin is None:
        prepared_histograms, reference_t0_bin = precompute_group_fourier_inputs(run)

    averaged_values: list[np.ndarray] = []
    complex_spectra: list[np.ndarray] = []
    average_freqs: np.ndarray | None = None
    first_group_dataset: MuonDataset | None = None
    selected_names: list[str] = []

    for group_id in selected:
        group_dataset = build_group_signal_dataset(
            run,
            group_id,
            center_signal=False,
            reference_t0_bin=reference_t0_bin,
            prepared_histograms=prepared_histograms,
        )
        if first_group_dataset is None:
            first_group_dataset = group_dataset
        selected_names.append(group_names.get(group_id, f"Group {group_id}"))

        phase_degrees = 0.0
        group_t0_offset_us = 0.0
        if apply_phase:
            phase_degrees = float(config.group_phase_degrees.get(group_id, 0.0))
            group_t0_offset_us = config.t0_offset_us

        freqs, spectrum = fft_complex_asymmetry(
            group_dataset,
            window=config.window,
            padding_factor=config.padding,
            t_min=config.t_min_us,
            t_max=config.t_max_us,
            phase_degrees=phase_degrees,
            t0_offset_us=group_t0_offset_us,
            subtract_average_signal=config.subtract_average_signal,
            filter_start_us=config.filter_start_us,
            filter_time_constant_us=config.filter_time_constant_us,
        )
        if average_freqs is None:
            average_freqs = freqs
        if is_entropy:
            complex_spectra.append(spectrum)
        else:
            averaged_values.append(fourier_display_values(spectrum, display=display))

    if average_freqs is None:
        return None

    if is_entropy and complex_spectra:
        avg_complex = np.mean(np.vstack([s[np.newaxis, :] for s in complex_spectra]), axis=0)
        averaged_display, _c0, _c1 = optimize_phase_entropy(avg_complex)
        averaged_error = np.zeros_like(averaged_display)
    elif averaged_values:
        averaged_display, averaged_error = average_fourier_display_values(
            averaged_values,
            estimate_error=config.estimate_average_error,
        )
    else:
        return None

    if len(selected) == len(all_ids):
        run_label = f"{run.run_number} Average"
    else:
        run_label = f"{run.run_number} Average ({', '.join(selected_names)})"

    source_metadata = dict(first_group_dataset.metadata) if first_group_dataset else {}
    metadata = dict(source_metadata)
    metadata.update(
        {
            "run_number": run.run_number,
            "run_label": run_label,
            "plot_domain": "frequency",
            "x_label": "Frequency (MHz)",
            "y_label": fourier_display_ylabel(display),
            "fourier_display": str(display),
            "fourier_group_output": "average",
            "group_ids": list(selected),
        }
    )
    return MuonDataset(
        time=np.asarray(average_freqs, dtype=float),
        asymmetry=np.asarray(averaged_display, dtype=float),
        error=np.asarray(averaged_error, dtype=float),
        metadata=metadata,
        run=run,
    )


__all__ = [
    "GroupSpectrumConfig",
    "compute_average_group_spectrum",
    "fourier_display_ylabel",
    "precompute_group_fourier_inputs",
]
