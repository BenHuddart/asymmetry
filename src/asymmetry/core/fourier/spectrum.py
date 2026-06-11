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
from asymmetry.core.fourier.burg import burg_spectrum
from asymmetry.core.fourier.conditioning import apply_spectrum_conditioning
from asymmetry.core.fourier.diamag import fit_and_subtract_diamagnetic
from asymmetry.core.fourier.fft import (
    average_fourier_display_values,
    canonical_fourier_display_mode,
    exclude_frequency_ranges,
    fft_complex_asymmetry,
    fourier_display_values,
    fourier_mode_uses_entropy_optimizer,
    fourier_mode_uses_phase_correction,
    optimize_phase_entropy,
    prepare_fft_time_signal,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.units import gauss_to_mhz
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import common_t0_for_groups

#: Minimum applied field (Gauss) for a diamagnetic fit to be attempted.
_MIN_DIAMAG_FIELD_GAUSS = 5.0

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
    "real_imag": "FFT Real + Imag (a.u.)",
    "burg": "Burg AR spectrum (a.u.)",
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
    # ── post-FFT conditioning (frequency-domain finishers) ──────────────
    #: Pulse frequency-response compensation (divide by R(ν) from pulse.py).
    pulse_compensation: bool = False
    #: Pulse half-width in µs; ``0`` resolves from instrument metadata.
    pulse_half_width_us: float = 0.0
    pulse_separation_us: float = 0.0
    pulse_n_pulses: int = 1
    pulse_max_gain: float = 25.0
    #: Robust baseline offset mode: ``"none" | "sigma_clip" | "wimda"``.
    baseline_mode: str = "none"
    baseline_kappa: float = 2.0
    #: Frequency-range exclusions as ``(centre_mhz, half_width_mhz)`` pairs.
    exclude_enabled: bool = False
    exclusion_ranges: list[tuple[float, float]] = field(default_factory=list)
    #: When set, prepend a diamagnetic exclusion centred on the reference field.
    diamag_exclusion: bool = False
    diamag_half_width_mhz: float = 0.3
    #: Burg all-poles pole-scan range (diagnostic "Resolution (Burg)" mode).
    burg_order_min: int = 2
    burg_order_max: int = 40
    #: Fit and subtract the diamagnetic line in the time domain before the FFT.
    remove_diamag: bool = False

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
            "group_phase_degrees": {int(k): float(v) for k, v in self.group_phase_degrees.items()},
            "pulse_compensation": self.pulse_compensation,
            "pulse_half_width_us": self.pulse_half_width_us,
            "pulse_separation_us": self.pulse_separation_us,
            "pulse_n_pulses": self.pulse_n_pulses,
            "pulse_max_gain": self.pulse_max_gain,
            "baseline_mode": self.baseline_mode,
            "baseline_kappa": self.baseline_kappa,
            "exclude_enabled": self.exclude_enabled,
            "exclusion_ranges": [[float(c), float(w)] for c, w in self.exclusion_ranges],
            "diamag_exclusion": self.diamag_exclusion,
            "diamag_half_width_mhz": self.diamag_half_width_mhz,
            "burg_order_min": self.burg_order_min,
            "burg_order_max": self.burg_order_max,
            "remove_diamag": self.remove_diamag,
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
            selected_group_ids=[int(g) for g in selected] if isinstance(selected, list) else None,
            group_phase_degrees=group_phase_degrees,
            pulse_compensation=bool(data.get("pulse_compensation", False)),
            pulse_half_width_us=float(data.get("pulse_half_width_us", 0.0)),
            pulse_separation_us=float(data.get("pulse_separation_us", 0.0)),
            pulse_n_pulses=int(data.get("pulse_n_pulses", 1)),
            pulse_max_gain=float(data.get("pulse_max_gain", 25.0)),
            baseline_mode=str(data.get("baseline_mode", "none")),
            baseline_kappa=float(data.get("baseline_kappa", 2.0)),
            exclude_enabled=bool(data.get("exclude_enabled", False)),
            exclusion_ranges=_coerce_exclusion_ranges(data.get("exclusion_ranges")),
            diamag_exclusion=bool(data.get("diamag_exclusion", False)),
            diamag_half_width_mhz=float(data.get("diamag_half_width_mhz", 0.3)),
            burg_order_min=int(data.get("burg_order_min", 2)),
            burg_order_max=int(data.get("burg_order_max", 40)),
            remove_diamag=bool(data.get("remove_diamag", False)),
        )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_exclusion_ranges(value: object) -> list[tuple[float, float]]:
    """Parse serialised ``[[centre, half_width], ...]`` exclusion ranges."""
    if not isinstance(value, (list, tuple)):
        return []
    ranges: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                ranges.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return ranges


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
    prepared_histograms, _ = prepare_histograms_with_deadtime(histograms, grouping, apply_deadtime)

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


def _reference_field_gauss(run: Run, dataset: MuonDataset | None) -> float | None:
    """Return the run's applied field in Gauss, or ``None``."""
    sources: list[dict] = []
    if dataset is not None and isinstance(dataset.metadata, dict):
        sources.append(dataset.metadata)
    run_metadata = getattr(run, "metadata", None)
    if isinstance(run_metadata, dict):
        sources.append(run_metadata)
    for metadata in sources:
        try:
            return float(metadata["field"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _reference_frequency_mhz(run: Run, dataset: MuonDataset | None) -> float | None:
    """Return the muon Larmor frequency (MHz) at the applied field, or ``None``."""
    field_gauss = _reference_field_gauss(run, dataset)
    if field_gauss is None:
        return None
    return float(gauss_to_mhz(field_gauss))


def _resolve_exclusion_ranges(
    config: GroupSpectrumConfig, reference_mhz: float | None
) -> list[tuple[float, float]]:
    """Return the active exclusion ranges, prepending the diamag slot if asked.

    The diamagnetic slot and the manual ranges are independent: either can be
    active on its own (the GUI happens to gate them together, but a programmatic
    config need not).
    """
    if not config.exclude_enabled and not config.diamag_exclusion:
        return []
    ranges: list[tuple[float, float]] = []
    if config.diamag_exclusion and reference_mhz is not None:
        ranges.append((float(reference_mhz), float(config.diamag_half_width_mhz)))
    if config.exclude_enabled:
        ranges.extend((float(c), float(w)) for c, w in config.exclusion_ranges)
    return ranges


def _metadata_pulse_half_width_us(run: Run, dataset: MuonDataset | None) -> float:
    """Resolve a pulse half-width (µs) from instrument metadata, else 0."""
    sources: list[dict] = []
    if dataset is not None and isinstance(dataset.metadata, dict):
        sources.append(dataset.metadata)
    run_metadata = getattr(run, "metadata", None)
    if isinstance(run_metadata, dict):
        sources.append(run_metadata)
    for metadata in sources:
        for key in ("pulse_half_width_us", "pulse_width_us", "pulse_fwhm_us"):
            try:
                value = float(metadata[key])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0.0:
                # FWHM keys are stored as a full width; halve to a half-width.
                return value / 2.0 if key == "pulse_fwhm_us" else value
    return 0.0


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
    canonical = canonical_fourier_display_mode(display)
    is_real_imag = canonical == "real_imag"
    is_burg = canonical == "burg"

    if prepared_histograms is None or reference_t0_bin is None:
        prepared_histograms, reference_t0_bin = precompute_group_fourier_inputs(run)

    averaged_values: list[np.ndarray] = []
    imag_values: list[np.ndarray] = []
    complex_spectra: list[np.ndarray] = []
    burg_orders: list[int] = []
    burg_hit_boundary = False
    diamag_fields: list[float] = []
    diamag_fit_curve: tuple[np.ndarray, np.ndarray] | None = None
    average_freqs: np.ndarray | None = None
    first_group_dataset: MuonDataset | None = None
    selected_names: list[str] = []
    # Shared across the sweep so a reference_run background loads + deadtime-
    # prepares once, not once per group.
    background_reference_cache: dict = {}

    for group_id in selected:
        group_dataset = build_group_signal_dataset(
            run,
            group_id,
            center_signal=False,
            reference_t0_bin=reference_t0_bin,
            prepared_histograms=prepared_histograms,
            background_reference_cache=background_reference_cache,
        )
        if config.remove_diamag:
            seed_field = _reference_field_gauss(run, group_dataset)
            # Diamagnetic precession only exists in a transverse field; skip the
            # fit at (near-)zero field, where a bounded fit would otherwise lock
            # onto a spurious low frequency.
            if seed_field is not None and abs(seed_field) > _MIN_DIAMAG_FIELD_GAUSS:
                group_dataset, diamag_fit = fit_and_subtract_diamagnetic(
                    group_dataset, seed_field_gauss=seed_field
                )
                if diamag_fit.success:
                    diamag_fields.append(diamag_fit.field_gauss)
                    if diamag_fit_curve is None:
                        diamag_fit_curve = (
                            np.asarray(group_dataset.time, dtype=float),
                            diamag_fit.model(np.asarray(group_dataset.time, dtype=float)),
                        )

        if first_group_dataset is None:
            first_group_dataset = group_dataset
        selected_names.append(group_names.get(group_id, f"Group {group_id}"))

        if is_burg:
            signal, dt = prepare_fft_time_signal(
                group_dataset,
                window=config.window,
                t_min=config.t_min_us,
                t_max=config.t_max_us,
                subtract_average_signal=config.subtract_average_signal,
                filter_start_us=config.filter_start_us,
                filter_time_constant_us=config.filter_time_constant_us,
            )
            n_padded = len(signal) * max(config.padding, 1)
            burg_freqs = np.fft.rfftfreq(n_padded, d=dt)
            spec, best_order, hit_boundary = burg_spectrum(
                signal,
                burg_freqs,
                dt,
                order_range=(config.burg_order_min, config.burg_order_max),
            )
            if average_freqs is None:
                average_freqs = burg_freqs
            averaged_values.append(spec)
            burg_orders.append(best_order)
            burg_hit_boundary = burg_hit_boundary or hit_boundary
            continue

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
        elif is_real_imag:
            averaged_values.append(fourier_display_values(spectrum, display="Cos"))
            imag_values.append(fourier_display_values(spectrum, display="Sin"))
        else:
            averaged_values.append(fourier_display_values(spectrum, display=display))

    if average_freqs is None:
        return None

    averaged_imag: np.ndarray | None = None
    if is_entropy and complex_spectra:
        avg_complex = np.mean(np.vstack([s[np.newaxis, :] for s in complex_spectra]), axis=0)
        averaged_display, _c0, _c1 = optimize_phase_entropy(avg_complex)
        averaged_error = np.zeros_like(averaged_display)
    elif averaged_values:
        averaged_display, averaged_error = average_fourier_display_values(
            averaged_values,
            estimate_error=config.estimate_average_error,
        )
        if is_real_imag and imag_values:
            averaged_imag, _ = average_fourier_display_values(imag_values, estimate_error=False)
    else:
        return None

    # ── post-FFT conditioning (compensation → baseline → exclusions) ────
    # Operates on the canonical MHz axis; the plot panel converts to the
    # display unit.  Entropy keeps its optimiser output untouched, and Burg is a
    # raw all-poles diagnostic (it already inherits the time-domain preprocessing
    # chain) — neither is post-conditioned.
    conditioning = None
    if not is_entropy and not is_burg:
        reference_mhz = _reference_frequency_mhz(run, first_group_dataset)
        exclusion_ranges = _resolve_exclusion_ranges(config, reference_mhz)
        pulse_half_width_us = config.pulse_half_width_us
        if config.pulse_compensation and pulse_half_width_us <= 0.0:
            pulse_half_width_us = _metadata_pulse_half_width_us(run, first_group_dataset)
        conditioning = apply_spectrum_conditioning(
            average_freqs,
            averaged_display,
            averaged_error,
            pulse_compensation=config.pulse_compensation,
            pulse_half_width_us=pulse_half_width_us,
            pulse_separation_us=config.pulse_separation_us,
            pulse_n_pulses=config.pulse_n_pulses,
            pulse_max_gain=config.pulse_max_gain,
            baseline_mode=config.baseline_mode,
            baseline_kappa=config.baseline_kappa,
            exclusion_ranges=exclusion_ranges,
        )
        averaged_display = conditioning.display
        averaged_error = conditioning.error
        if averaged_imag is not None:
            # Keep the imaginary quadrature on the same footing as the real
            # channel so the Real+Imag overlay shares one zero reference.
            if conditioning.gain is not None:
                averaged_imag = averaged_imag * conditioning.gain
            if conditioning.baseline:
                averaged_imag = averaged_imag - conditioning.baseline
            if exclusion_ranges:
                averaged_imag = exclude_frequency_ranges(
                    average_freqs, averaged_imag, exclusion_ranges
                )

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
    if averaged_imag is not None:
        metadata["fourier_imag"] = np.asarray(averaged_imag, dtype=float).tolist()
    if is_burg and burg_orders:
        metadata["fourier_burg_order"] = int(max(burg_orders))
        metadata["fourier_burg_hit_boundary"] = bool(burg_hit_boundary)
        metadata["fourier_diagnostic"] = True
    if config.remove_diamag and diamag_fields:
        metadata["fourier_diamag_field_gauss"] = float(np.mean(diamag_fields))
        if diamag_fit_curve is not None:
            metadata["fourier_diamag_fit_time_us"] = diamag_fit_curve[0].tolist()
            metadata["fourier_diamag_fit_signal"] = diamag_fit_curve[1].tolist()
    if conditioning is not None:
        if conditioning.cutoff_frequency_mhz is not None:
            metadata["fourier_compensation_cutoff_mhz"] = conditioning.cutoff_frequency_mhz
        if config.baseline_mode in {"sigma_clip", "wimda"}:
            metadata["fourier_baseline"] = conditioning.baseline
            metadata["fourier_baseline_noise"] = conditioning.noise_sigma
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
