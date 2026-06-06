"""Grouped-count maximum-entropy spectral estimation.

This module owns the scriptable MaxEnt API.  It follows the MULTIMAX data
shape: one non-negative frequency spectrum is reconstructed jointly from many
detector-group count signals with group phases, amplitudes, and backgrounds.

The numerical kernel is deliberately compact and isolated.  It uses the same
forward/adjoint contract and resumable state shape needed by the full
Skilling-Bryan port, while providing a deterministic entropy-regularized
projected-gradient V1 implementation for Asymmetry's GUI and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

_MAX_SPECTRUM_POINTS = 1 << 20
_MIN_POSITIVE = 1.0e-15
_MAX_DESIGN_CHUNK_ELEMENTS = 2_000_000

MaxEntProgressCallback = Callable[[int, int, str], None]
MaxEntCancelCallback = Callable[[], bool]


class MaxEntCancelledError(RuntimeError):
    """Raised when a MaxEnt calculation is cancelled by the caller."""


def default_n_spectrum_points(n_time_points: int) -> int:
    """Return Mantid-style power-of-two spectrum length for *n_time_points*."""
    points = 1
    required = max(1, int(n_time_points))
    while points < required:
        points *= 2
    return min(points, _MAX_SPECTRUM_POINTS)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _field_to_frequency_mhz(field_gauss: float) -> float:
    return float(field_gauss) * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA


@dataclass
class MaxEntConfig:
    """Concrete configuration for one grouped MaxEnt reconstruction."""

    n_spectrum_points: int | None = None
    default_level: float = 0.01
    f_min_mhz: float | None = None
    f_max_mhz: float | None = None
    auto_window: bool = True
    window_half_width_gauss: float = 300.0
    outer_cycles: int = 10
    inner_iterations: int = 12
    chi2_target_over_n: float = 1.0
    fit_phases: bool = True
    fit_amplitudes: bool = True
    fit_backgrounds: bool = True
    fit_constant_background: bool = True
    use_deadtime_correction: bool = True
    selected_group_ids: list[int] | None = None
    group_phase_degrees: dict[int, float] = field(default_factory=dict)
    t_min_us: float | None = None
    t_max_us: float | None = None
    time_binning_factor: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable recipe block."""
        return {
            "n_spectrum_points": self.n_spectrum_points,
            "default_level": float(self.default_level),
            "f_min_mhz": self.f_min_mhz,
            "f_max_mhz": self.f_max_mhz,
            "auto_window": bool(self.auto_window),
            "window_half_width_gauss": float(self.window_half_width_gauss),
            "outer_cycles": int(self.outer_cycles),
            "inner_iterations": int(self.inner_iterations),
            "chi2_target_over_n": float(self.chi2_target_over_n),
            "fit_phases": bool(self.fit_phases),
            "fit_amplitudes": bool(self.fit_amplitudes),
            "fit_backgrounds": bool(self.fit_backgrounds),
            "fit_constant_background": bool(self.fit_constant_background),
            "use_deadtime_correction": bool(self.use_deadtime_correction),
            "selected_group_ids": (
                None if self.selected_group_ids is None else list(self.selected_group_ids)
            ),
            "group_phase_degrees": {int(k): float(v) for k, v in self.group_phase_degrees.items()},
            "t_min_us": self.t_min_us,
            "t_max_us": self.t_max_us,
            "time_binning_factor": int(self.time_binning_factor),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> MaxEntConfig:
        """Build a config from a serialised recipe block."""
        data = data if isinstance(data, dict) else {}
        selected = data.get("selected_group_ids")
        phases = data.get("group_phase_degrees")
        n_points = data.get("n_spectrum_points")
        try:
            parsed_points = None if n_points is None else max(8, int(n_points))
        except (TypeError, ValueError):
            parsed_points = None
        return cls(
            n_spectrum_points=parsed_points,
            default_level=max(_MIN_POSITIVE, float(data.get("default_level", 0.01))),
            f_min_mhz=_optional_float(data.get("f_min_mhz")),
            f_max_mhz=_optional_float(data.get("f_max_mhz")),
            auto_window=bool(data.get("auto_window", True)),
            window_half_width_gauss=max(0.0, float(data.get("window_half_width_gauss", 300.0))),
            outer_cycles=max(1, int(data.get("outer_cycles", 10))),
            inner_iterations=max(1, int(data.get("inner_iterations", 12))),
            chi2_target_over_n=max(_MIN_POSITIVE, float(data.get("chi2_target_over_n", 1.0))),
            fit_phases=bool(data.get("fit_phases", True)),
            fit_amplitudes=bool(data.get("fit_amplitudes", True)),
            fit_backgrounds=bool(data.get("fit_backgrounds", True)),
            fit_constant_background=bool(data.get("fit_constant_background", True)),
            use_deadtime_correction=bool(data.get("use_deadtime_correction", True)),
            selected_group_ids=[int(g) for g in selected] if isinstance(selected, list) else None,
            group_phase_degrees=(
                {int(k): float(v) for k, v in phases.items()} if isinstance(phases, dict) else {}
            ),
            t_min_us=_optional_float(data.get("t_min_us")),
            t_max_us=_optional_float(data.get("t_max_us")),
            time_binning_factor=_parse_positive_int(data.get("time_binning_factor", 1)),
        )


@dataclass(frozen=True)
class MaxEntGroupInput:
    """One detector-group signal prepared for MaxEnt."""

    group_id: int
    group_name: str
    time_us: NDArray[np.float64]
    signal: NDArray[np.float64]
    sigma: NDArray[np.float64]
    phase_degrees: float = 0.0
    amplitude: float = 1.0
    background: float = 0.0
    mask: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class MaxEntInput:
    """Prepared raw-count input for joint MaxEnt reconstruction."""

    run_number: int
    groups: tuple[MaxEntGroupInput, ...]
    n_spectrum_points: int
    f_min_mhz: float
    f_max_mhz: float
    default_level: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def frequencies_mhz(self) -> NDArray[np.float64]:
        # Cached: this grid is requested in every projection of the iteration
        # hot loop and never changes after construction.
        return np.linspace(
            float(self.f_min_mhz), float(self.f_max_mhz), int(self.n_spectrum_points)
        )


@dataclass(frozen=True)
class MaxEntWorkloadEstimate:
    """Approximate MaxEnt compute and memory footprint for a configuration."""

    run_number: int
    selected_group_count: int
    time_points_per_group: tuple[int, ...]
    n_spectrum_points: int
    peak_dense_matrix_bytes: int
    total_dense_matrix_bytes: int

    @property
    def max_time_points(self) -> int:
        return max(self.time_points_per_group, default=0)

    @property
    def total_observations(self) -> int:
        return int(sum(self.time_points_per_group))


@dataclass
class MaxEntDiagnostics:
    """Per-cycle convergence diagnostics."""

    cycles: list[int] = field(default_factory=list)
    chi2: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    test: list[float] = field(default_factory=list)
    sconv: list[float] = field(default_factory=list)
    phases: list[dict[int, float]] = field(default_factory=list)
    amplitudes: list[dict[int, float]] = field(default_factory=list)
    backgrounds: list[dict[int, float]] = field(default_factory=list)

    def append(
        self,
        *,
        cycle: int,
        chi2: float,
        entropy: float,
        test: float,
        sconv: float,
        phases: dict[int, float],
        amplitudes: dict[int, float],
        backgrounds: dict[int, float],
    ) -> None:
        self.cycles.append(int(cycle))
        self.chi2.append(float(chi2))
        self.entropy.append(float(entropy))
        self.test.append(float(test))
        self.sconv.append(float(sconv))
        self.phases.append({int(k): float(v) for k, v in phases.items()})
        self.amplitudes.append({int(k): float(v) for k, v in amplitudes.items()})
        self.backgrounds.append({int(k): float(v) for k, v in backgrounds.items()})

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serialisable diagnostics payload."""
        return {
            "cycles": list(self.cycles),
            "chi2": list(self.chi2),
            "entropy": list(self.entropy),
            "test": list(self.test),
            "sconv": list(self.sconv),
            "phases": [dict(row) for row in self.phases],
            "amplitudes": [dict(row) for row in self.amplitudes],
            "backgrounds": [dict(row) for row in self.backgrounds],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> MaxEntDiagnostics:
        """Build diagnostics from a serialised payload."""
        diag = cls()
        if not isinstance(data, dict):
            return diag
        diag.cycles = [int(v) for v in data.get("cycles", [])]
        diag.chi2 = [float(v) for v in data.get("chi2", [])]
        diag.entropy = [float(v) for v in data.get("entropy", [])]
        diag.test = [float(v) for v in data.get("test", [])]
        diag.sconv = [float(v) for v in data.get("sconv", [])]
        diag.phases = [
            {int(k): float(v) for k, v in row.items()}
            for row in data.get("phases", [])
            if isinstance(row, dict)
        ]
        diag.amplitudes = [
            {int(k): float(v) for k, v in row.items()}
            for row in data.get("amplitudes", [])
            if isinstance(row, dict)
        ]
        diag.backgrounds = [
            {int(k): float(v) for k, v in row.items()}
            for row in data.get("backgrounds", [])
            if isinstance(row, dict)
        ]
        return diag


@dataclass
class MaxEntState:
    """Resumable MaxEnt iteration state."""

    frequencies_mhz: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    phases: dict[int, float]
    amplitudes: dict[int, float]
    backgrounds: dict[int, float]
    cycle: int = 0
    diagnostics: MaxEntDiagnostics = field(default_factory=MaxEntDiagnostics)
    signature: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MaxEntResult:
    """Completed or partially-completed MaxEnt reconstruction."""

    frequencies_mhz: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    state: MaxEntState
    diagnostics: MaxEntDiagnostics
    metadata: dict[str, Any]

    def as_dataset(self, run: Run | None = None) -> MuonDataset:
        """Return the primary MaxEnt spectrum as a plottable dataset."""
        error = np.zeros_like(self.spectrum, dtype=float)
        return MuonDataset(
            time=np.asarray(self.frequencies_mhz, dtype=float),
            asymmetry=np.asarray(self.spectrum, dtype=float),
            error=error,
            metadata=dict(self.metadata),
            run=run,
        )


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
        resolved[gid] = str(names.get(gid, names.get(str(gid), f"Group {gid}")))
    return resolved


def _parse_positive_int(value: object, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _run_with_maxent_binning(run: Run, config: MaxEntConfig) -> Run:
    """Return a shallow run copy with MaxEnt-only bunching applied."""
    factor = _parse_positive_int(config.time_binning_factor)
    if factor <= 1:
        return run
    grouping = dict(run.grouping) if isinstance(run.grouping, dict) else {}
    existing = _parse_positive_int(grouping.get("bunching_factor", 1))
    grouping["bunching_factor"] = int(existing * factor)
    return replace(run, grouping=grouping)


def estimate_maxent_workload(
    run: Run,
    config: MaxEntConfig | dict | None = None,
) -> MaxEntWorkloadEstimate:
    """Estimate dense-matrix work for a MaxEnt run without building matrices."""
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    group_names = _group_names(run)
    all_ids = sorted(group_names)
    if resolved_config.selected_group_ids is None:
        selected = all_ids
    else:
        wanted = {int(g) for g in resolved_config.selected_group_ids}
        selected = [gid for gid in all_ids if gid in wanted]

    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    histograms = list(run.histograms)
    if not histograms or not selected:
        return MaxEntWorkloadEstimate(
            run_number=int(run.run_number),
            selected_group_count=len(selected),
            time_points_per_group=tuple(),
            n_spectrum_points=int(resolved_config.n_spectrum_points or 0),
            peak_dense_matrix_bytes=0,
            total_dense_matrix_bytes=0,
        )

    n_bins = min(hist.n_bins for hist in histograms)
    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = min(n_bins - 1, int(grouping.get("last_good_bin", n_bins - 1)))
    except (TypeError, ValueError):
        last_good = n_bins - 1
    if first_good > last_good:
        first_good = 0
        last_good = n_bins - 1

    reference_t0 = grouping.get("t0_bin", histograms[0].t0_bin)
    try:
        reference_t0 = int(reference_t0)
    except (TypeError, ValueError):
        reference_t0 = int(histograms[0].t0_bin)
    bin_width = float(histograms[0].bin_width)
    bins = np.arange(first_good, last_good + 1, dtype=np.float64)
    time_us = (bins - float(reference_t0)) * bin_width
    mask = np.ones(time_us.size, dtype=bool)
    if resolved_config.t_min_us is not None:
        mask &= time_us >= float(resolved_config.t_min_us)
    if resolved_config.t_max_us is not None:
        mask &= time_us <= float(resolved_config.t_max_us)
    usable = int(np.count_nonzero(mask))
    base_bunch = _parse_positive_int(grouping.get("bunching_factor", 1))
    maxent_bunch = _parse_positive_int(resolved_config.time_binning_factor)
    effective_bunch = max(1, base_bunch * maxent_bunch)
    points = usable // effective_bunch if effective_bunch > 1 else usable
    time_points = tuple(int(points) for _group_id in selected)
    n_spectrum_points = int(
        resolved_config.n_spectrum_points or default_n_spectrum_points(max(time_points, default=0))
    )
    matrix_bytes = tuple(int(n * n_spectrum_points * 8) for n in time_points)
    return MaxEntWorkloadEstimate(
        run_number=int(run.run_number),
        selected_group_count=len(selected),
        time_points_per_group=time_points,
        n_spectrum_points=n_spectrum_points,
        peak_dense_matrix_bytes=max(matrix_bytes, default=0),
        total_dense_matrix_bytes=sum(matrix_bytes),
    )


def _resolve_frequency_window(run: Run, config: MaxEntConfig) -> tuple[float, float]:
    field_value = run.metadata.get("field") if isinstance(run.metadata, dict) else None
    try:
        center = _field_to_frequency_mhz(float(field_value))
    except (TypeError, ValueError):
        center = 0.0
    half_width = _field_to_frequency_mhz(float(config.window_half_width_gauss))

    # ``auto_window`` is an explicit request to derive the window from the run
    # field, so it must win over (possibly stale) explicit bounds.  Explicit
    # bounds are honoured when auto mode is off or the field is unusable.
    if config.auto_window and center > 0.0 and half_width > 0.0:
        return max(0.0, center - half_width), center + half_width

    explicit_min = config.f_min_mhz
    explicit_max = config.f_max_mhz
    if explicit_min is not None and explicit_max is not None and explicit_max > explicit_min:
        return float(explicit_min), float(explicit_max)
    if explicit_max is not None and explicit_max > 0.0:
        return max(0.0, float(explicit_min or 0.0)), float(explicit_max)
    return 0.0, max(10.0, center + half_width)


def build_maxent_input(
    run: Run,
    config: MaxEntConfig | dict | None = None,
    *,
    prepared_histograms: list[Histogram] | None = None,
    reference_t0_bin: int | None = None,
) -> MaxEntInput:
    """Build grouped raw-count MaxEnt input from *run* and *config*."""
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    run = _run_with_maxent_binning(run, resolved_config)
    group_names = _group_names(run)
    if not group_names:
        raise ValueError("MaxEnt requires detector groups.")
    if not run.histograms:
        raise ValueError("MaxEnt requires raw detector histograms.")

    if prepared_histograms is None:
        # ``use_deadtime_correction`` overrides the run's grouping metadata,
        # mirroring ``build_group_signal_dataset``'s explicit ``use_deadtime``
        # contract; passing prepared histograms there would otherwise make the
        # config flag inert.  ``reference_t0_bin`` (when not supplied) is
        # derived from these prepared histograms inside
        # ``build_group_signal_dataset``.
        grouping = run.grouping if isinstance(run.grouping, dict) else {}
        prepared_histograms, _applied = prepare_histograms_with_deadtime(
            list(run.histograms),
            grouping,
            bool(resolved_config.use_deadtime_correction),
        )

    all_ids = sorted(group_names)
    if resolved_config.selected_group_ids is None:
        selected = all_ids
    else:
        wanted = {int(g) for g in resolved_config.selected_group_ids}
        selected = [gid for gid in all_ids if gid in wanted]
    if not selected:
        raise ValueError("MaxEnt requires at least one selected group.")

    groups: list[MaxEntGroupInput] = []
    max_points = 0
    for group_id in selected:
        dataset = build_group_signal_dataset(
            run,
            group_id,
            center_signal=False,
            apply_lifetime_correction=True,
            use_deadtime=resolved_config.use_deadtime_correction,
            reference_t0_bin=reference_t0_bin,
            prepared_histograms=prepared_histograms,
        )
        time = np.asarray(dataset.time, dtype=np.float64)
        signal = np.asarray(dataset.asymmetry, dtype=np.float64)
        sigma = np.asarray(dataset.error, dtype=np.float64)
        mask = np.isfinite(time) & np.isfinite(signal) & np.isfinite(sigma) & (sigma > 0.0)
        if resolved_config.t_min_us is not None:
            mask &= time >= float(resolved_config.t_min_us)
        if resolved_config.t_max_us is not None:
            mask &= time <= float(resolved_config.t_max_us)
        if not np.any(mask):
            continue
        baseline = float(np.nanmean(signal[mask]))
        if not np.isfinite(baseline) or abs(baseline) <= _MIN_POSITIVE:
            baseline = 1.0
        normalized = (signal / baseline) - 1.0
        normalized_sigma = np.maximum(sigma / abs(baseline), _MIN_POSITIVE)
        max_points = max(max_points, int(np.count_nonzero(mask)))
        groups.append(
            MaxEntGroupInput(
                group_id=int(group_id),
                group_name=group_names.get(group_id, f"Group {group_id}"),
                time_us=time,
                signal=normalized,
                sigma=normalized_sigma,
                phase_degrees=float(resolved_config.group_phase_degrees.get(group_id, 0.0)),
                amplitude=1.0,
                background=0.0,
                mask=mask,
            )
        )

    if not groups:
        raise ValueError("MaxEnt input contains no valid grouped signals.")

    n_points = resolved_config.n_spectrum_points or default_n_spectrum_points(max_points)
    f_min, f_max = _resolve_frequency_window(run, resolved_config)
    if f_max <= f_min:
        f_max = f_min + 1.0
    return MaxEntInput(
        run_number=int(run.run_number),
        groups=tuple(groups),
        n_spectrum_points=int(n_points),
        f_min_mhz=float(f_min),
        f_max_mhz=float(f_max),
        default_level=float(resolved_config.default_level),
        metadata={
            "field": run.metadata.get("field") if isinstance(run.metadata, dict) else None,
            "group_ids": [group.group_id for group in groups],
            "time_binning_factor": int(resolved_config.time_binning_factor),
        },
    )


def _chunk_rows(n_time: int, n_frequency: int) -> int:
    """Return a row chunk that bounds temporary OPUS/TROPUS matrices."""
    if n_time <= 0:
        return 1
    return max(1, min(int(n_time), _MAX_DESIGN_CHUNK_ELEMENTS // max(1, int(n_frequency))))


def _project_forward(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    phase_degrees: float,
) -> NDArray[np.float64]:
    """Return dense-equivalent OPUS output without materialising all rows."""
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    f = np.asarray(spectrum, dtype=np.float64)
    phase = np.deg2rad(float(phase_degrees))
    output = np.empty(time.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        matrix = np.cos(
            2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :] + phase
        )
        output[start:stop] = matrix @ f
    return output


def _project_forward_components(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    spectrum: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(C @ f, S @ f)`` for the zero-phase cosine/sine kernels.

    ``cos(2πft + φ) = cosφ·C − sinφ·S``, so one component pair prices any
    number of phase candidates at vector cost instead of a kernel rebuild per
    candidate.
    """
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    f = np.asarray(spectrum, dtype=np.float64)
    cos_output = np.empty(time.size, dtype=np.float64)
    sin_output = np.empty(time.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        angle = 2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :]
        cos_output[start:stop] = np.cos(angle) @ f
        sin_output[start:stop] = np.sin(angle) @ f
    return cos_output, sin_output


def _project_adjoint(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    phase_degrees: float,
) -> NDArray[np.float64]:
    """Return dense-equivalent TROPUS output without materialising all rows."""
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    phase = np.deg2rad(float(phase_degrees))
    output = np.zeros(frequencies.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        matrix = np.cos(
            2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :] + phase
        )
        output += matrix.T @ v[start:stop]
    return output


def opus(
    spectrum: NDArray[np.float64],
    maxent_input: MaxEntInput,
    *,
    phases: dict[int, float] | None = None,
    amplitudes: dict[int, float] | None = None,
    backgrounds: dict[int, float] | None = None,
) -> dict[int, NDArray[np.float64]]:
    """Forward map from a shared spectrum to each group signal."""
    frequencies = maxent_input.frequencies_mhz
    f = np.asarray(spectrum, dtype=np.float64)
    predictions: dict[int, NDArray[np.float64]] = {}
    phases = phases or {}
    amplitudes = amplitudes or {}
    backgrounds = backgrounds or {}
    for group in maxent_input.groups:
        phase = float(phases.get(group.group_id, group.phase_degrees))
        amplitude = float(amplitudes.get(group.group_id, group.amplitude))
        background = float(backgrounds.get(group.group_id, group.background))
        predictions[group.group_id] = (
            amplitude * _project_forward(group.time_us, frequencies, f, phase_degrees=phase)
            + background
        )
    return predictions


def tropus(
    group_values: dict[int, NDArray[np.float64]],
    maxent_input: MaxEntInput,
    *,
    phases: dict[int, float] | None = None,
    amplitudes: dict[int, float] | None = None,
) -> NDArray[np.float64]:
    """Adjoint map from group time-domain values to one spectral vector."""
    frequencies = maxent_input.frequencies_mhz
    phases = phases or {}
    amplitudes = amplitudes or {}
    output = np.zeros(frequencies.size, dtype=np.float64)
    for group in maxent_input.groups:
        values = group_values.get(group.group_id)
        if values is None:
            continue
        phase = float(phases.get(group.group_id, group.phase_degrees))
        amplitude = float(amplitudes.get(group.group_id, group.amplitude))
        output += amplitude * _project_adjoint(
            group.time_us,
            frequencies,
            np.asarray(values, dtype=np.float64),
            phase_degrees=phase,
        )
    return output


def _state_signature(maxent_input: MaxEntInput, config: MaxEntConfig) -> tuple[Any, ...]:
    return (
        int(maxent_input.run_number),
        tuple(group.group_id for group in maxent_input.groups),
        int(maxent_input.n_spectrum_points),
        round(float(maxent_input.f_min_mhz), 12),
        round(float(maxent_input.f_max_mhz), 12),
        round(float(config.default_level), 12),
        bool(config.fit_phases),
        bool(config.fit_amplitudes),
        bool(config.fit_backgrounds),
        bool(config.fit_constant_background),
        # Data-preparation settings: changing any of these reshapes or rescales
        # the observed signals, so a resumed state would silently iterate a
        # stale spectrum against incompatible data.
        bool(config.use_deadtime_correction),
        None if config.t_min_us is None else round(float(config.t_min_us), 12),
        None if config.t_max_us is None else round(float(config.t_max_us), 12),
        int(config.time_binning_factor),
    )


def _initial_spectrum(maxent_input: MaxEntInput) -> NDArray[np.float64]:
    frequencies = maxent_input.frequencies_mhz
    estimate = np.full(frequencies.size, float(maxent_input.default_level), dtype=np.float64)
    for group in maxent_input.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        if not np.any(mask):
            continue
        weighted = np.asarray(group.signal, dtype=float)[mask]
        power = np.abs(
            _project_adjoint(
                np.asarray(group.time_us, dtype=float)[mask],
                frequencies,
                weighted,
                phase_degrees=group.phase_degrees,
            )
        )
        if power.size and np.nanmax(power) > 0.0:
            estimate += power / float(np.nanmax(power))
    area = np.trapezoid(estimate, frequencies) if frequencies.size > 1 else float(np.sum(estimate))
    if np.isfinite(area) and area > 0.0:
        estimate = estimate / area
    return np.maximum(estimate, _MIN_POSITIVE)


def initialize_state(
    maxent_input: MaxEntInput, config: MaxEntConfig | dict | None = None
) -> MaxEntState:
    """Return a fresh resumable MaxEnt state."""
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    phases = {group.group_id: group.phase_degrees for group in maxent_input.groups}
    amplitudes = {group.group_id: group.amplitude for group in maxent_input.groups}
    backgrounds = {group.group_id: group.background for group in maxent_input.groups}
    return MaxEntState(
        frequencies_mhz=maxent_input.frequencies_mhz,
        spectrum=_initial_spectrum(maxent_input),
        phases=phases,
        amplitudes=amplitudes,
        backgrounds=backgrounds,
        signature=_state_signature(maxent_input, resolved_config),
    )


def _residual_payload(
    state: MaxEntState,
    maxent_input: MaxEntInput,
) -> tuple[dict[int, NDArray[np.float64]], float, int]:
    """Return the full forward predictions plus χ² for the current state."""
    predictions = opus(
        state.spectrum,
        maxent_input,
        phases=state.phases,
        amplitudes=state.amplitudes,
        backgrounds=state.backgrounds,
    )
    chi2 = 0.0
    n_obs = 0
    for group in maxent_input.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        sigma = np.asarray(group.sigma, dtype=np.float64)
        pred = predictions[group.group_id]
        residual = (np.asarray(group.signal, dtype=np.float64)[mask] - pred[mask]) / np.maximum(
            sigma[mask],
            _MIN_POSITIVE,
        )
        chi2 += float(np.sum(residual**2))
        n_obs += int(np.count_nonzero(mask))
    return predictions, chi2, n_obs


def _residual_gradient_payload(
    state: MaxEntState,
    maxent_input: MaxEntInput,
) -> tuple[NDArray[np.float64], float, int]:
    """Return the χ² gradient (TROPUS of the weighted residuals) plus χ².

    Fuses the forward projection and the adjoint into one chunked pass per
    group, so each cosine kernel chunk is built once per inner iteration
    instead of twice (OPUS then TROPUS over identical chunks).
    """
    frequencies = maxent_input.frequencies_mhz
    f = np.asarray(state.spectrum, dtype=np.float64)
    grad = np.zeros(frequencies.size, dtype=np.float64)
    chi2 = 0.0
    n_obs = 0
    for group in maxent_input.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        phase = np.deg2rad(float(state.phases.get(group.group_id, group.phase_degrees)))
        amplitude = float(state.amplitudes.get(group.group_id, group.amplitude))
        background = float(state.backgrounds.get(group.group_id, group.background))
        time = np.asarray(group.time_us, dtype=np.float64)
        signal = np.asarray(group.signal, dtype=np.float64)
        sigma = np.maximum(np.asarray(group.sigma, dtype=np.float64), _MIN_POSITIVE)
        chunk = _chunk_rows(time.size, frequencies.size)
        for start in range(0, time.size, chunk):
            stop = min(start + chunk, time.size)
            rows = mask[start:stop]
            if not np.any(rows):
                continue
            matrix = np.cos(
                2.0 * np.pi * time[start:stop][rows, np.newaxis] * frequencies[np.newaxis, :]
                + phase
            )
            pred = amplitude * (matrix @ f) + background
            residual = (signal[start:stop][rows] - pred) / sigma[start:stop][rows]
            chi2 += float(np.dot(residual, residual))
            n_obs += int(np.count_nonzero(rows))
            grad += amplitude * (matrix.T @ (residual / sigma[start:stop][rows]))
    return grad, chi2, n_obs


def _check_cancel(cancel_callback: MaxEntCancelCallback | None) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise MaxEntCancelledError("MaxEnt calculation cancelled.")


def _fit_group_nuisance(
    state: MaxEntState,
    maxent_input: MaxEntInput,
    config: MaxEntConfig,
    *,
    cancel_callback: MaxEntCancelCallback | None = None,
    predictions: dict[int, NDArray[np.float64]] | None = None,
) -> None:
    # Callers that just evaluated the forward model on this exact state (the
    # end-of-cycle diagnostics) pass their predictions in to avoid a
    # back-to-back duplicate projection.
    if predictions is None:
        predictions = opus(
            state.spectrum,
            maxent_input,
            phases=state.phases,
            amplitudes=state.amplitudes,
            backgrounds=state.backgrounds,
        )
    for group in maxent_input.groups:
        _check_cancel(cancel_callback)
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        if np.count_nonzero(mask) < 2:
            continue
        y = np.asarray(group.signal, dtype=float)[mask]
        p = np.asarray(predictions[group.group_id], dtype=float)[mask]
        if config.fit_amplitudes or config.fit_backgrounds or config.fit_constant_background:
            # ``p`` already includes the current amplitude and background, so
            # recover the bare projection before regressing absolute values;
            # regressing against ``p`` itself would make the fitted amplitude
            # oscillate (amp_{n+1} ~ a / amp_n) instead of converging.
            amplitude = float(state.amplitudes.get(group.group_id, group.amplitude))
            background = float(state.backgrounds.get(group.group_id, group.background))
            safe_amplitude = amplitude if abs(amplitude) > _MIN_POSITIVE else 1.0
            base = (p - background) / safe_amplitude
            target = y
            columns: list[np.ndarray] = []
            if config.fit_amplitudes:
                columns.append(base)
            else:
                target = target - amplitude * base
            if config.fit_backgrounds or config.fit_constant_background:
                columns.append(np.ones_like(base))
            else:
                target = target - background
            if columns:
                x = np.vstack(columns).T
                try:
                    coeffs, *_ = np.linalg.lstsq(x, target, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None
                if coeffs is not None:
                    idx = 0
                    if config.fit_amplitudes:
                        state.amplitudes[group.group_id] = float(np.clip(coeffs[idx], 0.01, 100.0))
                        idx += 1
                    if config.fit_backgrounds or config.fit_constant_background:
                        state.backgrounds[group.group_id] = float(coeffs[idx])

        if config.fit_phases:
            current = float(state.phases.get(group.group_id, group.phase_degrees))
            candidates = current + np.linspace(-4.0, 4.0, 9)
            _check_cancel(cancel_callback)
            # One (C@f, S@f) pair prices every candidate: the kernel does not
            # depend on the scalar phase, so the scan is pure vector algebra.
            cos_proj, sin_proj = _project_forward_components(
                np.asarray(group.time_us, dtype=float)[mask],
                maxent_input.frequencies_mhz,
                state.spectrum,
            )
            amplitude = float(state.amplitudes[group.group_id])
            background = float(state.backgrounds[group.group_id])
            best_phase = current
            best_score = np.inf
            for candidate in candidates:
                radians = np.deg2rad(float(candidate))
                pred = (
                    amplitude * (np.cos(radians) * cos_proj - np.sin(radians) * sin_proj)
                    + background
                )
                residual = y - pred
                score = float(np.dot(residual, residual))
                if score < best_score:
                    best_score = score
                    best_phase = float(candidate)
            state.phases[group.group_id] = best_phase


def _entropy(spectrum: NDArray[np.float64], default_level: float) -> float:
    f = np.maximum(np.asarray(spectrum, dtype=float), _MIN_POSITIVE)
    default = max(float(default_level), _MIN_POSITIVE)
    return float(-np.sum(f * np.log(f / default)))


def _normalize_spectrum(
    spectrum: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    f = np.maximum(np.asarray(spectrum, dtype=float), _MIN_POSITIVE)
    area = np.trapezoid(f, frequencies) if frequencies.size > 1 else float(np.sum(f))
    if np.isfinite(area) and area > _MIN_POSITIVE:
        f = f / area
    return np.maximum(f, _MIN_POSITIVE)


def run_cycles(
    maxent_input: MaxEntInput,
    config: MaxEntConfig | dict | None = None,
    *,
    state: MaxEntState | None = None,
    cycles: int | None = None,
    progress_callback: MaxEntProgressCallback | None = None,
    cancel_callback: MaxEntCancelCallback | None = None,
) -> MaxEntResult:
    """Run *cycles* outer MaxEnt cycles, returning the updated result."""
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    active_state = state or initialize_state(maxent_input, resolved_config)
    signature = _state_signature(maxent_input, resolved_config)
    if active_state.signature and active_state.signature != signature:
        raise ValueError("MaxEnt state is incompatible with the current configuration; restart.")
    active_state.signature = signature

    n_cycles = resolved_config.outer_cycles if cycles is None else max(0, int(cycles))
    frequencies = maxent_input.frequencies_mhz
    total_steps = max(1, n_cycles * (int(resolved_config.inner_iterations) + 1))
    completed_steps = 0
    predictions: dict[int, NDArray[np.float64]] | None = None
    for _ in range(n_cycles):
        _check_cancel(cancel_callback)
        if progress_callback is not None:
            progress_callback(
                completed_steps,
                total_steps,
                f"Preparing cycle {active_state.cycle + 1}",
            )
        _fit_group_nuisance(
            active_state,
            maxent_input,
            resolved_config,
            cancel_callback=cancel_callback,
            predictions=predictions,
        )
        completed_steps += 1
        if progress_callback is not None:
            progress_callback(
                completed_steps,
                total_steps,
                f"Refining spectrum for cycle {active_state.cycle + 1}",
            )
        for _inner in range(resolved_config.inner_iterations):
            _check_cancel(cancel_callback)
            grad, chi2, n_obs = _residual_gradient_payload(active_state, maxent_input)
            entropy_grad = -np.log(
                np.maximum(active_state.spectrum, _MIN_POSITIVE)
                / max(float(resolved_config.default_level), _MIN_POSITIVE)
            )
            target = max(float(n_obs) * resolved_config.chi2_target_over_n, _MIN_POSITIVE)
            pressure = min(1.0, chi2 / target)
            direction = grad + 0.02 * pressure * entropy_grad
            scale = float(np.nanmax(np.abs(direction))) if direction.size else 1.0
            if not np.isfinite(scale) or scale <= 0.0:
                break
            step = 0.15 / scale
            active_state.spectrum *= np.exp(np.clip(step * direction, -0.5, 0.5))
            active_state.spectrum = _normalize_spectrum(active_state.spectrum, frequencies)
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(
                    completed_steps,
                    total_steps,
                    f"Cycle {active_state.cycle + 1}: inner iteration {_inner + 1}",
                )

        _check_cancel(cancel_callback)
        # The end-of-cycle predictions are reused by the next cycle's nuisance
        # fit — the state does not change between here and there.
        predictions, chi2, n_obs = _residual_payload(active_state, maxent_input)
        entropy = _entropy(active_state.spectrum, resolved_config.default_level)
        target = max(float(n_obs) * resolved_config.chi2_target_over_n, _MIN_POSITIVE)
        test = float(abs(chi2 - target) / target)
        sconv = float(np.nanmax(active_state.spectrum) / np.nanmean(active_state.spectrum))
        active_state.cycle += 1
        active_state.diagnostics.append(
            cycle=active_state.cycle,
            chi2=chi2,
            entropy=entropy,
            test=test,
            sconv=sconv,
            phases=active_state.phases,
            amplitudes=active_state.amplitudes,
            backgrounds=active_state.backgrounds,
        )

    metadata = dict(maxent_input.metadata)
    metadata.update(
        {
            "run_number": maxent_input.run_number,
            "run_label": f"{maxent_input.run_number} MaxEnt",
            "plot_domain": "frequency",
            "x_label": "Frequency (MHz)",
            "y_label": "MaxEnt spectral density (a.u.)",
            "fourier_display": "MaxEnt",
            "frequency_representation": "maxent",
            "maxent_cycles": int(active_state.cycle),
            "maxent_chi2": (
                float(active_state.diagnostics.chi2[-1]) if active_state.diagnostics.chi2 else None
            ),
        }
    )
    return MaxEntResult(
        frequencies_mhz=np.asarray(frequencies, dtype=float),
        spectrum=np.asarray(active_state.spectrum, dtype=float),
        state=active_state,
        diagnostics=active_state.diagnostics,
        metadata=metadata,
    )


def maxent(
    run: Run,
    config: MaxEntConfig | dict | None = None,
    *,
    cycles: int | None = None,
    state: MaxEntState | None = None,
    progress_callback: MaxEntProgressCallback | None = None,
    cancel_callback: MaxEntCancelCallback | None = None,
) -> MaxEntResult:
    """Compute a grouped MaxEnt spectrum for *run*."""
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    _check_cancel(cancel_callback)
    maxent_input = build_maxent_input(run, resolved_config)
    return run_cycles(
        maxent_input,
        resolved_config,
        state=state,
        cycles=cycles,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
