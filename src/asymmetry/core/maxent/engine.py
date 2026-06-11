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
from asymmetry.core.fourier.units import gauss_to_mhz
from asymmetry.core.maxent.pulse import pulse_amplitude_phase
from asymmetry.core.maxent.specbg import apply_maxent_specbg
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import group_names
from asymmetry.core.utils.coerce import optional_float

_MAX_SPECTRUM_POINTS = 1 << 20
_MIN_POSITIVE = 1.0e-15
_MAX_DESIGN_CHUNK_ELEMENTS = 2_000_000

# Interior-exclusion σ-inflation factor: points inside the exclusion window keep
# their place in the time grid but are de-weighted to ~zero (weight ∝ 1/σ²), so
# the FFT/grid length is preserved.  Large but finite (WiMDA uses 1e15; 1e8
# keeps σ² well clear of float overflow while contributing weight ~1e-16).
_EXCLUSION_SIGMA_INFLATION = 1.0e8

_PULSE_N_PULSES = {"ignore": 0, "single": 1, "double": 2}

# Early-stop guard for forced cycle counts.  Past the χ² optimum the projected
# gradient keeps minimising χ² by collapsing spectral weight onto a grid edge
# (DC at f=0 for field scans, the band edge otherwise) — the line is lost even
# though χ² is flat or still falling.  So treat an explicit ``cycles`` count as a
# maximum and stop at the χ² plateau: the first cycle whose relative χ²
# improvement over the previous cycle drops below the tolerance.
#
# The minimum-cycle gate is a short warm-up so the guard does not trip on the
# steep early descent.  Its value is tightly constrained: it must be small
# enough to fire before the spectrum degrades (on the test synthetic the line is
# intact through cycle 8 but flips to the band edge by cycle 9), yet ``> N`` so
# an exact ``cycles=N`` run is unaffected.  Tests/scripts that need an exact
# count regardless should pass ``early_stop=False`` rather than rely on staying
# under this gate.
_EARLY_STOP_MIN_CYCLES = 6
_EARLY_STOP_CHI2_REL_TOL = 5.0e-3

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


def _float_or_default(value: object, default: float) -> float:
    parsed = optional_float(value)
    return float(default) if parsed is None else parsed


def _parse_group_ids(value: object) -> list[int] | None:
    """Parse a serialised group-id list, skipping malformed entries."""
    if not isinstance(value, list):
        return None
    parsed: list[int] = []
    for entry in value:
        try:
            parsed.append(int(entry))
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_phase_table(value: object) -> dict[int, float]:
    """Parse a serialised {group_id: phase} table, skipping malformed entries.

    Recipes cross the project-file boundary, so a corrupted or hand-edited
    entry must degrade to "no seed for that group" rather than raising out of
    whichever GUI slot happens to touch the recipe.
    """
    if not isinstance(value, dict):
        return {}
    parsed: dict[int, float] = {}
    for key, raw in value.items():
        phase = optional_float(raw)
        if phase is None:
            continue
        try:
            parsed[int(key)] = phase
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_choice(value: object, choices: tuple[str, ...], default: str) -> str:
    """Return *value* if it is one of *choices* (case-insensitive), else *default*."""
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return default


def _field_to_frequency_mhz(field_gauss: float) -> float:
    return float(gauss_to_mhz(float(field_gauss)))


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
    # ISIS pulse-shape response folded into the forward model (Phase 2).
    # ``pulse_mode`` is one of "ignore" / "single" / "double"; the widths are in
    # microseconds.  Defaults: off (continuous-source data needs no shaping).
    pulse_mode: str = "ignore"
    pulse_half_width_us: float = 0.05
    pulse_separation_us: float = 0.324
    # Interior exclusion window (µs): points inside are σ-inflated (de-weighted),
    # not dropped, so the time grid length is preserved.  ``None`` disables it.
    exclude_t_min_us: float | None = None
    exclude_t_max_us: float | None = None
    # Reconstruction mode: "general" (the joint multi-group fit) or "zf_lf"
    # (zero/longitudinal field, exactly two forward/backward groups with phases
    # pinned 0/180 and amplitudes tied through the run's α).
    mode: str = "general"
    # Display-only SpecBG: subtract a zero-centred pseudo-Voigt model of the
    # static central peak from the displayed ZF/LF field-distribution spectrum.
    # Widths in MHz; does not change the computation (absent from the signature).
    specbg_enabled: bool = False
    specbg_gaussian_width_mhz: float = 0.1
    specbg_lorentzian_width_mhz: float = 0.1
    specbg_lorentzian_fraction: float = 0.5
    # Display-only: whether the time-domain reconstruction overlay is the active
    # view (vs the spectrum).  It does not change the computation, so it is
    # absent from ``_state_signature`` (toggling it must not invalidate a
    # resumed run).  Defaults off — the spectrum is MaxEnt's primary output.
    show_reconstruction: bool = False

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
            "pulse_mode": str(self.pulse_mode),
            "pulse_half_width_us": float(self.pulse_half_width_us),
            "pulse_separation_us": float(self.pulse_separation_us),
            "exclude_t_min_us": self.exclude_t_min_us,
            "exclude_t_max_us": self.exclude_t_max_us,
            "mode": str(self.mode),
            "specbg_enabled": bool(self.specbg_enabled),
            "specbg_gaussian_width_mhz": float(self.specbg_gaussian_width_mhz),
            "specbg_lorentzian_width_mhz": float(self.specbg_lorentzian_width_mhz),
            "specbg_lorentzian_fraction": float(self.specbg_lorentzian_fraction),
            "show_reconstruction": bool(self.show_reconstruction),
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
            default_level=max(_MIN_POSITIVE, _float_or_default(data.get("default_level"), 0.01)),
            f_min_mhz=optional_float(data.get("f_min_mhz")),
            f_max_mhz=optional_float(data.get("f_max_mhz")),
            auto_window=bool(data.get("auto_window", True)),
            window_half_width_gauss=max(
                0.0, _float_or_default(data.get("window_half_width_gauss"), 300.0)
            ),
            outer_cycles=_parse_positive_int(data.get("outer_cycles", 10), 10),
            inner_iterations=_parse_positive_int(data.get("inner_iterations", 12), 12),
            chi2_target_over_n=max(
                _MIN_POSITIVE, _float_or_default(data.get("chi2_target_over_n"), 1.0)
            ),
            fit_phases=bool(data.get("fit_phases", True)),
            fit_amplitudes=bool(data.get("fit_amplitudes", True)),
            fit_backgrounds=bool(data.get("fit_backgrounds", True)),
            fit_constant_background=bool(data.get("fit_constant_background", True)),
            use_deadtime_correction=bool(data.get("use_deadtime_correction", True)),
            selected_group_ids=_parse_group_ids(selected),
            group_phase_degrees=_parse_phase_table(phases),
            t_min_us=optional_float(data.get("t_min_us")),
            t_max_us=optional_float(data.get("t_max_us")),
            time_binning_factor=_parse_positive_int(data.get("time_binning_factor", 1)),
            pulse_mode=_parse_choice(
                data.get("pulse_mode"), ("ignore", "single", "double"), "ignore"
            ),
            pulse_half_width_us=max(0.0, _float_or_default(data.get("pulse_half_width_us"), 0.05)),
            pulse_separation_us=max(0.0, _float_or_default(data.get("pulse_separation_us"), 0.324)),
            exclude_t_min_us=optional_float(data.get("exclude_t_min_us")),
            exclude_t_max_us=optional_float(data.get("exclude_t_max_us")),
            mode=_parse_choice(data.get("mode"), ("general", "zf_lf"), "general"),
            specbg_enabled=bool(data.get("specbg_enabled", False)),
            specbg_gaussian_width_mhz=max(
                0.0, _float_or_default(data.get("specbg_gaussian_width_mhz"), 0.1)
            ),
            specbg_lorentzian_width_mhz=max(
                0.0, _float_or_default(data.get("specbg_lorentzian_width_mhz"), 0.1)
            ),
            specbg_lorentzian_fraction=float(
                np.clip(_float_or_default(data.get("specbg_lorentzian_fraction"), 0.5), 0.0, 1.0)
            ),
            show_reconstruction=bool(data.get("show_reconstruction", False)),
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
    # Pulse-shape response over ``frequencies_mhz`` as a per-frequency amplitude
    # ``R(ν)`` and phase shift ``δ(ν)`` (radians): the forward kernel becomes
    # ``R·cos(2πνt + φ − δ)``.  ``None`` means no pulse shaping (R = 1, δ = 0).
    pulse_amp: NDArray[np.float64] | None = None
    pulse_phase: NDArray[np.float64] | None = None
    # ZF/LF mode: the reconstruction mode and the α used to tie the F/B group
    # amplitudes/backgrounds.  ``mode == "general"`` leaves ``zf_lf_alpha`` None.
    mode: str = "general"
    zf_lf_alpha: float | None = None

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
    # How the cycle loop ended: ``"max_cycles"`` ran the full requested count,
    # ``"converged"`` stopped at the χ² plateau, ``"diverged"`` stopped because
    # χ² had started rising past the optimum.  ``converged``/``diverged`` are
    # convenience views; either implies ``early_stopped``.
    stop_reason: str = "max_cycles"
    converged: bool = False
    diverged: bool = False
    # The prepared input the cycles iterated.  Carried so callers (the GUI
    # reconstruction overlay) can reuse the exact grouped signals/kernel the
    # engine fitted instead of rebuilding them — the reconstruction's χ² then
    # equals the engine's by identity, not just by deterministic rebuild.  Not
    # serialised (the project persists recipes, not prepared arrays).
    maxent_input: MaxEntInput | None = None

    @property
    def early_stopped(self) -> bool:
        """Whether the run stopped before the requested cycle count."""
        return self.stop_reason != "max_cycles"

    def as_dataset(self, run: Run | None = None, config: MaxEntConfig | None = None) -> MuonDataset:
        """Return the primary MaxEnt spectrum as a plottable dataset.

        When *config* is supplied this is also the single point where the
        display-only SpecBG zero-frequency subtraction is applied (a no-op unless
        SpecBG is enabled in ZF/LF mode), so the on-demand and live render paths
        cannot diverge on whether/how the central peak is removed.
        """
        error = np.zeros_like(self.spectrum, dtype=float)
        dataset = MuonDataset(
            time=np.asarray(self.frequencies_mhz, dtype=float),
            asymmetry=np.asarray(self.spectrum, dtype=float),
            error=error,
            metadata=dict(self.metadata),
            run=run,
        )
        if config is not None:
            return apply_maxent_specbg(dataset, config)
        return dataset


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
    names_by_group = group_names(run)
    all_ids = sorted(names_by_group)
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
    names_by_group = group_names(run)
    if not names_by_group:
        raise ValueError("MaxEnt requires detector groups.")
    if not run.histograms:
        raise ValueError("MaxEnt requires raw detector histograms.")

    if prepared_histograms is None:
        # ``use_deadtime_correction`` means "honour the run's existing deadtime
        # setting": correction applies only when the config flag AND the run's
        # grouping flag agree.  This keeps MaxEnt input consistent with the FFT
        # path (which follows the grouping flag alone) in the default state
        # where loaders populate ``dead_time_us`` but leave the grouping flag
        # off, while still letting the panel checkbox force correction off.
        # ``reference_t0_bin`` (when not supplied) is derived from these
        # prepared histograms inside ``build_group_signal_dataset``.
        grouping = run.grouping if isinstance(run.grouping, dict) else {}
        prepared_histograms, _applied = prepare_histograms_with_deadtime(
            list(run.histograms),
            grouping,
            bool(resolved_config.use_deadtime_correction)
            and bool(grouping.get("deadtime_correction", False)),
        )

    all_ids = sorted(names_by_group)
    if resolved_config.selected_group_ids is None:
        selected = all_ids
    else:
        wanted = {int(g) for g in resolved_config.selected_group_ids}
        selected = [gid for gid in all_ids if gid in wanted]
    if not selected:
        raise ValueError("MaxEnt requires at least one selected group.")

    # ZF/LF mode: exactly two forward/backward groups, phases pinned 0/180, and
    # amplitudes/backgrounds tied through the run's α.
    zf_lf_phase_map: dict[int, float] = {}
    zf_lf_alpha: float | None = None
    if resolved_config.mode == "zf_lf":
        if len(selected) != 2:
            raise ValueError("ZF/LF mode requires exactly two selected groups (forward/backward).")
        grouping = run.grouping if isinstance(run.grouping, dict) else {}
        # Order the pair by the run's forward/backward designation (not the
        # sorted group id) so the forward group is pinned to 0° and the α tie
        # enforces F = α·B on the right detectors even when backward has the
        # lower id.  Fall back to selection order if neither matches.
        forward_group = grouping.get("forward_group")
        backward_group = grouping.get("backward_group")
        if backward_group in selected and forward_group not in selected:
            selected = [gid for gid in selected if gid != backward_group] + [int(backward_group)]
        elif forward_group in selected:
            selected = [int(forward_group)] + [gid for gid in selected if gid != forward_group]
        zf_lf_phase_map = {int(selected[0]): 0.0, int(selected[1]): 180.0}
        zf_lf_alpha = _float_or_default(grouping.get("alpha"), 1.0)
        if not np.isfinite(zf_lf_alpha) or zf_lf_alpha <= 0.0:
            zf_lf_alpha = 1.0

    groups: list[MaxEntGroupInput] = []
    max_points = 0
    # Shared across the sweep so a reference_run background loads + deadtime-
    # prepares once, not once per group.
    background_reference_cache: dict = {}
    for group_id in selected:
        dataset = build_group_signal_dataset(
            run,
            group_id,
            center_signal=False,
            apply_lifetime_correction=True,
            use_deadtime=resolved_config.use_deadtime_correction,
            reference_t0_bin=reference_t0_bin,
            prepared_histograms=prepared_histograms,
            background_reference_cache=background_reference_cache,
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
        # Interior exclusion: keep the points in place (grid length preserved)
        # but inflate σ so they carry ~zero weight, mirroring WiMDA's σ=1e15
        # sentinel for excluded channels rather than dropping them.
        ex_lo = resolved_config.exclude_t_min_us
        ex_hi = resolved_config.exclude_t_max_us
        if ex_lo is not None and ex_hi is not None and ex_hi > ex_lo:
            in_window = (time >= float(ex_lo)) & (time <= float(ex_hi))
            normalized_sigma = np.where(
                in_window, normalized_sigma * _EXCLUSION_SIGMA_INFLATION, normalized_sigma
            )
        max_points = max(max_points, int(np.count_nonzero(mask)))
        groups.append(
            MaxEntGroupInput(
                group_id=int(group_id),
                group_name=names_by_group.get(group_id, f"Group {group_id}"),
                time_us=time,
                signal=normalized,
                sigma=normalized_sigma,
                phase_degrees=float(
                    zf_lf_phase_map.get(
                        group_id, resolved_config.group_phase_degrees.get(group_id, 0.0)
                    )
                ),
                amplitude=1.0,
                background=0.0,
                mask=mask,
            )
        )

    if not groups:
        raise ValueError("MaxEnt input contains no valid grouped signals.")
    if resolved_config.mode == "zf_lf" and len(groups) != 2:
        # A group can be dropped above if its mask is entirely empty (e.g. an
        # exclusion/time window that removes all its points).  The α tie needs
        # both groups, so fail loudly rather than silently degrade to an untied
        # single-group fit.
        raise ValueError(
            "ZF/LF mode needs two groups with usable data; one group has no "
            "points in the selected time/exclusion window."
        )

    n_points = resolved_config.n_spectrum_points or default_n_spectrum_points(max_points)
    f_min, f_max = _resolve_frequency_window(run, resolved_config)
    if f_max <= f_min:
        f_max = f_min + 1.0
    n_points = int(n_points)
    frequencies = np.linspace(float(f_min), float(f_max), n_points)
    n_pulses = _PULSE_N_PULSES.get(resolved_config.pulse_mode, 0)
    pulse_amp, pulse_phase = pulse_amplitude_phase(
        frequencies,
        half_width_us=resolved_config.pulse_half_width_us,
        separation_us=resolved_config.pulse_separation_us,
        n_pulses=n_pulses,
    )
    return MaxEntInput(
        run_number=int(run.run_number),
        groups=tuple(groups),
        n_spectrum_points=n_points,
        f_min_mhz=float(f_min),
        f_max_mhz=float(f_max),
        default_level=float(resolved_config.default_level),
        pulse_amp=pulse_amp,
        pulse_phase=pulse_phase,
        mode=str(resolved_config.mode),
        zf_lf_alpha=zf_lf_alpha,
        metadata={
            "field": run.metadata.get("field") if isinstance(run.metadata, dict) else None,
            "group_ids": [group.group_id for group in groups],
            "time_binning_factor": int(resolved_config.time_binning_factor),
            "pulse_mode": str(resolved_config.pulse_mode),
            "maxent_mode": str(resolved_config.mode),
            "zf_lf_alpha": zf_lf_alpha,
        },
    )


def _chunk_rows(n_time: int, n_frequency: int) -> int:
    """Return a row chunk that bounds temporary OPUS/TROPUS matrices."""
    if n_time <= 0:
        return 1
    return max(1, min(int(n_time), _MAX_DESIGN_CHUNK_ELEMENTS // max(1, int(n_frequency))))


def _kernel_phase_offset(
    phase: float,
    pulse_phase: NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    """Return the per-frequency kernel angle offset ``phase − δ(ν)``."""
    return phase if pulse_phase is None else phase - pulse_phase[np.newaxis, :]


def _apply_pulse_amp(
    vector: NDArray[np.float64],
    pulse_amp: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Scale a per-frequency vector by the pulse amplitude ``R(ν)`` (identity if None).

    The pulse amplitude column-scales the kernel ``M·diag(R)``.  Rather than
    materialise that scaled matrix per chunk, the forward map folds ``R`` into
    the spectrum it multiplies (``M @ (R⊙f)``) and the adjoint folds it into the
    accumulated result (``R ⊙ (Mᵀ@v)``).  Both equal the column-scaled operator
    and its exact transpose, so the OPUS/TROPUS adjoint property is preserved
    bit-for-bit in the matrix ``M`` — only the cheap O(n_freq) scaling moves out
    of the hot ``n_time × n_freq`` block.
    """
    return vector if pulse_amp is None else vector * pulse_amp


def _project_forward(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    phase_degrees: float,
    pulse_amp: NDArray[np.float64] | None = None,
    pulse_phase: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return dense-equivalent OPUS output without materialising all rows.

    The pulse-shape response enters as the per-frequency amplitude ``pulse_amp``
    (R) and phase shift ``pulse_phase`` (δ): the kernel becomes
    ``R(ν)·cos(2πνt + φ − δ(ν))``, which preserves the OPUS/TROPUS adjoint pair
    because both maps build the identical matrix.
    """
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    # Fold the pulse amplitude R(ν) into the spectrum once: M @ (R⊙f) equals the
    # column-scaled kernel (M·diag(R)) @ f without allocating the scaled block.
    f = _apply_pulse_amp(np.asarray(spectrum, dtype=np.float64), pulse_amp)
    phase = np.deg2rad(float(phase_degrees))
    offset = _kernel_phase_offset(phase, pulse_phase)
    output = np.empty(time.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        matrix = np.cos(
            2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :] + offset
        )
        output[start:stop] = matrix @ f
    return output


def _project_forward_components(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    pulse_amp: NDArray[np.float64] | None = None,
    pulse_phase: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(C @ f, S @ f)`` for the (pulse-shaped) cosine/sine kernels.

    With the pulse response the bare kernel ``cos(2πνt)`` / ``sin(2πνt)`` is
    replaced by ``R(ν)·cos(2πνt − δ(ν))`` / ``R(ν)·sin(2πνt − δ(ν))`` so that the
    model at phase φ is still ``cosφ·C − sinφ·S`` — the phase scan formula is
    unchanged, the components simply carry the pulse shaping.
    """
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    # Fold R(ν) into the spectrum once (see :func:`_apply_pulse_amp`); both the
    # cosine and sine projections then carry the pulse shaping via this scaling.
    f = _apply_pulse_amp(np.asarray(spectrum, dtype=np.float64), pulse_amp)
    offset = _kernel_phase_offset(0.0, pulse_phase)
    cos_output = np.empty(time.size, dtype=np.float64)
    sin_output = np.empty(time.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        angle = 2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :] + offset
        cos_output[start:stop] = np.cos(angle) @ f
        sin_output[start:stop] = np.sin(angle) @ f
    return cos_output, sin_output


def _project_adjoint(
    time_us: NDArray[np.float64],
    frequencies_mhz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    phase_degrees: float,
    pulse_amp: NDArray[np.float64] | None = None,
    pulse_phase: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return dense-equivalent TROPUS output without materialising all rows.

    Uses the same pulse-shaped kernel as :func:`_project_forward`, so it remains
    its exact matrix transpose (the OPUS/TROPUS adjoint property is preserved).
    """
    time = np.asarray(time_us, dtype=np.float64)
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    phase = np.deg2rad(float(phase_degrees))
    offset = _kernel_phase_offset(phase, pulse_phase)
    output = np.zeros(frequencies.size, dtype=np.float64)
    chunk = _chunk_rows(time.size, frequencies.size)
    for start in range(0, time.size, chunk):
        stop = min(start + chunk, time.size)
        matrix = np.cos(
            2.0 * np.pi * time[start:stop, np.newaxis] * frequencies[np.newaxis, :] + offset
        )
        output += matrix.T @ v[start:stop]
    # Weight the accumulated adjoint by R(ν) once: R ⊙ (Mᵀ@v) is the exact
    # transpose of the column-scaled forward map M @ (R⊙f).
    return _apply_pulse_amp(output, pulse_amp)


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
            amplitude
            * _project_forward(
                group.time_us,
                frequencies,
                f,
                phase_degrees=phase,
                pulse_amp=maxent_input.pulse_amp,
                pulse_phase=maxent_input.pulse_phase,
            )
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
            pulse_amp=maxent_input.pulse_amp,
            pulse_phase=maxent_input.pulse_phase,
        )
    return output


@dataclass(frozen=True)
class ReconstructedGroup:
    """One group's time-domain reconstruction over its in-window points.

    All arrays are masked to the group's good (finite, positive-σ, in-window)
    points, in the engine's internal normalised signal space — the same space
    the χ² is computed in, so ``chi2`` here equals this group's contribution to
    the engine's reported χ² exactly.
    """

    group_id: int
    group_name: str
    time_us: NDArray[np.float64]
    data: NDArray[np.float64]
    model: NDArray[np.float64]
    sigma: NDArray[np.float64]
    residual: NDArray[np.float64]
    chi2: float
    n_obs: int


def reconstruct_group_signals(
    maxent_input: MaxEntInput,
    state: MaxEntState,
) -> dict[int, ReconstructedGroup]:
    """Return the per-group time-domain reconstruction for a converged state.

    The forward model (``opus``) of the shared spectrum is the reconstruction
    that the engine fits to each group's data; this packages it alongside the
    observed signal, σ, and the weighted residual ``(data − model)/σ`` for the
    overlay.  Summing :attr:`ReconstructedGroup.chi2` over groups reproduces the
    engine's reported χ² (same residual computation as ``_residual_payload``).
    """
    predictions = opus(
        state.spectrum,
        maxent_input,
        phases=state.phases,
        amplitudes=state.amplitudes,
        backgrounds=state.backgrounds,
    )
    reconstructions: dict[int, ReconstructedGroup] = {}
    for group in maxent_input.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        time = np.asarray(group.time_us, dtype=np.float64)[mask]
        data = np.asarray(group.signal, dtype=np.float64)[mask]
        model = np.asarray(predictions[group.group_id], dtype=np.float64)[mask]
        sigma = np.maximum(np.asarray(group.sigma, dtype=np.float64)[mask], _MIN_POSITIVE)
        residual = (data - model) / sigma
        chi2 = float(np.sum(residual**2))
        reconstructions[group.group_id] = ReconstructedGroup(
            group_id=int(group.group_id),
            group_name=str(group.group_name),
            time_us=time,
            data=data,
            model=model,
            sigma=sigma,
            residual=residual,
            chi2=chi2,
            n_obs=int(time.size),
        )
    return reconstructions


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
        # Pulse-shape response and the interior exclusion window both reshape the
        # forward model / the data weighting, so a resumed state with a changed
        # setting would iterate a stale spectrum against an incompatible model.
        str(config.pulse_mode),
        round(float(config.pulse_half_width_us), 12),
        round(float(config.pulse_separation_us), 12),
        None if config.exclude_t_min_us is None else round(float(config.exclude_t_min_us), 12),
        None if config.exclude_t_max_us is None else round(float(config.exclude_t_max_us), 12),
        # ZF/LF mode ties group amplitudes/backgrounds and pins phases, so it
        # reshapes the fit; a changed mode must invalidate a resumed state.
        str(config.mode),
        # Effective phase seeds: ``state.phases`` is seeded from these at
        # initialisation and a resumed state never re-reads the config, so an
        # edited seed must force a restart or it would be silently ignored
        # (with ``fit_phases`` off, permanently; with it on, the ±4°/cycle
        # refinement cannot follow a large seed change either).
        tuple(round(float(group.phase_degrees), 9) for group in maxent_input.groups),
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
                pulse_amp=maxent_input.pulse_amp,
                pulse_phase=maxent_input.pulse_phase,
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
    *,
    cancel_callback: MaxEntCancelCallback | None = None,
) -> tuple[NDArray[np.float64], float, int]:
    """Return the χ² gradient (TROPUS of the weighted residuals) plus χ².

    Fuses the forward projection and the adjoint into one chunked pass per
    group, so each cosine kernel chunk is built once per inner iteration
    instead of twice (OPUS then TROPUS over identical chunks).  Cancellation
    is checked per chunk: a full pass over a large workload can take tens of
    seconds, and window-close waits on cooperative cancel with a timeout.
    """
    frequencies = maxent_input.frequencies_mhz
    pulse_amp = maxent_input.pulse_amp
    pulse_phase = maxent_input.pulse_phase
    # Fold R(ν) into the spectrum once for every group's forward projection; the
    # adjoint gradient gets it back once at the end (R is common to all groups,
    # so it factors straight out of the group sum).  Net: no scaled kernel block
    # is materialised in the chunk loop.
    f = _apply_pulse_amp(np.asarray(state.spectrum, dtype=np.float64), pulse_amp)
    grad = np.zeros(frequencies.size, dtype=np.float64)
    chi2 = 0.0
    n_obs = 0
    for group in maxent_input.groups:
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        phase = np.deg2rad(float(state.phases.get(group.group_id, group.phase_degrees)))
        offset = _kernel_phase_offset(phase, pulse_phase)
        amplitude = float(state.amplitudes.get(group.group_id, group.amplitude))
        background = float(state.backgrounds.get(group.group_id, group.background))
        time = np.asarray(group.time_us, dtype=np.float64)
        signal = np.asarray(group.signal, dtype=np.float64)
        sigma = np.maximum(np.asarray(group.sigma, dtype=np.float64), _MIN_POSITIVE)
        chunk = _chunk_rows(time.size, frequencies.size)
        for start in range(0, time.size, chunk):
            _check_cancel(cancel_callback)
            stop = min(start + chunk, time.size)
            rows = mask[start:stop]
            if not np.any(rows):
                continue
            matrix = np.cos(
                2.0 * np.pi * time[start:stop][rows, np.newaxis] * frequencies[np.newaxis, :]
                + offset
            )
            pred = amplitude * (matrix @ f) + background
            residual = (signal[start:stop][rows] - pred) / sigma[start:stop][rows]
            chi2 += float(np.dot(residual, residual))
            n_obs += int(np.count_nonzero(rows))
            grad += amplitude * (matrix.T @ (residual / sigma[start:stop][rows]))
    return _apply_pulse_amp(grad, pulse_amp), chi2, n_obs


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

        # ZF/LF pins phases at 0/180; never refit them in that mode.
        if config.fit_phases and config.mode != "zf_lf":
            current = float(state.phases.get(group.group_id, group.phase_degrees))
            candidates = current + np.linspace(-4.0, 4.0, 9)
            _check_cancel(cancel_callback)
            # One (C@f, S@f) pair prices every candidate: the kernel does not
            # depend on the scalar phase, so the scan is pure vector algebra.
            cos_proj, sin_proj = _project_forward_components(
                np.asarray(group.time_us, dtype=float)[mask],
                maxent_input.frequencies_mhz,
                state.spectrum,
                pulse_amp=maxent_input.pulse_amp,
                pulse_phase=maxent_input.pulse_phase,
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

    _apply_zf_lf_tie(state, maxent_input, config)


def _apply_zf_lf_tie(
    state: MaxEntState,
    maxent_input: MaxEntInput,
    config: MaxEntConfig,
) -> None:
    """Tie the two F/B group amplitudes and backgrounds through α (ZF/LF mode).

    Mirrors WiMDA's per-cycle redistribution: the two independently fitted
    values are summed and split in ratio α:1 (group1 = α·group2), enforcing the
    F = α·B detector-efficiency balance after the least-squares step rather than
    as a constraint inside it.
    """
    if config.mode != "zf_lf":
        return
    alpha = maxent_input.zf_lf_alpha
    if alpha is None or not np.isfinite(alpha) or alpha <= 0.0 or len(maxent_input.groups) != 2:
        return
    forward_id = int(maxent_input.groups[0].group_id)
    backward_id = int(maxent_input.groups[1].group_id)
    # Only redistribute the quantities the user is actually fitting — a disabled
    # amplitude/background fit must stay frozen (mirrors WiMDA, which ties inside
    # MODAMP/MODBAK, i.e. only when those values are being refit).
    tied: list[tuple[dict[int, float], float, float]] = []
    if config.fit_amplitudes:
        tied.append((state.amplitudes, 0.01, 100.0))
    if config.fit_backgrounds or config.fit_constant_background:
        tied.append((state.backgrounds, -np.inf, np.inf))
    for table, lo, hi in tied:
        x_forward = float(table.get(forward_id, 0.0))
        x_backward = float(table.get(backward_id, 0.0))
        tied_backward = (x_forward + x_backward) / (1.0 + alpha)
        table[backward_id] = float(np.clip(tied_backward, lo, hi))
        table[forward_id] = float(np.clip(alpha * table[backward_id], lo, hi))


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
    early_stop: bool = True,
    progress_callback: MaxEntProgressCallback | None = None,
    cancel_callback: MaxEntCancelCallback | None = None,
) -> MaxEntResult:
    """Run up to *cycles* outer MaxEnt cycles, returning the updated result.

    With *early_stop* (the default) ``cycles`` is a maximum: the loop halts at
    the χ² plateau instead of iterating into the post-optimum divergence that a
    forced high cycle count (e.g. the GUI's 50-cycle Converge) otherwise drives.
    Pass ``early_stop=False`` to run the exact count regardless.
    """
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
    # Seed the plateau test from the resumed history so incremental cycle calls
    # (the GUI's 1/5/25 buttons) share one continuous χ² trajectory.
    prev_chi2 = active_state.diagnostics.chi2[-1] if active_state.diagnostics.chi2 else None
    stop_reason = "max_cycles"
    converged = False
    diverged = False
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
            grad, chi2, n_obs = _residual_gradient_payload(
                active_state, maxent_input, cancel_callback=cancel_callback
            )
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

        # Plateau guard: once χ² stops improving meaningfully, further cycles only
        # degrade the spectrum (collapse onto DC / the band edge), so stop and
        # keep this iterate.  The minimum-cycle gate keeps short exact-count runs
        # on the legacy path; the test rides on the resumed history so it spans
        # incremental cycle calls.
        if (
            early_stop
            and active_state.cycle > _EARLY_STOP_MIN_CYCLES
            and prev_chi2 not in (None, 0)
        ):
            improvement = (prev_chi2 - chi2) / prev_chi2
            # A non-finite improvement means χ² blew up (NaN/inf); treat that,
            # like a rising χ², as divergence so the loop bails rather than
            # spinning to the cycle cap (every NaN comparison is False).
            if not np.isfinite(improvement) or improvement < _EARLY_STOP_CHI2_REL_TOL:
                diverged = not np.isfinite(improvement) or improvement < 0.0
                converged = not diverged
                stop_reason = "diverged" if diverged else "converged"
                break
        prev_chi2 = chi2

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
            "maxent_stop_reason": stop_reason,
            "maxent_converged": bool(converged),
            "maxent_diverged": bool(diverged),
        }
    )
    return MaxEntResult(
        frequencies_mhz=np.asarray(frequencies, dtype=float),
        spectrum=np.asarray(active_state.spectrum, dtype=float),
        state=active_state,
        diagnostics=active_state.diagnostics,
        metadata=metadata,
        stop_reason=stop_reason,
        converged=converged,
        diverged=diverged,
        maxent_input=maxent_input,
    )


def maxent(
    run: Run,
    config: MaxEntConfig | dict | None = None,
    *,
    cycles: int | None = None,
    state: MaxEntState | None = None,
    early_stop: bool = True,
    progress_callback: MaxEntProgressCallback | None = None,
    cancel_callback: MaxEntCancelCallback | None = None,
) -> MaxEntResult:
    """Compute a grouped MaxEnt spectrum for *run*.

    *cycles* is an upper bound: the run stops early at the χ² plateau unless
    *early_stop* is ``False``.
    """
    resolved_config = config if isinstance(config, MaxEntConfig) else MaxEntConfig.from_dict(config)
    _check_cancel(cancel_callback)
    maxent_input = build_maxent_input(run, resolved_config)
    return run_cycles(
        maxent_input,
        resolved_config,
        state=state,
        cycles=cycles,
        early_stop=early_stop,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
