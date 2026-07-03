"""Fit-panel seeding helpers (phase/background/N0 seed math).

Split out of ``fit_panel.py`` (Phase 2 mechanical split). Pure helpers; no
intra-package dependencies.
"""

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import (
    split_parameter_name,
)
from asymmetry.core.fourier.fft import estimate_fft_phase, fft_complex_asymmetry
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def _field_value_overrides(model: CompositeModel, field_gauss: float) -> dict[str, float]:
    """Return a dict overriding field-like defaults with *field_gauss*.

    Only overrides parameters whose base name is ``"field"`` or ``"B_L"``
    and only when *field_gauss* is non-zero.
    """
    if field_gauss == 0.0:
        return {}
    overrides: dict[str, float] = {}
    for pname in model.param_names:
        base_name, _index = split_parameter_name(pname)
        if base_name in {"field", "B_L"}:
            overrides[pname] = field_gauss
    return overrides


def _seed_group_background_and_n0(
    counts: np.ndarray,
    *,
    time: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return heuristic grouped-count seeds for background, N0, and amplitude."""
    count_arr = np.asarray(counts, dtype=float)
    time_arr = np.asarray(time, dtype=float) if time is not None else None
    if time_arr is not None and time_arr.shape == count_arr.shape:
        finite_mask = np.isfinite(count_arr) & np.isfinite(time_arr)
        count_arr = count_arr[finite_mask]
        time_arr = time_arr[finite_mask]
    else:
        time_arr = None
        count_arr = count_arr[np.isfinite(count_arr)]

    if count_arr.size == 0:
        return 0.0, 100.0, 0.2

    if time_arr is None:
        time_arr = np.arange(count_arr.size, dtype=float)
        background_scale = np.ones_like(time_arr)
    else:
        background_scale = np.exp(time_arr / float(MUON_LIFETIME_US))

    def _window_mask(sample_time: np.ndarray, *, tail: bool) -> np.ndarray:
        if sample_time.size <= 1:
            return np.ones(sample_time.size, dtype=bool)

        start = float(np.min(sample_time))
        stop = float(np.max(sample_time))
        span = max(0.0, stop - start)
        width = min(1.0, span * 0.25) if span > 0.0 else 0.0
        if width > 0.0:
            mask = sample_time >= (stop - width) if tail else sample_time <= (start + width)
            if np.count_nonzero(mask) >= min(5, sample_time.size):
                return mask

        window_size = min(sample_time.size, max(1, int(np.ceil(sample_time.size * 0.2))))
        mask = np.zeros(sample_time.size, dtype=bool)
        if tail:
            mask[-window_size:] = True
        else:
            mask[:window_size] = True
        return mask

    late_mask = _window_mask(time_arr, tail=True)
    early_mask = _window_mask(time_arr, tail=False)

    raw_like_counts = count_arr / background_scale
    if time is not None and np.count_nonzero(late_mask) >= 2:
        late_time = np.asarray(time_arr[late_mask], dtype=float)
        late_raw_like = np.asarray(raw_like_counts[late_mask], dtype=float)
        design = np.column_stack(
            [np.exp(-late_time / float(MUON_LIFETIME_US)), np.ones_like(late_time)]
        )
        coeffs, *_ = np.linalg.lstsq(design, late_raw_like, rcond=None)
        background = max(float(coeffs[1]), 0.0)
    else:
        background = float(np.mean(raw_like_counts[late_mask]))
    residual = count_arr - background * background_scale
    if not np.any(np.isfinite(residual)):
        return float(background), 100.0, 0.2

    core_mask = (
        early_mask if np.count_nonzero(early_mask) >= 3 else np.ones_like(residual, dtype=bool)
    )
    core_residual = np.asarray(residual[core_mask], dtype=float)
    core_residual = core_residual[np.isfinite(core_residual)]
    if core_residual.size == 0:
        core_residual = np.asarray(residual[np.isfinite(residual)], dtype=float)

    n0 = max(float(np.median(core_residual)), 1.0)
    centered = core_residual - n0
    if centered.size >= 2:
        lower = float(np.percentile(centered, 10.0))
        upper = float(np.percentile(centered, 90.0))
        amplitude_scale = 0.5 * max(upper - lower, 0.0)
    elif centered.size == 1:
        amplitude_scale = abs(float(centered[0]))
    else:
        amplitude_scale = 0.0

    amplitude = amplitude_scale / n0 if n0 > 0.0 else 0.0
    amplitude = float(np.clip(amplitude, 0.01, 1.0))

    return float(background), n0, amplitude


def _group_phase_window_mhz(
    metadata: dict[str, object] | None,
    freqs: np.ndarray,
) -> tuple[float, float | None]:
    """Return a field-guided FFT phase-estimation window for one grouped trace."""
    frequencies = np.asarray(freqs, dtype=float)
    positive = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
    if positive.size == 0:
        return 0.0, None

    field_value = None if metadata is None else metadata.get("field")
    try:
        field_gauss = abs(float(field_value))
    except (TypeError, ValueError):
        return 0.0, None
    if not np.isfinite(field_gauss) or np.isclose(field_gauss, 0.0):
        return 0.0, None

    center = field_gauss * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    half_width = 10.0
    lo = max(0.0, center - half_width)
    hi = center + half_width
    if np.any((positive >= lo) & (positive <= hi)):
        return lo, hi
    return 0.0, None


#: Upper bound on the padded FFT length used for group-phase *seeding*.
#: Zero-padding only interpolates the spectrum, so capping it for very large
#: (high-resolution) histograms leaves the peak-phase seed essentially
#: unchanged while keeping the per-selection cost bounded (avoids multi-second
#: hangs when seeding is refreshed on every dataset/selection change).
_MAX_PHASE_SEED_FFT_POINTS = 1 << 17  # 131072


def _bounded_phase_seed_padding(n_points: int, *, desired: int = 8) -> int:
    """Return a padding factor capped so the seed FFT stays bounded in size."""
    if n_points <= 0:
        return 1
    max_factor = max(1, _MAX_PHASE_SEED_FFT_POINTS // int(n_points))
    return max(1, min(int(desired), int(max_factor)))


def _seed_group_phase_degrees(grouped_groups: list[object]) -> dict[str, float]:
    """Return the FFT-estimated absolute phase (degrees) of each group's oscillation."""
    phase_degrees_by_group: dict[str, float] = {}
    for group in grouped_groups:
        group_id = str(getattr(group, "group_id", ""))
        time = np.asarray(getattr(group, "time", []), dtype=float)
        counts = np.asarray(getattr(group, "counts", []), dtype=float)
        if time.size < 4 or counts.size != time.size:
            phase_degrees_by_group[group_id] = 0.0
            continue

        finite_mask = np.isfinite(time) & np.isfinite(counts)
        time = time[finite_mask]
        counts = counts[finite_mask]
        if time.size < 4:
            phase_degrees_by_group[group_id] = 0.0
            continue

        error = np.asarray(getattr(group, "error", np.ones_like(counts)), dtype=float)
        if error.shape != counts.shape:
            error = np.ones_like(counts, dtype=float)
        else:
            error = error[finite_mask]

        metadata = dict(getattr(group, "metadata", {}) or {})
        background_seed, n0_seed, _amplitude_seed = _seed_group_background_and_n0(
            counts,
            time=time,
        )
        residual = counts - (background_seed * np.exp(time / float(MUON_LIFETIME_US))) - n0_seed
        dataset = MuonDataset(
            time=time.copy(),
            asymmetry=np.asarray(residual, dtype=float),
            error=error.copy(),
            metadata=metadata,
            run=None,
        )
        freqs, spectrum = fft_complex_asymmetry(
            dataset,
            window="none",
            padding_factor=_bounded_phase_seed_padding(time.size),
            subtract_average_signal=True,
        )
        min_frequency, max_frequency = _group_phase_window_mhz(metadata, freqs)
        phase_degrees_by_group[group_id] = estimate_fft_phase(
            freqs,
            spectrum,
            method="peak",
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    return phase_degrees_by_group


def _wrap_phase_rad(phase_deg: float) -> float:
    """Wrap a phase in degrees to radians on ``(-pi, pi]``."""
    return float(np.angle(np.exp(1j * np.deg2rad(phase_deg))))


def _seed_group_absolute_phases(grouped_groups: list[object]) -> dict[str, float]:
    """Return per-group *absolute* phase seeds in radians (wrapped to ``(-pi, pi]``).

    Grouped fits hold the shared model phase at zero, so each group's per-group
    phase nuisance carries the full FFT-estimated phase rather than an offset
    relative to the first group.
    """
    return {
        group_id: _wrap_phase_rad(phase_deg)
        for group_id, phase_deg in _seed_group_phase_degrees(grouped_groups).items()
    }
