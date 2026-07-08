"""Frequency-domain fitting helpers."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

FREQUENCY_COMPONENT_NAMES: tuple[str, ...] = (
    "GaussianPeak",
    "LorentzianPeak",
    "ConstantBackground",
    "LinearBackground",
)


def default_frequency_model() -> CompositeModel:
    """Return the default V1 frequency-domain peak model."""
    return CompositeModel(["GaussianPeak", "ConstantBackground"], operators=["+"])


def frequency_mhz_to_field_gauss(value_mhz: float) -> float:
    """Convert a muon precession frequency or width in MHz to Gauss."""
    scale = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    return float(value_mhz) / scale


def field_gauss_to_frequency_mhz(value_gauss: float) -> float:
    """Convert a magnetic field or width in Gauss to MHz."""
    scale = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    return float(value_gauss) * scale


def append_frequency_field_derived_parameters(
    parameters: ParameterSet,
    uncertainties: dict[str, float] | None = None,
) -> tuple[ParameterSet, dict[str, float]]:
    """Return a copy with derived centre/width field parameters appended.

    The canonical fitted quantities remain ``nu0`` and ``fwhm`` in MHz.  The
    derived ``B0`` and ``Bwid`` values are appended for trend tables and exports.
    """
    result = ParameterSet()
    result_uncertainties: dict[str, float] = dict(uncertainties or {})

    for parameter in parameters:
        result.add(
            Parameter(
                name=parameter.name,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
            )
        )

    existing = {parameter.name for parameter in result}
    if "nu0" in parameters and "B0" not in existing:
        result.add(Parameter("B0", value=frequency_mhz_to_field_gauss(parameters["nu0"].value)))
        if "nu0" in result_uncertainties:
            result_uncertainties["B0"] = abs(
                frequency_mhz_to_field_gauss(result_uncertainties["nu0"])
            )
    if "fwhm" in parameters and "Bwid" not in existing:
        result.add(Parameter("Bwid", value=frequency_mhz_to_field_gauss(parameters["fwhm"].value)))
        if "fwhm" in result_uncertainties:
            result_uncertainties["Bwid"] = abs(
                frequency_mhz_to_field_gauss(result_uncertainties["fwhm"])
            )
    return result, result_uncertainties


def _peak_component_param_groups(model: CompositeModel) -> list[dict[str, str]]:
    """Return the (composite) ``height``/``nu0``/``fwhm`` names of each peak term.

    A "peak" component is one whose base parameters include ``height``,
    ``nu0`` and ``fwhm`` (``GaussianPeak`` / ``LorentzianPeak``).  The composite
    model suffixes repeated parameters (``nu0_1``, ``nu0_2``, …), so the names
    are read from the model's per-component parameter mapping — in model order,
    which is how detected peaks are assigned to components.
    """
    mappings = getattr(model, "_param_mappings", None)
    components = getattr(model, "components", None)
    groups: list[dict[str, str]] = []
    if not mappings or not components:
        return groups
    for component, mapping in zip(components, mappings, strict=False):
        base = set(getattr(component, "param_names", []))
        if not {"height", "nu0", "fwhm"}.issubset(base):
            continue
        names = {key: mapping.get(key) for key in ("height", "nu0", "fwhm")}
        if all(isinstance(name, str) and name for name in names.values()):
            groups.append(names)  # type: ignore[arg-type]
    return groups


def _detect_top_n_local_maxima(
    x: np.ndarray, y: np.ndarray, candidates: np.ndarray, baseline: float, df: float, n: int
) -> list[tuple[float, float, float]]:
    """Return up to *n* ``(nu0, height, fwhm)`` seeds for the most prominent peaks.

    Trusts the user's declared component count: this is *not* the noise-floor /
    sidelobe-gated detector used by the fit wizard (which is deliberately
    conservative and would drop a weak-but-real second line). It takes the *n*
    highest-prominence local maxima inside the guarded band, strongest-first,
    with sub-bin (parabolic) centre refinement and a half-maximum width.
    """
    from scipy.signal import find_peaks, peak_widths

    indices, properties = find_peaks(y, prominence=0.0)
    if indices.size == 0:
        return []
    in_band = np.isin(indices, candidates)
    indices = indices[in_band]
    prominences = np.asarray(properties["prominences"], dtype=float)[in_band]
    if indices.size == 0:
        return []

    order = np.argsort(prominences)[::-1][: max(0, int(n))]
    selected = indices[order]
    widths_samples, *_ = peak_widths(y, selected, rel_height=0.5)

    seeds: list[tuple[float, float, float]] = []
    for k, idx in enumerate(selected):
        idx = int(idx)
        # 3-point parabolic vertex for a sub-bin centre.
        nu0 = float(x[idx])
        if 1 <= idx < x.size - 1:
            y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if denom != 0.0:
                nu0 = float(x[idx]) + 0.5 * (y0 - y2) / denom * df
        height = max(float(y[idx]) - baseline, 1e-12)
        width = max(float(widths_samples[k]) * df, np.finfo(float).eps)
        seeds.append((nu0, height, width))
    return seeds


def _seed_multiple_peaks(
    x: np.ndarray,
    y: np.ndarray,
    model: CompositeModel,
    peak_groups: list[dict[str, str]],
    *,
    guard_bins: int,
    guard_freq_mhz: float,
) -> dict[str, float]:
    """Seed every peak term of a multi-peak spectral model from the spectrum.

    Detected peaks are assigned strongest-first to the peak components in model
    order.  When fewer peaks are detected than the model declares, the extra
    components are spread across the spectrum window (never left at the
    off-screen ``nu0`` default) so their preview stays visible — the same
    failure mode the single-peak seeding fixes, one component over.
    """
    candidates = np.arange(x.size)
    df = 1.0
    if x.size >= 2:
        df = float(np.median(np.diff(np.sort(x))))
        guard = max(guard_bins * df, guard_freq_mhz)
        guarded = np.flatnonzero(np.abs(x) >= guard)
        if guarded.size > 0:
            candidates = guarded
    if df <= 0.0:
        df = 1.0

    baseline = float(np.nanmedian(y[candidates]))
    n = len(peak_groups)
    detected = _detect_top_n_local_maxima(x, y, candidates, baseline, df, n)

    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    fallback_fwhm = max(span / 20.0, df, np.finfo(float).eps)
    fallback_height = max(detected[0][1] * 0.25 if detected else 1e-12, 1e-12)

    seeds: dict[str, float] = {"bg": baseline}
    for i, group in enumerate(peak_groups):
        if i < len(detected):
            nu0, height, width = detected[i]
        else:
            # Distribute the undetected components evenly across the window so
            # each stays on-screen for the preview and gives the optimiser a
            # sensible, non-degenerate start.
            nu0 = lo + (i + 0.5) / n * span if span > 0 else lo
            height, width = fallback_height, fallback_fwhm
        seeds[group["nu0"]] = nu0
        seeds[group["height"]] = height
        seeds[group["fwhm"]] = width

    if "slope" in model.param_names:
        edge = max(1, min(10, x.size // 10))
        dx = float(np.mean(x[-edge:]) - np.mean(x[:edge]))
        dy = float(np.mean(y[-edge:]) - np.mean(y[:edge]))
        seeds["slope"] = dy / dx if abs(dx) > 1e-12 else 0.0

    return {name: value for name, value in seeds.items() if name in model.param_names}


def seed_peak_parameters_from_dataset(
    dataset, model: CompositeModel, *, guard_bins: int = 3, guard_freq_mhz: float = 2.0
) -> dict[str, float]:
    """Return simple peak/background seeds for one displayed Fourier spectrum.

    The dominant-magnitude bin of a (Power)^1/2 spectrum is usually the
    DC/apodisation spike, not the physical precession peak, so the peak
    search excludes a low-frequency guard band before taking the argmax.
    The guard width is ``max(guard_bins * df, guard_freq_mhz)``, where ``df``
    is the spectrum's bin spacing (a proxy for ``1/T_obs``); it falls back to
    the unguarded global argmax if the guard would empty the search array.

    When the model declares two or more peak components, seeding delegates to
    :func:`_seed_multiple_peaks`, which assigns the strongest detected peaks to
    each component (see there for the under-detection fallback).
    """
    x = np.asarray(getattr(dataset, "time", []), dtype=float)
    y = np.asarray(getattr(dataset, "asymmetry", []), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0 or y.size == 0:
        return {}

    peak_groups = _peak_component_param_groups(model)
    if len(peak_groups) >= 2:
        return _seed_multiple_peaks(
            x, y, model, peak_groups, guard_bins=guard_bins, guard_freq_mhz=guard_freq_mhz
        )

    candidates = np.arange(x.size)
    if x.size >= 2:
        df = float(np.median(np.diff(np.sort(x))))
        guard = max(guard_bins * df, guard_freq_mhz)
        guarded = np.flatnonzero(np.abs(x) >= guard)
        if guarded.size > 0:
            candidates = guarded
    # ``candidates`` is never empty here: it starts as the full (non-empty,
    # per the size check above) index range and is only narrowed when the
    # narrowed set itself is non-empty.
    candidate_mask = np.zeros(x.size, dtype=bool)
    candidate_mask[candidates] = True

    baseline = float(np.nanmedian(y[candidates]))
    peak_index = int(candidates[np.nanargmax(y[candidates])])

    peak_y = float(y[peak_index])
    height = max(peak_y - baseline, 1e-12)
    nu0 = float(x[peak_index])
    half_height = baseline + 0.5 * height
    # Restrict the half-max crossing search to the same guarded region as the
    # peak search — otherwise a DC/apodisation spike that also exceeds
    # half_height (common, since it dwarfs the physical peak) drags the FWHM
    # span out to the DC spike's edge, producing a wildly inflated seed even
    # though nu0 itself correctly skipped it.
    above = np.flatnonzero((y >= half_height) & candidate_mask)
    if above.size >= 2:
        fwhm = max(float(x[above[-1]] - x[above[0]]), np.finfo(float).eps)
    elif x.size >= 2:
        fwhm = max(float((np.nanmax(x) - np.nanmin(x)) / 20.0), np.finfo(float).eps)
    else:
        fwhm = 0.1

    seeds = {"height": height, "nu0": nu0, "fwhm": fwhm, "bg": baseline}
    if "slope" in model.param_names:
        edge = max(1, min(10, x.size // 10))
        dx = float(np.mean(x[-edge:]) - np.mean(x[:edge]))
        dy = float(np.mean(y[-edge:]) - np.mean(y[:edge]))
        seeds["slope"] = dy / dx if abs(dx) > 1e-12 else 0.0
    return {name: value for name, value in seeds.items() if name in model.param_names}


__all__ = [
    "FREQUENCY_COMPONENT_NAMES",
    "append_frequency_field_derived_parameters",
    "default_frequency_model",
    "field_gauss_to_frequency_mhz",
    "frequency_mhz_to_field_gauss",
    "seed_peak_parameters_from_dataset",
]
