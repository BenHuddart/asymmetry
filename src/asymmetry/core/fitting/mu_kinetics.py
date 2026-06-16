"""Pulsed-source fast-muonium reaction kinetics.

At a pulsed source the small-transverse-field muonium (Mu) precession signal of a
*fast-reacting* sample has decayed before the first good bin (EMU: t_good ≈ 0.203
µs). Over the surviving window the signal is

    A(t) = A_Mu · exp(−λ_Mu·t) · cos(2π f_Mu t + φ)   [+ slow diamagnetic]

and, re-centred on the first good time t_g, the initial amplitude and the rate
trade off through the conserved product ``A_Mu·exp(−λ_Mu·t_g)`` — a per-run fit
cannot separate them and rails to the amplitude bound. See
``docs/porting/pulsed-fast-mu-kinetics/`` for the study.

The physical key: ``A_Mu`` is the **initial muonium fraction**, a property of muon
thermalisation in the solvent — *common across the concentration/temperature
series*; the maleic-acid scavenger changes only the rate ``λ_Mu``. So this module
**shares ``A_Mu`` (and the phase) across the series** while letting ``λ_Mu`` vary
per run: the slow members pin the shared amplitude, which then forces ``λ_Mu`` for
the truncated fast members. This is the standard muonium-chemistry "fraction"
method, realised over the implemented asymmetry-domain global fit
(:func:`asymmetry.core.fitting.asymmetry_global.fit_global`).

The kinetics are pseudo-first-order, ``λ_Mu = λ₀ + k_Mu·[x]`` (slope = the
bimolecular rate constant ``k_Mu``; intercept = the water background ``λ₀``), and
``k_Mu(T)`` follows the Arrhenius form the muon-school guide states verbatim,
``log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)`` → activation energy ``E``.

Public surface:

* :func:`fit_mu_relaxation_series` — per-run ``λ_Mu`` from a 2 G Mu series with
  ``A_Mu`` shared (the degeneracy break).
* :func:`fit_bimolecular_rate` — the ``λ_Mu = λ₀ + k_Mu·[x]`` line.
* :func:`fit_arrhenius` — the activation energy from ``k_Mu(T)``.
* :func:`mu_relaxation_from_amplitude` — single-run cross-check: ``λ_Mu`` with
  ``A_Mu`` *fixed* to a reference (the analytic limit of the shared fit).

All functions are Qt-free and work on percent-scale asymmetry ``MuonDataset``\\s.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.asymmetry_global import GlobalFitResult, fit_global
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

__all__ = [
    "MuRelaxationSeriesResult",
    "BimolecularRateResult",
    "ArrheniusResult",
    "fit_mu_relaxation_series",
    "fit_bimolecular_rate",
    "fit_arrhenius",
    "mu_relaxation_from_amplitude",
]

#: Molar gas constant (J·mol⁻¹·K⁻¹).
R_GAS = 8.314462618

#: The Mu-signal model: a relaxing transverse oscillation (Mu precession) plus a
#: slowly-relaxing diamagnetic background — the guide's "relaxing oscillation".
#: Parameters: A_1 (A_Mu), frequency (f_Mu, fixed), phase, Lambda_2 (λ_Mu),
#: A_3/Lambda_3 (diamagnetic amplitude/relaxation).
_MU_MODEL_EXPRESSION = "Oscillatory*Exponential + Exponential"

_LARGE_RATE = 500.0  # µs⁻¹ upper bound; degeneracy shows as a rail toward this.


@dataclass
class MuRelaxationSeriesResult:
    """Per-run ``λ_Mu`` from a shared-amplitude fit of a Mu series.

    ``lambda_mu`` / ``lambda_mu_error`` are aligned with the input dataset order.
    A non-finite ``lambda_mu_error`` entry flags a member whose rate iminuit could
    not bound (the degenerate case when ``A_Mu`` is *not* shared).
    """

    success: bool
    lambda_mu: list[float]
    lambda_mu_error: list[float]
    shared_amplitude: float
    shared_amplitude_error: float
    shared_phase: float
    shared_phase_error: float
    reduced_chi_squared: float
    global_result: GlobalFitResult = field(repr=False)


@dataclass
class BimolecularRateResult:
    """Pseudo-first-order line ``λ_Mu = λ₀ + k_Mu·[x]``.

    ``k_mu`` is in µs⁻¹ per relative-concentration unit (the corpus supplies no
    molarity — see GROUND_TRUTH §9); ``lambda0`` is the water background in µs⁻¹.
    ``reduced_chi_squared`` is ``nan`` when the fit was unweighted (no point
    errors supplied).
    """

    k_mu: float
    k_mu_error: float
    lambda0: float
    lambda0_error: float
    reduced_chi_squared: float


@dataclass
class ArrheniusResult:
    """Activation energy from ``log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)``.

    ``reduced_chi_squared`` is ``nan`` when the fit was unweighted (no ``k_errors``).
    """

    activation_energy: float  # in ``energy_unit`` (default kJ·mol⁻¹)
    activation_energy_error: float
    log10_a: float  # log₁₀ of the Arrhenius pre-exponential factor A
    log10_a_error: float
    energy_unit: str
    reduced_chi_squared: float


# ---------------------------------------------------------------------------
# Shared-amplitude per-run λ_Mu (the degeneracy break)
# ---------------------------------------------------------------------------


def _amplitude_seed(dataset: MuonDataset) -> float:
    """Robust oscillation-amplitude seed from a dataset's early asymmetry."""
    asym = np.asarray(dataset.asymmetry, dtype=float)
    if asym.size == 0:
        return 5.0
    early = asym[: max(1, asym.size // 4)]
    spread = float(np.percentile(np.abs(early - early.mean()), 95))
    return float(np.clip(spread, 1.0, 50.0))


def _seed_parameter_set(
    datasets: Sequence[MuonDataset],
    *,
    f_mu: float,
    initial: dict[str, float] | None,
) -> ParameterSet:
    """Build the broadcast seed ParameterSet for the Mu-signal model."""
    overrides = dict(initial or {})
    amp = overrides.get("A_1", _amplitude_seed(datasets[0]))
    dia = overrides.get("A_3", amp)
    params = [
        Parameter("A_1", amp, min=0.0, max=100.0),
        Parameter("frequency", overrides.get("frequency", f_mu), fixed=True),
        Parameter("phase", overrides.get("phase", 0.0)),
        Parameter("Lambda_2", overrides.get("Lambda_2", 1.0), min=0.0, max=_LARGE_RATE),
        Parameter("A_3", dia, min=-100.0, max=100.0),
        Parameter("Lambda_3", overrides.get("Lambda_3", 0.2), min=0.0, max=_LARGE_RATE),
    ]
    return ParameterSet(params)


def _finite_error(value: float | None) -> float:
    """Map a missing/invalid iminuit error to ``+inf`` (the degenerate signal)."""
    if value is None or not math.isfinite(value):
        return math.inf
    return float(value)


def _reject_unbounded_errors(errors: Sequence[float], *, name: str, hint: str) -> None:
    """Raise a pointed error if any weight is non-finite (an unbounded member).

    :func:`fit_mu_relaxation_series` returns ``+inf`` for a member whose rate the
    fit could not bound; feeding that straight into a trend fit would otherwise
    surface only the low-level "sigma must be finite" message. Point at the real
    cause instead.
    """
    bad = [i for i, e in enumerate(errors) if e is None or not math.isfinite(e)]
    if bad:
        raise ValueError(
            f"{name} has non-finite entries at index/indices {bad}: those series "
            f"members were not bounded by the relaxation fit, so they carry no "
            f"usable weight. {hint}"
        )


def fit_mu_relaxation_series(
    datasets: Sequence[MuonDataset],
    *,
    f_mu: float = 2.78,
    share_amplitude: bool = True,
    share_phase: bool = True,
    t_min: float | None = None,
    t_max: float | None = None,
    initial: dict[str, float] | None = None,
    minos: bool = False,
) -> MuRelaxationSeriesResult:
    """Fit a 2 G Mu series, recovering per-run ``λ_Mu`` past the truncation degeneracy.

    The series is fitted simultaneously (one shared muonium amplitude ``A_Mu`` and,
    by default, one shared phase) while ``λ_Mu`` and the diamagnetic background are
    estimated per run. The slow members pin the shared ``A_Mu``; that pinned value
    then forces ``λ_Mu`` for the fast members whose Mu signal decayed before the
    first good bin — which a free per-run fit cannot do (set ``share_amplitude=
    False`` to reproduce that degenerate baseline).

    Parameters
    ----------
    datasets
        2 G Mu asymmetry datasets at one temperature, ordered by concentration
        (the result's ``lambda_mu`` is returned in the same order).
    f_mu
        Mu precession frequency (MHz), held fixed (``γ_Mu·B`` ≈ 2.78 MHz at 2 G).
    share_amplitude, share_phase
        Whether ``A_Mu`` / the phase are shared globals (the default) or estimated
        per run.
    t_min, t_max
        Optional fit-window clip applied to every dataset (default: each dataset's
        good-bin window).
    initial
        Optional seed overrides for any of ``A_1, phase, Lambda_2, A_3, Lambda_3``.
    minos
        Run MINOS for asymmetric errors on top of HESSE.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("fit_mu_relaxation_series requires at least one dataset")

    model_fn = CompositeModel.from_expression(_MU_MODEL_EXPRESSION).function
    seed = _seed_parameter_set(datasets, f_mu=f_mu, initial=initial)

    global_params: list[str] = []
    local_params: list[str] = ["Lambda_2", "A_3", "Lambda_3"]
    (global_params if share_amplitude else local_params).append("A_1")
    (global_params if share_phase else local_params).append("phase")

    result = fit_global(
        datasets,
        model_fn,
        global_params=global_params,
        local_params=local_params,
        initial_params=seed,
        t_min=t_min,
        t_max=t_max,
        minos=minos,
    )

    lambdas: list[float] = []
    lambda_errors: list[float] = []
    amplitudes: list[float] = []
    phases: list[float] = []
    for index in range(len(datasets)):
        fit = result.dataset_results[index]
        lambdas.append(float(fit.parameters["Lambda_2"].value))
        lambda_errors.append(_finite_error(fit.uncertainties.get("Lambda_2")))
        amplitudes.append(float(fit.parameters["A_1"].value))
        phases.append(float(fit.parameters["phase"].value))

    if share_amplitude:
        shared_amp = float(result.global_parameters["A_1"].value)
        shared_amp_err = _finite_error(result.global_uncertainties.get("A_1"))
    else:
        shared_amp = float(np.mean(amplitudes))
        shared_amp_err = math.nan
    if share_phase:
        shared_phase = float(result.global_parameters["phase"].value)
        shared_phase_err = _finite_error(result.global_uncertainties.get("phase"))
    else:
        shared_phase = float(np.mean(phases))
        shared_phase_err = math.nan

    return MuRelaxationSeriesResult(
        success=result.success,
        lambda_mu=lambdas,
        lambda_mu_error=lambda_errors,
        shared_amplitude=shared_amp,
        shared_amplitude_error=shared_amp_err,
        shared_phase=shared_phase,
        shared_phase_error=shared_phase_err,
        reduced_chi_squared=result.reduced_chi_squared,
        global_result=result,
    )


def mu_relaxation_from_amplitude(
    dataset: MuonDataset,
    *,
    reference_amplitude: float,
    f_mu: float = 2.78,
    phase: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    """Single-run ``λ_Mu`` with ``A_Mu`` *fixed* to a reference amplitude.

    The analytic limit of the shared fit: fixing ``A_Mu`` to the known muonium
    fraction (``reference_amplitude``, e.g. the shared value from
    :func:`fit_mu_relaxation_series`) breaks the per-run degeneracy directly, so a
    single truncated run yields ``λ_Mu``. Returns ``(λ_Mu, σ_λ)`` in µs⁻¹. Use as
    an independent cross-check of the series fit, or as a manual fallback when a
    full series is unavailable but the Mu fraction is known.
    """
    model_fn = CompositeModel.from_expression(_MU_MODEL_EXPRESSION).function
    params = [
        Parameter("A_1", float(reference_amplitude), fixed=True),
        Parameter("frequency", f_mu, fixed=True),
        Parameter("phase", 0.0 if phase is None else float(phase), fixed=phase is not None),
        Parameter("Lambda_2", 1.0, min=0.0, max=_LARGE_RATE),
        Parameter("A_3", _amplitude_seed(dataset), min=-100.0, max=100.0),
        Parameter("Lambda_3", 0.2, min=0.0, max=_LARGE_RATE),
    ]
    local = ["Lambda_2", "A_3", "Lambda_3"]
    if phase is None:
        local.append("phase")
    result = fit_global(
        [dataset],
        model_fn,
        global_params=[],
        local_params=local,
        initial_params=ParameterSet(params),
        t_min=t_min,
        t_max=t_max,
    )
    fit = result.dataset_results[0]
    return float(fit.parameters["Lambda_2"].value), _finite_error(fit.uncertainties.get("Lambda_2"))


# ---------------------------------------------------------------------------
# Trend fits (weighted linear least squares)
# ---------------------------------------------------------------------------


def _weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None
) -> tuple[float, float, float, float, float]:
    """Weighted straight-line fit ``y = intercept + slope·x``.

    Returns ``(slope, intercept, slope_err, intercept_err, reduced_chi2)``. With
    ``sigma=None`` the fit is unweighted and the reduced χ² is ``nan``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("A straight-line fit needs at least two points")
    if y.size != n:
        raise ValueError("x and y must have the same length")

    if sigma is None:
        weights = np.ones(n)
        weighted = False
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.size != n:
            raise ValueError("sigma must match the number of points")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma must be finite and positive on every point")
        weights = 1.0 / sigma**2
        weighted = True

    s = float(np.sum(weights))
    sx = float(np.sum(weights * x))
    sy = float(np.sum(weights * y))
    sxx = float(np.sum(weights * x * x))
    sxy = float(np.sum(weights * x * y))
    denom = s * sxx - sx * sx
    if denom == 0.0:
        raise ValueError("Degenerate x values: cannot fit a line (need ≥2 distinct x)")

    slope = (s * sxy - sx * sy) / denom
    intercept = (sxx * sy - sx * sxy) / denom
    slope_err = math.sqrt(s / denom)
    intercept_err = math.sqrt(sxx / denom)

    model = intercept + slope * x
    if weighted:
        chi2 = float(np.sum(weights * (y - model) ** 2))
        reduced = chi2 / max(n - 2, 1)
    else:
        reduced = math.nan
    return slope, intercept, slope_err, intercept_err, reduced


def fit_bimolecular_rate(
    concentrations: Sequence[float],
    lambdas: Sequence[float],
    lambda_errors: Sequence[float] | None = None,
) -> BimolecularRateResult:
    """Fit ``λ_Mu = λ₀ + k_Mu·[x]`` → bimolecular rate ``k_Mu`` and background ``λ₀``.

    ``k_Mu`` is the slope (µs⁻¹ per relative-concentration unit); ``λ₀`` the
    intercept (the water/solvent background relaxation).
    """
    concentrations = list(concentrations)
    lambdas = list(lambdas)
    if len(concentrations) != len(lambdas):
        raise ValueError("concentrations and lambdas must have the same length")
    sigma = None if lambda_errors is None else list(lambda_errors)
    if sigma is not None:
        if len(sigma) != len(lambdas):
            raise ValueError("lambda_errors must match the number of points")
        _reject_unbounded_errors(
            sigma,
            name="lambda_errors",
            hint=(
                "Add a slow, well-surviving reference (e.g. deoxygenated water at "
                "[x]=0) to pin the shared A_Mu, or drop the unbounded members "
                "before fitting the rate."
            ),
        )

    slope, intercept, slope_err, intercept_err, reduced = _weighted_linear_fit(
        np.asarray(concentrations),
        np.asarray(lambdas),
        None if sigma is None else np.asarray(sigma),
    )
    return BimolecularRateResult(
        k_mu=slope,
        k_mu_error=slope_err,
        lambda0=intercept,
        lambda0_error=intercept_err,
        reduced_chi_squared=reduced,
    )


def fit_arrhenius(
    temperatures: Sequence[float],
    k_values: Sequence[float],
    k_errors: Sequence[float] | None = None,
    *,
    energy_unit: str = "kJ/mol",
) -> ArrheniusResult:
    """Fit ``log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)`` → activation energy ``E``.

    Linearises in ``(1/T, log₁₀ k)`` (the guide's exact form); the slope is
    ``−E/(ln10·R)``. ``E`` is returned in ``energy_unit`` — ``"kJ/mol"`` (default)
    or ``"J/mol"``. Point errors ``k_errors`` are propagated as
    ``σ(log₁₀ k) = σ_k/(k·ln10)``.
    """
    temperatures = np.asarray(list(temperatures), dtype=float)
    k_values = np.asarray(list(k_values), dtype=float)
    if temperatures.size != k_values.size:
        raise ValueError("temperatures and k_values must have the same length")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive (kelvin)")
    if np.any(k_values <= 0.0):
        raise ValueError("k_values must be positive to take log₁₀ k")

    unit_scale = {"kj/mol": 1.0e-3, "j/mol": 1.0}
    key = energy_unit.lower()
    if key not in unit_scale:
        raise ValueError(f"energy_unit must be one of {sorted(unit_scale)}, got {energy_unit!r}")

    x = 1.0 / temperatures
    y = np.log10(k_values)
    if k_errors is None:
        sigma_y = None
    else:
        k_err = np.asarray(list(k_errors), dtype=float)
        if k_err.size != k_values.size:
            raise ValueError("k_errors must match the number of points")
        _reject_unbounded_errors(
            k_err.tolist(),
            name="k_errors",
            hint="Drop the temperatures whose k_Mu could not be bounded.",
        )
        sigma_y = k_err / (k_values * math.log(10.0))

    slope, intercept, slope_err, intercept_err, reduced = _weighted_linear_fit(x, y, sigma_y)

    # slope = −E / (ln10 · R)  →  E = −slope · ln10 · R   (in J/mol).
    ln10_r = math.log(10.0) * R_GAS
    energy_j = -slope * ln10_r
    energy_err_j = slope_err * ln10_r
    scale = unit_scale[key]
    return ArrheniusResult(
        activation_energy=energy_j * scale,
        activation_energy_error=energy_err_j * scale,
        log10_a=intercept,
        log10_a_error=intercept_err,
        energy_unit=energy_unit,
        reduced_chi_squared=reduced,
    )
