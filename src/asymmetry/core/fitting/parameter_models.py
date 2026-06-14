"""Model fitting for fitted parameters as a function of field or temperature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from itertools import product

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.ballistic import lambda_total as ballistic_lambda_total
from asymmetry.core.fitting.composite import (
    QUADRATURE_OPERATOR,
    build_component_expression,
    parse_component_expression,
)
from asymmetry.core.fitting.diffusion import lambda_total as diffusion_lambda_total
from asymmetry.core.fitting.muon_proton import rf_resonance_mup
from asymmetry.core.fitting.muonium import (
    G_E_MHZ_PER_G,
    G_MU_MHZ_PER_G,
    VACUUM_MUONIUM_A_HF_MHZ,
)
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    ParamInfo,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.sc import models as sc_models
from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


@dataclass(frozen=True)
class ParameterModelComponentDefinition:
    """Descriptor for a parameter-vs-x basis function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]
    param_info: dict[str, ParamInfo]
    formula_template: str
    latex_equation: str = ""
    scopes: tuple[str, ...] = ("common",)
    #: For centred-peak components, FWHM = ``fwhm_factor * Bwid`` (None otherwise).
    fwhm_factor: float | None = None
    #: ``True`` for components registered through the user-function facade
    #: (:mod:`asymmetry.core.fitting.user_functions`); see
    #: ``ComponentDefinition.user``.
    user: bool = False


#: Recognised scope tokens for :attr:`ParameterModelComponentDefinition.scopes`
#: (see :func:`component_names_for_x`).
SCOPES: tuple[str, ...] = ("common", "field", "temperature")


def _constant(x: NDArray, c: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(x, dtype=float), float(c), dtype=float)


def _linear(x: NDArray, m: float, b: float) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    return m * xx + b


def _power_law(x: NDArray, a: float, n: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    safe_x = np.maximum(np.abs(xx), 1e-12)
    return a * np.power(safe_x, n) + c


def _polynomial(
    x: NDArray,
    c0: float = 0.0,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    c4: float = 0.0,
    c5: float = 0.0,
) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    # Horner evaluation of the quintic; fix unused coefficients at 0 to fit
    # lower orders (WiMDA "Polynomial fit up to fifth order term").
    result = np.full_like(xx, float(c5))
    for coeff in (c4, c3, c2, c1, c0):
        result = result * xx + float(coeff)
    return result


def _cubic(
    x: NDArray,
    c0: float = 0.0,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    # Horner evaluation of the cubic. This is the WiMDA/Mantid-prescribed ALC
    # background (a curved/sloping baseline a Linear fit cannot match); it is a
    # well-conditioned 4-parameter restriction of the quintic `Polynomial`.
    result = np.full_like(xx, float(c3))
    for coeff in (c2, c1, c0):
        result = result * xx + float(coeff)
    return result


def _power_law_quad_bg(x: NDArray, a: float, n: float, BG: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    safe_x = np.maximum(np.abs(xx), 1e-12)
    return np.hypot(a * np.power(safe_x, n), BG)


def _exp_decay(x: NDArray, a: float, tau: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    tau_value = float(tau)
    tau_sign = 1.0 if tau_value >= 0.0 else -1.0
    tau_safe = tau_sign * max(abs(tau_value), 1e-12)
    exponent = np.clip(-xx / tau_safe, -700, 700)
    return a * np.exp(exponent) + c


def _arrhenius(x: NDArray, a: float, Ea: float) -> NDArray[np.float64]:
    # x is T (K), k_B in meV/K to keep Ea in meV-scale units.
    kb_mev = 8.617333262e-2
    tt = np.asarray(x, dtype=float)
    safe_t = np.maximum(np.abs(tt), 1e-9)
    exponent = np.clip(-Ea / (kb_mev * safe_t), -700, 700)
    return a * np.exp(exponent)


def _critical_divergence(
    x: NDArray, a: float, Tc: float, nu: float, c: float = 0.0
) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    dist = np.maximum(np.abs(xx - Tc), 1e-9)
    return a * np.power(dist, -nu) + c


def _order_parameter(
    x: NDArray, y0: float, Tc: float, beta: float, alpha: float = 1.0
) -> NDArray[np.float64]:
    """Magnetic order-parameter temperature dependence.

    ``y(T) = y0 * [1 - (T/Tc)^alpha]^beta`` for ``0 <= T < Tc``, and ``0`` for
    ``T >= Tc``. The order parameter rises continuously from zero at ``Tc`` to
    ``y0`` at ``T = 0``.

    The amplitude ``y0`` carries the unit of whatever observable is being
    trended (e.g. MHz for a precession frequency, G or T for an internal field,
    % for an asymmetry). ``beta`` is the critical exponent governing the
    near-``Tc`` behaviour; ``alpha`` is a shape exponent describing how the
    curve departs from the pure power law away from ``Tc``. Fixing ``alpha = 1``
    recovers the simple form ``y0 (1 - T/Tc)^beta``.
    """
    tt = np.asarray(x, dtype=float)
    # Clip the exponents and Tc to their physical (non-negative) domain rather
    # than mirroring with abs(): the model is otherwise even in alpha/beta/Tc,
    # so an unbounded fit could converge to a negative value that is numerically
    # degenerate with its positive counterpart. Clipping collapses any negative
    # excursion onto the boundary instead of reporting a meaningless sign.
    alpha_safe = max(float(alpha), 0.0)
    beta_safe = max(float(beta), 0.0)
    Tc_safe = max(float(Tc), 1e-12)
    reduced = np.clip(tt / Tc_safe, 0.0, None)
    base = np.clip(1.0 - np.power(reduced, alpha_safe), 0.0, None)
    return np.where(
        base > 0.0,
        float(y0) * np.power(base, beta_safe),
        0.0,
    )


def _redfield(
    x: NDArray,
    D: float,
    nu: float,
    m: int = 2,
) -> NDArray[np.float64]:
    """0D Redfield contribution in paper notation.

    λ0D(B) = (D^2 / 4) * [2/nu] / [1 + (ω_mu/nu)^m],
    where ω_mu = gamma_mu * B.

    Notes
    -----
    - ``x`` is field in Gauss.
    - ``D`` and ``nu`` are in MHz.
    """
    xx = np.asarray(x, dtype=float)
    omega_mu = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * np.abs(xx)
    nu_safe = max(abs(float(nu)), 1e-12)
    m_safe = max(abs(float(m)), 1e-12)
    denom = 1.0 + np.power(omega_mu / nu_safe, m_safe)
    return ((float(D) ** 2) / 4.0) * (2.0 / nu_safe) / denom


#: (γₑ + γ_μ)/2π in MHz/G — converts a muonium hyperfine constant A_hf (MHz)
#: into the characteristic repolarisation field B₀ = A_hf / this (in G).
#: Shares the per-ratio constants with the Breit-Rabi muonium components so a
#: repolarisation-fitted A_hf stays consistent with a TF-muonium-fitted one.
_ISOTROPIC_MU_B0_DENOM_MHZ_PER_G = G_E_MHZ_PER_G + G_MU_MHZ_PER_G


def isotropic_mu_b0_gauss(A_hf_mhz: float) -> float:
    """Characteristic field B₀ = A_hf/(γₑ + γ_μ) of isotropic muonium, in G."""
    return max(abs(float(A_hf_mhz)), 1e-12) / _ISOTROPIC_MU_B0_DENOM_MHZ_PER_G


def _mu_repolarisation(
    x: NDArray, a_Mu: float, A_hf: float, a_Dia: float = 0.0
) -> NDArray[np.float64]:
    """Time-averaged longitudinal polarization of isotropic muonium.

    In LF only the 2↔4 transition mixes the muon spin states; averaging the
    unresolved fast oscillation leaves ``1 - a24 = (1/2 + r²)/(1 + r²)`` with
    ``r = B/B0`` and ``B0 = A_hf/(γₑ + γ_μ)``. ``x`` is field in Gauss,
    ``A_hf`` in MHz.
    """
    xx = np.asarray(x, dtype=float)
    r2 = (xx / isotropic_mu_b0_gauss(A_hf)) ** 2
    return float(a_Mu) * (0.5 + r2) / (1.0 + r2) + float(a_Dia)


def _lorentzian(x: NDArray, a: float, B0: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    B0_safe = np.sign(B0) * max(abs(B0), 1e-12)
    return a / (1.0 + (xx / B0_safe) ** 2) + c


def _lcr_gaussian(x: NDArray, f: float, B0: float, Bwid: float) -> NDArray[np.float64]:
    """Eq. (4) LCR Gaussian term: lambda_LCR(B) = f * G(B; B0; Bwid)."""
    xx = np.asarray(x, dtype=float)
    bwid_safe = max(abs(float(Bwid)), 1e-12)
    exponent = -0.5 * ((xx - float(B0)) / bwid_safe) ** 2
    return float(f) * np.exp(exponent)


def _lcr_lorentzian(x: NDArray, f: float, B0: float, Bwid: float) -> NDArray[np.float64]:
    """LCR Lorentzian peak centred at B0: f / (1 + ((B - B0)/Bwid)^2).

    The amplitude/centre/width parameter set (f, B0, Bwid) matches
    :func:`_lcr_gaussian` so the two are interchangeable ALC peak shapes.
    """
    xx = np.asarray(x, dtype=float)
    bwid_safe = max(abs(float(Bwid)), 1e-12)
    return float(f) / (1.0 + ((xx - float(B0)) / bwid_safe) ** 2)


def _lambda_bg(x: NDArray, lambda_BG: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(x, dtype=float), float(lambda_BG), dtype=float)


def _diffusion_lf_1d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        diffusion_lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=1),
        dtype=float,
    )


def _diffusion_lf_2d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        diffusion_lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=2),
        dtype=float,
    )


def _diffusion_lf_3d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        diffusion_lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=3),
        dtype=float,
    )


def _ballistic_lf_1d(
    x: NDArray,
    A: float,
    D_hop: float,
) -> NDArray[np.float64]:
    return np.asarray(
        ballistic_lambda_total(x, C=A, D_hop=D_hop, lambda_0D=0.0, n=1),
        dtype=float,
    )


def _ballistic_lf_2d(
    x: NDArray,
    A: float,
    D_hop: float,
) -> NDArray[np.float64]:
    return np.asarray(
        ballistic_lambda_total(x, C=A, D_hop=D_hop, lambda_0D=0.0, n=2),
        dtype=float,
    )


def _ballistic_lf_3d(
    x: NDArray,
    A: float,
    D_hop: float,
) -> NDArray[np.float64]:
    return np.asarray(
        ballistic_lambda_total(x, C=A, D_hop=D_hop, lambda_0D=0.0, n=3),
        dtype=float,
    )


PARAMETER_MODEL_COMPONENTS: dict[str, ParameterModelComponentDefinition] = {
    "Constant": ParameterModelComponentDefinition(
        name="Constant",
        description="c",
        function=_constant,
        param_names=["c"],
        param_defaults={"c": 0.0},
        param_info={"c": get_param_info("c")},
        formula_template="{c}",
        latex_equation=r"y(x) = c",
        scopes=("common",),
    ),
    "Linear": ParameterModelComponentDefinition(
        name="Linear",
        description="m*x + b",
        function=_linear,
        param_names=["m", "b"],
        param_defaults={"m": 1.0, "b": 0.0},
        param_info={"m": get_param_info("m"), "b": get_param_info("b")},
        formula_template="{m}*x + {b}",
        latex_equation=r"y(x) = m x + b",
        scopes=("common", "field", "temperature"),
    ),
    "Polynomial": ParameterModelComponentDefinition(
        name="Polynomial",
        description="c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5",
        function=_polynomial,
        param_names=["c0", "c1", "c2", "c3", "c4", "c5"],
        param_defaults={"c0": 0.0, "c1": 1.0, "c2": 0.0, "c3": 0.0, "c4": 0.0, "c5": 0.0},
        param_info={f"c{k}": get_param_info(f"c{k}") for k in range(6)},
        formula_template="{c0} + {c1}*x + {c2}*x^2 + {c3}*x^3 + {c4}*x^4 + {c5}*x^5",
        latex_equation=r"y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3 + c_4 x^4 + c_5 x^5",
        scopes=("common",),
    ),
    "Cubic": ParameterModelComponentDefinition(
        name="Cubic",
        description="c0 + c1*x + c2*x^2 + c3*x^3",
        function=_cubic,
        param_names=["c0", "c1", "c2", "c3"],
        param_defaults={"c0": 0.0, "c1": 1.0, "c2": 0.0, "c3": 0.0},
        param_info={f"c{k}": get_param_info(f"c{k}") for k in range(4)},
        formula_template="{c0} + {c1}*x + {c2}*x^2 + {c3}*x^3",
        latex_equation=r"y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3",
        scopes=("common", "field", "temperature"),
    ),
    "PowerLaw": ParameterModelComponentDefinition(
        name="PowerLaw",
        description="a*|x|^n + c",
        function=_power_law,
        param_names=["a", "n", "c"],
        param_defaults={"a": 1.0, "n": 1.0, "c": 0.0},
        param_info={
            "a": get_param_info("a"),
            "n": get_param_info("n"),
            "c": get_param_info("c"),
        },
        formula_template="{a}*|x|^{n} + {c}",
        latex_equation=r"y(x) = a \lvert x \rvert^{n} + c",
        scopes=("common",),
    ),
    "PowerLawQuadBG": ParameterModelComponentDefinition(
        name="PowerLawQuadBG",
        description="sqrt((a*|x|^n)^2 + BG^2)",
        function=_power_law_quad_bg,
        param_names=["a", "n", "BG"],
        param_defaults={"a": 1.0, "n": 1.0, "BG": 0.0},
        param_info={
            "a": get_param_info("a"),
            "n": get_param_info("n"),
            "BG": get_param_info("BG"),
        },
        formula_template="sqrt(({a}*|x|^{n})^2 + {BG}^2)",
        latex_equation=r"y(x) = \sqrt{(a \lvert x \rvert^{n})^2 + \mathrm{BG}^2}",
        scopes=("common",),
    ),
    "ExponentialDecay": ParameterModelComponentDefinition(
        name="ExponentialDecay",
        description="a*exp(-x/tau) + c",
        function=_exp_decay,
        param_names=["a", "tau", "c"],
        param_defaults={"a": 1.0, "tau": 10.0, "c": 0.0},
        param_info={
            "a": get_param_info("a"),
            "tau": get_param_info("tau"),
            "c": get_param_info("c"),
        },
        formula_template="{a}*exp(-x/{tau}) + {c}",
        latex_equation=r"y(x) = a e^{-x/\tau} + c",
        scopes=("common",),
    ),
    "Arrhenius": ParameterModelComponentDefinition(
        name="Arrhenius",
        description="a*exp(-Ea/(k_B T))",
        function=_arrhenius,
        param_names=["a", "Ea"],
        param_defaults={"a": 1.0, "Ea": 1.0},
        param_info={"a": get_param_info("a"), "Ea": get_param_info("Ea")},
        formula_template="{a}*exp(-{Ea}/(k_B*T))",
        latex_equation=r"y(T) = a e^{-E_a / (k_B T)}",
        scopes=("temperature",),
    ),
    "CriticalDivergence": ParameterModelComponentDefinition(
        name="CriticalDivergence",
        description="a*|T-Tc|^{-nu} + c",
        function=_critical_divergence,
        param_names=["a", "Tc", "nu", "c"],
        param_defaults={"a": 1.0, "Tc": 10.0, "nu": 1.0, "c": 0.0},
        param_info={
            "a": get_param_info("a"),
            "Tc": get_param_info("Tc"),
            "nu": ParamInfo("nu", "nu", "ν", r"$\\nu$", r"\\nu"),
            "c": get_param_info("c"),
        },
        formula_template="{a}*|x-{Tc}|^(-{nu}) + {c}",
        latex_equation=r"y(T) = a \lvert T - T_c \rvert^{-\nu} + c",
        scopes=("temperature",),
    ),
    "OrderParameter": ParameterModelComponentDefinition(
        name="OrderParameter",
        description="y0*(1 - (T/Tc)^alpha)^beta (0 above Tc)",
        function=_order_parameter,
        param_names=["y0", "Tc", "beta", "alpha"],
        param_defaults={"y0": 1.0, "Tc": 10.0, "beta": 0.36, "alpha": 1.0},
        param_info={
            "y0": get_param_info("y0"),
            "Tc": get_param_info("Tc"),
            "beta": get_param_info("beta"),
            "alpha": get_param_info("alpha"),
        },
        formula_template="{y0}*(1 - (T/{Tc})^{alpha})^{beta}",
        latex_equation=r"y(T) = y_0 \left[1 - (T/T_c)^{\alpha}\right]^{\beta}",
        scopes=("temperature",),
    ),
    "Redfield": ParameterModelComponentDefinition(
        name="Redfield",
        description="(D^2/4) * (2/nu)/(1 + (omega_mu/nu)^m)",
        function=_redfield,
        param_names=["D", "nu", "m"],
        param_defaults={"D": 10.0, "nu": 10.0, "m": 2},
        param_info={
            "D": get_param_info("D"),
            "nu": get_param_info("nu"),
            "m": get_param_info("m"),
        },
        formula_template="(({D}^2)/4)*(2/{nu})/(1 + ((gamma_mu*x)/{nu})^{m})",
        latex_equation=(
            r"\lambda(B) = \frac{D^2}{4} \cdot \frac{2/\nu}{1 + (\gamma_\mu B / \nu)^m}"
        ),
        scopes=("field",),
    ),
    "Lorentzian": ParameterModelComponentDefinition(
        name="Lorentzian",
        description="a/(1 + (B/B0)^2) + c",
        function=_lorentzian,
        param_names=["a", "B0", "c"],
        param_defaults={"a": 1.0, "B0": 100.0, "c": 0.0},
        param_info={
            "a": get_param_info("a"),
            "B0": get_param_info("B0"),
            "c": get_param_info("c"),
        },
        formula_template="{a}/(1 + (x/{B0})^2) + {c}",
        latex_equation=r"y(B) = \frac{a}{1 + (B/B_0)^2} + c",
        scopes=("field",),
    ),
    "MuRepolarisation": ParameterModelComponentDefinition(
        name="MuRepolarisation",
        description="a_Mu*(1/2 + (B/B0)^2)/(1 + (B/B0)^2) + a_Dia, B0 = A_hf/(γₑ+γµ)",
        function=_mu_repolarisation,
        param_names=["a_Mu", "A_hf", "a_Dia"],
        param_defaults={"a_Mu": 15.0, "A_hf": VACUUM_MUONIUM_A_HF_MHZ, "a_Dia": 5.0},
        param_info={
            "a_Mu": get_param_info("a_Mu"),
            "A_hf": get_param_info("A_hf"),
            "a_Dia": get_param_info("a_Dia"),
        },
        formula_template=(
            "{a_Mu}*(0.5 + (x/B0)^2)/(1 + (x/B0)^2) + {a_Dia} with B0 = {A_hf}/(γₑ+γµ)"
        ),
        latex_equation=(
            r"y(B) = a_{\mathrm{Mu}} \frac{\tfrac{1}{2} + (B/B_0)^2}{1 + (B/B_0)^2}"
            r" + a_{\mathrm{Dia}}, \quad B_0 = \frac{A_{\mathrm{hf}}}{\gamma_e + \gamma_\mu}"
        ),
        scopes=("field",),
    ),
    "GaussianLCR": ParameterModelComponentDefinition(
        name="GaussianLCR",
        description="f*G(B; B0; Bwid)",
        function=_lcr_gaussian,
        param_names=["f", "B0", "Bwid"],
        param_defaults={"f": 0.1, "B0": 1000.0, "Bwid": 100.0},
        param_info={
            "f": get_param_info("f"),
            "B0": get_param_info("B0"),
            "Bwid": get_param_info("Bwid"),
        },
        formula_template="{f}*G(x; {B0}; {Bwid})",
        latex_equation=r"\lambda_{LCR}(B) = f\,\exp\left(-\frac{(B-B_0)^2}{2 B_{wid}^2}\right)",
        scopes=("field",),
        fwhm_factor=2.0 * np.sqrt(2.0 * np.log(2.0)),  # ≈ 2.3548
    ),
    "LorentzianLCR": ParameterModelComponentDefinition(
        name="LorentzianLCR",
        description="f/(1 + ((B-B0)/Bwid)^2)",
        function=_lcr_lorentzian,
        param_names=["f", "B0", "Bwid"],
        param_defaults={"f": 0.1, "B0": 1000.0, "Bwid": 100.0},
        param_info={
            "f": get_param_info("f"),
            "B0": get_param_info("B0"),
            "Bwid": get_param_info("Bwid"),
        },
        formula_template="{f}*L(x; {B0}; {Bwid})",
        latex_equation=r"\lambda_{LCR}(B) = \frac{f}{1 + \left((B-B_0)/B_{wid}\right)^2}",
        scopes=("field",),
        fwhm_factor=2.0,  # half-max at |B - B0| = Bwid
    ),
    "DiffusionLF_1D": ParameterModelComponentDefinition(
        name="DiffusionLF_1D",
        description="(A^2/4) J(gamma_e B; n=1, D_2D)",
        function=_diffusion_lf_1d,
        param_names=["A", "D_2D", "D_perp"],
        param_defaults={"A": 1.0, "D_2D": 1.0, "D_perp": 0.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\\it A}", "MHz"),
            "D_2D": get_param_info("D_2D"),
            "D_perp": get_param_info("D_perp"),
        },
        formula_template="(({A}^2)/4)*J(x; n=1, D_2D={D_2D}, D_perp={D_perp})",
        latex_equation=(r"\lambda_{1D}(B) = \frac{A^2}{4} J\left(B; n=1, D_{2D}, D_{\perp}\right)"),
        scopes=("field",),
    ),
    "DiffusionLF_2D": ParameterModelComponentDefinition(
        name="DiffusionLF_2D",
        description="(A^2/4) J(gamma_e B; n=2, D_2D)",
        function=_diffusion_lf_2d,
        param_names=["A", "D_2D", "D_perp"],
        param_defaults={"A": 1.0, "D_2D": 1.0, "D_perp": 0.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\\it A}", "MHz"),
            "D_2D": get_param_info("D_2D"),
            "D_perp": get_param_info("D_perp"),
        },
        formula_template="(({A}^2)/4)*J(x; n=2, D_2D={D_2D}, D_perp={D_perp})",
        latex_equation=(r"\lambda_{2D}(B) = \frac{A^2}{4} J\left(B; n=2, D_{2D}, D_{\perp}\right)"),
        scopes=("field",),
    ),
    "DiffusionLF_3D": ParameterModelComponentDefinition(
        name="DiffusionLF_3D",
        description="(A^2/4) J(gamma_e B; n=3, D_2D)",
        function=_diffusion_lf_3d,
        param_names=["A", "D_2D", "D_perp"],
        param_defaults={"A": 1.0, "D_2D": 1.0, "D_perp": 0.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\\it A}", "MHz"),
            "D_2D": get_param_info("D_2D"),
            "D_perp": get_param_info("D_perp"),
        },
        formula_template="(({A}^2)/4)*J(x; n=3, D_2D={D_2D}, D_perp={D_perp})",
        latex_equation=(r"\lambda_{3D}(B) = \frac{A^2}{4} J\left(B; n=3, D_{2D}, D_{\perp}\right)"),
        scopes=("field",),
    ),
    "BallisticLF_1D": ParameterModelComponentDefinition(
        name="BallisticLF_1D",
        description="(A^2/4) J(gamma_e B; n=1, D_hop)",
        function=_ballistic_lf_1d,
        param_names=["A", "D_hop"],
        param_defaults={"A": 1.0, "D_hop": 1.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\it A}", "MHz"),
            "D_hop": get_param_info("D_hop"),
        },
        formula_template="(({A}^2)/4)*J(x; n=1, D_hop={D_hop})",
        latex_equation=(
            r"\lambda_{1D}^{ball}(B) = \frac{A^2}{4} J\left(B; n=1, D_{\mathrm{hop}}\right)"
        ),
        scopes=("field",),
    ),
    "BallisticLF_2D": ParameterModelComponentDefinition(
        name="BallisticLF_2D",
        description="(A^2/4) J(gamma_e B; n=2, D_hop)",
        function=_ballistic_lf_2d,
        param_names=["A", "D_hop"],
        param_defaults={"A": 1.0, "D_hop": 1.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\it A}", "MHz"),
            "D_hop": get_param_info("D_hop"),
        },
        formula_template="(({A}^2)/4)*J(x; n=2, D_hop={D_hop})",
        latex_equation=(
            r"\lambda_{2D}^{ball}(B) = \frac{A^2}{4} J\left(B; n=2, D_{\mathrm{hop}}\right)"
        ),
        scopes=("field",),
    ),
    "BallisticLF_3D": ParameterModelComponentDefinition(
        name="BallisticLF_3D",
        description="(A^2/4) J(gamma_e B; n=3, D_hop)",
        function=_ballistic_lf_3d,
        param_names=["A", "D_hop"],
        param_defaults={"A": 1.0, "D_hop": 1.0},
        param_info={
            "A": ParamInfo("A", "A", "A", r"$A$", r"{\it A}", "MHz"),
            "D_hop": get_param_info("D_hop"),
        },
        formula_template="(({A}^2)/4)*J(x; n=3, D_hop={D_hop})",
        latex_equation=(
            r"\lambda_{3D}^{ball}(B) = \frac{A^2}{4} J\left(B; n=3, D_{\mathrm{hop}}\right)"
        ),
        scopes=("field",),
    ),
    "Lambda_bg": ParameterModelComponentDefinition(
        name="Lambda_bg",
        description="lambda_BG",
        function=_lambda_bg,
        param_names=["lambda_BG"],
        param_defaults={"lambda_BG": 0.0},
        param_info={"lambda_BG": get_param_info("lambda_BG")},
        formula_template="{lambda_BG}",
        latex_equation=r"\lambda_{bg}(B) = \lambda_{BG}",
        scopes=("field",),
    ),
    "RFResonanceMuP": ParameterModelComponentDefinition(
        name="RFResonanceMuP",
        description=(
            "BG + two Lorentzians at the exact-diagonalisation RF resonance fields "
            "B1(A_mu,A_p,nu_RF), B2(...) of the mu+e+p radical"
        ),
        function=rf_resonance_mup,
        param_names=["A_mu", "A_p", "nu_RF", "ampl1", "wid1", "ampl2", "wid2", "BG"],
        param_defaults={
            "A_mu": 515.0,
            "A_p": 124.0,
            "nu_RF": 218.5,
            "ampl1": -18.0,
            "wid1": 25.0,
            "ampl2": -18.0,
            "wid2": 25.0,
            "BG": 0.0,
        },
        param_info={
            "A_mu": get_param_info("A_mu"),
            "A_p": get_param_info("A_p"),
            "nu_RF": get_param_info("nu_RF"),
            "ampl1": get_param_info("ampl1"),
            "wid1": get_param_info("wid1"),
            "ampl2": get_param_info("ampl2"),
            "wid2": get_param_info("wid2"),
            "BG": get_param_info("BG"),
        },
        formula_template=(
            "{BG} + {ampl1}*{wid1}^2/({wid1}^2 + (x-B1)^2)"
            " + {ampl2}*{wid2}^2/({wid2}^2 + (x-B2)^2),"
            " B1,B2 = exact-diag resonance fields(A_mu={A_mu}, A_p={A_p}, nu_RF={nu_RF})"
        ),
        latex_equation=(
            r"y(B) = \mathrm{BG} + \sum_{i=1,2} \mathrm{ampl}_i\,"
            r"\frac{\mathrm{wid}_i^2}{\mathrm{wid}_i^2 + (B - B_i)^2},\quad "
            r"E(B_i; A_\mu, A_p) = \nu_{\mathrm{RF}}"
        ),
        scopes=("field",),
    ),
}


def _register_superconducting_components() -> None:
    """Register superconducting sigma(T) components for temperature trending."""

    PARAMETER_MODEL_COMPONENTS.update(
        {
            "SC_SWave": ParameterModelComponentDefinition(
                name="SC_SWave",
                description="Isotropic s-wave (fully gapped): sigma_0*rho_s(T; g=1)+sigma_bg",
                function=sc_models.sc_s_wave,
                param_names=["sigma_0", "Tc", "gap_ratio", "sigma_bg"],
                param_defaults={"sigma_0": 1.0, "Tc": 20.0, "gap_ratio": 1.764, "sigma_bg": 0.0},
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template="{sigma_0}*rho_s(x; Tc={Tc}, Delta0/kBTc={gap_ratio}) + {sigma_bg}",
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_s\left(T; T_c, \Delta_0/(k_B T_c)\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_DWave": ParameterModelComponentDefinition(
                name="SC_DWave",
                description="d_{x^2-y^2} line-node gap: sigma_0*rho_d(T; g=cos(2phi))+sigma_bg",
                function=sc_models.sc_d_wave,
                param_names=["sigma_0", "Tc", "gap_ratio", "sigma_bg"],
                param_defaults={"sigma_0": 1.0, "Tc": 20.0, "gap_ratio": 2.14, "sigma_bg": 0.0},
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template="{sigma_0}*rho_d(x; Tc={Tc}, Delta0/kBTc={gap_ratio}) + {sigma_bg}",
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_d\left(T; T_c, \Delta_0/(k_B T_c)\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_AnisotropicS_Cos4": ParameterModelComponentDefinition(
                name="SC_AnisotropicS_Cos4",
                description="Anisotropic s-wave: sigma_0*rho(T; g=1+a*cos(4phi))+sigma_bg",
                function=sc_models.sc_anisotropic_s_cos4,
                param_names=["sigma_0", "Tc", "gap_ratio", "a_anis", "shape_factor_a", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio": 1.764,
                    "a_anis": 0.2,
                    "shape_factor_a": 0.0,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "a_anis": get_param_info("a_anis"),
                    "shape_factor_a": get_param_info("shape_factor_a"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "{sigma_0}*rho_ani(x; Tc={Tc}, Delta0/kBTc={gap_ratio}, a={a_anis}, a_shape={shape_factor_a}) + {sigma_bg}"
                ),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_{ani}\left(T; T_c, \Delta_0/(k_B T_c), a, a_{\mathrm{shape}}\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_NonmonotonicD": ParameterModelComponentDefinition(
                name="SC_NonmonotonicD",
                description="Nonmonotonic d-wave: sigma_0*rho(T; beta*cos(2phi)+(1-beta)*cos(6phi))+sigma_bg",
                function=sc_models.sc_nonmonotonic_d,
                param_names=["sigma_0", "Tc", "gap_ratio", "beta_nm", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio": 2.14,
                    "beta_nm": 0.8,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "beta_nm": get_param_info("beta_nm"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "{sigma_0}*rho_nm(x; Tc={Tc}, Delta0/kBTc={gap_ratio}, beta={beta_nm}) + {sigma_bg}"
                ),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_{nm}\left(T; T_c, \Delta_0/(k_B T_c), \beta\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_PWaveAxial": ParameterModelComponentDefinition(
                name="SC_PWaveAxial",
                description="2D axial p-wave example: sigma_0*rho_p(T; g=cos(phi))+sigma_bg",
                function=sc_models.sc_p_wave_axial,
                param_names=["sigma_0", "Tc", "gap_ratio", "shape_factor_a", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio": 2.0,
                    "shape_factor_a": 0.0,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "shape_factor_a": get_param_info("shape_factor_a"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "{sigma_0}*rho_p(x; Tc={Tc}, Delta0/kBTc={gap_ratio}, a_shape={shape_factor_a}) + {sigma_bg}"
                ),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_p\left(T; T_c, \Delta_0/(k_B T_c), a_{\mathrm{shape}}\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_ExtendedS": ParameterModelComponentDefinition(
                name="SC_ExtendedS",
                description="Extended s-wave from cos(2phi): sigma_0*rho_ext(T)+sigma_bg",
                function=sc_models.sc_extended_s,
                param_names=["sigma_0", "Tc", "gap_ratio", "signed_gap", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio": 2.14,
                    "signed_gap": 0.0,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "signed_gap": get_param_info("signed_gap"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "{sigma_0}*rho_ext(x; Tc={Tc}, Delta0/kBTc={gap_ratio}, signed={signed_gap}) + {sigma_bg}"
                ),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_{ext}\left(T; T_c, \Delta_0/(k_B T_c), s\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_SPlusG": ParameterModelComponentDefinition(
                name="SC_SPlusG",
                description="s+g anisotropic singlet: sigma_0*rho(T; g=(1-sin^4(theta)cos(4phi))/2)+sigma_bg",
                function=sc_models.sc_s_plus_g,
                param_names=["sigma_0", "Tc", "gap_ratio", "sigma_bg"],
                param_defaults={"sigma_0": 1.0, "Tc": 20.0, "gap_ratio": 2.77, "sigma_bg": 0.0},
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template="{sigma_0}*rho_sg(x; Tc={Tc}, Delta0/kBTc={gap_ratio}) + {sigma_bg}",
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_{s+g}\left(T; T_c, \Delta_0/(k_B T_c)\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_AlphaModel": ParameterModelComponentDefinition(
                name="SC_AlphaModel",
                description="Alpha model (scaled s-wave ratio): sigma_0*rho_alpha(T)+sigma_bg",
                function=sc_models.sc_alpha_model,
                param_names=["sigma_0", "Tc", "alpha_sc", "sigma_bg"],
                param_defaults={"sigma_0": 1.0, "Tc": 20.0, "alpha_sc": 1.0, "sigma_bg": 0.0},
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "alpha_sc": get_param_info("alpha_sc"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template="{sigma_0}*rho_alpha(x; Tc={Tc}, alpha={alpha_sc}) + {sigma_bg}",
                latex_equation=(
                    r"\sigma(T) = \sigma_0\,\rho_{\alpha}\left(T; T_c, \alpha\right) + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_TwoGap_SS": ParameterModelComponentDefinition(
                name="SC_TwoGap_SS",
                description="Two-gap s+s (MgB2-style): sigma_0*[w*rho_1+(1-w)*rho_2]+sigma_bg",
                function=sc_models.sc_two_gap_ss,
                param_names=["sigma_0", "Tc", "gap_ratio_1", "gap_ratio_2", "weight", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio_1": 1.2,
                    "gap_ratio_2": 2.2,
                    "weight": 0.5,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio_1": get_param_info("gap_ratio_1"),
                    "gap_ratio_2": get_param_info("gap_ratio_2"),
                    "weight": get_param_info("weight"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=("{sigma_0}*({weight}*rho_1 + (1-{weight})*rho_2) + {sigma_bg}"),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\left[w\rho_1(T) + (1-w)\rho_2(T)\right] + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_TwoGap_SD": ParameterModelComponentDefinition(
                name="SC_TwoGap_SD",
                description="Two-gap mixed s+d: sigma_0*[w*rho_s+(1-w)*rho_d]+sigma_bg",
                function=sc_models.sc_two_gap_sd,
                param_names=["sigma_0", "Tc", "gap_ratio_s", "gap_ratio_d", "weight", "sigma_bg"],
                param_defaults={
                    "sigma_0": 1.0,
                    "Tc": 20.0,
                    "gap_ratio_s": 1.764,
                    "gap_ratio_d": 2.14,
                    "weight": 0.5,
                    "sigma_bg": 0.0,
                },
                param_info={
                    "sigma_0": get_param_info("sigma_0"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio_s": get_param_info("gap_ratio_s"),
                    "gap_ratio_d": get_param_info("gap_ratio_d"),
                    "weight": get_param_info("weight"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=("{sigma_0}*({weight}*rho_s + (1-{weight})*rho_d) + {sigma_bg}"),
                latex_equation=(
                    r"\sigma(T) = \sigma_0\left[w\rho_s(T) + (1-w)\rho_d(T)\right] + \sigma_{bg}"
                ),
                scopes=("temperature",),
            ),
            "SC_SWave_Q": ParameterModelComponentDefinition(
                name="SC_SWave_Q",
                description="Quadrature isotropic s-wave: sqrt((sigma_sc*rho_s)^2 + sigma_nm^2)",
                function=sc_models.sc_s_wave_q,
                param_names=["sigma_sc", "sigma_nm", "Tc", "gap_ratio"],
                param_defaults={"sigma_sc": 1.0, "sigma_nm": 0.1, "Tc": 20.0, "gap_ratio": 1.764},
                param_info={
                    "sigma_sc": get_param_info("sigma_sc"),
                    "sigma_nm": get_param_info("sigma_nm"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                },
                formula_template="sqrt(({sigma_sc}*rho_s)^2 + {sigma_nm}^2)",
                latex_equation=(
                    r"\sigma(T) = \sqrt{\left(\sigma_{sc}\rho_s(T)\right)^2 + \sigma_{nm}^2}"
                ),
                scopes=("temperature",),
            ),
            "SC_DWave_Q": ParameterModelComponentDefinition(
                name="SC_DWave_Q",
                description="Quadrature d-wave: sqrt((sigma_sc*rho_d)^2 + sigma_nm^2)",
                function=sc_models.sc_d_wave_q,
                param_names=["sigma_sc", "sigma_nm", "Tc", "gap_ratio"],
                param_defaults={"sigma_sc": 1.0, "sigma_nm": 0.1, "Tc": 20.0, "gap_ratio": 2.14},
                param_info={
                    "sigma_sc": get_param_info("sigma_sc"),
                    "sigma_nm": get_param_info("sigma_nm"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                },
                formula_template="sqrt(({sigma_sc}*rho_d)^2 + {sigma_nm}^2)",
                latex_equation=(
                    r"\sigma(T) = \sqrt{\left(\sigma_{sc}\rho_d(T)\right)^2 + \sigma_{nm}^2}"
                ),
                scopes=("temperature",),
            ),
            "SC_SPlusG_Q": ParameterModelComponentDefinition(
                name="SC_SPlusG_Q",
                description=(
                    "Quadrature s+g: sqrt((sigma_sc*rho_s+g)^2 + sigma_nm^2); use when linewidth channels add in quadrature"
                ),
                function=sc_models.sc_s_plus_g_q,
                param_names=["sigma_sc", "sigma_nm", "Tc", "gap_ratio"],
                param_defaults={"sigma_sc": 1.0, "sigma_nm": 0.1, "Tc": 20.0, "gap_ratio": 2.77},
                param_info={
                    "sigma_sc": get_param_info("sigma_sc"),
                    "sigma_nm": get_param_info("sigma_nm"),
                    "Tc": get_param_info("Tc"),
                    "gap_ratio": get_param_info("gap_ratio"),
                },
                formula_template="sqrt(({sigma_sc}*rho_sg)^2 + {sigma_nm}^2)",
                latex_equation=(
                    r"\sigma(T) = \sqrt{\left(\sigma_{sc}\rho_{s+g}(T)\right)^2 + \sigma_{nm}^2}"
                ),
                scopes=("temperature",),
            ),
            "SC_Brandt_VortexLattice": ParameterModelComponentDefinition(
                name="SC_Brandt_VortexLattice",
                description=(
                    "Brandt field-dependent vortex-lattice line width (single crystal): "
                    "sigma(B0; lambda_ab, Bc2) for a type-II superconductor"
                ),
                function=sc_models.brandt_field_width_sigma,
                param_names=["lambda_ab", "Bc2", "sigma_bg"],
                param_defaults={"lambda_ab": 200.0, "Bc2": 10.0, "sigma_bg": 0.0},
                param_info={
                    "lambda_ab": get_param_info("lambda_ab"),
                    "Bc2": get_param_info("Bc2"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "sqrt((sigma0({lambda_ab})*(1-b)*(1+1.21*(1-sqrt(b))^3)/2.21)^2 + {sigma_bg}^2), "
                    "b=B0/{Bc2}"
                ),
                latex_equation=(
                    r"\sigma(B_0) = \sqrt{\left[\sigma_0(\lambda_{ab})\,"
                    r"\frac{(1-b)[1+1.21(1-\sqrt{b})^3]}{1+1.21}\right]^2 + \sigma_{bg}^2},\ "
                    r"b = B_0/B_{c2}"
                ),
                scopes=("field",),
            ),
            "SC_Brandt_VortexLattice_Powder": ParameterModelComponentDefinition(
                name="SC_Brandt_VortexLattice_Powder",
                description=(
                    "Brandt field-dependent vortex-lattice line width (polycrystalline): "
                    "sigma(B0; lambda_ab, Bc2) with the 3^(1/4) ab-plane powder average"
                ),
                function=sc_models.brandt_field_width_sigma_powder,
                param_names=["lambda_ab", "Bc2", "sigma_bg"],
                param_defaults={"lambda_ab": 200.0, "Bc2": 10.0, "sigma_bg": 0.0},
                param_info={
                    "lambda_ab": get_param_info("lambda_ab"),
                    "Bc2": get_param_info("Bc2"),
                    "sigma_bg": get_param_info("sigma_bg"),
                },
                formula_template=(
                    "sqrt((sigma0(3^0.25*{lambda_ab})*(1-b)*(1+1.21*(1-sqrt(b))^3)/2.21)^2 "
                    "+ {sigma_bg}^2), b=B0/{Bc2}"
                ),
                latex_equation=(
                    r"\sigma(B_0) = \sqrt{\left[\sigma_0(3^{1/4}\lambda_{ab})\,"
                    r"\frac{(1-b)[1+1.21(1-\sqrt{b})^3]}{1+1.21}\right]^2 + \sigma_{bg}^2},\ "
                    r"b = B_0/B_{c2}"
                ),
                scopes=("field",),
            ),
        }
    )


_register_superconducting_components()

_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
#: The parameter-vs-x grammar additionally supports the quadrature combinator
#: ``f ⊕ g = √(f² + g²)`` (binary, same precedence as ``+``/``-``, associative),
#: so e.g. ``PowerLaw ⊕ Constant`` reproduces ``PowerLawQuadBG``. Time-domain
#: composites keep the base operator set.
_PARAMETER_ALLOWED_OPERATORS: frozenset[str] = _ALLOWED_OPERATORS | {QUADRATURE_OPERATOR}


def component_names_for_x(x_key: str) -> list[str]:
    """Return available basis functions for a given x-variable."""
    if x_key == "field":
        scopes = {"common", "field"}
    elif x_key == "temperature":
        scopes = {"common", "temperature"}
    else:
        scopes = {"common"}

    names = [
        name
        for name, comp in PARAMETER_MODEL_COMPONENTS.items()
        if scopes.intersection(comp.scopes)
    ]
    return sorted(names)


class ParameterCompositeModel:
    """Flat composite model for parameter-vs-x fitting."""

    def __init__(
        self,
        component_names: list[str],
        operators: list[str] | None = None,
        open_parentheses: list[int] | None = None,
        close_parentheses: list[int] | None = None,
    ) -> None:
        if not component_names:
            raise ValueError("Model must contain at least one component")

        missing = [name for name in component_names if name not in PARAMETER_MODEL_COMPONENTS]
        if missing:
            raise ValueError(f"Unknown component(s): {missing}")

        if operators is None:
            operators = ["+"] * (len(component_names) - 1)

        if len(operators) != max(len(component_names) - 1, 0):
            raise ValueError("operators length must be len(component_names) - 1")
        if any(op not in _PARAMETER_ALLOWED_OPERATORS for op in operators):
            raise ValueError("operators must be one of '+', '-', '*', '/', '⊕'")

        if open_parentheses is None:
            open_parentheses = [0] * len(component_names)
        if close_parentheses is None:
            close_parentheses = [0] * len(component_names)
        if len(open_parentheses) != len(component_names):
            raise ValueError("open_parentheses length must be len(component_names)")
        if len(close_parentheses) != len(component_names):
            raise ValueError("close_parentheses length must be len(component_names)")
        if any((not isinstance(v, int)) or v < 0 for v in open_parentheses):
            raise ValueError("open_parentheses values must be non-negative integers")
        if any((not isinstance(v, int)) or v < 0 for v in close_parentheses):
            raise ValueError("close_parentheses values must be non-negative integers")

        balance = 0
        for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
            balance += open_count
            balance -= close_count
            if balance < 0:
                raise ValueError("Invalid parentheses: closing before opening")
        if balance != 0:
            raise ValueError("Invalid parentheses: unbalanced expression")

        self.component_names = list(component_names)
        self.operators = list(operators)
        self.open_parentheses = list(open_parentheses)
        self.close_parentheses = list(close_parentheses)
        self.components = [PARAMETER_MODEL_COMPONENTS[name] for name in self.component_names]
        self._param_mappings = self._build_param_mapping()

        param_names: list[str] = []
        param_defaults: dict[str, float] = {}
        param_info: dict[str, ParamInfo] = {}
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            for pname in component.param_names:
                unique_name = mapping[pname]
                param_names.append(unique_name)
                param_defaults[unique_name] = component.param_defaults[pname]
                base_info = component.param_info[pname]
                if unique_name == pname:
                    param_info[unique_name] = base_info
                else:
                    _base, index = split_parameter_name(unique_name)
                    param_info[unique_name] = base_info.with_index(index) if index else base_info

        self.param_names = param_names
        self.param_defaults = param_defaults
        self.param_info = param_info

    @classmethod
    def from_expression(cls, expression: str) -> ParameterCompositeModel:
        """Construct a ParameterCompositeModel from a component-name expression."""
        component_names, operators, open_parentheses, close_parentheses = (
            parse_component_expression(
                expression,
                allowed_components=set(PARAMETER_MODEL_COMPONENTS),
                allowed_operators=_PARAMETER_ALLOWED_OPERATORS,
            )
        )
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
        )

    def component_expression_string(self) -> str:
        """Return the builder-facing expression using component names."""
        return build_component_expression(
            self.component_names,
            self.operators,
            self.open_parentheses,
            self.close_parentheses,
        )

    def component_param_name(self, component_index: int, local_name: str) -> str:
        """Global (possibly index-suffixed) param name for a component's param.

        For ``["GaussianLCR", "LorentzianLCR"]`` the second component's ``"B0"``
        is ``"B0_2"``; for a single component it is the bare ``"B0"``. Lets callers
        address a component's parameters by name instead of positional arithmetic
        over :attr:`param_names`.
        """
        return self._param_mappings[component_index][local_name]

    def _build_param_mapping(self) -> list[dict[str, str]]:
        counts = Counter(pname for component in self.components for pname in component.param_names)
        mappings: list[dict[str, str]] = []
        for idx, component in enumerate(self.components, start=1):
            mapping: dict[str, str] = {}
            for pname in component.param_names:
                if counts[pname] > 1:
                    mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
            mappings.append(mapping)
        return mappings

    def formula_string(self) -> str:
        """Return a readable formula string for display."""
        terms: list[str] = []
        for idx, (mapping, component) in enumerate(
            zip(self._param_mappings, self.components, strict=True)
        ):
            mapping_text = {k: mapping[k] for k in component.param_names}
            term = component.formula_template.format(**mapping_text)
            if self.open_parentheses[idx] > 0:
                term = "(" * self.open_parentheses[idx] + term
            if self.close_parentheses[idx] > 0:
                term = term + ")" * self.close_parentheses[idx]
            terms.append(term)

        if not terms:
            return "0"

        expression = terms[0]
        for op, term in zip(self.operators, terms[1:], strict=True):
            expression = f"{expression} {op} {term}"
        return expression

    def function(self, x: NDArray, **kwargs: float) -> NDArray[np.float64]:
        """Evaluate the composite model with arithmetic precedence."""
        xx = np.asarray(x, dtype=float)
        values: list[NDArray[np.float64]] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            local_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            values.append(np.asarray(component.function(xx, **local_kwargs), dtype=float))

        if not values:
            return np.zeros_like(xx)

        if any(self.open_parentheses) or any(self.close_parentheses):
            return self._evaluate_parenthesized(xx, values)

        reduced_values: list[NDArray[np.float64]] = [values[0]]
        reduced_ops: list[str] = []
        i = 0
        while i < len(self.operators):
            op = self.operators[i]
            if op in {"*", "/"}:
                left = reduced_values.pop()
                right = values[i + 1]
                if op == "*":
                    reduced_values.append(left * right)
                else:
                    safe_right = np.where(np.abs(right) < 1e-12, np.nan, right)
                    reduced_values.append(left / safe_right)
            else:
                reduced_ops.append(op)
                reduced_values.append(values[i + 1])
            i += 1

        result = reduced_values[0]
        for op, value in zip(reduced_ops, reduced_values[1:], strict=True):
            if op == "+":
                result = result + value
            elif op == "-":
                result = result - value
            else:  # QUADRATURE_OPERATOR: f ⊕ g = √(f² + g²)
                result = np.sqrt(result**2 + value**2)

        return result

    def _evaluate_parenthesized(
        self,
        xx: NDArray[np.float64],
        values: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        value_stack: list[NDArray[np.float64]] = []
        op_stack: list[str] = []

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_top_operator() -> None:
            if len(value_stack) < 2 or not op_stack:
                raise ValueError("Invalid expression")
            op = op_stack.pop()
            rhs = value_stack.pop()
            lhs = value_stack.pop()
            if op == "+":
                value_stack.append(lhs + rhs)
            elif op == "-":
                value_stack.append(lhs - rhs)
            elif op == "*":
                value_stack.append(lhs * rhs)
            elif op == "/":
                safe_rhs = np.where(np.abs(rhs) < 1e-12, np.nan, rhs)
                value_stack.append(lhs / safe_rhs)
            else:  # QUADRATURE_OPERATOR: f ⊕ g = √(f² + g²)
                value_stack.append(np.sqrt(lhs**2 + rhs**2))

        for idx, value in enumerate(values):
            for _ in range(self.open_parentheses[idx]):
                op_stack.append("(")

            value_stack.append(value)

            for _ in range(self.close_parentheses[idx]):
                while op_stack and op_stack[-1] != "(":
                    apply_top_operator()
                if not op_stack or op_stack[-1] != "(":
                    raise ValueError("Invalid parentheses in expression")
                op_stack.pop()

            if idx < len(self.operators):
                op = self.operators[idx]
                while (
                    op_stack and op_stack[-1] != "(" and precedence(op_stack[-1]) >= precedence(op)
                ):
                    apply_top_operator()
                op_stack.append(op)

        while op_stack:
            if op_stack[-1] == "(":
                raise ValueError("Invalid parentheses in expression")
            apply_top_operator()

        if len(value_stack) != 1:
            raise ValueError("Invalid expression")
        if value_stack[0].shape != xx.shape:
            raise ValueError("Invalid expression result shape")
        return value_stack[0]

    def _extract_component_kwargs(
        self,
        component: ParameterModelComponentDefinition,
        mapping: dict[str, str],
        kwargs: dict[str, float],
    ) -> dict[str, float]:
        local_kwargs: dict[str, float] = {}
        for pname in component.param_names:
            unique_name = mapping[pname]
            if unique_name not in kwargs:
                raise KeyError(f"Missing parameter '{unique_name}'")
            local_kwargs[pname] = float(kwargs[unique_name])
        return local_kwargs

    def additive_component_indices(self) -> list[int]:
        """Return indices for additive components (first term, '+' and '⊕' terms).

        Quadrature (``⊕``) combines whole components like ``+`` does, so each
        operand is a distinct curve worth plotting on its own; subtractive and
        multiplicative terms are not standalone contributions.
        """
        if not self.components:
            return []
        indices = [0]
        for idx, op in enumerate(self.operators, start=1):
            if op in ("+", QUADRATURE_OPERATOR):
                indices.append(idx)
        return indices

    def evaluate_components(
        self,
        x: NDArray,
        *,
        additive_only: bool = False,
        **kwargs: float,
    ) -> list[tuple[str, NDArray[np.float64]]]:
        """Evaluate individual component curves for plotting/export."""
        xx = np.asarray(x, dtype=float)
        include = (
            set(self.additive_component_indices())
            if additive_only
            else set(range(len(self.components)))
        )

        out: list[tuple[str, NDArray[np.float64]]] = []
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            if idx not in include:
                continue
            local_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            values = np.asarray(component.function(xx, **local_kwargs), dtype=float)
            out.append((self.component_names[idx], values))
        return out

    def to_dict(self) -> dict[str, list[str] | list[int]]:
        """Return a JSON-serializable representation of the model."""
        return {
            "component_names": list(self.component_names),
            "operators": list(self.operators),
            "open_parentheses": list(self.open_parentheses),
            "close_parentheses": list(self.close_parentheses),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParameterCompositeModel:
        """Construct a ParameterCompositeModel from serialized data."""
        component_names = data.get("component_names")
        operators = data.get("operators")
        open_parentheses = data.get("open_parentheses")
        close_parentheses = data.get("close_parentheses")
        if not isinstance(component_names, list) or not all(
            isinstance(value, str) for value in component_names
        ):
            raise ValueError("Invalid parameter composite model data: component_names")
        if operators is not None:
            if not isinstance(operators, list) or not all(
                isinstance(value, str) for value in operators
            ):
                raise ValueError("Invalid parameter composite model data: operators")
        if open_parentheses is not None:
            if not isinstance(open_parentheses, list) or not all(
                isinstance(value, int) for value in open_parentheses
            ):
                raise ValueError("Invalid parameter composite model data: open_parentheses")
        if close_parentheses is not None:
            if not isinstance(close_parentheses, list) or not all(
                isinstance(value, int) for value in close_parentheses
            ):
                raise ValueError("Invalid parameter composite model data: close_parentheses")
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
        )


#: Trend components whose default ``Tc = 10`` seed is unphysical for real
#: field/temperature scans (which sit at tens-to-hundreds of K or G). The
#: critical temperature must instead be inferred from the fitted x-range,
#: otherwise the trend fit fails to converge until the user reseeds Tc by hand.
_TREND_TC_COMPONENTS = frozenset({"CriticalDivergence", "OrderParameter"})


def suggest_trend_seeds(
    model: ParameterCompositeModel,
    x: NDArray,
    y: NDArray,
) -> dict[str, float]:
    """Data-aware seed overrides for critical-temperature trend components.

    The :data:`CriticalDivergence` and :data:`OrderParameter` components default
    to ``Tc = 10``, which is far from where real datasets transition (spin-glass
    ``T_g ≈ 88 K``, EuO ``T_c ≈ 69 K``), so the trend fit will not converge until
    ``Tc`` is reseeded by hand. This helper derives a starting ``Tc`` from the
    boundary of the fitted x-range — and a cheap amplitude/baseline seed — so the
    fit converges out of the box.

    Returns a mapping of *unique* parameter name (matching ``model.param_names``,
    accounting for repeated-component suffixing) to suggested value. Only the
    parameters it is confident about are returned; the caller merges these over
    ``model.param_defaults`` and leaves everything else untouched. Returns an
    empty mapping when the model has no trend component or the data is unusable
    (all-NaN x). Pure and Qt-free.

    The seed direction follows each model's physics:

    * ``CriticalDivergence`` (``a·|x − Tc|^{-ν} + c``) diverges *at* ``Tc`` and is
      fitted with data *above* it, so ``Tc`` is placed just below ``min(x)`` (a
      small margin keeps the nearest point off the singularity) and the baseline
      ``c`` is seeded from the flat, far-from-``Tc`` end of the trace.
    * ``OrderParameter`` (``y0·[1 − (T/Tc)^α]^β``) vanishes *at* ``Tc`` and is
      fitted with data *below* it, so ``Tc`` is placed just above ``max(x)`` and
      the amplitude ``y0`` is seeded from the largest observed value.

    Exponents (``ν``, ``β``, ``α``) keep their physical defaults.
    """
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    finite = np.isfinite(xf)
    if not np.any(finite):
        return {}

    x_valid = xf[finite]
    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    span = x_max - x_min
    # A few percent of the span keeps Tc just off the data; the absolute floor
    # covers a single-point or zero-span trace where a relative margin vanishes.
    margin = max(span * 0.05, abs(x_max) * 0.01, 1e-3)

    y_finite = yf[np.isfinite(yf)]
    y_min = float(np.min(y_finite)) if y_finite.size else None
    y_max = float(np.max(y_finite)) if y_finite.size else None

    seeds: dict[str, float] = {}
    for component, mapping in zip(model.components, model._param_mappings, strict=True):
        if component.name == "CriticalDivergence":
            seeds[mapping["Tc"]] = x_min - margin
            if y_min is not None:
                seeds[mapping["c"]] = y_min
        elif component.name == "OrderParameter":
            seeds[mapping["Tc"]] = x_max + margin
            if y_max is not None:
                seeds[mapping["y0"]] = y_max
    return seeds


class ErrorMode(str, Enum):
    """How per-point σ values are assigned when weighting a model fit.

    ``COLUMN`` uses the propagated errors of the trended parameter (the
    default and the only mode where the stabilisation floor applies);
    ``PERCENT`` sets σᵢ = (value/100)·|yᵢ|; ``ABSOLUTE`` a constant σ =
    value; ``NONE`` unit weights; ``SCATTER`` fits with unit weights and
    rescales the parameter errors by √(χ²/ν) afterwards — the standard
    estimate of errors from the scatter of the points (and the fixed point
    of WiMDA's iterated Estimate mode).
    """

    COLUMN = "column"
    PERCENT = "percent"
    ABSOLUTE = "absolute"
    NONE = "none"
    SCATTER = "scatter"


def apply_error_mode(
    y: NDArray,
    yerr: NDArray | None,
    mode: ErrorMode | str = ErrorMode.COLUMN,
    value: float | None = None,
) -> NDArray[np.float64] | None:
    """Return the σ array a fit should weight with under ``mode``.

    ``value`` is the percentage for ``PERCENT`` and the constant σ for
    ``ABSOLUTE``. A missing, non-finite, or non-positive value falls back to
    σ = 1 for ``ABSOLUTE`` (matching WiMDA) and to 1 % for ``PERCENT`` (so a
    zero percentage cannot zero out every error and mask the whole fit).
    Returns ``None`` for ``COLUMN`` with no error column, which callers treat
    as unit weights. Percent-mode points with y = 0 get σ = 0 and are
    excluded by the standard validity mask — they carry no error information.
    """
    mode = ErrorMode(mode)
    yy = np.asarray(y, dtype=float)
    if mode is ErrorMode.COLUMN:
        return None if yerr is None else np.asarray(yerr, dtype=float)
    if mode is ErrorMode.PERCENT:
        pct = abs(float(value)) if value is not None and np.isfinite(value) else 1.0
        if pct <= 0.0:
            pct = 1.0
        return (pct / 100.0) * np.abs(yy)
    if mode is ErrorMode.ABSOLUTE:
        const = abs(float(value)) if value is not None and np.isfinite(value) else 1.0
        if const <= 0.0:
            const = 1.0
        return np.full_like(yy, const)
    # NONE and SCATTER both weight uniformly; SCATTER additionally rescales
    # the resulting parameter errors after the fit.
    return np.ones_like(yy)


@dataclass
class ParameterModelFitResult:
    """Result of fitting a parameter-vs-x model."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    message: str = ""
    #: Error mode the fit was run with (χ²ᵣ carries no goodness information
    #: for ``"none"``/``"scatter"`` — quality verdicts should be suppressed).
    error_mode: str = ErrorMode.COLUMN.value
    #: Number of points that entered the fit (0 when unknown/legacy).
    n_points: int = 0


@dataclass
class ModelFitRange:
    """A model and fit results over a specific x-range.

    ``windows`` optionally restricts the range to a union of (min, max)
    intervals: a point enters the fit if it falls in *any* window
    (OR-combination), with one model fitted across the union. When absent,
    the plain ``x_min``/``x_max`` bounds apply.
    """

    x_min: float | None
    x_max: float | None
    model: ParameterCompositeModel
    parameters: ParameterSet
    result: ParameterModelFitResult | None = None
    windows: list[tuple[float, float]] | None = None


def validate_fit_windows(
    windows: Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Normalise a window list: drop None, reject inverted windows."""
    if not windows:
        return None
    normalised: list[tuple[float, float]] = []
    for window in windows:
        lo, hi = float(window[0]), float(window[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"Fit window bounds must be finite, got ({lo}, {hi})")
        if lo > hi:
            raise ValueError(f"Fit window is inverted: ({lo}, {hi})")
        normalised.append((lo, hi))
    return normalised


def parse_fit_windows(state: object) -> list[tuple[float, float]] | None:
    """Leniently parse a serialized window list (sequence of [lo, hi] pairs).

    Malformed entries are dropped rather than raising — saved state must
    never prevent a project from opening. Returns ``None`` when nothing
    usable remains.
    """
    if not isinstance(state, (list, tuple)):
        return None
    windows: list[tuple[float, float]] = []
    for entry in state:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            try:
                windows.append((float(entry[0]), float(entry[1])))
            except (TypeError, ValueError):
                continue
    return windows or None


def windows_mask(
    x: NDArray,
    windows: Sequence[tuple[float, float]] | None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of points inside the window union (or x_min/x_max).

    With ``windows`` present the mask is the OR over all (min, max)
    intervals and ``x_min``/``x_max`` are ignored; otherwise the plain
    bounds apply (``None`` bounds are open).
    """
    xx = np.asarray(x, dtype=float)
    valid = validate_fit_windows(windows)
    if valid is not None:
        mask = np.zeros(xx.shape, dtype=bool)
        for lo, hi in valid:
            mask |= (xx >= lo) & (xx <= hi)
        return mask
    mask = np.ones(xx.shape, dtype=bool)
    if x_min is not None:
        mask &= xx >= float(x_min)
    if x_max is not None:
        mask &= xx <= float(x_max)
    return mask


def range_mask(x: NDArray, fit_range: ModelFitRange) -> NDArray[np.bool_]:
    """Window-union (or min/max) mask for a :class:`ModelFitRange`."""
    return windows_mask(x, fit_range.windows, fit_range.x_min, fit_range.x_max)


def effective_range_bounds(fit_range: ModelFitRange) -> tuple[float | None, float | None]:
    """The x-extent a range's fitted curve should span.

    With windows present this is the union envelope (so the curve is drawn
    continuously through excluded gaps); otherwise the plain bounds. Raises
    ``ValueError`` for invalid windows, like :func:`validate_fit_windows`.
    """
    windows = validate_fit_windows(fit_range.windows)
    if windows is not None:
        return min(lo for lo, _ in windows), max(hi for _, hi in windows)
    return fit_range.x_min, fit_range.x_max


@dataclass
class ParameterModelFit:
    """Model fits attached to a single parameter trace."""

    parameter_name: str
    x_key: str
    ranges: list[ModelFitRange] = field(default_factory=list)
    active: bool = True
    #: When the x-axis is a fitted parameter (``x_key == "param:<name>"``),
    #: account for its per-point uncertainty via effective-variance weighting.
    use_x_errors: bool = False


@dataclass
class ParameterModelFitExecution:
    """Fitted model with sampled curve ready for plotting/export."""

    range_index: int
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    result: ParameterModelFitResult


@dataclass
class ParameterGroupData:
    """Input data for one selected group in a cross-group parameter fit."""

    group_id: str
    group_name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yerr: NDArray[np.float64]
    group_variable_value: float
    #: Optional per-point x-uncertainty (aligned to ``x``), used only for an
    #: effective-variance fit when the abscissa is itself a fitted parameter.
    xerr: NDArray[np.float64] | None = None


@dataclass
class CrossGroupFitResult:
    """Result container for cross-group parameter fits."""

    success: bool
    chi_squared: float
    reduced_chi_squared: float
    global_parameters: ParameterSet = field(default_factory=ParameterSet)
    local_parameters: dict[str, ParameterSet] = field(default_factory=dict)
    fixed_parameters: ParameterSet = field(default_factory=ParameterSet)
    global_uncertainties: dict[str, float] = field(default_factory=dict)
    local_uncertainties: dict[str, dict[str, float]] = field(default_factory=dict)
    message: str = ""
    #: Error mode the fit was run with (χ²ᵣ carries no goodness information for
    #: ``"none"``/``"scatter"`` — quality verdicts should be suppressed).
    error_mode: str = ErrorMode.COLUMN.value
    #: Number of points that entered the fit across all groups (0 if unknown).
    n_points: int = 0


_TRANSPORT_RATE_SEED_GRID = (
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1.0e3,
    3.0e3,
    1.0e4,
    3.0e4,
    1.0e5,
)
_DYNAMIC_TRANSPORT_RATE_SEED_COUNT = 6
_PARAMETER_MODEL_ERROR_FLOOR_FRACTION = 0.5


def _parameter_values_by_name(parameters: ParameterSet) -> dict[str, float]:
    return {parameter.name: float(parameter.value) for parameter in parameters}


def _stabilize_parameter_model_errors(errors: NDArray[np.float64]) -> NDArray[np.float64]:
    positive_errors = np.asarray(errors, dtype=float)
    positive_errors = positive_errors[np.isfinite(positive_errors) & (positive_errors > 0.0)]
    if positive_errors.size == 0:
        return np.asarray(errors, dtype=float)

    median_positive_error = float(np.median(positive_errors))
    floor = _PARAMETER_MODEL_ERROR_FLOOR_FRACTION * median_positive_error
    if floor <= 0.0:
        return np.asarray(errors, dtype=float)
    return np.maximum(np.asarray(errors, dtype=float), floor)


def _is_additive_parameter_model(model: ParameterCompositeModel) -> bool:
    return (
        (not any(model.open_parentheses))
        and (not any(model.close_parentheses))
        and all(op == "+" for op in model.operators)
    )


def _transport_rate_parameter_for_component(
    component: ParameterModelComponentDefinition,
) -> str | None:
    if component.name.startswith("DiffusionLF_"):
        return "D_2D"
    if component.name.startswith("BallisticLF_"):
        return "D_hop"
    return None


def _clip_to_parameter_bounds(value: float, parameter: Parameter) -> float:
    return float(min(max(value, float(parameter.min)), float(parameter.max)))


def _dynamic_transport_rate_seeds(x_fit: NDArray[np.float64]) -> tuple[float, ...]:
    finite_x = np.asarray(x_fit, dtype=float)
    positive_fields = np.abs(finite_x[np.isfinite(finite_x) & (np.abs(finite_x) > 0.0)])
    if positive_fields.size == 0:
        return ()

    omega = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G * positive_fields
    omega_min = float(np.min(omega))
    omega_max = float(np.max(omega))
    if omega_max <= 0.0:
        return ()

    seed_min = max(0.3, omega_min / 100.0)
    seed_max = max(seed_min, omega_max * 10.0)
    if np.isclose(seed_min, seed_max):
        return (seed_min,)

    return tuple(
        float(value)
        for value in np.geomspace(seed_min, seed_max, num=_DYNAMIC_TRANSPORT_RATE_SEED_COUNT)
    )


def _candidate_rate_values(parameter: Parameter, x_fit: NDArray[np.float64]) -> list[float]:
    values = [_clip_to_parameter_bounds(float(parameter.value), parameter)]
    for seed in _TRANSPORT_RATE_SEED_GRID:
        if float(parameter.min) <= seed <= float(parameter.max):
            values.append(float(seed))
    for seed in _dynamic_transport_rate_seeds(x_fit):
        if float(parameter.min) <= seed <= float(parameter.max):
            values.append(float(seed))

    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(float(value))
    return deduped


def _solve_nonnegative_weighted_linear(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    if design.size == 0:
        return np.zeros(0, dtype=float)

    safe_sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    weighted_design = np.asarray(design, dtype=float) / safe_sigma[:, None]
    weighted_target = np.asarray(target, dtype=float) / safe_sigma

    active = np.ones(weighted_design.shape[1], dtype=bool)
    coefficients = np.zeros(weighted_design.shape[1], dtype=float)

    while np.any(active):
        solution, *_ = np.linalg.lstsq(weighted_design[:, active], weighted_target, rcond=None)
        if np.all(solution >= 0.0):
            coefficients[active] = solution
            return coefficients
        active_indices = np.flatnonzero(active)
        active[active_indices[int(np.argmin(solution))]] = False

    return coefficients


def _transport_seed_initial_values(
    x_fit: NDArray[np.float64],
    y_fit: NDArray[np.float64],
    e_fit: NDArray[np.float64],
    model: ParameterCompositeModel,
    parameters: ParameterSet,
) -> dict[str, float] | None:
    if not _is_additive_parameter_model(model):
        return None

    current_values = _parameter_values_by_name(parameters)
    fixed_contribution = np.zeros_like(y_fit, dtype=float)
    transport_entries: list[tuple[ParameterModelComponentDefinition, dict[str, str], str, str]] = []

    for component, mapping in zip(model.components, model._param_mappings, strict=True):
        rate_name = _transport_rate_parameter_for_component(component)
        amp_name = mapping.get("A")
        rate_param_name = mapping.get(rate_name) if rate_name is not None else None

        if amp_name is None or rate_param_name is None:
            local_kwargs = model._extract_component_kwargs(component, mapping, current_values)
            fixed_contribution += np.asarray(component.function(x_fit, **local_kwargs), dtype=float)
            continue

        amp_parameter = parameters[amp_name]
        if amp_parameter.fixed:
            local_kwargs = model._extract_component_kwargs(component, mapping, current_values)
            fixed_contribution += np.asarray(component.function(x_fit, **local_kwargs), dtype=float)
            continue

        transport_entries.append((component, mapping, amp_name, rate_param_name))

    if not transport_entries:
        return None

    target = np.asarray(y_fit - fixed_contribution, dtype=float)
    best_seed: dict[str, float] | None = None
    best_chi2 = float("inf")

    rate_grids = [
        _candidate_rate_values(parameters[rate_param_name], x_fit)
        for _component, _mapping, _amp_name, rate_param_name in transport_entries
    ]

    for rate_choice in product(*rate_grids):
        seed_values = dict(current_values)
        basis_columns: list[NDArray[np.float64]] = []

        for rate_value, (component, mapping, _amp_name, rate_param_name) in zip(
            rate_choice, transport_entries, strict=True
        ):
            seed_values[rate_param_name] = float(rate_value)
            local_kwargs = model._extract_component_kwargs(component, mapping, seed_values)
            local_kwargs["A"] = 2.0
            basis_columns.append(np.asarray(component.function(x_fit, **local_kwargs), dtype=float))

        design = np.column_stack(basis_columns)
        prefactors = _solve_nonnegative_weighted_linear(design, target, e_fit)
        model_y = np.zeros_like(target, dtype=float)

        for prefactor, basis, (_component, _mapping, amp_name, _rate_param_name) in zip(
            prefactors, basis_columns, transport_entries, strict=True
        ):
            seed_values[amp_name] = float(2.0 * np.sqrt(max(prefactor, 0.0)))
            model_y += prefactor * basis

        chi2 = float(np.sum(np.square((target - model_y) / np.maximum(e_fit, 1e-12))))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_seed = seed_values

    return best_seed


def _effective_variance_residual(
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma_y2: NDArray[np.float64],
    xerr2: NDArray[np.float64],
    x_step: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Orear/York effective-variance residual when both x and y are uncertain.

    ``σ²_eff = σ_y² + (∂f/∂x)²·σ_x²``, with the local slope evaluated by a
    central finite difference ``(f(x+h) − f(x−h)) / 2h`` (no analytic
    derivative needed). A model undefined just outside the data range yields a
    non-finite probe; the slope falls back to zero there rather than poisoning
    the whole cost (``σ_y² > 0`` is guaranteed by the caller's validity mask, so
    the denominator stays positive). Shared by the single-series and
    cross-group fits so the two paths cannot drift numerically.
    """
    pred = np.asarray(predict(x), dtype=float)
    slope = (
        np.asarray(predict(x + x_step), dtype=float) - np.asarray(predict(x - x_step), dtype=float)
    ) / (2.0 * x_step)
    slope = np.where(np.isfinite(slope), slope, 0.0)
    sigma2 = sigma_y2 + (slope**2) * xerr2
    return (y - pred) / np.sqrt(sigma2)


def _run_parameter_model_minuit(
    x_fit: NDArray[np.float64],
    y_fit: NDArray[np.float64],
    e_fit: NDArray[np.float64],
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    method: str,
    initial_values: dict[str, float] | None = None,
    xerr_fit: NDArray[np.float64] | None = None,
) -> tuple[ParameterModelFitResult, float]:
    try:
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
    except ImportError as exc:
        return (
            ParameterModelFitResult(success=False, message=f"iminuit import error: {exc}"),
            float("inf"),
        )

    free = parameters.free_parameters
    fixed_kw = {p.name: p.value for p in parameters if p.fixed}
    param_names = [p.name for p in free]

    def model_wrapper(x_local: NDArray, *args: float) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(param_names, args, strict=False))}
        return model.function(x_local, **kw)

    if xerr_fit is None:
        cost = LeastSquares(x_fit, y_fit, e_fit, model_wrapper)
    else:
        # Errors-in-variables (Orear/York effective variance): inflate each
        # point's variance by the x-error propagated through the local slope,
        # σ²_eff = σ_y² + (∂f/∂x)²·σ_x². The slope is a central finite
        # difference, so no analytic derivative is needed; iminuit re-evaluates
        # the cost every step, making the weighting self-consistent at the
        # minimum with no outer iteration. With σ_x = 0 this reduces exactly to
        # ordinary least squares (callers route the all-zero case to the
        # LeastSquares branch above to stay byte-identical).
        x_step = np.maximum(np.abs(x_fit), 1.0) * 1e-6
        e2 = e_fit**2
        xerr2 = np.asarray(xerr_fit, dtype=float) ** 2

        def cost(*args: float) -> float:
            resid = _effective_variance_residual(
                lambda x_local: model_wrapper(x_local, *args),
                x_fit,
                y_fit,
                e2,
                xerr2,
                x_step,
            )
            return float(np.sum(resid**2))

        cost.errordef = Minuit.LEAST_SQUARES

    if initial_values is None:
        initial_values = {}
    init = [float(initial_values.get(p.name, p.value)) for p in free]
    m = Minuit(cost, *init, name=param_names)

    for i, p in enumerate(free):
        if p.min != -float("inf"):
            m.limits[i] = (p.min, m.limits[i][1])
        if p.max != float("inf"):
            m.limits[i] = (m.limits[i][0], p.max)

    if method == "simplex":
        m.simplex()
    else:
        m.migrad()

    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}

    for p in parameters:
        if p.fixed:
            result_params.add(
                Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=True)
            )
        else:
            idx = param_names.index(p.name)
            value = float(m.values[idx])
            result_params.add(
                Parameter(name=p.name, value=value, min=p.min, max=p.max, fixed=False)
            )
            err = m.errors[idx]
            if err is not None and np.isfinite(err):
                uncertainties[p.name] = float(err)

    ndof = max(len(x_fit) - len(free), 1)
    return (
        ParameterModelFitResult(
            success=bool(m.valid),
            chi_squared=float(m.fval),
            reduced_chi_squared=float(m.fval) / ndof,
            parameters=result_params,
            uncertainties=uncertainties,
            message="Fit successful" if m.valid else "Fit failed",
        ),
        float(m.fval),
    )


def fit_parameter_model(
    x: NDArray,
    y: NDArray,
    yerr: NDArray | None,
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x_min: float | None = None,
    x_max: float | None = None,
    method: str = "migrad",
    error_mode: ErrorMode | str = ErrorMode.COLUMN,
    error_value: float | None = None,
    windows: Sequence[tuple[float, float]] | None = None,
    xerr: NDArray | None = None,
) -> ParameterModelFitResult:
    """Fit a parameter-vs-x model using iminuit.

    ``error_mode``/``error_value`` select the per-point σ assignment (see
    :class:`ErrorMode`); ``windows`` optionally restricts the fit to a union
    of (min, max) intervals, overriding ``x_min``/``x_max``.

    ``xerr`` optionally supplies per-point x-uncertainties for an
    errors-in-variables (Orear/York effective-variance) fit — used for
    parameter-vs-parameter trending where the abscissa is itself a fitted
    quantity. When ``xerr`` is ``None`` or all zero/non-finite the fit is
    ordinary least squares (the abscissa is treated as exact). It is ignored
    under the ``NONE``/``SCATTER`` error modes, whose unit y-weights have no
    physical scale to combine with the x-variance term.
    """
    error_mode = ErrorMode(error_mode)
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ee = apply_error_mode(yy, yerr, error_mode, error_value)
    if ee is None:
        ee = np.ones_like(xx)
    # Effective variance combines σ_x with a real per-point σ_y; under unit
    # weights (NONE) or scatter estimation (SCATTER) the "σ_y" is a placeholder
    # of arbitrary scale, so the combination σ_y² + slope²·σ_x² would be
    # scale-dependent — ignore x-errors in those modes.
    if error_mode in (ErrorMode.NONE, ErrorMode.SCATTER):
        xe = None
    else:
        xe = None if xerr is None else np.asarray(xerr, dtype=float)

    try:
        window_selection = windows_mask(xx, windows, x_min, x_max)
    except ValueError as exc:
        # Keep the documented failure contract: bad range inputs yield a
        # failed result, never an exception.
        return ParameterModelFitResult(success=False, message=str(exc), error_mode=error_mode.value)

    mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee) & (ee > 0)
    mask &= window_selection

    if not np.any(mask):
        return ParameterModelFitResult(
            success=False,
            message="No valid points in selected range",
            error_mode=error_mode.value,
        )

    x_fit = xx[mask]
    y_fit = yy[mask]
    e_fit = ee[mask]
    if error_mode is ErrorMode.COLUMN:
        # The stabilisation floor exists to tame near-zero propagated errors;
        # explicit Percent/Absolute/unit weights are honoured verbatim.
        e_fit = _stabilize_parameter_model_errors(e_fit)

    # Effective-variance weighting only when x carries usable uncertainty;
    # otherwise stay on the ordinary least-squares path (byte-identical).
    xerr_fit: NDArray[np.float64] | None = None
    if xe is not None:
        xe_fit = xe[mask]
        xe_fit = np.where(np.isfinite(xe_fit) & (xe_fit > 0.0), xe_fit, 0.0)
        if np.any(xe_fit > 0.0):
            xerr_fit = xe_fit

    initial_candidates: list[dict[str, float] | None] = [None]
    heuristic_seed = _transport_seed_initial_values(x_fit, y_fit, e_fit, model, parameters)
    if heuristic_seed is not None:
        initial_candidates.append(heuristic_seed)

    best_result: ParameterModelFitResult | None = None
    best_fval = float("inf")

    for initial_values in initial_candidates:
        result, fval = _run_parameter_model_minuit(
            x_fit=x_fit,
            y_fit=y_fit,
            e_fit=e_fit,
            model=model,
            parameters=parameters,
            method=method,
            initial_values=initial_values,
            xerr_fit=xerr_fit,
        )
        if best_result is None:
            best_result = result
            best_fval = fval
            continue
        if result.success and not best_result.success:
            best_result = result
            best_fval = fval
            continue
        if result.success == best_result.success and fval < best_fval:
            best_result = result
            best_fval = fval

    if best_result is None:
        return ParameterModelFitResult(
            success=False, message="Fit failed", error_mode=error_mode.value
        )

    best_result.error_mode = error_mode.value
    best_result.n_points = int(len(x_fit))
    if error_mode is ErrorMode.SCATTER and best_result.success:
        # Estimate errors from the scatter of the points: the unit-weight fit
        # location is independent of a uniform σ rescale, so multiplying the
        # parameter errors by √(χ²/ν) is exactly the fixed point of WiMDA's
        # iterated Estimate mode (σ ← σ·√χ²ᵣ until χ²ᵣ = 1).
        ndof = len(x_fit) - len(parameters.free_parameters)
        if ndof < 1:
            # An (over)determined interpolation has no residual scatter to
            # estimate errors from — χ² ≈ 0 would collapse the rescaled
            # errors to ~0, reporting an indeterminate fit as exact.
            best_result.uncertainties = {}
            best_result.message = (
                f"{best_result.message}; no degrees of freedom to estimate errors from scatter"
            )
        else:
            scale = float(np.sqrt(best_result.chi_squared / ndof))
            if np.isfinite(scale) and scale > 0.0:
                best_result.uncertainties = {
                    name: err * scale for name, err in best_result.uncertainties.items()
                }
    return best_result


def global_fit_parameter_model(
    groups: list[ParameterGroupData],
    model: ParameterCompositeModel,
    global_params: list[str],
    local_params: list[str],
    fixed_params: dict[str, float],
    initial_params: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    method: str = "migrad",
    error_mode: ErrorMode | str = ErrorMode.COLUMN,
    error_value: float | None = None,
    windows: Sequence[tuple[float, float]] | None = None,
    xerr: Mapping[str, NDArray] | None = None,
) -> CrossGroupFitResult:
    """Jointly fit a parameter model across multiple groups.

    Parameters are classified as:
    - global: one shared value across all groups
    - local: independent value per group
    - fixed: fixed constant

    ``error_mode``/``error_value`` select the per-point σ assignment for every
    group (see :class:`ErrorMode`); ``windows`` optionally restricts each
    group to a union of (min, max) intervals (OR-combined, one model across the
    union). ``SCATTER`` rescales *all* parameter errors (global and local) by
    √(χ²/ν) after the fit — the fixed point of WiMDA's Estimate iteration.

    ``xerr`` optionally maps ``group_id`` → per-point x-uncertainties (aligned
    to that group's stored ``x``) for an errors-in-variables (Orear/York
    effective-variance) fit, used for parameter-vs-parameter trending where the
    abscissa is itself a fitted quantity. It uses the same central-difference
    estimator as the single-series :func:`fit_parameter_model`. A group with no
    entry, or all-zero/non-finite σ_x, keeps ordinary least squares. Like the
    single-series path, ``xerr`` is ignored under ``NONE``/``SCATTER``, whose
    unit y-weights carry no physical scale to combine with the x-variance term.
    """
    error_mode = ErrorMode(error_mode)
    # Effective variance combines σ_x with a real per-point σ_y; under unit
    # weights (NONE) or scatter estimation (SCATTER) the "σ_y" is a placeholder
    # of arbitrary scale, so the combination would be scale-dependent — ignore
    # x-errors in those modes (matching the single-series path and the GUI
    # toggle's enable rule).
    use_xerr = xerr is not None and error_mode not in (ErrorMode.NONE, ErrorMode.SCATTER)
    if len(groups) < 2:
        return CrossGroupFitResult(
            success=False,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            message="Need at least two groups for cross-group fitting",
        )

    if initial_params is None:
        initial_params = dict(model.param_defaults)
    if parameter_bounds is None:
        parameter_bounds = {}

    all_param_names = set(model.param_names)
    fixed_names = set(fixed_params.keys())
    global_names = set(global_params)
    local_names = set(local_params)

    if (fixed_names | global_names | local_names) - all_param_names:
        unknown = sorted((fixed_names | global_names | local_names) - all_param_names)
        return CrossGroupFitResult(
            success=False,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            message=f"Unknown parameter classification: {unknown}",
        )

    unknown_bounds = set(parameter_bounds.keys()) - all_param_names
    if unknown_bounds:
        return CrossGroupFitResult(
            success=False,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            message=f"Unknown parameter bounds: {sorted(unknown_bounds)}",
        )

    unclassified = all_param_names - fixed_names - global_names - local_names
    # Default any unspecified parameters to global to avoid accidental omission.
    global_names |= unclassified

    bounds_by_param: dict[str, tuple[float, float]] = {}
    for pname in all_param_names:
        raw_bounds = parameter_bounds.get(pname, (-float("inf"), float("inf")))
        try:
            p_min = float(raw_bounds[0])
        except (TypeError, ValueError, IndexError):
            p_min = -float("inf")
        try:
            p_max = float(raw_bounds[1])
        except (TypeError, ValueError, IndexError):
            p_max = float("inf")
        if p_min > p_max:
            p_min, p_max = p_max, p_min
        bounds_by_param[pname] = (p_min, p_max)

    fixed_values: dict[str, float] = {}
    for pname in sorted(fixed_names):
        init_val = float(
            fixed_params.get(pname, initial_params.get(pname, model.param_defaults.get(pname, 0.0)))
        )
        p_min, p_max = bounds_by_param[pname]
        fixed_values[pname] = min(max(init_val, p_min), p_max)

    try:
        from iminuit import Minuit
    except ImportError as exc:
        return CrossGroupFitResult(
            success=False,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            message=f"iminuit import error: {exc}",
        )

    # Build Minuit parameter vector with deterministic names.
    fit_param_names: list[str] = []
    fit_init_values: list[float] = []
    fit_limits: dict[str, tuple[float, float]] = {}

    for pname in sorted(global_names):
        minuit_name = f"g__{pname}"
        fit_param_names.append(minuit_name)
        init_val = float(initial_params.get(pname, model.param_defaults.get(pname, 0.0)))
        p_min, p_max = bounds_by_param[pname]
        fit_init_values.append(min(max(init_val, p_min), p_max))
        fit_limits[minuit_name] = (p_min, p_max)

    for gidx, group in enumerate(groups):
        for pname in sorted(local_names):
            minuit_name = f"l__{gidx}__{pname}"
            fit_param_names.append(minuit_name)
            init_val = float(initial_params.get(pname, model.param_defaults.get(pname, 0.0)))
            p_min, p_max = bounds_by_param[pname]
            fit_init_values.append(min(max(init_val, p_min), p_max))
            fit_limits[minuit_name] = (p_min, p_max)

    def _build_kwargs(arg_map: dict[str, float], gidx: int) -> dict[str, float]:
        kwargs = {name: float(val) for name, val in fixed_values.items()}
        for pname in global_names:
            kwargs[pname] = float(arg_map[f"g__{pname}"])
        for pname in local_names:
            kwargs[pname] = float(arg_map[f"l__{gidx}__{pname}"])
        return kwargs

    # Precompute per-group fit arrays once: the window mask, finite mask and the
    # per-point σ under the chosen error mode are all data-only (independent of
    # the fitted parameters), so they need not be rebuilt every cost call. When
    # x-uncertainty is active a group also carries its σ_x² and the finite
    # -difference step (None → ordinary least squares for that group).
    group_fit_arrays: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
        ]
    ] = []
    total_points = 0
    for group in groups:
        x = np.asarray(group.x, dtype=float)
        y = np.asarray(group.y, dtype=float)
        sigma = apply_error_mode(y, group.yerr, error_mode, error_value)
        if sigma is None:
            sigma = np.ones_like(x)
        try:
            window_sel = windows_mask(x, windows)
        except ValueError as exc:
            return CrossGroupFitResult(
                success=False,
                chi_squared=0.0,
                reduced_chi_squared=0.0,
                message=str(exc),
                error_mode=error_mode.value,
            )
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & window_sel
        x_masked = x[mask]
        xe2: NDArray[np.float64] | None = None
        xstep: NDArray[np.float64] | None = None
        if use_xerr:
            raw_xe = xerr.get(group.group_id) if xerr is not None else None
            if raw_xe is not None:
                xe = np.asarray(raw_xe, dtype=float)
                if xe.shape == x.shape:
                    xe_masked = xe[mask]
                    xe_masked = np.where(np.isfinite(xe_masked) & (xe_masked > 0.0), xe_masked, 0.0)
                    if np.any(xe_masked > 0.0):
                        xe2 = xe_masked**2
                        xstep = np.maximum(np.abs(x_masked), 1.0) * 1e-6
        group_fit_arrays.append((x_masked, y[mask], sigma[mask], xe2, xstep))
        total_points += int(np.count_nonzero(mask))

    def cost_function(*args: float) -> float:
        arg_map = dict(zip(fit_param_names, args, strict=False))
        total = 0.0
        for gidx, (xx, yy, ee, xe2, xstep) in enumerate(group_fit_arrays):
            if xx.size == 0:
                continue
            kwargs = _build_kwargs(arg_map, gidx)
            if xe2 is None or xstep is None:
                pred = np.asarray(model.function(xx, **kwargs), dtype=float)
                resid = (yy - pred) / ee
            else:
                resid = _effective_variance_residual(
                    partial(model.function, **kwargs), xx, yy, ee**2, xe2, xstep
                )
            total += float(np.sum(resid**2))
        return total

    m = Minuit(cost_function, *fit_init_values, name=fit_param_names)
    for i, minuit_name in enumerate(fit_param_names):
        p_min, p_max = fit_limits[minuit_name]
        if p_min != -float("inf") or p_max != float("inf"):
            m.limits[i] = (p_min, p_max)
    if method == "simplex":
        m.simplex()
    else:
        m.migrad()

    ndof = max(total_points - len(fit_param_names), 1)

    global_parameter_set = ParameterSet()
    local_parameter_sets: dict[str, ParameterSet] = {}
    fixed_parameter_set = ParameterSet()
    global_unc: dict[str, float] = {}
    local_unc: dict[str, dict[str, float]] = {}

    values_by_name = {name: float(m.values[name]) for name in fit_param_names}

    for pname in sorted(global_names):
        key = f"g__{pname}"
        val = values_by_name[key]
        p_min, p_max = bounds_by_param[pname]
        global_parameter_set.add(
            Parameter(name=pname, value=val, min=p_min, max=p_max, fixed=False)
        )
        err = m.errors[key]
        if err is not None and np.isfinite(err):
            global_unc[pname] = float(err)

    for gidx, group in enumerate(groups):
        pset = ParameterSet()
        unc_map: dict[str, float] = {}
        for pname in sorted(local_names):
            key = f"l__{gidx}__{pname}"
            val = values_by_name[key]
            p_min, p_max = bounds_by_param[pname]
            pset.add(Parameter(name=pname, value=val, min=p_min, max=p_max, fixed=False))
            err = m.errors[key]
            if err is not None and np.isfinite(err):
                unc_map[pname] = float(err)
        local_parameter_sets[group.group_id] = pset
        local_unc[group.group_id] = unc_map

    for pname in sorted(fixed_names):
        p_min, p_max = bounds_by_param[pname]
        fixed_parameter_set.add(
            Parameter(name=pname, value=fixed_values[pname], min=p_min, max=p_max, fixed=True)
        )

    message = "Fit successful" if m.valid else "Fit failed"
    if error_mode is ErrorMode.SCATTER and m.valid:
        # Estimate errors from the scatter of the points: rescale *all* (global
        # and local) parameter errors by √(χ²/ν), the fixed point of WiMDA's
        # iterated Estimate mode. χ²ᵣ then carries no goodness information.
        ndof_free = total_points - len(fit_param_names)
        if ndof_free < 1:
            global_unc = {}
            local_unc = {gid: {} for gid in local_unc}
            message += "; no degrees of freedom to estimate errors from scatter"
        else:
            scale = float(np.sqrt(float(m.fval) / ndof_free))
            if np.isfinite(scale) and scale > 0.0:
                global_unc = {name: err * scale for name, err in global_unc.items()}
                local_unc = {
                    gid: {name: err * scale for name, err in unc.items()}
                    for gid, unc in local_unc.items()
                }

    return CrossGroupFitResult(
        success=bool(m.valid),
        chi_squared=float(m.fval),
        reduced_chi_squared=float(m.fval) / float(ndof),
        global_parameters=global_parameter_set,
        local_parameters=local_parameter_sets,
        fixed_parameters=fixed_parameter_set,
        global_uncertainties=global_unc,
        local_uncertainties=local_unc,
        message=message,
        error_mode=error_mode.value,
        n_points=int(total_points),
    )


def evaluate_parameter_model_fit(
    fit: ParameterModelFit,
    num_points: int = 200,
) -> list[ParameterModelFitExecution]:
    """Return sampled curves for each successful active range."""
    curves: list[ParameterModelFitExecution] = []

    for idx, fit_range in enumerate(fit.ranges):
        if fit_range.result is None or not fit_range.result.success:
            continue

        try:
            # One model across the window union: sample the full envelope so
            # the curve is drawn continuously through the excluded gaps.
            x_min, x_max = effective_range_bounds(fit_range)
        except ValueError:
            # Invalid windows (e.g. inverted mid-edit): skip the curve rather
            # than raising inside a plotting path.
            continue
        if x_min is None or x_max is None:
            continue
        if x_max <= x_min:
            continue

        xs = np.linspace(float(x_min), float(x_max), num=max(2, int(num_points)), dtype=float)
        kwargs = {p.name: p.value for p in fit_range.result.parameters}
        ys = fit_range.model.function(xs, **kwargs)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            continue

        curves.append(
            ParameterModelFitExecution(
                range_index=idx,
                x=xs[mask],
                y=ys[mask],
                result=fit_range.result,
            )
        )

    return curves
