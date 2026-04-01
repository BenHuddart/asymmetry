"""Model fitting for fitted parameters as a function of field or temperature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.diffusion import lambda_total
from asymmetry.core.fitting.sc import models as sc_models
from asymmetry.core.fitting.parameters import (
    ParamInfo,
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


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


def _constant(x: NDArray, c: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(x, dtype=float), float(c), dtype=float)


def _linear(x: NDArray, m: float, b: float) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    return m * xx + b


def _power_law(x: NDArray, a: float, n: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    safe_x = np.maximum(np.abs(xx), 1e-12)
    return a * np.power(safe_x, n) + c


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


def _critical_divergence(x: NDArray, a: float, Tc: float, nu: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    dist = np.maximum(np.abs(xx - Tc), 1e-9)
    return a * np.power(dist, -nu) + c


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


def _lambda_bg(x: NDArray, lambda_BG: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(x, dtype=float), float(lambda_BG), dtype=float)


def _diffusion_lf_1d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=1),
        dtype=float,
    )


def _diffusion_lf_2d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=2),
        dtype=float,
    )


def _diffusion_lf_3d(
    x: NDArray,
    A: float,
    D_2D: float,
    D_perp: float = 0.0,
) -> NDArray[np.float64]:
    return np.asarray(
        lambda_total(x, C=A, D_nD=D_2D, D_perp=D_perp, lambda_0D=0.0, n=3),
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
        latex_equation=(
            r"\lambda_{1D}(B) = \frac{A^2}{4} J\left(B; n=1, D_{2D}, D_{\perp}\right)"
        ),
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
        latex_equation=(
            r"\lambda_{2D}(B) = \frac{A^2}{4} J\left(B; n=2, D_{2D}, D_{\perp}\right)"
        ),
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
        latex_equation=(
            r"\lambda_{3D}(B) = \frac{A^2}{4} J\left(B; n=3, D_{2D}, D_{\perp}\right)"
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
                formula_template=(
                    "{sigma_0}*({weight}*rho_1 + (1-{weight})*rho_2) + {sigma_bg}"
                ),
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
                formula_template=(
                    "{sigma_0}*({weight}*rho_s + (1-{weight})*rho_d) + {sigma_bg}"
                ),
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
        }
    )


_register_superconducting_components()

_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})


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

    def __init__(self, component_names: list[str], operators: list[str] | None = None) -> None:
        if not component_names:
            raise ValueError("Model must contain at least one component")

        missing = [name for name in component_names if name not in PARAMETER_MODEL_COMPONENTS]
        if missing:
            raise ValueError(f"Unknown component(s): {missing}")

        if operators is None:
            operators = ["+"] * (len(component_names) - 1)

        if len(operators) != max(len(component_names) - 1, 0):
            raise ValueError("operators length must be len(component_names) - 1")
        if any(op not in _ALLOWED_OPERATORS for op in operators):
            raise ValueError("operators must be one of '+', '-', '*', '/'")

        self.component_names = list(component_names)
        self.operators = list(operators)
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

    def _build_param_mapping(self) -> list[dict[str, str]]:
        counts = Counter(
            pname
            for component in self.components
            for pname in component.param_names
        )
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
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            mapping_text = {k: mapping[k] for k in component.param_names}
            terms.append(component.formula_template.format(**mapping_text))

        if not terms:
            return "0"

        expression = terms[0]
        for op, term in zip(self.operators, terms[1:], strict=True):
            expression = f"({expression}) {op} ({term})"
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
            else:
                result = result - value

        return result

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
        """Return indices for additive components (first term and '+' terms)."""
        if not self.components:
            return []
        indices = [0]
        for idx, op in enumerate(self.operators, start=1):
            if op == "+":
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


@dataclass
class ParameterModelFitResult:
    """Result of fitting a parameter-vs-x model."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class ModelFitRange:
    """A model and fit results over a specific x-range."""

    x_min: float | None
    x_max: float | None
    model: ParameterCompositeModel
    parameters: ParameterSet
    result: ParameterModelFitResult | None = None


@dataclass
class ParameterModelFit:
    """Model fits attached to a single parameter trace."""

    parameter_name: str
    x_key: str
    ranges: list[ModelFitRange] = field(default_factory=list)
    active: bool = True


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


def fit_parameter_model(
    x: NDArray,
    y: NDArray,
    yerr: NDArray | None,
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x_min: float | None = None,
    x_max: float | None = None,
    method: str = "migrad",
) -> ParameterModelFitResult:
    """Fit a parameter-vs-x model using iminuit."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yerr is None:
        ee = np.ones_like(xx)
    else:
        ee = np.asarray(yerr, dtype=float)

    mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee) & (ee > 0)
    if x_min is not None:
        mask &= xx >= float(x_min)
    if x_max is not None:
        mask &= xx <= float(x_max)

    if not np.any(mask):
        return ParameterModelFitResult(success=False, message="No valid points in selected range")

    x_fit = xx[mask]
    y_fit = yy[mask]
    e_fit = ee[mask]

    try:
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
    except ImportError as exc:
        return ParameterModelFitResult(success=False, message=f"iminuit import error: {exc}")

    free = parameters.free_parameters
    fixed_kw = {p.name: p.value for p in parameters if p.fixed}
    param_names = [p.name for p in free]

    def model_wrapper(x_local: NDArray, *args: float) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(param_names, args, strict=False))}
        return model.function(x_local, **kw)

    cost = LeastSquares(x_fit, y_fit, e_fit, model_wrapper)
    init = [p.value for p in free]
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
            result_params.add(Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=True))
        else:
            idx = param_names.index(p.name)
            value = float(m.values[idx])
            result_params.add(Parameter(name=p.name, value=value, min=p.min, max=p.max, fixed=False))
            err = m.errors[idx]
            if err is not None and np.isfinite(err):
                uncertainties[p.name] = float(err)

    ndof = max(len(x_fit) - len(free), 1)
    return ParameterModelFitResult(
        success=bool(m.valid),
        chi_squared=float(m.fval),
        reduced_chi_squared=float(m.fval) / ndof,
        parameters=result_params,
        uncertainties=uncertainties,
        message="Fit successful" if m.valid else "Fit failed",
    )


def global_fit_parameter_model(
    groups: list[ParameterGroupData],
    model: ParameterCompositeModel,
    global_params: list[str],
    local_params: list[str],
    fixed_params: dict[str, float],
    initial_params: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    method: str = "migrad",
) -> CrossGroupFitResult:
    """Jointly fit a parameter model across multiple groups.

    Parameters are classified as:
    - global: one shared value across all groups
    - local: independent value per group
    - fixed: fixed constant
    """
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

    def cost_function(*args: float) -> float:
        arg_map = dict(zip(fit_param_names, args, strict=False))
        total = 0.0
        for gidx, group in enumerate(groups):
            x = np.asarray(group.x, dtype=float)
            y = np.asarray(group.y, dtype=float)
            e = np.asarray(group.yerr, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
            if not np.any(mask):
                continue
            xx = x[mask]
            yy = y[mask]
            ee = e[mask]
            kwargs = _build_kwargs(arg_map, gidx)
            pred = np.asarray(model.function(xx, **kwargs), dtype=float)
            resid = (yy - pred) / ee
            total += float(np.sum(resid ** 2))
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

    total_points = 0
    for group in groups:
        x = np.asarray(group.x, dtype=float)
        y = np.asarray(group.y, dtype=float)
        e = np.asarray(group.yerr, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        total_points += int(np.sum(mask))

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
        global_parameter_set.add(Parameter(name=pname, value=val, min=p_min, max=p_max, fixed=False))
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

    return CrossGroupFitResult(
        success=bool(m.valid),
        chi_squared=float(m.fval),
        reduced_chi_squared=float(m.fval) / float(ndof),
        global_parameters=global_parameter_set,
        local_parameters=local_parameter_sets,
        fixed_parameters=fixed_parameter_set,
        global_uncertainties=global_unc,
        local_uncertainties=local_unc,
        message="Fit successful" if m.valid else "Fit failed",
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

        x_min = fit_range.x_min
        x_max = fit_range.x_max
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
