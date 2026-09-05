"""Composite fit-function builder primitives.

This module exposes baseline-free muSR components that can be combined with
``+``, ``-``, ``*``, and ``/`` to produce a single model callable compatible
with :class:`asymmetry.core.fitting.engine.FitEngine`.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.component_tags import (
    ALL_GEOMETRIES,
    ComputationalCost,
    FieldGeometry,
    PhysicsClass,
)
from asymmetry.core.fitting.latex_preview import (
    LatexTerm,
    fallback_function_latex,
    param_symbol_latex,
    transform_template,
)
from asymmetry.core.fitting.models import (
    MODELS,
    ModelDefinition,
    abragam,
    bessel_oscillation,
    dynamic_gaussian_kt,
    dynamic_lorentzian_kt,
    exponential_relaxation,
    gaussian_broadened_kt,
    gaussian_relaxation,
    keren,
    longitudinal_field_kubo_toyabe,
    risch_kehr,
    static_gkt_zf,
    stretched_exponential,
)
from asymmetry.core.fitting.muon_fluorine.polarization import (
    dynamic_fmuf_polarization,
    fmuf_triangle_polarization,
    general_fmuf_polarization,
    linear_fmuf_polarization,
    mu_f_polarization,
)
from asymmetry.core.fitting.muonium import (
    VACUUM_MUONIUM_A_HF_MHZ,
    high_tf_muonium,
    high_tf_muonium_aniso,
    low_tf_muonium,
    muonium_lf_relaxation,
    tf_muonium,
    zf_muonium,
)
from asymmetry.core.fitting.nuclear_dipole import (
    dipolar_pair_field,
    dipolar_spin_j,
    electron_dipole,
    proton_dipole,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, ParamInfo, get_param_info
from asymmetry.core.fitting.sc.lineshape import (
    vortex_lattice_component,
    vortex_lattice_powder_component,
)
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

#: MODELS keys that have no COMPONENTS entry of the same name but DO map to a
#: component under a different (expression) name. Fit *expressions* take
#: COMPONENTS names; these are the standalone names in
#: :data:`asymmetry.core.fitting.models.MODELS`. Mapping each to its expression
#: name lets the parser point a mis-typed expression at the right token — the
#: single most-repeated discoverability miss in clean-room API testing
#: (e.g. ``GaussianRelaxation`` used in an expression where ``Gaussian`` is
#: required). Drift is guarded by ``tests/test_composite_unknown_component_hint``.
_MODEL_NAME_TO_COMPONENT: dict[str, str] = {
    "GaussianRelaxation": "Gaussian",
    "ExponentialRelaxation": "Exponential",
    "LFKuboToyabe": "LongitudinalFieldKT",
}


def _unknown_component_hint(
    name: str, allowed: set[str] | frozenset[str] | None = None
) -> tuple[str, tuple[str, ...]]:
    """Build a corrective message + suggestions for an unknown component name.

    Returns ``(message, suggestions)`` where ``suggestions`` are valid
    expression names the caller can actually type. ``allowed`` is the set the
    name was checked against — defaults to the full ``COMPONENTS`` registry, but
    a restricted grammar (e.g. parameter-domain models) passes its own set so
    suggestions never point at names that are invalid in that context. Resolved
    at call time so it sees the live ``COMPONENTS``/``MODELS`` registries.
    """
    available = COMPONENTS if allowed is None else allowed
    base = f"Unknown component '{name}'"
    alias = _MODEL_NAME_TO_COMPONENT.get(name)
    if alias is not None and alias in available:
        message = (
            f"{base}. In fit expressions use '{alias}', not '{name}': "
            f"'{name}' is the standalone name in MODELS, while expressions take "
            f"COMPONENTS names (e.g. '{alias} + Constant')."
        )
        return message, (alias,)

    # Suggest from the allowed set only, and never echo the rejected name back.
    matches = difflib.get_close_matches(name, sorted(available), n=4, cutoff=0.6)
    suggestions = tuple(m for m in matches if m != name)[:3]
    if name in MODELS and name not in available:
        # A MODELS name with no direct component analogue: still steer the
        # caller toward COMPONENTS names, listing any near matches we found.
        hint = f" Did you mean {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
        message = (
            f"{base}. '{name}' is a MODELS name; fit expressions take COMPONENTS "
            f"names (see asymmetry.core.fitting.composite.COMPONENTS).{hint}"
        )
        return message, suggestions
    if suggestions:
        joined = ", ".join(repr(s) for s in suggestions)
        message = f"{base}. Did you mean {joined}? (expressions take COMPONENTS names)"
        return message, suggestions
    return f"{base}.", ()


class UnknownComponentError(ValueError):
    """An expression referenced a component name that is not registered.

    Carries the offending ``name`` so callers (e.g. the GUI builder) can
    produce targeted guidance without parsing the message text, and a tuple of
    valid-expression-name ``suggestions`` (closest COMPONENTS names, or the
    expression name for a mis-used MODELS key).
    """

    def __init__(self, name: str, allowed: set[str] | frozenset[str] | None = None) -> None:
        message, suggestions = _unknown_component_hint(name, allowed)
        super().__init__(message)
        self.name = name
        self.suggestions: tuple[str, ...] = suggestions


@dataclass(frozen=True)
class ComponentDefinition:
    """Descriptor for a baseline-free component function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]
    param_info: dict[str, ParamInfo]
    formula_template: str
    latex_equation: str = ""
    category: str = "General"
    domain: str = "time"
    #: Parameters that should start *fixed* in a fit (e.g. a nuclear spin the
    #: model is piecewise-constant in, or a hyperfine constant that is known).
    #: The GUI pre-checks the fix box; the user can always free them.
    fixed_params: tuple[str, ...] = ()
    #: ``True`` for components registered through the user-function facade
    #: (:mod:`asymmetry.core.fitting.user_functions`). Provenance is keyed off
    #: this flag — picker badges and the docs-enforcement exemptions — never
    #: off name lists.
    user: bool = False
    #: ``True`` for the per-instance placeholder definitions that stand in for
    #: a user component referenced by a project but not currently registered.
    #: Placeholders evaluate to zero and are never inserted into ``COMPONENTS``.
    missing: bool = False
    #: For a precession component whose fitted value is the *local* field/frequency
    #: at the muon (so it can be converted to a Knight shift), the name of that
    #: parameter (``"frequency"`` or ``"field"``). ``None`` for everything else —
    #: crucially including muonium components, whose ``field`` parameter is the
    #: *applied* field fed into the Breit–Rabi levels, not a local field.
    knight_observable: str | None = None
    #: Applied-field geometries the component is physically meaningful for. The
    #: fit wizard uses this to drop components that cannot apply to a run's
    #: geometry. Defaults to all three (no restriction).
    field_geometries: frozenset[FieldGeometry] = ALL_GEOMETRIES
    #: Physics-class tags for wizard scoping. ``CUSTOM`` is the untagged
    #: sentinel default for user functions; every built-in must override it
    #: with real class(es) — enforced by ``tests/test_component_tags.py``.
    physics_classes: frozenset[PhysicsClass] = frozenset({PhysicsClass.CUSTOM})
    #: Relative evaluation-cost hint for tiered wizard screening (the wizard
    #: trials cheap components before expensive ones).
    cost: ComputationalCost = ComputationalCost.MODERATE


def _exp_component(t: NDArray, A: float, Lambda: float) -> NDArray[np.float64]:
    return exponential_relaxation(t, A0=A, Lambda=Lambda, baseline=0.0)


def _gaussian_component(t: NDArray, A: float, sigma: float) -> NDArray[np.float64]:
    return gaussian_relaxation(t, A0=A, sigma=sigma, baseline=0.0)


def _oscillatory_component(
    t: NDArray,
    A: float,
    frequency: float,
    phase: float,
) -> NDArray[np.float64]:
    return A * np.cos(2.0 * np.pi * frequency * t + phase)


def _oscillatory_field_component(
    t: NDArray,
    A: float,
    field: float,
    phase: float,
) -> NDArray[np.float64]:
    frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * float(field)
    return A * np.cos(2.0 * np.pi * frequency * t + phase)


def _muonium_tf_component(
    t: NDArray, A: float, field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    return A * tf_muonium(t, field, A_hf, phase)


def _muonium_low_tf_component(
    t: NDArray, A: float, field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    return A * low_tf_muonium(t, field, A_hf, phase)


def _muonium_zf_component(
    t: NDArray, A: float, A_hf: float, D_mu: float, f_cut: float, phase: float
) -> NDArray[np.float64]:
    return A * zf_muonium(t, A_hf, D_mu, f_cut, phase)


def _stretched_component(
    t: NDArray,
    A: float,
    Lambda: float,
    beta: float,
) -> NDArray[np.float64]:
    return stretched_exponential(t, A0=A, Lambda=Lambda, beta=beta, baseline=0.0)


def _gkt_component(t: NDArray, A: float, Delta: float) -> NDArray[np.float64]:
    return static_gkt_zf(t, A0=A, Delta=Delta, baseline=0.0)


def _lf_kt_component(t: NDArray, A: float, Delta: float, B_L: float) -> NDArray[np.float64]:
    """Longitudinal-field Kubo-Toyabe depolarization function.

    Wrapper adapting longitudinal_field_kubo_toyabe for use as a composite component.
    """
    return longitudinal_field_kubo_toyabe(t, A0=A, Delta=Delta, B_L=B_L, baseline=0.0)


def _dynamic_gkt_component(
    t: NDArray, A: float, Delta: float, nu: float, B_L: float
) -> NDArray[np.float64]:
    """Dynamic (strong-collision) Gaussian Kubo-Toyabe composite component."""
    return dynamic_gaussian_kt(t, A0=A, Delta=Delta, nu=nu, B_L=B_L, baseline=0.0)


def _dynamic_lkt_component(
    t: NDArray, A: float, a_L: float, nu: float, B_L: float
) -> NDArray[np.float64]:
    """Dynamic (strong-collision) Lorentzian Kubo-Toyabe composite component."""
    return dynamic_lorentzian_kt(t, A0=A, a_L=a_L, nu=nu, B_L=B_L, baseline=0.0)


def _keren_component(
    t: NDArray, A: float, Delta: float, nu: float, B_L: float
) -> NDArray[np.float64]:
    """Keren dynamic Gaussian LF relaxation composite component."""
    return keren(t, A0=A, Delta=Delta, nu=nu, B_L=B_L, baseline=0.0)


def _abragam_component(t: NDArray, A: float, Delta: float, nu: float) -> NDArray[np.float64]:
    """Abragam relaxation composite component."""
    return abragam(t, A0=A, Delta=Delta, nu=nu, baseline=0.0)


def _constant_component(t: NDArray, A_bg: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(t, dtype=float), fill_value=A_bg, dtype=float)


def _gaussian_peak_component(
    t: NDArray,
    height: float,
    nu0: float,
    fwhm: float,
) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    width = max(abs(float(fwhm)), 1e-12)
    exponent = -4.0 * np.log(2.0) * ((x - float(nu0)) / width) ** 2
    return float(height) * np.exp(exponent)


def _lorentzian_peak_component(
    t: NDArray,
    height: float,
    nu0: float,
    fwhm: float,
) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    width = max(abs(float(fwhm)), 1e-12)
    return float(height) / (1.0 + 4.0 * ((x - float(nu0)) / width) ** 2)


def _constant_background_component(t: NDArray, bg: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(t, dtype=float), fill_value=float(bg), dtype=float)


def _linear_background_component(t: NDArray, bg: float, slope: float) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    return float(bg) + float(slope) * x


def _risch_kehr_component(t: NDArray, A: float, Gamma: float) -> NDArray[np.float64]:
    return A * risch_kehr(t, Gamma)


def _bessel_component(t: NDArray, A: float, frequency: float, phase: float) -> NDArray[np.float64]:
    return A * bessel_oscillation(t, frequency, phase)


def _gaussian_broadened_kt_component(
    t: NDArray, A: float, Delta: float, B_L: float, w_rel: float
) -> NDArray[np.float64]:
    return A * gaussian_broadened_kt(t, Delta, B_L, w_rel)


def _muonium_high_tf_component(
    t: NDArray, A: float, field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    return A * high_tf_muonium(t, field, A_hf, phase)


def _muonium_high_tf_aniso_component(
    t: NDArray, A: float, field: float, A_hf: float, D_mu: float, phase: float
) -> NDArray[np.float64]:
    return A * high_tf_muonium_aniso(t, field, A_hf, D_mu, phase)


def _muonium_lf_relax_component(
    t: NDArray, A: float, delta_ex: float, tau_c: float, B_L: float, A_hf: float
) -> NDArray[np.float64]:
    return A * muonium_lf_relaxation(t, delta_ex, tau_c, B_L, A_hf)


def _dipolar_pair_field_component(
    t: NDArray, A: float, B_dip: float, lambda_T: float
) -> NDArray[np.float64]:
    return A * dipolar_pair_field(t, B_dip, lambda_T)


def _invalid_trial_penalty(t: NDArray) -> NDArray[np.float64]:
    """Flat penalty curve returned for invalid trial geometries.

    Keeps minimization alive when the optimiser probes an unphysical point
    (e.g. a distance at its inclusive zero bound) instead of aborting the fit
    with an exception.
    """
    return np.full_like(np.asarray(t, dtype=float), fill_value=1.0e3, dtype=float)


def _proton_dipole_component(
    t: NDArray, A: float, r_muH: float, lambda_T: float
) -> NDArray[np.float64]:
    try:
        return A * proton_dipole(t, r_muH, lambda_T)
    except ValueError:
        return _invalid_trial_penalty(t)


def _electron_dipole_component(
    t: NDArray, A: float, r_mue: float, lambda_T: float
) -> NDArray[np.float64]:
    try:
        return A * electron_dipole(t, r_mue, lambda_T)
    except ValueError:
        return _invalid_trial_penalty(t)


def _dipolar_spin_j_component(
    t: NDArray, A: float, f_dip: float, f_quad: float, J_spin: float
) -> NDArray[np.float64]:
    return A * dipolar_spin_j(t, f_dip, f_quad, J_spin)


def _dynamic_fmuf_component(t: NDArray, A: float, r_muF: float, nu: float) -> NDArray[np.float64]:
    try:
        return A * dynamic_fmuf_polarization(t, r_muF, nu)
    except ValueError:
        return _invalid_trial_penalty(t)


def _fmuf_triangle_component(
    t: NDArray, A: float, r_muF: float, r3: float, phi3: float
) -> NDArray[np.float64]:
    try:
        return A * fmuf_triangle_polarization(t, r_muF, r3, phi3)
    except ValueError:
        return _invalid_trial_penalty(t)


def _muf_component(t: NDArray, A: float, r_muF: float) -> NDArray[np.float64]:
    try:
        return A * mu_f_polarization(t, r_muF)
    except ValueError:
        return _invalid_trial_penalty(t)


def _linear_fmuf_component(t: NDArray, A: float, r_muF: float) -> NDArray[np.float64]:
    try:
        return A * linear_fmuf_polarization(t, r_muF)
    except ValueError:
        return _invalid_trial_penalty(t)


def _general_fmuf_component(
    t: NDArray,
    A: float,
    r1: float,
    r2: float,
    theta: float,
) -> NDArray[np.float64]:
    try:
        return A * general_fmuf_polarization(t, r1, r2, theta)
    except ValueError:
        return _invalid_trial_penalty(t)


#: Canonical component-picker categories in display order, mapped to the stem
#: of their reference page under ``docs/reference/fit_functions/``.  This is
#: the single source of truth consumed by the GUI picker (submenu order) and
#: by the documentation-placement test — a new category must be registered
#: here (with a docs page) before components can use it.  "General" is the
#: default bucket and renders at the top level of the picker rather than as a
#: submenu; it must stay empty for time-domain components.
CATEGORY_REGISTRY: dict[str, str] = {
    "General": "",
    "Relaxation": "relaxation",
    "Oscillation": "oscillation",
    "Kubo-Toyabe": "kubo_toyabe",
    "Muonium": "muonium",
    "Nuclear dipolar": "nuclear_dipolar",
    "Background": "background",
    "Frequency Domain": "frequency_domain",
}


COMPONENTS: dict[str, ComponentDefinition] = {
    "Exponential": ComponentDefinition(
        name="Exponential",
        description="A exp(-Lambda t)",
        function=_exp_component,
        param_names=["A", "Lambda"],
        param_defaults={"A": 25.0, "Lambda": 0.5},
        param_info={"A": get_param_info("A"), "Lambda": get_param_info("Lambda")},
        formula_template="{A}*exp(-{Lambda}*t)",
        latex_equation=r"A(t) = A e^{-\Lambda t}",
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.GENERIC_RELAXATION}),
        cost=ComputationalCost.CHEAP,
    ),
    "Gaussian": ComponentDefinition(
        name="Gaussian",
        description="A exp(-(sigma t)^2)",
        function=_gaussian_component,
        param_names=["A", "sigma"],
        param_defaults={"A": 25.0, "sigma": 0.5},
        param_info={"A": get_param_info("A"), "sigma": get_param_info("sigma")},
        formula_template="{A}*exp(-({sigma}*t)^2)",
        latex_equation=r"A(t) = A e^{-(\sigma t)^2}",
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.GENERIC_RELAXATION}),
        cost=ComputationalCost.CHEAP,
    ),
    "Oscillatory": ComponentDefinition(
        name="Oscillatory",
        description="A cos(2 pi f t + phase)",
        function=_oscillatory_component,
        param_names=["A", "frequency", "phase"],
        param_defaults={"A": 25.0, "frequency": 1.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "frequency": get_param_info("frequency"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*cos(2*pi*{frequency}*t + {phase})",
        latex_equation=r"A(t) = A \cos(2\pi f t + \phi)",
        category="Oscillation",
        knight_observable="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "OscillatoryField": ComponentDefinition(
        name="OscillatoryField",
        description="A cos(2 pi gamma_mu B t + phase)",
        function=_oscillatory_field_component,
        param_names=["A", "field", "phase"],
        param_defaults={"A": 25.0, "field": 100.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*cos(2*pi*gamma_mu*{field}*t + {phase})",
        latex_equation=r"A(t) = A \cos(2\pi \gamma_\mu B t + \phi)",
        category="Oscillation",
        knight_observable="field",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "VortexLattice": ComponentDefinition(
        name="VortexLattice",
        description=(
            "Single-crystal vortex-lattice TF oscillation with the modified-London "
            "field distribution (non-Gaussian, skewed); fits lambda and Bc2"
        ),
        function=vortex_lattice_component,
        param_names=["A", "field", "phase", "lambda_ab", "Bc2"],
        param_defaults={"A": 20.0, "field": 100.0, "phase": 0.0, "lambda_ab": 200.0, "Bc2": 10.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "phase": get_param_info("phase"),
            "lambda_ab": get_param_info("lambda_ab"),
            "Bc2": get_param_info("Bc2"),
        },
        formula_template="{A}*Re[exp(i(2*pi*gamma_mu*{field}*t + {phase})) * R_VL(t; {lambda_ab}, {Bc2})]",
        latex_equation=r"A(t) = A\,\mathrm{Re}\!\left[e^{i(2\pi\gamma_\mu B t + \phi)} R_{VL}(t;\lambda,B_{c2})\right]",
        category="Oscillation",
        fixed_params=("field",),
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.SUPERCONDUCTIVITY}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "VortexLatticePowder": ComponentDefinition(
        name="VortexLatticePowder",
        description=(
            "Polycrystalline vortex-lattice TF oscillation (modified-London, skewed "
            "line, 3^(1/4) powder average); fits ab-plane lambda_ab and Bc2"
        ),
        function=vortex_lattice_powder_component,
        param_names=["A", "field", "phase", "lambda_ab", "Bc2"],
        param_defaults={"A": 20.0, "field": 100.0, "phase": 0.0, "lambda_ab": 200.0, "Bc2": 10.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "phase": get_param_info("phase"),
            "lambda_ab": get_param_info("lambda_ab"),
            "Bc2": get_param_info("Bc2"),
        },
        formula_template="{A}*Re[exp(i(2*pi*gamma_mu*{field}*t + {phase})) * R_VL_powder(t; {lambda_ab}, {Bc2})]",
        latex_equation=r"A(t) = A\,\mathrm{Re}\!\left[e^{i(2\pi\gamma_\mu B t + \phi)} R_{VL}^{\mathrm{pow}}(t;\lambda_{ab},B_{c2})\right]",
        category="Oscillation",
        fixed_params=("field",),
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.SUPERCONDUCTIVITY}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "Bessel": ComponentDefinition(
        name="Bessel",
        description="Zeroth-order Bessel oscillation for incommensurate (SDW) order",
        function=_bessel_component,
        param_names=["A", "frequency", "phase"],
        param_defaults={"A": 25.0, "frequency": 1.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "frequency": get_param_info("frequency"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*J0(2*pi*{frequency}*t + {phase})",
        latex_equation=r"A(t) = A\,J_0(2\pi f t + \phi)",
        category="Oscillation",
        knight_observable="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuoniumTF": ComponentDefinition(
        name="MuoniumTF",
        description="Transverse-field muonium: four Mu0 transitions about gamma_mu B",
        function=_muonium_tf_component,
        param_names=["A", "field", "A_hf", "phase"],
        param_defaults={"A": 25.0, "field": 100.0, "A_hf": 0.24, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "A_hf": get_param_info("A_hf"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*TFmuonium(t; {field}, {A_hf}, {phase})",
        latex_equation=(
            r"A(t) = \frac{A}{4}\sum_{ij}(1\pm\delta)\cos(2\pi w_{ij} t + \phi),"
            r"\ \ w_{ij}=E_i-E_j,\ \delta=\frac{x}{\sqrt{1+x^2}},"
            r"\ x=\frac{(\gamma_e+\gamma_\mu)B}{A_\mu}"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuoniumLowTF": ComponentDefinition(
        name="MuoniumLowTF",
        description="Low transverse-field muonium: two Mu0 satellite frequencies",
        function=_muonium_low_tf_component,
        param_names=["A", "field", "A_hf", "phase"],
        param_defaults={"A": 25.0, "field": 100.0, "A_hf": 0.24, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "A_hf": get_param_info("A_hf"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*LowTFmuonium(t; {field}, {A_hf}, {phase})",
        latex_equation=(
            r"A(t) = \frac{A}{4}\left[(1+\delta)\cos(2\pi w_{12} t + \phi)"
            r"+(1-\delta)\cos(2\pi w_{23} t + \phi)\right]"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuoniumZF": ComponentDefinition(
        name="MuoniumZF",
        description="Zero-field axial muonium: three hyperfine lines",
        function=_muonium_zf_component,
        param_names=["A", "A_hf", "D_mu", "f_cut", "phase"],
        param_defaults={"A": 25.0, "A_hf": 1.0, "D_mu": 0.5, "f_cut": 0.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "A_hf": get_param_info("A_hf"),
            "D_mu": get_param_info("D_mu"),
            "f_cut": get_param_info("f_cut"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*ZFmuonium(t; {A_hf}, {D_mu}, {f_cut}, {phase})",
        latex_equation=(
            r"A(t) = \frac{A}{6}\sum_k a_k\cos(2\pi f_k t + \phi),"
            r"\ f_1=A_\mu-D,\ f_2=A_\mu+\frac{D}{2},\ f_3=\frac{3D}{2}"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuoniumHighTF": ComponentDefinition(
        name="MuoniumHighTF",
        description="High transverse-field muonium: the nu_12/nu_34 intratriplet pair",
        function=_muonium_high_tf_component,
        param_names=["A", "field", "A_hf", "phase"],
        param_defaults={
            "A": 25.0,
            "field": 3000.0,
            "A_hf": VACUUM_MUONIUM_A_HF_MHZ,
            "phase": 0.0,
        },
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "A_hf": get_param_info("A_hf"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*HighTFmuonium(t; {field}, {A_hf}, {phase})",
        latex_equation=(
            r"A(t) = \frac{A}{2}\left[\cos(2\pi\nu_{12}t+\phi)+\cos(2\pi\nu_{34}t+\phi)\right],"
            r"\ \nu_{12}+\nu_{34}=A_\mu"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuoniumHighTFAniso": ComponentDefinition(
        name="MuoniumHighTFAniso",
        description="Powder-averaged anisotropic high-TF muonium pair (axial D)",
        function=_muonium_high_tf_aniso_component,
        param_names=["A", "field", "A_hf", "D_mu", "phase"],
        param_defaults={
            "A": 25.0,
            "field": 3000.0,
            "A_hf": VACUUM_MUONIUM_A_HF_MHZ,
            "D_mu": 10.0,
            "phase": 0.0,
        },
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "A_hf": get_param_info("A_hf"),
            "D_mu": get_param_info("D_mu"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*HighTFmuoniumAniso(t; {field}, {A_hf}, {D_mu}, {phase})",
        latex_equation=(
            r"A(t) = \frac{A}{2}\left\langle\cos\left(2\pi(\nu_{34}+\frac{d}{2})t+\phi\right)"
            r"+\cos\left(2\pi(\nu_{12}-\frac{d}{2})t+\phi\right)\right\rangle_{\cos\theta},"
            r"\ d=\frac{D}{2}(3\cos^2\theta-1)"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "MuoniumLFRelax": ComponentDefinition(
        name="MuoniumLFRelax",
        description="Muonium longitudinal-field T1 relaxation (BPP at the nu_12 transition)",
        function=_muonium_lf_relax_component,
        param_names=["A", "delta_ex", "tau_c", "B_L", "A_hf"],
        param_defaults={
            "A": 25.0,
            "delta_ex": 0.5,
            "tau_c": 0.1,
            "B_L": 10.0,
            "A_hf": VACUUM_MUONIUM_A_HF_MHZ,
        },
        param_info={
            "A": get_param_info("A"),
            "delta_ex": get_param_info("delta_ex"),
            "tau_c": get_param_info("tau_c"),
            "B_L": get_param_info("B_L"),
            "A_hf": get_param_info("A_hf"),
        },
        formula_template="{A}*exp(-lambda({delta_ex},{tau_c},{B_L},{A_hf})*t)",
        fixed_params=("A_hf",),
        latex_equation=(
            r"A(t) = A e^{-\lambda t},\ \ \lambda = "
            r"\frac{(1-\delta)\,\delta_{ex}^2\,\tau_c}{1+(2\pi\nu_{12}\tau_c)^2},"
            r"\ \delta=\frac{x}{\sqrt{1+x^2}}"
        ),
        category="Muonium",
        field_geometries=frozenset({FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.MUONIUM, PhysicsClass.DYNAMICS}),
        cost=ComputationalCost.CHEAP,
    ),
    "StretchedExponential": ComponentDefinition(
        name="StretchedExponential",
        description="A exp(-(|Lambda| t)^beta)",
        function=_stretched_component,
        param_names=["A", "Lambda", "beta"],
        param_defaults={"A": 25.0, "Lambda": 0.5, "beta": 1.0},
        param_info={
            "A": get_param_info("A"),
            "Lambda": get_param_info("Lambda"),
            "beta": get_param_info("beta"),
        },
        formula_template="{A}*exp(-(abs({Lambda})*t)^({beta}))",
        latex_equation=r"A(t) = A \exp\left(-(|\Lambda| t)^\beta\right)",
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.GENERIC_RELAXATION}),
        cost=ComputationalCost.CHEAP,
    ),
    "RischKehr": ComponentDefinition(
        name="RischKehr",
        description="Risch-Kehr relaxation from 1D diffusive spin transport",
        function=_risch_kehr_component,
        param_names=["A", "Gamma"],
        param_defaults={"A": 25.0, "Gamma": 1.0},
        param_info={"A": get_param_info("A"), "Gamma": get_param_info("Gamma")},
        formula_template="{A}*exp({Gamma}*t)*erfc(sqrt({Gamma}*t))",
        latex_equation=r"A(t) = A\, e^{\Gamma t}\,\mathrm{erfc}\!\left(\sqrt{\Gamma t}\right)",
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.DYNAMICS}),
        cost=ComputationalCost.CHEAP,
    ),
    "StaticGKT_ZF": ComponentDefinition(
        name="StaticGKT_ZF",
        description="Static Gaussian Kubo-Toyabe (zero field)",
        function=_gkt_component,
        param_names=["A", "Delta"],
        param_defaults={"A": 25.0, "Delta": 0.5},
        param_info={"A": get_param_info("A"), "Delta": get_param_info("Delta")},
        formula_template=("{A}*(1/3 + 2/3*(1-({Delta}*t)^2)*exp(-({Delta}*t)^2/2))"),
        latex_equation=(
            r"A(t) = A\left[\frac{1}{3} + \frac{2}{3}\left(1-(\Delta t)^2\right)e^{-(\Delta t)^2/2}\right]"
        ),
        category="Kubo-Toyabe",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "LongitudinalFieldKT": ComponentDefinition(
        name="LongitudinalFieldKT",
        description="Static Gaussian Kubo-Toyabe with longitudinal field (Hayano et al. 1979)",
        function=_lf_kt_component,
        param_names=["A", "Delta", "B_L"],
        param_defaults={"A": 25.0, "Delta": 0.5, "B_L": 0.0},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "B_L": get_param_info("B_L"),
        },
        formula_template="{A}*Gz(t; Delta={Delta}, B_L={B_L})",
        latex_equation=(
            r"A(t) = A\left[1 - \frac{2\Delta^2}{\omega_0^2}\left(1 - e^{-\Delta^2 t^2/2}\cos(\omega_0 t)\right) "
            r"+ \frac{2\Delta^4}{\omega_0^3}\int_0^t e^{-\Delta^2\tau^2/2}\sin(\omega_0\tau)\,d\tau\right] "
            r"\quad\text{where}\quad \omega_0 = \gamma_\mu B_L"
        ),
        category="Kubo-Toyabe",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.MODERATE,
    ),
    "DynamicGaussianKT": ComponentDefinition(
        name="DynamicGaussianKT",
        description=(
            "Dynamic Gaussian Kubo-Toyabe (strong-collision; Hayano et al., "
            "Phys. Rev. B 20, 850 (1979))"
        ),
        function=_dynamic_gkt_component,
        param_names=["A", "Delta", "nu", "B_L"],
        param_defaults={"A": 25.0, "Delta": 0.5, "nu": 1.0, "B_L": 0.0},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "nu": get_param_info("nu"),
            "B_L": get_param_info("B_L"),
        },
        formula_template="{A}*G_dyn(t; Delta={Delta}, nu={nu}, B_L={B_L})",
        latex_equation=(
            r"A(t)=A\,G^{\mathrm{dyn}}_{\mathrm{GKT}}(t;\Delta,\nu,B_L),\quad "
            r"G_d(t)=g(t)+\nu\!\int_0^t\! g(t-\tau)\,G_d(\tau)\,d\tau,\ "
            r"g(t)=e^{-\nu t}G^{\mathrm{stat}}_{\mathrm{GKT}}(t)"
        ),
        category="Kubo-Toyabe",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM, PhysicsClass.DYNAMICS}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "DynamicLorentzianKT": ComponentDefinition(
        name="DynamicLorentzianKT",
        description=(
            "Dynamic Lorentzian Kubo-Toyabe (strong-collision; Uemura et al., "
            "Phys. Rev. B 31, 546 (1985))"
        ),
        function=_dynamic_lkt_component,
        param_names=["A", "a_L", "nu", "B_L"],
        param_defaults={"A": 25.0, "a_L": 0.5, "nu": 1.0, "B_L": 0.0},
        param_info={
            "A": get_param_info("A"),
            "a_L": get_param_info("a_L"),
            "nu": get_param_info("nu"),
            "B_L": get_param_info("B_L"),
        },
        formula_template="{A}*G_dyn_L(t; a_L={a_L}, nu={nu}, B_L={B_L})",
        latex_equation=(
            r"A(t)=A\,G^{\mathrm{dyn}}_{\mathrm{LKT}}(t;a_L,\nu,B_L),\quad "
            r"G^{\mathrm{stat}}_{\mathrm{LKT}}(t)=\frac{1}{3}+\frac{2}{3}(1-a_L t)e^{-a_L t}"
        ),
        category="Kubo-Toyabe",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM, PhysicsClass.DYNAMICS}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "GaussianBroadenedKT": ComponentDefinition(
        name="GaussianBroadenedKT",
        description="Static (LF) Gaussian Kubo-Toyabe averaged over a Gaussian spread of Delta",
        function=_gaussian_broadened_kt_component,
        param_names=["A", "Delta", "B_L", "w_rel"],
        param_defaults={"A": 25.0, "Delta": 0.5, "B_L": 0.0, "w_rel": 0.2},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "B_L": get_param_info("B_L"),
            "w_rel": get_param_info("w_rel"),
        },
        formula_template="{A}*<G_KT(t; Delta', {B_L})>_(Delta'~N({Delta},{w_rel}*{Delta}))",
        latex_equation=(
            r"A(t)=A\!\int\! d\Delta'\,p(\Delta')\,G_{\mathrm{KT}}(t;\Delta',B_L),\ "
            r"p=\mathcal{N}(\Delta,(w_\Delta\Delta)^2)"
        ),
        category="Kubo-Toyabe",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "Keren": ComponentDefinition(
        name="Keren",
        description=(
            "Keren dynamic Gaussian relaxation in a longitudinal field "
            "(Keren, Phys. Rev. B 50, 10039 (1994))"
        ),
        function=_keren_component,
        param_names=["A", "Delta", "nu", "B_L"],
        param_defaults={"A": 25.0, "Delta": 0.5, "nu": 1.0, "B_L": 0.0},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "nu": get_param_info("nu"),
            "B_L": get_param_info("B_L"),
        },
        formula_template="{A}*exp(-Gamma(t; Delta={Delta}, nu={nu}, B_L={B_L}))",
        latex_equation=(
            r"A(t)=A\exp[-\Gamma(t)],\ \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}"
            r"\left[(\omega_0^2+\nu^2)\nu t+(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)"
            r"-2\nu\omega_0 e^{-\nu t}\sin\omega_0 t\right],\ \omega_0=\gamma_\mu B_L"
        ),
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "Abragam": ComponentDefinition(
        name="Abragam",
        description=(
            "Abragam relaxation, Gaussian-to-exponential crossover "
            "(Abragam, Principles of Nuclear Magnetism, 1961)"
        ),
        function=_abragam_component,
        param_names=["A", "Delta", "nu"],
        param_defaults={"A": 25.0, "Delta": 0.5, "nu": 1.0},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "nu": get_param_info("nu"),
        },
        formula_template="{A}*exp(-({Delta}^2/{nu}^2)*(exp(-{nu}*t)-1+{nu}*t))",
        latex_equation=(
            r"A(t)=A\exp\!\left[-\frac{\Delta^2}{\nu^2}\left(e^{-\nu t}-1+\nu t\right)\right]"
        ),
        category="Relaxation",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.DYNAMICS, PhysicsClass.GENERIC_RELAXATION}),
        cost=ComputationalCost.CHEAP,
    ),
    "MuF": ComponentDefinition(
        name="MuF",
        description="Analytical mu-F polarization function D_z(t)",
        function=_muf_component,
        param_names=["A", "r_muF"],
        param_defaults={"A": 25.0, "r_muF": 1.17},
        param_info={"A": get_param_info("A"), "r_muF": get_param_info("r_muF")},
        formula_template="{A}*Dz_muF(t,{r_muF})",
        latex_equation=(
            r"A(t)=A\frac{1}{6}\left[1+2\cos\left(\frac{\omega_d t}{2}\right)+\cos(\omega_d t)+2\cos\left(\frac{3\omega_d t}{2}\right)\right]"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.CHEAP,
    ),
    "FmuF_Linear": ComponentDefinition(
        name="FmuF_Linear",
        description="Analytical collinear F-mu-F polarization function",
        function=_linear_fmuf_component,
        param_names=["A", "r_muF"],
        param_defaults={"A": 25.0, "r_muF": 1.17},
        param_info={"A": get_param_info("A"), "r_muF": get_param_info("r_muF")},
        formula_template="{A}*G_FmuF_linear(t,{r_muF})",
        latex_equation=(r"A(t)=A\,G_{F\mu F}(t)"),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.CHEAP,
    ),
    "DynamicFmuF": ComponentDefinition(
        name="DynamicFmuF",
        description="Strong-collision dynamicized collinear F-mu-F polarization",
        function=_dynamic_fmuf_component,
        param_names=["A", "r_muF", "nu"],
        param_defaults={"A": 25.0, "r_muF": 1.17, "nu": 0.5},
        param_info={
            "A": get_param_info("A"),
            "r_muF": get_param_info("r_muF"),
            "nu": get_param_info("nu"),
        },
        formula_template="{A}*G_FmuF_dyn(t; {r_muF}, {nu})",
        latex_equation=(
            r"A(t)=A\,G_d(t),\ G_d(t)=g(t)+\nu\!\int_0^t\! g(t-\tau)G_d(\tau)d\tau,\ "
            r"g(t)=e^{-\nu t}G_{F\mu F}(t)"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR, PhysicsClass.DYNAMICS}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "FmuF_General": ComponentDefinition(
        name="FmuF_General",
        description="Numerical powder-averaged F-mu-F polarization (r1, r2, theta)",
        function=_general_fmuf_component,
        param_names=["A", "r1", "r2", "theta"],
        param_defaults={"A": 25.0, "r1": 1.17, "r2": 1.17, "theta": 180.0},
        param_info={
            "A": get_param_info("A"),
            "r1": get_param_info("r1"),
            "r2": get_param_info("r2"),
            "theta": get_param_info("theta"),
        },
        formula_template="{A}*Dz_FmuF_general(t,{r1},{r2},{theta})",
        latex_equation=(r"A(t)=A\,D_z^{\mathrm{powder}}\!(t;r_1,r_2,\theta)"),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "FmuF_Triangle": ComponentDefinition(
        name="FmuF_Triangle",
        description="Collinear F-mu-F plus a third fluorine (16-dim powder average)",
        function=_fmuf_triangle_component,
        param_names=["A", "r_muF", "r3", "phi3"],
        param_defaults={"A": 25.0, "r_muF": 1.17, "r3": 2.5, "phi3": 90.0},
        param_info={
            "A": get_param_info("A"),
            "r_muF": get_param_info("r_muF"),
            "r3": get_param_info("r3"),
            "phi3": get_param_info("phi3"),
        },
        formula_template="{A}*Dz_FmuF_F(t; {r_muF}, {r3}, {phi3})",
        latex_equation=(r"A(t)=A\,D_z^{\mathrm{powder}}\!(t;r_{\mu F},r_3,\phi_3)"),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.EXPENSIVE,
    ),
    "DipolarPairField": ComponentDefinition(
        name="DipolarPairField",
        description="Spin-1/2 dipole pair parameterised by the dipolar field B_dip",
        function=_dipolar_pair_field_component,
        param_names=["A", "B_dip", "lambda_T"],
        param_defaults={"A": 25.0, "B_dip": 10.0, "lambda_T": 0.0},
        param_info={
            "A": get_param_info("A"),
            "B_dip": get_param_info("B_dip"),
            "lambda_T": get_param_info("lambda_T"),
        },
        formula_template=(
            "{A}/6*(1 + exp(-{lambda_T}*t)*(2*cos(w*t/2)+cos(w*t)+2*cos(3*w*t/2))),"
            " w=gamma_mu*{B_dip}"
        ),
        latex_equation=(
            r"A(t)=\frac{A}{6}\left[1+e^{-\lambda_T t}\left(2\cos\frac{\omega_d t}{2}"
            r"+\cos\omega_d t+2\cos\frac{3\omega_d t}{2}\right)\right],\ "
            r"\omega_d=\gamma_\mu B_{dip}"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.CHEAP,
    ),
    "ProtonDipole": ComponentDefinition(
        name="ProtonDipole",
        description="Spin-1/2 dipole pair: muon + proton at distance r",
        function=_proton_dipole_component,
        param_names=["A", "r_muH", "lambda_T"],
        param_defaults={"A": 25.0, "r_muH": 1.7, "lambda_T": 0.0},
        param_info={
            "A": get_param_info("A"),
            "r_muH": get_param_info("r_muH"),
            "lambda_T": get_param_info("lambda_T"),
        },
        formula_template="{A}*Dz_pair(t; omega_d({r_muH}), {lambda_T})",
        latex_equation=(
            r"A(t)=\frac{A}{6}\left[1+e^{-\lambda_T t}\left(2\cos\frac{\omega_d t}{2}"
            r"+\cos\omega_d t+2\cos\frac{3\omega_d t}{2}\right)\right],\ "
            r"\hbar\omega_d=\frac{\mu_0\hbar^2\gamma_\mu\gamma_p}{4\pi r^3}"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.CHEAP,
    ),
    "ElectronDipole": ComponentDefinition(
        name="ElectronDipole",
        description="Spin-1/2 dipole pair: muon + localized electron moment at distance r",
        function=_electron_dipole_component,
        param_names=["A", "r_mue", "lambda_T"],
        param_defaults={"A": 25.0, "r_mue": 5.0, "lambda_T": 0.0},
        param_info={
            "A": get_param_info("A"),
            "r_mue": get_param_info("r_mue"),
            "lambda_T": get_param_info("lambda_T"),
        },
        formula_template="{A}*Dz_pair(t; omega_d({r_mue}), {lambda_T})",
        latex_equation=(
            r"A(t)=\frac{A}{6}\left[1+e^{-\lambda_T t}\left(2\cos\frac{\omega_d t}{2}"
            r"+\cos\omega_d t+2\cos\frac{3\omega_d t}{2}\right)\right],\ "
            r"\hbar\omega_d=\frac{\mu_0\hbar^2\gamma_\mu\gamma_e}{4\pi r^3}"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR, PhysicsClass.MAGNETISM}),
        cost=ComputationalCost.CHEAP,
    ),
    "DipolarSpinJ": ComponentDefinition(
        name="DipolarSpinJ",
        description="Muon coupled to one spin-J nucleus with dipolar + quadrupolar terms",
        function=_dipolar_spin_j_component,
        param_names=["A", "f_dip", "f_quad", "J_spin"],
        param_defaults={"A": 25.0, "f_dip": 0.2, "f_quad": 0.0, "J_spin": 1.5},
        param_info={
            "A": get_param_info("A"),
            "f_dip": get_param_info("f_dip"),
            "f_quad": get_param_info("f_quad"),
            "J_spin": get_param_info("J_spin"),
        },
        formula_template="{A}*Dz_spinJ(t; {f_dip}, {f_quad}, {J_spin})",
        fixed_params=("J_spin",),
        latex_equation=(
            r"A(t)=A\,\frac{P_z(t)+2P_x(t)}{3},\ \ \text{Celio-Meier spin-}J"
            r"\text{ eigen-solution}"
        ),
        category="Nuclear dipolar",
        field_geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset({PhysicsClass.MOLECULAR}),
        cost=ComputationalCost.MODERATE,
    ),
    "Constant": ComponentDefinition(
        name="Constant",
        description="Constant background A_bg",
        function=_constant_component,
        param_names=["A_bg"],
        param_defaults={"A_bg": 0.0},
        param_info={"A_bg": get_param_info("A_bg")},
        formula_template="{A_bg}",
        latex_equation=r"A(t) = A_{bg}",
        category="Background",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.BACKGROUND}),
        cost=ComputationalCost.CHEAP,
    ),
    "GaussianPeak": ComponentDefinition(
        name="GaussianPeak",
        description="Gaussian spectral line, parameterised by its full width at half maximum",
        function=_gaussian_peak_component,
        param_names=["height", "nu0", "fwhm"],
        param_defaults={"height": 1.0, "nu0": 1.0, "fwhm": 0.1},
        param_info={
            "height": get_param_info("height"),
            "nu0": get_param_info("nu0"),
            "fwhm": get_param_info("fwhm"),
        },
        formula_template="{height}*exp(-4*ln(2)*((nu-{nu0})/{fwhm})^2)",
        latex_equation=(
            r"S(\nu)=h\exp\left[-4\ln 2\,\frac{(\nu-\nu_0)^2}{w^2}\right],"
            r"\quad w \equiv \mathrm{FWHM}"
        ),
        category="Frequency Domain",
        domain="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.SPECTRAL}),
        cost=ComputationalCost.CHEAP,
    ),
    "LorentzianPeak": ComponentDefinition(
        name="LorentzianPeak",
        description="Lorentzian spectral line, parameterised by its full width at half maximum",
        function=_lorentzian_peak_component,
        param_names=["height", "nu0", "fwhm"],
        param_defaults={"height": 1.0, "nu0": 1.0, "fwhm": 0.1},
        param_info={
            "height": get_param_info("height"),
            "nu0": get_param_info("nu0"),
            "fwhm": get_param_info("fwhm"),
        },
        formula_template="{height}/(1+4*((nu-{nu0})/{fwhm})^2)",
        latex_equation=(r"S(\nu)=\frac{h}{1+4\,(\nu-\nu_0)^2/w^2},\quad w \equiv \mathrm{FWHM}"),
        category="Frequency Domain",
        domain="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.SPECTRAL}),
        cost=ComputationalCost.CHEAP,
    ),
    "ConstantBackground": ComponentDefinition(
        name="ConstantBackground",
        description="Frequency-domain constant background",
        function=_constant_background_component,
        param_names=["bg"],
        param_defaults={"bg": 0.0},
        param_info={"bg": get_param_info("bg")},
        formula_template="{bg}",
        latex_equation=r"S(\nu)=b_g",
        category="Frequency Domain",
        domain="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.SPECTRAL, PhysicsClass.BACKGROUND}),
        cost=ComputationalCost.CHEAP,
    ),
    "LinearBackground": ComponentDefinition(
        name="LinearBackground",
        description="Frequency-domain linear background",
        function=_linear_background_component,
        param_names=["bg", "slope"],
        param_defaults={"bg": 0.0, "slope": 0.0},
        param_info={"bg": get_param_info("bg"), "slope": get_param_info("slope")},
        formula_template="{bg}+{slope}*nu",
        latex_equation=r"S(\nu)=b_g+m\nu",
        category="Frequency Domain",
        domain="frequency",
        field_geometries=frozenset({FieldGeometry.ZF, FieldGeometry.TF, FieldGeometry.LF}),
        physics_classes=frozenset({PhysicsClass.SPECTRAL, PhysicsClass.BACKGROUND}),
        cost=ComputationalCost.CHEAP,
    ),
}


_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
#: Quadrature-sum operator ``f ⊕ g = √(f² + g²)``. It is *not* part of the
#: time-domain composite grammar (only the parameter-vs-x grammar enables it via
#: ``parse_component_expression(..., allowed_operators=...)``). The tokenizer
#: recognises the glyph as a single token so a time-domain expression using it
#: fails the parse cleanly (an "unexpected operator" where an operator is
#: expected, or an unknown-component error where an operand is expected) rather
#: than a confusing character-level tokenise failure.
QUADRATURE_OPERATOR = "⊕"
_UNIT_AMPLITUDE_SENTINEL = "__UNIT_AMPLITUDE__"
_FRACTION_GROUP_DECORATOR = "frac"


def _tokenize_component_expression(expression: str) -> list[str]:
    """Return infix expression tokens for component-name expressions."""
    stripped = expression.strip()
    if not stripped:
        raise ValueError("Expression is required")

    token_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|⊕|[(){}+\-*/]")
    tokens: list[str] = []
    position = 0
    for match in token_pattern.finditer(stripped):
        gap = stripped[position : match.start()]
        if gap.strip():
            raise ValueError(f"Unexpected token near '{gap.strip()}'")
        tokens.append(match.group(0))
        position = match.end()

    trailing = stripped[position:]
    if trailing.strip():
        raise ValueError(f"Unexpected token near '{trailing.strip()}'")
    return tokens


def _parse_group_decorator(tokens: list[str], idx: int) -> tuple[str | None, int]:
    """Return an optional group decorator starting at ``idx``."""
    if idx >= len(tokens) or tokens[idx] != "{":
        return None, idx
    if idx + 2 >= len(tokens) or tokens[idx + 2] != "}":
        raise ValueError("Invalid group decorator")
    decorator = tokens[idx + 1]
    if decorator != _FRACTION_GROUP_DECORATOR:
        raise ValueError(f"Unknown group decorator '{decorator}'")
    return decorator, idx + 3


def parse_component_expression(
    expression: str,
    *,
    allowed_components: set[str] | frozenset[str],
    allowed_operators: set[str] | frozenset[str] = _ALLOWED_OPERATORS,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Parse a component expression into constructor-ready parts.

    ``allowed_operators`` defaults to ``+ - * /``; the parameter-vs-x grammar
    passes an extended set including the quadrature operator ``⊕`` so it is
    accepted there but rejected in the (default) time-domain grammar.
    """
    tokens = _tokenize_component_expression(expression)

    component_names: list[str] = []
    operators: list[str] = []
    open_parentheses: list[int] = []
    close_parentheses: list[int] = []
    pending_open = 0
    expecting_operand = True

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if expecting_operand:
            if token == "(":
                pending_open += 1
                idx += 1
                continue
            if token in allowed_operators or token == ")":
                raise ValueError(f"Expected component before '{token}'")
            if token not in allowed_components:
                raise UnknownComponentError(token, allowed=set(allowed_components))

            component_names.append(token)
            open_parentheses.append(pending_open)
            close_parentheses.append(0)
            pending_open = 0
            expecting_operand = False
            idx += 1
            continue

        if token in allowed_operators:
            operators.append(token)
            expecting_operand = True
            idx += 1
            continue
        if token == ")":
            if not component_names:
                raise ValueError("Closing parenthesis has no matching component")
            close_parentheses[-1] += 1
            idx += 1
            continue
        if token == "(":
            raise ValueError("Expected operator before '('")
        raise ValueError(f"Expected operator before '{token}'")

    if pending_open:
        raise ValueError("Invalid parentheses: unbalanced expression")
    if expecting_operand:
        raise ValueError("Expression cannot end with an operator")

    balance = 0
    for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
        balance += open_count
        balance -= close_count
        if balance < 0:
            raise ValueError("Invalid parentheses: closing before opening")
    if balance != 0:
        raise ValueError("Invalid parentheses: unbalanced expression")

    return component_names, operators, open_parentheses, close_parentheses


def parse_composite_expression(
    expression: str,
) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
    """Parse a composite expression including optional group decorators."""
    tokens = _tokenize_component_expression(expression)

    component_names: list[str] = []
    operators: list[str] = []
    open_parentheses: list[int] = []
    close_parentheses: list[int] = []
    fraction_groups: list[tuple[int, int]] = []
    pending_open = 0
    expecting_operand = True
    paren_component_stack: list[int] = []
    last_closed_group: tuple[int, int] | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if expecting_operand:
            if token == "(":
                pending_open += 1
                idx += 1
                continue
            if token in _ALLOWED_OPERATORS or token in {")", "{", "}"}:
                raise ValueError(f"Expected component before '{token}'")
            if token not in COMPONENTS:
                raise UnknownComponentError(token)

            component_index = len(component_names)
            component_names.append(token)
            open_parentheses.append(pending_open)
            close_parentheses.append(0)
            for _ in range(pending_open):
                paren_component_stack.append(component_index)
            pending_open = 0
            expecting_operand = False
            last_closed_group = None
            idx += 1
            continue

        if token in _ALLOWED_OPERATORS:
            operators.append(token)
            expecting_operand = True
            last_closed_group = None
            idx += 1
            continue
        if token == ")":
            if not component_names:
                raise ValueError("Closing parenthesis has no matching component")
            if not paren_component_stack:
                raise ValueError("Invalid parentheses: closing before opening")
            close_parentheses[-1] += 1
            start_index = paren_component_stack.pop()
            last_closed_group = (start_index, len(component_names) - 1)
            idx += 1
            decorator, idx = _parse_group_decorator(tokens, idx)
            if decorator == _FRACTION_GROUP_DECORATOR:
                fraction_groups.append(last_closed_group)
            continue
        if token == "(":
            raise ValueError("Expected operator before '('")
        if token == "{":
            raise ValueError("Group decorator must follow a closing parenthesis")
        raise ValueError(f"Expected operator before '{token}'")

    if pending_open:
        raise ValueError("Invalid parentheses: unbalanced expression")
    if expecting_operand:
        raise ValueError("Expression cannot end with an operator")

    balance = 0
    for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
        balance += open_count
        balance -= close_count
        if balance < 0:
            raise ValueError("Invalid parentheses: closing before opening")
    if balance != 0:
        raise ValueError("Invalid parentheses: unbalanced expression")

    return component_names, operators, open_parentheses, close_parentheses, fraction_groups


@dataclass(frozen=True)
class _ClosingParenthesis:
    """One closing parenthesis: the component span it delimits, and its decorator."""

    span: tuple[int, int]
    is_fraction_group: bool


def _closing_parentheses(
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_groups: Sequence[tuple[int, int]] = (),
) -> list[list[_ClosingParenthesis]]:
    """Return, per component index, the parentheses closing there, innermost first.

    Two parentheses can delimit the same component span (``((a + b))``), while a
    ``{frac}`` decorator belongs to exactly one pair: it lands on the innermost
    pair with that span — the pair :func:`parse_composite_expression` read it
    from. Everything that has to know *which* parenthesis carries a group (the
    tree builder, the expression writer, the formula and mathtext renderers)
    reads it from here, so they all agree.
    """
    pending = Counter(fraction_groups)
    stack: list[int] = []
    closures: list[list[_ClosingParenthesis]] = []
    for index in range(len(open_parentheses)):
        stack.extend([index] * open_parentheses[index])
        closing_here: list[_ClosingParenthesis] = []
        for _ in range(close_parentheses[index]):
            if not stack:
                raise ValueError("Invalid parentheses: closing before opening")
            span = (stack.pop(), index)
            is_fraction_group = pending[span] > 0
            if is_fraction_group:
                pending[span] -= 1
            closing_here.append(_ClosingParenthesis(span, is_fraction_group))
        closures.append(closing_here)
    if stack:
        raise ValueError("Invalid parentheses: unbalanced expression")
    return closures


def build_component_expression(
    component_names: list[str],
    operators: list[str],
    open_parentheses: list[int] | None = None,
    close_parentheses: list[int] | None = None,
    fraction_groups: list[tuple[int, int]] | None = None,
) -> str:
    """Return a human-editable expression string using component names."""
    if not component_names:
        return ""

    opens = list(open_parentheses or [0] * len(component_names))
    closes = list(close_parentheses or [0] * len(component_names))
    closures = _closing_parentheses(opens, closes, fraction_groups or [])
    parts: list[str] = []
    for idx, name in enumerate(component_names):
        prefix = "(" * opens[idx]
        suffix_parts = [
            ")" + ("{frac}" if closure.is_fraction_group else "") for closure in closures[idx]
        ]
        token = prefix + name + "".join(suffix_parts)
        if idx == 0:
            parts.append(token)
        else:
            parts.append(f"{operators[idx - 1]} {token}")
    return " ".join(parts)


# --- expression tree ---------------------------------------------------------
#
# The serialised form of a composite model is flat: component names, the
# operators between them, and per-component parenthesis counts. Amplitude
# policy *and* evaluation are both defined on the tree that form denotes, so
# parentheses that do not change the tree change neither the parameter names
# nor the value.


@dataclass(frozen=True)
class ExprLeaf:
    """One component, addressed by its index in ``component_names``."""

    index: int


@dataclass(frozen=True)
class ExprProduct:
    """A ``*``/``/`` chain; ``ops[k]`` joins ``factors[k]`` to ``factors[k + 1]``.

    Normalised so that no factor is itself an :class:`ExprProduct`: every
    factor is an :class:`ExprLeaf` or an :class:`ExprSum`.
    """

    factors: tuple[ExpressionNode, ...]
    ops: tuple[str, ...]


@dataclass(frozen=True)
class ExprSum:
    """A ``+``/``-`` chain; ``signs[k]`` is ``+1``/``-1`` for ``terms[k]``.

    ``fraction_group`` carries the ``(start, end)`` component range when the sum
    is a fraction group: its group amplitude scales the weighted sum of its
    terms, and its terms stay aligned one-for-one with the group's additive term
    ranges (so a fraction sum is never flattened into a parent sum).
    """

    terms: tuple[ExpressionNode, ...]
    signs: tuple[int, ...]
    fraction_group: tuple[int, int] | None = None


ExpressionNode = ExprLeaf | ExprProduct | ExprSum

_INVERTED_OPERATOR = {"*": "/", "/": "*"}


def _node_children(node: ExpressionNode) -> tuple[ExpressionNode, ...]:
    if isinstance(node, ExprLeaf):
        return ()
    return node.factors if isinstance(node, ExprProduct) else node.terms


def leaf_indices(node: ExpressionNode) -> tuple[int, ...]:
    """Return the component indices under ``node``, left to right."""
    if isinstance(node, ExprLeaf):
        return (node.index,)
    return tuple(index for child in _node_children(node) for index in leaf_indices(child))


def iter_nodes(node: ExpressionNode) -> Iterator[ExpressionNode]:
    """Yield ``node`` and every node beneath it, parents before children."""
    yield node
    for child in _node_children(node):
        yield from iter_nodes(child)


def _flattened_sum(terms: list[ExpressionNode], signs: list[int]) -> ExprSum:
    """Return a plain sum with plain-sum terms lifted into it.

    ``a - (b + c)`` is ``a - b - c``, so a lifted term's sign multiplies the
    signs it brings with it. Fraction sums own a scale and are never lifted.
    """
    flat_terms: list[ExpressionNode] = []
    flat_signs: list[int] = []
    for sign, term in zip(signs, terms, strict=True):
        if isinstance(term, ExprSum) and term.fraction_group is None:
            for inner_sign, inner_term in zip(term.signs, term.terms, strict=True):
                flat_terms.append(inner_term)
                flat_signs.append(sign * inner_sign)
        else:
            flat_terms.append(term)
            flat_signs.append(sign)
    return ExprSum(tuple(flat_terms), tuple(flat_signs), None)


def _flattened_product(factors: list[ExpressionNode], ops: list[str]) -> ExprProduct:
    """Return a product with sub-products lifted into it.

    ``x / (a * b)`` is ``x / a / b``: dividing *into* a sub-product inverts the
    operators the sub-product brings with it.
    """
    flat_factors: list[ExpressionNode] = []
    flat_ops: list[str] = []
    for order, factor in enumerate(factors):
        join = ops[order - 1] if order else ""
        if isinstance(factor, ExprProduct):
            inverted = join == "/"
            for inner_order, inner_factor in enumerate(factor.factors):
                if inner_order:
                    inner_op = factor.ops[inner_order - 1]
                    flat_ops.append(_INVERTED_OPERATOR[inner_op] if inverted else inner_op)
                elif join:
                    flat_ops.append(join)
                flat_factors.append(inner_factor)
        else:
            if join:
                flat_ops.append(join)
            flat_factors.append(factor)
    return ExprProduct(tuple(flat_factors), tuple(flat_ops))


def build_expression_tree(
    component_count: int,
    operators: Sequence[str],
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_groups: Sequence[tuple[int, int]] = (),
) -> ExpressionNode:
    """Return the expression tree denoted by a composite model's flat form.

    ``*``/``/`` bind tighter than ``+``/``-``; parentheses group. The result is
    normalised so that only the *structure* survives: a group holding a single
    node **is** that node (redundant parentheses vanish), a product factor that
    is itself a product is flattened into its parent, and a plain sum term that
    is itself a plain sum is flattened. A fraction-group sum keeps its own terms.

    The caller has already validated the parenthesis balance, and the fraction
    groups are the ones *requested*: a group is tagged onto the sum of the
    parenthesis whose span it names, and
    :meth:`CompositeModel._validate_fraction_groups` then reads the tagged sums
    back to accept or reject the request.
    """
    # ``(`` and ``)`` tokens both carry the parenthesis they delimit, so an
    # opening parenthesis knows whether it starts a fraction group.
    closures = _closing_parentheses(open_parentheses, close_parentheses, fraction_groups)
    tokens: list[tuple[str, int | str | _ClosingParenthesis | None]] = []
    open_positions: list[int] = []
    for index in range(component_count):
        for _ in range(open_parentheses[index]):
            open_positions.append(len(tokens))
            tokens.append(("(", None))
        tokens.append(("leaf", index))
        for closure in closures[index]:
            tokens[open_positions.pop()] = ("(", closure)
            tokens.append((")", closure))
        if index < component_count - 1:
            tokens.append(("op", operators[index]))

    position = 0

    def at_operator(symbols: set[str]) -> bool:
        return (
            position < len(tokens)
            and tokens[position][0] == "op"
            and tokens[position][1] in symbols
        )

    def parse_group(fraction_group: tuple[int, int] | None) -> ExpressionNode:
        nonlocal position
        terms = [parse_product()]
        signs = [1]
        while at_operator({"+", "-"}):
            signs.append(1 if tokens[position][1] == "+" else -1)
            position += 1
            terms.append(parse_product())
        if fraction_group is not None:
            # A tagged sum keeps its own terms, however many: validation counts
            # them here and rejects a group that does not span two or more.
            return ExprSum(tuple(terms), tuple(signs), fraction_group)
        if len(terms) == 1:
            return terms[0]
        return _flattened_sum(terms, signs)

    def parse_product() -> ExpressionNode:
        nonlocal position
        factors = [parse_atom()]
        ops: list[str] = []
        while at_operator({"*", "/"}):
            ops.append(cast(str, tokens[position][1]))
            position += 1
            factors.append(parse_atom())
        if len(factors) == 1:
            return factors[0]
        return _flattened_product(factors, ops)

    def parse_atom() -> ExpressionNode:
        nonlocal position
        kind, payload = tokens[position]
        position += 1
        if kind == "leaf":
            return ExprLeaf(index=cast(int, payload))
        closure = cast(_ClosingParenthesis, payload)
        node = parse_group(closure.span if closure.is_fraction_group else None)
        position += 1  # the matching ``)``
        return node

    return parse_group(None)


def _fraction_group_nodes(tree: ExpressionNode) -> dict[tuple[int, int], ExprSum]:
    """Return the sum node tagged with each fraction group, keyed by group."""
    return {
        node.fraction_group: node
        for node in iter_nodes(tree)
        if isinstance(node, ExprSum) and node.fraction_group is not None
    }


def _missing_component_function(t: NDArray, **_params: float) -> NDArray[np.float64]:
    """Zero-valued stand-in evaluation for a missing user component."""
    return np.zeros_like(np.asarray(t, dtype=float))


def placeholder_component_definition(name: str) -> ComponentDefinition:
    """Return a named placeholder for an unregistered (user) component.

    Used when a project references a component that is not registered in this
    session (typically a user function whose plugin is not installed): the
    model still opens with its original expression — the placeholder evaluates
    to zero and is flagged ``missing`` so fitting can be blocked with a clear
    message instead of the model being silently dropped. Placeholders are
    per-instance and are **never** inserted into ``COMPONENTS``.
    """
    return ComponentDefinition(
        name=name,
        description=f"Missing user function '{name}' (not registered in this session)",
        function=_missing_component_function,
        param_names=[],
        param_defaults={},
        param_info={},
        formula_template="0",
        latex_equation="",
        category="User",
        domain="time",
        user=True,
        missing=True,
    )


class CompositeModel:
    """A flat composite model built from baseline-free components.

    ``allow_missing`` lets a model materialise even when some component names
    are not registered (see :func:`placeholder_component_definition`); callers
    that fit or edit the model must check :attr:`missing_component_names`.
    """

    def __init__(
        self,
        component_names: list[str],
        operators: list[str] | None = None,
        open_parentheses: list[int] | None = None,
        close_parentheses: list[int] | None = None,
        fraction_groups: list[tuple[int, int]] | None = None,
        *,
        allow_missing: bool = False,
    ) -> None:
        if not component_names:
            raise ValueError("Composite model must contain at least one component")

        missing = [name for name in component_names if name not in COMPONENTS]
        if missing and not allow_missing:
            raise ValueError(f"Unknown component(s): {missing}")

        if operators is None:
            operators = ["+"] * (len(component_names) - 1)

        if len(operators) != max(len(component_names) - 1, 0):
            raise ValueError("operators length must be len(component_names) - 1")
        if any(op not in _ALLOWED_OPERATORS for op in operators):
            raise ValueError("operators must be one of '+', '-', '*', '/'")

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
        # The expression tree is the only source of structure, so it is built
        # first — from the groups as *requested* — and the fraction groups are
        # then validated against the sum nodes they tag.
        requested_fraction_groups = self._checked_fraction_group_spans(fraction_groups or [])
        self._tree = build_expression_tree(
            len(self.component_names),
            self.operators,
            self.open_parentheses,
            self.close_parentheses,
            requested_fraction_groups,
        )
        self._fraction_group_nodes = _fraction_group_nodes(self._tree)
        self.fraction_groups = self._validate_fraction_groups(requested_fraction_groups)
        self._fraction_term_number_by_component = self._build_fraction_term_number_map()
        self._fraction_group_by_component = self._build_fraction_group_component_map()
        self.missing_component_names: tuple[str, ...] = tuple(missing)
        self.components = [
            COMPONENTS[name] if name in COMPONENTS else placeholder_component_definition(name)
            for name in component_names
        ]
        self._suppress_component_amplitude = self._suppressed_scaling_parameters()
        self._param_mappings = self._build_param_mapping()
        # Component-based fraction names depend on the (already-built) non-fraction
        # parameter mapping so a candidate f_<Component> can dodge collisions.
        (
            self._fraction_param_name_by_component,
            self._derived_fraction_name_by_group,
        ) = self._build_fraction_naming()

        param_names: list[str] = []
        defaults: dict[str, float] = {}
        param_info: dict[str, ParamInfo] = {}
        for idx, (mapping, component) in enumerate(
            zip(self._param_mappings, self.components, strict=True)
        ):
            group = self._fraction_group_by_component.get(idx)
            if group is not None and group[0] == idx:
                amplitude_name = self._fraction_group_amplitude_name(group)
                param_names.append(amplitude_name)
                defaults[amplitude_name] = self._fraction_group_default_amplitude(group)
                param_info[amplitude_name] = get_param_info(amplitude_name)

            for pname in component.param_names:
                unique_name = mapping[pname]
                if unique_name == _UNIT_AMPLITUDE_SENTINEL:
                    continue
                if unique_name not in defaults:
                    param_names.append(unique_name)
                    defaults[unique_name] = component.param_defaults[pname]
                    param_info[unique_name] = get_param_info(unique_name)

            # Only the first n-1 additive terms of a group carry a free fraction
            # parameter; the last term's weight is 1 - Σ (the remainder), so it
            # has no parameter. Each free fraction defaults to 1/n.
            if idx in self._fraction_param_name_by_component:
                fraction_name = self._fraction_param_name_by_component[idx]
                term_starts = self._fraction_group_term_starts(group)
                param_names.append(fraction_name)
                defaults[fraction_name] = 1.0 / float(len(term_starts))
                param_info[fraction_name] = get_param_info(fraction_name)
        self.param_names = param_names
        self.param_defaults = defaults
        self.param_info = param_info

    @classmethod
    def from_expression(cls, expression: str) -> CompositeModel:
        """Construct a CompositeModel from a component-name expression."""
        component_names, operators, open_parentheses, close_parentheses, fraction_groups = (
            parse_composite_expression(expression)
        )
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=fraction_groups,
        )

    def expression_tree(self) -> ExpressionNode:
        """Return the model's expression tree (see :func:`build_expression_tree`).

        The nodes are frozen dataclasses addressing components by index, so a
        caller can walk the products and sums the expression denotes.
        """
        return self._tree

    def parameter_mapping(self) -> list[dict[str, str]]:
        """Return per-component maps of local parameter name → unique fit name.

        One dict per entry of :attr:`components`, in the same order.  Copies
        are returned so callers (e.g. the RRF frequency-offset wrapper) cannot
        mutate the model's internal mapping.
        """
        return [dict(mapping) for mapping in self._param_mappings]

    def knight_observable_params(self) -> dict[str, str]:
        """Map fitted parameter name → kind for Knight-convertible components.

        For each component whose value is a *local* precession field/frequency
        (``ComponentDefinition.knight_observable`` set — Oscillatory/Bessel give
        ``"frequency"``, OscillatoryField gives ``"field"``), return its unique
        fitted parameter name pointing at that kind. Components whose ``field``
        is the *applied* field (muonium TF) are excluded, so a Knight-shift
        conversion never mistakes the applied field for a local one.
        """
        observables: dict[str, str] = {}
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            local = component.knight_observable
            if local and local in mapping:
                observables[mapping[local]] = local
        return observables

    def component_expression_string(self) -> str:
        """Return the builder-facing expression using component names."""
        return build_component_expression(
            self.component_names,
            self.operators,
            self.open_parentheses,
            self.close_parentheses,
            self.fraction_groups,
        )

    def _checked_fraction_group_spans(
        self,
        fraction_groups: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Return the requested group spans, in order, after shape checks.

        Only the checks the expression tree cannot make: a span must be a pair
        of in-range integers and must be asked for once. Whether it *is* a
        group is decided on the tree (:meth:`_validate_fraction_groups`).
        """
        spans: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for group in fraction_groups:
            if not isinstance(group, tuple) or len(group) != 2:
                raise ValueError("fraction_groups must contain (start, end) pairs")
            start, end = group
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("fraction_groups indices must be integers")
            if start < 0 or end >= len(self.component_names) or start >= end:
                raise ValueError("Invalid fraction group range")
            if group in seen:
                raise ValueError("Duplicate fraction group")
            seen.add(group)
            spans.append(group)
        return spans

    def _validate_fraction_groups(
        self,
        fraction_groups: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Return the accepted groups (sorted), validated against the tree.

        Each requested span must have tagged a parenthesised sum; that sum must
        hold two or more terms, all joined by ``+``; and no two groups may claim
        the same component (which is also what rejects nested groups).
        """
        validated: list[tuple[int, int]] = []
        occupied_components: set[int] = set()
        for group in fraction_groups:
            node = self._fraction_group_nodes.get(group)
            if node is None:
                raise ValueError("Fraction groups must map to one parenthesized expression")
            if len(node.terms) < 2:
                raise ValueError("Fraction groups require at least two additive terms")
            if any(sign != 1 for sign in node.signs):
                raise ValueError("Fraction groups only support additive '+' terms")
            for idx in range(group[0], group[1] + 1):
                if idx in occupied_components:
                    raise ValueError("Fraction groups cannot overlap")
                occupied_components.add(idx)
            validated.append(group)
        validated.sort()
        return validated

    def _parenthesis_closures(self) -> list[list[_ClosingParenthesis]]:
        """Return this model's closing parentheses (see :func:`_closing_parentheses`)."""
        return _closing_parentheses(
            self.open_parentheses, self.close_parentheses, self.fraction_groups
        )

    def _build_fraction_group_component_map(self) -> dict[int, tuple[int, int]]:
        mapping: dict[int, tuple[int, int]] = {}
        for group in self.fraction_groups:
            start, end = group
            for idx in range(start, end + 1):
                mapping[idx] = group
        return mapping

    def _non_fraction_param_names(self) -> set[str]:
        """Return every non-fraction unique parameter name the model exposes.

        Group amplitudes (``A_{start+1}``) plus each component's mapped
        parameter name (dropping the unit-amplitude sentinel). Fraction names
        are disambiguated against this set so a synthesized ``f_<Component>``
        never shadows a real parameter.
        """
        names: set[str] = set()
        for group in self.fraction_groups:
            names.add(self._fraction_group_amplitude_name(group))
        for mapping in self._param_mappings:
            for unique_name in mapping.values():
                if unique_name != _UNIT_AMPLITUDE_SENTINEL:
                    names.add(unique_name)
        return names

    def _build_fraction_naming(
        self,
    ) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
        """Assign component-based fraction names for free terms and remainders.

        For a group with additive term starts ``t_1..t_n``:
        - ``t_1..t_{n-1}`` each get a free parameter ``f_<Component>`` (the
          component at the term start), disambiguated with ``_2``, ``_3``, …
          when a base name would repeat across all fraction parameters or
          collide with any other model parameter name.
        - ``t_n`` gets no parameter; its remainder weight is displayed under a
          synthesized ``f_<Component>`` name guaranteed not to collide with any
          real parameter name (free fractions or otherwise).
        """
        if not self.fraction_groups:
            return {}, {}

        reserved = self._non_fraction_param_names()
        used = set(reserved)
        free_names: dict[int, str] = {}

        def disambiguate(base: str) -> str:
            if base not in used:
                used.add(base)
                return base
            suffix = 2
            while f"{base}_{suffix}" in used:
                suffix += 1
            candidate = f"{base}_{suffix}"
            used.add(candidate)
            return candidate

        for group in self.fraction_groups:
            term_starts = self._fraction_group_term_starts(group)
            for idx in term_starts[:-1]:
                base = f"f_{self.component_names[idx]}"
                free_names[idx] = disambiguate(base)

        # Remainder display names are chosen after every free name is fixed, so
        # they only need to avoid the real parameter set (reserved + free) — which
        # is exactly what ``used`` already tracks, so the same disambiguator applies.
        derived_names: dict[tuple[int, int], str] = {}
        for group in self.fraction_groups:
            last_start = self._fraction_group_term_starts(group)[-1]
            derived_names[group] = disambiguate(f"f_{self.component_names[last_start]}")

        return free_names, derived_names

    def _build_fraction_term_number_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        next_number = 1
        for group in self.fraction_groups:
            for term_start, term_end in self._fraction_group_term_ranges(group):
                for idx in range(term_start, term_end + 1):
                    mapping[idx] = next_number
                next_number += 1
        return mapping

    def _fraction_group_term_ranges(self, group: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the additive term ranges represented by one fraction group.

        Read straight off the sum node the group tags: its terms *are* the
        group's additive terms, whatever else is parenthesised around or inside
        them.
        """
        ranges: list[tuple[int, int]] = []
        for term in self._fraction_group_nodes[group].terms:
            indices = leaf_indices(term)
            ranges.append((min(indices), max(indices)))
        return ranges

    def _fraction_group_term_starts(self, group: tuple[int, int]) -> list[int]:
        """Return component indices where each weighted fraction term starts."""
        return [start for start, _end in self._fraction_group_term_ranges(group)]

    def _fraction_group_amplitude_name(self, group: tuple[int, int]) -> str:
        return f"A_{group[0] + 1}"

    def _fraction_param_name(self, component_index: int) -> str | None:
        """Return the free fraction parameter name for a term start, if any.

        The last additive term of a group is the derived remainder and has no
        free parameter, so this returns ``None`` for it.
        """
        return self._fraction_param_name_by_component.get(component_index)

    def _fraction_group_default_amplitude(self, group: tuple[int, int]) -> float:
        start, end = group
        for idx in range(start, end + 1):
            component = self.components[idx]
            for pname in component.param_names:
                if self._is_scaling_parameter(pname):
                    return float(component.param_defaults[pname])
        return 1.0

    def _fraction_group_weights(
        self,
        group: tuple[int, int],
        kwargs: dict[str, float],
    ) -> dict[int, float]:
        """Return ``{term_start: weight}`` for one group under the n-1 scheme.

        Each of the n-1 free terms weighs ``clamp(value, 0, 1)``; the remainder
        term weighs ``clamp(1 - Σ free, 0, 1)``. No sum-normalization: with the
        free values already in [0, 1] the weights are the physical partition
        (they sum to 1 whenever Σ free ≤ 1, and the remainder floors at 0 once
        the free weights over-subscribe the group).
        """
        component_indices = self._fraction_group_term_starts(group)
        weights: dict[int, float] = {}
        free_total = 0.0
        for idx in component_indices[:-1]:
            name = self._fraction_param_name(idx)
            if name is None or name not in kwargs:
                raise KeyError(f"Missing composite parameter '{name}'")
            weight = min(max(float(kwargs[name]), 0.0), 1.0)
            weights[idx] = weight
            free_total += weight
        weights[component_indices[-1]] = min(max(1.0 - free_total, 0.0), 1.0)
        return weights

    def fraction_weights(self, values: dict[str, float]) -> dict[str, float]:
        """Return ``{name: weight}`` for every fraction term across all groups.

        Both the n-1 free parameters (keyed by their real names, carrying the
        clamped [0, 1] value) and the derived remainder of each group (keyed by
        its synthesized display name, carrying ``1 - Σ free``) appear. A group
        is **skipped entirely** when any of its free parameters is missing from
        ``values``, so callers never receive a partial partition.
        """
        out: dict[str, float] = {}
        for group in self.fraction_groups:
            free_starts = self._fraction_group_term_starts(group)[:-1]
            names = [self._fraction_param_name(idx) for idx in free_starts]
            if not all(name in values for name in names):
                continue
            weights = self._fraction_group_weights(group, values)
            last_start = self._fraction_group_term_starts(group)[-1]
            for idx, weight in weights.items():
                if idx == last_start:
                    out[self._derived_fraction_name_by_group[group]] = weight
                else:
                    out[self._fraction_param_name(idx)] = weight
        return out

    def normalized_parameter_values(self, values: dict[str, float]) -> dict[str, float]:
        """Return a copy with free fraction parameters clamped into [0, 1].

        Under the n-1 scheme the fitted values are already the physical free
        weights, so this only clamps each free fraction into its [0, 1] range
        (leaving every other entry untouched). Kept because callers rely on it
        to present display-ready values.
        """
        normalized = dict(values)
        for idx, name in self._fraction_param_name_by_component.items():
            if name in normalized:
                normalized[name] = min(max(float(normalized[name]), 0.0), 1.0)
        return normalized

    def fraction_parameter_groups(self) -> list[list[str]]:
        """Return the free fraction-parameter names grouped by fraction group.

        Each group contributes its n-1 free parameter names; the remainder term
        has no parameter and is not listed here (see :meth:`derived_fraction_names`
        for its display label).
        """
        groups: list[list[str]] = []
        for group in self.fraction_groups:
            free_starts = self._fraction_group_term_starts(group)[:-1]
            groups.append([self._fraction_param_name(idx) for idx in free_starts])
        return groups

    def derived_fraction_names(self) -> list[str]:
        """Return one synthesized display name per group for its remainder term.

        These are the ``f_<Component>`` labels of each group's last additive
        term — display-only names (no fitted parameter) that never collide with
        a real parameter name. Ordered to match :attr:`fraction_groups`.
        """
        return [self._derived_fraction_name_by_group[group] for group in self.fraction_groups]

    def derived_fraction_terms(self) -> list[tuple[str, tuple[int, int]]]:
        """Return ``(display_name, group)`` for each group's remainder term.

        Pairs each derived remainder name with the fraction group it labels, so
        a caller can place the remainder row and know which group it belongs to.
        """
        return [
            (self._derived_fraction_name_by_group[group], group) for group in self.fraction_groups
        ]

    def with_default_fraction_groups(self) -> CompositeModel:
        """Return a copy with a top-level additive fraction group when suitable."""
        if self.fraction_groups or len(self.component_names) < 2:
            return self

        # Only a model that already *is* a sum of two or more ``+`` terms can
        # become one fraction group; anything else (a product, a single leaf, a
        # sum with a ``-``) is returned untouched.
        root = self._tree
        if not isinstance(root, ExprSum) or len(root.terms) < 2:
            return self
        if any(sign != 1 for sign in root.signs):
            return self

        whole_span = (0, len(self.component_names) - 1)
        open_parentheses = list(self.open_parentheses)
        close_parentheses = list(self.close_parentheses)
        if not any(closure.span == whole_span for closure in self._parenthesis_closures()[-1]):
            open_parentheses[0] += 1
            close_parentheses[-1] += 1

        return CompositeModel(
            component_names=list(self.component_names),
            operators=list(self.operators),
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=[whole_span],
            # Rebuilding from an existing instance: its names were already
            # vetted (possibly as placeholders), so never re-raise here.
            allow_missing=True,
        )

    def domains(self) -> set[str]:
        """Return the set of analysis domains of the model's components.

        A well-formed model has a single domain (``{"time"}`` or
        ``{"frequency"}``); a mixed set indicates a model that combines
        time- and frequency-domain components (e.g. restored from a project
        saved before domain filtering existed) and should be surfaced to the
        user rather than silently fitted.

        Missing-component placeholders are skipped: their domain is unknowable,
        and the missing-ness itself is surfaced separately (fit blocking via
        :attr:`missing_component_names`).
        """
        return {component.domain for component in self.components if not component.missing}

    def fixed_by_default_params(self) -> set[str]:
        """Unique parameter names that should start fixed in a fit.

        Collected from each component's :attr:`ComponentDefinition.fixed_params`
        through the model's parameter mapping (so duplicated components yield
        their indexed names, e.g. ``J_spin_2``).
        """
        fixed: set[str] = set()
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            for pname in component.fixed_params:
                unique = mapping.get(pname)
                if unique and unique != _UNIT_AMPLITUDE_SENTINEL:
                    fixed.add(unique)
        return fixed

    def _build_param_mapping(self) -> list[dict[str, str]]:
        """Return per-component maps of local parameter name → unique fit name.

        A suppressed scaling parameter maps to the unit-amplitude sentinel and
        is not exposed at all, so it is also left out of the collision counts
        that drive the ``_2``/``_3`` suffixes: adding a factor whose scale the
        policy suppresses never renames a parameter the user already has.
        ``A`` is always indexed by its own component (``A_1``, ``A_3``, …).
        """
        name_counts = Counter(
            pname
            for idx, component in enumerate(self.components)
            for pname in component.param_names
            if not self._is_suppressed_scaling_parameter(idx, pname)
        )
        mappings: list[dict[str, str]] = []
        used_names: set[str] = set()

        for idx, component in enumerate(self.components, start=1):
            mapping: dict[str, str] = {}
            for pname in component.param_names:
                if self._is_suppressed_scaling_parameter(idx - 1, pname):
                    mapping[pname] = _UNIT_AMPLITUDE_SENTINEL
                    continue
                if pname == "A":
                    mapping[pname] = f"A_{idx}"
                elif name_counts[pname] > 1:
                    term_number = self._fraction_term_number_by_component.get(idx - 1)
                    if term_number is not None:
                        candidate = f"{pname}_{term_number}"
                        mapping[pname] = (
                            candidate if candidate not in used_names else f"{pname}_{idx}"
                        )
                    else:
                        mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
                used_names.add(mapping[pname])
            mappings.append(mapping)
        return mappings

    def _is_suppressed_scaling_parameter(self, component_index: int, pname: str) -> bool:
        return (
            self._is_scaling_parameter(pname)
            and self._suppress_component_amplitude[component_index]
        )

    def _suppressed_scaling_parameters(self) -> list[bool]:
        """Return, per component, whether its scaling parameter is unit-valued.

        Every product carries exactly one scale. A product with a sum factor
        (plain or fraction group) takes its scale from that sum's terms, so all
        of its leaf factors are suppressed; otherwise the first leaf factor that
        declares a scale keeps it and every later one is suppressed. Inside a
        fraction group the group amplitude carries the scale, so every component
        of the group is suppressed.
        """
        suppress = [False] * len(self.components)
        for index in self._fraction_group_by_component:
            suppress[index] = True
        for node in iter_nodes(self._tree):
            if not isinstance(node, ExprProduct):
                continue
            scale_leaves = [
                factor.index
                for factor in node.factors
                if isinstance(factor, ExprLeaf)
                and self._component_has_scaling_parameter(factor.index)
            ]
            survivors = 0 if any(isinstance(f, ExprSum) for f in node.factors) else 1
            for index in scale_leaves[survivors:]:
                suppress[index] = True
        return suppress

    def _is_scaling_parameter(self, pname: str) -> bool:
        """Return True for parameters that act as component scale factors."""
        return pname in {"A", "A_bg"}

    def _component_has_scaling_parameter(self, idx: int) -> bool:
        return any(self._is_scaling_parameter(pname) for pname in self.components[idx].param_names)

    def function(self, t: NDArray, **kwargs: float) -> NDArray[np.float64]:
        """Evaluate the composite function with standard arithmetic precedence."""
        return self._evaluate_node(self._tree, np.asarray(t, dtype=float), kwargs)

    def _evaluate_node(
        self,
        node: ExpressionNode,
        t_arr: NDArray[np.float64],
        kwargs: dict[str, float],
    ) -> NDArray[np.float64]:
        """Evaluate one expression-tree node (see :func:`build_expression_tree`)."""
        if isinstance(node, ExprLeaf):
            component = self.components[node.index]
            component_kwargs = self._extract_component_kwargs(
                component, self._param_mappings[node.index], kwargs
            )
            return np.asarray(component.function(t_arr, **component_kwargs), dtype=float)

        if isinstance(node, ExprProduct):
            result = self._evaluate_node(node.factors[0], t_arr, kwargs)
            for op, factor in zip(node.ops, node.factors[1:], strict=True):
                rhs = self._evaluate_node(factor, t_arr, kwargs)
                if op == "*":
                    result = result * rhs
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        divided = np.full_like(result, 1e30, dtype=float)
                        np.divide(result, rhs, out=divided, where=np.abs(rhs) > 1e-30)
                    result = divided
            return result

        if node.fraction_group is None:
            result = self._evaluate_node(node.terms[0], t_arr, kwargs)
            for sign, term in zip(node.signs[1:], node.terms[1:], strict=True):
                value = self._evaluate_node(term, t_arr, kwargs)
                result = result + value if sign > 0 else result - value
            return result

        amplitude_name = self._fraction_group_amplitude_name(node.fraction_group)
        if amplitude_name not in kwargs:
            raise KeyError(f"Missing composite parameter '{amplitude_name}'")
        weights = self._fraction_group_weights(node.fraction_group, kwargs)
        weighted = np.zeros_like(t_arr)
        for term in node.terms:
            weight = weights[leaf_indices(term)[0]]
            weighted = weighted + weight * self._evaluate_node(term, t_arr, kwargs)
        return float(kwargs[amplitude_name]) * weighted

    def _extract_component_kwargs(
        self,
        component: ComponentDefinition,
        mapping: dict[str, str],
        kwargs: dict[str, float],
    ) -> dict[str, float]:
        component_kwargs: dict[str, float] = {}
        for pname in component.param_names:
            unique_name = mapping[pname]
            if unique_name == _UNIT_AMPLITUDE_SENTINEL:
                component_kwargs[pname] = 1.0
                continue
            if unique_name not in kwargs:
                raise KeyError(f"Missing composite parameter '{unique_name}'")
            component_kwargs[pname] = float(kwargs[unique_name])
        return component_kwargs

    def additive_component_indices(self) -> list[int]:
        """Return component indices that contribute in additive (+) form.

        This includes the first component and any component joined with a
        ``+`` operator. Components joined with ``-``, ``*``, or ``/`` are
        excluded because their visual contribution is not an additive area.
        """
        if not self.components:
            return []

        indices = [0]
        for idx, op in enumerate(self.operators, start=1):
            if op == "+":
                indices.append(idx)
        return indices

    def evaluate_components(
        self,
        t: NDArray,
        *,
        additive_only: bool = False,
        **kwargs: float,
    ) -> list[tuple[str, NDArray[np.float64]]]:
        """Evaluate individual component curves.

        Parameters
        ----------
        t : array-like
            Time points where components are evaluated.
        additive_only : bool, optional
            If True, only return additive components (first component and
            components joined with ``+`` operators).
        **kwargs : float
            Composite-model parameters using unique parameter names.
        """
        t_arr = np.asarray(t, dtype=float)
        curves: list[tuple[str, NDArray[np.float64]]] = []

        if additive_only:
            include = set(self.additive_component_indices())
        else:
            include = set(range(len(self.components)))

        fraction_weights: dict[int, float] = {}
        for group in self.fraction_groups:
            fraction_weights.update(self._fraction_group_weights(group, kwargs))

        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            if idx not in include:
                continue
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            y_vals = np.asarray(component.function(t_arr, **component_kwargs), dtype=float)
            group = self._fraction_group_by_component.get(idx)
            if group is not None:
                weight = fraction_weights[idx] if idx in fraction_weights else 1.0
                y_vals = float(kwargs[self._fraction_group_amplitude_name(group)]) * weight * y_vals
            curves.append((self.component_names[idx], y_vals))
        return curves

    def formula_string(self) -> str:
        """Return a symbolic formula preview string."""
        if self.fraction_groups:
            return self._formula_string_with_fraction_groups()

        parts: list[str] = []
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True), start=1
        ):
            term = self._component_formula_term(component, mapping)

            if self.open_parentheses[idx - 1] > 0:
                term = "(" * self.open_parentheses[idx - 1] + term
            if self.close_parentheses[idx - 1] > 0:
                term = term + ")" * self.close_parentheses[idx - 1]
            parts.append(term)

        if not parts:
            return ""
        expression = parts[0]
        for op, term in zip(self.operators, parts[1:], strict=True):
            if op == "*" and term == "1":
                continue
            if op == "/" and term == "1":
                continue
            if op == "*" and expression == "1":
                expression = term
                continue
            expression = f"{expression} {op} {term}"
        return expression

    def _component_formula_term(
        self,
        component: ComponentDefinition,
        mapping: dict[str, str],
    ) -> str:
        fmt_values = {
            pname: ("1" if mapping[pname] == _UNIT_AMPLITUDE_SENTINEL else mapping[pname])
            for pname in component.param_names
        }
        term = component.formula_template.format(**fmt_values)
        if fmt_values.get("A") == "1" and term.startswith("1*"):
            term = term[2:]
        return term

    def _formula_string_with_fraction_groups(self) -> str:
        # Weight prefix for each additive term start: free terms use their real
        # fraction parameter; the remainder term renders its derived weight
        # explicitly as (1-f_X-f_Y) over the group's free parameter names.
        weight_prefix_by_start: dict[int, str] = {}
        for group in self.fraction_groups:
            term_starts = self._fraction_group_term_starts(group)
            for idx in term_starts[:-1]:
                weight_prefix_by_start[idx] = self._fraction_param_name(idx)
            free_names = [self._fraction_param_name(idx) for idx in term_starts[:-1]]
            weight_prefix_by_start[term_starts[-1]] = (
                "(1" + "".join(f"-{name}" for name in free_names) + ")"
            )

        terms: list[str] = []
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            term = self._component_formula_term(component, mapping)
            if idx in weight_prefix_by_start:
                prefix = weight_prefix_by_start[idx]
                term = prefix if term == "1" else f"{prefix}*{term}"
            terms.append(term)

        value_stack: list[str] = []
        op_stack: list[str] = []
        closures = self._parenthesis_closures()

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_top_operator() -> None:
            op = op_stack.pop()
            rhs = value_stack.pop()
            lhs = value_stack.pop()
            value_stack.append(f"({lhs} {op} {rhs})")

        for idx, term in enumerate(terms):
            for _ in range(self.open_parentheses[idx]):
                op_stack.append("(")

            value_stack.append(term)

            for closure in closures[idx]:
                while op_stack[-1] != "(":
                    apply_top_operator()
                op_stack.pop()
                if closure.is_fraction_group:
                    amplitude_name = self._fraction_group_amplitude_name(closure.span)
                    grouped_term = value_stack.pop()
                    value_stack.append(f"{amplitude_name}*({grouped_term})")

            if idx < len(self.operators):
                op = self.operators[idx]
                while (
                    op_stack and op_stack[-1] != "(" and precedence(op_stack[-1]) >= precedence(op)
                ):
                    apply_top_operator()
                op_stack.append(op)

        while op_stack:
            apply_top_operator()

        expression = value_stack[0]
        if expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1]
        return expression

    # --- typeset (mathtext) preview -----------------------------------------

    def _top_level_terms(self) -> list[tuple[int, int, str]]:
        """Return ``(start, end, separator)`` for each top-level additive term.

        Read off the root of the expression tree. A plain sum contributes one
        entry per term, carrying the operator that *joins* it to the running
        expression (``""`` for the first, ``" + "``/``" - "`` for the rest); any
        other root — a product, a single component, or a fraction group whose
        weights render inside it — is one term spanning the whole model.
        """
        root = self._tree
        if isinstance(root, ExprSum) and root.fraction_group is None:
            terms: list[tuple[int, int, str]] = []
            for order, (term, sign) in enumerate(zip(root.terms, root.signs, strict=True)):
                indices = leaf_indices(term)
                separator = "" if order == 0 else (" + " if sign == 1 else " - ")
                terms.append((min(indices), max(indices), separator))
            return terms
        return [(0, len(self.component_names) - 1, "")]

    def _latex_component_body(
        self,
        component: ComponentDefinition,
        mapping: dict[str, str],
    ) -> str:
        """Return the mathtext body for one component, matching formula_string.

        Mirrors :meth:`_component_formula_term`: the same amplitude symbol is
        emitted, a suppressed (``__UNIT_AMPLITUDE__``) amplitude collapses to
        ``1`` and drops a leading ``1\\,`` factor, and the same parameter
        symbols appear — only rendered as mathtext. When the template cannot be
        transformed confidently, a ``\\mathrm{Name}(t; ...)`` fallback is
        returned instead (the designed output for gnarly components).
        """
        # Resolve, per local param, whether it renders as the literal ``1`` (a
        # suppressed amplitude) or as a mathtext symbol.
        symbol_for_param: dict[str, str] = {}
        is_unit: dict[str, bool] = {}
        for pname in component.param_names:
            unique = mapping[pname]
            if unique == _UNIT_AMPLITUDE_SENTINEL:
                is_unit[pname] = True
                symbol_for_param[pname] = "1"
                continue
            is_unit[pname] = False
            symbol_for_param[pname] = param_symbol_latex(get_param_info(unique).latex, unique)

        # Parameter symbols that actually surface (unit amplitudes excluded),
        # in template order — used for the function-name fallback.
        surfaced_symbols = [
            symbol_for_param[pname] for pname in component.param_names if not is_unit[pname]
        ]

        transformed = self._transform_component_template(component, symbol_for_param, is_unit)
        if transformed is None:
            return fallback_function_latex(component.name, surfaced_symbols)
        return transformed

    def _transform_component_template(
        self,
        component: ComponentDefinition,
        symbol_for_param: dict[str, str],
        is_unit: dict[str, bool],
    ) -> str | None:
        """Substitute sentinels into the template and transform to mathtext.

        Returns ``None`` when the template is outside the transformable subset.
        Unit-amplitude params are substituted with the literal ``1`` (matching
        ``formula_string``); every other param becomes an opaque sentinel that
        the transformer resolves back to its mathtext symbol.
        """
        template = component.formula_template
        symbols: dict[str, str] = {}
        fmt_values: dict[str, str] = {}
        for order, pname in enumerate(component.param_names):
            if is_unit.get(pname, False):
                fmt_values[pname] = "1"
                continue
            sentinel = f"\x00{order}\x00"
            symbols[sentinel] = symbol_for_param[pname]
            fmt_values[pname] = sentinel
        try:
            substituted = template.format(**fmt_values)
        except (KeyError, IndexError, ValueError):
            return None
        # Drop a leading ``1*`` factor exactly as _component_formula_term does,
        # so a shared/suppressed amplitude never leaves a dangling 1.
        if fmt_values.get("A") == "1" and substituted.startswith("1*"):
            substituted = substituted[2:]
        return transform_template(substituted, symbols)

    def _latex_span_fragment(self, start: int, end: int) -> str:
        """Render the component span ``[start, end]`` as one mathtext fragment.

        Reuses the same shunting-yard as
        :meth:`_formula_string_with_fraction_groups` — including the fraction
        weight prefixes and per-group amplitude factor — but emits mathtext.
        The span is a single top-level additive term, so a parenthesis it opens
        may still close outside it (``(a + b)`` splits into two terms). Only the
        pairs that open *and* close within the span are walked; a fraction group
        is always one of them, since a group sum lies inside a single top-level
        term.
        """
        weight_prefix_by_start = self._latex_weight_prefixes()
        closures = [
            [closure for closure in row if closure.span[0] >= start]
            for row in self._parenthesis_closures()[start : end + 1]
        ]
        opens_within_span = Counter(closure.span[0] for row in closures for closure in row)

        value_stack: list[str] = []
        op_stack: list[str] = []

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_top_operator() -> None:
            op = op_stack.pop()
            rhs = value_stack.pop()
            lhs = value_stack.pop()
            if op == "*":
                value_stack.append(f"({lhs}\\,{rhs})")
            elif op == "/":
                value_stack.append(f"({lhs}/{rhs})")
            else:
                value_stack.append(f"({lhs} {op} {rhs})")

        for idx in range(start, end + 1):
            component = self.components[idx]
            mapping = self._param_mappings[idx]
            body = self._latex_component_body(component, mapping)
            if idx in weight_prefix_by_start:
                prefix = weight_prefix_by_start[idx]
                body = prefix if body == "1" else f"{prefix}\\,{body}"

            for _ in range(opens_within_span[idx]):
                op_stack.append("(")

            value_stack.append(body)

            for closure in closures[idx - start]:
                while op_stack[-1] != "(":
                    apply_top_operator()
                op_stack.pop()
                if closure.is_fraction_group:
                    amplitude_name = self._fraction_group_amplitude_name(closure.span)
                    amp = param_symbol_latex(get_param_info(amplitude_name).latex, amplitude_name)
                    grouped = value_stack.pop()
                    value_stack.append(f"{amp}\\,({grouped})")

            if idx < end:
                op = self.operators[idx]
                while (
                    op_stack and op_stack[-1] != "(" and precedence(op_stack[-1]) >= precedence(op)
                ):
                    apply_top_operator()
                op_stack.append(op)

        while op_stack:
            apply_top_operator()

        fragment = value_stack[-1]
        if fragment.startswith("(") and fragment.endswith(")"):
            fragment = fragment[1:-1]
        return fragment

    def _latex_weight_prefixes(self) -> dict[int, str]:
        """Return per-term-start mathtext fraction weight prefixes.

        Mirrors the ``weight_prefix_by_start`` map in
        :meth:`_formula_string_with_fraction_groups`, rendering the free
        fraction symbols and the ``(1 - f_X - f_Y)`` remainder in mathtext.
        """
        prefixes: dict[int, str] = {}
        for group in self.fraction_groups:
            term_starts = self._fraction_group_term_starts(group)
            free_symbols: list[str] = []
            for idx in term_starts[:-1]:
                name = self._fraction_param_name(idx)
                symbol = param_symbol_latex(get_param_info(name).latex, name)
                prefixes[idx] = symbol
                free_symbols.append(symbol)
            remainder = "(1" + "".join(f" - {sym}" for sym in free_symbols) + ")"
            prefixes[term_starts[-1]] = remainder
        return prefixes

    def _group_within_span(self, start: int, end: int) -> tuple[int, int] | None:
        """Return the sole fraction group intersecting ``[start, end]``, else None.

        ``group`` on a :class:`LatexTerm` is set only when exactly one fraction
        group's component range intersects the term's range; when a term
        contains two (or zero) groups the accent is ambiguous, so ``None``.
        """
        hits = [group for group in self.fraction_groups if not (group[1] < start or group[0] > end)]
        return hits[0] if len(hits) == 1 else None

    def latex_terms(self) -> list[LatexTerm]:
        """Return the model as a list of typeset (mathtext) additive terms.

        Splits at top-level ``+``/``-`` operators; each term carries its
        ``separator`` ("" for the first, " + "/" - " otherwise) and the sole
        fraction ``group`` its component range intersects (or ``None``). Never
        raises and never returns an empty list for a valid model — a single
        multiplicative chain yields one term.
        """
        terms: list[LatexTerm] = []
        for start, end, separator in self._top_level_terms():
            try:
                latex = self._latex_span_fragment(start, end)
            except Exception:  # noqa: BLE001 - preview must never raise
                names = self.component_names[start : end + 1]
                latex = fallback_function_latex("+".join(names) or "model", [])
            group = self._group_within_span(start, end)
            terms.append(LatexTerm(latex=latex, separator=separator, group=group))
        if not terms:
            terms.append(
                LatexTerm(
                    latex=fallback_function_latex("model", []),
                    separator="",
                    group=None,
                )
            )
        return terms

    def latex_string(self) -> str:
        """Return the whole model as one mathtext string.

        The concatenation ``"".join(sep + latex)`` over :meth:`latex_terms`.
        """
        return "".join(term.separator + term.latex for term in self.latex_terms())

    def to_model_definition(self, name: str = "Composite") -> ModelDefinition:
        """Create a ModelDefinition-compatible wrapper for the fit engine."""
        return ModelDefinition(
            name=name,
            description=self.formula_string(),
            function=self.function,
            param_names=list(self.param_names),
            param_defaults=dict(self.param_defaults),
            param_info=dict(self.param_info),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the model."""
        return {
            "component_names": list(self.component_names),
            "operators": list(self.operators),
            "open_parentheses": list(self.open_parentheses),
            "close_parentheses": list(self.close_parentheses),
            "fraction_groups": [[start, end] for start, end in self.fraction_groups],
        }

    @classmethod
    def from_dict(cls, data: dict, *, allow_missing: bool = False) -> CompositeModel:
        """Construct a CompositeModel from serialized data.

        With ``allow_missing=True``, component names that are not registered
        materialise as named zero-valued placeholders instead of raising —
        the degrade path for projects referencing user functions that are not
        installed in this session (the original names round-trip unchanged
        through :meth:`to_dict`).
        """
        component_names = data.get("component_names")
        operators = data.get("operators")
        open_parentheses = data.get("open_parentheses")
        close_parentheses = data.get("close_parentheses")
        fraction_groups = data.get("fraction_groups")
        if not isinstance(component_names, list) or not all(
            isinstance(v, str) for v in component_names
        ):
            raise ValueError("Invalid composite model data: component_names")
        if operators is not None:
            if not isinstance(operators, list) or not all(isinstance(v, str) for v in operators):
                raise ValueError("Invalid composite model data: operators")
        if open_parentheses is not None:
            if not isinstance(open_parentheses, list) or not all(
                isinstance(v, int) for v in open_parentheses
            ):
                raise ValueError("Invalid composite model data: open_parentheses")
        if close_parentheses is not None:
            if not isinstance(close_parentheses, list) or not all(
                isinstance(v, int) for v in close_parentheses
            ):
                raise ValueError("Invalid composite model data: close_parentheses")
        if fraction_groups is not None:
            if not isinstance(fraction_groups, list) or not all(
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(idx, int) for idx in value)
                for value in fraction_groups
            ):
                raise ValueError("Invalid composite model data: fraction_groups")
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=[(start, end) for start, end in (fraction_groups or [])],
            allow_missing=allow_missing,
        )


def _legacy_fraction_numbering(model: CompositeModel) -> list[list[int]]:
    """Return the old ``fraction_<k>`` numbers per group, in group order.

    The retired scheme numbered one ``fraction_<k>`` per additive term,
    consecutively across all groups in group order (see the pre-migration
    ``_build_fraction_param_number_map``). This reconstructs that numbering so
    legacy value dicts can be located and migrated.
    """
    numbering: list[list[int]] = []
    next_number = 1
    for group in model.fraction_groups:
        term_starts = model._fraction_group_term_starts(group)
        group_numbers = list(range(next_number, next_number + len(term_starts)))
        numbering.append(group_numbers)
        next_number += len(term_starts)
    return numbering


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Best-effort ``float(value)``, falling back to *default* when malformed.

    A corrupted legacy project can carry ``None`` or a non-numeric string for a
    fraction value (hand-edited or truncated ``.asymp`` file). The old
    positional scheme's parse path silently defaulted such values, so this
    preserves that behavior instead of letting ``TypeError``/``ValueError``
    abort project loading (see the migration functions below, which are called
    outside any try/except by ``FitSlot.from_dict`` and
    ``migrate_legacy_fraction_state``).
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def has_legacy_fraction_values(model: CompositeModel, values: Mapping[str, float]) -> bool:
    """Return True when ``values`` carries old ``fraction_<k>`` keys for ``model``.

    Only the numbers that the model's fraction groups would have used under the
    retired positional scheme count — a stray ``fraction_9`` unrelated to any
    group does not trigger migration.
    """
    for group_numbers in _legacy_fraction_numbering(model):
        if any(f"fraction_{k}" in values for k in group_numbers):
            return True
    return False


def migrate_legacy_fraction_values(
    model: CompositeModel, values: Mapping[str, float]
) -> dict[str, float]:
    """Convert a legacy fraction value dict to the n-1 free-parameter scheme.

    For each group whose legacy keys ``fraction_a..fraction_b`` are present (one
    per term), compute the old normalized weights (clamp ≥ 0, divide by the sum;
    equal weights when the sum ≤ 1e-30) and assign the first n-1 to the group's
    new free parameters, dropping the last (now the derived remainder). All
    non-fraction keys pass through untouched; consumed legacy keys are removed.
    The model's structural :meth:`CompositeModel.to_dict` is unaffected — only
    value dicts need migrating.
    """
    migrated = dict(values)
    for group, group_numbers in zip(
        model.fraction_groups, _legacy_fraction_numbering(model), strict=True
    ):
        legacy_keys = [f"fraction_{k}" for k in group_numbers]
        if not any(key in migrated for key in legacy_keys):
            continue
        raw = [max(_coerce_float(migrated.get(key, 0.0)), 0.0) for key in legacy_keys]
        total = sum(raw)
        if total <= 1e-30:
            weights = [1.0 / float(len(raw))] * len(raw)
        else:
            weights = [value / total for value in raw]
        # Assign the first n-1 normalized weights to the new free parameters;
        # the last term is the derived remainder and gets no key.
        term_starts = model._fraction_group_term_starts(group)
        for idx, weight in zip(term_starts[:-1], weights[:-1], strict=True):
            migrated[model._fraction_param_name(idx)] = weight
        for key in legacy_keys:
            migrated.pop(key, None)
    return migrated


def _legacy_fraction_rename_map(model: CompositeModel) -> dict[str, str | None]:
    """Map each legacy ``fraction_<k>`` name to its new name (or ``None`` to drop).

    For every group the positional legacy keys ``fraction_a..fraction_b`` (one
    per additive term) map to the group's term starts ``t_1..t_n``. Under the
    n-1 scheme ``t_1..t_{n-1}`` become free parameters (their new names) and
    ``t_n`` — the derived remainder — has no parameter, so its legacy key maps
    to ``None`` (drop). Only names that a value/entry dict must be rewritten
    against appear here.
    """
    rename: dict[str, str | None] = {}
    for group, group_numbers in zip(
        model.fraction_groups, _legacy_fraction_numbering(model), strict=True
    ):
        term_starts = model._fraction_group_term_starts(group)
        for number, idx in zip(group_numbers[:-1], term_starts[:-1], strict=True):
            rename[f"fraction_{number}"] = model._fraction_param_name(idx)
        rename[f"fraction_{group_numbers[-1]}"] = None
    return rename


def migrate_legacy_fraction_parameter_entries(
    model: CompositeModel, parameters: list[dict]
) -> list[dict]:
    """Migrate a list of parameter-state dicts from the legacy fraction scheme.

    Each entry is a GUI/serialised parameter-state dict carrying at least a
    ``"name"`` and ``"value"`` (plus metadata such as ``fixed``/``min``/``max``/
    ``uncertainty``/``type``/``bounds``). Legacy ``fraction_<k>`` entries are
    rewritten in place: the first n-1 terms of each group keep their metadata but
    are renamed to the new free-parameter name and take the normalised migrated
    weight as their value; the dropped last-term entry is removed. Every other
    entry passes through untouched, preserving order. A no-op (returns a shallow
    copy) when the parameter list carries no legacy fraction keys for the model.
    """
    value_map = {
        str(entry["name"]): _coerce_float(entry.get("value", 0.0))
        for entry in parameters
        if isinstance(entry, dict) and "name" in entry
    }
    if not has_legacy_fraction_values(model, value_map):
        return [dict(entry) for entry in parameters]

    migrated_values = migrate_legacy_fraction_values(model, value_map)
    rename = _legacy_fraction_rename_map(model)

    migrated: list[dict] = []
    for entry in parameters:
        if not isinstance(entry, dict):
            migrated.append(entry)
            continue
        name = str(entry.get("name", ""))
        if name in rename:
            new_name = rename[name]
            if new_name is None:
                # Dropped derived-remainder term: no parameter under the new scheme.
                continue
            new_entry = dict(entry)
            new_entry["name"] = new_name
            new_entry["value"] = migrated_values.get(new_name, entry.get("value"))
            migrated.append(new_entry)
        else:
            migrated.append(dict(entry))
    return migrated


def migrate_legacy_fraction_parameter_set(
    model: CompositeModel, parameter_set: ParameterSet
) -> ParameterSet:
    """Migrate a :class:`ParameterSet` off the legacy ``fraction_<k>`` scheme.

    The :class:`ParameterSet` analogue of
    :func:`migrate_legacy_fraction_parameter_entries`: for each group the first
    n-1 legacy ``fraction_<k>`` parameters are renamed to their new
    free-parameter name and take the normalised migrated weight as their value
    (all other metadata — bounds, ``fixed``, ``expr``, links, ties — preserved);
    the derived last-term parameter is dropped. Every other parameter passes
    through untouched, preserving order. Returns the input unchanged when it
    carries no legacy fraction keys for this model.
    """
    value_map = {parameter.name: float(parameter.value) for parameter in parameter_set}
    if not has_legacy_fraction_values(model, value_map):
        return parameter_set

    migrated_values = migrate_legacy_fraction_values(model, value_map)
    rename = _legacy_fraction_rename_map(model)

    migrated = ParameterSet()
    for parameter in parameter_set:
        if parameter.name in rename:
            new_name = rename[parameter.name]
            if new_name is None:
                # Dropped derived-remainder term: no parameter under the new scheme.
                continue
            migrated.add(
                Parameter(
                    name=new_name,
                    value=migrated_values.get(new_name, parameter.value),
                    min=parameter.min,
                    max=parameter.max,
                    fixed=parameter.fixed,
                    expr=parameter.expr,
                    link_group=parameter.link_group,
                    tie=parameter.tie,
                )
            )
        else:
            migrated.add(parameter)
    return migrated


def migrate_legacy_fraction_state(state: Mapping[str, object]) -> dict[str, object]:
    """Migrate a saved single/global fit-state blob to the n-1 fraction scheme.

    ``state`` is the GUI single-/global-fit form payload carrying a
    ``composite_model`` dict and a ``parameters`` list of parameter-state dicts.
    When the model reconstructs and its parameter list carries legacy
    ``fraction_<k>`` keys, the ``parameters`` list is migrated via
    :func:`migrate_legacy_fraction_parameter_entries`; otherwise (missing/
    malformed model, or no legacy keys) the state is returned with a shallow-
    copied ``parameters`` list and is otherwise unchanged. Never raises — a
    model that fails to reconstruct simply skips migration.
    """
    migrated = dict(state)
    parameters = state.get("parameters")
    model_data = state.get("composite_model")
    if not isinstance(parameters, list) or not isinstance(model_data, dict):
        return migrated
    try:
        model = CompositeModel.from_dict(model_data, allow_missing=True)
    except (ValueError, KeyError, TypeError):
        return migrated
    migrated["parameters"] = migrate_legacy_fraction_parameter_entries(model, parameters)
    return migrated
