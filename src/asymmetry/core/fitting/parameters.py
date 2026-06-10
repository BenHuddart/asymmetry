"""Parameter objects with bounds, constraints, and linking."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INDEXED_PARAM_RE = re.compile(r"^(.+)_([0-9]+)$")
_GLE_CONTROL_WORD_RE = re.compile(r"^\\[A-Za-z]+$")


@dataclass(frozen=True)
class ParamInfo:
    """Display and unit metadata for a fit parameter."""

    name: str
    plain: str
    unicode: str
    latex: str
    gle: str
    unit: str | None = None
    default_min: float | None = None  # None means no lower bound (-inf)
    description: str | None = None

    def with_index(self, index: str) -> ParamInfo:
        """Return indexed metadata (e.g. ``A`` -> ``A_2``)."""
        return ParamInfo(
            name=f"{self.name}_{index}",
            plain=f"{self.plain}_{index}",
            unicode=f"{self.unicode}_{index}",
            latex=_append_latex_index(self.latex, index),
            gle=f"{self.gle}_{{{index}}}",
            unit=self.unit,
            default_min=self.default_min,
            description=self.description,
        )

    def plain_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.plain} ({self.unit})"
        return self.plain

    def unicode_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.unicode} ({self.unit})"
        return self.unicode

    def latex_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.latex} ({self.unit})"
        return self.latex

    def gle_label(self, *, include_unit: bool = True) -> str:
        gle_symbol = _gle_symbol_with_spacing_boundary(self.gle)
        if include_unit and self.unit:
            return f"{gle_symbol} ({_unit_to_gle(self.unit)})"
        return gle_symbol


def split_parameter_name(name: str) -> tuple[str, str | None]:
    """Split a parameter name into base and optional numeric suffix."""
    match = _INDEXED_PARAM_RE.match(name)
    if not match:
        return name, None
    return match.group(1), match.group(2)


def _append_latex_index(symbol: str, index: str) -> str:
    if symbol.startswith("$") and symbol.endswith("$"):
        return f"${symbol[1:-1]}_{{{index}}}$"
    return f"{symbol}_{index}"


def _unit_to_gle(unit: str) -> str:
    # Keep GLE labels in native markup (no $...$ math mode).
    unit_gle = unit.replace("μ", r"{\rm \mu}{}")
    # Normalize common unicode superscripts to ASCII exponent markup.
    unit_gle = unit_gle.replace("⁻¹", "^{-1}")
    unit_gle = unit_gle.replace("⁻²", "^{-2}")
    unit_gle = unit_gle.replace("⁻³", "^{-3}")
    return unit_gle


def _gle_symbol_with_spacing_boundary(symbol: str) -> str:
    # For bare control words (e.g. \lambda), add {} so following spaces are not swallowed.
    if _GLE_CONTROL_WORD_RE.match(symbol):
        return f"{symbol}{{}}"
    return symbol


PARAM_INFO_REGISTRY: dict[str, ParamInfo] = {
    "A": ParamInfo("A", "A", "A", r"$A$", r"{\it A}", "%", default_min=0.0),
    "A0": ParamInfo("A0", "A0", "A₀", r"$A_0$", r"{\it A}_{0}", "%", default_min=0.0),
    "A_bg": ParamInfo("A_bg", "A_bg", "A_bg", r"$A_{bg}$", r"{\it A}_{bg}", "%", default_min=0.0),
    "Lambda": ParamInfo("Lambda", "Lambda", "λ", r"$\lambda$", r"\lambda", "μs⁻¹", default_min=0.0),
    "sigma": ParamInfo("sigma", "sigma", "σ", r"$\sigma$", r"\sigma", "μs⁻¹", default_min=0.0),
    "Delta": ParamInfo("Delta", "Delta", "Δ", r"$\Delta$", r"\Delta", "μs⁻¹", default_min=0.0),
    "a_L": ParamInfo("a_L", "a_L", "a", r"$a$", r"{\it a}", "μs⁻¹", default_min=0.0),
    "beta": ParamInfo("beta", "beta", "β", r"$\beta$", r"\beta", default_min=0.0),
    "alpha": ParamInfo("alpha", "alpha", "α", r"$\alpha$", r"\alpha", default_min=0.0),
    "y0": ParamInfo(
        "y0",
        "y0",
        "y₀",
        r"$y_0$",
        r"{\it y}_{0}",
        default_min=0.0,
        description="Saturated (T=0) value of an order-parameter trend.",
    ),
    "phase": ParamInfo("phase", "phase", "φ", r"$\phi$", r"\phi", "rad"),
    "frequency": ParamInfo(
        "frequency", "frequency", "f", r"$f$", r"{\it f}", "MHz", default_min=0.0
    ),
    "height": ParamInfo(
        "height",
        "height",
        "height",
        r"$h$",
        r"{\it h}",
        "a.u.",
        default_min=0.0,
        description="Frequency-domain peak height.",
    ),
    "nu0": ParamInfo(
        "nu0",
        "nu0",
        "ν₀",
        r"$\nu_0$",
        r"\nu_{0}",
        "MHz",
        default_min=0.0,
        description="Frequency-domain peak centre.",
    ),
    "fwhm": ParamInfo(
        "fwhm",
        "FWHM",
        "FWHM",
        r"$\mathrm{FWHM}$",
        r"{\rm FWHM}",
        "MHz",
        default_min=0.0,
        description="Full width at half maximum of a frequency-domain peak.",
    ),
    "bg": ParamInfo(
        "bg",
        "bg",
        "bg",
        r"$b_g$",
        r"{\it b}_{g}",
        "a.u.",
        description="Frequency-domain background level.",
    ),
    "slope": ParamInfo(
        "slope",
        "slope",
        "slope",
        r"$m$",
        r"{\it m}",
        "a.u./MHz",
        description="Linear frequency-domain background slope.",
    ),
    "field": ParamInfo("field", "field", "B", r"$B$", r"{\it B}", "G"),
    "A_hf": ParamInfo("A_hf", "A_hf", "Aµ", r"$A_\mu$", r"{\it A}_{\mu}", "MHz", default_min=0.0),
    "D_mu": ParamInfo("D_mu", "D_mu", "D_µ", r"$D_\mu$", r"{\it D}_{\mu}", "MHz"),
    "f_cut": ParamInfo(
        "f_cut", "f_cut", "f_cut", r"$f_{\mathrm{cut}}$", r"{\it f}_{cut}", "MHz", default_min=0.0
    ),
    "B_L": ParamInfo("B_L", "B_L", "B_L", r"$B_L$", r"{\it B}_{L}", "G"),
    "r_muF": ParamInfo(
        "r_muF", "r_muF", "r_μF", r"$r_{\mu F}$", r"{\it r}_{\mu F}", "Å", default_min=0.0
    ),
    "r1": ParamInfo("r1", "r1", "r₁", r"$r_1$", r"{\it r}_{1}", "Å", default_min=0.0),
    "r2": ParamInfo("r2", "r2", "r₂", r"$r_2$", r"{\it r}_{2}", "Å", default_min=0.0),
    "r3": ParamInfo("r3", "r3", "r₃", r"$r_3$", r"{\it r}_{3}", "Å", default_min=0.0),
    "theta": ParamInfo("theta", "theta", "θ", r"$\theta$", r"\theta", "°", default_min=0.0),
    "phi3": ParamInfo("phi3", "phi3", "φ₃", r"$\phi_3$", r"\phi_{3}", "°", default_min=0.0),
    "Gamma": ParamInfo("Gamma", "Gamma", "Γ", r"$\Gamma$", r"\Gamma", "μs⁻¹", default_min=0.0),
    "delta_ex": ParamInfo(
        "delta_ex", "delta_ex", "δ_ex", r"$\delta_{ex}$", r"\delta_{ex}", "MHz", default_min=0.0
    ),
    "tau_c": ParamInfo("tau_c", "tau_c", "τ_c", r"$\tau_c$", r"\tau_{c}", "μs", default_min=0.0),
    "w_rel": ParamInfo(
        "w_rel", "w_rel", "w_Δ", r"$w_\Delta$", r"{\it w}_{\Delta}", default_min=0.0
    ),
    "B_dip": ParamInfo(
        "B_dip", "B_dip", "B_dip", r"$B_{dip}$", r"{\it B}_{dip}", "G", default_min=0.0
    ),
    "lambda_T": ParamInfo(
        "lambda_T", "lambda_T", "λ_T", r"$\lambda_T$", r"\lambda_{T}", "μs⁻¹", default_min=0.0
    ),
    "r_muH": ParamInfo(
        "r_muH", "r_muH", "r_μH", r"$r_{\mu H}$", r"{\it r}_{\mu H}", "Å", default_min=0.0
    ),
    "r_mue": ParamInfo(
        "r_mue", "r_mue", "r_μe", r"$r_{\mu e}$", r"{\it r}_{\mu e}", "Å", default_min=0.0
    ),
    "f_dip": ParamInfo(
        "f_dip", "f_dip", "f_dip", r"$f_{dip}$", r"{\it f}_{dip}", "MHz", default_min=0.0
    ),
    "f_quad": ParamInfo("f_quad", "f_quad", "f_quad", r"$f_{quad}$", r"{\it f}_{quad}", "MHz"),
    "J_spin": ParamInfo("J_spin", "J", "J", r"$J$", r"{\it J}", default_min=0.5),
    "baseline": ParamInfo("baseline", "baseline", "baseline", "baseline", "baseline", "%"),
    "a": ParamInfo("a", "a", "a", r"$a$", r"{\it a}"),
    "b": ParamInfo("b", "b", "b", r"$b$", r"{\it b}"),
    "c": ParamInfo("c", "c", "c", r"$c$", r"{\it c}"),
    "n": ParamInfo("n", "n", "n", r"$n$", r"{\it n}"),
    "tau": ParamInfo("tau", "tau", "τ", r"$\tau$", r"\tau", default_min=0.0),
    "B0": ParamInfo("B0", "B0", "B₀", r"$B_0$", r"{\it B}_{0}", "G"),
    "Bwid": ParamInfo(
        "Bwid", "Bwid", "B_wid", r"$B_{wid}$", r"{\it B}_{wid}", "G", default_min=0.0
    ),
    "Tc": ParamInfo("Tc", "Tc", "T_c", r"$T_c$", r"{\it T}_{c}", "K", default_min=0.0),
    "Ea": ParamInfo("Ea", "Ea", "E_a", r"$E_a$", r"{\it E}_{a}", "meV", default_min=0.0),
    "D": ParamInfo("D", "D", "D", r"$D$", r"{\it D}", "MHz", default_min=0.0),
    "nu": ParamInfo("nu", "nu", "ν", r"$\nu$", r"\nu", "MHz", default_min=0.0),
    "m": ParamInfo("m", "m", "m", r"$m$", r"{\it m}"),
    "f": ParamInfo("f", "f", "f", r"$f$", r"{\it f}", "μs⁻¹", default_min=0.0),
    "D_2D": ParamInfo(
        "D_2D", "D_2D", "D_2D", r"$D_{2D}$", r"{\it D}_{2D}", "μs⁻¹", default_min=0.0
    ),
    "D_hop": ParamInfo(
        "D_hop",
        "D_hop",
        "D_hop",
        r"$D_{\mathrm{hop}}$",
        r"{\it D}_{hop}",
        "μs⁻¹",
        default_min=0.0,
    ),
    "D_nD": ParamInfo(
        "D_nD", "D_nD", "D_nD", r"$D_{nD}$", r"{\it D}_{nD}", "μs⁻¹", default_min=0.0
    ),
    "D_perp": ParamInfo(
        "D_perp", "D_perp", "D_⊥", r"$D_{\perp}$", r"{\it D}_{\perp}", "μs⁻¹", default_min=0.0
    ),
    "lambda_BG": ParamInfo(
        "lambda_BG",
        "lambda_BG",
        "λ_BG",
        r"$\lambda_{BG}$",
        r"\lambda_{BG}",
        "μs⁻¹",
        default_min=0.0,
    ),
    "lambda_0D": ParamInfo(
        "lambda_0D",
        "lambda_0D",
        "λ_0D",
        r"$\lambda_{0D}$",
        r"\lambda_{0D}",
        "μs⁻¹",
        default_min=0.0,
    ),
    "C": ParamInfo("C", "C", "C", r"$C$", r"{\it C}", "MHz", default_min=0.0),
    "sigma_0": ParamInfo(
        "sigma_0", "sigma_0", "σ_0", r"$\sigma_0$", r"\sigma_{0}", "μs⁻¹", default_min=0.0
    ),
    "sigma_bg": ParamInfo(
        "sigma_bg", "sigma_bg", "σ_bg", r"$\sigma_{bg}$", r"\sigma_{bg}", "μs⁻¹", default_min=0.0
    ),
    "sigma_sc": ParamInfo(
        "sigma_sc", "sigma_sc", "σ_sc", r"$\sigma_{sc}$", r"\sigma_{sc}", "μs⁻¹", default_min=0.0
    ),
    "sigma_nm": ParamInfo(
        "sigma_nm", "sigma_nm", "σ_nm", r"$\sigma_{nm}$", r"\sigma_{nm}", "μs⁻¹", default_min=0.0
    ),
    "gap_ratio": ParamInfo(
        "gap_ratio",
        "gap_ratio",
        "Δ0/kBTc",
        r"$\Delta_0/(k_B T_c)$",
        r"\Delta_{0}/(k_{B} T_{c})",
        default_min=0.0,
    ),
    "gap_ratio_1": ParamInfo(
        "gap_ratio_1",
        "gap_ratio_1",
        "Δ01/kBTc",
        r"$\Delta_{01}/(k_B T_c)$",
        r"\Delta_{01}/(k_{B} T_{c})",
        default_min=0.0,
    ),
    "gap_ratio_2": ParamInfo(
        "gap_ratio_2",
        "gap_ratio_2",
        "Δ02/kBTc",
        r"$\Delta_{02}/(k_B T_c)$",
        r"\Delta_{02}/(k_{B} T_{c})",
        default_min=0.0,
    ),
    "gap_ratio_s": ParamInfo(
        "gap_ratio_s",
        "gap_ratio_s",
        "Δ0s/kBTc",
        r"$\Delta_{0s}/(k_B T_c)$",
        r"\Delta_{0s}/(k_{B} T_{c})",
        default_min=0.0,
    ),
    "gap_ratio_d": ParamInfo(
        "gap_ratio_d",
        "gap_ratio_d",
        "Δ0d/kBTc",
        r"$\Delta_{0d}/(k_B T_c)$",
        r"\Delta_{0d}/(k_{B} T_{c})",
        default_min=0.0,
    ),
    "a_anis": ParamInfo("a_anis", "a_anis", "a", r"$a$", r"{\it a}"),
    "shape_factor_a": ParamInfo(
        "shape_factor_a",
        "shape_factor_a",
        "a_shape",
        r"$a_{\mathrm{shape}}$",
        r"{\it a}_{shape}",
        default_min=0.0,
    ),
    "beta_nm": ParamInfo("beta_nm", "beta_nm", "β_nm", r"$\beta_{nm}$", r"\beta_{nm}"),
    "alpha_sc": ParamInfo("alpha_sc", "alpha_sc", "α", r"$\alpha$", r"\alpha", default_min=0.0),
    "weight": ParamInfo("weight", "weight", "w", r"$w$", r"{\it w}", default_min=0.0),
    "fraction": ParamInfo(
        "fraction",
        "fraction",
        "fraction",
        r"$f$",
        r"{\it f}",
        default_min=0.0,
    ),
    "signed_gap": ParamInfo(
        "signed_gap", "signed_gap", "signed", "signed", "signed", default_min=0.0
    ),
}

_PARAM_DESCRIPTIONS: dict[str, str] = {
    "A": "Amplitude prefactor for a component contribution.",
    "A0": "Initial asymmetry amplitude at t = 0.",
    "A_bg": "Time-independent background asymmetry level.",
    "Lambda": "Exponential relaxation rate constant.",
    "sigma": "Gaussian relaxation rate related to field-distribution width.",
    "Delta": "Static Gaussian field-distribution width in Kubo-Toyabe models.",
    "a_L": "Static Lorentzian field-distribution half-width (rate) in Lorentzian Kubo-Toyabe models.",
    "beta": "Stretching exponent controlling deviation from simple exponential relaxation.",
    "phase": "Phase offset of oscillatory precession.",
    "frequency": "Muon spin precession frequency.",
    "field": "Applied or effective magnetic field magnitude.",
    "A_hf": "Muonium hyperfine coupling constant; sets the satellite splitting about the central line.",
    "D_mu": "Axial anisotropy of the zero-field muonium hyperfine interaction.",
    "f_cut": "Lorentzian cutoff frequency damping high-frequency muonium lines (0 disables it).",
    "B_L": "Applied longitudinal magnetic field (Gauss) to decouple muons from the static local-field distribution.",
    "r_muF": "Muon-fluorine distance for two-spin or linear F-mu-F polarization functions.",
    "r1": "First muon-fluorine distance in the general F-mu-F geometry.",
    "r2": "Second muon-fluorine distance in the general F-mu-F geometry.",
    "r3": "Distance from the muon to the third fluorine in the F-mu-F + F geometry.",
    "theta": "F-mu-F bond angle in degrees for the general three-spin geometry.",
    "phi3": "Angle in degrees between the F-mu-F axis and the third fluorine direction.",
    "Gamma": "Risch-Kehr relaxation rate set by the 1D diffusion of the depolarizing carrier.",
    "delta_ex": "Amplitude of the fluctuating (nuclear-hyperfine or spin-exchange) coupling relaxing muonium.",
    "tau_c": "Correlation time of the fluctuating coupling (inverse hop or collision rate).",
    "w_rel": "Fractional standard deviation of the Gaussian distribution of the Kubo-Toyabe width Delta.",
    "B_dip": "Dipolar field at the muon from the coupled nuclear spin (omega_d = gamma_mu B_dip).",
    "lambda_T": "Transverse damping applied to the oscillating part of the dipole-pair polarization.",
    "r_muH": "Muon-proton distance for the spin-1/2 dipole-pair polarization.",
    "r_mue": "Muon-electron distance for the spin-1/2 dipole-pair polarization.",
    "f_dip": "Dipolar coupling frequency between the muon and the spin-J nucleus.",
    "f_quad": "Quadrupolar splitting frequency of the spin-J nucleus (sign-sensitive).",
    "J_spin": "Nuclear spin quantum number J (half-integer); normally held fixed.",
    "baseline": "Additive constant baseline contribution.",
    "a": "Scale prefactor for the component term.",
    "b": "Additive intercept term.",
    "c": "Constant offset term.",
    "n": "Power-law exponent controlling curvature.",
    "tau": "Characteristic decay scale of x in the exponential term.",
    "B0": "Characteristic field scale or resonance-center field.",
    "Bwid": "Characteristic Gaussian field width around the resonance center.",
    "Tc": "Critical temperature where the ordered/superconducting state emerges.",
    "Ea": "Activation energy for thermally activated behavior.",
    "D": "Dynamic coupling scale entering the Redfield relaxation contribution.",
    "nu": "Fluctuation rate for local-field dynamics.",
    "m": "Field-dependence exponent in generalized Redfield-like denominators.",
    "f": "Amplitude of the Gaussian level-crossing resonance contribution.",
    "D_2D": "In-plane diffusion rate used in diffusion-assisted LF relaxation models.",
    "D_hop": "Ballistic hopping rate entering the Bessel-function transport autocorrelation.",
    "D_nD": "Effective n-dimensional diffusion rate.",
    "D_perp": "Perpendicular (interlayer) diffusion rate component.",
    "lambda_BG": "Field-independent background relaxation contribution.",
    "lambda_0D": "Field-independent local (0D) dynamic relaxation contribution.",
    "C": "Overall coupling prefactor in transport-based relaxation expressions.",
    "sigma_0": "Additive superconducting scale in sigma(T)=sigma_0*rho_s(T)+sigma_bg; approximately the T->0 superconducting linewidth.",
    "sigma_bg": "Additive non-superconducting background linewidth (temperature-independent within the model).",
    "sigma_sc": "Quadrature superconducting linewidth scale in sigma(T)=sqrt((sigma_sc*rho_s)^2+sigma_nm^2).",
    "sigma_nm": "Quadrature normal/nuclear linewidth floor combined in quadrature with superconducting broadening.",
    "gap_ratio": "Dimensionless zero-temperature gap ratio Delta_0/(k_B T_c); weak-coupling references include 1.764 (s-wave), 2.14 (d-wave), and 2.77 (s+g).",
    "gap_ratio_1": "Gap ratio Delta_01/(k_B T_c) for band/component 1 in two-gap s+s models.",
    "gap_ratio_2": "Gap ratio Delta_02/(k_B T_c) for band/component 2 in two-gap s+s models.",
    "gap_ratio_s": "s-wave channel gap ratio in mixed two-gap s+d models.",
    "gap_ratio_d": "d-wave channel gap ratio in mixed two-gap s+d models.",
    "a_anis": "Fourfold anisotropy amplitude in g(phi)=1+a*cos(4phi); |a|<1 is nodeless while |a|>=1 can generate accidental nodes.",
    "shape_factor_a": "Optional weak-coupling shape-factor parameter a used in the generalized reduced-gap law; values > 0 enable the generalized form, while 0 falls back to the Carrington-Manzano interpolation.",
    "beta_nm": "Harmonic-mixing parameter in nonmonotonic d-wave g(phi)=beta*cos(2phi)+(1-beta)*cos(6phi).",
    "alpha_sc": "Alpha-model scaling factor multiplying the weak-coupling s-wave ratio: Delta_0/(k_B T_c)=alpha_sc*1.764.",
    "weight": "Band mixing weight w constrained to [0,1] for weighted sums rho=w*rho_1+(1-w)*rho_2.",
    "fraction": "Normalized mixture fraction within a fraction-defined composite component group.",
    "signed_gap": "Flag-like control for extended-s convention; nonzero values preserve sign of cos(2phi), zero uses magnitude |cos(2phi)|.",
}


def _attach_description(info: ParamInfo) -> ParamInfo:
    return ParamInfo(
        name=info.name,
        plain=info.plain,
        unicode=info.unicode,
        latex=info.latex,
        gle=info.gle,
        unit=info.unit,
        default_min=info.default_min,
        description=_PARAM_DESCRIPTIONS.get(info.name),
    )


PARAM_INFO_REGISTRY = {
    name: _attach_description(info) for name, info in PARAM_INFO_REGISTRY.items()
}


def get_param_info(name: str) -> ParamInfo:
    """Return metadata for a parameter name, including indexed variants."""
    base_name, index = split_parameter_name(name)
    info = PARAM_INFO_REGISTRY.get(base_name)
    if info is None:
        info = ParamInfo(base_name, base_name, base_name, f"${base_name}$", base_name)
    if index is None:
        return ParamInfo(
            name,
            info.plain,
            info.unicode,
            info.latex,
            info.gle,
            info.unit,
            info.default_min,
            info.description,
        )
    return info.with_index(index)


def param_info_map(param_names: list[str]) -> dict[str, ParamInfo]:
    """Build a parameter metadata mapping for a parameter-name sequence."""
    return {name: get_param_info(name) for name in param_names}


@dataclass
class Parameter:
    """A single fit parameter."""

    name: str
    value: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    fixed: bool = False
    expr: str | None = None  # Expression constraint (e.g. tie to another param)
    link_group: int | None = None  # Equality link group id (WiMDA-style); None = unlinked

    @property
    def is_constrained(self) -> bool:
        return self.fixed or self.expr is not None


class ParameterSet:
    """Ordered collection of :class:`Parameter` objects."""

    def __init__(self, params: list[Parameter] | None = None) -> None:
        self._params: dict[str, Parameter] = {}
        for p in params or []:
            self.add(p)

    def add(self, param: Parameter) -> None:
        self._params[param.name] = param

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    @property
    def free_parameters(self) -> list[Parameter]:
        followers = set(self.link_followers())
        return [p for p in self if not p.is_constrained and p.name not in followers]

    @property
    def names(self) -> list[str]:
        return list(self._params)

    # --- equality link groups (WiMDA "Ties") ---------------------------------

    def link_groups(self) -> dict[int, list[Parameter]]:
        """Return members of each equality link group, keyed by group id.

        Singletons (a group with one member) are dropped — linking one
        parameter to nothing is a no-op.
        """
        groups: dict[int, list[Parameter]] = {}
        for p in self:
            if p.link_group is not None:
                groups.setdefault(p.link_group, []).append(p)
        return {gid: members for gid, members in groups.items() if len(members) > 1}

    def link_main(self, group: int) -> Parameter:
        """Return the "main" parameter of a link group.

        Mirrors WiMDA: the first member, unless a later member is free
        (non-fixed) — then the first free member is the main, so the free-fit
        set always contains the group main.
        """
        members = self.link_groups()[group]
        for member in members:
            if not member.fixed:
                return member
        return members[0]

    def link_followers(self) -> dict[str, str]:
        """Map each non-main linked parameter name to its group main's name."""
        followers: dict[str, str] = {}
        for gid, members in self.link_groups().items():
            main = self.link_main(gid)
            for member in members:
                if member.name != main.name:
                    followers[member.name] = main.name
        return followers

    def values_array(self) -> list[float]:
        return [p.value for p in self]

    def update_values(self, values: dict[str, float]) -> None:
        for name, val in values.items():
            if name in self._params:
                self._params[name].value = val
