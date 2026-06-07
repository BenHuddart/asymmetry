"""Human-readable applicability notes for fit-function components."""

from __future__ import annotations

PARAMETER_MODEL_APPLICABILITY: dict[str, str] = {
    "Constant": (
        "Use when the fitted parameter is effectively independent of x over the selected range. "
        "This is often a baseline model for testing whether more complex trends are justified by data."
    ),
    "Linear": (
        "Use for first-order monotonic trends where the parameter changes approximately linearly with x. "
        "It is usually appropriate over narrow windows where curvature is not yet significant."
    ),
    "PowerLaw": (
        "Use near scale-invariant regimes or crossover windows where the observable follows a power law. "
        "It is especially useful for phenomenological critical-like behavior without a full microscopic model."
    ),
    "ExponentialDecay": (
        "Use for relaxation toward an asymptote with a characteristic x-scale tau. "
        "This form is common for activated suppression or screening effects that saturate at large x."
    ),
    "Arrhenius": (
        "Use for thermally activated processes where a barrier controls the trend versus temperature. "
        "The fitted Ea parameter can be interpreted as an activation energy in the model's chosen units."
    ),
    "CriticalDivergence": (
        "Use close to a critical temperature where a quantity grows strongly as |T-Tc| decreases. "
        "The exponent nu captures the divergence rate and should be interpreted within the model range only."
    ),
    "OrderParameter": (
        "Use for a magnetic order parameter that grows continuously below an ordering temperature Tc and "
        "vanishes above it, such as a spontaneous precession frequency, internal field, or ordered-moment-like "
        "asymmetry tracked across a second-order transition. The amplitude y0 is the saturated T=0 value (in the "
        "unit of the trended observable), beta is the critical exponent (about 0.33-0.37 for 3D Heisenberg/Ising "
        "magnets and 0.5 for mean field), and the shape exponent alpha controls the departure from a pure power "
        "law away from Tc; fix alpha=1 to fit the near-Tc power law y0*(1-T/Tc)^beta. The model is exactly zero "
        "for T>=Tc, so include the ordered-phase points and let Tc fall within the fitted range. It is not "
        "appropriate for diverging quantities (use CriticalDivergence instead) or for first-order transitions."
    ),
    "Redfield": (
        "Use for longitudinal-field relaxation in the motional-narrowing picture of dynamic local fields. "
        "It is appropriate when a single dominant fluctuation rate and coupling scale are physically meaningful."
    ),
    "Lorentzian": (
        "Use for field response with a central peak and a characteristic width B0. "
        "This is a compact phenomenological shape for broad resonant-like suppression away from center."
    ),
    "GaussianLCR": (
        "Use for localized level-crossing resonance features around a specific field B0. "
        "Bwid controls broadening and f sets the resonance amplitude contribution."
    ),
    "DiffusionLF_1D": (
        "Use when LF relaxation is governed by diffusion that is effectively one-dimensional. "
        "Typical scenarios include channel-like transport pathways with strong directional constraint."
    ),
    "DiffusionLF_2D": (
        "Use when LF relaxation is governed by diffusion on layered or planar pathways. "
        "This model is suited to quasi-two-dimensional transport in anisotropic materials."
    ),
    "DiffusionLF_3D": (
        "Use for isotropic or weakly anisotropic bulk diffusion in three dimensions. "
        "It is the most natural diffusion-LF baseline when no reduced dimensionality is evident."
    ),
    "BallisticLF_1D": (
        "Use when LF relaxation is governed by effectively one-dimensional ballistic transport, such as spinon propagation along chains. "
        "The 1D model is the relevant choice when a logarithmic lambda versus field signature is expected on a semilog field axis."
    ),
    "BallisticLF_2D": (
        "Use when excitations propagate ballistically within an effectively two-dimensional manifold. "
        "This form is appropriate for coherent in-plane transport where a diffusive random-walk picture is not justified."
    ),
    "BallisticLF_3D": (
        "Use for three-dimensional ballistic transport with coherent propagation across the active volume. "
        "It is the ballistic analogue of the 3D diffusion LF model when reduced dimensionality is not physically motivated."
    ),
    "Lambda_bg": (
        "Use as an additive field-independent relaxation background for lambda-like observables. "
        "It is commonly combined with explicit field-dependent terms to capture residual broadening."
    ),
    "SC_SWave": (
        "Use for superconductors consistent with a single isotropic fully gapped order parameter. "
        "It is a standard baseline for nodeless BCS-like superfluid-density behavior."
    ),
    "SC_DWave": (
        "Use for nodal d-wave candidates where low-temperature excitations remain significant. "
        "This often captures stronger low-T deviations from activated behavior than s-wave forms."
    ),
    "SC_AnisotropicS_Cos4": (
        "Use for anisotropic nodeless gaps with fourfold modulation on the Fermi surface. "
        "The anisotropy parameter controls angular variation while preserving an s-wave-like symmetry class."
    ),
    "SC_NonmonotonicD": (
        "Use for non-monotonic d-wave scenarios with mixed angular harmonics in the gap function. "
        "The beta term adjusts the relative weight of harmonics and can capture shape changes in rho(T)."
    ),
    "SC_SPlusG": (
        "Use for anisotropic singlet s+g phenomenology when pure isotropic s-wave and pure d-wave are both too restrictive. "
        "This model captures strong angular modulation while retaining a single superconducting channel."
    ),
    "SC_PWaveAxial": (
        "Use for candidate odd-parity axial p-wave superconducting states. "
        "Apply when symmetry arguments or complementary probes motivate an axial p-wave gap structure."
    ),
    "SC_ExtendedS": (
        "Use for extended-s scenarios where anisotropy or sign structure modifies the isotropic s-wave kernel. "
        "This is useful when data sit between simple isotropic s-wave and nodal alternatives."
    ),
    "SC_AlphaModel": (
        "Use when a single-gap BCS shape is retained but coupling strength differs from weak coupling. "
        "The alpha parameter rescales the gap ratio and can emulate strong-coupling effects."
    ),
    "SC_TwoGap_SS": (
        "Use when two distinct nodeless superconducting gaps contribute to the measured response. "
        "The weight parameter partitions spectral contribution between the two gap channels."
    ),
    "SC_TwoGap_SD": (
        "Use for mixed-symmetry two-gap interpretations combining s-wave and d-wave components. "
        "This is appropriate when neither pure s-wave nor pure d-wave forms fit the full temperature range."
    ),
    "SC_SWave_Q": (
        "Use when superconducting and normal contributions combine in quadrature with an s-wave kernel. "
        "It is suitable when linewidth sources are independent and add as variances rather than linearly."
    ),
    "SC_DWave_Q": (
        "Use when superconducting and normal contributions combine in quadrature with a d-wave kernel. "
        "It captures nodal-gap temperature dependence while preserving quadrature mixing assumptions."
    ),
    "SC_SPlusG_Q": (
        "Use when an s+g anisotropic gap is required and superconducting/non-superconducting linewidth channels "
        "are modeled as independent Gaussian contributions that add in quadrature."
    ),
}

FIT_COMPONENT_APPLICABILITY: dict[str, str] = {
    "Exponential": (
        "Use for homogeneous dynamic relaxation with a single dominant rate scale. "
        "This is often the first model tested for simple paramagnetic or fluctuation-dominated signals."
    ),
    "Gaussian": (
        "Use when depolarization is driven by a static or quasi-static Gaussian field distribution. "
        "It is common for dense random local fields where a Gaussian second moment is appropriate."
    ),
    "Oscillatory": (
        "Use for coherent muon spin precession at a well-defined frequency in an internal or applied field. "
        "The phase and frequency track magnetic order and field calibration effects."
    ),
    "OscillatoryField": (
        "Use when precession frequency is parameterized via field B using gamma_mu instead of direct f. "
        "This is convenient when field is the physically controlled variable in the experiment model."
    ),
    "StretchedExponential": (
        "Use for distributed relaxation-rate environments where a single exponential is insufficient. "
        "The stretching exponent beta captures broad dynamical heterogeneity phenomenologically."
    ),
    "StaticGKT_ZF": (
        "Use for zero-field static Gaussian Kubo-Toyabe relaxation from randomly oriented nuclear moments. "
        "It is a standard baseline for nonmagnetic or weakly magnetic static local-field distributions."
    ),
    "DynamicGaussianKT": (
        "Use when a dense (Gaussian) static local-field distribution of width Delta is partially averaged by "
        "fluctuations at rate nu, such as muon hopping or thermally fluctuating moments. It is the strong-collision "
        "generalisation of the static Gaussian Kubo-Toyabe: nu -> 0 recovers the static function (with its 1/3 tail in "
        "zero field), while nu >> Delta gives motional narrowing toward exponential decay with rate 2*Delta^2/nu. Set "
        "B_L for longitudinal-field decoupling studies (e.g. ionic-motion field sweeps). This is the standard model for "
        "extracting a hop/fluctuation rate and its activation energy in metals (Cu) and ionic conductors."
    ),
    "DynamicLorentzianKT": (
        "Use instead of the Gaussian dynamic KT when the local-field distribution is Lorentzian rather than Gaussian, "
        "i.e. for dilute or randomly diluted moments (spin glasses, dilute-spin systems), with half-width a_L fluctuating "
        "at rate nu. nu -> 0 recovers the static Lorentzian Kubo-Toyabe. Zero field is exact; longitudinal-field support "
        "currently dynamicises the zero-field line shape (see the dynamic-relaxation porting study)."
    ),
    "Keren": (
        "Use as the analytic longitudinal-field dynamic Gaussian relaxation function: an exact strong-collision result in "
        "the fast/intermediate fluctuation regime that avoids the numerical convolution of the full dynamic KT. It is the "
        "model named for longitudinal-field decoupling analyses (e.g. ionic diffusion) and reduces to the Abragam function "
        "at zero field. Prefer the full DynamicGaussianKT when fluctuations are slow (nu <~ Delta) or the static 1/3 tail matters."
    ),
    "Abragam": (
        "Use for single-component relaxation that crosses over from a Gaussian line shape (slow fluctuations, nu -> 0: "
        "exp(-Delta^2 t^2/2)) to an exponential (fast fluctuations, nu >> Delta: exp(-(Delta^2/nu) t)). It is the classic "
        "model for extracting a hop/correlation rate from a transverse-field line shape, e.g. the Cu diffusion line-shape "
        "change from Gaussian to Lorentzian on warming."
    ),
    "Constant": (
        "Use for a time-independent background term from non-relaxing or spectrometer-background contributions. "
        "It is typically included additively with dynamic or oscillatory components."
    ),
    "MuF": (
        "Use for Case I muon-fluorine stopping states where the muon is strongly coupled to one dominant 19F nucleus, "
        "giving the characteristic three-frequency oscillation of a two-spin mu-F pair. This is the appropriate starting "
        "model for fluorine-containing molecular materials where a symmetric site between two fluorines is not favored, as in "
        "the CuF2(H2O)2(pyz) example discussed by Lancaster et al. It should not be used when the data clearly require two comparable "
        "fluorine couplings or an additional nearby spin such as a proton."
    ),
    "FmuF_Linear": (
        "Use for the classic hydrogen-bond-like linear F-mu-F center found in simple ionic fluorides, where the muon sits approximately "
        "midway between two equivalent fluorines and the zero-field polarization is described by the analytical collinear three-spin form. "
        "This is the natural model for LiF, NaF, CaF2, and BaF2 type systems and for any material where a nearly symmetric two-fluorine site "
        "is expected. It is not the right model for crooked or asymmetric molecular geometries, because its closed form assumes equal mu-F distances "
        "and the collinear powder-averaged geometry of the Brewer et al. treatment."
    ),
    "FmuF_General": (
        "Use for distorted or asymmetric three-spin F-mu-F stopping states where the muon couples to two fluorines with inequivalent distances and/or "
        "a bent bond angle. This is the relevant model for Case II molecular-magnet scenarios such as the crooked F-mu-F geometry identified in "
        "[Cu(NO3)(pyz)2]PF6, where a linear FmuF model is too restrictive. The present implementation still assumes only three coupled spins "
        "(F-mu-F); it is therefore not intended for Case III situations that require an additional nearby nucleus, such as the proton in HF2-."
    ),
}


def get_component_applicability(component_name: str) -> str:
    """Return physical-applicability text for a known component name."""
    if component_name in PARAMETER_MODEL_APPLICABILITY:
        return PARAMETER_MODEL_APPLICABILITY[component_name]
    if component_name in FIT_COMPONENT_APPLICABILITY:
        return FIT_COMPONENT_APPLICABILITY[component_name]
    return (
        "Use this component when its mathematical form matches the expected physics "
        "and the fit remains stable across your selected data range."
    )
