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
    "Constant": (
        "Use for a time-independent background term from non-relaxing or spectrometer-background contributions. "
        "It is typically included additively with dynamic or oscillatory components."
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
