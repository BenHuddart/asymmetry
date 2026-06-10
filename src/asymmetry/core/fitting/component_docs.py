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
        "This is convenient when field is the physically controlled variable in the experiment model. "
        "For a transverse-field muonium experiment, model the central diamagnetic Mu+ line with this "
        "component and add MuoniumTF for the Mu0 satellites."
    ),
    "MuoniumTF": (
        "Use for transverse-field muonium (Mu0): the four hyperfine transitions about the applied field, "
        "parameterized by field B and the hyperfine coupling A_hf. In the shallow-donor (small A_hf) limit "
        "it reduces to two satellites straddling the diamagnetic line with separation A_hf, so A_hf reads "
        "off the hyperfine constant directly. Compose with *Exponential for relaxation and add a separate "
        "OscillatoryField for the central diamagnetic line."
    ),
    "MuoniumLowTF": (
        "Low-field approximation to transverse-field muonium: the two dominant Mu0 satellite frequencies "
        "(WiMDA's LowTFMuonium). Use when only two satellites are resolved; otherwise prefer MuoniumTF."
    ),
    "MuoniumZF": (
        "Use for zero-field axial muonium: three hyperfine lines f1=A_hf-D, f2=A_hf+D/2, f3=3D/2 set by the "
        "hyperfine A_hf and axial anisotropy D, with an optional Lorentzian cutoff f_cut. There is no applied "
        "field, so no central diamagnetic line."
    ),
    "StretchedExponential": (
        "Use for distributed relaxation-rate environments where a single exponential is insufficient. "
        "The stretching exponent beta captures broad dynamical heterogeneity phenomenologically."
    ),
    "StaticGKT_ZF": (
        "Use for zero-field static Gaussian Kubo-Toyabe relaxation from randomly oriented nuclear moments. "
        "It is a standard baseline for nonmagnetic or weakly magnetic static local-field distributions."
    ),
    "LongitudinalFieldKT": (
        "Use for a static, dense (Gaussian) local-field distribution of width Δ with an applied longitudinal "
        "field Bₗ — the workhorse for frozen/static magnetism and dilute nuclear-dipole hosts. Sweeping Bₗ "
        "through the decoupling crossover (γµBₗ ~ Δ) and watching the polarisation recover toward 1 is the "
        "unambiguous signature that the local field is *static*; the Bₗ → 0 limit is the zero-field ⅓-tail "
        "Kubo–Toyabe (StaticGKT_ZF). If the polarisation does not recover, or relaxation persists at high "
        "field, the field is *dynamic* — use DynamicGaussianKT or Keren instead. Δ in µs⁻¹, Bₗ in Gauss."
    ),
    "DynamicGaussianKT": (
        "Use when a dense (Gaussian) static local-field distribution of width Δ is partially averaged by "
        "fluctuations at rate ν, such as muon hopping or thermally fluctuating moments. It is the strong-collision "
        "generalisation of the static Gaussian Kubo–Toyabe: ν → 0 recovers the static function (with its ⅓ tail in "
        "zero field), while ν ≫ Δ gives motional narrowing toward exponential decay with rate 2Δ²/ν. Set "
        "Bₗ for longitudinal-field decoupling studies (e.g. ionic-motion field sweeps). This is the standard model for "
        "extracting a hop/fluctuation rate and its activation energy in metals (Cu) and ionic conductors. "
        "Computed via the strong-collision dynamicisation integral; grid-independent to better than 0.5%."
    ),
    "DynamicLorentzianKT": (
        "Use instead of the Gaussian dynamic KT when the local-field distribution is Lorentzian rather than Gaussian, "
        "i.e. for dilute or randomly diluted moments (spin glasses, dilute-spin systems), with half-width aₗ fluctuating "
        "at rate ν. ν → 0 recovers the static Lorentzian Kubo–Toyabe. Zero field is exact (analytic); the "
        "longitudinal-field line shape is computed by an analytic angular average (≈0.2% accurate for Bₗ ≳ 20 G). "
        "See the 'Dynamic and fluctuating-field relaxation functions' user-guide page for the method and accuracy."
    ),
    "Keren": (
        "Use as the analytic longitudinal-field dynamic Gaussian relaxation function: an exact strong-collision result in "
        "the fast/intermediate fluctuation regime that avoids the numerical convolution of the full dynamic KT. It is the "
        "model named for longitudinal-field decoupling analyses (e.g. ionic diffusion) and reduces to the Abragam function "
        "at zero field. Prefer the full DynamicGaussianKT when fluctuations are slow (ν ≲ Δ) or the static ⅓ tail matters."
    ),
    "Abragam": (
        "Use for single-component relaxation that crosses over from a Gaussian line shape (slow fluctuations, ν → 0: "
        "exp(−Δ²t²/2)) to an exponential (fast fluctuations, ν ≫ Δ: exp(−(Δ²/ν) t)). It is the classic "
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
    "RischKehr": (
        "Use when the muon (or muonium) polarization is relaxed by a spin carrier diffusing in one dimension, "
        "such as a polaron moving along a conducting-polymer chain or an excitation confined to a structural channel. "
        "The Risch-Kehr function G(t) = exp(Γt)·erfc(√(Γt)) (Risch and Kehr, Phys. Rev. B 46, 5246 (1992)) replaces the "
        "exponential expected for 3D fluctuations: the 1D random walk keeps returning the carrier to the muon, giving a "
        "characteristic (πΓt)^(-1/2) long-time tail. A signature of the 1D regime is a stretched-exponential fit returning "
        "beta near 1/2 at early times; prefer this function over a stretched exponential when 1D transport is physically "
        "motivated (cf. the mobile-excitation discussion in Blundell, De Renzi, Lancaster & Pratt, Muon Spectroscopy, "
        "OUP 2022, section 8.4). Γ is a rate in μs⁻¹ and must be non-negative."
    ),
    "Bessel": (
        "Use for magnets with incommensurate order, such as spin-density-wave states, where the muon samples the "
        "Overhauser distribution of local fields between -B₁ and +B₁ rather than a single value. The zero-field "
        "polarization is A·J₀(2πft) with f = γ_μB₁/2π set by the field-distribution edge (Muon Spectroscopy, OUP 2022, "
        "eqn 6.47; the (TMTSF)₂PF₆ example of section 6.4). At late times it resembles a damped cosine with a "
        "characteristic -45° phase shift (eqn 6.48) — if a free-phase Oscillatory fit insists on a phase near -45°, "
        "try this component instead. Compose with *Exponential or *Gaussian for additional relaxation. For "
        "commensurate order use Oscillatory/OscillatoryField instead."
    ),
    "MuoniumHighTF": (
        "Use for transverse-field muonium at high field (B well above B₀ = A_hf/(γ_e+γ_μ) ≈ 1585 G for vacuum muonium), "
        "where only the two intramultiplet transitions ν₁₂ and ν₃₄ carry appreciable amplitude and are observed as a pair "
        "of lines whose frequencies sum to the hyperfine constant: ν₁₂ + ν₃₄ = A_hf (Muon Spectroscopy, OUP 2022, "
        "eqn 4.65). Fitting the pair therefore measures A_hf directly even when neither line is individually assigned. "
        "Both lines are given equal weight (the high-field limit); at lower fields, where the (1±δ) weights and the other "
        "two transitions matter, use MuoniumTF. The frequencies are computed from the exact Breit-Rabi levels, so the "
        "component remains correct down to intermediate fields apart from the equal-weight approximation."
    ),
    "MuoniumHighTFAniso": (
        "Use for high transverse-field muonium with an axially anisotropic hyperfine interaction — bond-centred muonium "
        "in semiconductors or muoniated radicals — measured on a polycrystalline or powder sample. The hyperfine tensor "
        "is written as an isotropic part A_hf plus an axial (traceless) part D (Muon Spectroscopy, OUP 2022, eqn 4.68); "
        "each crystallite shifts the two high-field lines by ±d/2 with d = (D/2)(3cos²θ-1), and the powder average over "
        "cosθ produces the characteristic asymmetric Pake-like broadening of the pair. D = 0 reduces exactly to "
        "MuoniumHighTF. For single crystals fit the orientation-dependent lines directly with MuoniumTF or Oscillatory "
        "components instead."
    ),
    "MuoniumLFRelax": (
        "Use for the longitudinal-field spin-lattice (T1) relaxation of muonium when a fluctuating coupling — nuclear "
        "hyperfine fields modulated by muonium hopping, or electron spin exchange with carriers — relaxes the muon spin "
        "via the intratriplet ν₁₂ transition. The rate follows the BPP/Redfield form λ = (1-δ)·δ_ex²·τ_c/(1+(2πν₁₂τ_c)²) "
        "with ν₁₂ from the exact Breit-Rabi levels and δ = x/√(1+x²) (cf. the quantum-diffusion analyses of Kiefl et al., "
        "Phys. Rev. Lett. 62, 792 (1989) and Kadono et al., Phys. Rev. Lett. 64, 665 (1990); Redfield form as in Muon "
        "Spectroscopy, OUP 2022, eqn 5.53). Measuring λ versus B_L and locating the T1 minimum (2πν₁₂τ_c ≈ 1) "
        "determines both δ_ex and τ_c. A_hf defaults to vacuum muonium (4463 MHz) and should normally be fixed. "
        "This component is a relaxation envelope: multiply an amplitude component, or use it standalone for the "
        "repolarized muonium fraction."
    ),
    "GaussianBroadenedKT": (
        "Use when a static Kubo-Toyabe fit is qualitatively right but the dip is too sharp and the 1/3-tail recovery too "
        "pronounced — the signature of a *distribution* of static widths Δ across muon sites, as in structurally "
        "disordered hosts, dilute magnetic alloys, or systems with several inequivalent sites. The component averages the "
        "static (longitudinal-field) Gaussian Kubo-Toyabe over a Gaussian distribution of Δ with fractional standard "
        "deviation w_Δ (the Gaussian-broadened Gaussian of Noakes and Kalvius, Phys. Rev. B 56, 2352 (1997); WiMDA's "
        "'Gau broad KT', whose 'rel width' equals w_Δ·√2). w_Δ = 0 reduces exactly to LongitudinalFieldKT. For dynamic "
        "(fluctuating) fields use DynamicGaussianKT instead — broadening and dynamics both fill in the dip and are "
        "difficult to distinguish from a single spectrum, so vary temperature or field before trusting either."
    ),
    "DynamicFmuF": (
        "Use when an F-mu-F signal (clear at low temperature) progressively damps and loses its oscillations on warming "
        "because the muon hops away from the F-mu-F site or the coupling fluctuates: the static collinear F-mu-F "
        "polarization (Muon Spectroscopy, OUP 2022, eqn 4.81) is dynamicized by the strong-collision model (eqn 5.30) "
        "with fluctuation rate ν. ν = 0 recovers FmuF_Linear exactly; large ν gives motional narrowing toward "
        "exp(-2ω_d²t/ν). Fitting a temperature series with shared r_muF and free ν extracts the hop rate and hence an "
        "activation energy for muon diffusion in the fluoride. Assumes the equal-distance collinear geometry of "
        "FmuF_Linear."
    ),
    "FmuF_Triangle": (
        "Use for fluorides where the muon couples to the two nearest fluorines of an F-mu-F centre *and* a non-negligible "
        "third fluorine — the situation identified in, e.g., second-neighbour analyses of ionic fluorides (cf. Wilkinson "
        "and Blundell, Phys. Rev. Lett. 125, 087201 (2020); Muon Spectroscopy, OUP 2022, section 4.5). The collinear "
        "F-mu-F pair sits at r_muF and the third fluorine at distance r₃, at angle φ₃ to the F-mu-F axis; the 16-dimensional "
        "four-spin problem is solved exactly with all mu-F and F-F dipolar couplings and a full powder average. "
        "As r₃ → ∞ it approaches FmuF_General's collinear limit (FmuF_Linear plus the F-F coupling). Unlike WiMDA's "
        "'F-u-F-F' function it includes the F-F couplings and a proper powder average, so fitted distances are not "
        "directly comparable with WiMDA results. Evaluation is cached per geometry; fits are slower than the analytic "
        "F-mu-F components."
    ),
    "DipolarPairField": (
        "Use for a muon dipolar-coupled to a single spin-1/2 nucleus when you want to fit the dipolar field at the muon "
        "directly rather than assume a nucleus and distance. The zero-field polycrystalline polarization is the "
        "two-spin form (1/6)[1 + e^(-λ_T·t)(2cos(ω_d t/2) + cos(ω_d t) + 2cos(3ω_d t/2))] with ω_d = γ_μB_dip "
        "(Muon Spectroscopy, OUP 2022, eqn 4.80; Meier, Hyperfine Interact. 17-19, 427 (1984)). The transverse damping "
        "λ_T applies only to the oscillating 5/6 part, modelling couplings to more distant nuclei; the non-oscillating "
        "1/6 component is preserved. Use MuF (fluorine), ProtonDipole, or ElectronDipole to parameterize by distance "
        "instead; B_dip relates to a distance via B_dip = μ₀ħγ_j/(4πr³)."
    ),
    "ProtonDipole": (
        "Use for muon stopping sites adjacent to a single dominant proton — hydroxyl groups, hydrides, or water of "
        "crystallization — where the zero-field signal shows the characteristic two-spin beat pattern. Identical physics "
        "to MuF but with the proton gyromagnetic ratio: the fitted r_μH is the muon-proton distance from "
        "ħω_d = μ₀ħ²γ_μγ_p/(4πr³) (Muon Spectroscopy, OUP 2022, eqns 4.76 and 4.80). The transverse damping λ_T acts "
        "on the oscillating part only, absorbing weaker couplings to more distant nuclei. Note that proton moments are "
        "~10x weaker than ¹⁹F at the same distance, so resolvable oscillations require a close, well-defined mu-H pair."
    ),
    "ElectronDipole": (
        "Use for a muon coupled by dipolar interaction to a single *localized* electronic moment at distance r_μe in zero "
        "field — for example a dilute paramagnetic defect or rare-earth ion adjacent to the muon site — when the moment "
        "is static on the muon timescale. Same two-spin form as the nuclear pairs but with the electron gyromagnetic "
        "ratio, so frequencies are ~3 orders of magnitude higher at the same distance and r_μe of several Å still gives "
        "MHz-scale oscillations. Not appropriate for muonium (where the contact hyperfine dominates — use the Muonium "
        "components) or for dense magnets (use Oscillatory/Bessel with an internal field)."
    ),
    "DipolarSpinJ": (
        "Use for zero-field precession of a muon coupled to one nucleus of spin J > 1/2 with both dipolar and "
        "quadrupolar interactions — e.g. mu-⁹³Nb (J=9/2) or mu-⁶³,⁶⁵Cu (J=3/2) pairs, where the electric-field gradient "
        "produced by the muon itself quadrupole-splits the neighbouring nucleus (cf. the quadrupolar discussion around "
        "eqn 4.87 of Muon Spectroscopy, OUP 2022). Implements the closed-form polycrystalline eigen-solution of Celio "
        "and Meier, Hyperfine Interact. 17-19, 435 (1984): f_dip sets the dipolar coupling, f_quad the quadrupolar "
        "splitting (sign-sensitive), and J the nuclear spin, which should be held fixed at the known value. For J = 1/2 "
        "it reduces to the two-spin pair (quadrupole inactive). For more than one strongly coupled nucleus use the "
        "F-mu-F family or a dedicated multi-spin model."
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
