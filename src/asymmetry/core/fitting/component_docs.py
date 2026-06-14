"""Human-readable applicability notes and literature references for fit components.

Each user-facing component carries:

* an **applicability** paragraph — a concise statement of when the model is
  physically relevant (shown in the component-info dialog), written with
  rendered symbols (Greek letters, unicode sub/superscripts) rather than
  ASCII names; and
* a short **reference list** in APS style, citing the original literature for
  the functional form (shown directly below the applicability).

Expressions are kept consistent with the conventions of Blundell, De Renzi,
Lancaster & Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022), but the
user-facing text cites the primary sources rather than textbook equation
numbers.
"""

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
    "Polynomial": (
        "Use as a flexible empirical trend or background up to fifth order in x when no physical model "
        "is available. Fix the coefficients of unused orders at 0 to fit lower orders (for a quadratic, "
        "fix c₃–c₅). Polynomials extrapolate poorly: keep the order as low as the residuals allow and "
        "interpret the coefficients only within the fitted window."
    ),
    "Cubic": (
        "Use as the curved or sloping background under an avoided-level-crossing (ALC) field scan, where "
        "a straight line cannot follow the off-resonance baseline. Fitted over the non-resonant regions and "
        "subtracted before the resonance peak is fitted. It is a well-conditioned third-order restriction of "
        "the general Polynomial; like any polynomial background it extrapolates poorly outside the fitted window."
    ),
    "PowerLaw": (
        "Use near scale-invariant regimes or crossover windows where the observable follows a power law. "
        "It is especially useful for phenomenological critical-like behavior without a full microscopic model. "
        "When a width-like quantity rides on a background that adds in quadrature rather than linearly, use "
        "PowerLawQuadBG (equivalently PowerLaw ⊕ Constant) instead of an additive constant."
    ),
    "PowerLawQuadBG": (
        "Use when a power-law signal combines with a constant background in quadrature, "
        "y = √((a·|x|ⁿ)² + BG²), as for width-like quantities (relaxation rates, linewidths) whose "
        "independent broadening channels add as squares. Unlike an additive constant, the curve "
        "saturates smoothly at BG where the power-law term is small; prefer PowerLaw with an additive "
        "constant when the background is a genuine offset of the observable itself. This fixed "
        "component is exactly the composite PowerLaw ⊕ Constant (the quadrature combinator), with BG "
        "mapping to the constant term and the power law's own additive constant fixed at 0; reach for "
        "the ⊕ form when either side is a richer model than a bare power law or constant."
    ),
    "ExponentialDecay": (
        "Use for relaxation toward an asymptote with a characteristic x-scale τ. "
        "This form is common for activated suppression or screening effects that saturate at large x."
    ),
    "Arrhenius": (
        "Use for thermally activated processes where an energy barrier controls the trend versus "
        "temperature: y = a·exp(−Eₐ/k_B·T). The fitted Eₐ can be interpreted as an activation energy."
    ),
    "CriticalDivergence": (
        "Use close to a critical temperature where a quantity grows strongly as |T − Tc| decreases. "
        "The exponent ν captures the divergence rate and should be interpreted within the fitted range only."
    ),
    "OrderParameter": (
        "Use for a magnetic order parameter that grows continuously below an ordering temperature Tc and "
        "vanishes above it, such as a spontaneous precession frequency, internal field, or ordered-moment-like "
        "asymmetry tracked across a second-order transition. The amplitude y₀ is the saturated T = 0 value (in "
        "the unit of the trended observable), β is the critical exponent (about 0.33–0.37 for 3D "
        "Heisenberg/Ising magnets and 0.5 for mean field), and the shape exponent α controls the departure from "
        "a pure power law away from Tc; fix α = 1 to fit the near-Tc power law y₀(1 − T/Tc)^β. The model is "
        "exactly zero for T ≥ Tc, so include the ordered-phase points and let Tc fall within the fitted range. "
        "It is not appropriate for diverging quantities (use CriticalDivergence instead) or for first-order "
        "transitions."
    ),
    "Redfield": (
        "Use for longitudinal-field relaxation in the motional-narrowing picture of dynamic local fields, "
        "where λ(B) = 2Δ²ν/(ν² + ωµ²) with ωµ = γµB. It is appropriate when a single dominant fluctuation "
        "rate ν and coupling scale are physically meaningful; the suppression of λ with field measures ν "
        "directly."
    ),
    "MuRepolarisation": (
        "Use for the longitudinal-field repolarisation of isotropic muonium: the time-averaged muon "
        "polarization rises from half the muonium amplitude at B = 0 to the full amplitude once "
        "B ≫ B₀ = A_hf/(γₑ + γµ), on top of a field-independent diamagnetic baseline a_Dia. Fit it to "
        "initial- or integral-asymmetry LF scans (e.g. built with the integral-vs-field observable) to "
        "estimate the hyperfine constant A_hf when the muonium precession is too fast to resolve — the "
        "standard method for large hyperfine couplings. It assumes an isotropic hyperfine interaction "
        "and time-averaged observation; anisotropic or chemically reacting muonium states distort the "
        "curve, and any missing fraction appears as a reduced a_Mu."
    ),
    "Lorentzian": (
        "Use for field response with a central peak and a characteristic width B₀. "
        "This is a compact phenomenological shape for broad resonant-like suppression away from center."
    ),
    "GaussianLCR": (
        "Use for localized avoided-level-crossing resonance features around a specific field B₀, where the "
        "muon relaxation is resonantly enhanced when its Zeeman splitting matches a transition of a nearby "
        "(typically quadrupolar) nucleus. B_wid controls broadening and f sets the resonance amplitude."
    ),
    "LorentzianLCR": (
        "Use as the Lorentzian-shaped alternative to GaussianLCR for avoided-level-crossing resonance peaks, "
        "appropriate when lifetime (rather than inhomogeneous) broadening dominates the resonance width."
    ),
    "DiffusionLF_1D": (
        "Use when longitudinal-field relaxation is governed by spin excitations diffusing in effectively one "
        "dimension, such as transport along chains or channel-like pathways. The 1D spectral density gives the "
        "characteristic λ ∝ ω^(−1/2) low-frequency divergence that distinguishes 1D diffusion on a field scan."
    ),
    "DiffusionLF_2D": (
        "Use when longitudinal-field relaxation is governed by diffusion on layered or planar pathways. "
        "This model is suited to quasi-two-dimensional transport in anisotropic materials."
    ),
    "DiffusionLF_3D": (
        "Use for isotropic or weakly anisotropic bulk diffusion in three dimensions. "
        "It is the natural diffusion baseline when no reduced dimensionality is evident."
    ),
    "BallisticLF_1D": (
        "Use when longitudinal-field relaxation is governed by effectively one-dimensional ballistic "
        "transport, such as spinon propagation along chains. The 1D ballistic model is the relevant choice "
        "when λ versus B is logarithmic on a semilog field axis, in contrast to the power-law signature of "
        "diffusive transport."
    ),
    "BallisticLF_2D": (
        "Use when excitations propagate ballistically within an effectively two-dimensional manifold. "
        "This form is appropriate for coherent in-plane transport where a diffusive random-walk picture is "
        "not justified."
    ),
    "BallisticLF_3D": (
        "Use for three-dimensional ballistic transport with coherent propagation across the active volume. "
        "It is the ballistic analogue of the 3D diffusion model when reduced dimensionality is not physically "
        "motivated."
    ),
    "Lambda_bg": (
        "Use as an additive field-independent relaxation background for λ-like observables. "
        "It is commonly combined with explicit field-dependent terms to capture residual broadening."
    ),
    "SC_SWave": (
        "Use for superconductors consistent with a single isotropic, fully gapped order parameter. "
        "It is the standard baseline for nodeless BCS-like superfluid-density behavior: σ(T) saturates "
        "exponentially at low temperature, reflecting the absence of low-energy quasiparticles."
    ),
    "SC_DWave": (
        "Use for nodal d-wave candidates where low-temperature quasiparticle excitations remain significant. "
        "The line nodes give σ(T) a linear low-temperature slope rather than activated saturation, which is "
        "the key qualitative difference from s-wave forms."
    ),
    "SC_AnisotropicS_Cos4": (
        "Use for anisotropic nodeless gaps with fourfold modulation on the Fermi surface. "
        "The anisotropy parameter controls angular variation while preserving an s-wave-like symmetry class."
    ),
    "SC_NonmonotonicD": (
        "Use for non-monotonic d-wave scenarios with mixed angular harmonics in the gap function. "
        "The β parameter adjusts the relative weight of harmonics and can capture shape changes in ρ_s(T)."
    ),
    "SC_SPlusG": (
        "Use for anisotropic singlet s+g phenomenology when pure isotropic s-wave and pure d-wave are both "
        "too restrictive. This model captures strong angular modulation while retaining a single "
        "superconducting channel."
    ),
    "SC_PWaveAxial": (
        "Use for candidate odd-parity axial p-wave superconducting states. "
        "Apply when symmetry arguments or complementary probes motivate an axial p-wave gap structure."
    ),
    "SC_ExtendedS": (
        "Use for extended-s scenarios where anisotropy or sign structure modifies the isotropic s-wave "
        "kernel. This is useful when data sit between simple isotropic s-wave and nodal alternatives."
    ),
    "SC_AlphaModel": (
        "Use when a single-gap BCS temperature dependence is retained but the coupling strength differs from "
        "weak coupling. The α parameter rescales the gap ratio and can emulate strong-coupling effects."
    ),
    "SC_TwoGap_SS": (
        "Use when two distinct nodeless superconducting gaps contribute to the measured response, as in "
        "multiband superconductors. The weight parameter partitions spectral contribution between the two "
        "gap channels."
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
        "Use when an s+g anisotropic gap is required and superconducting/non-superconducting linewidth "
        "channels are modeled as independent Gaussian contributions that add in quadrature."
    ),
}

FIT_COMPONENT_APPLICABILITY: dict[str, str] = {
    "Exponential": (
        "Use for homogeneous dynamic relaxation with a single dominant rate λ, as produced by rapidly "
        "fluctuating local fields in the motional-narrowing regime, or by a dilute (Lorentzian) distribution "
        "of static fields in transverse field. It is usually the first model tested for paramagnetic or "
        "fluctuation-dominated signals."
    ),
    "Gaussian": (
        "Use when depolarization is driven by a static or quasi-static, dense (Gaussian) distribution of "
        "local fields, with σ set by the second moment of the distribution. Typical sources are dense nuclear "
        "moments in transverse field; in zero field with the full ⅓ tail visible, prefer the Kubo–Toyabe "
        "forms, which expose the field-distribution width Δ directly. Note the convention A·exp[−(σt)²]: "
        "rates quoted for the e^(−Δ²t²/2) convention correspond to σ = Δ/√2."
    ),
    "Oscillatory": (
        "Use for coherent muon spin precession at a well-defined frequency ν, e.g. from long-range magnetic "
        "order in zero field (where ν = γµB_int/2π acts as an order parameter) or from an applied transverse "
        "field. Compose with a relaxation component to describe dephasing from field inhomogeneity or "
        "dynamics."
    ),
    "OscillatoryField": (
        "Use when precession is more naturally parameterized by the local or applied field B than by "
        "frequency, with ν = γµB/2π. This is convenient when field is the controlled experimental variable. "
        "For a transverse-field muonium experiment, model the central diamagnetic Mu⁺ line with this "
        "component and add MuoniumTF for the Mu⁰ satellites."
    ),
    "Bessel": (
        "Use for magnets with incommensurate order, such as spin-density-wave states, where muons sample the "
        "full (Overhauser) distribution of local fields between −B₁ and +B₁ rather than a single value. The "
        "zero-field polarization is A·J₀(2πνt) with ν = γµB₁/2π set by the field-distribution edge. At late "
        "times it resembles a damped cosine with a characteristic −45° phase shift, so a free-phase "
        "Oscillatory fit that insists on a phase near −45° is the usual hint to try this component. Compose "
        "with a relaxation component for additional damping; for commensurate order use "
        "Oscillatory/OscillatoryField instead."
    ),
    "MuoniumTF": (
        "Use for transverse-field muonium (Mu⁰): the four hyperfine transitions about the applied field, "
        "parameterized by field B and the hyperfine coupling A_hf. In the shallow-donor (small A_hf) limit it "
        "reduces to two satellites straddling the diamagnetic line with separation A_hf, so A_hf can be read "
        "off directly. Compose with a relaxation component for damping, and add a separate OscillatoryField "
        "for the central diamagnetic line."
    ),
    "MuoniumLowTF": (
        "Low-field approximation to transverse-field muonium: the two dominant intratriplet Mu⁰ frequencies. "
        "Use when only the two low-frequency lines are resolved (νB ≪ A_hf); otherwise prefer MuoniumTF."
    ),
    "MuoniumZF": (
        "Use for zero-field muonium with an axially anisotropic hyperfine interaction: three lines at "
        "ν₁ = A_hf − D, ν₂ = A_hf + D/2 and ν₃ = 3D/2, set by the isotropic coupling A_hf and anisotropy D, "
        "with an optional Lorentzian cutoff f_cut suppressing lines beyond the spectrometer bandwidth. There "
        "is no applied field, so no central diamagnetic line."
    ),
    "MuoniumHighTF": (
        "Use for transverse-field muonium at high field, B well above B₀ = A_hf/(γe + γµ) (≈ 1585 G for "
        "vacuum muonium), where only the two muon-spin-flip transitions ν₁₂ and ν₃₄ carry appreciable "
        "amplitude. Their frequencies sum to the hyperfine constant, ν₁₂ + ν₃₄ = A_hf, so fitting the pair "
        "measures A_hf directly even when neither line is individually assigned. Both lines are given equal "
        "weight (the high-field limit); at lower fields, where the transition amplitudes differ and the other "
        "two transitions matter, use MuoniumTF. The frequencies are computed from the exact Breit–Rabi "
        "levels."
    ),
    "MuoniumHighTFAniso": (
        "Use for high transverse-field muonium with an axially anisotropic hyperfine interaction — "
        "bond-centred muonium in semiconductors or muoniated radicals — measured on a polycrystalline or "
        "powder sample. The hyperfine tensor is written as an isotropic part A_hf plus an axial part D; each "
        "crystallite shifts the two high-field lines by ±d/2 with d = (D/2)(3cos²θ − 1), and the powder "
        "average over cosθ produces the characteristic asymmetric (Pake-like) broadening of the pair. D = 0 "
        "reduces exactly to MuoniumHighTF. For single crystals fit the orientation-dependent lines directly "
        "with MuoniumTF or Oscillatory components instead."
    ),
    "MuoniumLFRelax": (
        "Use for the longitudinal-field spin-lattice (T₁) relaxation of muonium when a fluctuating coupling — "
        "nuclear hyperfine fields modulated by muonium hopping, or electron spin exchange with carriers — "
        "relaxes the muon spin via the intratriplet ν₁₂ transition. The rate takes the Redfield form "
        "λ = δ_ex²·τ_c/[1 + (2πν₁₂τ_c)²], with ν₁₂ from the exact Breit–Rabi levels; the field decoupling "
        "enters solely through ν₁₂(B) moving the spectral density off the fluctuation peak (Kadono PRL 64, "
        "665 (1990); WiMDA's extra (1 − δ) prefactor is not in that source and is not applied here). "
        "Measuring λ versus B and locating the T₁ minimum (2πν₁₂τ_c ≈ 1) determines both δ_ex and τ_c, the "
        "classic route to muonium hop rates. A_hf defaults to vacuum muonium (4463 MHz) and should normally "
        "be held fixed. This component is a relaxation envelope: multiply an oscillating component, or use it "
        "standalone for the repolarized muonium fraction."
    ),
    "StretchedExponential": (
        "Use for distributed relaxation-rate environments where a single exponential is insufficient — spin "
        "glasses near freezing, dilute or disordered magnets, and other hosts with broad rate distributions. "
        "The stretching exponent β captures the breadth of the distribution phenomenologically: β drifting "
        "from 1 toward ⅓ on cooling through a transition is a canonical signature of glassy freezing, and "
        "β = ½ arises for dilute static moments in the fast-fluctuation limit."
    ),
    "RischKehr": (
        "Use when the muon (or muonium) polarization is relaxed by a spin carrier diffusing in one dimension, "
        "such as a polaron moving along a conducting-polymer chain or an excitation confined to a structural "
        "channel. The 1D random walk keeps returning the carrier to the muon, giving "
        "G(t) = e^(Γt)·erfc(√(Γt)) with a characteristic (πΓt)^(−1/2) long-time tail in place of an "
        "exponential. A stretched-exponential fit returning β near ½ at early times is the usual hint; prefer "
        "this function when 1D transport is physically motivated. Γ is a rate in µs⁻¹ and must be "
        "non-negative."
    ),
    "StaticGKT_ZF": (
        "Use for zero-field relaxation from a static, dense (Gaussian) distribution of local fields of width "
        "Δ — most commonly randomly oriented nuclear moments in nonmagnetic hosts. The Kubo–Toyabe dip and "
        "recovery to the ⅓ tail is the unambiguous zero-field signature of static disorder; if the tail "
        "relaxes, the field is partly dynamic (use DynamicGaussianKT)."
    ),
    "LongitudinalFieldKT": (
        "Use for a static, dense (Gaussian) local-field distribution of width Δ with an applied longitudinal "
        "field Bₗ — the workhorse for frozen/static magnetism and nuclear-dipole hosts. Sweeping Bₗ through "
        "the decoupling crossover (γµBₗ ~ Δ) and watching the polarization recover toward 1 is the "
        "unambiguous signature that the local field is static; the Bₗ → 0 limit is the zero-field ⅓-tail "
        "Kubo–Toyabe (StaticGKT_ZF). If the polarization does not recover, or relaxation persists at high "
        "field, the field is dynamic — use DynamicGaussianKT or Keren instead. Δ in µs⁻¹, Bₗ in Gauss."
    ),
    "DynamicGaussianKT": (
        "Use when a dense (Gaussian) static local-field distribution of width Δ is partially averaged by "
        "fluctuations at rate ν, such as muon hopping or thermally fluctuating moments. It is the "
        "strong-collision generalization of the static Gaussian Kubo–Toyabe: ν → 0 recovers the static "
        "function (with its ⅓ tail in zero field), while ν ≫ Δ gives motional narrowing toward exponential "
        "decay with rate 2Δ²/ν. Set Bₗ for longitudinal-field decoupling studies. This is the standard model "
        "for extracting a hop/fluctuation rate and its activation energy in metals and ionic conductors. "
        "Computed via the strong-collision integral equation; grid-independent to better than 0.5%."
    ),
    "DynamicLorentzianKT": (
        "Use instead of the Gaussian dynamic KT when the local-field distribution is Lorentzian rather than "
        "Gaussian, i.e. for dilute or randomly diluted moments (spin glasses, dilute-spin systems), with "
        "half-width aₗ fluctuating at rate ν. ν → 0 recovers the static Lorentzian Kubo–Toyabe. Zero field is "
        "exact (analytic); the longitudinal-field line shape is computed by an analytic angular average "
        "(≈0.2% accurate for Bₗ ≳ 20 G). See the Kubo-Toyabe page of the fit-function user guide "
        "for the method and accuracy."
    ),
    "Keren": (
        "Use as the analytic longitudinal-field dynamic Gaussian relaxation function: an accurate "
        "strong-collision result in the fast/intermediate fluctuation regime (ν ≳ Δ) that avoids the "
        "numerical convolution of the full dynamic Kubo–Toyabe. It is the standard model for "
        "longitudinal-field decoupling analyses (e.g. ionic diffusion) and reduces to the Abragam function at "
        "zero field. Prefer the full DynamicGaussianKT when fluctuations are slow (ν ≲ Δ) or the static ⅓ "
        "tail matters."
    ),
    "Abragam": (
        "Use for single-component relaxation that crosses over from a Gaussian line shape (slow fluctuations, "
        "ν → 0: exp(−Δ²t²/2)) to an exponential (fast fluctuations, ν ≫ Δ: exp(−(Δ²/ν)t)). It is the classic "
        "model for extracting a hop or correlation rate from a transverse-field line shape, e.g. the "
        "Gaussian-to-Lorentzian change of the Cu line shape as muon diffusion sets in on warming."
    ),
    "GaussianBroadenedKT": (
        "Use when a static Kubo–Toyabe fit is qualitatively right but the dip is too sharp and the ⅓-tail "
        "recovery too pronounced — the signature of a distribution of static widths Δ across muon sites, as "
        "in structurally disordered hosts, dilute magnetic alloys, or systems with several inequivalent "
        "sites. The component averages the static (longitudinal-field) Gaussian Kubo–Toyabe over a Gaussian "
        "distribution of Δ with fractional standard deviation w_Δ (WiMDA's 'Gau broad KT', whose 'rel width' "
        "equals w_Δ·√2); w_Δ = 0 reduces exactly to LongitudinalFieldKT. Beware: broadening and dynamics both "
        "fill in the dip and are difficult to distinguish from a single spectrum — vary temperature or field "
        "before trusting either."
    ),
    "Constant": (
        "Use for a time-independent background term from non-relaxing or spectrometer-background "
        "contributions, such as muons stopping in silver sample holders or cryostat tails. It is typically "
        "included additively with relaxing or oscillating components."
    ),
    "MuF": (
        "Use for muon–fluorine stopping states in which the muon is strongly coupled to one dominant ¹⁹F "
        "nucleus, giving the characteristic three-frequency oscillation of an entangled two-spin µ–F pair. "
        "This is the appropriate starting model for fluorine-containing molecular materials where a symmetric "
        "site between two fluorines is not favored. It should not be used when the data clearly require two "
        "comparable fluorine couplings (use FmuF_Linear or FmuF_General) or an additional nearby spin such as "
        "a proton."
    ),
    "FmuF_Linear": (
        "Use for the classic linear F–µ–F centre found in simple ionic fluorides (LiF, NaF, CaF₂, BaF₂ and "
        "similar), where the muon sits midway between two equivalent fluorines and the zero-field "
        "polarization follows the analytical collinear three-spin form. It is the natural model whenever a "
        "nearly symmetric two-fluorine site is expected. It is not the right model for bent or asymmetric "
        "geometries, because the closed form assumes equal µ–F distances and a collinear powder-averaged "
        "geometry."
    ),
    "FmuF_General": (
        "Use for distorted or asymmetric F–µ–F stopping states where the muon couples to two fluorines with "
        "inequivalent distances and/or a bent bond angle, as found in some molecular magnets where the linear "
        "model is too restrictive. The implementation still assumes only three coupled spins (F–µ–F); it is "
        "therefore not intended for situations that require an additional nearby nucleus, such as the proton "
        "in HF₂⁻ (and for a third fluorine, use FmuF_Triangle)."
    ),
    "DynamicFmuF": (
        "Use when an F–µ–F signal that is clear at low temperature progressively damps and loses its "
        "oscillations on warming, because the muon hops away from the F–µ–F site or the coupling fluctuates: "
        "the static collinear F–µ–F polarization is dynamicized by the strong-collision model with "
        "fluctuation rate ν. ν = 0 recovers FmuF_Linear exactly; large ν gives motional narrowing toward "
        "exp(−2ω_d²t/ν). Fitting a temperature series with shared r_µF and free ν extracts the hop rate and "
        "hence an activation energy for muon diffusion in the fluoride. Assumes the equal-distance collinear "
        "geometry of FmuF_Linear."
    ),
    "FmuF_Triangle": (
        "Use for fluorides where the muon couples to the two nearest fluorines of an F–µ–F centre and a "
        "non-negligible third fluorine, as established by second-neighbour analyses of ionic fluorides. The "
        "collinear F–µ–F pair sits at r_µF and the third fluorine at distance r₃, at angle φ₃ to the F–µ–F "
        "axis; the 16-dimensional four-spin problem is solved exactly with all µ–F and F–F dipolar couplings "
        "and a full powder average. As r₃ → ∞ it approaches the collinear limit of FmuF_General (FmuF_Linear "
        "plus the F–F coupling). Unlike WiMDA's 'F-u-F-F' function it includes the F–F couplings and a proper "
        "powder average, so fitted distances are not directly comparable with WiMDA results. Evaluation is "
        "cached per geometry; fits are slower than the analytic F–µ–F components."
    ),
    "DipolarPairField": (
        "Use for a muon dipolar-coupled to a single spin-½ nucleus when it is preferable to fit the dipolar "
        "field at the muon directly rather than assume a nucleus and distance: ω_d = γµB_dip. The zero-field "
        "polycrystalline polarization is the entangled two-spin form, ⅙[1 + e^(−λ_T·t)(2cos(ω_d t/2) + "
        "cos(ω_d t) + 2cos(3ω_d t/2))]. The transverse damping λ_T applies only to the oscillating 5/6 part, "
        "modelling weaker couplings to more distant nuclei, while the non-oscillating ⅙ component is "
        "preserved. Use MuF (fluorine), ProtonDipole, or ElectronDipole to parameterize by distance instead."
    ),
    "ProtonDipole": (
        "Use for muon stopping sites adjacent to a single dominant proton — hydroxyl groups, hydrides, or "
        "water of crystallization — where the zero-field signal shows the characteristic two-spin beat "
        "pattern. Identical physics to MuF but with the proton gyromagnetic ratio: the fitted r_µH is the "
        "muon–proton distance through the r⁻³ dipolar coupling. The transverse damping λ_T acts on the "
        "oscillating part only, absorbing weaker couplings to more distant nuclei. Proton moments are roughly "
        "ten times weaker than ¹⁹F at the same distance, so resolvable oscillations require a close, "
        "well-defined µ–H pair."
    ),
    "ElectronDipole": (
        "Use for a muon coupled by the dipolar interaction to a single localized electronic moment at "
        "distance r_µe in zero field — for example a dilute paramagnetic defect or rare-earth ion adjacent to "
        "the muon site — when the moment is static on the muon timescale. Same two-spin form as the nuclear "
        "pairs but with the electron gyromagnetic ratio, so frequencies are about three orders of magnitude "
        "higher at the same distance and r_µe of several Å still gives MHz-scale oscillations. Not "
        "appropriate for muonium (where the contact hyperfine dominates — use the Muonium components) or for "
        "dense magnets (use Oscillatory or Bessel with an internal field)."
    ),
    "GaussianPeak": (
        "Use for a spectral line whose underlying time-domain envelope is Gaussian — a static, dense "
        "(Gaussian) distribution of local fields, as in nuclear-dipole hosts or frozen disorder. The line is "
        "parameterised by its full width at half maximum: a time-domain envelope e^(−(σt)²) transforms to a "
        "Gaussian line of FWHM = 2σ√(ln2)/π, so the fitted width converts back to the relaxation rate via "
        "σ = π·FWHM/(2√(ln2)). The centre ν₀ gives the local field through B₀ = ν₀/(γµ/2π). Peak heights are "
        "in the arbitrary units of the displayed spectrum (they depend on apodization and normalisation), so "
        "physical conclusions should rest on positions and widths rather than absolute heights."
    ),
    "LorentzianPeak": (
        "Use for a spectral line whose underlying time-domain envelope is exponential — dynamic (motionally "
        "narrowed) relaxation, or a dilute (Lorentzian) static field distribution. The line is parameterised "
        "by its full width at half maximum: a time-domain envelope e^(−λt) transforms to a Lorentzian line of "
        "FWHM = λ/π, so the fitted width converts back to the relaxation rate via λ = π·FWHM. The centre ν₀ "
        "gives the local field through B₀ = ν₀/(γµ/2π). Lorentzian tails are heavy: fit windows should extend "
        "several FWHM beyond the peak or the width will be underestimated. A line that is neither Gaussian "
        "nor Lorentzian usually signals overlapping sites — prefer two peaks over one broadened one."
    ),
    "ConstantBackground": (
        "Use for the flat spectral baseline present in essentially every displayed Fourier spectrum, arising "
        "from white noise in the time-domain data and the flat part of any apodization pedestal. Include it "
        "additively in every frequency-domain model unless the baseline has already been subtracted; leaving "
        "it out biases peak heights and widths upward."
    ),
    "LinearBackground": (
        "Use instead of ConstantBackground when the baseline visibly slopes across the fit window — typically "
        "the shoulder of an intense line outside the window or low-frequency leakage from a non-zero mean. "
        "The slope is strongly correlated with the peak parameters in narrow windows, so prefer the constant "
        "form unless the slope is clearly resolved."
    ),
    "DipolarSpinJ": (
        "Use for zero-field precession of a muon coupled to one nucleus of spin J > ½ with both dipolar and "
        "quadrupolar interactions — e.g. µ⁺–⁶³Cu (J = 3/2) or µ⁺–⁹³Nb (J = 9/2) pairs, where the electric "
        "field gradient produced by the muon itself quadrupole-splits the neighbouring nucleus. Implements "
        "the closed-form polycrystalline eigen-solution with dipolar coupling f_dip, quadrupolar splitting "
        "f_quad (sign-sensitive), and nuclear spin J, which should be held fixed at the known value. For "
        "J = ½ it reduces to the two-spin pair (quadrupole inactive). For more than one strongly coupled "
        "nucleus use the F–µ–F family or a dedicated multi-spin model."
    ),
}

#: APS-style literature references shown below the applicability text.
FIT_COMPONENT_REFERENCES: dict[str, tuple[str, ...]] = {
    "StretchedExponential": (
        "Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, and E. J. Ansaldo, "
        "Phys. Rev. B 31, 546 (1985).",
        "I. A. Campbell et al., Phys. Rev. Lett. 72, 1291 (1994).",
    ),
    "RischKehr": ("R. Risch and K. W. Kehr, Phys. Rev. B 46, 5246 (1992).",),
    "Bessel": ("L. P. Le et al., Phys. Rev. B 48, 7284 (1993).",),
    "StaticGKT_ZF": (
        "R. Kubo and T. Toyabe, in Magnetic Resonance and Relaxation, "
        "edited by R. Blinc (North-Holland, Amsterdam, 1967), p. 810.",
        "R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo, "
        "Phys. Rev. B 20, 850 (1979).",
    ),
    "LongitudinalFieldKT": (
        "R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo, "
        "Phys. Rev. B 20, 850 (1979).",
    ),
    "DynamicGaussianKT": (
        "R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo, "
        "Phys. Rev. B 20, 850 (1979).",
    ),
    "DynamicLorentzianKT": (
        "Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, and E. J. Ansaldo, "
        "Phys. Rev. B 31, 546 (1985).",
    ),
    "Keren": ("A. Keren, Phys. Rev. B 50, 10039 (1994).",),
    "Abragam": (
        "A. Abragam, The Principles of Nuclear Magnetism (Oxford University Press, Oxford, 1961).",
    ),
    "GaussianBroadenedKT": ("D. R. Noakes and G. M. Kalvius, Phys. Rev. B 56, 2352 (1997).",),
    "MuoniumTF": ("B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",),
    "MuoniumLowTF": ("B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",),
    "MuoniumZF": ("B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",),
    "MuoniumHighTF": ("B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",),
    "MuoniumHighTFAniso": (
        "B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",
        "E. Roduner and H. Fischer, Chem. Phys. 54, 261 (1981).",
    ),
    "MuoniumLFRelax": (
        "R. F. Kiefl et al., Phys. Rev. Lett. 62, 792 (1989).",
        "R. Kadono et al., Phys. Rev. Lett. 64, 665 (1990).",
        "T. U. Ito and R. Kadono, J. Phys. Soc. Jpn. 94, 064601 (2025).",
    ),
    "MuF": ("T. Lancaster et al., Phys. Rev. Lett. 99, 267601 (2007).",),
    "FmuF_Linear": ("J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986).",),
    "FmuF_General": (
        "T. Lancaster et al., Phys. Rev. Lett. 99, 267601 (2007).",
        "J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986).",
    ),
    "DynamicFmuF": (
        "J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986).",
        "R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and R. Kubo, "
        "Phys. Rev. B 20, 850 (1979).",
    ),
    "FmuF_Triangle": (
        "J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986).",
        "J. M. Wilkinson and S. J. Blundell, Phys. Rev. Lett. 125, 087201 (2020).",
    ),
    "DipolarPairField": ("P. F. Meier, Hyperfine Interact. 18, 427 (1984).",),
    "ProtonDipole": ("P. F. Meier, Hyperfine Interact. 18, 427 (1984).",),
    "ElectronDipole": ("P. F. Meier, Hyperfine Interact. 18, 427 (1984).",),
    "DipolarSpinJ": (
        "M. Celio and P. F. Meier, Hyperfine Interact. 18, 435 (1984).",
        "O. Hartmann, Phys. Rev. Lett. 39, 832 (1977).",
    ),
}

PARAMETER_MODEL_REFERENCES: dict[str, tuple[str, ...]] = {
    "MuRepolarisation": (
        "S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, "
        "Muon Spectroscopy: An Introduction (Oxford University Press, Oxford, 2022).",
        "B. D. Patterson, Rev. Mod. Phys. 60, 69 (1988).",
    ),
    "Redfield": ("N. Bloembergen, E. M. Purcell, and R. V. Pound, Phys. Rev. 73, 679 (1948).",),
    "GaussianLCR": ("S. R. Kreitzman et al., Phys. Rev. Lett. 56, 181 (1986).",),
    "LorentzianLCR": ("S. R. Kreitzman et al., Phys. Rev. Lett. 56, 181 (1986).",),
    "DiffusionLF_1D": ("F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",),
    "DiffusionLF_2D": ("F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",),
    "DiffusionLF_3D": ("F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",),
    "BallisticLF_1D": (
        "F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",
        "B. M. Huddart et al., Phys. Rev. B 103, L060405 (2021).",
    ),
    "BallisticLF_2D": (
        "F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",
        "B. M. Huddart et al., Phys. Rev. B 103, L060405 (2021).",
    ),
    "BallisticLF_3D": (
        "F. L. Pratt, J. Phys.: Conf. Ser. 2462, 012038 (2023).",
        "B. M. Huddart et al., Phys. Rev. B 103, L060405 (2021).",
    ),
}

# All SC_* parameter models share the penetration-depth methodology references;
# the d-wave/interpolation family additionally cites Carrington and Manzano.
_SC_BASE_REFERENCES: tuple[str, ...] = (
    "R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).",
)
_SC_CM_REFERENCES: tuple[str, ...] = _SC_BASE_REFERENCES + (
    "A. Carrington and F. Manzano, Physica C 385, 205 (2003).",
)
for _sc_name in (
    "SC_SWave",
    "SC_AnisotropicS_Cos4",
    "SC_SPlusG",
    "SC_PWaveAxial",
    "SC_ExtendedS",
    "SC_AlphaModel",
    "SC_TwoGap_SS",
    "SC_SWave_Q",
    "SC_SPlusG_Q",
):
    PARAMETER_MODEL_REFERENCES[_sc_name] = _SC_BASE_REFERENCES
for _sc_name in ("SC_DWave", "SC_NonmonotonicD", "SC_TwoGap_SD", "SC_DWave_Q"):
    PARAMETER_MODEL_REFERENCES[_sc_name] = _SC_CM_REFERENCES


_GENERIC_APPLICABILITY = (
    "Use this component when its mathematical form matches the expected physics "
    "and the fit remains stable across your selected data range."
)


def get_component_applicability(component_name: str, kind: str | None = None) -> str:
    """Return physical-applicability text for a known component name.

    ``kind`` disambiguates names registered both as a fit component and as a
    parameter-trend model (currently ``Constant``): pass ``"fit"`` or
    ``"parameter_model"`` so the caller's registry wins.  Without ``kind`` the
    parameter-model dictionary is consulted first (historical behaviour).
    """
    if kind == "fit":
        return FIT_COMPONENT_APPLICABILITY.get(component_name, _GENERIC_APPLICABILITY)
    if kind == "parameter_model":
        return PARAMETER_MODEL_APPLICABILITY.get(component_name, _GENERIC_APPLICABILITY)
    if component_name in PARAMETER_MODEL_APPLICABILITY:
        return PARAMETER_MODEL_APPLICABILITY[component_name]
    if component_name in FIT_COMPONENT_APPLICABILITY:
        return FIT_COMPONENT_APPLICABILITY[component_name]
    return _GENERIC_APPLICABILITY


def get_component_references(component_name: str, kind: str | None = None) -> tuple[str, ...]:
    """Return the APS-style reference list for a component (may be empty).

    ``kind`` behaves as in :func:`get_component_applicability`.
    """
    if kind == "fit":
        return FIT_COMPONENT_REFERENCES.get(component_name, ())
    if kind == "parameter_model":
        return PARAMETER_MODEL_REFERENCES.get(component_name, ())
    if component_name in FIT_COMPONENT_REFERENCES:
        return FIT_COMPONENT_REFERENCES[component_name]
    return PARAMETER_MODEL_REFERENCES.get(component_name, ())


def register_component_documentation(
    component_name: str,
    *,
    kind: str,
    applicability: str = "",
    references: tuple[str, ...] = (),
) -> None:
    """Insert documentation for a facade-registered user component.

    ``kind`` is ``"fit"`` or ``"parameter_model"`` (the same registry kinds the
    lookup functions disambiguate by). Empty values are not inserted, so the
    lookups fall back to the generic placeholder text — acceptable for user
    components, which the docs-enforcement tests exempt by their ``user`` flag.
    """
    if kind == "fit":
        applicability_dict, references_dict = (
            FIT_COMPONENT_APPLICABILITY,
            FIT_COMPONENT_REFERENCES,
        )
    elif kind == "parameter_model":
        applicability_dict, references_dict = (
            PARAMETER_MODEL_APPLICABILITY,
            PARAMETER_MODEL_REFERENCES,
        )
    else:
        raise ValueError(f"Unknown documentation kind {kind!r}")
    if applicability:
        applicability_dict[component_name] = str(applicability)
    if references:
        references_dict[component_name] = tuple(str(ref) for ref in references)
