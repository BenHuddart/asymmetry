"""Trace scientific definitions to citations, owning prose and independent cases.

Citation text belongs to component_docs; equations belong to the owning pages.
The cases below establish the stated observable/domain, not whole-family
certification. Registry/composition round trips are deliberately separate.
"""

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

from asymmetry.core.fitting.component_docs import get_component_references
from asymmetry.core.fitting.composite import CATEGORY_REGISTRY, COMPONENTS
from asymmetry.core.fitting.models import MODELS

ROOT = Path(__file__).resolve().parents[2]


class Evidence(NamedTuple):
    anchor: str
    basis: str
    cases: tuple[str, ...]
    domain: str
    definition_only: bool = False
    unresolved_issue: int | None = None


EVIDENCE = {
    "Exponential": Evidence(
        "fit-exponential",
        "Defined exponential envelope; Lambda in inverse microseconds.",
        ("tests/core/test_model_definitions.py::test_relaxation_half_height_landmarks",),
        "Half-life landmark; not proof of a sample's fluctuation mechanism.",
        True,
    ),
    "Gaussian": Evidence(
        "fit-gaussian",
        "Defined Gaussian envelope; sigma differs from Cartesian TF width by sqrt(2).",
        (
            "tests/core/test_fitting_models.py::test_transverse_field_envelope_uses_delta_over_root_two",
        ),
        "Independent Gaussian frequency average and ZF curvature cases; no static ZF recovery.",
        True,
    ),
    "StretchedExponential": Evidence(
        "fit-stretched-exponential",
        "Explicit Kohlrausch envelope; cited applications do not define every beta's mechanism.",
        ("tests/core/test_model_definitions.py::test_stretched_half_is_a_positive_rate_mixture",),
        "Independent Levy-rate integral at beta=1/2; positive rate mixtures require 0<beta<=1.",
    ),
    "RischKehr": Evidence(
        "fit-risch-kehr",
        "Risch-Kehr family; Telling-Kilcoyne Eq.(2) gives the implemented reduced finite-field form.",
        ("tests/core/test_model_definitions.py::test_risch_kehr_is_the_continuous_rate_integral",),
        "Independent positive-rate integral for Gamma*t=0 through 1e4; microscopic transport needs additional assumptions.",
    ),
    "Keren": Evidence(
        "fit-keren",
        "Keren Eq.(4), angular Delta and omega0, inverse-time nu; approximation to dynamic Gaussian KT.",
        ("tests/core/test_dynamic_relaxation.py::test_abragam_and_keren_match_decimal_oracle",),
        "Independent high-precision defining expression; physical approximation is not the full static or renewal model.",
    ),
    "Abragam": Evidence(
        "fit-abragam",
        "Abragam Gaussian stochastic-phase envelope; one transverse frequency component.",
        ("tests/core/test_dynamic_relaxation.py::test_abragam_and_keren_match_decimal_oracle",),
        "Decimal defining-expression oracle, including small rates; not a zero-field isotropic KT model.",
    ),
    "Oscillatory": Evidence(
        "fit-oscillatory",
        "Defined cosine, cyclic frequency in MHz and phase in radians.",
        ("tests/core/test_model_definitions.py::test_oscillation_quarter_period_and_phase",),
        "Quarter-period and quadrature detector-phase landmarks; amplitude in percent.",
        True,
    ),
    "OscillatoryField": Evidence(
        "fit-oscillatory-field",
        "Defined cosine with frequency=(gamma_mu/2pi)*field, Gauss to Tesla explicit.",
        ("tests/core/test_model_definitions.py::test_oscillation_quarter_period_and_phase",),
        "Field/frequency unit correspondence; shared constants are not independently remeasured.",
        True,
    ),
    "Bessel": Evidence(
        "fit-bessel",
        "Signed arcsine-field definition, supported by cited SDW applications.",
        ("tests/gui/test_wimda_parity_components.py::test_bessel_matches_overhauser_integral",),
        "Independent arcsine-field integral at phase zero; shifted argument is a fitting convention.",
    ),
    "OverhauserPowder": Evidence(
        "fit-overhauser-powder",
        "Signed arcsine transform with isotropic 1/3 and 2/3 weights; separate empirical damping.",
        (
            "tests/core/test_overhauser_powder.py::test_undamped_precessing_part_is_the_overhauser_transform",
        ),
        "Independent undamped field average; a Bessel-like line does not establish an ordering wavevector.",
    ),
    "OverhauserPowderCutoff": Evidence(
        "fit-overhauser-powder-two-cutoff",
        "Amato Eqs.(15)-(17): shifted arcsine approximation, not exact rotating-ellipse density Eq.(10).",
        (
            "tests/core/test_overhauser_powder.py::test_closed_form_approximates_the_exact_two_cutoff_transform",
            "tests/core/test_model_definitions.py::test_offset_arcsine_components_match_independent_average",
        ),
        "Quantified approximation against the elliptical-density integral; exact only for the defined offset arcsine geometry.",
    ),
    "OverhauserPowderCentre": Evidence(
        "fit-overhauser-powder-two-cutoff",
        "Amato Eqs.(15)-(17), center/halfwidth parameterization of the same shifted arcsine form.",
        (
            "tests/core/test_model_definitions.py::test_offset_arcsine_components_match_independent_average",
        ),
        "Independent fixed-direction offset arcsine average; not an exact rotating-ellipse transform.",
    ),
    "HelicalPowder": Evidence(
        "fit-helical-powder",
        "Amato Eq.(10) elliptical-magnitude density; isotropic static spin-rotation average.",
        ("tests/core/test_helical.py::test_powder_line_is_the_helix_phase_average",),
        "Independent elliptical-vector average at ratios .05-.999 and phases 0,.7; phenomenological damping is separate.",
    ),
    "HelicalCrystal": Evidence(
        "fit-helical-crystal",
        "Explicit elliptical-vector geometry with matching initial/detector axis; source supports field family.",
        ("tests/core/test_helical.py::test_crystal_line_is_the_helix_phase_average",),
        "Independent vector spin projection for selected ratios and orientations; orientation and field-magnitude weights both matter.",
    ),
    "VortexLattice": Evidence(
        "fit-vortex-lattice",
        "Defined Gaussian-cutoff modified-London shape, rescaled to Brandt Eq.(13) second moment.",
        ("tests/core/test_sc_vl_lineshape.py::test_fft_build_matches_direct_reciprocal_sum",),
        "Independent reciprocal field sum only; full time-envelope accuracy unresolved, histogram error exceeds 2e-3 on a sampled case.",
        unresolved_issue=657,
    ),
    "VortexLatticePowder": Evidence(
        "fit-vortex-lattice",
        "Same defined shape with effective lambda=3^(1/4)*lambda_ab; Pratt Eq.(3) supports width conversion.",
        (
            "tests/core/test_sc_vl_lineshape.py::test_fft_build_matches_direct_reciprocal_sum",
            "tests/core/test_sc_vl_lineshape.py::test_relaxation_matches_full_spatial_average",
        ),
        "Reciprocal field sum plus powder-kernel histogram comparison at 2e-3, not 1e-5; not a physical powder orientation average.",
    ),
    "StaticGKT_ZF": Evidence(
        "fit-static-gkt-zf",
        "Kubo-Toyabe/Hayano static isotropic Gaussian Cartesian fields; Delta is angular component width.",
        (
            "tests/core/test_fitting_models.py::test_static_gkt_is_an_isotropic_gaussian_field_average",
        ),
        "Independent classical 3D Gaussian spin rotation; normalized before amplitude/baseline.",
    ),
    "LongitudinalFieldKT": Evidence(
        "fit-lf-kubo-toyabe",
        "Hayano static Gaussian longitudinal-field expression; field in Gauss.",
        (
            "tests/core/test_static_gaussian_accuracy.py::test_gaussian_spin_rotation_average",
            "tests/core/test_static_gaussian_accuracy.py::test_extended_time_adaptive_integral",
        ),
        "Independent Gaussian spin rotations and adaptive time integrals; owning page gives sampled dimensionless domain.",
    ),
    "DynamicGaussianKT": Evidence(
        "fit-dynamic-gaussian-kt",
        "Hayano strong-collision renewal equation applied to the static Gaussian field kernel.",
        ("tests/core/test_dynamic_kt_accuracy.py::test_independent_scalar_renewal_refinement",),
        "Independent scalar renewal solve/refinement; complete Poisson resets, not every microscopic fluctuation process.",
    ),
    "DynamicLorentzianKT": Evidence(
        "fit-dynamic-lorentzian-kt",
        "Explicit strong-collision composition of isotropic Lorentzian static kernel; Uemura supplies static-field context.",
        (
            "tests/core/test_dynamic_kt_accuracy.py::test_independent_scalar_renewal_refinement",
            "tests/core/test_dynamic_kt_accuracy.py::test_lorentzian_weak_field_against_published_expression",
        ),
        "Independent published static expression and renewal solve; not an average of site-dependent Gaussian dynamic kernels.",
    ),
    "GaussianBroadenedKT": Evidence(
        "fit-gaussian-broadened-kt",
        "Noakes-Kalvius ZF motivation; explicit Gaussian-width average extends the static LF kernel.",
        (
            "tests/core/test_gaussian_broadened_accuracy.py::test_continuous_zero_field_average_through_long_tail",
            "tests/core/test_gaussian_broadened_accuracy.py::test_independent_long_time_finite_field_average",
        ),
        "Gaussian in width, not variance; independent continuous ZF and nested LF integral cases, sampled limits in owning prose.",
    ),
    "MuoniumTF": Evidence(
        "fit-muonium-tf",
        "Explicit isotropic electron-muon Hamiltonian; Patterson provides muonium spectroscopy context.",
        ("tests/core/test_muonium_hamiltonian.py::test_full_tf_matches_normalized_density_matrix",),
        "Unpolarized electron and initially transverse polarized muon, phase zero; five field/hyperfine pairs, t<=2/A_hf.",
    ),
    "MuoniumLowTF": Evidence(
        "fit-muonium-low-tf",
        "Defined selected two-transition approximation of isotropic muonium; exact Breit-Rabi frequencies.",
        (
            "tests/core/test_muonium_hamiltonian.py::test_low_and_high_field_pairs_match_independent_transition_selection",
        ),
        "Selected transition weights and frequencies only; low pair carries half the initial polarization, not the full signal.",
    ),
    "MuoniumZF": Evidence(
        "fit-muonium-zf",
        "Defined three-line axial component; independent Hamiltonian derivation, no verified exact Patterson-equation attribution.",
        (
            "tests/core/test_muonium_hamiltonian.py::test_zf_three_lines_omit_additional_static_sixth",
        ),
        "Phase=cutoff=0 gives full static powder polarization minus 1/6; tensor(A-D,A-D,A+2D). Filtering is empirical.",
    ),
    "MuoniumHighTF": Evidence(
        "fit-muonium-high-tf",
        "Exact isotropic transition frequencies with equal high-field weights; Patterson spectroscopy context.",
        (
            "tests/core/test_muonium_hamiltonian.py::test_low_and_high_field_pairs_match_independent_transition_selection",
        ),
        "Independent selected-transition comparison at high field; dropping other transitions is a physical approximation.",
    ),
    "MuoniumHighTFAniso": Evidence(
        "fit-muonium-high-tf-aniso",
        "Explicit axial Hamiltonian tensor(A-D/2,A-D/2,A+D); cited papers support anisotropic muonium family.",
        (
            "tests/gui/test_wimda_parity_components.py::test_high_tf_aniso_pair_frequencies_match_exact_hamiltonian",
        ),
        "Independent per-angle gap construction at three angles; equal weights/two-line selection and 32-node powder integration are not certified globally.",
    ),
    "MuoniumLFRelax": Evidence(
        "fit-muonium-lf-relax",
        "Kadono-Ito Eq.(22) intratriplet rate envelope; Eq.(19) repolarization amplitude is separate.",
        (
            "tests/core/test_muonium_hamiltonian.py::test_lf_rate_matches_eq22_using_independent_energy_gap",
        ),
        "Independent Hamiltonian gap at four fields; delta_ex angular inverse-time amplitude, tau_c microseconds, no mixing prefactor.",
    ),
    "MuF": Evidence(
        "fit-muf",
        "Lancaster Eqs.(1),(2): point-dipole Hamiltonian and spectral polarization trace.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_mu_f_and_linear_fmuf_match_independent_spin_trace",
        ),
        "Independent two-spin powder trace, three distances, t=0..8 microseconds; shared constants.",
    ),
    "FmuF_Linear": Evidence(
        "fit-fmuf-linear",
        "Brewer collinear family; explicit symmetric two-neighbor dipole Hamiltonian.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_mu_f_and_linear_fmuf_match_independent_spin_trace",
        ),
        "Independent three-spin powder trace without nuclear-nuclear interaction; equal distances, no field or dynamics.",
    ),
    "FmuF_General": Evidence(
        "fit-fmuf-general",
        "Lancaster Eqs.(1),(2), bent two-fluorine point-dipole geometry including F-F coupling.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_general_and_triangle_geometries_match_independent_spin_trace",
        ),
        "Independent crystal-frame three-axis powder identity at three angles and unequal distances; no Euler grid reused.",
    ),
    "FmuF_Triangle": Evidence(
        "fit-fmuf-triangle",
        "Explicit four-spin all-pairs point-dipole composition; cited studies motivate additional fluorines.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_general_and_triangle_geometries_match_independent_spin_trace",
        ),
        "Independent 16-state powder trace, three angles, t=0..8; source-family support is not an exact geometry attribution.",
    ),
    "DynamicFmuF": Evidence(
        "fit-dynamic-fmuf",
        "Defined Hayano-style strong-collision renewal composition of Brewer's static collinear kernel.",
        (
            "tests/gui/test_wimda_parity_components.py::test_dynamic_fmuf_closed_form_matches_reference_solver",
        ),
        "Independent fine-grid Volterra solver versus eigenmodes; complete Poisson resets do not automatically represent irreversible escape.",
    ),
    "DipolarPairField": Evidence(
        "fit-dipolar-pair-field",
        "Meier spin-pair context; explicit point-dipole Hamiltonian with omega_d=gamma_mu*B_dip.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_distance_and_field_pair_parameterizations_match_spin_trace",
        ),
        "Independent undamped two-spin powder trace; transverse-only exponential damping is an empirical extension.",
    ),
    "ProtonDipole": Evidence(
        "fit-proton-dipole",
        "Same defined spin-pair Hamiltonian with proton gyromagnetic ratio and distance in angstroms.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_distance_and_field_pair_parameterizations_match_spin_trace",
        ),
        "Independent SI coupling conversion and spin trace at zero damping; constants shared.",
    ),
    "ElectronDipole": Evidence(
        "fit-electron-dipole",
        "Same defined spin-pair Hamiltonian for one isolated electronic spin-1/2 dipole; no contact coupling.",
        (
            "tests/core/test_dipolar_hamiltonians.py::test_distance_and_field_pair_parameterizations_match_spin_trace",
        ),
        "Independent SI coupling conversion and spin trace at zero damping; not contact-hyperfine muonium.",
    ),
    "DipolarSpinJ": Evidence(
        "fit-dipolar-spin-j",
        "Explicit axial dipolar-plus-quadrupolar Hamiltonian; Celio-Meier/Hartmann support physical family.",
        ("tests/gui/test_wimda_parity_components.py::test_spin_j_matches_exact_diagonalization",),
        "Independent full matrix versus block solution for J=.5,1,1.5,2.5,4.5 and signed quadrupole; axial I_z^2 convention only.",
    ),
    "Constant": Evidence(
        "fit-constant",
        "Defined additive time-independent asymmetry in percent.",
        ("tests/core/test_model_definitions.py::test_background_intercepts_and_slope",),
        "Signed intercept and arbitrary coordinate landmarks, not an instrument-background calibration.",
        True,
    ),
    "GaussianPeak": Evidence(
        "fit-gaussian-peak",
        "Defined height/center/FWHM Gaussian, frequency in MHz and height in spectrum units.",
        (
            "tests/core/test_model_definitions.py::test_spectral_peak_height_width_and_integrated_area",
        ),
        "Independent half-height and integrated-area checks; finite-window Fourier data may not be an ideal Gaussian.",
        True,
    ),
    "LorentzianPeak": Evidence(
        "fit-lorentzian-peak",
        "Defined height/center/FWHM Lorentzian, frequency in MHz.",
        (
            "tests/core/test_model_definitions.py::test_spectral_peak_height_width_and_integrated_area",
        ),
        "Independent half-height and integrated-area checks; apodization and finite-window effects remain separate.",
        True,
    ),
    "ConstantBackground": Evidence(
        "fit-constant-background",
        "Defined additive constant in spectrum units.",
        ("tests/core/test_model_definitions.py::test_background_intercepts_and_slope",),
        "Signed intercept landmarks; no physical noise-spectrum assumption required.",
        True,
    ),
    "LinearBackground": Evidence(
        "fit-linear-background",
        "Defined intercept+slope*frequency; slope in spectrum units/MHz.",
        ("tests/core/test_model_definitions.py::test_background_intercepts_and_slope",),
        "Independent intercept and secant slope at signed coordinates; empirical local baseline.",
        True,
    ),
}

MODEL_FAMILIES = {
    "ExponentialRelaxation": "Exponential",
    "GaussianRelaxation": "Gaussian",
    "Oscillatory": "Oscillatory",
    "StretchedExponential": "StretchedExponential",
    "StaticGKT_ZF": "StaticGKT_ZF",
    "LFKuboToyabe": "LongitudinalFieldKT",
    "DynamicGaussianKT": "DynamicGaussianKT",
    "DynamicLorentzianKT": "DynamicLorentzianKT",
    "Keren": "Keren",
    "Abragam": "Abragam",
}


def test_every_builtin_has_an_evidence_contract():
    builtins = {name for name, definition in COMPONENTS.items() if not definition.user}
    assert set(EVIDENCE) == builtins
    assert set(MODEL_FAMILIES) == set(MODELS)
    assert set(MODEL_FAMILIES.values()) <= set(EVIDENCE)


def test_unresolved_components_remain_within_the_reviewed_allowance():
    unresolved = {
        name for name, evidence in EVIDENCE.items() if evidence.unresolved_issue is not None
    }
    assert unresolved <= {"VortexLattice"}
    for name in unresolved:
        assert EVIDENCE[name].unresolved_issue > 0


@pytest.mark.parametrize("name,evidence", EVIDENCE.items(), ids=EVIDENCE)
def test_evidence_targets_resolve(name, evidence):
    page = (
        ROOT
        / "docs/reference/fit_functions"
        / f"{CATEGORY_REGISTRY[COMPONENTS[name].category]}.rst"
    )
    assert f".. _{evidence.anchor}:" in page.read_text()
    assert evidence.basis and evidence.domain and evidence.cases
    if not evidence.definition_only:
        assert get_component_references(name, kind="fit"), name
    for node in evidence.cases:
        module, function = node.split("::")
        tree = ast.parse((ROOT / module).read_text())
        assert any(
            isinstance(item, ast.FunctionDef) and item.name == function for item in tree.body
        ), node
