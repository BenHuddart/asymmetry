"""Tests for the WiMDA fit-function-parity components.

Verification criteria from docs/porting/wimda-fit-function-parity/:
limit identities against established components, t = 0 normalisation,
registry/doc hygiene, serialization round-trips, and golden checks against
independent evaluations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from asymmetry.core.fitting.component_docs import FIT_COMPONENT_APPLICABILITY
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.models import (
    bessel_oscillation,
    gaussian_broadened_kt,
    longitudinal_field_kubo_toyabe,
    risch_kehr,
)
from asymmetry.core.fitting.muon_fluorine.polarization import (
    dynamic_fmuf_polarization,
    fmuf_triangle_polarization,
    general_fmuf_polarization,
    linear_fmuf_polarization,
)
from asymmetry.core.fitting.muonium import (
    VACUUM_MUONIUM_A_HF_MHZ,
    _tf_levels,
    high_tf_muonium,
    high_tf_muonium_aniso,
    muonium_lf_relaxation,
)
from asymmetry.core.fitting.nuclear_dipole import (
    dipolar_pair_field,
    dipolar_pair_kernel,
    dipolar_spin_j,
    electron_dipole,
    proton_dipole,
)

NEW_COMPONENTS = [
    "RischKehr",
    "Bessel",
    "GaussianBroadenedKT",
    "MuoniumHighTF",
    "MuoniumHighTFAniso",
    "MuoniumLFRelax",
    "DynamicFmuF",
    "FmuF_Triangle",
    "DipolarPairField",
    "ProtonDipole",
    "ElectronDipole",
    "DipolarSpinJ",
]

T = np.linspace(0.0, 16.0, 321)


# --- registry and documentation hygiene -------------------------------------


def test_new_components_registered_with_metadata() -> None:
    for name in NEW_COMPONENTS:
        definition = COMPONENTS[name]
        assert definition.domain == "time"
        assert definition.category in {
            "Relaxation",
            "Oscillation",
            "Kubo-Toyabe",
            "Muonium",
            "Nuclear dipolar",
        }
        assert definition.formula_template
        assert definition.latex_equation
        assert set(definition.param_defaults) == set(definition.param_names)
        for param in definition.param_names:
            assert param in definition.param_info


def test_new_components_have_applicability_docs() -> None:
    for name in NEW_COMPONENTS:
        text = FIT_COMPONENT_APPLICABILITY[name]
        assert len(text) > 100


def test_new_components_have_aps_style_references() -> None:
    import re

    from asymmetry.core.fitting.component_docs import get_component_references

    for name in NEW_COMPONENTS:
        refs = get_component_references(name)
        assert refs, name
        for ref in refs:
            # APS style ends with "(year)." (journal) or "year)." (book).
            assert re.search(r"\(.*\d{4}\)\.$", ref), ref


def test_new_components_finite_and_normalised_at_defaults() -> None:
    for name in NEW_COMPONENTS:
        definition = COMPONENTS[name]
        y = definition.function(T, **definition.param_defaults)
        assert y.shape == T.shape, name
        assert np.all(np.isfinite(y)), name
        # All new components are cosine-like at phase 0 / pure relaxation:
        # the t = 0 value equals the amplitude default (A = 25).
        assert y[0] == pytest.approx(25.0, rel=1e-9), name


def test_new_components_serialization_round_trip() -> None:
    for name in NEW_COMPONENTS:
        model = CompositeModel([name, "Constant"], ["+"])
        restored = CompositeModel.from_dict(model.to_dict())
        assert restored.component_names == model.component_names


def test_fixed_by_default_params_resolve_through_param_mapping() -> None:
    """J_spin (piecewise-constant in the fit) and MuoniumLFRelax's A_hf start
    fixed; duplicated components expose their indexed names."""
    assert CompositeModel(["DipolarSpinJ"]).fixed_by_default_params() == {"J_spin"}
    assert CompositeModel(["MuoniumLFRelax"]).fixed_by_default_params() == {"A_hf"}
    duplicated = CompositeModel(["DipolarSpinJ", "DipolarSpinJ"], ["+"])
    assert duplicated.fixed_by_default_params() == {"J_spin_1", "J_spin_2"}
    assert CompositeModel(["Exponential"]).fixed_by_default_params() == set()


def test_indexed_subscripted_params_render_in_mathtext() -> None:
    """Duplicated components index their params (lambda_T -> lambda_T_2); the
    label machinery must merge subscripts (lambda_{T,2}) — a naive suffix
    produces a double subscript that matplotlib rejects at plot-label time."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.mathtext import MathTextParser

    from asymmetry.core.fitting.parameters import PARAM_INFO_REGISTRY, get_param_info

    parser = MathTextParser("agg")
    for base in PARAM_INFO_REGISTRY:
        info = get_param_info(f"{base}_2")
        parser.parse(info.latex.strip())  # raises ValueError on bad mathtext


# --- RischKehr ---------------------------------------------------------------


def test_risch_kehr_matches_erfc_form_and_asymptote() -> None:
    gamma = 1.3
    g = gamma * T[1:]
    direct = np.exp(g) * erfc(np.sqrt(g))  # safe for these moderate g values
    assert np.allclose(risch_kehr(T[1:], gamma), direct, rtol=1e-12)

    t_large = np.array([1.0e4])
    tail = risch_kehr(t_large, gamma)[0]
    assert tail == pytest.approx(1.0 / np.sqrt(np.pi * gamma * 1.0e4), rel=1e-2)
    # Continuity across WiMDA's branch point Gamma*t = 20 is inherent to erfcx;
    # check smoothness there anyway.
    near = risch_kehr(np.array([19.99, 20.01]), 1.0)
    assert abs(near[1] - near[0]) < 1e-4


def test_risch_kehr_negative_rate_uses_magnitude() -> None:
    assert np.allclose(risch_kehr(T, -2.0), risch_kehr(T, 2.0))


# --- Bessel ------------------------------------------------------------------


def test_bessel_matches_overhauser_integral() -> None:
    freq = 0.7  # MHz
    omega = 2.0 * np.pi * freq
    t = np.linspace(0.0, 8.0, 50)
    # MS-Intro eqn 6.45: P(t) = (1/pi) int_-B1^B1 dB cos(gamma B t)/sqrt(B1^2-B^2)
    phi = np.linspace(0.0, np.pi, 20001)  # B = B1 cos(phi) substitution
    integral = np.trapezoid(np.cos(omega * np.outer(t, np.cos(phi))), phi, axis=1) / np.pi
    assert np.allclose(bessel_oscillation(t, freq), integral, atol=1e-10)
    assert bessel_oscillation(np.array([0.0]), freq)[0] == 1.0


# --- Gaussian-broadened KT ---------------------------------------------------


def test_gbkt_zero_width_reduces_to_lf_kt() -> None:
    for b_l in (0.0, 30.0):
        broadened = gaussian_broadened_kt(T, 0.5, b_l, 0.0)
        static = longitudinal_field_kubo_toyabe(T, 1.0, 0.5, b_l)
        assert np.allclose(broadened, static, atol=1e-12)


def test_gbkt_matches_brute_force_average() -> None:
    width = 0.3
    nodes = np.linspace(-5.0, 5.0, 2001)
    pdf = np.exp(-0.5 * nodes**2) / np.sqrt(2.0 * np.pi)
    brute = np.zeros_like(T)
    for x, p in zip(nodes, pdf, strict=True):
        brute += (
            p
            * (nodes[1] - nodes[0])
            * longitudinal_field_kubo_toyabe(T, 1.0, abs(0.5 * (1.0 + width * x)), 30.0)
        )
    assert np.allclose(gaussian_broadened_kt(T, 0.5, 30.0, width), brute, atol=1e-4)


def test_gbkt_is_continuous_in_width_and_accurate_at_large_delta() -> None:
    t = np.linspace(0.0, 32.0, 641)
    # Continuity: w_rel -> 0 must approach the single-width LF KT smoothly
    # (the previous grid-interpolated implementation had a ~2.5% branch step).
    for b_l in (0.0, 30.0, 300.0):
        broadened = gaussian_broadened_kt(t, 0.5, b_l, 1e-8)
        single = longitudinal_field_kubo_toyabe(t, 1.0, 0.5, b_l)
        assert np.allclose(broadened, single, atol=1e-9), f"B_L={b_l}"

    # Accuracy at large Delta*tmax, where the old fixed 800-point grid lost
    # 2.5% around the dip.
    width = 0.3
    nodes = np.linspace(-6.0, 6.0, 2001)
    pdf = np.exp(-0.5 * nodes**2) / np.sqrt(2.0 * np.pi)
    brute = np.zeros_like(t)
    for x, p in zip(nodes, pdf, strict=True):
        brute += (
            p
            * (nodes[1] - nodes[0])
            * longitudinal_field_kubo_toyabe(t, 1.0, abs(8.0 * (1.0 + width * x)), 0.0)
        )
    assert np.allclose(gaussian_broadened_kt(t, 8.0, 0.0, width), brute, atol=2e-3)


def test_gbkt_broadening_softens_the_dip() -> None:
    sharp = gaussian_broadened_kt(T, 0.5, 0.0, 0.0)
    broad = gaussian_broadened_kt(T, 0.5, 0.0, 0.4)
    assert broad.min() > sharp.min()


# --- muonium -----------------------------------------------------------------


def test_high_tf_pair_frequencies_sum_to_hyperfine() -> None:
    a_hf = VACUUM_MUONIUM_A_HF_MHZ
    _d, e1, e2, e3, e4 = _tf_levels(3000.0, a_hf)
    assert abs(e1 - e2) + abs(e3 - e4) == pytest.approx(a_hf, rel=1e-12)


def test_high_tf_aniso_reduces_to_isotropic_pair() -> None:
    iso = high_tf_muonium(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.3)
    aniso = high_tf_muonium_aniso(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.0, 0.3)
    assert np.allclose(iso, aniso, atol=1e-10)


def test_high_tf_aniso_pair_frequencies_match_exact_hamiltonian() -> None:
    """The per-orientation pair frequencies must match the exact 4-level
    anisotropic-muonium Hamiltonian (axial hyperfine tensor at angle theta to
    the field) to first order in D — in particular the *direction* of the
    nu_12 shift, which a literal port of WiMDA's ±d/2 convention gets wrong.
    """
    from asymmetry.core.fitting.muonium import G_E_MHZ_PER_G, G_MU_MHZ_PER_G

    field = 3000.0
    a_hf = VACUUM_MUONIUM_A_HF_MHZ
    big_d = 20.0

    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    spins = (sx, sy, sz)
    id2 = np.eye(2)

    for theta in (0.0, np.deg2rad(54.7356), np.deg2rad(90.0)):
        # Axial hyperfine tensor with symmetry axis at angle theta in the x-z
        # plane: A = A_iso * 1 + diag(-D/2, -D/2, D) rotated by theta about y.
        n_axis = np.array([np.sin(theta), 0.0, np.cos(theta)])
        a_tensor = (a_hf - 0.5 * big_d) * np.eye(3) + 1.5 * big_d * np.outer(n_axis, n_axis)

        h = G_E_MHZ_PER_G * field * np.kron(sz, id2) - G_MU_MHZ_PER_G * field * np.kron(id2, sz)
        for i in range(3):
            for j in range(3):
                h = h + a_tensor[i, j] * np.kron(spins[i], spins[j])
        evals = np.sort(np.linalg.eigvalsh(h))

        # The two muon-spin-flip (high-field-observable) transitions are the
        # splittings within each electron-spin manifold: the two smallest
        # level gaps between adjacent eigenvalues pair up as nu_12 and nu_34.
        gaps = np.diff(evals)
        nu_pair_exact = np.sort(gaps)[:2]

        # The component's solver selects the pair by sigma_x^mu amplitude —
        # an independent construction that must agree with the gap analysis.
        from asymmetry.core.fitting.muonium import _aniso_pair_frequencies

        f_lo, f_hi = _aniso_pair_frequencies(field, a_hf, big_d, np.array([np.cos(theta)]))
        nu_pair_model = np.sort([f_lo[0], f_hi[0]])

        assert np.allclose(nu_pair_exact, nu_pair_model, atol=1e-8), (
            f"theta={np.degrees(theta):.1f}: exact {nu_pair_exact}, model {nu_pair_model}"
        )

        # The pair sum must track the secular effective coupling to first
        # order: A_eff = A_hf + (D/2)(3cos^2 theta - 1).
        d_shift = 0.5 * big_d * (3.0 * np.cos(theta) ** 2 - 1.0)
        assert np.sum(nu_pair_model) == pytest.approx(a_hf + d_shift, abs=0.1)


def test_high_tf_aniso_pair_sum_tracks_effective_hyperfine() -> None:
    """With the exact treatment, each orientation's pair sum equals
    A_eff(theta) = A_hf + d — for theta = 0 (d = D) the spectrum must contain
    frequencies summing above A_hf, not stay pinned at A_hf."""
    from asymmetry.core.fitting.muonium import _tf_levels

    big_d = 50.0
    _d, e1, e2, e3, e4 = _tf_levels(3000.0, VACUUM_MUONIUM_A_HF_MHZ + big_d)
    assert abs(e1 - e2) + abs(e3 - e4) == pytest.approx(VACUUM_MUONIUM_A_HF_MHZ + big_d, rel=1e-12)


def test_high_tf_aniso_powder_average_damps_pair() -> None:
    iso = high_tf_muonium(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 0.0)
    aniso = high_tf_muonium_aniso(T, 3000.0, VACUUM_MUONIUM_A_HF_MHZ, 25.0, 0.0)
    # The distribution of anisotropy shifts dephases the pair envelope.
    assert np.max(np.abs(aniso[T > 4.0])) < np.max(np.abs(iso[T > 4.0]))


def test_muonium_lf_relaxation_quenches_with_field() -> None:
    low = muonium_lf_relaxation(T, 0.5, 0.01, 10.0, VACUUM_MUONIUM_A_HF_MHZ)
    high = muonium_lf_relaxation(T, 0.5, 0.01, 5000.0, VACUUM_MUONIUM_A_HF_MHZ)
    assert low[-1] < high[-1] <= 1.0
    assert low[0] == 1.0


# --- nuclear dipolar ---------------------------------------------------------


def _exact_spin_j_polarization(t: np.ndarray, f_dip: float, f_quad: float, J: float) -> np.ndarray:
    """Exact-diagonalization reference for the muon + spin-J ZF problem.

    H = w_d (S·I − 3 S_z I_z) + w_q I_z², muon spin-1/2; polycrystalline
    average (P_z + 2 P_x)/3 (exact for this axially symmetric Hamiltonian).
    """
    wd = 2.0 * np.pi * f_dip
    wq = 2.0 * np.pi * f_quad
    dim_n = int(round(2.0 * J)) + 1
    m = J - np.arange(dim_n)  # m = J, J-1, ..., -J
    iz = np.diag(m)
    ladder = np.sqrt(J * (J + 1.0) - m[1:] * (m[1:] + 1.0))
    iplus = np.zeros((dim_n, dim_n))
    iplus[np.arange(dim_n - 1), np.arange(1, dim_n)] = ladder
    ix = 0.5 * (iplus + iplus.T)
    iy = -0.5j * (iplus - iplus.T)

    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    id_n = np.eye(dim_n)
    id_mu = np.eye(2)

    h = wd * (
        np.kron(sx, ix) + np.kron(sy, iy) - 2.0 * np.kron(sz, iz)  # S·I − 3 S_z I_z
    ) + wq * np.kron(id_mu, iz @ iz)
    evals, evecs = np.linalg.eigh(h)
    dim = 2 * dim_n

    out = np.zeros((2, t.size))
    for k, sigma in enumerate((2.0 * np.kron(sz, id_n), 2.0 * np.kron(sx, id_n))):
        sig_eig = evecs.conj().T @ sigma @ evecs
        weights = (np.abs(sig_eig) ** 2) / dim
        omega = evals[:, None] - evals[None, :]
        out[k] = np.tensordot(weights, np.cos(np.multiply.outer(omega, t)), axes=2)
    pz, px = out
    return (pz + 2.0 * px) / 3.0


@pytest.mark.parametrize("J", [0.5, 1.0, 1.5, 2.5, 4.5])
@pytest.mark.parametrize("f_quad", [0.0, 0.1, -0.3])
def test_spin_j_matches_exact_diagonalization(J: float, f_quad: float) -> None:
    t = np.linspace(0.0, 16.0, 161)
    closed_form = dipolar_spin_j(t, 0.2, f_quad, J)
    exact = _exact_spin_j_polarization(t, 0.2, f_quad, J)
    assert np.allclose(closed_form, exact, atol=1e-10), (
        f"J={J}, f_quad={f_quad}: max dev {np.max(np.abs(closed_form - exact))}"
    )


def test_spin_half_reduces_to_meier_pair() -> None:
    kernel = dipolar_pair_kernel(T, 2.0 * np.pi * 0.2, 0.0)
    spin_j = dipolar_spin_j(T, 0.2, 0.0, 0.5)
    assert np.allclose(kernel, spin_j, atol=1e-12)


def test_spin_j_quadrupole_inactive_for_spin_half() -> None:
    without = dipolar_spin_j(T, 0.2, 0.0, 0.5)
    with_quad = dipolar_spin_j(T, 0.2, 0.7, 0.5)
    # A spin-1/2 nucleus has no quadrupole moment; the closed form keeps the
    # 1<->2 splitting quadrupole-free (the spectator levels shift together).
    assert np.allclose(without, with_quad, atol=1e-12)


def test_dipolar_pair_transverse_damping_preserves_static_sixth() -> None:
    damped = dipolar_pair_field(np.array([60.0]), 10.0, 5.0)[0]
    assert damped == pytest.approx(1.0 / 6.0, abs=1e-12)


def test_dipole_pair_frequency_scales_with_gyromagnetic_ratio() -> None:
    # At equal distance the electron pair beats ~660x faster than the proton
    # pair; check via the first zero-crossing ordering instead of exact ratios.
    t_short = np.linspace(0.0, 0.02, 2000)
    e = electron_dipole(t_short, 2.0, 0.0)
    p = proton_dipole(t_short, 2.0, 0.0)
    assert e.min() < 0.9  # electron pair has already oscillated
    assert p.min() > 0.999  # proton pair has barely moved


# --- F-mu-F dynamics and triangle ---------------------------------------------


def test_dynamic_fmuf_static_limit() -> None:
    static = linear_fmuf_polarization(T, 1.17)
    dynamic = dynamic_fmuf_polarization(T, 1.17, 0.0)
    assert np.allclose(static, dynamic, atol=1e-12)


def test_dynamic_fmuf_slow_fluctuation_damps_tail() -> None:
    static = linear_fmuf_polarization(T, 1.17)
    dynamic = dynamic_fmuf_polarization(T, 1.17, 0.3)
    assert dynamic[0] == pytest.approx(1.0)
    assert dynamic[-1] < static[-1]
    assert np.all(dynamic <= 1.0 + 1e-9)


def test_dynamic_fmuf_fast_fluctuation_motional_narrowing() -> None:
    from asymmetry.core.fitting.muon_fluorine.dipolar import omega_d_mu_f_rad_per_us

    omega_d = omega_d_mu_f_rad_per_us(1.17)
    nu = 500.0  # far above the solver range: Abragam-form branch
    narrowed = np.exp(-2.0 * omega_d**2 * T / nu)
    # The Abragam-form branch approaches the bare narrowing exponential for
    # nu*t >> 1 (and is more accurate at early times).
    result = dynamic_fmuf_polarization(T, 1.17, nu)
    assert np.allclose(result[T > 0.1], narrowed[T > 0.1], atol=2e-3)
    assert result[0] == pytest.approx(1.0)


def test_dynamic_fmuf_branch_seam_is_small() -> None:
    """The solver/interpolation crossover must not put a step in the model.

    The previous fixed switch at nu = 12 caused 2.5-30 % discontinuities; the
    crossover now follows nu = 12*omega_d (where the solver and the
    Abragam-form interpolation are both accurate), keeping the seam ~1 % or
    below across mu-F distances.
    """
    from asymmetry.core.fitting.muon_fluorine.dipolar import omega_d_mu_f_rad_per_us
    from asymmetry.core.fitting.muon_fluorine.polarization import (
        _DYN_FMUF_GRID_CAP,
        _DYN_FMUF_NU_H_STABILITY,
        _DYN_FMUF_SWITCH_MIN,
        _DYN_FMUF_SWITCH_RATIO,
    )

    tmax = float(T.max())
    # Sub-percent at physical mu-F distances; a looser guard at r = 0.6 A
    # (minimizer-exploration territory) where both branches sit at their
    # accuracy edge.
    for r, seam_tol in ((1.17, 5e-3), (0.8, 8e-3), (0.6, 3e-2)):
        nu_seam = min(
            max(_DYN_FMUF_SWITCH_MIN, _DYN_FMUF_SWITCH_RATIO * omega_d_mu_f_rad_per_us(r)),
            _DYN_FMUF_NU_H_STABILITY * (_DYN_FMUF_GRID_CAP - 1) / tmax,
        )
        below = dynamic_fmuf_polarization(T, r, nu_seam * 0.999)
        above = dynamic_fmuf_polarization(T, r, nu_seam * 1.001)
        assert np.max(np.abs(below - above)) < seam_tol, f"r={r}"


def test_triangle_distant_third_fluorine_matches_general_collinear() -> None:
    # r3 -> infinity decouples the third spin; the remaining physics is the
    # collinear three-spin problem with F-F coupling = FmuF_General(r, r, 180).
    tri = fmuf_triangle_polarization(T, 1.17, 60.0, 90.0)
    gen = general_fmuf_polarization(T, 1.17, 1.17, 180.0)
    assert np.allclose(tri, gen, atol=1e-6)


def test_triangle_component_handles_invalid_trial_point() -> None:
    y = COMPONENTS["FmuF_Triangle"].function(T, A=0.8, r_muF=1.2, r3=-1.0, phi3=90.0)
    assert np.all(np.isfinite(y))
    assert y[0] > 100.0


def test_triangle_angle_is_periodic_not_rejected() -> None:
    # Any phi3 is geometrically valid; out-of-canonical-range angles map onto
    # their mirror image instead of producing a flat penalty plateau.
    direct = fmuf_triangle_polarization(T, 1.17, 2.5, 160.0)
    mirrored = fmuf_triangle_polarization(T, 1.17, 2.5, 200.0)  # 360 - 200 = 160
    assert np.allclose(direct, mirrored, atol=1e-9)
    along_axis = fmuf_triangle_polarization(T, 1.17, 2.5, 0.0)
    assert np.all(np.isfinite(along_axis))


def test_distance_components_penalise_zero_distance_trials() -> None:
    # The optimiser may probe the inclusive r = 0 bound; the wrappers must
    # return the flat penalty rather than aborting the fit with a ValueError.
    for name, params in (
        ("ProtonDipole", {"A": 25.0, "r_muH": 0.0, "lambda_T": 0.0}),
        ("ElectronDipole", {"A": 25.0, "r_mue": 0.0, "lambda_T": 0.0}),
        ("DynamicFmuF", {"A": 25.0, "r_muF": 0.0, "nu": 0.5}),
        ("MuF", {"A": 25.0, "r_muF": 0.0}),
        ("FmuF_Linear", {"A": 25.0, "r_muF": 0.0}),
    ):
        y = COMPONENTS[name].function(T, **params)
        assert np.all(np.isfinite(y)), name
        assert y[0] > 100.0, name


def test_triangle_third_fluorine_changes_lineshape() -> None:
    near = fmuf_triangle_polarization(T, 1.17, 2.0, 90.0)
    far = fmuf_triangle_polarization(T, 1.17, 30.0, 90.0)
    assert np.max(np.abs(near - far)) > 0.01


# --- fit recovery smoke test ---------------------------------------------------


def test_risch_kehr_fit_recovery() -> None:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 12.0, 240)
    truth = {"A": 22.0, "Gamma": 1.5}
    definition = COMPONENTS["RischKehr"]
    y = definition.function(t, **truth) + rng.normal(0.0, 0.2, t.size)
    dataset = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.2))

    model = CompositeModel(["RischKehr"])
    definition_composite = model.to_model_definition()
    start = {"A_1": 18.0, "Gamma": 0.8}
    params = ParameterSet(
        [
            Parameter(name, value=start.get(name, 1.0), min=0.0)
            for name in definition_composite.param_names
        ]
    )
    result = FitEngine().fit(dataset, model.function, params)
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["A_1"] == pytest.approx(truth["A"], abs=0.5)
    assert fitted["Gamma"] == pytest.approx(truth["Gamma"], abs=0.2)


def test_plot_sample_count_accounts_for_hyperfine_frequencies() -> None:
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from asymmetry.gui.panels.fit_panel import _fit_curve_sample_count

    model = CompositeModel(["MuoniumHighTF"])
    params = {"A_1": 25.0, "field": 3000.0, "A_hf": 4463.302, "phase": 0.0}
    n = _fit_curve_sample_count(model, params, 0.0, 0.2)
    # 4463 MHz over 0.2 us needs ~36k points at 40/cycle -> hits the cap,
    # far above what the 41 MHz field-only estimate (~330 points) would give.
    assert n == 20000
