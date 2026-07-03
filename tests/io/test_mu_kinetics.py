"""Pulsed-source fast-muonium reaction kinetics.

RED gate for the ``pulsed-fast-mu-kinetics`` port (docs/porting/). The transverse
-field Mu signal of a fast-reacting sample has decayed before the first good bin
at a pulsed source, so a per-run fit cannot separate the Mu amplitude ``A_Mu``
from its relaxation ``lambda_Mu``. The kinetics module breaks the degeneracy by
*sharing* ``A_Mu`` across the concentration/temperature series (slow members pin
it; fast members then yield lambda_Mu), then fits the pseudo-first-order line
``lambda_Mu = lambda0 + k_Mu*[x]`` and the Arrhenius ``log10 k_Mu = log10 A -
E/(2.3 R T)``.

Truth is planted and synthetic (``core.simulate``); no binary fixtures. See
``docs/porting/pulsed-fast-mu-kinetics/verification-plan.md``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from asymmetry.core.fitting import CompositeModel
from asymmetry.core.fitting.mu_kinetics import (
    fit_arrhenius,
    fit_bimolecular_rate,
    fit_mu_relaxation_series,
    mu_relaxation_from_amplitude,
)
from asymmetry.core.simulate import (
    BUILTIN_TEMPLATES,
    reduce_run_to_dataset,
    simulate_run,
)

R_GAS = 8.314462618  # J / mol / K

# Fixed Mu precession frequency at 2 G (gamma_Mu ~ 1.394 MHz/G).
F_MU = 2.78


def _build_mu_series(
    concentrations,
    *,
    lambda0,
    k_mu,
    amplitude=8.0,
    phase=0.2,
    dia_amp=6.0,
    dia_lambda=0.1,
    temperature=290.0,
    good_bin_start=113,
    total_events=4.0e8,
    seed0=0,
):
    """Synthesise a 2 G Mu concentration series with a planted kinetics law.

    The shared ``amplitude`` is the (concentration-independent) muonium fraction;
    only ``lambda_Mu = lambda0 + k_mu*[x]`` varies across the series. ``good_bin_start``
    is raised so the fast members have decayed before the window opens (the real
    EMU t_good ~ 0.203 us at first_good_bin 21), reproducing the truncation.
    """
    base = BUILTIN_TEMPLATES["ideal_pulsed_fb"]
    template_def = dataclasses.replace(base, good_bin_start=good_bin_start)
    model = CompositeModel.from_expression("Oscillatory*Exponential + Exponential")
    datasets = []
    planted_lambdas = []
    for index, conc in enumerate(concentrations):
        lam = lambda0 + k_mu * conc
        params = {
            "A_1": amplitude,
            "frequency": F_MU,
            "phase": phase,
            "Lambda_2": lam,
            "A_3": dia_amp,
            "Lambda_3": dia_lambda,
        }
        run = simulate_run(
            template_def.build(),
            model,
            params,
            total_events=total_events,
            seed=seed0 + index,
        )
        dataset = reduce_run_to_dataset(run)
        dataset.metadata["temperature"] = temperature
        datasets.append(dataset)
        planted_lambdas.append(lam)
    return datasets, planted_lambdas


# Relative concentrations: water + quarter:half:full = 0:1:2:4 (GROUND_TRUTH).
CONCENTRATIONS = [0.0, 1.0, 2.0, 4.0]


class TestSharedAmplitudeBreaksDegeneracy:
    def test_recovers_lambda_for_all_members_including_truncated(self):
        lambda0, k_mu = 0.5, 2.5  # -> lambda = [0.5, 3.0, 5.5, 10.5]
        datasets, planted = _build_mu_series(CONCENTRATIONS, lambda0=lambda0, k_mu=k_mu)
        result = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True)

        assert result.success
        # Every member recovered, including the fast (truncated) full/half samples.
        for fitted, truth in zip(result.lambda_mu, planted, strict=True):
            assert fitted == pytest.approx(truth, rel=0.18)
        # The shared amplitude is the common muonium fraction (planted 8%).
        assert result.shared_amplitude == pytest.approx(8.0, rel=0.2)
        assert result.reduced_chi_squared < 3.0

    def test_free_fit_is_worse_on_the_truncated_member(self):
        """Without sharing, the fast member's lambda is degenerate (huge error)."""
        datasets, planted = _build_mu_series(CONCENTRATIONS, lambda0=0.5, k_mu=2.5)
        shared = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True)
        free = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=False)

        fast = len(CONCENTRATIONS) - 1  # full concentration -> fastest decay
        # The degeneracy shows up as a far larger lambda uncertainty when A_Mu is free.
        assert free.lambda_mu_error[fast] > 2.0 * shared.lambda_mu_error[fast]


class TestBimolecularRate:
    def test_recovers_slope_and_intercept(self):
        lambda0, k_mu = 0.47, 2.6
        lambdas = [lambda0 + k_mu * x for x in CONCENTRATIONS]
        errors = [0.05] * len(lambdas)
        result = fit_bimolecular_rate(CONCENTRATIONS, lambdas, errors)

        assert result.k_mu == pytest.approx(k_mu, rel=1e-6)
        assert result.lambda0 == pytest.approx(lambda0, rel=1e-6)
        assert result.k_mu_error > 0.0

    def test_single_concentration_raises(self):
        with pytest.raises(ValueError):
            fit_bimolecular_rate([1.0], [3.0], [0.1])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            fit_bimolecular_rate([0.0, 1.0, 2.0], [0.5, 3.0], [0.1, 0.1])

    def test_unbounded_member_error_is_actionable(self):
        # An inf lambda-error (unbounded series member) raises a pointed error
        # rather than a low-level "sigma must be finite" message.
        with pytest.raises(ValueError, match="not bounded by the relaxation fit"):
            fit_bimolecular_rate(
                [0.0, 1.0, 2.0, 4.0], [0.5, 1.0, 1.5, 2.0], [0.05, 0.05, 0.05, math.inf]
            )


class TestArrhenius:
    def test_recovers_activation_energy_kjmol(self):
        # log10 k = log10 A - E/(2.3 R T); plant E and recover it.
        ea_j = 17600.0  # 17.6 kJ/mol (diffusion-controlled, GROUND_TRUTH section 6)
        pre = 3700.0
        temperatures = [278.0, 288.0, 298.0, 308.0, 318.0, 328.0, 338.0, 358.0]
        k_values = [pre * math.exp(-ea_j / (R_GAS * T)) for T in temperatures]
        errors = [0.03 * k for k in k_values]
        result = fit_arrhenius(temperatures, k_values, errors)

        assert result.activation_energy == pytest.approx(17.6, rel=0.05)  # kJ/mol
        assert result.activation_energy_error > 0.0


class TestEndToEndPipeline:
    """Series -> shared-amplitude lambda_Mu -> k_Mu(T) -> Arrhenius E_a."""

    def test_recovers_planted_kinetics(self):
        ea_j = 17600.0
        pre = 3700.0
        lambda0 = 0.5
        temperatures = [278.0, 298.0, 338.0]
        k_per_temperature = []
        for t_index, temperature in enumerate(temperatures):
            k_true = pre * math.exp(-ea_j / (R_GAS * temperature))
            datasets, _ = _build_mu_series(
                CONCENTRATIONS,
                lambda0=lambda0,
                k_mu=k_true,
                temperature=temperature,
                total_events=2.0e8,
                seed0=100 * (t_index + 1),
            )
            relax = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True)
            rate = fit_bimolecular_rate(CONCENTRATIONS, relax.lambda_mu, relax.lambda_mu_error)
            assert rate.k_mu == pytest.approx(k_true, rel=0.25)
            k_per_temperature.append(rate.k_mu)

        arr = fit_arrhenius(temperatures, k_per_temperature, [0.1 * k for k in k_per_temperature])
        assert arr.activation_energy == pytest.approx(17.6, rel=0.3)  # kJ/mol


import os  # noqa: E402

_MALEIC_DIR = os.environ.get("ASYMMETRY_MALEIC_DIR")


def _load_maleic(run_number):
    from asymmetry.core.io import load

    path = os.path.join(_MALEIC_DIR, "Data_hdf5", f"emu{run_number:08d}.nxs")
    return load(path)


@pytest.mark.skipif(_MALEIC_DIR is None, reason="ASYMMETRY_MALEIC_DIR not set")
class TestRealCorpusSweep:
    """End-to-end on the real EMU maleic-acid corpus (env-gated, not in CI).

    Grades the physics (linearity, positive slope, finite fast-member rates,
    Arrhenius order) — no reference fit log exists for this example (GT §7).
    """

    # Room-T (~290 K) 2 G Mu runs, relative [x] = {0, 1, 2, 4}.
    ROOM_T = {0.0: 78251, 1.0: 78279, 2.0: 78277, 4.0: 78257}
    # Deoxygenated-water reference (x = 0) — its slow, well-surviving Mu signal
    # pins the shared A_Mu when the maleic members are all fast (the amplitude
    # calibration the method needs; see the user guide).
    WATER_REF = 78251
    # 2 G Mu concentration triplets {quarter, half, full} at three temperatures.
    ARRHENIUS = {
        278.0: {1.0: 78282, 2.0: 78294, 4.0: 78259},
        298.0: {1.0: 78284, 2.0: 78296, 4.0: 78263},
        338.0: {1.0: 78288, 2.0: 78300, 4.0: 78271},
    }

    def test_room_temperature_concentration_line(self):
        concentrations = sorted(self.ROOM_T)
        datasets = [_load_maleic(self.ROOM_T[c]) for c in concentrations]
        relax = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True, t_max=8.0)

        assert relax.success
        # The half/full rates are now finite and physical (the S5 ❌ -> ✅ flip).
        assert all(0.0 < lam < 50.0 for lam in relax.lambda_mu)
        # Monotonic rise with concentration.
        assert relax.lambda_mu == sorted(relax.lambda_mu)
        rate = fit_bimolecular_rate(concentrations, relax.lambda_mu, relax.lambda_mu_error)
        assert rate.k_mu > 0.0
        assert 0.0 < rate.lambda0 < 5.0  # physical water background

    def test_arrhenius_activation_energy_order(self):
        temperatures = sorted(self.ARRHENIUS)
        k_per_temperature = []
        k_errors = []
        for temperature in temperatures:
            triplet = self.ARRHENIUS[temperature]
            # Anchor the shared A_Mu with the slow water reference at [x] = 0.
            concentrations = [0.0, *sorted(triplet)]
            datasets = [_load_maleic(self.WATER_REF)] + [
                _load_maleic(triplet[c]) for c in sorted(triplet)
            ]
            relax = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True, t_max=8.0)
            rate = fit_bimolecular_rate(concentrations, relax.lambda_mu, relax.lambda_mu_error)
            assert rate.k_mu > 0.0
            k_per_temperature.append(rate.k_mu)
            k_errors.append(max(rate.k_mu_error, 1e-3))

        arr = fit_arrhenius(temperatures, k_per_temperature, k_errors)
        # Diffusion-controlled order of magnitude (literature ~17.6 kJ/mol).
        assert 3.0 < arr.activation_energy < 60.0


class TestAmplitudeInversionCrossCheck:
    def test_agrees_with_shared_fit_on_truncated_member(self):
        datasets, planted = _build_mu_series(CONCENTRATIONS, lambda0=0.5, k_mu=2.5)
        shared = fit_mu_relaxation_series(datasets, f_mu=F_MU, share_amplitude=True)

        fast = len(CONCENTRATIONS) - 1
        lam, sigma = mu_relaxation_from_amplitude(
            datasets[fast],
            reference_amplitude=shared.shared_amplitude,
            f_mu=F_MU,
            phase=shared.shared_phase,
        )
        assert sigma > 0.0
        assert lam == pytest.approx(planted[fast], rel=0.30)
