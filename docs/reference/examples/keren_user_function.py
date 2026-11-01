"""Keren dynamic relaxation as an Asymmetry user function.

Drop this file in ``~/.asymmetry/user_functions/`` and restart Asymmetry:
``KerenUser`` appears in the fit-function builder under *User*, fits like
any built-in component, and survives project save/load.

The function reproduces the shipped ``Keren`` component exactly (it is the
worked example from the "User functions" chapter of the user guide) — for
your own physics, replace the body and metadata.
"""

import numpy as np

from asymmetry import register_component
from asymmetry.core.fitting.phi_functions import phi2
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


def keren_user(t, A, Delta, nu, B_L):
    """Keren's analytic dynamic Gaussian relaxation in a longitudinal field.

    P(t) = A exp[-Gamma(t)] with omega_0 = gamma_mu B_L and

        Gamma(t) = (2 Delta^2 / (omega_0^2 + nu^2)^2) * [
            (omega_0^2 + nu^2) nu t
            + (omega_0^2 - nu^2) (1 - e^{-nu t} cos(omega_0 t))
            - 2 nu omega_0 e^{-nu t} sin(omega_0 t) ]

    after A. Keren, Phys. Rev. B 50, 10039 (1994). At B_L = 0 it reduces to
    twice the transverse-field Abragam exponent; as nu and B_L vanish it tends to the
    Gaussian envelope exp(-Delta^2 t^2).
    """
    if not np.isfinite(nu) or nu < 0:
        raise ValueError("fluctuation rate nu must be finite and >= 0")
    t = np.abs(np.asarray(t, dtype=float))
    omega0 = 2 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * float(B_L)
    exponent = 2 * float(Delta) ** 2 * t**2 * phi2(-(float(nu) + 1j * omega0) * t).real
    return A * np.exp(-exponent)


register_component(
    "KerenUser",
    keren_user,
    ["A", "Delta", "nu", "B_L"],
    domain="time",
    description=(
        "Keren dynamic Gaussian relaxation in a longitudinal field "
        "(user-function copy of the built-in Keren component)"
    ),
    formula_template="{A}*exp(-Gamma(t; Delta={Delta}, nu={nu}, B_L={B_L}))",
    latex_equation=(
        r"A(t)=A\exp[-\Gamma(t)],\ \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}"
        r"\left[(\omega_0^2+\nu^2)\nu t+(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)"
        r"-2\nu\omega_0 e^{-\nu t}\sin\omega_0 t\right],\ \omega_0=\gamma_\mu B_L"
    ),
    applicability=(
        "Use for dynamically fluctuating Gaussian local fields in an applied "
        "longitudinal field, where the fluctuation rate ν and the Larmor "
        "frequency ω₀ are comparable — the analytic fast-fluctuation "
        "alternative to the numerical dynamic Kubo-Toyabe function."
    ),
    references=("A. Keren, Phys. Rev. B 50, 10039 (1994).",),
    param_defaults={"A": 25.0, "Delta": 0.5, "nu": 1.0, "B_L": 0.0},
)
