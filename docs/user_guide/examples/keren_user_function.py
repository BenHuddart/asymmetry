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
    the Abragam exponent; as nu and B_L both vanish it tends to the static
    Gaussian envelope exp(-Delta^2 t^2).
    """
    t = np.asarray(t, dtype=float)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)  # rad/us
    delta2 = float(Delta) * float(Delta)
    w2 = omega0 * omega0
    n2 = float(nu) * float(nu)
    denom = w2 + n2

    if denom < 1e-20:
        # nu -> 0 and B_L -> 0: Gamma -> Delta^2 t^2 (static Gaussian limit)
        exponent = np.clip(-delta2 * t * t, -700, 0)
        return A * np.exp(exponent)

    e = np.exp(np.clip(-float(nu) * np.abs(t), -700, 0))
    gamma = (2.0 * delta2 / (denom * denom)) * (
        denom * float(nu) * np.abs(t)
        + (w2 - n2) * (1.0 - e * np.cos(omega0 * t))
        - 2.0 * float(nu) * omega0 * e * np.sin(omega0 * t)
    )
    exponent = np.clip(-gamma, -700, 0)
    return A * np.exp(exponent)


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
