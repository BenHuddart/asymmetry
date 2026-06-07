"""Phase-3 validation of the core period-selection API on photo-µSR silicon.

Runs the start of the carrier-recombination workflow described in
``CarrierRecombinationSilicon 2026.docx`` purely through the scriptable core
API (no GUI):

1. Load a period-mode HIFI run and extract the light-OFF (Green) and
   light-ON (Red) spectra with ``asymmetry.core.io.select_period``.
2. Confirm light-OFF shows small relaxation and light-ON significant
   relaxation.
3. Fit light-OFF with a single exponential to get A0, fix A0, then fit the
   first ~1 µs of light-ON for the relaxation rate λ — the first point of the
   λ-vs-Δn calibration.

Usage::

    .venv/bin/python docs/porting/period-selection/validate_photomusr.py
"""

from __future__ import annotations

import os

import numpy as np

from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.io import load, period_labels, select_period

DATA_DIR = os.path.expanduser(
    "~/Documents/WiMDA muon school/Semiconductors/Photo-muSR in silicon/Data_hdf5"
)
RUN = "HIFI00103277.nxs"


def _relaxation_proxy(dataset) -> float:
    """Peak-to-tail drop of asymmetry over the first 4 µs (a crude rate proxy)."""
    mask = dataset.time < 4.0
    asym = dataset.asymmetry[mask]
    return float(asym[0] - asym[-1])


def main() -> None:
    path = os.path.join(DATA_DIR, RUN)
    combined = load(path)
    print(f"Loaded {RUN}: {type(combined).__name__}, periods = {period_labels(combined)}")
    print(f"  period_number/count = "
          f"{combined.metadata.get('period_number')}/{combined.metadata.get('period_count')}")

    light_off = select_period(combined, "green")  # period 2
    light_on = select_period(combined, "red")     # period 1

    print("\nPer-period extraction:")
    for name, ds in (("light-OFF (Green)", light_off), ("light-ON (Red)", light_on)):
        print(f"  {name:18s} label={ds.run_label} n={ds.n_points} "
              f"T={ds.run.temperature} K  B={ds.run.field} G  drop={_relaxation_proxy(ds):+.3f}")

    assert np.allclose(light_off.time, light_on.time), "period time axes should match"
    assert _relaxation_proxy(light_on) > _relaxation_proxy(light_off), (
        "light-ON should relax more than light-OFF"
    )
    print("\n  ✓ light-ON relaxes more than light-OFF")

    # --- Step 1: fit light-OFF single exponential to obtain A0 ----------------
    engine = FitEngine()
    model = MODELS["ExponentialRelaxation"]

    off_params = ParameterSet()
    off_params.add(Parameter(name="A0", value=20.0, min=0.0, max=100.0))
    off_params.add(Parameter(name="Lambda", value=0.05, min=0.0))
    off_params.add(Parameter(name="baseline", value=0.0, fixed=True))
    off_fit = engine.fit(light_off, model.function, off_params, t_min=0.0, t_max=8.0)
    a0 = off_fit.parameters["A0"].value
    print(f"\nlight-OFF fit: success={off_fit.success} "
          f"A0={a0:.3f} λ={off_fit.parameters['Lambda'].value:.4f} µs⁻¹ "
          f"χ²ᵣ={off_fit.reduced_chi_squared:.3f}")

    # --- Step 2: fix A0, fit light-ON first ~1 µs for λ -----------------------
    on_params = ParameterSet()
    on_params.add(Parameter(name="A0", value=a0, fixed=True))
    on_params.add(Parameter(name="Lambda", value=0.5, min=0.0))
    on_params.add(Parameter(name="baseline", value=0.0, fixed=True))
    on_fit = engine.fit(light_on, model.function, on_params, t_min=0.0, t_max=1.0)
    lam = on_fit.parameters["Lambda"].value
    print(f"light-ON fit (A0 fixed, 0–1 µs): success={on_fit.success} "
          f"λ={lam:.4f} ± {on_fit.uncertainties.get('Lambda', float('nan')):.4f} µs⁻¹ "
          f"χ²ᵣ={on_fit.reduced_chi_squared:.3f}")

    assert off_fit.success and on_fit.success, "both fits should converge"
    assert lam > off_fit.parameters["Lambda"].value, "light-ON λ should exceed light-OFF λ"
    print("\n  ✓ workflow reproduced end-to-end via the core API")


if __name__ == "__main__":
    main()
