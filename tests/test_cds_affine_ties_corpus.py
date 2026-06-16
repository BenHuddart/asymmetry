"""Deliverable guard for affine ties on the real CdS shallow-donor corpus.

Session-5 API finding (``_findings/windows-api/Nuclear+Semi.md``): with three
*free* satellite frequencies the CdS amplitudes scatter, so the Mu0 ionisation
energy E_i is un-extractable (the API run got E_i = 43 +- 1090 meV). Tying the
satellites to ``f_centre +- delta`` (delta free) stabilises the amplitudes and
makes E_i extractable — the capability this branch adds.

Grade target (``Shallow donor state in cadmium sulphide/GROUND_TRUTH.md``):
A_mu ~ 0.20 MHz (soft literature); satellites collapse on warming (~30 K);
E_i ~ tens of meV. **Corpus-conditional**: skips cleanly when the WiMDA muon
-school corpus is absent (always in CI). Run locally to exercise it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet

_K_B = 0.0861733  # meV/K
_EXPR = "Oscillatory*Exponential + Oscillatory*Exponential + Oscillatory*Exponential + Constant"
# Cold runs up the temperature ramp (Tlog spans ~5 -> ~30 K), enough to define
# the Arrhenius slope. Run -> Tlog from GROUND_TRUTH §3.
_RUNS = [20721, 20720, 20719, 20718, 20722, 20723, 20724, 20725, 20726, 20727, 20728]


def _corpus_root() -> Path | None:
    candidates = [
        os.environ.get("WIMDA_CORPUS_ROOT"),
        r"C:\Users\benhu\Source\wimda-corpus",
        str(Path.home() / "Documents" / "WiMDA muon school"),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _cds_dir() -> Path | None:
    root = _corpus_root()
    if root is None:
        return None
    matches = sorted(root.rglob("EMU00020721.nxs"))
    hdf5 = [m for m in matches if "hdf5" in str(m).lower()]
    matches = hdf5 or matches
    return matches[0].parent if matches else None


pytestmark = pytest.mark.skipif(_cds_dir() is None, reason="WiMDA CdS corpus not present")


def _model_fn():
    return CompositeModel.from_expression(_EXPR).to_model_definition().function


def _tied_seed(fc: float = 1.39, dsat: float = 0.126) -> ParameterSet:
    """Three damped lines; phase/relaxation/amplitude shared via link groups,
    satellites tied to f_centre +- delta with delta a free auxiliary param."""
    ps = ParameterSet(
        [
            Parameter("A_1", 8, min=0, max=30),
            Parameter("frequency_1", fc, min=fc - 0.3, max=fc + 0.3),
            Parameter("phase_1", 0, min=-3.2, max=3.2, link_group=3),
            Parameter("Lambda_2", 0.3, min=0, max=10, link_group=1),
            Parameter("A_3", 4, min=0, max=30, link_group=2),
            Parameter("phase_3", 0, min=-3.2, max=3.2, link_group=3),
            Parameter("Lambda_4", 0.3, min=0, max=10, link_group=1),
            Parameter("A_5", 4, min=0, max=30, link_group=2),
            Parameter("phase_5", 0, min=-3.2, max=3.2, link_group=3),
            Parameter("Lambda_6", 0.3, min=0, max=10, link_group=1),
            Parameter("A_bg", 0, min=-10, max=20),
        ]
    )
    ps.add(
        Parameter(
            "frequency_3",
            fc - dsat,
            tie=AffineTie("frequency_1", offset="delta", offset_scale=-1.0),
        )
    )
    ps.add(
        Parameter(
            "frequency_5",
            fc + dsat,
            tie=AffineTie("frequency_1", offset="delta", offset_scale=+1.0),
        )
    )
    ps.add(Parameter("delta", dsat, min=0.0, max=0.5))
    return ps


def _fit(run: int):
    from asymmetry.core.io import load

    ds = load(str(_cds_dir() / f"EMU000{run}.nxs"))
    result = FitEngine().fit(ds, _model_fn(), _tied_seed(), 0.1, 8.0)
    return ds, result


def test_base_temperature_tied_fit_recovers_hyperfine() -> None:
    ds, result = _fit(20721)  # coldest, highest-stats (Tlog ~ 5.18 K)
    assert result.success
    assert ds.sample_temperature_logged < 6.0
    assert result.reduced_chi_squared < 1.6

    fitted = {p.name: p.value for p in result.parameters}
    # Symmetry exactly enforced by the tie.
    assert fitted["frequency_3"] == pytest.approx(fitted["frequency_1"] - fitted["delta"], abs=1e-9)
    assert fitted["frequency_5"] == pytest.approx(fitted["frequency_1"] + fitted["delta"], abs=1e-9)
    # A_mu = 2*delta ~ 0.24 MHz (GT soft 0.20; program 0.242), with a tight error.
    assert 2.0 * fitted["delta"] == pytest.approx(0.24, abs=0.04)
    assert result.uncertainties["delta"] < 0.01


def test_satellites_collapse_on_warming() -> None:
    _ds_cold, cold = _fit(20721)
    _ds_warm, warm = _fit(20728)  # Tlog ~ 30 K, above the Mu0 onset
    delta_cold = {p.name: p.value for p in cold.parameters}["delta"]
    delta_warm = {p.name: p.value for p in warm.parameters}["delta"]
    assert delta_cold > 0.1  # well-resolved satellites when cold
    assert delta_warm < 0.05  # collapsed onto the central line when warm


def test_ionisation_energy_is_extractable() -> None:
    """The deliverable: tied amplitudes yield a finite, physical E_i.

    With three free frequencies this fit produced E_i = 43 +- 1090 meV (the
    uncertainty dwarfs the value). Tying the satellites must bring the relative
    uncertainty under control and the value into the tens-of-meV scale.
    """
    from scipy.optimize import curve_fit

    temps, fracs = [], []
    for run in _RUNS:
        ds, result = _fit(run)
        d = {p.name: p.value for p in result.parameters}
        neutral = d["A_3"] + d["A_5"]
        temps.append(ds.sample_temperature_logged)
        fracs.append(neutral / (d["A_1"] + neutral))

    t = np.asarray(temps)
    f = np.asarray(fracs)

    def ionisation(T, f0, C, Ei):  # noqa: N803 — physics symbols (temperature, prefactor, energy)
        return f0 / (1.0 + C * np.exp(-Ei / (_K_B * T)))

    popt, pcov = curve_fit(
        ionisation,
        t,
        f,
        p0=[f.max(), 1.0, 15.0],
        bounds=([0, 0, 0], [1, 1e6, 500]),
        maxfev=20000,
    )
    e_i = popt[2]
    sigma = float(np.sqrt(pcov[2, 2]))

    assert 1.0 < e_i < 60.0  # tens-of-meV scale (GT), not the 43-meV-with-huge-error noise floor
    assert sigma / e_i < 1.0  # finite & extractable, unlike the free-frequency 43 +- 1090
