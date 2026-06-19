"""Higher-degree ALC baseline models (Quartic / Quintic / Sextic).

The ALC scan-view baseline used to top out at ``Cubic`` (degree 3). A steep
0–3 T muonium-repolarisation (µLCR) envelope needs degree ≥6 to flatten
cleanly; Cubic leaves the radical resonance dips riding on a curved residual,
so a GUI-only user could not reach a paper-grade baseline (corpus MED #6,
corannulene). These tests pin the new degree-4/5/6 components and that they
leave a flatter residual than Cubic on a steep synthetic envelope while still
recovering the dip positions.

The baseline math lives in ``asymmetry.core.fitting`` (Qt-free); the GUI combo
exposure is asserted separately in the ALC panel GUI tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting import fit_scan_baseline
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.core.transform import FieldScan

# Two narrow radical dips sitting on a steep, multi-scale repolarisation rise,
# on a Gauss-scale field axis (0–3 T ≈ 0–30000 G) — the regime where Cubic is
# too stiff. The dips are excluded from the fitted (non-resonant) regions.
_DIP1, _DIP2 = 11000.0, 21000.0
_REGIONS = [(0.0, 8500.0), (13500.0, 18000.0), (24000.0, 30000.0)]


def _steep_envelope_scan() -> FieldScan:
    x = np.linspace(0.0, 30000.0, 80)
    envelope = 9.0 * (1.0 - np.exp(-x / 4500.0)) - 3.0 * np.exp(-x / 16000.0)

    def _dip(centre: float, width: float, amp: float) -> np.ndarray:
        return -amp * np.exp(-0.5 * ((x - centre) / width) ** 2)

    value = envelope + _dip(_DIP1, 350.0, 1.4) + _dip(_DIP2, 350.0, 1.1)
    return FieldScan(
        x=x,
        value=value,
        error=np.full_like(x, 0.05),
        run_numbers=list(range(1, x.size + 1)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )


def _region_residual_rms(scan: FieldScan, baseline: np.ndarray) -> float:
    in_region = np.zeros(scan.x.size, dtype=bool)
    for lo, hi in _REGIONS:
        in_region |= (scan.x >= lo) & (scan.x <= hi)
    resid = np.asarray(scan.value, dtype=float) - np.asarray(baseline, dtype=float)
    return float(np.sqrt(np.mean(resid[in_region] ** 2)))


@pytest.mark.parametrize(
    "name, degree",
    [("Quartic", 4), ("Quintic", 5), ("Sextic", 6)],
)
def test_high_degree_polynomial_components_registered(name, degree):
    comp = PARAMETER_MODEL_COMPONENTS[name]
    expected = [f"c{k}" for k in range(degree + 1)]
    assert comp.param_names == expected
    # Defaults and per-parameter info must cover exactly the named coefficients.
    assert set(comp.param_defaults) == set(expected)
    assert set(comp.param_info) == set(expected)


def test_sextic_baseline_flatter_than_cubic_on_steep_envelope():
    scan = _steep_envelope_scan()

    cubic = fit_scan_baseline(scan, _REGIONS, model="Cubic")
    sextic = fit_scan_baseline(scan, _REGIONS, model="Sextic")
    assert cubic.fit.success and sextic.fit.success

    cubic_rms = _region_residual_rms(scan, cubic.baseline)
    sextic_rms = _region_residual_rms(scan, sextic.baseline)
    # Degree 6 must flatten the off-resonance baseline materially better than
    # Cubic — the whole point of raising the ceiling.
    assert sextic_rms < 0.5 * cubic_rms


def test_higher_degrees_all_beat_cubic_on_steep_envelope():
    scan = _steep_envelope_scan()
    rms = {
        name: _region_residual_rms(scan, fit_scan_baseline(scan, _REGIONS, model=name).baseline)
        for name in ("Cubic", "Quartic", "Quintic", "Sextic")
    }
    # Every higher order flattens this steep envelope better than Cubic; the
    # ladder gives the user a way to pick the lowest adequate degree. The fits
    # run through migrad (a non-linear optimiser, not an exact nested
    # least-squares solve), so only the comfortable margins are asserted — not a
    # strict adjacent-degree ordering, which could flip on optimiser noise.
    assert rms["Quartic"] < rms["Cubic"]
    assert rms["Quintic"] < rms["Cubic"]
    assert rms["Sextic"] < rms["Cubic"]
    # Degree 6 is well clear of degree 4 (a wide, stable gap, not an adjacent tie).
    assert rms["Sextic"] < rms["Quartic"]


def test_sextic_baseline_recovers_dip_positions():
    scan = _steep_envelope_scan()
    corrected = fit_scan_baseline(scan, _REGIONS, model="Sextic").corrected
    x = np.asarray(scan.x, dtype=float)
    y = np.asarray(corrected.value, dtype=float)

    # After a flat baseline subtraction the two most negative excursions are the
    # radical dips; each should sit within one field step of its true centre.
    step = float(x[1] - x[0])
    for centre in (_DIP1, _DIP2):
        window = np.abs(x - centre) <= 1500.0
        argmin_x = x[window][int(np.argmin(y[window]))]
        assert abs(argmin_x - centre) <= step
