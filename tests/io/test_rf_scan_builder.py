"""RF-µSR (Green − Red) field-scan builder + resonance fit.

Covers :func:`asymmetry.core.io.periods.build_rf_difference_scan` (the scan
acquisition that closes the GUI side of parity gap PC1) and
:func:`asymmetry.core.fitting.field_scan.fit_rf_resonance` (the A_µ/A_p read-out).

Two layers:

* **Synthetic, always-on** — two-period runs whose (Green − Red) curve is a known
  two-Lorentzian resonance (from the verified ``RFResonanceMuP`` model); the
  builder must reproduce it and the fit must invert it back to the input
  couplings. Also exercises ordering, windowing, and exclusion rules.
* **Corpus-conditional** — the benzene DEVA RF scan (runs 56426–56462), skipped
  when the WiMDA muon-school corpus is absent. Recovers A_µ ≈ 514.78 MHz.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.field_scan import fit_rf_resonance
from asymmetry.core.fitting.muon_proton import rf_resonance_mup
from asymmetry.core.io.periods import build_rf_difference_scan
from asymmetry.core.utils.constants import PeriodMode

# --- synthetic fixtures -------------------------------------------------------

_TIME = np.linspace(0.1, 8.0, 64)
# Truth couplings, deliberately off the fit's default seeds (515 / 124) so the
# round-trip exercises convergence rather than a no-op.
_A_MU_TRUE = 520.0
_A_P_TRUE = 130.0
_NU_RF = 218.5


def _green_red_value(field: float) -> float:
    """The (Green − Red) integral-asymmetry value (%) at *field*, from the model."""
    return float(
        rf_resonance_mup(
            np.array([field], dtype=float),
            A_mu=_A_MU_TRUE,
            A_p=_A_P_TRUE,
            nu_RF=_NU_RF,
            ampl1=2.0,
            wid1=30.0,
            ampl2=2.0,
            wid2=30.0,
            BG=0.1,
        )[0]
    )


def _two_period_run(run_number: int, field: float, *, green_minus_red: float) -> Run:
    """A two-period run whose Green − Red curve is the flat value *green_minus_red* (%)."""
    n = _TIME.size
    red = np.zeros(n, dtype=float)
    green = np.full(n, float(green_minus_red), dtype=float)
    err = np.full(n, 0.01, dtype=float)
    grouping = {
        "period_reduced": [
            (_TIME.copy(), red, err.copy()),  # red  = period 1
            (_TIME.copy(), green, err.copy()),  # green = period 2
        ],
        "period_count": 2,
    }
    return Run(
        run_number=run_number,
        histograms=[],
        metadata={"field": float(field), "temperature": 293.0, "run_number": run_number},
        grouping=grouping,
        source_file="synthetic",
    )


def _single_period_run(run_number: int, field: float) -> Run:
    """A run with no per-period data (should be excluded by the RF builder)."""
    return Run(
        run_number=run_number,
        histograms=[Histogram(counts=np.ones(8, dtype=float), bin_width=0.1)],
        metadata={"field": float(field), "run_number": run_number},
        grouping={"period_count": 1},
        source_file="synthetic",
    )


def _synthetic_runs(fields: np.ndarray) -> list[Run]:
    return [
        _two_period_run(56000 + i, float(b), green_minus_red=_green_red_value(float(b)))
        for i, b in enumerate(fields)
    ]


# --- builder behaviour --------------------------------------------------------


def test_builder_forms_green_minus_red_value_per_run() -> None:
    """Each scan point is the run's (Green − Red) integral, fractional (÷100)."""
    runs = _synthetic_runs(np.array([700.0, 800.0]))
    scan = build_rf_difference_scan(runs)
    assert scan.n_points == 2
    # Fractional convention (build_field_scan-style): value == model% / 100.
    assert scan.value[0] == pytest.approx(_green_red_value(700.0) / 100.0, rel=1e-6)
    assert scan.value[1] == pytest.approx(_green_red_value(800.0) / 100.0, rel=1e-6)


def test_builder_orders_by_field_and_excludes_single_period() -> None:
    """Points sort by field ascending; non-two-period runs land in `excluded`."""
    runs = _synthetic_runs(np.array([900.0, 600.0, 750.0]))
    runs.insert(0, _single_period_run(57000, 650.0))
    scan = build_rf_difference_scan(runs, order_key="field")
    assert scan.n_points == 3
    assert list(scan.x) == sorted(scan.x)
    assert list(scan.x) == [600.0, 750.0, 900.0]
    assert (57000, "not a two-period (red/green) run") in scan.excluded


def test_builder_window_restricts_integration() -> None:
    """A t_min/t_max window outside the curve excludes the run with a reason."""
    runs = _synthetic_runs(np.array([770.0, 860.0]))
    scan = build_rf_difference_scan(runs, t_min=100.0, t_max=200.0)
    assert scan.n_points == 0
    assert len(scan.excluded) == 2


def test_builder_accepts_muondataset_and_green_plus_red_mode() -> None:
    """MuonDataset inputs resolve to their run; GREEN_PLUS_RED is accepted."""
    run = _two_period_run(58000, 800.0, green_minus_red=1.5)
    ds = MuonDataset(
        time=_TIME.copy(),
        asymmetry=np.zeros_like(_TIME),
        error=np.ones_like(_TIME),
        metadata=run.metadata,
        run=run,
    )
    scan = build_rf_difference_scan([ds], mode=PeriodMode.GREEN_PLUS_RED)
    assert scan.n_points == 1
    # Green + Red = 0 + 1.5 (% ) → 0.015 fractional.
    assert scan.value[0] == pytest.approx(1.5 / 100.0, rel=1e-6)


def test_builder_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        build_rf_difference_scan([], mode="not-a-mode")
    with pytest.raises(ValueError):
        build_rf_difference_scan([], order_key="bogus")


# --- round-trip fit -----------------------------------------------------------


def test_fit_recovers_input_couplings_on_synthetic_scan() -> None:
    """Build a noiseless model scan, then fit it back to the input A_µ / A_p.

    Seeds are placed near the truth (a few MHz off): the RF-resonance χ² surface
    is multimodal in A_p, so a fit needs a reasonable starting guess — which is
    exactly why the GUI exposes the A_µ₀ / A_p₀ seed inputs. Convergence from the
    benzene *default* seeds on real data is covered by the corpus test below.
    """
    fields = np.arange(560.0, 1081.0, 10.0)
    scan = build_rf_difference_scan(_synthetic_runs(fields))
    assert scan.n_points == fields.size

    result = fit_rf_resonance(scan, nu_rf=_NU_RF, a_mu=_A_MU_TRUE - 2.0, a_p=_A_P_TRUE - 3.0)
    assert result.success
    assert result.parameters["A_mu"].value == pytest.approx(_A_MU_TRUE, abs=0.5)
    assert result.parameters["A_p"].value == pytest.approx(_A_P_TRUE, abs=1.0)
    # nu_RF is held fixed.
    assert result.parameters["nu_RF"].value == pytest.approx(_NU_RF, abs=1e-9)


# --- corpus-conditional recovery ----------------------------------------------


def _corpus_rf_dir() -> Path | None:
    """Locate the benzene RF-resonance run directory, or None when absent."""
    candidates = []
    env = os.environ.get("WIMDA_CORPUS_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.home() / "Documents" / "WiMDA muon school",
        Path.home() / "Source" / "wimda-corpus",
        Path("C:/Users/benhu/Source/wimda-corpus"),
    ]
    rel = Path("Chemistry") / "Muon spectroscopy of benzene"
    for root in candidates:
        benzene = root / rel
        for sub in ("data_hdf5", "data"):
            rf_dir = benzene / sub / "RF resonance"
            if rf_dir.is_dir() and any(rf_dir.glob("*.nxs")):
                return rf_dir
    return None


_RF_DIR = _corpus_rf_dir()


# Loads 37 real HDF4 files + a Migrad fit (~5s alone) — a genuine standard-tier
# outlier among otherwise-fast unit tests. Flaky under -n auto standard-tier
# runs (observed 2026-07-13, recurred 2026-07-17): xdist schedules it onto a
# worker alongside dozens of other tests with no wall-clock margin, and under
# full-suite contention it has intermittently run past the tier's tight
# per-test budget. `slow` moves it to the full tier (2 workers, 600s timeout —
# see tools/harness.py _FULL_TIER_WORKERS/_FULL_TIER_TIMEOUT_S), matching the
# precedent already set for the other corpus-conditional heavy test,
# test_corpus_hdf4_parity in tests/io/test_hdf4_loader.py.
@pytest.mark.slow
@pytest.mark.skipif(_RF_DIR is None, reason="WiMDA benzene RF corpus not present")
def test_benzene_rf_scan_recovers_paper_couplings() -> None:
    """End-to-end: load benzene RF runs → Green − Red scan → fit → A_µ ≈ 514.78."""
    from asymmetry.core.io import load

    assert _RF_DIR is not None
    datasets = []
    for path in sorted(_RF_DIR.glob("*.nxs")):
        try:
            loaded = load(str(path))
        except Exception:  # noqa: BLE001 - a single unreadable file must not abort
            continue
        datasets.append(loaded[0] if isinstance(loaded, list) else loaded)

    assert len(datasets) >= 20, "expected the benzene RF field series to load"

    # Early-time window where the resonance contrast is strongest (WiMDA's
    # time-integral of the period difference).
    scan = build_rf_difference_scan(datasets, t_min=0.0, t_max=1.0, order_key="field")
    assert scan.n_points >= 20
    assert 560.0 <= scan.x.min() and scan.x.max() <= 1100.0

    result = fit_rf_resonance(scan, nu_rf=218.5)
    assert result.success
    # Paper (McKenzie 2013, Table 1): A_µ = 514.78(4), A_p = 124.6(14) MHz; the
    # core port recovers 516.0 / 125.4. Tolerances bracket both.
    assert result.parameters["A_mu"].value == pytest.approx(514.78, abs=3.0)
    assert result.parameters["A_p"].value == pytest.approx(124.6, abs=6.0)
    assert result.reduced_chi_squared < 5.0
