"""RF scan estimators, explicit historical means, and synthetic fit inversion."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.field_scan import fit_rf_resonance
from asymmetry.core.fitting.muon_proton import rf_resonance_mup
from asymmetry.core.io.periods import build_rf_difference_scan
from asymmetry.core.utils.constants import PeriodMode
from tests.corpus import corpus_path, requires_corpus

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
    scan = build_rf_difference_scan(runs, method="differential")
    assert scan.n_points == 2
    # Fractional convention (build_field_scan-style): value == model% / 100.
    assert scan.value[0] == pytest.approx(_green_red_value(700.0) / 100.0, rel=1e-6)
    assert scan.value[1] == pytest.approx(_green_red_value(800.0) / 100.0, rel=1e-6)


def test_builder_orders_by_field_and_excludes_single_period() -> None:
    """Points sort by field ascending; non-two-period runs land in `excluded`."""
    runs = _synthetic_runs(np.array([900.0, 600.0, 750.0]))
    runs.insert(0, _single_period_run(57000, 650.0))
    scan = build_rf_difference_scan(runs, method="differential", order_key="field")
    assert scan.n_points == 3
    assert list(scan.x) == sorted(scan.x)
    assert list(scan.x) == [600.0, 750.0, 900.0]
    assert (57000, "not a two-period (red/green) run") in scan.excluded


def test_builder_window_restricts_integration() -> None:
    """A t_min/t_max window outside the curve excludes the run with a reason."""
    runs = _synthetic_runs(np.array([770.0, 860.0]))
    scan = build_rf_difference_scan(runs, method="differential", t_min=100.0, t_max=200.0)
    assert scan.n_points == 0
    assert len(scan.excluded) == 2


def test_builder_rejects_curve_only_pooled_reconstruction() -> None:
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
    assert scan.n_points == 0
    assert "reload raw period data" in scan.excluded[0][1]


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
    scan = build_rf_difference_scan(_synthetic_runs(fields), method="differential")
    assert scan.n_points == fields.size

    result = fit_rf_resonance(scan, nu_rf=_NU_RF, a_mu=_A_MU_TRUE - 2.0, a_p=_A_P_TRUE - 3.0)
    assert result.success
    assert result.parameters["A_mu"].value == pytest.approx(_A_MU_TRUE, abs=0.5)
    assert result.parameters["A_p"].value == pytest.approx(_A_P_TRUE, abs=1.0)
    # nu_RF is held fixed.
    assert result.parameters["nu_RF"].value == pytest.approx(_NU_RF, abs=1e-9)


# --- corpus-conditional recovery ----------------------------------------------


_RF_DIR = corpus_path("Chemistry/Muon spectroscopy of benzene/data/RF resonance")


@requires_corpus(_RF_DIR)
def test_benzene_rf_count_integrals_match_workflow() -> None:
    """Matched raw observations test the changed operation, without fitted truth."""
    from dataclasses import replace

    from asymmetry.core.io import load
    from asymmetry.core.workflow.integral_scan import build_integral_scan
    from asymmetry.core.workflow.reduction import ReductionSettings

    assert _RF_DIR is not None
    runs = []
    for number in (56426, 56435, 56450, 56462):
        dataset = load(str(_RF_DIR / f"{number}.nxs"))
        grouping = dict(
            dataset.run.grouping,
            alpha=1.0,
            beta=1.0,
            deadtime_correction=False,
            background_correction=False,
        )
        grouping.pop("alpha_policy", None)
        runs.append(replace(dataset.run, grouping=grouping))
    scan = build_rf_difference_scan(runs, t_min=0, t_max=1)
    workflow = build_integral_scan(
        runs, ReductionSettings(period="green_minus_red"), t_min=0, t_max=1
    )
    assert scan.n_points == workflow.n_points == 4
    np.testing.assert_allclose(scan.value, workflow.value, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(scan.error, workflow.error, rtol=1e-12, atol=1e-14)
    mean = build_rf_difference_scan(runs, method="differential", t_min=0, t_max=1)
    assert np.any(np.abs(mean.value - scan.value) > 1e-8)


def test_mixed_historical_and_count_rf_points_keep_reports_and_pair_positions():
    from dataclasses import replace

    from asymmetry.core.fitting.scan_observations import scan_fit_input
    from asymmetry.core.transform.preparation import histogram_observation
    from tests.core.test_integrated_observation_validity import shared_reference_run

    historical = _two_period_run(10, 100.0, green_minus_red=0.0)
    time = np.array([0.0, 1.0])
    historical.grouping["period_reduced"] = [
        (time, np.zeros(2), np.zeros(2)),
        (time, np.array([20.0, 80.0]), np.array([10.0, 0.0])),
    ]
    run, reference = shared_reference_run(green=(700.0, 350.0))
    datasets = []
    for number, field in ((20, 200.0), (30, 300.0)):
        source = replace(
            run, run_number=number, source_file=f"sample-{number}.nxs", metadata={"field": field}
        )
        datasets.append(
            MuonDataset(
                np.array([]),
                np.array([]),
                np.array([]),
                run=source,
                histogram_observation=histogram_observation(source, reference=reference),
            )
        )
    scan = build_rf_difference_scan([datasets[1], historical, datasets[0]], method="differential")
    assert scan.run_numbers == [10, 20, 30]
    assert scan.value[0] == pytest.approx(0.2)
    assert scan.error[0] == pytest.approx(0.1)
    assert len(scan.observations) == 3
    assert scan.observations[0].report.zero_variance == 1
    assert scan.correlated_pairs == ((1, 2),)
    with pytest.raises(ValueError, match="reuse counting observations"):
        scan_fit_input(scan)
    assert scan_fit_input(scan, x_max=200.0).n_points == 2
