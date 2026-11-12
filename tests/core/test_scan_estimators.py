"""Independent changing-count controls through the production scan consumers."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import build_rf_difference_scan
from asymmetry.core.transform.preparation import histogram_observation
from asymmetry.core.workflow.integral_scan import build_integral_scan
from asymmetry.core.workflow.reduction import ReductionSettings


def two_bin_run(number=13001):
    periods = [
        [Histogram(np.array(f, dtype=float), 1.0), Histogram(np.array(b, dtype=float), 1.0)]
        for f, b in (([6000, 10], [4000, 90]), ([6000, 90], [4000, 10]))
    ]
    return Run(
        number,
        histograms=periods[0],
        metadata={"field": 100.0, "period_count": 2},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "beta": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 1,
            "period_count": 2,
            "period_histograms": periods,
            "period_mode": "green_minus_red",
        },
    )


def test_rf_default_and_workflow_use_count_integrals():
    run = two_bin_run()
    expected = (6090 - 4010) / 10100 - (6010 - 4090) / 10100
    assert expected == pytest.approx(0.015841584158415828, abs=1e-16)
    actual = build_rf_difference_scan([run], t_min=0, t_max=1)
    dataset = MuonDataset(
        np.arange(2.0),
        np.zeros(2),
        np.ones(2),
        metadata=dict(run.metadata),
        run=run,
        histogram_observation=histogram_observation(run),
    )
    workflow = build_integral_scan(
        [dataset], ReductionSettings(period="green_minus_red"), t_min=0, t_max=1
    )
    assert actual.value == pytest.approx([expected])
    assert workflow.value == pytest.approx([expected])


def test_rf_explicit_time_bin_mean_is_distinct():
    scan = build_rf_difference_scan([two_bin_run()], method="differential")
    assert scan.method == "differential"
    assert scan.value == pytest.approx([0.8])


@pytest.mark.parametrize("mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_shared_nonunit_balances_native_window_and_display_rebin(mode):
    from asymmetry.core.transform.integral import build_field_scan

    run = two_bin_run()
    run.grouping.update(alpha=0.7, beta=1.2, period_mode=mode)

    def ratio(f, b):
        return (f - 0.7 * b) / (1.2 * f + 0.7 * b)

    expected = {
        "red": ratio(6010, 4090),
        "green": ratio(6090, 4010),
        "green_minus_red": ratio(6090, 4010) - ratio(6010, 4090),
        "green_plus_red": ratio(12100, 8100),
    }[mode]
    for bunching in (1, 2, 20):
        altered = replace(run, grouping=dict(run.grouping, bunching_factor=bunching))
        scan = build_field_scan([altered], t_min=0, t_max=1)
        assert scan.value == pytest.approx([expected])


@pytest.mark.parametrize(
    "method,expected", [("integral", 0.015841584158415828), ("differential", 0.8)]
)
def test_actual_cli_records_nonflat_observable(
    workflow_folder, tmp_path, capsys, monkeypatch, method, expected
):
    import json

    import asymmetry.core.io as io
    from asymmetry import cli

    run = two_bin_run(102)
    dataset = MuonDataset(
        np.arange(2.0),
        np.zeros(2),
        np.ones(2),
        metadata=dict(run.metadata),
        run=run,
        histogram_observation=histogram_observation(run),
    )
    monkeypatch.setattr(io, "load", lambda path: dataset)
    cli.main(
        [
            "integral-scan",
            str(workflow_folder),
            "--runs",
            "102",
            "--period",
            "green-red",
            "--method",
            method,
            "--tmin",
            "0",
            "--tmax",
            "1",
            "--order",
            "run",
            "--json",
            "--workdir",
            str(tmp_path / "scan-work"),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["scan"]["method"] == method
    assert output["scan"]["points"][0]["value"] == pytest.approx(expected)
    stored = json.loads((tmp_path / "scan-work" / "scans" / "integral-scan.json").read_text())
    assert stored["scan"]["points"][0]["value"] == pytest.approx(expected)


def test_attached_run_is_not_count_provenance_and_explicit_curve_mean_survives():
    from asymmetry.core.transform.integral import build_field_scan, integrate_run

    run = two_bin_run()
    dataset = MuonDataset(np.arange(2.0), np.array([20.0, -80.0]), np.array([1.0, 2.0]), run=run)
    unavailable = build_field_scan([dataset])
    assert not unavailable.n_points
    assert "verified count preparation" in unavailable.excluded[0][1]
    assert integrate_run(dataset, method="differential")[0] == pytest.approx(-0.3)
    dataset.run = None
    assert integrate_run(dataset, method="differential")[0] == pytest.approx(-0.3)


def test_verified_snapshot_owns_effective_settings_and_reference_corrections(monkeypatch):
    import asymmetry.core.transform.integral as integral
    from asymmetry.core.transform.integral import build_field_scan
    from asymmetry.core.transform.preparation import histogram_observation

    run = two_bin_run()
    run.grouping.update(
        alpha=0.7,
        beta=1.2,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[4, 2],
    )
    observation = histogram_observation(run)
    dataset = MuonDataset(
        np.arange(2.0), np.zeros(2), np.ones(2), run=run, histogram_observation=observation
    )
    run.grouping.update(alpha=5, beta=9, background_correction=False)
    calls = []
    prepare = integral.corrected_period_counts

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(integral, "corrected_period_counts", counted)
    scan = build_field_scan([dataset], t_min=0, t_max=1)

    def ratio(f, b):
        return (f - 0.7 * b) / (1.2 * f + 0.7 * b)

    expected = ratio(6090 - 8, 4010 - 4) - ratio(6010 - 8, 4090 - 4)
    assert scan.value == pytest.approx([expected])
    assert calls == [1]
    assert set(scan.correction_outcomes[run.run_number]) == {"red", "green"}


def test_native_center_window_selects_whole_bins():
    run = two_bin_run()
    assert build_rf_difference_scan([run], t_min=1, t_max=1).value == pytest.approx([1.6])
    assert build_rf_difference_scan([run], t_min=0, t_max=0).value == pytest.approx([0.0])


def test_workflow_cannot_promote_arbitrary_attached_counts():
    run = two_bin_run()
    curve = MuonDataset(np.arange(2.0), np.array([-50.0, -50.0]), np.ones(2), run=run)
    scan = build_integral_scan([curve], ReductionSettings(), order_key="run")
    assert scan.n_points == 0
    assert "verified count preparation" in scan.excluded[0][1]
    mean = build_integral_scan([curve], ReductionSettings(), method="differential", order_key="run")
    assert mean.value == pytest.approx([-0.5])
    raw = build_integral_scan([run], ReductionSettings(period="green_minus_red"), order_key="run")
    assert raw.value == pytest.approx([0.015841584158415828])


def test_curve_mean_reduction_has_no_fabricated_correction_provenance():
    from asymmetry.core.transform.integral import reduce_integral_run

    curve = MuonDataset(np.arange(2.0), np.array([20.0, -80.0]), np.ones(2), run=two_bin_run())
    result = reduce_integral_run(curve, method="differential")
    assert result.value == pytest.approx(-0.3)
    assert result.corrections is None


def test_current_condition_coordinates_do_not_replace_verified_counts():
    from asymmetry.core.transform.integral import build_field_scan

    run = two_bin_run()
    observation = histogram_observation(run)
    curve = MuonDataset(
        np.arange(2.0),
        np.zeros(2),
        np.ones(2),
        metadata=dict(run.metadata),
        run=run,
        histogram_observation=observation,
    )
    crop = curve.time_range(0, 1)
    curve.metadata["field"] = curve.run.metadata["field"] = 200.0
    assert crop.metadata["field"] == 100
    run.grouping["period_histograms"][1][0].counts[:] = 1
    scan = build_field_scan([crop])
    assert scan.x == pytest.approx([200])
    assert scan.value == pytest.approx([0.015841584158415828])
    workflow = build_integral_scan([crop], ReductionSettings(period="green_minus_red"))
    assert workflow.x == pytest.approx([200])
    assert workflow.value == pytest.approx(scan.value)
