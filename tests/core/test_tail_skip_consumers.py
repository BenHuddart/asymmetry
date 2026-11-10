"""Accepted tail skips survive real scientific consumers and saved results."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.corrections import correction_warnings
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def tail_run(*, bad=True):
    base = 100 * np.exp(-np.arange(200) * 0.1 / MUON_LIFETIME_US) + 5
    backward = base * 0.8
    if bad:
        backward[99:] = 0
    return Run(
        run_number=42,
        histograms=[
            Histogram(a, 0.1, good_bin_end=199, count_variance=np.abs(a) + 1)
            for a in (base, backward)
        ],
        metadata={"run_number": 42, "field": 10.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "forward_group": 1,
            "backward_group": 2,
            "t0_bin": 0,
            "first_good_bin": 0,
            "last_good_bin": 199,
            "alpha": 1.0,
            "background_correction": True,
            "background_mode": "tail_fit",
        },
    )


def test_group_cohort_uses_all_selected_groups_and_retains_outcomes():
    from asymmetry.core.fourier.group_preparation import prepare_group_counts
    from asymmetry.core.fourier.grouped import build_group_signal_dataset
    from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum

    run = tail_run()
    with pytest.warns(UserWarning, match="subtraction skipped"):
        prepared = prepare_group_counts(run, [1, 2])
    for gid in (1, 2):
        dataset = build_group_signal_dataset(
            run, gid, prepared=prepared, center_signal=False, apply_lifetime_correction=False
        )
        np.testing.assert_array_equal(dataset.asymmetry, run.histograms[gid - 1].counts)
        assert not dataset.metadata["fourier_background_corrected"]
        assert dataset.metadata["fourier_background_value"] is None
        assert len(correction_warnings(dataset.metadata)) == 1
    with pytest.warns(UserWarning, match="subtraction skipped"):
        spectrum = compute_average_group_spectrum(run, GroupSpectrumConfig())
    assert correction_warnings(spectrum.metadata)
    selected = prepare_group_counts(run, [1])
    assert selected.backgrounds[1].outcome.status == "applied"


def test_maxent_keeps_common_skip_provenance():
    from asymmetry.core.maxent import MaxEntConfig, build_maxent_input

    with pytest.warns(UserWarning, match="subtraction skipped"):
        result = build_maxent_input(tail_run(), MaxEntConfig())
    assert len(correction_warnings(result.metadata)) == 1
    assert all(
        record["background"]["status"] == "skipped"
        for record in result.metadata["correction_outcomes"].values()
    )


def test_joint_count_inputs_are_unsubtracted_and_fit_summary_keeps_preflight():
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.result_summary import fit_result_summary

    run = tail_run()
    source = MuonDataset(np.arange(200) * 0.1, np.zeros(200), np.ones(200), run=run)
    datasets = build_grouped_time_domain_datasets(source, lifetime_corrected=False)
    for index, dataset in enumerate(datasets):
        np.testing.assert_array_equal(dataset.asymmetry, run.histograms[index].counts)
        assert len(correction_warnings(dataset.metadata)) == 1
    result = FitEngine().fit(
        datasets[0], lambda t, level: np.full(t.size, level), ParameterSet([Parameter("level", 10)])
    )
    assert result.warnings
    summary = fit_result_summary(result)
    preflight = summary["correction_metadata"]["count_background_preflight"]
    assert all(
        record["status"] == "skipped" and record["requested"] == "tail_fit"
        for record in preflight.values()
    )


def test_integral_and_period_integral_preserve_atomic_decision_and_payload():
    from asymmetry.core.transform.integral import build_field_scan, reduce_integral_run
    from asymmetry.core.workflow.integral_scan import field_scan_from_payload, field_scan_payload

    run = tail_run()
    with pytest.warns(UserWarning, match="subtraction skipped"):
        result = reduce_integral_run(run)
    forward, backward = [h.counts.sum() for h in run.histograms]
    variance_f, variance_b = [h.count_variance.sum() for h in run.histograms]
    assert result.value == pytest.approx((forward - backward) / (forward + backward))
    assert result.error**2 == pytest.approx(
        4 * (backward**2 * variance_f + forward**2 * variance_b) / (forward + backward) ** 4
    )
    assert result.corrections.background.status == "skipped"
    combined = replace(
        run,
        grouping=dict(
            run.grouping,
            period_mode="green_minus_red",
            period_histograms=[tail_run(bad=False).histograms, run.histograms],
        ),
    )
    with pytest.warns(UserWarning, match="subtraction skipped"):
        scan = build_field_scan([combined])
    restored = field_scan_from_payload(field_scan_payload(scan))
    assert restored.correction_outcomes == scan.correction_outcomes
    assert all(
        record.background.status == "skipped"
        for record in restored.correction_outcomes[42].values()
    )
    assert len(restored.correction_outcomes[42]) == 2


def test_cli_cold_warm_warnings_and_recovery(tmp_path, capsys):
    import json

    from asymmetry import cli
    from asymmetry.core.io.nexus_writer import write_nexus_v1

    folder = tmp_path / "runs"
    folder.mkdir()
    path = folder / "SIM00000042.nxs"
    write_nexus_v1(tail_run(), path)
    args = [
        "reduce",
        str(folder),
        "--runs",
        "42",
        "--background",
        "tail_fit",
        "--workdir",
        str(tmp_path / "work"),
    ]
    for recomputed in (True, False):
        cli.main([*args, "--json"])
        captured = capsys.readouterr()
        entry = json.loads(captured.out)["entries"][0]
        assert entry["recomputed"] == recomputed
        record = entry["run"]["correction_outcomes"]["red"]["background"]
        assert record["requested"] == "tail_fit" and record["status"] == "skipped"
        assert record["reason"]
        assert captured.err.count("Automatic tail background subtraction skipped") == 1
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        cli.main(args)
    captured = capsys.readouterr()
    assert "42" in captured.out
    assert "background subtraction skipped" in captured.err
    write_nexus_v1(tail_run(bad=False), path)
    cli.main([*args, "--json"])
    captured = capsys.readouterr()
    entry = json.loads(captured.out)["entries"][0]
    assert entry["recomputed"]
    assert entry["run"]["correction_outcomes"]["red"]["background"]["status"] == "applied"
    assert "background subtraction skipped" not in captured.err


def test_automatic_alpha_matches_atomic_native_counts_with_remainder():
    from asymmetry.core.project.profiles import (
        AlphaPolicy,
        BackgroundPolicy,
        DeadtimePolicy,
        GroupingProfile,
        ProfileFingerprint,
        resolve_effective_grouping,
    )
    from asymmetry.core.transform.rebin import BinningSpec

    red, green = tail_run(bad=False), tail_run()
    red.histograms[1].counts = (red.histograms[0].counts - 5) * 0.6 + 8
    red.grouping.update(
        period_histograms=[red.histograms, green.histograms],
        period_mode="green_minus_red",
        good_frames=1000.0,
    )
    profile = GroupingProfile(
        "test",
        ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        period_mode="green_minus_red",
        binning=BinningSpec(factor=3),
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
        background_policy=BackgroundPolicy(mode="tail_fit"),
        deadtime_policy=DeadtimePolicy(mode="manual", values=[0.01, 0.02]),
    )
    with pytest.warns(UserWarning, match="subtraction skipped"):
        grouping = resolve_effective_grouping(profile, red)
    totals = [
        np.sum(hist.counts / (1 - hist.counts * tau / (1000 * hist.bin_width)))
        for hist, tau in zip(red.histograms, (0.01, 0.02), strict=True)
    ]
    assert grouping["alpha"] == pytest.approx(totals[0] / totals[1], rel=1e-13)
    for mode in ("fixed", "calibrated"):
        fixed = resolve_effective_grouping(
            replace(profile, alpha_policy=AlphaPolicy(mode=mode, value=1.17)), red
        )
        assert fixed["alpha"] == 1.17


@pytest.mark.parametrize("module_name", ["fit_wizard", "global_fit_wizard"])
def test_wizard_result_round_trip_keeps_corrections_and_warnings(module_name):
    import importlib
    import json

    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionOutcomes

    module = importlib.import_module(f"asymmetry.core.fitting.{module_name}")
    skipped = CorrectionOutcome("background", "tail_fit", "skipped", "backward: empty tail")
    result = FitResult(
        objective=Objective.GAUSSIAN,
        success=True,
        warnings=[skipped.warning],
        correction_metadata={
            "correction_outcomes": {"red": CorrectionOutcomes(background=skipped).to_dict()}
        },
    )
    payload = json.loads(json.dumps(module._serialize_fit_result(result)))
    restored = module._deserialize_fit_result(payload)
    assert restored.correction_metadata == result.correction_metadata
    assert restored.warnings == result.warnings
    payload.pop("warnings")
    payload.pop("correction_metadata")
    legacy = module._deserialize_fit_result(payload)
    assert legacy.warnings == [] and legacy.correction_metadata == {}


def test_period_integrals_use_explicit_native_support_and_one_skip_warning():
    from asymmetry.core.transform.integral import integrate_run, reduce_period_integral

    red, green = tail_run(bad=False), tail_run()
    red.grouping.update(
        first_good_bin=20,
        last_good_bin=149,
        good_frames=1000,
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.01, 0.02],
    )
    green.grouping = dict(red.grouping)
    # The selected tail remains unusable, while explicit integration includes early context.
    green.histograms[1].counts[74:150] = 0
    combined = replace(
        red,
        grouping=dict(
            red.grouping,
            period_histograms=[red.histograms, green.histograms],
            period_mode="green_minus_red",
        ),
    )
    with pytest.warns(UserWarning, match="subtraction skipped") as caught:
        result = reduce_period_integral(combined, t_min=0, t_max=19.9)
    assert len(caught) == 1
    plain = [
        replace(run, grouping=dict(run.grouping, background_correction=False))
        for run in (red, green)
    ]
    (r, er), (g, eg) = [integrate_run(run, t_min=0, t_max=19.9) for run in plain]
    assert result.value == pytest.approx(g - r, abs=1e-14)
    assert result.error == pytest.approx(np.hypot(er, eg), rel=1e-14)
