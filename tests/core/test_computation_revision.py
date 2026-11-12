"""Execution compatibility composes with persistent scientific dependency owners."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry import cli
from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.fit_wizard import single_fit_build_signature
from asymmetry.core.fitting.input_attribution import fit_input_attribution, saved_fit_matches
from asymmetry.core.fitting.likelihood import dataset_observations
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.representation.base import FitSlot
from asymmetry.core.representation.series import FitSeries
from asymmetry.core.transform.integral import build_field_scan
from asymmetry.core.transform.preparation import wizard_observation_matches
from asymmetry.core.workflow.integral_scan import field_scan_from_payload, field_scan_payload
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from asymmetry.core.workflow.workdir import ReducedEntry, StaleWorkDirError, WorkDir, series_digest
from tests.core.conftest import SCAN_RUNS
from tests.core.test_cli_commands import _json_output
from tests.core.test_workflow_reduction import _continuous_run


def constant(time, c):
    return np.full_like(time, c)


def reduced_curve():
    return reduce_run(_continuous_run(0), ReductionSettings(alpha=1.0, alpha_source="user"))


@pytest.mark.parametrize("revision", [None, "historical"])
def test_fresh_fit_over_historical_observations_records_both_executions(revision):
    data = replace(reduced_curve(), computation_revision=revision).time_range(0.3, 0.5)
    assert data.computation_revision == revision
    fitted = FitEngine().fit(data, constant, ParameterSet([Parameter("c", 0.0)]))
    assert fitted.success
    assert fitted.computation_revision == computation.COMPUTATION_REVISION
    assert fitted.input_attribution["computation_revision"] == revision
    assert saved_fit_matches(FitSlot(result=fit_result_summary(fitted)), data)
    assert data.computation_revision == revision


@pytest.mark.parametrize("revision", [None, "historical", "1"])
def test_fit_codec_and_nested_history_never_stamp_on_load_or_resave(revision, monkeypatch):
    data = reduced_curve().time_range(0.3, 0.5)
    result = FitEngine().fit(data, constant, ParameterSet([Parameter("c", 0.0)]))
    payload = _serialize_fit_result(result)
    if revision is None:
        payload.pop("computation_revision")
    else:
        payload["computation_revision"] = revision
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "next")
    restored = _deserialize_fit_result(payload)
    assert _serialize_fit_result(restored)["computation_revision"] == revision
    summary = fit_result_summary(restored) | {"npar": 1, "ndof": result.dof}
    # The same nested summaries are used by grouped and joint member histories.
    slot = FitSlot.from_dict(FitSlot(result={"grouped": {"member": summary}, **summary}).to_dict())
    series = FitSeries(
        "owned", "time_fb_asymmetry", results_by_run={7: summary}, extra={"joint": slot.to_dict()}
    )
    reopened = FitSeries.from_dict(json.loads(json.dumps(series.to_dict())))
    assert reopened.results_by_run[7]["computation_revision"] == revision
    assert (
        reopened.extra["joint"]["result"]["grouped"]["member"]["computation_revision"] == revision
    )
    assert reopened.is_stale(None)
    assert not reopened.same_observations(series)
    assert slot.data_key() is None
    assert not saved_fit_matches(slot, data)
    assert slot.fitted_values() == summary["parameters"]


def test_revision_does_not_change_observation_or_recipe_identity(monkeypatch):
    data = reduced_curve()
    before = fit_input_attribution(dataset_observations(data), data)
    signature = single_fit_build_signature(None, dataset=data)
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "next")
    after_data = reduced_curve()
    after = fit_input_attribution(dataset_observations(after_data), after_data)
    assert before["identity"] == after["identity"]
    assert data.effective_recipe.identity == after_data.effective_recipe.identity
    assert data.histogram_observation.identity == after_data.histogram_observation.identity
    assert not wizard_observation_matches(after_data, signature)
    assert wizard_observation_matches(
        after_data, single_fit_build_signature(None, dataset=after_data)
    )


@pytest.mark.parametrize("revision", [None, "historical"])
def test_workdir_history_resave_and_completion_collectors(tmp_path, revision):
    wd = WorkDir(tmp_path)
    data = replace(reduced_curve(), computation_revision=revision)
    entry = ReducedEntry(
        7, "digest", "owned.bin", data.n_points, ReductionSettings(), {}, 1.0, "off", 1, 2, []
    )
    wd.write_reduced(data, entry)
    restored = wd.reduced(7)
    wd.write_reduced(restored, wd.entry(7))
    assert wd.entry(7).computation_revision == revision
    assert not wd.is_current(7, "digest")
    with pytest.raises(StaleWorkDirError, match="historical computation"):
        wd.reduced(7, require_current=True)
    record = {
        "computation_revision": revision,
        "kind": "series",
        "trend": {"rows": []},
        "runs": [7],
    }
    wd.write_series("old", record)
    wd.write_wizard(7, record)
    wd.write_scan("old", record)
    wd.write_fit(7, "old", record)
    wd.write_survey(record)
    for reader, args in [
        (wd.read_series, ("old",)),
        (wd.read_wizard, (7,)),
        (wd.read_scan, ("old",)),
        (wd.read_survey, ()),
    ]:
        with pytest.raises(StaleWorkDirError, match="historical computation"):
            reader(*args)
        assert reader(*args, require_current=False)["computation_revision"] == revision
    assert wd.fitted_runs() == set()
    assert wd.screened_runs() == []
    assert wd.fit_series() == {}


def test_derived_series_rejects_unchanged_historical_member_bytes(tmp_path):
    wd = WorkDir(tmp_path)
    member = {
        "computation_revision": "historical",
        "kind": "series",
        "results": [],
        "trend_fits": {},
    }
    wd.write_series("member", member)
    historical = wd.read_series("member", require_current=False)
    wd.write_series(
        "derived",
        {
            "computation_revision": computation.COMPUTATION_REVISION,
            "kind": "global-batch",
            "members": [
                {"series": "member", "fit": None, "digest": series_digest(historical, None)}
            ],
        },
    )
    with pytest.raises(StaleWorkDirError, match="historical computation"):
        wd.read_series("derived")
    assert wd.read_series("derived", require_current=False)["members"][0][
        "digest"
    ] == series_digest(historical, None)


@pytest.mark.parametrize("revision", [None, "historical"])
def test_scan_codec_preserves_historical_execution(revision):
    scan = build_field_scan(
        [reduced_curve()], method="integral", t_min=0.3, t_max=0.5, order_key="run"
    )
    payload = field_scan_payload(scan)
    payload["computation_revision"] = revision
    restored = field_scan_from_payload(payload)
    assert field_scan_payload(restored)["computation_revision"] == revision
    np.testing.assert_array_equal(restored.value, scan.value)
    assert restored.effective_recipe == scan.effective_recipe


def test_package_metadata_does_not_invalidate_real_reduction_reuse(
    workflow_folder, tmp_path, capsys, monkeypatch
):
    from asymmetry.core.workflow import workdir

    args = [
        "reduce",
        str(workflow_folder),
        "--runs",
        "102",
        "--workdir",
        str(tmp_path / "wd"),
        "--json",
    ]
    cli.main(args)
    first = json.loads(capsys.readouterr().out)["entries"][0]
    monkeypatch.setattr(workdir, "__version__", "documentation-only")
    cli.main(args)
    second = json.loads(capsys.readouterr().out)["entries"][0]
    assert first["digest"] == second["digest"]
    assert not second["recomputed"]
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "next")
    cli.main(args)
    third = json.loads(capsys.readouterr().out)["entries"][0]
    assert third["recomputed"]
    assert third["computation_revision"] == "next"


def test_historical_linear_period_cache_is_recomputed_by_reduce(tmp_path, capsys, monkeypatch):
    """Owned historical fixture: linear G+R values, without running an old executable."""
    from tests.core.test_pooled_periods import two_periods

    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "SIM00000001.nxs").write_bytes(b"owned unequal-exposure periods")
    run = two_periods(green=(270, 30))
    source = MuonDataset(np.arange(8.0), np.zeros(8), np.ones(8), run=run)
    monkeypatch.setattr(
        "asymmetry.core.workflow.reduction.load_reduction_source", lambda paths, period: source
    )
    wd = WorkDir(tmp_path / "wd")
    args = [
        "reduce",
        str(folder),
        "--runs",
        "1",
        "--period",
        "green_plus_red",
        "--json",
        "--workdir",
        str(wd.root),
    ]
    cli.main(args)
    capsys.readouterr()
    # Earlier curve addition gave 20% + 80% = 100%; the count pool gives 65%.
    historical = replace(
        wd.reduced(1),
        asymmetry=np.full(8, 100.0),
        computation_revision="before-count-pooled-periods",
    )
    wd.write_reduced(historical, wd.entry(1))
    np.testing.assert_array_equal(wd.reduced(1).asymmetry, 100.0)
    cli.main(args)
    assert json.loads(capsys.readouterr().out)["entries"][0]["recomputed"]
    fresh = wd.reduced(1)
    np.testing.assert_allclose(fresh.asymmetry, 100 * (330 - 70) / (330 + 70))
    np.testing.assert_allclose(fresh.error, 200 * np.sqrt(330 * 70 / 400**3))
    assert fresh.computation_revision == computation.COMPUTATION_REVISION


def test_reduce_reuses_the_cache_and_recomputes_when_a_setting_changes(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    args = [
        "reduce",
        str(workflow_folder),
        "--runs",
        str(SCAN_RUNS[0]),
        "--json",
        "--workdir",
        str(workdir),
    ]

    cli.main(args)
    assert _json_output(capsys)["entries"][0]["recomputed"] is True

    cli.main(args)
    assert _json_output(capsys)["entries"][0]["recomputed"] is False

    cli.main([*args, "--rebin", "2"])
    second = _json_output(capsys)["entries"][0]
    assert second["recomputed"] is True
    assert second["settings"]["rebin"] == 2


@pytest.mark.parametrize("revision", [None, "historical-computation"])
def test_reduce_recomputes_historical_execution_even_with_matching_digest(
    workflow_folder, tmp_path, capsys, revision
):
    workdir = tmp_path / "wd"
    args = [
        "reduce",
        str(workflow_folder),
        "--runs",
        str(SCAN_RUNS[0]),
        "--json",
        "--workdir",
        str(workdir),
    ]
    cli.main(args)
    _json_output(capsys)
    sidecar = workdir / "reduced" / f"{SCAN_RUNS[0]}.json"
    record = json.loads(sidecar.read_text())
    record["computation_revision"] = revision
    sidecar.write_text(json.dumps(record))
    cli.main(args)
    assert _json_output(capsys)["entries"][0]["recomputed"] is True


def test_observation_aliases_and_revision_exclusion_preserve_existing_identity():
    from asymmetry.core.transform.preparation import histogram_observation

    source = _continuous_run(0)
    shared = {"mode": "fixed", "value": 1.0}
    source.grouping["owned_shared_policies"] = [shared, shared]
    before = histogram_observation(source).identity
    assert computation.without_revision(source.grouping) is source.grouping
    shared["computation_revision"] = "historical"
    cleaned = computation.without_revision(source.grouping)["owned_shared_policies"]
    assert cleaned[0] is cleaned[1]
    assert histogram_observation(source).identity == before


@pytest.mark.parametrize("revision", [None, "historical"])
def test_reopened_derived_series_checks_selected_trend_execution(tmp_path, revision):
    wd = WorkDir(tmp_path)
    wd.write_series(
        "source",
        {
            "computation_revision": computation.COMPUTATION_REVISION,
            "kind": "series",
            "trend_fits": {"law": {"computation_revision": revision, "m": 1.0}},
        },
    )
    source = wd.read_series("source")
    wd.write_series(
        "derived",
        {
            "computation_revision": computation.COMPUTATION_REVISION,
            "kind": "fit-trend",
            "members": [{"series": "source", "fit": "law", "digest": series_digest(source, "law")}],
        },
    )
    with pytest.raises(KeyError, match="stale"):
        wd.read_series("derived")
    assert wd.read_series("derived", require_current=False)["members"][0][
        "digest"
    ] == series_digest(source, "law")


@pytest.mark.parametrize("revision", [None, "historical"])
def test_moment_execution_is_captured_before_trend_serialization(revision, monkeypatch):
    from asymmetry.core.fourier.moments import moments_trend_row, spectrum_moments

    axis = np.linspace(-3.0, 3.0, 61)
    result = spectrum_moments(
        axis, np.exp(-(axis**2)), uncertainty="none", x_range=(-3.0, 3.0), cutoff_fraction=0.0
    )
    assert result.computation_revision == computation.COMPUTATION_REVISION
    assert result.b_ave == pytest.approx(0.0, abs=1e-15)
    historical = replace(result, computation_revision=revision)
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "next")
    assert moments_trend_row(historical, run_number=7)["computation_revision"] == revision


def test_dependent_identity_retains_numerical_execution_but_ignores_display_metadata():
    series = {
        "computation_revision": "1",
        "results": [{"parameters": {"c": 2.0}}],
        "name": "first",
        "timestamp": "first",
        "trend_fits": {"law": {"computation_revision": "1", "m": 1.0, "plot": "first.png"}},
    }
    before = series_digest(series, "law")
    series.update(name="renamed", timestamp="later", asymmetry_version="documentation-only")
    series["trend_fits"]["law"]["plot"] = "later.png"
    assert series_digest(series, "law") == before
    series["computation_revision"] = "historical"
    assert series_digest(series, "law") != before


def test_publishable_calibration_execution_is_separate_from_recipe_identity():
    from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe

    facts = {
        "version": 1,
        "calibration": {
            "records": [
                {
                    "calibration_id": "owned",
                    "computation_revision": "historical",
                    "source_file": "/private/owned.bin",
                }
            ]
        },
    }
    recipe = EffectiveReductionRecipe.from_dict(facts)
    public = recipe.publishable()
    assert public["calibration"]["records"][0]["computation_revision"] == "historical"
    assert "source_file" not in public["calibration"]["records"][0]
    facts["calibration"]["records"][0]["computation_revision"] = "next"
    assert EffectiveReductionRecipe.from_dict(facts).identity == recipe.identity


def test_acquisition_context_aliases_survive_execution_metadata_removal():
    import pickle

    from asymmetry.core.transform.constituent_counts import AcquiredCounts
    from asymmetry.core.transform.projections import reduction_identity

    shared = {"alpha": 1.0}
    acquired = AcquiredCounts([], {"policy": shared}, {})
    grouping = {"policy": shared, "inputs": (acquired,)}
    original = pickle.dumps(grouping)
    shared["computation_revision"] = "historical"
    cleaned = computation.without_revision(grouping)
    assert pickle.dumps(cleaned) == original
    assert cleaned["policy"] is cleaned["inputs"][0].grouping["policy"]
    plain = {"groups": {1: [1], 2: [2]}, "alpha_policy": {"value": 1.0}}
    before = reduction_identity(plain)
    plain["alpha_policy"]["computation_revision"] = "historical"
    assert reduction_identity(plain) == before


@pytest.mark.parametrize("revision", [None, "historical"])
def test_representation_restore_computes_arrays_without_replacing_saved_snapshot(revision):
    from asymmetry.core.representation.time import TimeFBAsymmetry

    rep = TimeFBAsymmetry()
    run = _continuous_run(0)
    rep.ensure_computed(run)
    rep.result_metadata["effective_recipes"][0]["computation_revision"] = revision
    rep.invalidate()
    rep.ensure_computed(run, replace_history=False)
    assert rep.primary.computation_revision == computation.COMPUTATION_REVISION
    assert (
        rep.to_dict()["result_metadata"]["effective_recipes"][0]["computation_revision"] == revision
    )
    rep.invalidate()
    rep.ensure_computed(run)
    assert (
        rep.to_dict()["result_metadata"]["effective_recipes"][0]["computation_revision"]
        == computation.COMPUTATION_REVISION
    )
