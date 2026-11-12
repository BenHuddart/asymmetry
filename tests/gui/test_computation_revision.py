"""GUI persistence and export retain the execution that produced a result."""

import json

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from tests.core.test_workflow_reduction import _continuous_run

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("revision", [None, "historical", "1"])
def test_parameter_result_decoration_and_cross_group_codec_keep_revision(qapp, revision):
    window = GlobalParameterFitWindow()
    panel = FitParametersPanel()
    model = ParameterCompositeModel(["Linear"])
    parameters = ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)])
    window._local_model_fits = {
        "Lambda": ParameterModelFit(
            parameter_name="Lambda",
            x_key="temperature",
            active=True,
            ranges=[
                ModelFitRange(
                    0.0,
                    10.0,
                    model,
                    parameters,
                    result=ParameterModelFitResult(
                        success=True, parameters=parameters, computation_revision=revision
                    ),
                )
            ],
        )
    }
    saved = json.loads(json.dumps(window._serialize_local_model_fits()))
    if revision is None:
        saved["Lambda"]["ranges"][0]["result"].pop("computation_revision")
    window._local_model_fits = window._deserialize_local_model_fits(saved)
    assert (
        window._serialize_local_model_fits()["Lambda"]["ranges"][0]["result"][
            "computation_revision"
        ]
        == revision
    )
    group = ParameterGroupData("owned", "owned", np.arange(3.0), np.ones(3), np.ones(3), 0.0)
    panel._last_cross_group_fit = {
        "model": model,
        "groups": [group],
        "fit_result": CrossGroupFitResult(True, 1.0, 1.0, computation_revision=revision),
    }
    saved = json.loads(json.dumps(panel._serialize_last_cross_group_fit()))
    if revision is None:
        saved["fit_result"].pop("computation_revision")
    panel._last_cross_group_fit = panel._deserialize_last_cross_group_fit(saved)
    assert panel._serialize_last_cross_group_fit()["fit_result"]["computation_revision"] == revision
    window.close()
    panel.close()


@pytest.mark.parametrize("revision", [None, "historical", "1"])
def test_export_of_historical_plot_preserves_execution_stamp(qapp, tmp_path, revision):
    panel = PlotPanel()
    data = reduce_run(_continuous_run(0), ReductionSettings())
    data.computation_revision = revision
    panel._current_datasets = [data]
    payload = panel.get_current_plot_export_data()[0]
    assert payload["computation_revision"] == revision
    path = tmp_path / "owned.dat"
    panel._write_data_file(path, payload)
    line = next(
        line for line in path.read_text().splitlines() if line.startswith("! computation_revision:")
    )
    assert json.loads(line.split(": ", 1)[1]) == revision
    assert data.computation_revision == revision
    panel.close()


@pytest.mark.parametrize("kind", ["single", "global"])
@pytest.mark.parametrize("revision", [None, "historical", "1"])
def test_real_wizard_restore_preserves_execution_and_blocks_historical_apply(qapp, kind, revision):
    from dataclasses import replace

    from asymmetry.core import computation
    from asymmetry.core.fitting.fit_wizard import (
        deserialize_fit_wizard_recommendation,
        serialize_fit_wizard_recommendation,
    )
    from asymmetry.core.fitting.global_fit_wizard import (
        deserialize_global_fit_wizard_recommendation,
        serialize_global_fit_wizard_recommendation,
    )
    from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
    from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
    from tests.gui.test_fit_wizard_window import _fake_recommendation as single_recommendation
    from tests.gui.test_global_fit_wizard_window import (
        _fake_recommendation as global_recommendation,
    )

    data = reduce_run(_continuous_run(0), ReductionSettings())
    data.metadata.update(run_number=7, field=10.0, temperature=5.0)
    if kind == "single":
        window = FitWizardWindow()
        window.set_analysis_context(data)
        result = single_recommendation(data)
        encode, decode = serialize_fit_wizard_recommendation, deserialize_fit_wizard_recommendation
    else:
        window = GlobalFitWizardWindow()
        window.set_analysis_context([data])
        result = global_recommendation([data])
        encode, decode = (
            serialize_global_fit_wizard_recommendation,
            deserialize_global_fit_wizard_recommendation,
        )
    payload = encode(replace(result, computation_revision=revision))
    if revision is None:
        payload.pop("computation_revision")
    restored = decode(payload)
    assert restored.computation_revision == revision
    assert encode(restored)["computation_revision"] == revision
    emitted = []
    window.apply_assessment_requested.connect(lambda *args: emitted.append(args))
    window.set_cached_recommendation(restored, signature=window._analysis_signature())
    assert window.current_recommendation() is restored
    assert window._analysis_stale == (not computation.is_current(revision))
    if not computation.is_current(revision):
        if kind == "single":
            window._apply_candidate(restored.recommended_key)
        else:
            window._apply_chosen()
        assert emitted == []
    window.close()


@pytest.mark.parametrize("revision", [None, "historical"])
def test_study_status_checks_unchanged_source_result_revisions(qapp, monkeypatch, revision):
    from asymmetry.core import computation
    from asymmetry.core.representation.global_fit_study import GlobalFitStudy
    from asymmetry.core.representation.series import FitSeries
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    source = FitSeries(
        "owned",
        "time_fb_asymmetry",
        results_by_run={
            7: {"computation_revision": computation.COMPUTATION_REVISION, "parameters": {"m": 1.0}}
        },
    )
    window._project_model.add_batch(source)
    group = ParameterGroupData(
        "owned",
        "owned",
        np.arange(3.0),
        np.ones(3),
        np.ones(3),
        0.0,
        input_computation_revisions=(computation.COMPUTATION_REVISION,),
    )
    study = GlobalFitStudy(
        "study",
        "owned",
        "m",
        "field",
        "field",
        "temperature",
        "temperature",
        "",
        "",
        source_group_ids=["owned"],
        groups=[group],
        model=ParameterCompositeModel(["Linear"]),
        result=CrossGroupFitResult(
            True, 1.0, 1.0, computation_revision=computation.COMPUTATION_REVISION
        ),
    )
    study = GlobalFitStudy.from_dict(json.loads(json.dumps(study.to_dict())))
    monkeypatch.setattr(window, "_compute_study_staleness", lambda study: (False, ""))
    assert window._study_staleness(study) == (False, "")
    source.results_by_run[7]["computation_revision"] = revision
    assert window._study_staleness(study)[0]
    assert source.results_by_run[7]["parameters"] == {"m": 1.0}
    assert study.groups[0].input_computation_revisions == (computation.COMPUTATION_REVISION,)
    window._dirty = False
    window.close()


def test_live_maxent_adapters_preserve_result_execution_on_conversion(qapp, monkeypatch):
    from dataclasses import replace

    from asymmetry.core import computation
    from asymmetry.core.maxent.engine import MaxEntConfig, maxent
    from asymmetry.core.representation import RepresentationType
    from asymmetry.gui.mainwindow import MainWindow
    from tests.core.test_maxent import _synthetic_run

    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=1,
        inner_iterations=1,
        fit_phases=False,
    )
    result = maxent(run, config)
    assert result.computation_revision == computation.COMPUTATION_REVISION
    window = MainWindow()
    window._maxent_active_run = run
    for revision in (None, "historical", computation.COMPUTATION_REVISION):
        captured = replace(result, computation_revision=revision)
        spectrum = captured.as_dataset(run, config)
        assert spectrum.computation_revision == revision
        window._record_frequency_maxent_recipe(run.run_number, config, spectrum, {})
        rep = window._project_model.representation(run.run_number, RepresentationType.FREQ_MAXENT)
        assert rep.result_metadata["computation_revision"] == revision
        window._record_maxent_reconstruction(run.run_number, config, captured)
        recon = window._project_model.representation(
            run.run_number, RepresentationType.TIME_MAXENT_RECON
        )
        assert recon.result_metadata["computation_revision"] == computation.COMPUTATION_REVISION
        assert recon.result_metadata["input_computation_revision"] == revision
    monkeypatch.setattr(computation, "COMPUTATION_REVISION", "next")
    assert result.as_dataset(run, config).computation_revision == "1"
    window._dirty = False
    window.close()


@pytest.mark.parametrize("revision", [None, "historical"])
def test_project_load_resave_preserves_snapshot_until_explicit_reduction(
    qapp, tmp_path, monkeypatch, revision
):
    from dataclasses import replace

    from asymmetry.core import computation
    from asymmetry.core.project.schema import load_project, save_project
    from asymmetry.core.representation import RepresentationType
    from asymmetry.gui.mainwindow import MainWindow

    source = tmp_path / "owned.nxs"
    source.write_bytes(b"owned loader-boundary fixture")

    def load_owned(path):
        run = replace(_continuous_run(0), run_number=7, source_file=str(source))
        run.metadata.update(run_number=7, source_file=str(source))
        return reduce_run(run, ReductionSettings())

    window = MainWindow()
    window._data_browser.add_dataset(load_owned(source))
    window.collect_project_state()
    representation = window._project_model.representation(7, RepresentationType.TIME_FB_ASYMMETRY)
    snapshot = representation.result_metadata["effective_recipes"][0]
    if revision is None:
        snapshot.pop("computation_revision")
    else:
        snapshot["computation_revision"] = revision
    path = tmp_path / "owned.asymp"
    save_project(window.collect_project_state(), path)
    restored = MainWindow()
    monkeypatch.setattr(restored, "_load_file", load_owned)
    restored.restore_project_state(load_project(path), str(path))
    current = restored._data_browser.get_dataset(7)
    assert current.computation_revision == computation.COMPUTATION_REVISION
    restored.collect_project_state()
    representation = restored._project_model.representation(7, RepresentationType.TIME_FB_ASYMMETRY)
    assert (
        representation.result_metadata["effective_recipes"][0].get("computation_revision")
        == revision
    )
    save_project(restored.collect_project_state(), path)
    assert (
        representation.result_metadata["effective_recipes"][0].get("computation_revision")
        == revision
    )
    applied, _outcomes = restored._apply_grouping_settings_to_dataset(
        current, dict(current.run.grouping)
    )
    assert applied
    assert (
        representation.result_metadata["effective_recipes"][0]["computation_revision"]
        == computation.COMPUTATION_REVISION
    )
    for gui in (window, restored):
        gui._dirty = False
        gui.close()
