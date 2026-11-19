"""Real desktop launch paths prepare the same requests as headless callers."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.cross_group import (
    prepare_cross_group_fit,
    run_cross_group_fit_from_config,
)
from asymmetry.core.fitting.parameter_models import ParameterGroupData
from asymmetry.core.fitting.recipe_inputs import build_recipe_engine_inputs
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog
from asymmetry.gui.panels.fit import global_tab as batch_module
from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
from asymmetry.gui.windows.joint_fit_window import JointFitWindow, JointSeriesEntry
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui


def _trace(run: int) -> MuonDataset:
    time = np.linspace(0.0, 6.0, 121)
    return MuonDataset(
        time, 0.2 * np.exp(-0.4 * time) + 0.02, np.full_like(time, 0.01), run=Run(run_number=run)
    )


def _parameter_snapshots(by_run):
    return {
        key: [(p.name, p.value, p.min, p.max, p.fixed) for p in params]
        for key, params in by_run.items()
    }


def test_batch_launch_uses_core_recipe_values(qapp, monkeypatch):
    datasets = [_trace(1), _trace(2)]
    tab = GlobalFitTab(member_kind="runs")
    tab.set_datasets(datasets)
    tab._set_composite_model(CompositeModel.from_expression("Exponential + Constant"))
    tab._user_initial_values_by_run = {1: {"Lambda": 0.61}}
    captured = []

    def capture(panel, call, **callbacks):
        captured.append(call)
        return None

    monkeypatch.setattr(batch_module, "_start_fit_call", capture)
    expected = build_recipe_engine_inputs(
        tab.current_recipe(),
        tab._composite_model,
        datasets,
        values_by_run=tab._effective_initial_values_by_run(tab._parse_parameter_configuration()),
    )
    tab._run_global_fit()
    assert len(captured) == 1
    call = captured[0]
    assert call.args[2] == expected.global_params
    assert call.args[3] == expected.local_params
    assert _parameter_snapshots(call.args[4]) == _parameter_snapshots(expected.initial_params)


def test_joint_launch_uses_each_core_recipe_without_batch_overrides(qapp, monkeypatch):
    model = CompositeModel.from_expression("Exponential + Constant")
    recipe = {
        "parameters": [
            {"name": "A_1", "value": 0.18, "type": "Global", "bounds": "0, 1"},
            {"name": "Lambda", "value": 0.3, "type": "Local", "bounds": "0, 4"},
            {"name": "A_bg", "value": 0.02, "type": "Fixed", "bounds": ""},
        ],
        "fit_range": {"min": 0.2, "max": 5.0},
    }
    entries = [
        JointSeriesEntry(
            str(n),
            str(n),
            RepresentationType.TIME_FB_ASYMMETRY,
            "Exp",
            (n,),
            "1/1",
            "",
            model,
            {"A_1": "global", "Lambda": "local", "A_bg": "fixed"},
            recipe,
            str(n),
        )
        for n in (1, 2)
    ]
    datasets = {str(n): [_trace(n)] for n in (1, 2)}
    window = JointFitWindow()
    window.set_providers(lambda: entries, datasets.__getitem__)
    window.start_new(RepresentationType.TIME_FB_ASYMMETRY)
    for row in range(2):
        window._series_table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    window._on_suggest_clicked()
    captured = []

    def capture(task, **callbacks):
        captured.append(task)
        return None

    monkeypatch.setattr(window._tasks, "start", capture)
    from asymmetry.gui.windows import joint_fit_window as module

    problems = []
    monkeypatch.setattr(
        module,
        "run_joint_fit_with_curves",
        lambda member_problems, *args, **kwargs: problems.extend(member_problems),
    )
    window._on_run_clicked()
    assert len(captured) == 1
    from types import SimpleNamespace

    captured[0](SimpleNamespace(is_cancelled=lambda: False))
    assert len(problems) == 2
    for problem in problems:
        expected = build_recipe_engine_inputs(recipe, model, datasets[problem.key])
        assert problem.global_params == expected.global_params
        assert problem.local_params == expected.local_params
        assert (problem.t_min, problem.t_max) == (expected.t_min, expected.t_max)
        assert _parameter_snapshots(problem.initial_params) == _parameter_snapshots(
            expected.initial_params
        )
    window._set_busy(False)


def test_cross_group_live_fit_and_recorded_refit_share_inputs_and_results(qapp, monkeypatch):
    x = np.arange(1.0, 9.0)
    groups = [
        ParameterGroupData(str(i), str(i), x, 0.02 * x + b, np.full_like(x, 0.005), float(i))
        for i, b in enumerate((0.1, 0.4))
    ]
    dialog = CrossGroupFitDialog(parameter_name="Lambda", x_key="field", groups=groups)
    config = dialog._collect_config()
    expected = prepare_cross_group_fit(groups, config)
    captured = []
    original = dialog._start_fit_task

    def capture(task, on_done):
        captured.append(task.args[0])
        return original(task, on_done)

    monkeypatch.setattr(dialog, "_start_fit_task", capture)
    dialog._run_fit(0)
    wait_for(lambda: not dialog._fit_in_progress, qapp, timeout_s=10.0)
    assert len(captured) == 1
    assert captured[0].initial_params == expected.initial_params
    assert captured[0].parameter_bounds == expected.parameter_bounds
    for actual, wanted in zip(captured[0].groups, expected.groups, strict=True):
        np.testing.assert_array_equal(actual.x, wanted.x)
        np.testing.assert_array_equal(actual.y, wanted.y)
    replay = run_cross_group_fit_from_config(groups, dialog._last_config)
    assert dialog._result.success == replay.result.success
    assert dialog._result.chi_squared == pytest.approx(replay.result.chi_squared, abs=1e-20)
    assert _parameter_snapshots({0: dialog._result.global_parameters}) == _parameter_snapshots(
        {0: replay.result.global_parameters}
    )
