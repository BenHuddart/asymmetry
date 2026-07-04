"""Tests for the MainWindow global-fit *studies* registry (Phase 2).

Covers the registry that replaced the trend panel's single-slot
``last_cross_group_fit``: create/update, project round-trip, legacy migration,
staleness, and off-thread Refit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.project import CURRENT_SCHEMA_VERSION  # noqa: E402
from asymmetry.core.representation.global_fit_study import (  # noqa: E402
    compute_group_input_digest,
)
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from tests._qt_helpers import wait_for  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


# ── synthetic builders ──────────────────────────────────────────────────────


def _groups(param: str = "A", *, scale: float = 1.0) -> list[ParameterGroupData]:
    return [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0, 300.0], dtype=float),
            y=np.array([0.20, 0.15, 0.10], dtype=float) * scale,
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=10.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([100.0, 200.0, 300.0], dtype=float),
            y=np.array([0.22, 0.16, 0.11], dtype=float) * scale,
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]


def _result() -> CrossGroupFitResult:
    return CrossGroupFitResult(
        success=True,
        chi_squared=2.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("m", value=-0.0005), Parameter("b", value=0.25)]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
        fixed_parameters=ParameterSet(),
    )


def _config(param: str = "A") -> dict:
    return {
        "model": ParameterCompositeModel(["Linear"]).to_dict(),
        "fit_x_min": None,
        "fit_x_max": None,
        "parameter_rows": [
            {"name": "m", "initial": -0.0005, "min": -1.0, "max": 1.0, "type": "Global"},
            {"name": "b", "initial": 0.25, "min": -1.0, "max": 1.0, "type": "Global"},
        ],
        "error_mode": "column",
        "error_value": None,
        "windows": None,
        "use_x_errors": False,
    }


def _output(param: str, x_key: str, groups: list[ParameterGroupData]) -> SimpleNamespace:
    return SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"]),
        x_key=x_key,
        fit_x_min=float("nan"),
        fit_x_max=float("nan"),
        config=_config(param),
    )


def _wait_idle(window: MainWindow, timeout_s: float = 10.0) -> None:
    win = window._global_parameter_fit_window
    if win is None:
        return
    wait_for(
        lambda: not win._fit_curve_compute_active,
        QApplication.instance(),
        timeout_s=timeout_s,
    )


# ── tests ────────────────────────────────────────────────────────────────────


def test_two_studies_persist_and_most_recent_displayed(mainwindow: MainWindow) -> None:
    g_a = _groups("A")
    g_b = _groups("D_2D")
    mainwindow._on_cross_group_fit_completed("A", g_a, _output("A", "field", g_a))
    mainwindow._on_cross_group_fit_completed("D_2D", g_b, _output("D_2D", "field", g_b))
    _wait_idle(mainwindow)

    assert len(mainwindow._global_fit_studies) == 2
    ids = list(mainwindow._global_fit_studies)
    id_a = mainwindow._cross_group_batch_id("A", "field", g_a)
    id_b = mainwindow._cross_group_batch_id("D_2D", "field", g_b)
    assert set(ids) == {id_a, id_b}

    # The most-recently-created study (D_2D) is the one shown.
    assert mainwindow._global_parameter_fit_window.batch_id() == id_b

    state = mainwindow.collect_project_state()
    assert state["schema_version"] == CURRENT_SCHEMA_VERSION
    assert len(state["global_fit_studies"]) == 2

    restored = MainWindow()
    restored.restore_project_state(state, "")
    _wait_idle(restored)

    assert len(restored._global_fit_studies) == 2
    assert set(restored._global_fit_studies) == {id_a, id_b}
    # Both studies carry a readable name and their x-label.
    names = {s.name for s in restored._global_fit_studies.values()}
    assert any("A vs" in n for n in names)
    # Most-recently-updated study displayed after load.
    assert restored._global_parameter_fit_window is not None
    assert restored._global_parameter_fit_window.batch_id() in {id_a, id_b}


def test_rerun_same_fit_updates_study_in_place(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)
    study = mainwindow._global_fit_studies[study_id]
    # User renames the study.
    study.name = "My renamed study"
    original_created = study.created

    # Re-run the same fit (same parameter/x/groups → same id).
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)

    assert len(mainwindow._global_fit_studies) == 1
    updated = mainwindow._global_fit_studies[study_id]
    assert updated.name == "My renamed study"  # preserved
    assert updated.created == original_created  # preserved
    assert updated.updated >= original_created  # refreshed (monotonic)


def test_default_name_collision_suffixed_for_new_study(mainwindow: MainWindow) -> None:
    # Two distinct studies whose default name "A vs ..." would collide: the
    # second gets a " (2)" suffix (they differ by group set → different ids).
    g_a = _groups("A")
    g_a2 = [
        ParameterGroupData(
            group_id="g2",
            group_name="G2",
            x=np.array([100.0, 200.0, 300.0], dtype=float),
            y=np.array([0.2, 0.15, 0.1], dtype=float),
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=30.0,
        ),
        _groups("A")[0],
    ]
    mainwindow._on_cross_group_fit_completed("A", g_a, _output("A", "field", g_a))
    mainwindow._on_cross_group_fit_completed("A", g_a2, _output("A", "field", g_a2))
    _wait_idle(mainwindow)
    names = [s.name for s in mainwindow._global_fit_studies.values()]
    assert len(names) == 2
    assert names[0] != names[1]
    assert any(n.endswith("(2)") for n in names)


def test_legacy_project_migrates_to_one_study(mainwindow: MainWindow) -> None:
    # Hand-build a v12 project whose panel state carries the legacy single-slot
    # ``last_cross_group_fit`` payload (the on-disk serialized shape).
    groups_payload = [
        {
            "group_id": "g0",
            "group_name": "G0",
            "x": [100.0, 200.0, 300.0],
            "y": [0.2, 0.15, 0.1],
            "yerr": [0.01, 0.01, 0.01],
            "group_variable_value": 10.0,
        },
        {
            "group_id": "g1",
            "group_name": "G1",
            "x": [100.0, 200.0, 300.0],
            "y": [0.22, 0.16, 0.11],
            "yerr": [0.01, 0.01, 0.01],
            "group_variable_value": 20.0,
        },
    ]
    legacy_fit = {
        "parameter_name": "Lambda",
        "x_key": "field",
        "fit_x_min": None,
        "fit_x_max": None,
        "config": {"config_key": "Lambda::field::g0|g1"},
        "config_key": "Lambda::field::g0|g1",
        "groups": groups_payload,
        "model": ParameterCompositeModel(["Linear"]).to_dict(),
        "fit_result": {
            "success": True,
            "chi_squared": 2.0,
            "reduced_chi_squared": 1.0,
            "message": "",
            "global_parameters": [
                {"name": "m", "value": -0.0005, "min": -1, "max": 1, "fixed": False}
            ],
            "global_uncertainties": {"m": 1e-5},
            "local_parameters": {"g0": [], "g1": []},
            "fixed_parameters": [],
            "local_uncertainties": {},
        },
    }

    base = mainwindow.collect_project_state()
    base["schema_version"] = 12
    base.pop("global_fit_studies", None)
    fp_state = dict(base.get("fit_parameters_state") or {})
    fp_state["last_cross_group_fit"] = legacy_fit
    base["fit_parameters_state"] = fp_state

    restored = MainWindow()
    restored.restore_project_state(base, "")
    _wait_idle(restored)

    assert len(restored._global_fit_studies) == 1
    study = next(iter(restored._global_fit_studies.values()))
    assert study.parameter_name == "Lambda"
    assert study.x_key == "field"
    assert len(study.groups) == 2
    # Window shows the migrated study.
    assert restored._global_parameter_fit_window is not None
    assert restored._global_parameter_fit_window.batch_id() == study.study_id


def test_exclusion_flips_study_stale_and_refit_clears_it(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)
    study = mainwindow._global_fit_studies[study_id]

    # Fresh study over the same groups is not stale.
    live_groups = _groups("A")

    def _assemble(param, x_key, group_ids):
        return list(live_groups)

    mainwindow._fit_parameters_panel.assemble_cross_group_groups = _assemble  # type: ignore
    stale, reason = mainwindow._study_staleness(study)
    assert not stale

    # Simulate an exclusion: the live rows now differ (one group's data shifts),
    # so the digest no longer matches the stored snapshot.
    shifted = _groups("A", scale=1.5)

    def _assemble_shifted(param, x_key, group_ids):
        return list(shifted)

    mainwindow._fit_parameters_panel.assemble_cross_group_groups = _assemble_shifted  # type: ignore
    stale, reason = mainwindow._study_staleness(study)
    assert stale
    assert compute_group_input_digest(shifted) != study.input_digest

    # Refit re-runs against the current (shifted) data off-thread and clears it.
    shifted_digest = compute_group_input_digest(shifted)
    mainwindow._on_global_fit_refit_requested(study_id)
    wait_for(
        lambda: mainwindow._global_fit_studies[study_id].input_digest == shifted_digest,
        QApplication.instance(),
        timeout_s=15.0,
    )
    _wait_idle(mainwindow)

    updated = mainwindow._global_fit_studies[study_id]
    assert compute_group_input_digest(shifted) == updated.input_digest
    stale, reason = mainwindow._study_staleness(updated)
    assert not stale


def test_deleted_source_series_marks_stale_and_disables_refit(
    mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Silence the modal warning that Refit raises when the source is gone (a
    # blocking exec() would hang the headless test).
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)
    study = mainwindow._global_fit_studies[study_id]

    def _assemble_missing(param, x_key, group_ids):
        return None  # a source series was deleted

    mainwindow._fit_parameters_panel.assemble_cross_group_groups = _assemble_missing  # type: ignore
    stale, reason = mainwindow._study_staleness(study)
    assert stale
    assert reason == "source series missing"

    # Displaying it disables the Refit button.
    mainwindow._display_global_fit_study(study_id)
    _wait_idle(mainwindow)
    window = mainwindow._global_parameter_fit_window
    assert window is not None
    assert window._stale_banner.isVisible()
    assert not window._refit_btn.isEnabled()

    # Refit is a no-op warning path (no study change) when the source is gone.
    mainwindow._on_global_fit_refit_requested(study_id)
    assert mainwindow._global_fit_studies[study_id] is study


# ── Phase 3 sidebar handlers: rename / duplicate / delete ────────────────────


def test_rename_updates_name_and_menu(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)

    mainwindow._on_global_fit_study_rename_requested(study_id, "My study")
    assert mainwindow._global_fit_studies[study_id].name == "My study"
    # The window title reflects the new name.
    window = mainwindow._global_parameter_fit_window
    assert window is not None
    assert "My study" in window.windowTitle()
    # The sidebar carries the new name too.
    entries = mainwindow._global_fit_sidebar_entries()
    assert any(name == "My study" for _sid, name, _stale in entries)


def test_duplicate_creates_suffixed_id_marked_stale(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)

    mainwindow._on_global_fit_study_duplicate_requested(study_id)
    _wait_idle(mainwindow)

    copy_id = f"{study_id}-copy"
    assert copy_id in mainwindow._global_fit_studies
    assert copy_id != study_id
    copy = mainwindow._global_fit_studies[copy_id]
    assert "(copy)" in copy.name
    # The copy's digest is deliberately mismatched so it reads as stale.
    assert copy.input_digest == "duplicated"
    # The copy is displayed and its banner is up.
    window = mainwindow._global_parameter_fit_window
    assert window is not None
    assert window.batch_id() == copy_id
    assert window._stale_banner.isVisible()


def test_duplicate_twice_avoids_id_collision(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)

    mainwindow._on_global_fit_study_duplicate_requested(study_id)
    _wait_idle(mainwindow)
    mainwindow._on_global_fit_study_duplicate_requested(study_id)
    _wait_idle(mainwindow)

    assert f"{study_id}-copy" in mainwindow._global_fit_studies
    assert f"{study_id}-copy2" in mainwindow._global_fit_studies


def test_delete_displays_remaining_study(mainwindow: MainWindow) -> None:
    g_a = _groups("A")
    g_b = _groups("D_2D")
    mainwindow._on_cross_group_fit_completed("A", g_a, _output("A", "field", g_a))
    mainwindow._on_cross_group_fit_completed("D_2D", g_b, _output("D_2D", "field", g_b))
    _wait_idle(mainwindow)
    id_a = mainwindow._cross_group_batch_id("A", "field", g_a)
    id_b = mainwindow._cross_group_batch_id("D_2D", "field", g_b)

    # Delete the displayed study (id_b, the most recent) → id_a displayed.
    mainwindow._on_global_fit_study_delete_requested(id_b)
    _wait_idle(mainwindow)
    assert id_b not in mainwindow._global_fit_studies
    assert id_a in mainwindow._global_fit_studies
    window = mainwindow._global_parameter_fit_window
    assert window is not None
    assert window.batch_id() == id_a


def test_delete_last_study_clears_window(mainwindow: MainWindow) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)

    mainwindow._on_global_fit_study_delete_requested(study_id)
    _wait_idle(mainwindow)
    assert not mainwindow._global_fit_studies
    window = mainwindow._global_parameter_fit_window
    assert window is not None
    assert not window.has_result()
    assert window.windowTitle() == "Global Parameter Fit"


def test_edit_fit_routes_through_fresh_fit_path(
    mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = _groups("A")
    mainwindow._on_cross_group_fit_completed("A", g, _output("A", "field", g))
    _wait_idle(mainwindow)
    study_id = mainwindow._cross_group_batch_id("A", "field", g)
    study = mainwindow._global_fit_studies[study_id]
    study.name = "Kept name"

    # Live groups available for re-assembly.
    live = _groups("A")
    mainwindow._fit_parameters_panel.assemble_cross_group_groups = (  # type: ignore
        lambda param, x_key, ids: list(live)
    )

    # Stub the dialog so exec() returns Accepted with an output over the same
    # groups (same batch id → in-place update).
    from types import SimpleNamespace

    fresh_output = SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"]),
        x_key="field",
        fit_x_min=float("nan"),
        fit_x_max=float("nan"),
        config=_config("A"),
        groups=live,
    )

    class _FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def output(self):
            return fresh_output

    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog.CrossGroupFitDialog", _FakeDialog
    )

    mainwindow._on_global_fit_edit_requested(study_id)
    _wait_idle(mainwindow)

    # In-place update: same id, name preserved.
    assert study_id in mainwindow._global_fit_studies
    assert mainwindow._global_fit_studies[study_id].name == "Kept name"
