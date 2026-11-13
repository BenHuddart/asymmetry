"""Changed acquisitions require a decision before the current session is replaced."""

import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.io import load
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, save_project
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.test_project_shell import _ds
from tests.io.test_acquisition_content import acquisition

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(str(args[-1])))
    instance = MainWindow()
    yield instance
    instance._clear_dirty()
    instance.close()


def test_changed_actual_open_cancel_keeps_unsaved_work(qapp, tmp_path, monkeypatch, window):
    path = acquisition(tmp_path)
    original = load(str(path))
    project = tmp_path / "changed.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {
                    "run_number": original.run_number,
                    "source_file": str(path),
                    "source_binding": original.run.source_binding.to_dict(),
                }
            ],
        },
        project,
    )
    acquisition(tmp_path, 5)
    questions = []

    def cancel(box):
        questions.append(box.windowTitle())
        return 0

    monkeypatch.setattr(QMessageBox, "exec", cancel)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: None)
    current = _ds(11)
    window._data_browser.add_dataset(current)
    window._mark_dirty()
    window._open_project_file(str(project))
    assert questions == ["Source Content Changed"]
    assert window._data_browser.get_dataset(11) is current
    assert window._data_browser.get_dataset(original.run_number) is None
    assert window._dirty
    assert window._current_project_path is None
    window._clear_dirty()
    window.close()


@pytest.mark.parametrize("choice", ["Open History Only", "Accept Replacement"])
def test_changed_open_resave_preserves_history(qapp, tmp_path, monkeypatch, choice, window):
    from copy import deepcopy

    from asymmetry.core.project import load_prepared_project, verify_project_sources
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    old_history = deepcopy(state["datasets"][0]["representations"])
    project = tmp_path / "history.asymp"
    save_project(state, project)
    acquisition(tmp_path, 5)
    clicks = []

    def decide(box):
        clicks.append(box.windowTitle())
        box._chosen = next(b for b in box.buttons() if b.text() == choice)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", decide)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._chosen)
    window._open_project_file(str(project))
    number = state["datasets"][0]["run_number"]
    dataset = window._data_browser.get_dataset(number)
    assert (dataset is not None) == (choice == "Accept Replacement")
    saved = window.collect_project_state()
    rep = saved["datasets"][0]["representations"]["time_fb_asymmetry"]
    assert rep["single_fits"] == old_history["time_fb_asymmetry"]["single_fits"]
    assert rep["result_metadata"] == old_history["time_fb_asymmetry"]["result_metadata"]
    save_project(saved, project)
    if dataset is not None:
        fresh = verify_project_sources(load_prepared_project(project), project).facts
        assert fresh[0].status == "matched"
        assert all(f.status == "changed" for f in fresh[1:])
        window._open_project_file(str(project))
        assert clicks == ["Source Content Changed"]
    else:
        assert saved["datasets"][0]["source_binding"] == state["datasets"][0]["source_binding"]
    window._clear_dirty()
    window.close()


@pytest.mark.parametrize("route", ["direct", "autosave", "shell"])
def test_all_open_routes_cancel_before_replacement(qapp, tmp_path, monkeypatch, window, route):
    from asymmetry.gui.shell import ProjectShell
    from tests.gui.test_project_autosave import _patch_recovery_choice
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    project = tmp_path / "routes.asymp"
    save_project(state, project)
    acquisition(tmp_path, 5)
    current = _ds(11)
    window._data_browser.add_dataset(current)
    window._mark_dirty()
    questions = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: questions.append(box.windowTitle()) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: None)
    if route == "direct":
        window.restore_project_state(state, str(project))
    elif route == "autosave":
        import os

        autosave = project.with_name("routes.autosave.asymp")
        save_project(state, autosave)
        os.utime(project, (1, 1))
        _patch_recovery_choice(monkeypatch, "load")
        # Recovery has its own question; source choice remains Cancel.
        import asymmetry.gui.utils.project_sources as ui

        monkeypatch.setattr(
            ui,
            "select_replacements",
            lambda parent, facts: questions.append("Source Content Changed") or None,
        )
        window._open_project_file(str(project))
        assert autosave.exists()
    else:
        shell = ProjectShell()
        page = shell.add_project()
        page._data_browser.add_dataset(_ds(13))
        shell.open_project(str(project))
        assert page._data_browser.get_dataset(13) is not None
        assert page._dirty
        for p in shell.pages():
            p._clear_dirty()
        shell.close()
    assert "Source Content Changed" in questions
    assert window._data_browser.get_dataset(11) is current
    assert window._dirty


@pytest.mark.parametrize("mode", ["none", "reference_run"])
def test_inactive_retained_reference_is_informational(qapp, tmp_path, monkeypatch, window, mode):
    from asymmetry.core.project import BackgroundPolicy, GroupingProfile, ProfileFingerprint
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    old = state["datasets"][0]["source_binding"]["expected"]
    reference = tmp_path / "reference.nxs"
    # Only owned input bytes are duplicated/changed.
    reference.write_bytes(path.read_bytes())
    with reference.open("ab") as stream:
        stream.write(b"owned annotation")
    profile = GroupingProfile(
        "inactive",
        ProfileFingerprint("test", 2),
        background_policy=BackgroundPolicy(
            mode,
            {
                "background_run": {
                    "source_file": str(reference),
                    "content_identity": old,
                    "run_number": 7,
                    "period_number": 1,
                }
            },
        ),
    )
    state["grouping_profiles"] = [profile.to_dict()]
    # An explicit per-run override keeps the retained profile unused.
    state["datasets"][0]["grouping_overrides"] = {
        "background_correction": False,
        "background_mode": "none",
        "background_run": {"source_file": str(reference), "content_identity": old},
    }
    project = tmp_path / "off.asymp"
    save_project(state, project)
    questions = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: questions.append(box.windowTitle()) or 0)
    window._open_project_file(str(project))
    assert not questions
    assert window._data_browser.get_dataset(7) is not None


def test_verify_read_change_is_unavailable_and_marks_partial(qapp, tmp_path, monkeypatch, window):
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    project = tmp_path / "race.asymp"
    save_project(state, project)
    load_paths = window._load_paths_with_progress

    def changed(paths):
        acquisition(tmp_path, 5)
        return load_paths(paths)

    monkeypatch.setattr(window, "_load_paths_with_progress", changed)
    window._open_project_file(str(project))
    assert window._data_browser.get_dataset(7) is None
    assert window._project_load_incomplete
    assert (
        window.collect_project_state()["datasets"][0]["source_binding"]
        == state["datasets"][0]["source_binding"]
    )


def test_verified_alias_uses_resolved_target(qapp, tmp_path, monkeypatch, window):
    import asymmetry.gui.utils.project_sources as ui
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    alias = tmp_path / "alias.nxs"
    alias.symlink_to(path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = acquisition(other_dir, 5)
    state["datasets"][0]["source_file"] = str(alias)
    project = tmp_path / "alias.asymp"
    save_project(state, project)
    verify = ui.verify_project_sources

    def retarget(*args, **kwargs):
        result = verify(*args, **kwargs)
        alias.unlink()
        alias.symlink_to(other)
        return result

    monkeypatch.setattr(ui, "verify_project_sources", retarget)
    window._open_project_file(str(project))
    actual = window._data_browser.get_dataset(7)
    assert actual.run.source_file == str(path)
    assert (
        actual.run.source_binding.expected.to_dict()
        == state["datasets"][0]["source_binding"]["expected"]
    )
    assert actual.run.source_binding.expected == actual.run.source_binding.observed


def test_legacy_reference_retarget_does_not_bypass_capture(qapp, tmp_path, monkeypatch, window):
    import asymmetry.gui.utils.project_sources as ui
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    directory = tmp_path / "reference"
    directory.mkdir()
    reference = acquisition(directory)
    state["datasets"][0]["grouping_overrides"] = {
        "groups": {"1": [1], "2": [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "alpha_source": "user",
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {
            "source_file": str(reference),
            "legacy_run_number": 7,
            "scale_request": {"mode": "manual", "coefficient": 1.0},
        },
    }
    project = tmp_path / "reference-race.asymp"
    save_project(state, project)
    verify = ui.verify_project_sources

    def retarget(*args, **kwargs):
        result = verify(*args, **kwargs)
        acquisition(directory, 5)
        return result

    monkeypatch.setattr(ui, "verify_project_sources", retarget)
    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(7)
    assert dataset is not None
    assert "after verification" in dataset.metadata["correction_error"]
    assert dataset.run.grouping["background_correction"]
    assert (
        dataset.run.grouping["background_run"].get("legacy_run_number") == 7
        or dataset.run.grouping["background_run"].get("run_number") == 7
    )


@pytest.mark.parametrize("assigned", [True, False])
def test_profile_reference_classification_uses_actual_fingerprint(
    qapp, tmp_path, monkeypatch, window, assigned
):
    from asymmetry.core.project import BackgroundPolicy, GroupingProfile, ProfileFingerprint
    from asymmetry.core.project.profiles import profile_fingerprint_for_run
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    identity = load(str(path)).run.source_binding.expected
    fingerprint = profile_fingerprint_for_run(load(str(path)).run)
    profile = GroupingProfile("same", fingerprint, groups={1: [1], 2: [2]}, active=True)
    unrelated = GroupingProfile(
        "same",
        ProfileFingerprint("unrelated", 99),
        background_policy=BackgroundPolicy(
            "reference_run",
            {
                "background_run": {
                    "source_file": str(path),
                    "content_identity": {"sha256": "e" * 64},
                }
            },
        ),
    )
    state["grouping_profiles"] = [profile.to_dict(), unrelated.to_dict()]
    if assigned:
        state["datasets"][0]["profile"] = "same"
    project = tmp_path / "fingerprint.asymp"
    save_project(state, project)
    questions = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: questions.append(box.windowTitle()) or 0)
    window._open_project_file(str(project))
    assert not questions
    assert window._data_browser.get_dataset(7).run.source_binding.expected == identity


def test_retained_series_source_status_does_not_hash_or_block_fitting(qapp, tmp_path, window):
    from dataclasses import replace

    from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
    from asymmetry.core.representation.series import FitSeries
    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    dataset = load(str(path))
    window._data_browser.add_dataset(dataset)
    recipe = state["datasets"][0]["representations"]["time_fb_asymmetry"]["result_metadata"][
        "effective_recipes"
    ][0]["effective_recipe"]
    series = FitSeries(
        "old",
        "time_fb_asymmetry",
        member_run_numbers=[7],
        results_by_run={
            7: {"computation_revision": "1", "input_attribution": {"effective_recipe": recipe}}
        },
    )
    assert "source contents" not in window._series_stale_reason(series, None)
    dataset.run = replace(
        dataset.run,
        source_binding=SourceBinding(ContentIdentity("e" * 64), ContentIdentity("e" * 64)),
    )
    assert "Historical source contents" in window._series_stale_reason(series, None)
    assert series.refit_block_reason() is None


@pytest.mark.parametrize("phase", ["question", "directory"])
@pytest.mark.parametrize("action", ["new", "close"])
def test_locate_cancellation_retains_current_work(
    qapp, tmp_path, monkeypatch, window, phase, action
):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog

    from tests.project.test_source_decisions import owned_project

    path, state = owned_project(tmp_path)
    project = tmp_path / "locate.asymp"
    save_project(state, project)
    path.unlink()
    current = _ds(11)
    window._data_browser.add_dataset(current)
    window._mark_dirty()
    entered = []
    original_question = QMessageBox.exec
    original_directory = QFileDialog.exec

    def stop():
        entered.append(phase)
        if action == "new":
            window._on_new_project()
        else:
            assert not window.close()

    def question(box):
        assert box.windowTitle() == "Data Files Not Found"
        if phase == "directory":
            return QMessageBox.StandardButton.Yes
        QTimer.singleShot(0, window, stop)
        return original_question(box)

    def directory(dialog):
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        QTimer.singleShot(0, window, stop)
        return original_directory(dialog)

    monkeypatch.setattr(QMessageBox, "exec", question)
    monkeypatch.setattr(QFileDialog, "exec", directory)
    window._open_project_file(str(project))
    assert entered == [phase]
    assert window._data_browser.get_dataset(11) is current
    assert window._dirty
    assert window._current_project_path is None
    assert "existing work retained" in window.statusBar().currentMessage()


def test_custom_decoder_legacy_unknown_stays_loadable_without_backfill(
    qapp, tmp_path, monkeypatch, window
):
    from asymmetry.core.io.base import BaseLoader, LoaderRegistry
    from asymmetry.core.project import (
        apply_source_decisions,
        load_prepared_project,
        verify_project_sources,
    )

    class OwnedLegacyLoader(BaseLoader):
        extensions = [".owned"]
        format_name = "Owned legacy decoder control"

        def load(self, filepath):
            from pathlib import Path

            dataset = _ds(7)
            dataset.run.source_file = filepath
            dataset.run.histograms[0].counts[:] = float(Path(filepath).read_text())
            return dataset

    monkeypatch.setattr(
        LoaderRegistry, "_loaders", dict(LoaderRegistry._loaders, **{".owned": OwnedLegacyLoader})
    )
    path = tmp_path / "legacy.owned"
    path.write_text("110")
    project = tmp_path / "legacy.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [{"run_number": 7, "source_file": str(path)}],
        },
        project,
    )
    window._open_project_file(str(project))
    loaded = window._data_browser.get_dataset(7)
    assert loaded is not None
    assert loaded.run.source_binding.expected is None
    assert loaded.run.source_binding.observed is None
    saved = window.collect_project_state()["datasets"][0]["source_binding"]
    assert saved == {"expected": None, "observed": None}
    verified = verify_project_sources(load_prepared_project(project), project)
    adopted = apply_source_decisions(verified, replacements=[0])
    with pytest.raises(ValueError, match="after verification"):
        adopted.check_capture(0, None)

    from asymmetry.core.project import BackgroundPolicy, GroupingProfile
    from asymmetry.core.project.profiles import profile_fingerprint_for_run

    state = load_prepared_project(project).state
    state["grouping_profiles"] = [
        GroupingProfile(
            "legacy",
            profile_fingerprint_for_run(loaded.run),
            groups={1: [1], 2: [2]},
            background_policy=BackgroundPolicy(
                "reference_run",
                {
                    "background_run": {
                        "source_file": str(path),
                        "run_number": 7,
                        "period_number": 1,
                        "scale_request": {"mode": "manual", "coefficient": 0.1},
                    }
                },
            ),
        ).to_dict()
    ]
    save_project(state, project)
    window._open_project_file(str(project))
    restored = window._data_browser.get_dataset(7)
    assert restored is not None
    assert restored.run.source_binding.expected is None
    assert restored.run.source_binding.observed is None


def test_computed_scan_uses_direct_captured_recipe_for_source_status(
    qapp, tmp_path, window, monkeypatch
):
    from dataclasses import replace

    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
    from tests.core.test_workflow_reduction import _continuous_run

    sources, datasets = [], []
    for number in (13001, 13002):
        run = _continuous_run(0)
        run = replace(
            run,
            run_number=number,
            metadata=dict(run.metadata, run_number=number, field=float(number)),
        )
        path = tmp_path / f"scan-{number}.nxs"
        write_nexus_v1(run, path)
        dataset = reduce_run(load(str(path)).run, ReductionSettings(alpha=1.0, alpha_source="user"))
        sources.append(path)
        datasets.append(dataset)
        window._data_browser.add_dataset(dataset)
    from tests._qt_helpers import wait_for

    messages = []
    monkeypatch.setattr(window, "_guidance_message", lambda *args, **kwargs: messages.append(args))
    window._plot_workspace.set_active_view("integral_scan")
    window._data_browser._table.selectAll()
    window._on_new_series_from_selection()
    wait_for(lambda: len(window._fit_panel.batch_datasets()) == 2, qapp)
    window._plot_panel.set_fit_range(0.1, 1.0)
    window._on_scan_requested()
    assert not messages, messages
    wait_for(lambda: window._alc_scan_series_id is not None, qapp)
    series = window._project_model.batch(window._alc_scan_series_id)
    assert series.is_computed
    assert all(
        summary["effective_recipe"] is not None and "input_attribution" not in summary
        for summary in series.results_by_run.values()
    )
    assert window._series_stale_reason(series, None) == ""
    with sources[0].open("ab") as stream:
        stream.write(b"owned source annotation")
    fresh = reduce_run(load(str(sources[0])).run, ReductionSettings(alpha=1.0, alpha_source="user"))
    window._data_browser.add_dataset(fresh)
    assert "Historical source contents" in window._series_stale_reason(series, None)
