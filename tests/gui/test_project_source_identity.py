"""Actual project restoration and collection preserve historical byte expectations."""

import pytest

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
from asymmetry.core.io import load
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, load_prepared_project, save_project
from asymmetry.gui.mainwindow import MainWindow
from tests.io.test_acquisition_content import acquisition

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("known", [False, True])
def test_actual_open_and_resave_retains_expected_and_observed(qapp, tmp_path, known):
    path = acquisition(tmp_path)
    actual = load(str(path)).run.source_binding.observed
    expected = actual if known else None
    entry = {"run_number": load(str(path)).run_number, "source_file": str(path)}
    if expected is not None:
        entry["source_binding"] = SourceBinding(expected, expected).to_dict()
    project = tmp_path / "source.asymp"
    save_project({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]}, project)
    window = MainWindow()
    window._open_project_file(str(project))
    ds = window._data_browser.get_dataset(entry["run_number"])
    assert ds is not None
    assert ds.run.source_binding == SourceBinding(expected, actual)
    state = window.collect_project_state()
    assert state["datasets"][0]["source_binding"] == SourceBinding(expected, actual).to_dict()
    save_project(state, project)
    reopened = load_prepared_project(project)
    assert reopened.sources[0].expected == expected
    window._clear_dirty()
    window.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_disk_reference_capture_uses_worker_and_cancel_discards(
    qapp, tmp_path, monkeypatch, cancel
):
    import threading
    import time

    from PySide6.QtCore import QThread, QTimer

    from asymmetry.core.io.reference_selection import reference_selection
    from asymmetry.core.transform.corrections import CorrectionUnavailableError
    from asymmetry.gui.utils.scientific_input import ScientificInputDialog

    path = acquisition(tmp_path)
    loaded = load(str(path))
    payload = dict(
        reference_selection(loaded.run),
        source_file=str(path),
        scale_request={"mode": "manual", "coefficient": 1.0},
    )
    threads = []
    entered = threading.Event()
    real = ContentIdentity.read

    def read(path, **kwargs):
        threads.append(QThread.currentThread())
        entered.set()
        if cancel:
            time.sleep(0.05)
        return real(path, **kwargs)

    monkeypatch.setattr(ContentIdentity, "read", read)
    window = MainWindow()
    if cancel:

        def stop():
            dialog = window.findChild(ScientificInputDialog)
            if dialog is None or not entered.is_set():
                QTimer.singleShot(1, window, stop)
            else:
                dialog.canceled.emit()

        QTimer.singleShot(1, window, stop)
        with pytest.raises(CorrectionUnavailableError, match="cancelled"):
            window._resolve_background_reference(dict(loaded.run.grouping, background_run=payload))
        assert not window._background_run_cache
    else:
        result = window._resolve_background_reference(
            dict(loaded.run.grouping, background_run=payload)
        )
        assert result.content_identity == loaded.run.source_binding.observed
    assert threads and all(t != qapp.thread() for t in threads)
    window._clear_dirty()
    window.close()


@pytest.mark.parametrize("unavailable", [False, True])
def test_combined_open_resave_keeps_own_expectations(qapp, tmp_path, monkeypatch, unavailable):
    import asymmetry.gui.utils.project_sources as sources_ui

    monkeypatch.setattr(
        sources_ui,
        "select_replacements",
        lambda parent, facts: tuple(
            i for i, f in enumerate(facts) if f.source.owner.active and f.status == "changed"
        ),
    )
    from dataclasses import replace

    from asymmetry.core.data.acquisition import constituent_sources_for_project
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from tests.core.test_workflow_reduction import _continuous_run

    entries, observations, terms = [], [], []
    old = ContentIdentity("c" * 64)
    for ordinal, number in enumerate((3, 4), 1):
        source = _continuous_run(0)
        source = replace(
            source, run_number=number, metadata=dict(source.metadata, run_number=number)
        )
        path = tmp_path / f"run{number}.nxs"
        write_nexus_v1(source, path)
        actual = load(str(path)).run.source_binding
        observations.append(actual.observed)
        entries.append(
            {"run_number": number, "source_file": str(path), "source_binding": actual.to_dict()}
        )
        terms.append(
            {
                "source_file": str(path),
                "source_run_number": number,
                "data_period_number": None,
                "dae_period_number": None,
                "input_ordinal": ordinal,
                "coefficient": 1.0,
                "source_binding": SourceBinding(old if ordinal == 1 else None, old).to_dict(),
            }
        )
    definition = {
        "combined_run_number": -1,
        "source_run_numbers": [3, 4],
        "constituent_sources": terms,
    }
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": entries,
        "combined_datasets": [] if unavailable else [definition],
    }
    project = tmp_path / "combined.asymp"
    save_project(state, project)
    window = MainWindow()
    window._open_project_file(str(project))
    if unavailable:
        window._data_browser.restore_unavailable_combination(definition, "unavailable control")
    combined = window._data_browser.get_dataset(-1)
    assert combined is not None
    facts = constituent_sources_for_project(combined)
    expected_terms = [old, None] if unavailable else [observations[0], None]
    assert [r["source_binding"]["expected"] for r in facts] == [
        v.to_dict() if v else None for v in expected_terms
    ]
    assert [r["coefficient"] for r in facts] == [1.0, 1.0]
    if not unavailable:
        assert [r["source_binding"]["observed"] for r in facts] == [
            v.to_dict() for v in observations
        ]
        assert combined.effective_recipe is not None
        assert all(d.expected != old for d in combined.effective_recipe.sources())
    saved = window.collect_project_state()
    save_project(saved, project)
    prepared = load_prepared_project(project)
    assert [d.expected for d in prepared.sources if d.role == "constituent"] == expected_terms
    window._clear_dirty()
    window.close()


def test_actual_open_missing_sample_and_combined_sources_survive_resave(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    identity = ContentIdentity("e" * 64)
    terms = [
        {
            "source_file": f"absent{n}.nxs",
            "source_run_number": n,
            "coefficient": c,
            "input_ordinal": i,
            "source_binding": SourceBinding(identity, None).to_dict(),
        }
        for i, (n, c) in enumerate(((3, 1.0), (4, -2.0)), 1)
    ]
    definition = {
        "combined_run_number": -1,
        "source_run_numbers": [3, 4],
        "operation": "subtract_signed",
        "constituent_sources": terms,
    }
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {
                "run_number": 3,
                "source_file": "absent3.nxs",
                "source_binding": SourceBinding(identity, None).to_dict(),
            }
        ],
        "combined_datasets": [definition],
    }
    path = tmp_path / "unavailable.asymp"
    save_project(state, path)
    monkeypatch.setattr(QMessageBox, "exec", lambda box: QMessageBox.StandardButton.No)
    window = MainWindow()
    window._open_project_file(str(path))
    saved = window.collect_project_state()
    assert saved["datasets"][0]["source_binding"] == state["datasets"][0]["source_binding"]
    assert saved["combined_datasets"][0]["constituent_sources"] == terms
    assert saved["combined_datasets"][0]["operation"] == "subtract_signed"
    save_project(saved, path)
    prepared = load_prepared_project(path)
    assert len(prepared.sources) == 3
    assert all(d.expected == identity for d in prepared.sources)
    window._clear_dirty()
    window.close()
