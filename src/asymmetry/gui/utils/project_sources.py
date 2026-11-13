"""Thin source-location and decision UI before deliberate project application."""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from asymmetry.core.project.sources import (
    apply_source_decisions,
    source_candidates,
    verify_project_sources,
)
from asymmetry.gui.tasks import check_task_cancelled
from asymmetry.gui.utils.project_preparation import ProjectPreparation


def select_replacements(parent, facts):
    """Return explicit current-owner selections, history only, or pre-apply cancel."""
    changed = tuple(
        i for i, f in enumerate(facts) if f.source.owner.active and f.status == "changed"
    )
    if not changed:
        return ()
    box, history, replace = source_change_question(parent, facts, changed)
    parent._bulk_load_cancel = box.reject
    box.exec()
    clicked = box.clickedButton()
    if clicked is history:
        return ()
    if clicked is replace:
        return changed
    return None


def source_change_question(parent, facts, changed):
    """Build the reviewable source question without entering a modal event loop."""
    box = QMessageBox(parent)
    box.setWindowTitle("Source Content Changed")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText("Current inputs differ from the content recorded in this project.")
    box.setInformativeText(
        "Open saved history without using these inputs, or accept their replacement "
        "and recompute supported current observations. Saved fits and calibrations stay historical. "
        "Arrays that the project did not save cannot be recovered in history mode.\n\n"
        + "\n".join(
            f"Input {i}: {facts[i].source.role}, run {facts[i].source.run_number}"
            for i in changed[:8]
        )
        + ("\nAdditional current owners are listed in Details." if len(changed) > 8 else "")
    )
    box.setDetailedText(
        "\n".join(
            f"Input {i}: {facts[i].source.role}, owner {facts[i].source.owner.address}\n"
            f"  {facts[i].candidate_file}"
            for i in changed
        )
    )
    history = box.addButton("Open History Only", QMessageBox.ButtonRole.ActionRole)
    replace = box.addButton("Accept Replacement", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box, history, replace


def prepare_source_application(parent, prepared, project_path):
    """Locate candidates, verify on the existing worker, then collect one decision."""
    candidates = source_candidates(prepared, project_path)
    missing = [
        source
        for source, candidate in zip(prepared.sources, candidates, strict=True)
        if source.owner.active and not Path(candidate).is_file() and source.source_file
    ]
    if missing:
        directory = locate_source_directory(parent, missing, project_path)
        if directory is None:
            return None
        if directory:
            candidates = source_candidates(prepared, project_path, directory=directory)
    dialog = ProjectPreparation(parent)
    parent._bulk_load_cancel = dialog.cancel_preparation
    dialog.setLabelText("Verifying project sources…")
    dialog.setMinimumDuration(0)
    verified = dialog.run(
        parent._tasks,
        operation=lambda worker: verify_project_sources(
            prepared,
            project_path,
            candidates=candidates,
            checkpoint=lambda: check_task_cancelled(worker),
            progress=lambda done, total: worker.progress.emit(
                int(100 * done / max(total, 1)), 100, "Verifying project sources…"
            ),
        ),
    )
    if verified is None:
        return None
    selected = select_replacements(parent, verified.facts)
    if selected is None:
        return None
    dialog = ProjectPreparation(parent)
    parent._bulk_load_cancel = dialog.cancel_preparation
    return dialog.run(
        parent._tasks,
        operation=lambda worker: apply_source_decisions(
            verified, replacements=selected, checkpoint=lambda: check_task_cancelled(worker)
        ),
    )


def locate_source_directory(parent, missing, project_path):
    """Return a proposed directory, an unavailable choice, or explicit cancellation."""
    import ntpath

    names = ", ".join(ntpath.basename(source.source_file) for source in missing[:5])
    box = QMessageBox(
        QMessageBox.Icon.Question,
        "Data Files Not Found",
        f"The following current input files could not be found:\n\n  {names}\n\n"
        "Would you like to locate the directory where these files are now stored?",
        QMessageBox.StandardButton.Yes
        | QMessageBox.StandardButton.No
        | QMessageBox.StandardButton.Cancel,
        parent,
    )
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    parent._bulk_load_cancel = box.reject
    answer = box.exec()
    if answer == QMessageBox.StandardButton.No:
        return ""
    if answer != QMessageBox.StandardButton.Yes:
        return None
    dialog = QFileDialog(parent, "Locate Data Directory", str(Path(project_path).absolute().parent))
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    parent._bulk_load_cancel = dialog.reject
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return None
    return dialog.selectedFiles()[0]


def apply_project_application(parent, application, project_path):
    """Publish deliberate decisions and retain the completed-item restore contract."""
    previous_restoring = parent._restoring_project
    parent._restoring_project = True
    parent._project_source_application = application
    parent._background_run_cache.clear()
    for fact in application.report():
        parent._log_panel.log(f"Source {fact['index']} ({fact['role']}): {fact['status']}")
    try:
        parent._restore_project_state_impl(application.prepared, project_path)
    finally:
        parent._restoring_project = previous_restoring


def series_source_reason(parent, series):
    """Qualify retained series using actual captured inputs, without reducing or hashing."""
    from asymmetry.core.fitting.input_attribution import source_context_status

    statuses = {
        source_context_status(
            summary.get("input_attribution") or summary,
            parent._data_browser.get_dataset(series.source_run_for(run)),
        )
        for run, summary in series.results_by_run.items()
    }
    if statuses - {"matched", "in_memory"}:
        return "Historical source contents are different, unavailable, or unverified; results remain available."
    return ""
