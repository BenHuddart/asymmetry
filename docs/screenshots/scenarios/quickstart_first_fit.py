"""Generate, save, reload, fit, assess, and export the public first analysis."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from weakref import finalize

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox

from asymmetry.gui.shell import ProjectShell
from asymmetry.gui.windows.simulate_dialog import SimulateDialog

from ._base import Scenario, _process_events_for, register


def _quickstart_action(shell, menu_name: str, label: str) -> None:
    menu_action = next(
        action
        for action in shell._menu_bar.actions()
        if action.text().replace("&", "") == menu_name
    )
    menu = menu_action.menu()
    action = next(item for item in menu.actions() if item.text().replace("&", "") == label)
    action.trigger()
    _process_events_for(milliseconds=50)


def _quickstart_wait(predicate, stage: str) -> None:
    deadline = time.monotonic() + 60
    while not predicate() and time.monotonic() < deadline:
        _process_events_for(milliseconds=50)
    if not predicate():
        raise TimeoutError(f"Quickstart {stage} did not finish within 60 seconds")


def run_quickstart(directory: Path) -> ProjectShell:
    """Follow the documented controls, replacing native file picking only."""
    shell = ProjectShell()
    first_page = shell.add_project()
    shell.show()
    _process_events_for(milliseconds=80)
    paths = {
        "Save Synthetic Run as NeXus": directory / "first_run.nxs",
        "Save Project As": directory / "first_analysis.asymp",
        "Export fit report": directory / "first_fit.txt",
    }

    def save_path(_parent, title, *_args, **_kwargs):
        return str(paths[title]), ""

    def generate(dialog):
        dialog.show()
        assert dialog._template_combo.currentData() == "ideal_pulsed_fb"
        values = {"A_1": 24, "Lambda": 0.3, "A_bg": 2}
        for row in range(dialog._param_table.rowCount()):
            name = dialog._param_table.item(row, 0).text()
            dialog._param_table.item(row, 1).setText(str(values[name]))
        dialog._events_spin.setValue(40)
        dialog._seed_check.setChecked(True)
        dialog._seed_spin.setValue(23)
        dialog._generate_button.click()
        _quickstart_wait(lambda: dialog._tasks.active_count == 0, "generation")
        assert dialog._last_run is not None
        dialog._save_button.click()
        assert paths["Save Synthetic Run as NeXus"].is_file()
        buttons = dialog.findChild(QDialogButtonBox)
        buttons.button(QDialogButtonBox.StandardButton.Close).click()
        return dialog.result()

    with (
        patch.object(QFileDialog, "getSaveFileName", side_effect=save_path),
        patch.object(SimulateDialog, "exec", generate),
    ):
        _quickstart_action(shell, "File", "Generate Synthetic Run…")
        _quickstart_action(shell, "File", "New Project")
        page = shell.current_page()
        assert page is not first_page
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
            shell._tabs.tabCloseRequested.emit(0)
        _process_events_for(milliseconds=80)
        assert shell.pages() == [page]
        with patch.object(
            QFileDialog,
            "getOpenFileNames",
            return_value=([str(paths["Save Synthetic Run as NeXus"])], ""),
        ):
            _quickstart_action(shell, "File", "Open Data File(s)…")
        _quickstart_wait(lambda: not page._bulk_load_active, "source loading")
        datasets = page._data_browser.get_all_datasets()
        assert len(datasets) == 1
        page._on_dataset_selected(datasets[0].run_number)
        _quickstart_action(shell, "Analysis", "Fit")
        page.resizeDocks([page._dock_data_browser], [320], Qt.Orientation.Horizontal)
        single_tab = page._fit_panel._single_tab
        single_tab._fit_wizard_btn.click()
        wizard = single_tab._fit_wizard_window
        assert wizard is not None
        zero_field = next(
            button
            for button in wizard._picker._direction_group.buttons()
            if button.text() == "Zero field"
        )
        if zero_field.isEnabled():
            zero_field.click()
        assert wizard._analyze_btn.text() == "Analyze"
        wizard._analyze_btn.click()
        _quickstart_wait(lambda: wizard._tasks.active_count == 0, "wizard analysis")
        recommendation = wizard._recommendation
        assert recommendation is not None
        assessment = recommendation.recommended_assessment
        assert assessment.is_successful
        assert assessment.template.model.component_names == ["Exponential", "Constant"]
        residuals = wizard._answer_card._residuals_toggle
        assert residuals.text() == "Show residuals"
        residuals.setChecked(True)
        _process_events_for(milliseconds=80)
        apply_button = wizard._answer_card._apply_btn
        assert apply_button.text() == "Apply this fit" and apply_button.isEnabled()
        apply_button.click()
        wizard.close()
        _, _, y_min, y_max = page._plot_panel.get_view_limits()
        page._plot_panel.set_view_limits(0, 10, y_min, y_max)
        _quickstart_action(shell, "File", "Save Project")
        _quickstart_wait(lambda: not page._project_save_active, "project saving")
        assert paths["Save Project As"].is_file()
        _quickstart_action(shell, "Analysis", "Export fit report…")
        assert paths["Export fit report"].is_file()

    page._clear_dirty()
    return shell


class QuickstartFirstFitScenario(Scenario):
    name = "quickstart_first_fit"
    description = "Public synthetic analysis with a saved source, applied wizard fit, and report."
    size = (1500, 920)
    requires_fit = True

    def build(self):
        directory = TemporaryDirectory(prefix="asymmetry-quickstart-")
        shell = run_quickstart(Path(directory.name))
        finalize(shell, directory.cleanup)
        return shell

    def teardown(self, widget):
        for page in widget.pages():
            page._clear_dirty()
        super().teardown(widget)


register(QuickstartFirstFitScenario())
