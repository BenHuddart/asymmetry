"""Fit the acquired ARGUS copper run through the public Quickstart controls."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog

import asymmetry
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from asymmetry.gui.shell import ProjectShell

from .._base import _process_events_for
from ._corpus import CorpusScenario, corpus_path, register
from .copper_diffusion import EXAMPLE

ARGUS_COPPER_RUN = f"{EXAMPLE}/Data/ARGUS00076935.nxs"
_MODEL = "StaticGKT_ZF + Constant"


def trigger_menu(shell: ProjectShell, menu_name: str, label: str) -> None:
    menu_action = next(
        action
        for action in shell._menu_bar.actions()
        if action.text().replace("&", "") == menu_name
    )
    menu = menu_action.menu()
    action = next(item for item in menu.actions() if item.text().replace("&", "") == label)
    action.trigger()
    _process_events_for(milliseconds=50)


def wait_for_quickstart(predicate: Callable[[], bool], stage: str) -> None:
    deadline = time.monotonic() + 60
    while not predicate() and time.monotonic() < deadline:
        _process_events_for(milliseconds=50)
    if not predicate():
        raise TimeoutError(f"Quickstart {stage} did not finish within 60 seconds")


def fit_quickstart(source: Path) -> ProjectShell:
    """Open the NeXus run and fit the expression entered in the Single tab."""
    shell = ProjectShell()
    page = shell.add_project()
    shell.show()

    with patch.object(QFileDialog, "getOpenFileNames", return_value=([str(source)], "")):
        trigger_menu(shell, "File", "Open Data File(s)…")
    wait_for_quickstart(lambda: not page._bulk_load_active, "source loading")
    (dataset,) = page._data_browser.get_all_datasets()
    page._data_browser.select_runs([dataset.run_number])
    trigger_menu(shell, "Analysis", "Fit")
    single_tab = page._fit_panel._single_tab

    def enter_model(dialog: FitFunctionBuilderDialog) -> int:
        dialog._text_mode_button.click()
        dialog._text_edit.setPlainText(_MODEL)
        dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).click()
        return dialog.result()

    with patch.object(FitFunctionBuilderDialog, "exec", enter_model):
        single_tab._edit_model_btn.click()
    assert single_tab._composite_model.component_expression_string() == _MODEL
    single_tab._fit_btn.click()
    assert single_tab.wait_for_fit(timeout_ms=60_000), "Quickstart fit timed out"
    ((_, record),) = page._collect_latest_fit_records()
    assert record["success"] and record["model_name"] == _MODEL
    assert record["producing_version"] == asymmetry.__version__
    return shell


class QuickstartScenario(CorpusScenario):
    name = "corpus_quickstart_first_fit"
    description = "Single-tab static Gaussian Kubo–Toyabe fit of the acquired ARGUS copper run."
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> ProjectShell:
        shell = fit_quickstart(corpus_path(ARGUS_COPPER_RUN))
        page = shell.current_page()
        plot = page._plot_panel
        plot._x_min.setValue(0)
        plot._x_max.setValue(10)
        plot._x_max.editingFinished.emit()
        page._dock_log.hide()
        page.resizeDocks([page._dock_data_browser], [290], Qt.Orientation.Horizontal)
        page.resizeDocks([page._dock_fit], [440], Qt.Orientation.Horizontal)
        return shell

    def teardown(self, widget: ProjectShell) -> None:
        widget.current_page()._clear_dirty()
        super().teardown(widget)


register(QuickstartScenario())
