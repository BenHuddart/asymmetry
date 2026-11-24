"""Generate the quickstart preset, analyse it, and apply the wizard's fit."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


class QuickstartFirstFitScenario(Scenario):
    name = "quickstart_first_fit"
    description = "Quickstart's first fit applied through the Fit Wizard."
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_generate_preset("ag_zf_kt")
        dataset = window._data_browser.get_all_datasets()[0]
        window._on_dataset_selected(dataset.run_number)
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        single_tab = window._fit_panel._single_tab
        assert single_tab._fit_wizard_btn.text() == "Wizard…"
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
        deadline = time.monotonic() + 60.0
        while wizard._tasks.active_count and time.monotonic() < deadline:
            _process_events_for(milliseconds=50)
        if wizard._tasks.active_count:
            wizard.close()
            raise TimeoutError("Quickstart wizard analysis did not finish within 60 seconds")
        recommendation = wizard._recommendation
        assert recommendation is not None
        assessment = recommendation.recommended_assessment
        assert assessment.is_successful
        assert assessment.template.title == "Static GKT + Constant", assessment.template.title
        assert wizard._answer_card._apply_btn.text() == "Apply this fit"
        assert wizard._answer_card._apply_btn.isEnabled()
        wizard._answer_card._apply_btn.click()
        wizard.close()
        _, _, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y_min, y_max)
        window._clear_dirty()
        return window


register(QuickstartFirstFitScenario())
