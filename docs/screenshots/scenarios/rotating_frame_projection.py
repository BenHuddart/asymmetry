"""The rotating-frame projection of a synthetic RF nutation run, after Auto-detect.

A two-period run on an idealised six-detector vector polarimeter
(:func:`~docs.screenshots.data.archetypes.make_rf_nutation_vector`, invented
numbers): B₀ = 110 G ∥ z, RF on in red with B₁ ∥ x at ν₀ ≈ 1.49 MHz, RF off
in green. The plot shows red with all three chips selected and the switch on
**Rotating**. ν_RF is typed, as a user would from the generator; **Auto-detect…**
then runs on its worker exactly as in the app, and its review is applied with
the rows it pre-ticks. P′_y carries the nutation into +y′, P′_x stays near
zero, and P_z swings between ±1.

The scenario's second capture is that review dialog
(``rotating_frame_review``), opened from the same state before Apply.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget

from ..data import make_rf_nutation_vector
from ._base import Scenario, _grab_at_dpr, _process_events_for, register

#: The generator frequency a user would type: ν₀ of B₀ = 110 G.
_NU_RF_TEXT = "1.49"


def _open_review(window):
    """Show the synthetic run rotated with ν_RF typed, run Auto-detect, return its review."""
    from asymmetry.gui.widgets.rotating_frame_review import FrameReviewDialog

    dataset = make_rf_nutation_vector()
    window._data_browser.add_dataset(dataset)
    window._data_browser.select_runs([dataset.run_number])
    panel = window._plot_panel
    bar = panel._projection_bar
    bar.set_selected(["P_x", "P_y", "P_z"])
    bar.selection_changed.emit(bar.selected_labels())
    if not panel._auto_x_btn.isChecked():
        panel._auto_x_btn.click()
    _process_events_for(milliseconds=60)
    bar._rotating_btn.click()
    edit = panel.frame_bar._edits["frequency_mhz"]
    edit.setText(_NU_RF_TEXT)
    edit.setModified(True)
    edit.editingFinished.emit()
    _process_events_for(milliseconds=60)

    panel.frame_bar._detect_btn.click()
    deadline = time.monotonic() + 30.0
    while not window.findChildren(FrameReviewDialog):
        if time.monotonic() > deadline:
            raise RuntimeError("Auto-detect did not open its review")
        QApplication.processEvents()
        time.sleep(0.01)
    return window.findChild(FrameReviewDialog)


class _MainWindowScenario(Scenario):
    """Builds the main window off-screen and closes it after the grab."""

    def _window(self):
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resize(1500, 1000)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        self._main_window = window
        return window

    def teardown(self, widget: QWidget) -> None:
        self._main_window._dirty = False
        self._main_window.close()
        self._main_window.deleteLater()
        super().teardown(widget)


class RotatingFrameProjectionScenario(_MainWindowScenario):
    name = "rotating_frame_projection"
    description = "P′_x, P′_y and P_z of a synthetic RF nutation run, frame filled by Auto-detect."
    size = (1000, 760)

    def build(self) -> QWidget:
        window = self._window()
        for dock in (
            window._dock_data_browser,
            window._dock_fit,
            window._dock_fit_parameters,
            window._dock_log,
        ):
            dock.hide()
        window.resize(1150, 980)
        review = _open_review(window)
        review._apply_btn.click()
        panel = window._plot_panel
        # Rotate first, then bunch (the bins stay independent): ×4 = 64 ns bins.
        panel.set_bunch_factor(4)
        if not panel._auto_y_btn.isChecked():
            panel._auto_y_btn.click()
        _process_events_for(milliseconds=120)

        # The figure is the plot panel alone: chips, frame bar and subplots.
        self.size = (panel.width(), panel.height())
        shot = QLabel()
        shot.setPixmap(_grab_at_dpr(panel, 2.0))
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(shot)
        return container


class RotatingFrameReviewScenario(_MainWindowScenario):
    name = "rotating_frame_review"
    description = "Auto-detect's review: each estimate beside the current value, pre-ticked."
    size = (720, 300)

    def build(self) -> QWidget:
        review = _open_review(self._window())
        review.setParent(None)
        hint = review.sizeHint()
        self.size = (max(hint.width(), 640), hint.height())
        return review


register(RotatingFrameProjectionScenario())
register(RotatingFrameReviewScenario())
