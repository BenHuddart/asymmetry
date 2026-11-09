"""Shared calibration controls on owned count-domain synthetic acquisitions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data.calibration import calibration_control
from ._base import CaptureContext, Scenario, register


class AlphaCalibrationDialogScenario(Scenario):
    name = "alpha_calibration_dialog"
    description = (
        "Inline alpha calibration in the grouping window's Corrections column, with the "
        "shared before/after (α = 1 ↔ α̂) asymmetry preview."
    )
    size = (1220, 780)
    fit_beta = False
    vector = False
    method = "tf_calibration_fit"

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = calibration_control(
            beta=0.8 if self.fit_beta else 1.0,
            vector=self.vector,
            relaxing=self.method == "general",
        )

        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)

        section = dialog._alpha_section
        section._method_combo.setCurrentIndex(section._method_combo.findData(self.method))
        section._fit_beta.setChecked(self.fit_beta)
        dialog._alpha_card.set_expanded(True)
        section._on_estimate()
        _pump_until(lambda: section._tasks.active_count == 0)
        if not section.pending():
            raise RuntimeError(section._result_label.text())
        preview = dialog._preview_pane
        preview.flush()
        _pump_until(
            lambda: (
                preview._state is not None
                and preview._tasks.active_count == 0
                and not preview._debounce.isActive()
            )
        )
        corr_scroll = dialog._corrections_scroll
        corr_scroll.ensureWidgetVisible(section)
        corr_scroll.horizontalScrollBar().setValue(0)
        _pump_events(80)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        # Estimating α leaves the correction draft dirty, so ``close()`` would
        # otherwise hit the dialog's "Discard changes" guard. Nothing here is
        # ever saved and the grab is already on disk, so discard the draft
        # explicitly. (The capture harness also auto-dismisses modals — see
        # ``_base.auto_dismiss_modals`` — but stating the intent here keeps the
        # scenario readable and lets the dialog actually close.)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 10_000) -> None:
    """Pump a nested event loop until *predicate* holds (or the timeout lapses).

    The estimate lands via a queued cross-thread signal, so the loop must be
    entered for the callback to run; the timeout is only a backstop.
    """
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(int(timeout_ms))
    loop.exec()
    check.stop()
    guard.stop()


class FreeBetaScenario(AlphaCalibrationDialogScenario):
    name = "calibration_free_beta"
    fit_beta = True


class DecayBalanceScenario(AlphaCalibrationDialogScenario):
    name = "calibration_decay_balance"
    method = "general"


class VectorCalibrationScenario(AlphaCalibrationDialogScenario):
    name = "calibration_vector"
    vector = True


for scenario in (
    AlphaCalibrationDialogScenario(),
    FreeBetaScenario(),
    DecayBalanceScenario(),
    VectorCalibrationScenario(),
):
    register(scenario)
