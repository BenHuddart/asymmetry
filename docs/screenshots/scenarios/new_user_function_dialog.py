"""New User Function dialog mid-authoring, in a valid state.

Drives :class:`~asymmetry.gui.windows.new_user_function_dialog.NewUserFunctionDialog`
with a small, realistic component — a stretched exponential modulating a
Larmor oscillation, ``A*exp(-(lam*x)**beta)*cos(2*pi*f*x + phi)`` — filled in
with sensible start values so the live preview draws a few visible cycles
under a stretched-exponential envelope. Validation is triggered directly (not
via the debounce timer) so the captured state shows the green "Function is
valid" status and OK enabled, without writing or registering anything (the
dialog is never accepted). Companion to :doc:`/reference/user_functions`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ._base import CaptureContext, Scenario, register


class NewUserFunctionDialogScenario(Scenario):
    name = "new_user_function_dialog"
    description = "New User Function dialog authoring a stretched-exponential oscillation."
    size = (620, 760)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.new_user_function_dialog import NewUserFunctionDialog

        dialog = NewUserFunctionDialog("component", domain="time")
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)

        # Fill in a realistic small function. StretchedOsc is not a registered
        # name, so validation passes global uniqueness; every bare name in the
        # formula (lam, beta, f, phi, plus the seeded amplitude A) is declared
        # as a parameter with a start value tuned for a legible preview.
        dialog._name_edit.setText("StretchedOsc")
        dialog._description_edit.setText(
            "Stretched-exponential envelope on a Larmor oscillation"
        )
        dialog._formula_edit.setText("A*exp(-(lam*x)**beta)*cos(2*pi*f*x + phi)")

        # Seed the parameter table: the pre-seeded A row plus the four detected
        # parameters, with start values chosen for a clear envelope + a few
        # visible cycles over the 0-32 us preview grid.
        _set_param(dialog, "A", 0.2)
        _add_param(dialog, "lam", 0.08)
        _add_param(dialog, "beta", 0.6)
        _add_param(dialog, "f", 0.15)
        _add_param(dialog, "phi", 0.0)

        # Trigger validation and the preview redraw deterministically rather
        # than waiting on the 300 ms debounce timer.
        dialog._run_validation()
        _pump_events(150)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _set_param(dialog: object, name: str, value: float) -> None:
    """Set the first parameter row (the pre-seeded amplitude) to *name*/*value*."""
    table = dialog._param_table  # type: ignore[attr-defined]
    if table.rowCount() == 0:
        _add_param(dialog, name, value)
        return
    table.item(0, 0).setText(name)
    spin = table.cellWidget(0, 1)
    spin.setValue(float(value))


def _add_param(dialog: object, name: str, value: float) -> None:
    dialog._append_param_row(name, float(value))  # type: ignore[attr-defined]


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(NewUserFunctionDialogScenario())
