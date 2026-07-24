"""Inline β (asymmetry balance) calibration in the grouping window's Corrections column.

Opens the Grouping window directly on a synthesised YBCO transverse-field run —
the same weak-TF calibration archetype :mod:`alpha_calibration_dialog` uses,
since β is measured on the same run type as α. β (the intrinsic-asymmetry
balance) is estimated **inline** in the **β (asymmetry balance)** card's
"Measure from run" section: a calibration-run picker, a **Protocol** selector
("Count fit (recommended)" / "Single-histogram ratio"), and an **Estimate β**
button. Pressing **Apply β** writes the result, focuses the β compare stage,
and drives the shared grouping preview (pinned below both columns), which
overlays the β = 1 "before" ghost against the calibrated-β "after" curve.
Companion to :doc:`/reference/detector_grouping` and
:doc:`/reference/data_reduction/beta_calibration`.

The count-fit protocol's real estimate (:func:`fit_fb_alpha` with
``estimate_beta=True``) is an iminuit count-domain fit — the same fitting
stack ``alpha_count_calibration`` marks ``requires_fit = True`` for, because
it trips on numpy >= 2.3 in dev environments (see docs/README.md). Rather than
tie this docs screenshot to that stack (and to a real fit's convergence),
this scenario patches ``estimate_beta_detailed`` at the same seam
``tests/gui/test_beta_section.py`` patches it — the module-level name the
β-section worker looks up — with a fixed, canonical
:class:`~asymmetry.core.fitting.beta_calibration.BetaEstimate`
(β = 0.8732 ± 0.0041, fitted α = 1.1520 ± 0.0081, method ``"count_fit"``).
The capture therefore needs no fitting backend at all, is instant, and is
byte-identical on every run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register

#: The fixed β estimate the scenario substitutes for a real count-domain fit.
#: These are the same canonical numbers quoted in the widget's own docstring
#: (``gui/windows/grouping/beta_section.py``) and in
#: ``tests/gui/test_beta_section.py``'s default fixture.
_BETA = 0.8732
_BETA_ERR = 0.0041
_ALPHA = 1.1520
_ALPHA_ERR = 0.0081


class BetaCalibrationDialogScenario(Scenario):
    name = "beta_calibration_dialog"
    description = (
        "Inline β (asymmetry-balance) calibration in the grouping window's Corrections "
        "column, with a completed count-fit estimate, the fitted-α consistency readout, "
        "and the β = 1 ↔ β̂ asymmetry preview."
    )
    size = (1220, 760)

    def capture(self, ctx: CaptureContext) -> Path:
        import asymmetry.gui.windows.grouping.beta_section as beta_section_module
        from asymmetry.core.fitting.beta_calibration import BetaEstimate
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = make_ybco_knight_grouped()

        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)

        # Pre-set the card's currently-applied α to the value the fixed estimate
        # will also report, so the fitted-α consistency readout renders clean
        # (no spurious ">3σ disagreement" warn-tint) — a genuine agreement is
        # the more representative capture of a well-calibrated instrument.
        dialog._alpha_spin.setValue(_ALPHA)

        fixed_estimate = BetaEstimate(
            beta=_BETA,
            beta_error=_BETA_ERR,
            alpha=_ALPHA,
            alpha_error=_ALPHA_ERR,
            alpha_beta_correlation=-0.31,
            method="count_fit",
            n_bins_used=1800,
            reduced_chi2=1.08,
            ok=True,
            message="β estimated by simultaneous forward/backward count fit.",
        )
        original_estimate = beta_section_module.estimate_beta_detailed
        beta_section_module.estimate_beta_detailed = lambda *a, **k: fixed_estimate
        try:
            section = dialog._beta_section
            section._on_estimate()
            # The estimate runs on a background worker thread even though it is
            # patched to return instantly, so a single click does not populate
            # the result before the grab — pump until the worker's queued
            # finished callback has landed (deterministic, not a fixed sleep).
            _pump_until(lambda: section._tasks.active_count == 0)
            # Apply so the pipeline chip, card header, β spin, and preview all
            # agree with the result row rather than showing the pre-apply β = 1
            # alongside a completed β̂ = 0.8732 estimate.
            section._on_apply()
            # Let the shared preview redraw the β = 1 ↔ β̂ overlay before grabbing.
            _pump_events(500)
            corr_scroll = getattr(dialog, "_corrections_scroll", None)
            if corr_scroll is not None:
                corr_scroll.ensureWidgetVisible(section)
                # See alpha_calibration_dialog.py: ensureWidgetVisible can also
                # scroll right when the run combo's popup content is wide under
                # the app stylesheet; pin the column flush-left.
                corr_scroll.horizontalScrollBar().setValue(0)
                _pump_events(80)
        finally:
            beta_section_module.estimate_beta_detailed = original_estimate

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

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


register(BetaCalibrationDialogScenario())
