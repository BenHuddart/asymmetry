"""Scan-point exclusion on an integral (ALC) field scan.

Illustrates one of the six mechanisms catalogued in
:doc:`/reference/exclusions`: *scan-point exclusion*. A synthetic
longitudinal-field scan (31 field-stepped runs with an avoided-level-crossing
resonance dip at ~3100 G, from the archetype gallery) is integrated
run-by-run with the real :func:`~asymmetry.core.transform.integral.build_field_scan`
reduction and rendered in the integral-scan view
(:class:`~asymmetry.gui.panels.alc_panel.ALCScanView`). One run — the baseline
point that scatters most from its neighbours — is marked excluded, so it draws
greyed and hollow: it is dropped from the baseline / peak / RF fits but stays
visible and clickable to restore, exactly as the page's table row describes.

The frame is cropped to the plot section (x-axis selector, canvas, provenance
line) — the analysis tables below hold no fit here, so they are excluded from
the capture. Fully deterministic: the archetype seeds its own RNG and no fit
runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data.archetypes import make_alc_field_scan
from ._base import CaptureContext, Scenario, register


class ALCScanExclusionScenario(Scenario):
    name = "alc_scan_exclusion"
    description = (
        "Integral-scan view with one run click-excluded (greyed) from the "
        "field-scan baseline and resonance fits."
    )
    size = (900, 440)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.transform.integral import build_field_scan
        from asymmetry.gui.panels.alc_panel import ALCScanView
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        # The very first matplotlib canvas painted in a fresh process settles one
        # right-margin column non-deterministically (an offscreen first-paint
        # artifact); every canvas after that is byte-stable. Prime the process
        # with a throwaway draw so the real capture is never that first paint —
        # this scenario must be byte-identical whether or not it runs first.
        _warm_fig, _warm_canvas = create_canvas(layout="constrained")
        _warm_canvas.draw()
        _pump_events(60)

        datasets = make_alc_field_scan()
        runs = [d.run for d in datasets if d.run is not None]
        scan = build_field_scan(runs, method="integral", order_key="field")

        fields = np.asarray(scan.x, dtype=float)
        value_pct = np.asarray(scan.value, dtype=float) * 100.0
        error_pct = np.asarray(scan.error, dtype=float) * 100.0
        run_numbers = list(scan.run_numbers)

        excluded_mask = _pick_excluded(fields, value_pct)

        view = ALCScanView()
        # The Baseline/Peaks/RF analysis tables hold no fit here; hiding them
        # lets the plot section fill the whole (fixed) view, which keeps the
        # grab geometry byte-stable across runs (a sub-widget grab drifts by a
        # pixel with layout timing) and avoids shipping empty tables.
        view.analysis_widget().hide()
        view.resize(*self.size)
        view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        view.show()
        _pump_events(120)

        view.show_scan(
            fields,
            value_pct,
            error_pct,
            run_numbers,
            x_label="B (G)",
            y_label="Integral asymmetry (%)",
            excluded_mask=excluded_mask,
        )
        # The panel's constrained-layout engine solves the axes rectangle
        # iteratively, landing the right spine on a sub-pixel boundary that
        # antialiases two ways between runs. Pin the axes to a fixed rectangle
        # so the geometry — and therefore every pixel — is byte-stable.
        view._figure.set_layout_engine("none")
        view._ax.set_position((0.10, 0.17, 0.87, 0.77))
        # draw_idle() defers; force synchronous draws so the offscreen grab
        # captures the rendered scan rather than a blank canvas. The first paint
        # of a fresh canvas settles one right-margin column non-deterministically
        # (an offscreen first-paint artifact); a warm-up draw + settle lands the
        # second draw on the converged, byte-stable render.
        view._canvas.draw()
        _pump_events(80)
        view._canvas.draw()
        _pump_events(120)

        # Grab the matplotlib canvas itself rather than the surrounding view:
        # the Agg render is byte-deterministic, whereas the Qt widget's right
        # margin has a first-paint-flaky boundary column. The scan plot (greyed
        # excluded point, "excluded" legend, labelled axes) is the whole story.
        pix = view._canvas.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        view.close()
        view.deleteLater()
        _pump_events(40)
        return out_path


def _pick_excluded(fields: np.ndarray, value_pct: np.ndarray) -> np.ndarray:
    """Mark the baseline run that scatters most from its neighbours.

    Detrends the scan with a short rolling mean and picks the largest-residual
    point that lies on the flat baseline (away from the resonance dip at
    ~3100 G and off the array edges, where the rolling mean is unreliable), so
    the excluded run is a genuine baseline outlier rather than a hand-picked
    point. Deterministic: the scan is built from a fixed-seed archetype.
    """
    n = fields.size
    kernel = np.ones(5) / 5.0
    trend = np.convolve(value_pct, kernel, mode="same")
    residual = np.abs(value_pct - trend)
    baseline = (
        (np.abs(fields - 3100.0) > 400.0)  # not the resonance dip
        & (np.arange(n) >= 2)  # not the left edge
        & (np.arange(n) < n - 2)  # not the right edge
    )
    candidate = np.where(baseline, residual, -np.inf)
    mask = np.zeros(n, dtype=bool)
    mask[int(np.argmax(candidate))] = True
    return mask


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(ALCScanExclusionScenario())
