"""Three-up FFT spectrum comparison: no apodisation, Gaussian, Lorentzian.

Companion to :doc:`/reference/fourier_analysis` — illustrates how
apodisation trades line-sharpness for sidelobe suppression on the same
TF signal. Unlike a bare-matplotlib reproduction, every panel is a real
GUI grab: the setting is applied through the Fourier panel's actual
**Apodisation** controls (the ``Filter τ (µs)`` field and the
``Lorentzian``/``Gaussian``/``None`` radios), the FFT is recomputed
through :meth:`MainWindow._on_compute_fourier` — the same worker-thread
code path a user's "Compute FFT" click drives — and the spectrum canvas
is grabbed after each recompute. Uses the existing YBCO vortex-lattice
synthetic dataset so the asymmetric P(B) lineshape is recognisable in
each panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

from ..data import make_ybco_vortex_lattice
from ..data.archetypes import GAMMA_MU_MHZ_PER_G
from ._base import CaptureContext, Scenario, _optimize_png, _process_events_for, register

#: (radio attribute, filter mode key, τ typed into "Filter τ (µs)", caption).
#: ``None`` for the τ leaves the field at whatever it already holds — the
#: value is inert while the "None" radio is selected.
_SETTINGS: tuple[tuple[str, str, float | None, str], ...] = (
    ("_filter_none_radio", "none", None, "None (no apodisation)"),
    ("_filter_gaussian_radio", "gaussian", 4.0, "Gaussian, Filter τ = 4.0 µs"),
    ("_filter_lorentzian_radio", "lorentzian", 3.0, "Lorentzian, Filter τ = 3.0 µs"),
)


class ApodisationComparisonScenario(Scenario):
    name = "apodisation_comparison"
    description = (
        "Three-up FFT spectrum showing none / Gaussian / Lorentzian apodisation, "
        "driven through the real Fourier panel controls."
    )
    size = (1500, 920)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resize(*self.size)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        window._on_fourier()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_ybco_vortex_lattice()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=120)

        # Switch the central plot workspace to the Frequency domain so the
        # FFT spectrum canvas, not the time-domain signal, is what gets
        # grabbed for each panel.
        window._on_domain_button_clicked("frequency")
        _process_events_for(milliseconds=80)

        freq_panel = window._frequency_plot_panel
        fourier_panel = window._fourier_panel
        center_mhz = GAMMA_MU_MHZ_PER_G * 2000.0
        x_min, x_max = center_mhz - 1.0, center_mhz + 1.5

        panels: list[QPixmap] = []
        try:
            for radio_attr, mode, tau, caption in _SETTINGS:
                if tau is not None:
                    fourier_panel._filter_time_constant_edit.setText(f"{tau:.1f}")
                getattr(fourier_panel, radio_attr).setChecked(True)

                _compute_fourier_and_wait(window)

                spectrum_x = np.asarray(freq_panel._last_plot_time, dtype=float)
                spectrum_y = np.asarray(freq_panel._last_plot_asymmetry, dtype=float)
                in_window = (spectrum_x >= x_min) & (spectrum_x <= x_max)
                peak = float(np.max(spectrum_y[in_window])) if np.any(in_window) else 1.0
                # Frame the Larmor line the same way for every panel so the
                # three spectra are visually comparable, but re-derive the
                # y-headroom per panel: apodisation changes the peak height.
                freq_panel.set_view_limits(x_min, x_max, -0.04 * peak, 1.10 * peak)
                _process_events_for(milliseconds=150)

                panels.append(_grab_canvas(freq_panel, caption=caption))
        finally:
            if hasattr(window, "_dirty"):
                window._dirty = False
            window.close()
            window.deleteLater()
            _process_events_for(milliseconds=50)

        canvas_pix = _compose(panels)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not canvas_pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)
        return out_path


def _compute_fourier_and_wait(window, *, timeout_ms: int = 15000) -> None:
    """Trigger a Fourier recompute and pump events until it finishes.

    ``_on_compute_fourier`` dispatches to the shared ``TaskRunner`` and
    returns immediately; the panel and ``_fourier_compute_active`` flag are
    only updated once the queued ``on_finished`` callback runs on the GUI
    thread (see ``MainWindow._on_fourier_payload_finished``). That callback
    flips the flag to ``False`` as its first statement and then applies the
    spectrum to the plot panel within the same synchronous call, so once the
    flag is observed ``False`` here the panel is already up to date.
    """
    window._on_compute_fourier()
    elapsed = 0
    step = 100
    while window._fourier_compute_active and elapsed < timeout_ms:
        _process_events_for(milliseconds=step)
        elapsed += step
    if window._fourier_compute_active:
        raise RuntimeError("Fourier recompute did not finish within the timeout")


def _grab_canvas(freq_panel, *, caption: str) -> QPixmap:
    """Grab just the matplotlib spectrum canvas (not the surrounding axis
    controls / header / footer chrome) and stamp a caption above it."""
    pix = freq_panel._canvas.grab()

    captioned = QPixmap(pix.width(), pix.height() + 40)
    captioned.fill(QColor("#f8fafc"))
    painter = QPainter(captioned)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = QFont(painter.font())
    font.setBold(True)
    font.setPointSizeF(14)
    painter.setFont(font)
    painter.setPen(QColor("#0f172a"))
    painter.drawText(
        0, 0, pix.width(), 40,
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        caption,
    )
    painter.drawPixmap(0, 40, pix)
    painter.end()
    return captioned


def _compose(panels: list[QPixmap]) -> QPixmap:
    padding = 20
    gap = 24
    total_w = sum(p.width() for p in panels) + gap * (len(panels) - 1) + padding * 2
    total_h = max(p.height() for p in panels) + padding * 2
    canvas = QPixmap(total_w, total_h)
    canvas.fill(QColor("#f8fafc"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    x = padding
    for p in panels:
        painter.drawPixmap(x, padding, p)
        x += p.width() + gap
    painter.end()
    return canvas


register(ApodisationComparisonScenario())
