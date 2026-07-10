"""Spectral moments of a YBCO vortex-lattice field distribution.

The spectral-moments control reduces a whole field/frequency line shape p(B) to
a handful of physics numbers — the mean field, the RMS width (which sets the
penetration depth in a type-II mixed state), and the skewness/asymmetry that
fingerprint the vortex lattice. This scenario captures that readout in its
Fourier-panel home, computed on the canonical asymmetric spectrum: the YBCO
vortex-lattice P(B), with its sharp low-field van-Hove peak and long high-field
tail toward the vortex cores (Brandt PRB 37, 2349, 1988; Sonier RMP 72, 769,
2000).

The scenario:

1. Loads the YBCO vortex-lattice run (full Run with F/B detector histograms +
   grouping) and switches the workspace to the **Frequency** domain.
2. Selects the **Phase** display (the phase-corrected real FFT) and fills the
   per-group phases from the data, so the spectrum is the lineshape-faithful
   absorption line the moments require — power/magnitude/phaseOptReal modes are
   squared or dispersive and are greyed out for moments.
3. Computes the FFT, drags a **range** over the line and sets a **cutoff**, and
   opens the **Spectral moments** section so its live readout is on screen: the
   peak field B_pk ≈ 1987 G (≈ 26.9 MHz, γ_μ = 0.01355 MHz/G), the mean B_ave
   sitting above the peak, the RMS width, and the positive skewness/asymmetry
   (β > 0, γ₁ > 0) that are the vortex distribution's expected signature.

The central plot shows the same shaded range and dotted cutoff drawn over the
P(B) line. The moment values are computed by the real core path
(``asymmetry.core.fourier.moments.spectrum_moments``) on the real FFT, so the
readout is an honest reduction of the displayed spectrum.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QTabBar, QWidget

from ..data import make_ybco_vortex_lattice
from ..data.archetypes import GAMMA_MU_MHZ_PER_G
from ._base import Scenario, _process_events_for, register


def _raise_inspector_tab(window, tab_label: str) -> None:
    """Select *tab_label* in the right inspector deck's tab bar.

    ``QDockWidget.raise_()`` is a silent no-op for tabified docks under the
    offscreen QPA platform the capture runs on, so the deck's ``QTabBar`` is
    driven directly. The deck's bar is identified by carrying both *tab_label*
    and the always-present "Fit" tab.
    """
    for tab_bar in window.findChildren(QTabBar):
        labels = [tab_bar.tabText(i) for i in range(tab_bar.count())]
        if tab_label in labels and "Fit" in labels:
            tab_bar.setCurrentIndex(labels.index(tab_label))
            _process_events_for(milliseconds=80)
            return
    raise RuntimeError(f"Inspector deck tab bar with {tab_label!r} not found")


class SpectralMomentsReadoutScenario(Scenario):
    name = "spectral_moments_readout"
    description = (
        "Spectral-moments readout in the Fourier panel, computed on the YBCO "
        "vortex-lattice phase-corrected FFT."
    )
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.widgets.panel_section import PanelSection

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_ybco_vortex_lattice()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=120)

        # Moments need a lineshape-faithful spectrum: the phase-corrected real
        # FFT. Select the "Phase" display and fill the per-group phases from the
        # data so the real projection is the clean absorption line (the
        # entropy-optimised phaseOptReal is dispersive for this signal, and the
        # default (Power)^1/2 is squared and moment-ineligible).
        fourier = window._fourier_panel
        fourier._set_display_mode("Phase")
        fourier._auto_phase_btn.click()
        _process_events_for(milliseconds=80)

        window._on_domain_button_clicked("frequency")
        _process_events_for(milliseconds=80)

        # Frame onto the Larmor line (γ_μ·B_app ≈ 27.1 MHz for 2000 G).
        center_mhz = GAMMA_MU_MHZ_PER_G * 2000.0
        x_min, x_max = center_mhz - 1.8, center_mhz + 3.6

        window._on_compute_fourier()

        # The recompute lands via a queued signal from a worker thread, so poll
        # until the rendered spectrum reaches the Larmor window before framing
        # it (mirrors the fourier scenarios).
        spectrum_x = spectrum_y = None
        for _ in range(100):  # bounded ~10 s; typically well under 1 s
            _process_events_for(milliseconds=100)
            x = window._frequency_plot_panel._last_plot_time
            y = window._frequency_plot_panel._last_plot_asymmetry
            if x is not None and y is not None and len(x) and float(np.nanmax(x)) >= x_max:
                spectrum_x = np.asarray(x, dtype=float)
                spectrum_y = np.asarray(y, dtype=float)
                break
        if spectrum_x is None:
            raise RuntimeError("Fourier recompute did not render within 10 s")

        # Open the (collapsed) Spectral moments section and drive its range and
        # cutoff over the vortex line, then compute the moments through the real
        # core path. The range brackets the peak and its high-field tail; the
        # 5 % cutoff trims the spectral floor before the integral.
        for section in fourier.findChildren(PanelSection):
            if section.title() == "Spectral moments":
                section.setExpanded(True)
                break
        moments = fourier.moments_widget
        moments.set_range_mhz(26.0, 30.0)
        moments.set_cutoff_fraction(0.05)
        window._refresh_spectral_moments()
        _process_events_for(milliseconds=80)

        # Frame the plot on the line so the shaded range and dotted cutoff are
        # recognisable rather than lost in a wide near-empty axis.
        in_window = (spectrum_x >= x_min) & (spectrum_x <= x_max)
        peak = float(np.max(spectrum_y[in_window])) if np.any(in_window) else 1.0
        window._frequency_plot_panel.set_view_limits(x_min, x_max, -0.04 * peak, 1.10 * peak)
        window._refresh_spectral_moments()
        _process_events_for(milliseconds=120)
        return window

    def settle(self, widget: QWidget) -> None:
        # Bring the Spectrum inspector tab to the front and scroll its control
        # deck down to the Spectral moments readout — the subject of the shot —
        # rather than the display-mode radios at the top of the panel.
        _process_events_for(milliseconds=100)
        _raise_inspector_tab(widget, widget._dock_fourier.windowTitle())
        # The Fourier panel holds its own inner QScrollArea (its control deck is
        # taller than the dock), so walk up from the moments widget to that
        # scroll area and bring the readout into view.
        moments = widget._fourier_panel.moments_widget
        ancestor = moments.parent()
        while ancestor is not None and not isinstance(ancestor, QScrollArea):
            ancestor = ancestor.parent()
        if isinstance(ancestor, QScrollArea):
            ancestor.ensureWidgetVisible(moments, 0, 40)
        _process_events_for(milliseconds=80)
        super().settle(widget)


register(SpectralMomentsReadoutScenario())
