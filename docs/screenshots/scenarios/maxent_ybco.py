"""Maximum-entropy spectrum of a YBCO vortex-lattice TF μSR signal.

Companion to :mod:`docs.screenshots.scenarios.fourier_tf`: the *same*
vortex-state dataset reconstructed with the maximum-entropy method instead of
the FFT, for the side-by-side comparison in ``reference/fourier_analysis.rst``.

The scenario loads the YBCO vortex-lattice run, switches to the **MaxEnt**
domain, runs the reconstruction to convergence, and waits for the background
worker to finish (pumping the event loop until the result is stored) so the
screenshot shows the converged spectrum rather than the transient state. It
then frames the reconstructed line (excluding the spectral-leakage spikes at
the transform boundary) and raises the **MaxEnt** inspector tab — the
reconstruction's own cycle/convergence controls, which are what the
embedding pages discuss — rather than the Fit tab, whose carried-over model
state is irrelevant to these pages.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ybco_vortex_lattice
from ._base import Scenario, _process_events_for, register
from .fourier_tf import _raise_inspector_tab


class MaxEntYbcoScenario(Scenario):
    name = "maxent_ybco"
    description = "MaxEnt reconstruction of a YBCO vortex-lattice TF μSR signal."
    size = (1500, 920)
    requires_fit = True  # MaxEnt uses the numba-backed solver (numpy < 2.3)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_ybco_vortex_lattice()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=120)

        # Enter the MaxEnt domain (enabled once the dataset supports it).
        window._on_domain_button_clicked("maxent")
        _process_events_for(milliseconds=120)

        # Run MaxEnt to convergence, then wait for the background worker to
        # store its result before the grab.
        run_number = int(dataset.run_number)
        window._on_compute_maxent(50)
        self._wait_for_maxent(window, run_number, timeout_s=120)
        _process_events_for(milliseconds=250)

        # The reconstructed spectrum's own transform-boundary bins carry sharp
        # spectral-leakage spikes (≈4x the physical line's height) right at
        # its extreme edges (~23.04 and ~31.17 MHz), so an Auto X / Auto Y
        # pair that frames the *full* data extent leaves the real
        # vortex-lattice line squashed into a quarter of the plot height.
        # Restrict the X range to sit inside those artifacts through the
        # panel's real X-range pathway (the same toolbar fields a user would
        # type into, via set_view_limits) with Auto X switched off so it
        # cannot snap back to the full extent, then re-autoscale Y from that
        # narrower window so the line fills the frame.
        freq_panel = window._frequency_plot_panel
        freq_panel._auto_x_btn.setChecked(False)
        freq_panel.set_view_limits(23.5, 30.5, 0.0, 1.0)
        freq_panel._auto_y_btn.setChecked(True)
        freq_panel._auto_y_limits()
        _process_events_for(milliseconds=120)
        return window

    def settle(self, widget: QWidget) -> None:
        # Bring the MaxEnt tab of the inspector deck to the front so the
        # screenshot shows the reconstruction's own controls (cycle buttons,
        # convergence diagnostics) — the settings the embedding pages
        # actually discuss (the caption instructs "click Compute MaxEnt") —
        # not the Fit tab, whose carried-over model state ("Model carried —
        # not fitted for this run") is irrelevant to these pages. Done here,
        # not in build(): the deck's tab bar only exists once the window is
        # shown and laid out; see _raise_inspector_tab for why raise_() is
        # not enough.
        _process_events_for(milliseconds=100)
        _raise_inspector_tab(widget, widget._dock_fourier.windowTitle())
        super().settle(widget)

    def _wait_for_maxent(self, window, run_number: int, timeout_s: int = 120) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            _process_events_for(milliseconds=100)
            if run_number in window._maxent_result_by_run:
                break


register(MaxEntYbcoScenario())
