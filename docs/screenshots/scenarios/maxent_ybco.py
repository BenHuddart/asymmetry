"""Maximum-entropy spectrum of a YBCO vortex-lattice TF μSR signal.

Companion to :mod:`docs.screenshots.scenarios.fourier_tf`: the *same*
vortex-state dataset reconstructed with the maximum-entropy method instead of
the FFT, for the side-by-side comparison in ``reference/fourier_analysis.rst``.

The scenario loads the YBCO vortex-lattice run, switches to the **MaxEnt**
domain, runs the reconstruction to convergence, and waits for the background
worker to finish (pumping the event loop until the result is stored) so the
screenshot shows the converged spectrum rather than the transient state.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ybco_vortex_lattice
from ._base import Scenario, _process_events_for, register


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

        # The spectrum renders with the panel's existing (wide) x-limits
        # preserved, leaving the reconstructed P(B) feature off-frame, so
        # frame it with Auto X / Auto Y.
        freq_panel = window._frequency_plot_panel
        freq_panel._auto_x_btn.setChecked(True)
        freq_panel._auto_y_btn.setChecked(True)
        freq_panel._auto_x_limits()
        freq_panel._auto_y_limits()
        _process_events_for(milliseconds=200)
        return window

    def _wait_for_maxent(self, window, run_number: int, timeout_s: int = 120) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            _process_events_for(milliseconds=100)
            if run_number in window._maxent_result_by_run:
                break


register(MaxEntYbcoScenario())
