"""Bespoke ALC-mode panels: build an integral-asymmetry field scan and view it.

ALC mode is toggled from the main toolbar when the Forward-Backward asymmetry
representation is active. Enabling it swaps the Fit and Parameters docks for
these focused widgets:

* :class:`ALCFitPanel` — the build controls (integration window + Build Scan).
  ALC mode *integrates* the asymmetry over the fit-range window for each
  selected run (one value per run) rather than fitting a model.
* :class:`ALCScanView` — the resulting field scan (integral asymmetry vs the
  run variable) as a plot plus a table.

The scan is recorded as a model-less ``FitSeries`` internally (see
``MainWindow._on_scan_requested``); these widgets are pure presentation and hold
no analysis logic of their own.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ALCFitPanel(QWidget):
    """Build controls for an integral-asymmetry field scan (ALC mode).

    The integration window reuses the time-spectrum fit-range: the user drags
    the shaded range on the plot and the current bounds are echoed here. Clicking
    *Build Scan* emits :attr:`build_requested`; the main window does the work.
    """

    build_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Integral scan (ALC)")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        intro = QLabel(
            "Integrate the asymmetry over the window for each selected run to "
            "build a field scan (ALC / repolarisation / QLCR)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        window_box = QGroupBox("Integration window")
        window_layout = QVBoxLayout(window_box)
        self._window_label = QLabel(self._format_window(None, None))
        window_layout.addWidget(self._window_label)
        hint = QLabel("Drag the shaded range on the time plot to change it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        window_layout.addWidget(hint)
        layout.addWidget(window_box)

        self._build_btn = QPushButton("Build Scan")
        self._build_btn.clicked.connect(self.build_requested.emit)
        layout.addWidget(self._build_btn)

        layout.addStretch()

    @staticmethod
    def _format_window(t_min: float | None, t_max: float | None) -> str:
        if t_min is None or t_max is None:
            return "Window: full good-bin range"
        return f"Window: {t_min:.3f} – {t_max:.3f} μs"

    def set_integration_window(self, t_min: float | None, t_max: float | None) -> None:
        """Echo the current fit-range/integration window."""
        self._window_label.setText(self._format_window(t_min, t_max))


class ALCScanView(QWidget):
    """Plot + table view of one integral-asymmetry field scan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout.addWidget(self._canvas, 1)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Run", "x", "A (%)", "± err"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

        self.clear()

    def clear(self) -> None:
        """Show the empty-state placeholder."""
        self._ax.clear()
        self._ax.text(
            0.5,
            0.5,
            "Build a scan to see the ALC curve",
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            color="gray",
        )
        self._ax.set_axis_off()
        self._canvas.draw_idle()
        self._table.setRowCount(0)

    def show_scan(
        self,
        x: NDArray[np.float64],
        value: NDArray[np.float64],
        error: NDArray[np.float64],
        run_numbers: list[int],
        *,
        x_label: str,
        y_label: str,
    ) -> None:
        """Render a scan from parallel arrays (already in display units)."""
        x = np.asarray(x, dtype=float)
        value = np.asarray(value, dtype=float)
        error = np.asarray(error, dtype=float)

        self._ax.clear()
        self._ax.set_axis_on()
        if x.size:
            self._ax.errorbar(
                x, value, yerr=error, fmt="o-", markersize=4, capsize=2, linewidth=1.0
            )
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

        self._table.setRowCount(int(x.size))
        for row in range(int(x.size)):
            run = run_numbers[row] if row < len(run_numbers) else ""
            cells = [
                str(run),
                f"{x[row]:.4g}",
                f"{value[row]:.4f}",
                f"{error[row]:.4f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)
