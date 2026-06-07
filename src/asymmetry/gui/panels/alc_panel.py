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
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles.fonts import mono_font


class ALCFitPanel(QWidget):
    """Build controls for an integral-asymmetry field scan (ALC mode).

    The integration window IS the time-spectrum fit-range, mirroring the regular
    fitting machinery: the user can drag the shaded range on the plot, or set it
    precisely with the spinboxes here (the two stay in sync). Editing a spinbox
    emits :attr:`fit_range_edit_committed`; clicking *Build Scan* emits
    :attr:`build_requested`. The main window does the work.
    """

    build_requested = Signal()
    #: (x_min, x_max) committed from the spinboxes — pushed to the plot fit-range.
    fit_range_edit_committed = Signal(float, float)

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
        range_row = QHBoxLayout()
        range_row.setContentsMargins(6, 4, 6, 4)
        range_row.setSpacing(4)
        self._min_spin = self._make_time_spin()
        mid_label = QLabel("≤ <i>t</i> ≤")
        mid_label.setTextFormat(Qt.TextFormat.RichText)
        self._max_spin = self._make_time_spin()
        range_row.addWidget(self._min_spin)
        range_row.addWidget(mid_label)
        range_row.addWidget(self._max_spin)
        range_row.addWidget(QLabel("μs"))
        range_row.addStretch()
        window_layout.addLayout(range_row)
        hint = QLabel("Drag the shaded range on the time plot, or set it here.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        window_layout.addWidget(hint)
        layout.addWidget(window_box)

        self._min_spin.editingFinished.connect(self._on_spin_committed)
        self._max_spin.editingFinished.connect(self._on_spin_committed)

        self._build_btn = QPushButton("Build Scan")
        self._build_btn.clicked.connect(self.build_requested.emit)
        layout.addWidget(self._build_btn)

        layout.addStretch()

    @staticmethod
    def _make_time_spin() -> QDoubleSpinBox:
        """A μs fit-range spinbox configured like the regular fit panel's."""
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setMinimumWidth(90)
        spin.setFont(mono_font(11.0))
        spin.setEnabled(False)  # enabled once the plot has a fit-range
        return spin

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update the spinboxes from the plot fit-range without re-emitting.

        Mirrors ``SingleFitTab.set_fit_range_display``; the spinboxes are
        disabled until the plot has a fit-range.
        """
        have_range = x_min is not None and x_max is not None
        self._min_spin.setEnabled(have_range)
        self._max_spin.setEnabled(have_range)
        if not have_range:
            return
        with QSignalBlocker(self._min_spin):
            self._min_spin.setValue(float(x_min))
        with QSignalBlocker(self._max_spin):
            self._max_spin.setValue(float(x_max))

    def _on_spin_committed(self) -> None:
        self.fit_range_edit_committed.emit(self._min_spin.value(), self._max_spin.value())


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
