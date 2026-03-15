"""Dialog for plotting full time-series run logs."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout


class LogPlotDialog(QDialog):
    """Plot an arbitrary log time series in a small modal dialog."""

    def __init__(
        self,
        *,
        title: str,
        time_values: list[float],
        data_values: list[float],
        units: str = "",
        parent=None,
    ) -> None:
        """Initialise the dialog and render the supplied time-series.

        Parameters
        ----------
        title
            Window title and plot title.
        time_values
            X-axis values, typically seconds from log start.
        data_values
            Y-axis values from the log.
        units
            Optional Y-axis units string.
        parent
            Parent Qt widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 420)

        layout = QVBoxLayout(self)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception:
            return

        figure = Figure(figsize=(6.0, 3.6))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        ax = figure.add_subplot(111)
        t = np.asarray(time_values, dtype=np.float64)
        y = np.asarray(data_values, dtype=np.float64)
        ax.plot(t, y, lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ylabel = "Value" if not units else f"Value ({units})"
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        figure.tight_layout()
        canvas.draw_idle()
