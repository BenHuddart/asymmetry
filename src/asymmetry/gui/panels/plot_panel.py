"""Central plot panel using Matplotlib embedded in Qt.

Displays time-domain asymmetry with error bars and optional fit overlay,
similar to WiMDA's main plot area.

The bunch-factor control rebins the displayed data and also defines the
dataset passed to fitting in the GUI. The original MuonDataset is preserved,
so changing the bunch factor after fitting only changes the plotted data while
the existing fit curve remains overlaid.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.rebin import rebin


class PlotPanel(QWidget):
    """Matplotlib canvas for time- and frequency-domain plots.

    Notes
    -----
    The bunch factor controls both the plotted representation and the dataset
    prepared for fitting in the GUI. The stored source dataset remains
    unchanged, and any rebinned fit dataset is produced as a temporary copy.
    """

    bunch_factor_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._figure = Figure(tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._ax = self._figure.add_subplot(111)
            self._ax.set_xlabel("Time (μs)")
            self._ax.set_ylabel("Asymmetry (%)")

            # Add plot limit controls toolbar
            self._create_limit_controls()
            layout.addLayout(self._limit_toolbar)

            layout.addWidget(self._canvas)
            self._has_mpl = True

            # Store current dataset for rebunching
            self._current_dataset = None

            # Store fit curve data to persist across redraws
            self._fit_curve = None  # (t_fit, y_fit, label) for single fits
            self._fit_curves = {}   # {run_number: (t_fit, y_fit, label)} for global fits
        except ImportError:
            from PySide6.QtWidgets import QLabel

            layout.addWidget(QLabel("matplotlib not installed — plotting disabled"))
            self._has_mpl = False

    def _create_limit_controls(self) -> None:
        """Create toolbar for adjusting plot limits.

        Uses a grid layout for compactness:
        Row 1: X min/max  Y min/max  Apply Auto
        Row 2: Bunch factor
        """
        self._limit_toolbar = QGridLayout()
        self._limit_toolbar.setSpacing(4)  # Tight spacing
        self._limit_toolbar.setContentsMargins(4, 4, 4, 4)  # Minimal margins

        # X-axis limits
        self._limit_toolbar.addWidget(QLabel("X:"), 0, 0)
        self._x_min = QDoubleSpinBox()
        self._x_min.setRange(-1e6, 1e6)
        self._x_min.setDecimals(3)
        self._x_min.setValue(0.0)
        self._x_min.setSuffix(" μs")
        self._x_min.setMaximumWidth(80)  # Constrain width
        self._limit_toolbar.addWidget(self._x_min, 0, 1)

        self._limit_toolbar.addWidget(QLabel("–"), 0, 2)  # Dash separator

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(-1e6, 1e6)
        self._x_max.setDecimals(3)
        self._x_max.setValue(10.0)
        self._x_max.setSuffix(" μs")
        self._x_max.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._x_max, 0, 3)

        # Y-axis limits
        self._limit_toolbar.addWidget(QLabel("Y:"), 0, 4)
        self._y_min = QDoubleSpinBox()
        self._y_min.setRange(-1e6, 1e6)
        self._y_min.setDecimals(3)
        self._y_min.setValue(-30.0)
        self._y_min.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._y_min, 0, 5)

        self._limit_toolbar.addWidget(QLabel("–"), 0, 6)  # Dash separator

        self._y_max = QDoubleSpinBox()
        self._y_max.setRange(-1e6, 1e6)
        self._y_max.setDecimals(3)
        self._y_max.setValue(30.0)
        self._y_max.setMaximumWidth(80)
        self._limit_toolbar.addWidget(self._y_max, 0, 7)

        # Apply and Auto buttons
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_limits)
        apply_btn.setMaximumWidth(60)
        self._limit_toolbar.addWidget(apply_btn, 0, 8)

        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self._auto_limits)
        auto_btn.setMaximumWidth(50)
        self._limit_toolbar.addWidget(auto_btn, 0, 9)

        # Stretch to fill remaining space
        self._limit_toolbar.setColumnStretch(10, 1)
        self._limit_toolbar.addWidget(QWidget(), 0, 10)

        # Bunch factor on second row
        self._limit_toolbar.addWidget(QLabel("Bunch:"), 1, 0)
        self._bunch_factor = QSpinBox()
        self._bunch_factor.setRange(1, 1000)
        self._bunch_factor.setValue(1)
        self._bunch_factor.setMaximumWidth(60)
        self._bunch_factor.valueChanged.connect(self._on_bunch_changed)
        self._limit_toolbar.addWidget(self._bunch_factor, 1, 1)

    def get_analysis_dataset(self, dataset: MuonDataset | None) -> MuonDataset | None:
        """Return the dataset that should be used for plotting and fitting.

        When the bunch factor is 1, the original dataset is returned. For a
        larger bunch factor, a rebinned MuonDataset copy is returned with the
        same metadata and run association.
        """
        if dataset is None:
            return None

        bunch_factor = self._bunch_factor.value()
        if bunch_factor <= 1:
            return dataset

        time, asymmetry, error = rebin(
            dataset.time,
            dataset.asymmetry,
            dataset.error,
            bunch_factor,
        )
        return MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=error,
            metadata=dict(dataset.metadata),
            run=dataset.run,
        )

    def plot_dataset(self, dataset: MuonDataset) -> None:
        """Plot a dataset, optionally rebinned according to the bunch factor.

        The input dataset is stored unchanged as the current dataset. If the
        bunch factor is greater than 1, temporary rebinned arrays are created
        for plotting. The source dataset itself is never mutated.
        """
        if not self._has_mpl:
            return

        # Store the original dataset
        self._current_dataset = dataset

        analysis_dataset = self.get_analysis_dataset(dataset)
        if analysis_dataset is None:
            return
        time = analysis_dataset.time
        asymmetry = analysis_dataset.asymmetry
        error = analysis_dataset.error

        self._ax.clear()
        self._ax.errorbar(
            time,
            asymmetry,
            yerr=error,
            fmt=".",
            markersize=3,
            label=f"Run {dataset.run_number}",
        )
        self._ax.set_xlabel("Time (μs)")
        self._ax.set_ylabel("Asymmetry (%)")

        # Re-plot fit curve if it exists (check both single and global fits)
        fit_to_plot = None
        if self._fit_curve is not None:
            fit_to_plot = self._fit_curve
        elif dataset.run_number in self._fit_curves:
            fit_to_plot = self._fit_curves[dataset.run_number]

        if fit_to_plot is not None:
            t_fit, y_fit, fit_label = fit_to_plot
            self._ax.plot(t_fit, y_fit, 'r-', linewidth=2, label=fit_label)

        self._ax.legend()

        # Set default limits based on data range (including error bars)
        x_min, x_max = time.min(), time.max()
        y_min = (asymmetry - error).min()
        y_max = (asymmetry + error).max()

        # Add 5% padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05

        self._x_min.setValue(x_min - x_padding)
        self._x_max.setValue(x_max + x_padding)
        self._y_min.setValue(y_min - y_padding)
        self._y_max.setValue(y_max + y_padding)

        # Apply the limits
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Apply the specified axis limits to the plot."""
        if not self._has_mpl:
            return

        self._ax.set_xlim(self._x_min.value(), self._x_max.value())
        self._ax.set_ylim(self._y_min.value(), self._y_max.value())
        self._canvas.draw()

    def _auto_limits(self) -> None:
        """Auto-scale the plot to fit all data."""
        if not self._has_mpl:
            return

        self._ax.relim()
        self._ax.autoscale()

        # Update spinboxes to reflect the new limits
        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()

        self._x_min.setValue(x_lim[0])
        self._x_max.setValue(x_lim[1])
        self._y_min.setValue(y_lim[0])
        self._y_max.setValue(y_lim[1])

        self._canvas.draw()

    def _on_bunch_changed(self) -> None:
        """Re-plot and refresh fit inputs when the bunch factor changes."""
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)
        self.bunch_factor_changed.emit(self._bunch_factor.value())

    def plot_fit(self, t_fit, y_fit, label: str = "Fit") -> None:
        """Overlay a fit curve on the current plot.

        The fit curve will be retained even when bunching or limits change.

        Parameters
        ----------
        t_fit : array
            Time points for the fit curve.
        y_fit : array
            Fitted asymmetry values.
        label : str, optional
            Label for the fit curve in the legend.
        """
        if not self._has_mpl:
            return

        # Store fit curve data for persistence across redraws (single fit)
        self._fit_curve = (t_fit, y_fit, label)
        # Clear global fits when doing a single fit
        self._fit_curves = {}

        # Plot the fit curve
        self._ax.plot(t_fit, y_fit, 'r-', linewidth=2, label=label)
        self._ax.legend()
        self._canvas.draw()

    def set_global_fits(self, fit_curves_dict: dict) -> None:
        """Set fit curves from global fitting.

        Parameters
        ----------
        fit_curves_dict : dict
            Dictionary mapping run_number -> (t_fit, y_fit, label).
        """
        if not self._has_mpl:
            return

        # Store all fit curves
        self._fit_curves = fit_curves_dict
        # Clear single fit curve
        self._fit_curve = None

        # Redraw current dataset with its fit
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    def clear(self) -> None:
        """Clear the plot and reset stored data."""
        if self._has_mpl:
            self._ax.clear()
            self._canvas.draw()
            self._current_dataset = None
            self._fit_curve = None
            self._fit_curves = {}

    def clear_fit(self) -> None:
        """Clear all fit curves and redraw the plot."""
        if not self._has_mpl:
            return

        self._fit_curve = None
        self._fit_curves = {}
        if self._current_dataset is not None:
            self.plot_dataset(self._current_dataset)

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the plot panel state.

        This captures the bunch factor, axis limits, the currently displayed
        run number, and any stored fit curves.  Fit curve arrays are
        serialised as plain Python lists for JSON compatibility.

        Returns
        -------
        dict
            Plot state suitable for inclusion in a project file.
        """
        state: dict = {
            "current_run_number": (
                self._current_dataset.run_number
                if self._current_dataset is not None
                else None
            ),
            "bunch_factor": self._bunch_factor.value() if self._has_mpl else 1,
            "x_min": self._x_min.value() if self._has_mpl else 0.0,
            "x_max": self._x_max.value() if self._has_mpl else 10.0,
            "y_min": self._y_min.value() if self._has_mpl else -30.0,
            "y_max": self._y_max.value() if self._has_mpl else 30.0,
            "fit_curve": None,
            "fit_curves": {},
        }

        if self._has_mpl:
            if self._fit_curve is not None:
                t_fit, y_fit, label = self._fit_curve
                state["fit_curve"] = {
                    "t": list(t_fit),
                    "y": list(y_fit),
                    "label": label,
                }
            for run_number, (t_fit, y_fit, label) in self._fit_curves.items():
                state["fit_curves"][str(run_number)] = {
                    "t": list(t_fit),
                    "y": list(y_fit),
                    "label": label,
                }

        return state

    def restore_state(
        self,
        state: dict,
        dataset: "MuonDataset | None" = None,
    ) -> None:
        """Restore plot panel state from a saved dict.

        Parameters
        ----------
        state : dict
            Plot state as returned by :meth:`get_state`.
        dataset : MuonDataset, optional
            Dataset to re-plot after restoring limits.  If *None* no plot is
            drawn, but all other state (limits, bunch factor, fit curves) is
            still applied.
        """
        if not self._has_mpl:
            return

        import numpy as np

        # Restore bunch factor without triggering bunch_factor_changed signal.
        self._bunch_factor.blockSignals(True)
        self._bunch_factor.setValue(state.get("bunch_factor", 1))
        self._bunch_factor.blockSignals(False)

        # Restore axis limit spinboxes (will be applied after optional re-plot).
        for spin, key, default in (
            (self._x_min, "x_min", 0.0),
            (self._x_max, "x_max", 10.0),
            (self._y_min, "y_min", -30.0),
            (self._y_max, "y_max", 30.0),
        ):
            spin.blockSignals(True)
            spin.setValue(state.get(key, default))
            spin.blockSignals(False)

        # Restore fit curves.
        self._fit_curve = None
        self._fit_curves = {}

        fit_curve_data = state.get("fit_curve")
        if fit_curve_data:
            self._fit_curve = (
                np.array(fit_curve_data["t"]),
                np.array(fit_curve_data["y"]),
                fit_curve_data.get("label", "Fit"),
            )

        for run_str, curve_data in state.get("fit_curves", {}).items():
            self._fit_curves[int(run_str)] = (
                np.array(curve_data["t"]),
                np.array(curve_data["y"]),
                curve_data.get("label", "Global Fit"),
            )

        # Re-plot the current dataset if one was provided.
        if dataset is not None:
            self._current_dataset = dataset
            self.plot_dataset(dataset)

        # Always apply the restored limits.
        self._apply_limits()
