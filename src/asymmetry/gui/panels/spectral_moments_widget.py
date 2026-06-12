"""Spectral-moments readout widget (a self-contained ``QGroupBox``).

A single widget class mounted in *both* the Fourier advanced stack and the MaxEnt
panel (the shared-widget hosting decision): it shows the moment set of the active
field/frequency spectrum, carries the *range* and *cutoff* controls, and offers a
**Send to trend** action. It owns no spectrum data and does no analysis — the host
``MainWindow`` reads the active spectrum (the W15 accessor), calls
:func:`asymmetry.core.fourier.moments.spectrum_moments`, and pushes the result
back via :meth:`show_moments`. The window range is held internally in **canonical
absolute MHz** (the unit-invariant), and shown / edited in the user's chosen unit.

Per the F8 vocabulary this is a *range* control, never an "Exclude". The widget
adds no background / diamagnetic / exclusion handling — those ladders are settled
elsewhere; it consumes whatever conditioned spectrum it is handed.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fourier.moments import SpectrumMoments
from asymmetry.core.fourier.units import FieldUnit, convert

# Display unit options (label, FieldUnit). Gauss first — the penetration-depth
# reading and the field-default decision.
_UNIT_OPTIONS: tuple[tuple[str, FieldUnit], ...] = (
    ("Field (G)", FieldUnit.GAUSS),
    ("Field (T)", FieldUnit.TESLA),
    ("Frequency (MHz)", FieldUnit.MHZ),
)
_UNIT_SUFFIX = {FieldUnit.GAUSS: "G", FieldUnit.TESLA: "T", FieldUnit.MHZ: "MHz"}

# Readout rows: (label, value attr, error attr) in physics-meaningful order.
_READOUT_ROWS: tuple[tuple[str, str, str], ...] = (
    ("B_pk", "b_pk", "b_pk_err"),
    ("B_ave", "b_ave", "b_ave_err"),
    ("⟨B_ave−B_pk⟩", "b_diff", "b_diff_err"),
    ("B_rms (vs mean)", "b_rms_mean", "b_rms_mean_err"),
    ("B_rms (vs peak)", "b_rms_peak", "b_rms_peak_err"),
    ("Skewness α", "skewness", "skewness_err"),
    ("Skewness γ₁", "skewness_g1", "skewness_g1_err"),
    ("Asymmetry β", "beta", "beta_err"),
)
_DIMENSIONLESS = {"skewness", "skewness_g1", "beta"}


def _format_value_error(value: float, error: float, suffix: str) -> str:
    """Return a compact ``value ± error unit`` string (blank for NaN value)."""
    if value is None or not math.isfinite(value):
        return "—"
    unit = f" {suffix}" if suffix else ""
    if error is not None and math.isfinite(error):
        return f"{value:.4g} ± {error:.2g}{unit}"
    return f"{value:.4g}{unit}"


class SpectralMomentsWidget(QGroupBox):
    """Range / cutoff controls + moment readout + send-to-trend, for one host."""

    #: Emitted when the user edits the unit, range, or cutoff (host recomputes).
    settings_changed = Signal()
    #: Emitted when the user clicks Send to trend.
    send_to_trend_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Spectral moments", parent)
        self._unit: FieldUnit = FieldUnit.GAUSS
        self._range_mhz: tuple[float, float] | None = None
        self._bounds_mhz: tuple[float, float] | None = None
        self._cutoff_fraction: float = 0.0
        self._eligible = False

        outer = QVBoxLayout(self)

        # ── controls ──────────────────────────────────────────────────────
        controls = QFormLayout()
        self._unit_combo = QComboBox()
        for label, unit in _UNIT_OPTIONS:
            self._unit_combo.addItem(label, userData=unit.value)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        controls.addRow("Unit:", self._unit_combo)

        range_row = QHBoxLayout()
        self._range_min_spin = self._make_range_spin()
        self._range_max_spin = self._make_range_spin()
        self._range_min_spin.editingFinished.connect(self._on_range_edited)
        self._range_max_spin.editingFinished.connect(self._on_range_edited)
        range_row.addWidget(self._range_min_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._range_max_spin)
        self._range_suffix_label = QLabel(_UNIT_SUFFIX[self._unit])
        range_row.addWidget(self._range_suffix_label)
        controls.addRow("Range:", self._wrap(range_row))

        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setRange(0.0, 99.0)
        self._cutoff_spin.setDecimals(1)
        self._cutoff_spin.setSingleStep(1.0)
        self._cutoff_spin.setSuffix(" % of peak")
        self._cutoff_spin.setValue(0.0)
        self._cutoff_spin.editingFinished.connect(self._on_cutoff_edited)
        controls.addRow("Cutoff:", self._cutoff_spin)
        outer.addLayout(controls)

        # ── readout ───────────────────────────────────────────────────────
        readout = QGridLayout()
        readout.setColumnStretch(1, 1)
        self._value_labels: dict[str, QLabel] = {}
        for row, (label, attr, _err) in enumerate(_READOUT_ROWS):
            readout.addWidget(QLabel(label), row, 0)
            value = QLabel("—")
            self._value_labels[attr] = value
            readout.addWidget(value, row, 1)
        self._points_label = QLabel("Points: —")
        readout.addWidget(self._points_label, len(_READOUT_ROWS), 0, 1, 2)
        outer.addLayout(readout)

        # ── action + status ──────────────────────────────────────────────
        self._send_btn = QPushButton("Send to trend")
        self._send_btn.clicked.connect(self.send_to_trend_requested.emit)
        outer.addWidget(self._send_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        outer.addWidget(self._status_label)

        self.set_eligible(False, "No lineshape-faithful spectrum is active.")

    # ── small builders ────────────────────────────────────────────────────

    @staticmethod
    def _make_range_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1.0e12, 1.0e12)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    # ── eligibility ───────────────────────────────────────────────────────

    def set_eligible(self, eligible: bool, reason: str = "") -> None:
        """Enable/disable the controls; show *reason* when ineligible.

        Ineligible display modes (power, magnitude, phase, Burg, correlation)
        grey the whole control out with the explanatory tooltip.
        """
        self._eligible = bool(eligible)
        for w in (
            self._unit_combo,
            self._range_min_spin,
            self._range_max_spin,
            self._cutoff_spin,
            self._send_btn,
        ):
            w.setEnabled(self._eligible)
        self.setToolTip("" if self._eligible else reason)
        if not self._eligible:
            self._status_label.setText(reason)
            self.clear_readout()
        else:
            self._status_label.setText("")

    def is_eligible(self) -> bool:
        return self._eligible

    # ── unit / range / cutoff state ───────────────────────────────────────

    def unit(self) -> FieldUnit:
        return self._unit

    def cutoff_fraction(self) -> float:
        return self._cutoff_fraction

    def range_mhz(self) -> tuple[float, float] | None:
        return self._range_mhz

    def set_spectrum_bounds(self, lo_mhz: float, hi_mhz: float) -> None:
        """Record the active spectrum's MHz extent; default the window to full."""
        lo, hi = sorted((float(lo_mhz), float(hi_mhz)))
        self._bounds_mhz = (lo, hi)
        if self._range_mhz is None:
            self._range_mhz = (lo, hi)
        else:  # clamp an existing window into the new spectrum
            clamped_lo = max(lo, self._range_mhz[0])
            clamped_hi = min(hi, self._range_mhz[1])
            # A non-overlapping new spectrum would invert the window; fall back to
            # the full extent rather than analysing a back-to-front/empty range.
            self._range_mhz = (clamped_lo, clamped_hi) if clamped_lo < clamped_hi else (lo, hi)
        self._refresh_range_spins()

    def set_range_mhz(self, lo_mhz: float, hi_mhz: float, *, emit: bool = False) -> None:
        """Set the window (canonical MHz), e.g. from a plot drag."""
        self._range_mhz = tuple(sorted((float(lo_mhz), float(hi_mhz))))
        self._refresh_range_spins()
        if emit:
            self.settings_changed.emit()

    def set_cutoff_fraction(self, fraction: float, *, emit: bool = False) -> None:
        """Set the cutoff (fraction in [0, 1)), e.g. from a plot drag."""
        self._cutoff_fraction = float(min(max(fraction, 0.0), 0.99))
        blocked = self._cutoff_spin.blockSignals(True)
        self._cutoff_spin.setValue(self._cutoff_fraction * 100.0)
        self._cutoff_spin.blockSignals(blocked)
        if emit:
            self.settings_changed.emit()

    def recipe(self) -> dict:
        """Return the extraction recipe (unit, range in MHz, cutoff fraction)."""
        return {
            "unit": self._unit.value,
            "range_mhz": None if self._range_mhz is None else list(self._range_mhz),
            "cutoff_fraction": self._cutoff_fraction,
        }

    # ── readout ───────────────────────────────────────────────────────────

    def show_moments(self, moments: SpectrumMoments | None) -> None:
        """Fill the readout from *moments* (or blank when ``None``/empty)."""
        if moments is None or moments.n_sample == 0:
            self.clear_readout()
            if moments is not None:
                self._points_label.setText("Points: 0 (empty window)")
            return
        suffix = _UNIT_SUFFIX[self._unit]
        for _label, attr, err_attr in _READOUT_ROWS:
            value = float(getattr(moments, attr))
            error = float(getattr(moments, err_attr))
            row_suffix = "" if attr in _DIMENSIONLESS else suffix
            self._value_labels[attr].setText(_format_value_error(value, error, row_suffix))
        note = "" if moments.peak_refined else "  (B_pk: discrete bin — edge)"
        self._points_label.setText(f"Points: {moments.n_sample}{note}")

    def clear_readout(self) -> None:
        for label in self._value_labels.values():
            label.setText("—")
        self._points_label.setText("Points: —")

    # ── persistence ───────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot (additive, namespaced by the host)."""
        return self.recipe()

    def restore_state(self, state: dict | None) -> None:
        """Restore from *state*; tolerant of absence / partial dicts."""
        if not isinstance(state, dict):
            return
        unit = FieldUnit.coerce(state.get("unit"), FieldUnit.GAUSS)
        idx = self._unit_combo.findData(unit.value)
        if idx >= 0:
            blocked = self._unit_combo.blockSignals(True)
            self._unit_combo.setCurrentIndex(idx)
            self._unit_combo.blockSignals(blocked)
            self._unit = unit
            self._range_suffix_label.setText(_UNIT_SUFFIX[self._unit])
        rng = state.get("range_mhz")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                self._range_mhz = (float(rng[0]), float(rng[1]))
            except (TypeError, ValueError):
                pass
        cutoff = state.get("cutoff_fraction")
        if isinstance(cutoff, (int, float)) and math.isfinite(float(cutoff)):
            self.set_cutoff_fraction(float(cutoff))
        self._refresh_range_spins()

    # ── internal handlers ─────────────────────────────────────────────────

    def _refresh_range_spins(self) -> None:
        self._range_suffix_label.setText(_UNIT_SUFFIX[self._unit])
        if self._range_mhz is None:
            return
        lo = float(convert(self._range_mhz[0], FieldUnit.MHZ, self._unit))
        hi = float(convert(self._range_mhz[1], FieldUnit.MHZ, self._unit))
        for spin, val in ((self._range_min_spin, lo), (self._range_max_spin, hi)):
            blocked = spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(blocked)

    def _on_unit_changed(self, _index: int) -> None:
        self._unit = FieldUnit.coerce(self._unit_combo.currentData(), FieldUnit.GAUSS)
        self._refresh_range_spins()
        self.settings_changed.emit()

    def _on_range_edited(self) -> None:
        lo_disp = self._range_min_spin.value()
        hi_disp = self._range_max_spin.value()
        lo = float(convert(lo_disp, self._unit, FieldUnit.MHZ))
        hi = float(convert(hi_disp, self._unit, FieldUnit.MHZ))
        self._range_mhz = tuple(sorted((lo, hi)))
        self.settings_changed.emit()

    def _on_cutoff_edited(self) -> None:
        self._cutoff_fraction = float(min(max(self._cutoff_spin.value() / 100.0, 0.0), 0.99))
        self.settings_changed.emit()
