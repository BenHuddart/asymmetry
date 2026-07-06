"""Rotating-reference-frame display controls for the time-domain plot panel.

The widget is self-contained (W10): :func:`install_rrf_controls` builds it,
parents it into the plot panel and wires the signals, so the panel itself
needs only a one-line insertion hook.  The display transform lives here too —
:func:`rrf_display_dataset` / :func:`rrf_display_fit_curve` wrap the panel's
analysis datasets at draw time, so the fit data path
(``get_analysis_dataset`` → ``mainwindow._get_fit_dataset``) keeps consuming
raw lab-frame data.  Quantitative rotating-frame work belongs to
:mod:`asymmetry.core.fitting.rrf_offset`, not to fits of this display curve.

ν₀ entry converts between MHz and Gauss exclusively through
:func:`asymmetry.core.fourier.units.convert` (W16); the canonical stored
value is MHz.  State persists as ``plot_state["rrf"]`` — additive, restore
tolerates absence (W1).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.spectrum import reference_field_gauss
from asymmetry.core.fourier.units import FieldUnit, convert, gauss_to_mhz
from asymmetry.core.transform.rrf import rrf_demodulate, rrf_demodulate_values
from asymmetry.gui.styles.metrics import field_width_for

__all__ = [
    "RRFControls",
    "install_rrf_controls",
    "rrf_display_dataset",
    "rrf_display_fit_curve",
    "rrf_draw_badge",
]

_COMPONENT_CHOICES: list[tuple[str, str]] = [
    ("In-phase (Re)", "real"),
    ("Quadrature (Im)", "imag"),
    ("Magnitude", "magnitude"),
]

_UNIT_CHOICES: list[tuple[str, FieldUnit]] = [
    ("MHz", FieldUnit.MHZ),
    ("Gauss", FieldUnit.GAUSS),
]


class RRFControls(QWidget):
    """Enable / ν₀ (MHz⇄Gauss) / phase / bandwidth / component controls."""

    #: Emitted whenever any control changes in a way that affects the display.
    rrf_changed = Signal()

    def __init__(self, plot_panel: QWidget) -> None:
        super().__init__(plot_panel)
        self._panel = plot_panel
        self._active_view_token: str = "fb_asymmetry"
        self._current_unit = FieldUnit.MHZ
        # RRF is advanced/niche functionality gated behind Options → Advanced →
        # "Rotating reference frame" (an app-level QSettings preference, default
        # off). Until the feature is enabled the controls are genuinely absent
        # (zero layout footprint) and the display transform never applies, so
        # the majority of users who never use RRF pay nothing for it.
        self._feature_enabled: bool = False

        self._enable_check = QCheckBox("Rotating frame")
        self._enable_check.setToolTip(
            "Demodulate the FB asymmetry into a frame rotating at ν₀ "
            "(complex demodulation + low-pass). Display only — fits keep "
            "using the raw data."
        )

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setDecimals(4)
        self._freq_spin.setRange(0.0, 10_000_000.0)
        self._freq_spin.setValue(0.0)
        self._freq_spin.setMinimumWidth(field_width_for(11, self._freq_spin))
        self._freq_spin.setToolTip(
            "Frame frequency ν₀. Auto-seeded from the run's field metadata "
            "(γ_μB/2π) when first enabled."
        )

        self._unit_combo = QComboBox()
        for label, unit in _UNIT_CHOICES:
            self._unit_combo.addItem(label, unit.value)

        self._phase_spin = QDoubleSpinBox()
        self._phase_spin.setDecimals(1)
        self._phase_spin.setRange(-360.0, 360.0)
        self._phase_spin.setSuffix("°")
        self._phase_spin.setToolTip("Frame phase φ; the carrier is e^(−i(2πν₀t+φ)).")

        self._bandwidth_spin = QDoubleSpinBox()
        self._bandwidth_spin.setDecimals(3)
        self._bandwidth_spin.setRange(0.0, 100_000.0)
        self._bandwidth_spin.setSuffix(" MHz")
        self._bandwidth_spin.setSpecialValueText("Auto")
        self._bandwidth_spin.setValue(0.0)
        self._bandwidth_spin.setToolTip(
            "Low-pass cutoff for the demodulated curve. Auto picks ν₀/2, "
            "clamped below the (possibly aliased) 2ν₀ image."
        )

        self._component_combo = QComboBox()
        for label, key in _COMPONENT_CHOICES:
            self._component_combo.addItem(label, key)
        self._component_combo.setToolTip(
            "In-phase follows the frame phase; quadrature diagnoses a phase "
            "error; magnitude is the phase-free envelope (noise-biased where "
            "the signal is comparable to its errors)."
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(4)
        row.addWidget(self._enable_check)
        row.addSpacing(4)
        self._freq_label = QLabel("ν₀:")
        row.addWidget(self._freq_label)
        row.addWidget(self._freq_spin)
        row.addWidget(self._unit_combo)
        self._phase_label = QLabel("φ:")
        row.addWidget(self._phase_label)
        row.addWidget(self._phase_spin)
        self._bandwidth_label = QLabel("Bandwidth:")
        row.addWidget(self._bandwidth_label)
        row.addWidget(self._bandwidth_spin)
        row.addWidget(self._component_combo)
        row.addStretch()

        self._enable_check.toggled.connect(self._on_enable_toggled)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        self._freq_spin.valueChanged.connect(self._emit_changed)
        self._phase_spin.valueChanged.connect(self._emit_changed)
        self._bandwidth_spin.valueChanged.connect(self._emit_changed)
        self._component_combo.currentIndexChanged.connect(self._emit_changed)

        self._sync_field_enabled()
        self.refresh_visibility()

    # ── state ──────────────────────────────────────────────────────────────

    def has_active_frame(self) -> bool:
        """True when a frame is configured (enabled + ν₀ > 0), feature-independent.

        The single definition of "this project carries an active RRF frame",
        shared by :meth:`is_active` and the host's open-project auto-enable
        check, so the two never drift.
        """
        return bool(self._enable_check.isChecked()) and self.frequency_mhz() > 0.0

    def is_active(self) -> bool:
        """True when RRF display should transform the plotted curve.

        Gated by the Advanced feature toggle: when the feature is off the
        controls are hidden, so a stale ``enabled`` flag (e.g. restored from a
        project saved before the toggle existed) must not silently demodulate.
        """
        return self._feature_enabled and self.has_active_frame()

    def feature_enabled(self) -> bool:
        """Whether the Options → Advanced RRF toggle is on."""
        return self._feature_enabled

    def set_feature_enabled(self, enabled: bool) -> None:
        """Enable/disable the whole RRF surface (the Advanced toggle).

        When off the controls are removed from the layout (zero footprint) and
        the display transform is inert; when on they reappear under the usual
        view condition (the FB-asymmetry time view).
        """
        self._feature_enabled = bool(enabled)
        self.refresh_visibility()

    def frequency_mhz(self) -> float:
        """Canonical frame frequency in MHz, whatever the display unit."""
        return float(convert(self._freq_spin.value(), self._display_unit(), FieldUnit.MHZ))

    def phase_deg(self) -> float:
        return float(self._phase_spin.value())

    def bandwidth_mhz(self) -> float | None:
        """User cutoff in MHz, or ``None`` for the sampling-aware default."""
        value = float(self._bandwidth_spin.value())
        return value if value > 0.0 else None

    def component(self) -> str:
        return str(self._component_combo.currentData())

    def demodulation_kwargs(self) -> dict:
        """Keyword arguments for :func:`asymmetry.core.transform.rrf.rrf_demodulate`."""
        return {
            "frequency_mhz": self.frequency_mhz(),
            "phase_deg": self.phase_deg(),
            "bandwidth_mhz": self.bandwidth_mhz(),
        }

    def get_state(self) -> dict:
        """Serialisable snapshot for ``plot_state["rrf"]``."""
        return {
            "enabled": bool(self._enable_check.isChecked()),
            "frequency_mhz": self.frequency_mhz(),
            "display_unit": self._display_unit().value,
            "phase_deg": self.phase_deg(),
            "bandwidth_mhz": self.bandwidth_mhz(),
            "component": self.component(),
        }

    def set_state(self, state: object) -> None:
        """Restore from ``plot_state["rrf"]``.

        Absence or junk restores the defaults (RRF off, ν₀ cleared): a state
        saved without the key means "raw display", and silently keeping the
        previous session's frame would demodulate the restored project at a
        stale frequency.
        """
        if not isinstance(state, dict):
            state = {}
        blockers = [
            QSignalBlocker(w)
            for w in (
                self._enable_check,
                self._freq_spin,
                self._unit_combo,
                self._phase_spin,
                self._bandwidth_spin,
                self._component_combo,
            )
        ]
        try:
            unit = FieldUnit.coerce(state.get("display_unit"), default=FieldUnit.MHZ)
            if unit not in {u for _, u in _UNIT_CHOICES}:
                unit = FieldUnit.MHZ
            idx = self._unit_combo.findData(unit.value)
            self._unit_combo.setCurrentIndex(max(idx, 0))
            self._current_unit = unit

            freq_mhz = _safe_float(state.get("frequency_mhz"), default=0.0)
            self._freq_spin.setValue(
                float(convert(freq_mhz, FieldUnit.MHZ, unit)) if freq_mhz > 0.0 else 0.0
            )

            self._phase_spin.setValue(_safe_float(state.get("phase_deg"), default=0.0))

            bandwidth = state.get("bandwidth_mhz")
            self._bandwidth_spin.setValue(
                _safe_float(bandwidth, default=0.0) if bandwidth is not None else 0.0
            )

            comp_idx = self._component_combo.findData(str(state.get("component", "real")))
            self._component_combo.setCurrentIndex(max(comp_idx, 0))

            self._enable_check.setChecked(bool(state.get("enabled", False)))
        finally:
            del blockers
        self._sync_field_enabled()
        self.refresh_visibility()

    # ── view gating ────────────────────────────────────────────────────────

    def set_active_view_token(self, token: object) -> None:
        """Track the workspace's active representation token (post-#53 seam)."""
        self._active_view_token = str(token or "")
        self.refresh_visibility()

    def applies_to_current_view(self) -> bool:
        """True when the active representation is the FB-asymmetry time view."""
        panel = self._panel
        if getattr(panel, "_is_frequency_plot_panel", lambda: False)():
            return False
        if self._active_view_token not in {"", "fb_asymmetry"}:
            return False
        current_mode = getattr(panel, "current_time_view_mode", lambda: "fb_asymmetry")()
        return current_mode == "fb_asymmetry"

    def refresh_visibility(self) -> None:
        """Show the controls only when the feature is on and the view fits (W16)."""
        self.setVisible(self._feature_enabled and self.applies_to_current_view())

    # ── internals ──────────────────────────────────────────────────────────

    def _display_unit(self) -> FieldUnit:
        return FieldUnit.coerce(self._unit_combo.currentData(), default=FieldUnit.MHZ)

    def _sync_field_enabled(self) -> None:
        enabled = bool(self._enable_check.isChecked())
        for widget in (
            self._freq_label,
            self._freq_spin,
            self._unit_combo,
            self._phase_label,
            self._phase_spin,
            self._bandwidth_label,
            self._bandwidth_spin,
            self._component_combo,
        ):
            widget.setEnabled(enabled)

    def _on_enable_toggled(self, checked: bool) -> None:
        if checked and self._freq_spin.value() <= 0.0:
            seed = self._field_seed_mhz()
            if seed is not None and seed > 0.0:
                with QSignalBlocker(self._freq_spin):
                    self._freq_spin.setValue(
                        float(convert(seed, FieldUnit.MHZ, self._display_unit()))
                    )
        self._sync_field_enabled()
        self._emit_changed()

    def _field_seed_mhz(self) -> float | None:
        """ν₀ = γ_μB/2π from the run's field metadata, via the shared resolver."""
        dataset = getattr(self._panel, "_current_dataset", None)
        if dataset is None:
            return None
        field_gauss = reference_field_gauss(getattr(dataset, "run", None), dataset)
        if field_gauss is None or not np.isfinite(field_gauss) or field_gauss <= 0.0:
            return None
        return float(gauss_to_mhz(field_gauss))

    def _on_unit_changed(self) -> None:
        # Convert the displayed number so the physical ν₀ is unchanged. The
        # previous unit is tracked explicitly — inferring it from "the other
        # combo entry" breaks the moment a third unit (Tesla) joins.
        new_unit = self._display_unit()
        old_unit = self._current_unit
        self._current_unit = new_unit
        value = self._freq_spin.value()
        if value > 0.0 and new_unit is not old_unit:
            with QSignalBlocker(self._freq_spin):
                self._freq_spin.setValue(float(convert(value, old_unit, new_unit)))
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.rrf_changed.emit()


def _safe_float(raw: object, *, default: float) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def install_rrf_controls(plot_panel: QWidget) -> RRFControls:
    """Build the controls, attach them to *plot_panel*, and wire redraws.

    The panel's only obligation is to add the returned widget to its layout;
    everything else (visibility tracking via ``time_view_changed``, redraw on
    change) is wired here.
    """
    controls = RRFControls(plot_panel)
    plot_panel._rrf_controls = controls  # noqa: SLF001 — the panel's own attribute
    if hasattr(plot_panel, "time_view_changed"):
        plot_panel.time_view_changed.connect(lambda _mode: controls.refresh_visibility())
    if hasattr(plot_panel, "_redraw_current_view"):
        controls.rrf_changed.connect(plot_panel._redraw_current_view)  # noqa: SLF001
    return controls


def _active_controls(plot_panel: object) -> RRFControls | None:
    controls = getattr(plot_panel, "_rrf_controls", None)
    if controls is None or not controls.is_active() or not controls.applies_to_current_view():
        return None
    return controls


def rrf_display_dataset(plot_panel: object, dataset: MuonDataset | None) -> MuonDataset | None:
    """Apply the RRF display transform to an analysis dataset, if active.

    Returns *dataset* unchanged when RRF is off, the view is not the
    FB-asymmetry time view, or the dataset is a frequency-domain / derived
    plot (recognised by ``plot_domain`` / ``x_label`` metadata).  Invalid
    bins (filter edges, non-finite holes) become NaN so the existing
    finite-mask drawing skips them.
    """
    if dataset is None:
        return None
    controls = _active_controls(plot_panel)
    if controls is None:
        return dataset
    metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
    is_frequency = getattr(plot_panel, "_is_frequency_domain_dataset", None)
    if callable(is_frequency):
        if is_frequency(dataset):
            return dataset
    elif str(metadata.get("plot_domain", "")).strip().lower() == "frequency":
        return dataset
    if metadata.get("x_label"):
        # Derived plots (integral scan, reconstruction views) plot something
        # other than the FB asymmetry against time.
        return dataset
    try:
        curve = rrf_demodulate(
            dataset.time, dataset.asymmetry, dataset.error, **controls.demodulation_kwargs()
        )
    except ValueError:
        return dataset
    valid_idx = np.flatnonzero(curve.valid)
    if valid_idx.size == 0:
        return dataset
    # Trim the filter-edge region by slicing (the panel's limit init uses
    # plain .min()/.max(), which NaN edges would poison); interior holes keep
    # NaN, matching how non-finite bins behave on the raw display.
    start, stop = int(valid_idx[0]), int(valid_idx[-1]) + 1
    sel = slice(start, stop)
    values, errors = curve.component(controls.component())
    values = np.where(curve.valid, values, np.nan)[sel]
    errors = np.where(curve.valid, errors, np.nan)[sel]
    frame_label = curve.frame_label(controls.component())
    new_metadata = dict(metadata)
    new_metadata["rrf_frame"] = frame_label
    # The resolved demodulation parameters: overlays (fit curves, model
    # components) must use exactly these — re-resolving an "Auto" bandwidth
    # on a differently-sampled overlay grid would filter the two curves
    # differently. The trim record restores the cheap mask-projection paths.
    new_metadata["rrf_frequency_mhz"] = float(curve.frequency_mhz)
    new_metadata["rrf_phase_deg"] = float(curve.phase_deg)
    new_metadata["rrf_bandwidth_mhz"] = float(curve.bandwidth_mhz)
    new_metadata["rrf_trim"] = [start, stop, int(np.asarray(dataset.time).size)]
    plot_panel._rrf_frame_drawn = frame_label  # noqa: SLF001 — badge handshake
    return MuonDataset(
        time=np.asarray(dataset.time, dtype=float)[sel],
        asymmetry=np.asarray(values, dtype=float),
        error=np.asarray(errors, dtype=float),
        metadata=new_metadata,
        run=dataset.run,
    )


def rrf_display_fit_curve(
    plot_panel: object,
    fit_to_plot: tuple | None,
    analysis_dataset: MuonDataset | None,
) -> tuple | None:
    """Demodulate a stored fit-curve overlay through the same pipeline.

    The model curve must transform with the data or the overlay turns into
    the fast lab-frame oscillation on top of a slow envelope — the WiMDA
    comparison-ledger item 4 trap.  The overlay transforms exactly when the
    *displayed* dataset did, with the parameters recorded on it (so an
    "Auto" bandwidth resolved on the data grid is not re-resolved on the
    overlay's, typically finer, grid).  Invalid bins become NaN (matplotlib
    breaks the line there).
    """
    if fit_to_plot is None:
        return None
    metadata = (
        analysis_dataset.metadata
        if analysis_dataset is not None and isinstance(analysis_dataset.metadata, dict)
        else {}
    )
    if not metadata.get("rrf_frame"):
        return fit_to_plot
    controls = getattr(plot_panel, "_rrf_controls", None)
    if controls is None:
        return fit_to_plot
    t_fit, y_fit, label = fit_to_plot
    try:
        curve = rrf_demodulate_values(
            np.asarray(t_fit, dtype=float),
            np.asarray(y_fit, dtype=float),
            frequency_mhz=float(metadata["rrf_frequency_mhz"]),
            phase_deg=float(metadata.get("rrf_phase_deg", 0.0)),
            bandwidth_mhz=float(metadata["rrf_bandwidth_mhz"]),
        )
    except (KeyError, TypeError, ValueError):
        return fit_to_plot
    values, _ = curve.component(controls.component())
    return (
        np.asarray(t_fit, dtype=float),
        np.where(curve.valid, values, np.nan),
        label,
    )


def rrf_draw_badge(plot_panel: object, ax: object) -> None:
    """Draw the self-describing frame badge on the axes.

    The badge reflects what was *actually drawn*: ``rrf_display_dataset``
    flags the frame label on the panel when (and only when) it transformed a
    curve in the current draw cycle, and the draw paths clear the flag on
    entry — so a declined transform (degenerate grid, all-invalid filter)
    never mislabels raw data as rotating-frame.  Anchored top-right inside
    the axes so every figure export carries the frame; styled to the muted
    tick-label grey of the BENCH plot grammar.
    """
    label = getattr(plot_panel, "_rrf_frame_drawn", None)
    if not label:
        return
    try:
        from asymmetry.gui.styles import tokens

        ax.text(
            0.985,
            0.985,
            str(label),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=tokens.PLOT_TICK_LABEL,
            zorder=6,
        )
    except Exception:  # noqa: BLE001 — annotation must never break the draw
        return
