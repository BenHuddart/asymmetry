"""Dedicated background-configuration dialog for the grouping profile editor.

Moved out of the main grouping form (:mod:`asymmetry.gui.windows.grouping.dialog`)
so the main window can show a compact status row + "Configure…" button instead of
the mode combo + status label. This module mirrors the historical modes exactly —
it moves the widgets/logic rather than reinventing them (the reference-run pick
flow, the tail-fit preview text, and the ``background_run`` payload shape are
unchanged) — and adds a small raw-counts preview using the shared
:func:`~asymmetry.gui.widgets.mpl_canvas.create_canvas` foundation.

Modes (:data:`asymmetry.core.project.profiles.BACKGROUND_POLICY_MODES`):

* ``none`` — no subtraction.
* ``range`` — musrfit's pre-t0 range average (continuous-source data only).
* ``tail_fit`` — WiMDA's late-time exponential + flat fit (pulsed-source mode).
* ``reference_run`` — subtract a designated background run, frame-scaled.
* ``fixed`` — subtract stored per-group constants (from a loaded ``.grp``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.project.profiles import BackgroundPolicy
from asymmetry.core.transform import fit_tail_background
from asymmetry.gui.windows.grouping.grp_io import (
    format_value_with_uncertainty as _format_value_with_uncertainty,
)

__all__ = ["BackgroundDialog", "BackgroundReferenceRunCandidate", "background_status_text"]


_MODE_ITEMS = (
    ("None", "none", "No background subtraction."),
    (
        "Range average (pre-t0)",
        "range",
        "Mean count over a pre-t0 bin window (musrfit convention) — "
        "continuous-source data only; pulsed files have no pre-t0 region.",
    ),
    (
        "Tail fit (late-time)",
        "tail_fit",
        "Fit muon exponential + flat rate to the late half of the good "
        "window and subtract the flat part — the pulsed-source mode.",
    ),
    (
        "Background run…",
        "reference_run",
        "Subtract a reference run (sample holder / silver / laser-off) "
        "scaled by the good-frame ratio.",
    ),
    (
        "Fixed values",
        "fixed",
        "Subtract stored per-group constants (from the loaded grouping).",
    ),
)


class BackgroundReferenceRunCandidate:
    """One candidate run offered for the reference-run background pick."""

    def __init__(
        self, run_number: int, label: str, source_file: str, good_frames: float | None
    ) -> None:
        """Store the candidate's run number, display label, source file, and good frames."""
        self.run_number = run_number
        self.label = label
        self.source_file = source_file
        self.good_frames = good_frames


def background_status_text(policy: BackgroundPolicy) -> str:
    """Return the main window's compact status-row text for *policy*."""
    if policy.mode == "none":
        return "Background: none"
    if policy.mode == "range":
        ranges = policy.details.get("background_ranges") or policy.details.get("background_range")
        if ranges:
            return f"Background: pre-t0 range {ranges}"
        return "Background: pre-t0 range"
    if policy.mode == "tail_fit":
        return "Background: tail fit (late-time)"
    if policy.mode == "reference_run":
        run_payload = policy.details.get("background_run") or {}
        run_number = run_payload.get("run_number")
        label = f"run {run_number}" if run_number else str(run_payload.get("source_file", ""))
        return f"Background: reference {label}".rstrip()
    if policy.mode == "fixed":
        values = policy.details.get("background_fixed_values")
        if values:
            return f"Background: fixed {list(values)}"
        return "Background: fixed values"
    return "Background: none"


class BackgroundDialog(QDialog):
    """Edit the background-subtraction policy for the current grouping draft.

    Parameters
    ----------
    available_modes
        Modes gated in per the dataset (:func:`available_background_modes`);
        ``"none"`` is always available. Modes outside this set are shown but
        disabled, matching the historical inline combo.
    has_fixed_values
        Whether the draft carries stored fixed background values, gating the
        "Fixed values" entry's visibility (as the inline combo did).
    initial_mode
        Starting mode.
    background_run_payload
        Current ``background_run`` payload dict (``run_number``/``source_file``/
        good-frame keys), or ``None``.
    reference_run_candidates
        Datasets offered by the reference-run picker (excludes the preview run).
    preview
        Optional ``(forward_counts, backward_counts, bin_width_us, t0_bin,
        last_good_bin)`` for the raw-counts preview; ``None`` disables it.
    forward_label, backward_label
        Group labels for the preview legend.
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        *,
        available_modes: tuple[str, ...],
        has_fixed_values: bool,
        initial_mode: str,
        background_run_payload: dict[str, Any] | None,
        reference_run_candidates: list[BackgroundReferenceRunCandidate],
        preview: tuple[Any, Any, float, int, int] | None = None,
        forward_label: str = "F",
        backward_label: str = "B",
        parent=None,
    ) -> None:
        """Build the dialog; see the class docstring for parameter semantics."""
        super().__init__(parent)
        self.setWindowTitle("Background Correction")
        self.resize(560, 480)

        self._available_modes = set(available_modes) | {"none"}
        self._has_fixed_values = bool(has_fixed_values)
        self._background_run_payload = (
            dict(background_run_payload) if isinstance(background_run_payload, dict) else None
        )
        self._reference_run_candidates = list(reference_run_candidates)
        self._preview = preview
        self._forward_label = forward_label
        self._backward_label = backward_label
        self._figure = None
        self._canvas = None

        root = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._populate_modes()
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        root.addLayout(mode_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        self._reference_row_widget = QWidget()
        reference_row = QHBoxLayout(self._reference_row_widget)
        reference_row.setContentsMargins(0, 0, 0, 0)
        self._reference_summary_label = QLabel("")
        self._reference_summary_label.setWordWrap(True)
        reference_row.addWidget(self._reference_summary_label)
        self._pick_reference_btn = QPushButton("Choose run…")
        self._pick_reference_btn.setAutoDefault(False)
        self._pick_reference_btn.setDefault(False)
        self._pick_reference_btn.clicked.connect(self._pick_background_run)
        reference_row.addWidget(self._pick_reference_btn)
        reference_row.addStretch()
        root.addWidget(self._reference_row_widget)

        self._canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._canvas_container, stretch=1)
        self._build_preview_canvas(canvas_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._set_mode(initial_mode)
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Mode combo
    # ------------------------------------------------------------------

    def _populate_modes(self) -> None:
        self._mode_combo.blockSignals(True)
        self._mode_combo.clear()
        for label, key, tooltip in _MODE_ITEMS:
            if key == "fixed" and not self._has_fixed_values:
                continue
            self._mode_combo.addItem(label, key)
            index = self._mode_combo.count() - 1
            self._mode_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
            if key != "none" and key not in self._available_modes:
                item = self._mode_combo.model().item(index)
                if item is not None:
                    item.setEnabled(False)
        self._mode_combo.blockSignals(False)

    def _set_mode(self, mode: str) -> None:
        idx = self._mode_combo.findData(str(mode).strip().lower())
        self._mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def current_mode(self) -> str:
        """Return the mode key currently selected in the mode combo."""
        return str(self._mode_combo.currentData() or "none")

    def _on_mode_changed(self) -> None:
        mode = self.current_mode()
        if mode == "reference_run" and not self._background_run_payload:
            self._pick_background_run()
        self._reference_row_widget.setVisible(mode == "reference_run")
        self._update_status()
        self._update_preview_visibility()
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Reference-run picker (moved verbatim from the inline dialog)
    # ------------------------------------------------------------------

    def _pick_background_run(self) -> None:
        labels = [c.label for c in self._reference_run_candidates] + ["Browse for file…"]
        choice, accepted = QInputDialog.getItem(
            self,
            "Background Run",
            "Reference run to subtract (scaled by the good-frame ratio):",
            labels,
            0,
            False,
        )
        if not accepted:
            self._set_mode("none")
            return
        if choice == "Browse for file…":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Background Run File",
                "",
                "Muon data (*.nxs *.bin *.mdu *.root);;All files (*)",
            )
            if not path:
                self._set_mode("none")
                return
            self._background_run_payload = {"run_number": None, "source_file": path}
        else:
            candidate = self._reference_run_candidates[labels.index(choice)]
            self._background_run_payload = {
                "run_number": int(candidate.run_number),
                "source_file": candidate.source_file,
                "good_frames_reference": candidate.good_frames,
            }
        self._update_status()

    # ------------------------------------------------------------------
    # Status line (moved verbatim from ``_update_background_status``)
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        mode = self.current_mode()
        if mode == "tail_fit":
            self._status_label.setText(self._tail_fit_preview_text())
            return
        if mode == "reference_run":
            payload = self._background_run_payload or {}
            run_number = payload.get("run_number")
            label = f"run {run_number}" if run_number else str(payload.get("source_file", ""))
            self._reference_summary_label.setText(f"Selected: {label}" if label else "")
            sample = payload.get("good_frames_sample")
            reference = payload.get("good_frames_reference")
            try:
                scale = float(sample) / float(reference)
                self._status_label.setText(f"Subtract {label}, frame-ratio scale {scale:.4g}.")
            except (TypeError, ValueError, ZeroDivisionError):
                self._status_label.setText(f"Subtract {label} (frame ratio resolved at reduction).")
            return
        self._status_label.setText("")

    def _tail_fit_preview_text(self) -> str:
        if self._preview is None:
            return ""
        forward, backward, bin_width_us, t0_bin, last_good_bin = self._preview
        parts: list[str] = []
        for name, counts in ((self._forward_label, forward), (self._backward_label, backward)):
            fit = fit_tail_background(
                counts,
                bin_width_us=bin_width_us,
                t0_bin=int(t0_bin),
                last_good_bin=int(last_good_bin),
            )
            if not fit.ok:
                parts.append(f"{name}: {fit.message}")
                continue
            value = _format_value_with_uncertainty(fit.rate_per_us, fit.rate_error_per_us)
            note = " (consistent with zero)" if fit.consistent_with_zero else ""
            parts.append(f"{name}: {value} counts/µs{note}")
        return "Tail-fit background — " + "; ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Preview canvas (F+B raw counts, log y, shaded active window)
    # ------------------------------------------------------------------

    def _build_preview_canvas(self, layout: QVBoxLayout) -> None:
        if self._preview is None:
            return
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas = create_canvas(layout="tight", figsize=(5.0, 3.0))
        except ImportError:
            self._figure = None
            self._canvas = None
            return
        layout.addWidget(self._canvas)

    def _update_preview_visibility(self) -> None:
        if self._canvas is None:
            return
        # "Fixed" mode has nothing sensible to preview (no run-derived window).
        self._canvas_container.setVisible(
            self.current_mode() != "fixed" and self._preview is not None
        )

    def _refresh_preview(self) -> None:
        if self._canvas is None or self._figure is None or self._preview is None:
            return
        mode = self.current_mode()
        if mode == "fixed":
            return
        forward, backward, bin_width_us, t0_bin, last_good_bin = self._preview

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        forward_arr = np.asarray(forward, dtype=float)
        backward_arr = np.asarray(backward, dtype=float)
        n = min(forward_arr.size, backward_arr.size)
        times_us = np.arange(n, dtype=float) * float(bin_width_us)
        ax.plot(times_us, np.clip(forward_arr[:n], 1e-6, None), label=self._forward_label, lw=0.8)
        ax.plot(times_us, np.clip(backward_arr[:n], 1e-6, None), label=self._backward_label, lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("time (µs)")
        ax.set_ylabel("counts")
        ax.legend(fontsize="small")

        window = self._active_window_us(bin_width_us, t0_bin, last_good_bin)
        if window is not None:
            start_us, end_us = window
            ax.axvspan(start_us, end_us, color="orange", alpha=0.2)
        self._canvas.draw_idle()

    def _active_window_us(
        self, bin_width_us: float, t0_bin: int, last_good_bin: int
    ) -> tuple[float, float] | None:
        mode = self.current_mode()
        if mode == "range":
            start_bin = int(float(t0_bin) * 0.1)
            end_bin = int(float(t0_bin) * 0.6)
            return start_bin * bin_width_us, end_bin * bin_width_us
        if mode == "tail_fit":
            end = int(last_good_bin)
            start = int(t0_bin) + (end - int(t0_bin)) // 2
            return start * bin_width_us, end * bin_width_us
        return None

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def get_policy(self) -> BackgroundPolicy:
        """Return the edited :class:`BackgroundPolicy`."""
        mode = self.current_mode()
        if mode == "none":
            return BackgroundPolicy(mode="none")
        details: dict[str, Any] = {}
        if mode == "reference_run" and self._background_run_payload:
            details["background_run"] = dict(self._background_run_payload)
        return BackgroundPolicy(mode=mode, details=details)
