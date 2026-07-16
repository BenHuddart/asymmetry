"""Inline background-configuration controls for the grouping Corrections panel.

This is the embeddable body of the retired ``BackgroundDialog`` — the mode combo,
a per-mode status line, and the reference-run picker — minus the modal shell and
the raw-counts canvas. The unified grouping preview
(:class:`~asymmetry.gui.windows.grouping.preview_pane.GroupingPreviewPane`) now
shows the effect of the subtraction, so the section carries no plot of its own.

The widget owns no reduction state: it emits :attr:`changed` when the mode or the
reference run changes, and the owning grouping dialog reads :meth:`mode` /
:meth:`background_run_payload` and re-previews. Modes match
:data:`asymmetry.core.project.profiles.BACKGROUND_POLICY_MODES` exactly (the
picker flow and ``background_run`` payload shape are unchanged from the modal).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.project.profiles import BackgroundPolicy

__all__ = ["BackgroundReferenceRunCandidate", "BackgroundSectionWidget", "background_status_text"]


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
    """Return a compact status-line description for *policy*."""
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


#: Provider the owner supplies so the reference picker always offers the current
#: fingerprint's datasets (excluding the preview run).
CandidatesProvider = Callable[[], "list[BackgroundReferenceRunCandidate]"]


class BackgroundSectionWidget(QWidget):
    """Mode combo + status + reference picker for the Corrections panel.

    Parameters
    ----------
    candidates_provider
        Called (lazily, when the picker opens) to list the reference-run
        candidates for the current fingerprint.
    parent
        Parent Qt widget.
    """

    #: Emitted when the mode or the selected reference run changes.
    changed = Signal()

    def __init__(
        self, candidates_provider: CandidatesProvider, parent: QWidget | None = None
    ) -> None:
        """Build the mode combo, status line and reference row."""
        super().__init__(parent)
        self._candidates_provider = candidates_provider
        self._available_modes: set[str] = {"none"}
        self._has_fixed_values = False
        self._background_run_payload: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, stretch=1)
        root.addLayout(mode_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        self._reference_row = QWidget()
        reference_layout = QHBoxLayout(self._reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        self._reference_summary_label = QLabel("")
        self._reference_summary_label.setWordWrap(True)
        reference_layout.addWidget(self._reference_summary_label, stretch=1)
        self._pick_reference_btn = QPushButton("Choose run…")
        self._pick_reference_btn.setAutoDefault(False)
        self._pick_reference_btn.setDefault(False)
        self._pick_reference_btn.clicked.connect(self._pick_background_run)
        reference_layout.addWidget(self._pick_reference_btn)
        root.addWidget(self._reference_row)

    # -- configuration ---------------------------------------------------

    def configure(
        self,
        *,
        available_modes: tuple[str, ...],
        has_fixed_values: bool,
        mode: str,
        background_run_payload: dict[str, Any] | None,
    ) -> None:
        """(Re)seed the section for the current dataset/draft, without emitting."""
        self._available_modes = set(available_modes) | {"none"}
        self._has_fixed_values = bool(has_fixed_values)
        self._background_run_payload = (
            dict(background_run_payload) if isinstance(background_run_payload, dict) else None
        )
        self._mode_combo.blockSignals(True)
        self._populate_modes()
        idx = self._mode_combo.findData(str(mode).strip().lower())
        self._mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._mode_combo.blockSignals(False)
        self._refresh_visibility_and_status()

    def _populate_modes(self) -> None:
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

    # -- queries ---------------------------------------------------------

    def mode(self) -> str:
        """The mode key currently selected."""
        return str(self._mode_combo.currentData() or "none")

    def background_run_payload(self) -> dict[str, Any] | None:
        """The selected reference-run payload (``reference_run`` mode)."""
        return dict(self._background_run_payload) if self._background_run_payload else None

    # -- reactions -------------------------------------------------------

    def _on_mode_changed(self) -> None:
        mode = self.mode()
        if mode == "reference_run" and not self._background_run_payload:
            # Prompt for a run the first time the mode is chosen, matching the
            # old dialog; a cancelled pick falls back to "none".
            self._pick_background_run(_from_mode_change=True)
            if self.mode() != "reference_run":
                return  # the pick was cancelled and reset the mode (already emitted)
        self._refresh_visibility_and_status()
        self.changed.emit()

    def _pick_background_run(self, _from_mode_change: bool = False) -> None:
        candidates = list(self._candidates_provider())
        labels = [c.label for c in candidates] + ["Browse for file…"]
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
            candidate = candidates[labels.index(choice)]
            self._background_run_payload = {
                "run_number": int(candidate.run_number),
                "source_file": candidate.source_file,
                "good_frames_reference": candidate.good_frames,
            }
        self._refresh_visibility_and_status()
        if not _from_mode_change:
            # A re-pick while already in reference_run mode still edits the policy.
            self.changed.emit()

    def _set_mode(self, mode: str) -> None:
        """Programmatically move to *mode* and notify (used by a cancelled pick)."""
        idx = self._mode_combo.findData(str(mode).strip().lower())
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._mode_combo.blockSignals(False)
        self._refresh_visibility_and_status()
        self.changed.emit()

    def _refresh_visibility_and_status(self) -> None:
        mode = self.mode()
        self._reference_row.setVisible(mode == "reference_run")
        if mode == "reference_run":
            payload = self._background_run_payload or {}
            run_number = payload.get("run_number")
            label = f"run {run_number}" if run_number else str(payload.get("source_file", ""))
            self._reference_summary_label.setText(f"Selected: {label}" if label else "")
        self._status_label.setText(background_status_text(self._policy()))

    def _policy(self) -> BackgroundPolicy:
        mode = self.mode()
        if mode == "none":
            return BackgroundPolicy(mode="none")
        details: dict[str, Any] = {}
        if mode == "reference_run" and self._background_run_payload:
            details["background_run"] = dict(self._background_run_payload)
        return BackgroundPolicy(mode=mode, details=details)
