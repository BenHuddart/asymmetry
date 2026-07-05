"""Author a new user fit function from the GUI (a formula or an advanced body).

This dialog is the interactive front end to
:mod:`asymmetry.core.fitting.user_function_authoring`. The author types a name,
a description, a maths formula in ``x`` and the parameter names, and a small
parameter table; the dialog live-validates the resulting
:class:`~asymmetry.core.fitting.user_function_authoring.UserFunctionDraft`
against exactly the checks a real registration runs, previews the curve, and on
accept writes+registers the function through
:func:`~asymmetry.core.fitting.user_function_authoring.create_user_function`.

Design notes worth keeping in view of the invariants:

* Building the draft callable runs *user code* on the GUI thread. That is
  acceptable **only** because the probe grids are tiny (~400 points) and the
  work is debounced behind a single-shot timer, so it never runs per-keystroke
  and never blocks long enough to need the worker machinery. Do not grow this
  into a long-running evaluation without moving it off the GUI thread.
* The generated function is an ordinary Asymmetry plugin — nothing here is a
  private on-disk format. The trust note echoes the wording in
  :mod:`asymmetry.gui.windows.user_functions_dialog`.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.user_function_authoring import (
    CreatedUserFunction,
    DraftParameter,
    UserFunctionDraft,
    create_user_function,
    detect_parameter_names,
    evaluate_draft,
    generate_function_body,
    validate_draft,
)
from asymmetry.core.fitting.user_functions import UserFunctionError
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.screen_sizing import resize_to_available

__all__ = ["NewUserFunctionDialog"]

#: How long the dialog waits after the last edit before re-validating. Long
#: enough that typing does not trigger a validation (which runs user code) per
#: keystroke, short enough to feel live.
_VALIDATION_DEBOUNCE_MS = 300

#: Preview grids mirror the core probe ranges (see
#: ``asymmetry.core.fitting.user_functions._PROBE_GRIDS``) so the curve the user
#: sees is drawn over the same domain the load-time probe validates against. A
#: coarser point count than the probe keeps the redraw cheap on the GUI thread.
_PREVIEW_POINTS = 400
_PREVIEW_RANGES: dict[str, tuple[float, float]] = {
    "time": (0.0, 32.0),
    "frequency": (0.0, 50.0),
    "parameter": (1e-3, 300.0),
}

_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "time": ("Time (µs)", "Asymmetry"),
    "frequency": ("Frequency (MHz)", "Amplitude"),
    "parameter": ("Trend variable x", "Parameter value"),
}

#: The muted caption under the formula field, keyed by the effective preview
#: kind ("time"/"frequency"/"parameter").
_FORMULA_HINTS: dict[str, str] = {
    "time": "x is time in µs.",
    "frequency": "x is frequency in MHz.",
    "parameter": "x is the trend variable (temperature, field, …).",
}

_MATH_NOTE = (
    "Bare maths names (exp, cos, sqrt, pi, …) and np.* are both fine, "
    "e.g. A*exp(-lam*x)*cos(2*pi*f*x + phi)."
)


def _in_test_mode() -> bool:
    """Return True when running under pytest, to suppress modal popups."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


class NewUserFunctionDialog(QDialog):
    """Create a new fit component or parameter-trend from a formula.

    Args:
        kind: ``"component"`` (a fit component) or ``"parameter"`` (a
            parameter-vs-x trend).
        domain: ``"time"`` or ``"frequency"``; component kind only. Controls the
            preview grid/labels and the registration domain. Ignored for the
            parameter kind (which always uses the parameter probe grid).
        directory: Directory to write the plugin file into, passed through to
            :func:`create_user_function`. ``None`` uses the default
            user-functions directory. Tests pass ``tmp_path``.
        parent: Parent widget.
    """

    def __init__(
        self,
        kind: str,
        *,
        domain: str = "time",
        directory: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._domain = domain
        self._directory = directory
        # The effective grid/label key: parameter kind ignores the domain.
        self._preview_kind = "parameter" if kind == "parameter" else domain
        self._created: CreatedUserFunction | None = None
        #: The last body text generated as the Advanced editor's prefill — used
        #: to tell an untouched prefill from an edited body when toggling off.
        self._last_generated_body: str = ""
        #: The advanced-editor body while advanced mode is on, else None. This
        #: is the draft's ``advanced_body``.
        self._advanced_active = False

        self.setWindowTitle("New User Function")

        self._build_ui()
        self._wire_signals()

        # Validate once so the initial state (OK disabled, hint shown) is
        # correct without waiting for an edit.
        self._run_validation()

        resize_to_available(self, 640, 760, min_width=560, min_height=560, center=True)

    # ── construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("StretchedOsc")
        layout.addLayout(self._labelled_row("Name", self._name_edit))

        # 2. Description
        self._description_edit = QLineEdit()
        self._description_edit.setPlaceholderText(
            "Shown in the library and the component info dialog"
        )
        layout.addLayout(self._labelled_row("Description", self._description_edit))

        # 3. Formula (single line — the core's math-name rewriting is
        #    single-expression). Advanced mode hides this and shows the editor.
        self._formula_edit = QLineEdit()
        self._formula_edit.setPlaceholderText("A*exp(-lam*x)*cos(2*pi*f*x + phi)")
        self._formula_row = self._labelled_row("Formula", self._formula_edit)
        layout.addLayout(self._formula_row)

        self._formula_hint = self._muted_label(f"{_FORMULA_HINTS[self._preview_kind]} {_MATH_NOTE}")
        layout.addWidget(self._formula_hint)

        # 6. Advanced editor (created here, shown when the toggle is on).
        self._advanced_editor = QPlainTextEdit()
        self._advanced_editor.setPlaceholderText(
            "x = np.asarray(x, dtype=float)\nresult = ...\nreturn result"
        )
        self._apply_mono_font(self._advanced_editor)
        self._advanced_editor.setVisible(False)
        self._advanced_editor.setMinimumHeight(96)
        layout.addWidget(self._advanced_editor)

        # 4/6. Detect parameters + advanced toggle, side by side.
        self._detect_button = QPushButton("Detect parameters")
        self._advanced_toggle = QCheckBox("Edit as Python (advanced)")
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self._detect_button)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self._advanced_toggle)
        layout.addLayout(toggle_row)

        # 5. Parameter table.
        layout.addWidget(self._muted_label("Parameters", bold=True))
        self._param_table = QTableWidget(0, 2)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Start value"])
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self._param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._param_table.setMinimumHeight(120)
        layout.addWidget(self._param_table)

        add_button = QPushButton("Add row")
        remove_button = QPushButton("Remove row")
        self._add_row_button = add_button
        self._remove_row_button = remove_button
        param_buttons = QHBoxLayout()
        param_buttons.addWidget(add_button)
        param_buttons.addWidget(remove_button)
        param_buttons.addStretch(1)
        layout.addLayout(param_buttons)

        if self._kind == "component":
            # By convention the first parameter of a component is its amplitude,
            # so pre-seed one A row and say so.
            self._append_param_row("A", 0.2)
            layout.addWidget(
                self._muted_label("By convention the first parameter is the amplitude.")
            )

        # 7. Preview canvas.
        self._build_preview(layout)

        # 8. Status label.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._status_label)

        # 9. Trust + persistence note (echoes user_functions_dialog wording).
        trust = self._muted_label(
            "User functions are ordinary Python run with full privileges — the "
            "same trust model as WiMDA's plugin DLLs. The file is saved to your "
            "user-functions folder and reloads every time Asymmetry starts, so "
            "you can edit it freely afterwards."
        )
        layout.addWidget(trust)

        # 10. OK / Cancel.
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self._buttons)

    def _build_preview(self, layout: QVBoxLayout) -> None:
        """Add the preview canvas, degrading to a label without matplotlib."""
        self._figure = None
        self._canvas = None
        self._axes = None
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas = create_canvas(layout="tight", figsize=(4.0, 2.2))
            self._axes = self._figure.add_subplot(111)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._canvas.setMinimumHeight(150)
            layout.addWidget(self._canvas, stretch=1)
        except ImportError:
            layout.addWidget(self._muted_label("matplotlib is not installed — no preview."))

    def _wire_signals(self) -> None:
        # Debounced live validation: restart the timer on any edit.
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(_VALIDATION_DEBOUNCE_MS)
        self._validation_timer.timeout.connect(self._run_validation)

        self._name_edit.textChanged.connect(self._schedule_validation)
        self._description_edit.textChanged.connect(self._schedule_validation)
        self._formula_edit.textChanged.connect(self._schedule_validation)
        self._advanced_editor.textChanged.connect(self._schedule_validation)
        self._param_table.itemChanged.connect(self._schedule_validation)

        self._detect_button.clicked.connect(self._on_detect_parameters)
        self._add_row_button.clicked.connect(lambda: self._append_param_row("", 1.0))
        self._remove_row_button.clicked.connect(self._remove_selected_row)
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)

        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

    # ── small widget helpers ────────────────────────────────────────────────

    def _labelled_row(self, text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(text)
        label.setMinimumWidth(88)
        row.addWidget(label)
        row.addWidget(widget, stretch=1)
        return row

    def _muted_label(self, text: str, *, bold: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        weight = "font-weight: 600;" if bold else ""
        label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; {weight}")
        return label

    def _apply_mono_font(self, editor: QPlainTextEdit) -> None:
        try:
            from asymmetry.gui.styles.fonts import mono_font

            editor.setFont(mono_font(10.0))
        except Exception:
            # Font styling is cosmetic; never let it break the dialog.
            pass

    # ── parameter table ─────────────────────────────────────────────────────

    def _append_param_row(self, name: str, value: float) -> None:
        row = self._param_table.rowCount()
        self._param_table.insertRow(row)
        self._param_table.setItem(row, 0, QTableWidgetItem(name))
        spin = QDoubleSpinBox()
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setValue(float(value))
        spin.valueChanged.connect(self._schedule_validation)
        self._param_table.setCellWidget(row, 1, spin)

    def _remove_selected_row(self) -> None:
        rows = sorted({index.row() for index in self._param_table.selectedIndexes()}, reverse=True)
        if not rows:
            # No selection: remove the last row as a sensible default.
            if self._param_table.rowCount():
                rows = [self._param_table.rowCount() - 1]
        for row in rows:
            self._param_table.removeRow(row)
        self._schedule_validation()

    def _current_param_names(self) -> list[str]:
        names = []
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 0)
            names.append(item.text().strip() if item is not None else "")
        return names

    def _draft_parameters(self) -> list[DraftParameter]:
        params: list[DraftParameter] = []
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 0)
            name = item.text().strip() if item is not None else ""
            spin = self._param_table.cellWidget(row, 1)
            value = float(spin.value()) if isinstance(spin, QDoubleSpinBox) else 1.0
            params.append(DraftParameter(name, value))
        return params

    def _on_detect_parameters(self) -> None:
        """Add parameter names found in the formula, keeping existing rows.

        Never removes or reorders rows the user already has: detection is
        additive so a start value already entered survives a re-detect.
        """
        try:
            detected = detect_parameter_names(self._formula_edit.text())
        except UserFunctionError as exc:
            self._show_status(str(exc), ok=False)
            return
        existing = set(self._current_param_names())
        for name in detected:
            if name not in existing:
                self._append_param_row(name, 1.0)
                existing.add(name)
        self._schedule_validation()

    # ── advanced toggle ─────────────────────────────────────────────────────

    def _on_advanced_toggled(self, checked: bool) -> None:
        if checked:
            # Pre-fill with the formula body (or keep a previously edited body).
            if not self._advanced_editor.toPlainText().strip():
                prefill = self._generate_body_prefill()
                self._last_generated_body = prefill
                self._advanced_editor.setPlainText(prefill)
            self._advanced_active = True
            self._set_formula_visible(False)
            self._advanced_editor.setVisible(True)
            self._detect_button.setEnabled(False)
        else:
            # Reverting to formula mode discards the body. If the body was
            # edited away from the last generated prefill, confirm the discard.
            body = self._advanced_editor.toPlainText()
            if body.strip() and body != self._last_generated_body and not self._confirm_discard():
                # Re-check the toggle without re-entering this handler.
                self._advanced_toggle.blockSignals(True)
                self._advanced_toggle.setChecked(True)
                self._advanced_toggle.blockSignals(False)
                return
            self._advanced_active = False
            self._advanced_editor.setVisible(False)
            self._set_formula_visible(True)
            self._detect_button.setEnabled(True)
        self._schedule_validation()

    def _generate_body_prefill(self) -> str:
        """The formula body used to seed the Advanced editor.

        Falls back to a neutral template when the formula cannot yet be parsed,
        so toggling on always leaves the author with editable starting text.
        """
        try:
            return generate_function_body(self._current_draft())
        except UserFunctionError:
            return "x = np.asarray(x, dtype=float)\nresult = x\nreturn result"

    def _confirm_discard(self) -> bool:
        """Confirm discarding advanced-editor edits; auto-proceed under pytest."""
        if _in_test_mode():
            return True
        answer = QMessageBox.question(
            self,
            "Discard advanced edits?",
            "Switching back to the formula discards your Python edits. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _set_formula_visible(self, visible: bool) -> None:
        self._formula_edit.setVisible(visible)
        self._formula_hint.setVisible(visible)

    # ── draft assembly + validation ─────────────────────────────────────────

    def _current_draft(self) -> UserFunctionDraft:
        return UserFunctionDraft(
            kind=self._kind,
            name=self._name_edit.text().strip(),
            description=self._description_edit.text().strip(),
            formula=self._formula_edit.text().strip(),
            parameters=self._draft_parameters(),
            domain=self._domain,
            advanced_body=(self._advanced_editor.toPlainText() if self._advanced_active else None),
        )

    def _schedule_validation(self) -> None:
        self._validation_timer.start()

    def _missing_required_fields(self, draft: UserFunctionDraft) -> list[str]:
        """Human-readable names of the required fields still empty in *draft*.

        Empty-so-far is a normal state (first open, mid-typing), not a mistake,
        so the dialog shows these as a neutral hint rather than routing them
        through ``validate_draft`` and surfacing the core's grammar error for
        an empty name.
        """
        missing: list[str] = []
        if not draft.name:
            missing.append("a name")
        if draft.advanced_body is not None:
            if not draft.advanced_body.strip():
                missing.append("a Python body")
        elif not draft.formula:
            missing.append("a formula")
        if not draft.description:
            missing.append("a description")
        return missing

    def _run_validation(self) -> None:
        """Build the draft, validate it, and update status/OK/preview.

        Runs user code (via ``validate_draft``) on the GUI thread — acceptable
        because the probe grids are tiny and this is debounced, never
        per-keystroke. Any failure is shown inline; the dialog never crashes.
        """
        draft = self._current_draft()

        # Still-empty required fields are not errors — show a neutral to-do
        # hint instead of the core's red grammar message for an empty name.
        missing = self._missing_required_fields(draft)
        if missing:
            self._set_valid(False)
            self._show_hint(f"To create a function, fill in: {', '.join(missing)}.")
            self._clear_preview()
            return

        try:
            validate_draft(draft)
        except UserFunctionError as exc:
            self._set_valid(False)
            self._show_status(str(exc), ok=False)
            self._clear_preview()
            return
        except Exception as exc:  # noqa: BLE001 — defensive: never crash the dialog.
            self._set_valid(False)
            self._show_status(f"Unexpected error: {type(exc).__name__}: {exc}", ok=False)
            self._clear_preview()
            return

        self._set_valid(True)
        target = self._directory if self._directory is not None else "your user-functions folder"
        self._show_status(
            f"Function is valid — it will be saved to {target} and available immediately.",
            ok=True,
        )
        self._update_preview(draft)

    def _set_valid(self, valid: bool) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def _show_status(self, text: str, *, ok: bool) -> None:
        colour = tokens.OK if ok else tokens.ERROR
        self._status_label.setStyleSheet(f"color: {colour};")
        self._status_label.setText(text)

    def _show_hint(self, text: str) -> None:
        """Neutral (muted) status for the incomplete-draft state — not an error."""
        self._status_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._status_label.setText(text)

    # ── preview ─────────────────────────────────────────────────────────────

    def _update_preview(self, draft: UserFunctionDraft) -> None:
        if self._axes is None or self._canvas is None:
            return
        import numpy as np

        low, high = _PREVIEW_RANGES[self._preview_kind]
        grid = np.linspace(low, high, _PREVIEW_POINTS)
        try:
            values = evaluate_draft(draft, grid)
        except UserFunctionError:
            self._clear_preview()
            return
        xlabel, ylabel = _AXIS_LABELS[self._preview_kind]
        self._axes.clear()
        self._axes.plot(grid, values, color=tokens.PLOT_FIT_PREVIEW, linewidth=1.5)
        self._axes.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.5)
        self._axes.set_xlabel(xlabel, fontsize=8)
        self._axes.set_ylabel(ylabel, fontsize=8)
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def _clear_preview(self) -> None:
        if self._axes is None or self._canvas is None:
            return
        self._axes.clear()
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()

    # ── accept ──────────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        """Re-validate and create the function; keep open on failure.

        ``create_user_function`` re-runs validation before it writes anything,
        so a name that became a collision *between* the last live validation and
        this click (or any other late failure) surfaces here with nothing
        half-registered — the core deletes the file if the load fails.
        """
        draft = self._current_draft()
        try:
            self._created = create_user_function(draft, directory=self._directory)
        except UserFunctionError as exc:
            self._created = None
            self._set_valid(False)
            self._show_status(str(exc), ok=False)
            return
        self.accept()

    # ── public accessors ────────────────────────────────────────────────────

    def created(self) -> CreatedUserFunction | None:
        """The created function after a successful accept, else ``None``."""
        return self._created

    def created_name(self) -> str | None:
        """The registered name after a successful accept, else ``None``."""
        return self._created.definition.name if self._created is not None else None
