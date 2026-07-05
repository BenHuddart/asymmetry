"""Window-agnostic decision-trail widget for the fit wizards.

The trail renders a sequence of :class:`~asymmetry.core.fitting.wizard_narrative.TrailStep`
rows — a plain-sentence headline with an expand chevron. Expanding a step reveals
its ``detail_lines`` verbatim and, when the host window has injected a bespoke
panel for that step's ``key`` (via :meth:`set_step_detail_widget`), reveals that
re-parented panel too.

This module holds no analysis logic and imports no window: prose comes entirely
from the core narrative module, and the host is responsible for supplying (and
re-parenting) any deep panel. That keeps the widget reusable by both the
single-spectrum wizard and a future multi-dataset wizard.

Two rendering modes:

* :meth:`set_steps` — the completed trail. Every step gets a chevron and can be
  expanded. This is the Result-state trail.
* :meth:`stream_placeholders` / :meth:`activate_step` / :meth:`set_status` — the
  Running-state stream. Headlines light up as stages report; unknown progress
  text only updates the status line.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.gui.styles import tokens


class _TrailRow(QWidget):
    """One expandable step: chevron + headline, with an inline detail area.

    The detail area holds the step's ``detail_lines`` as a wrapped label and,
    optionally, a host-injected panel appended beneath them. The row never
    creates that panel — it only shows/hides whatever :meth:`set_detail_widget`
    was handed.
    """

    def __init__(self, step: TrailStep, *, expandable: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = step.key
        self._detail_widget: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._header = QToolButton(self)
        self._header.setText(step.headline)
        self._header.setCheckable(expandable)
        self._header.setEnabled(expandable)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.RightArrow if expandable else Qt.ArrowType.NoArrow)
        self._header.setStyleSheet(
            "QToolButton { border: none; text-align: left; font-weight: 600; }"
        )
        self._header.setSizePolicy(
            self._header.sizePolicy().horizontalPolicy(),
            self._header.sizePolicy().verticalPolicy(),
        )
        if expandable:
            self._header.toggled.connect(self._on_toggled)
        layout.addWidget(self._header)

        self._detail_area = QWidget(self)
        self._detail_layout = QVBoxLayout(self._detail_area)
        self._detail_layout.setContentsMargins(20, 0, 0, 6)
        self._detail_layout.setSpacing(4)

        self._detail_label = QLabel("\n".join(step.detail_lines), self._detail_area)
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._detail_layout.addWidget(self._detail_label)
        self._detail_area.setVisible(False)
        layout.addWidget(self._detail_area)

    @property
    def key(self) -> str:
        return self._key

    def set_headline(self, text: str) -> None:
        self._header.setText(text)

    def set_active(self, active: bool) -> None:
        """Style the headline as the in-progress step during streaming."""
        weight = "700" if active else "500"
        colour = tokens.ACCENT if active else tokens.TEXT
        self._header.setStyleSheet(
            f"QToolButton {{ border: none; text-align: left;"
            f" font-weight: {weight}; color: {colour}; }}"
        )

    def set_detail_widget(self, widget: QWidget | None) -> None:
        """Inject (or clear) the host panel shown when this row is expanded.

        The widget is re-parented into the detail area; passing ``None`` removes
        the current one. Visibility follows the row's expanded state.
        """
        if self._detail_widget is widget:
            return
        if self._detail_widget is not None:
            self._detail_layout.removeWidget(self._detail_widget)
            self._detail_widget.setParent(None)
        self._detail_widget = widget
        if widget is not None:
            self._detail_layout.addWidget(widget)
            widget.setVisible(self._detail_area.isVisible())

    def set_expanded(self, expanded: bool) -> None:
        if self._header.isCheckable():
            self._header.setChecked(expanded)

    def is_expanded(self) -> bool:
        # Reflects the toggle's checked state, not live visibility — a collapsed
        # ancestor would make ``isVisible()`` False even for an expanded row.
        return self._header.isCheckable() and self._header.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        self._header.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._detail_area.setVisible(checked)
        if self._detail_widget is not None:
            self._detail_widget.setVisible(checked)


class DecisionTrail(QWidget):
    """A vertical list of expandable decision-trail steps.

    Window-agnostic: fed :class:`TrailStep` objects via :meth:`set_steps` (the
    finished trail) or driven step-by-step during a run via
    :meth:`stream_placeholders`, :meth:`activate_step`, and :meth:`set_status`.
    Host windows inject deep panels per step key with
    :meth:`set_step_detail_widget`; the trail never builds or imports those.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, _TrailRow] = {}
        self._pending_details: dict[str, QWidget] = {}
        self._active_key: str | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._status_label = QLabel("", self)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._status_label.setVisible(False)
        self._layout.addWidget(self._status_label)

        self._rows_container = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._layout.addWidget(self._rows_container)

    # ── Status line ────────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        """Show a free-text status line above the rows (or hide it when empty)."""
        self._status_label.setText(text or "")
        self._status_label.setVisible(bool(text))

    # ── Streaming mode ─────────────────────────────────────────────────────

    def stream_placeholders(self, steps: tuple[TrailStep, ...]) -> None:
        """Show pending, non-expandable placeholder headlines for a run.

        Headlines are shown greyed/light; :meth:`activate_step` lights one as it
        starts. Called at run start with the known-in-advance step skeleton.
        """
        self._clear_rows()
        for step in steps:
            row = _TrailRow(step, expandable=False, parent=self._rows_container)
            row.set_active(False)
            self._rows_layout.addWidget(row)
            self._rows[step.key] = row

    def activate_step(self, key: str) -> None:
        """Mark the step with ``key`` as the currently-running step.

        Earlier steps are shown as done (inactive), the named step active. No-op
        for an unknown key, so unmapped progress messages never raise.
        """
        if key not in self._rows:
            return
        self._active_key = key
        for row_key, row in self._rows.items():
            row.set_active(row_key == key)

    def active_step_key(self) -> str | None:
        """Return the key of the step currently marked active (streaming), if any."""
        return self._active_key

    # ── Finished mode ──────────────────────────────────────────────────────

    def set_steps(self, steps: tuple[TrailStep, ...]) -> None:
        """Render the finished trail: every step expandable to its detail.

        Re-applies any panels previously registered via
        :meth:`set_step_detail_widget` so a host can inject them once, before or
        after the final trail arrives.
        """
        self.set_status("")
        self._clear_rows()
        for step in steps:
            row = _TrailRow(step, expandable=True, parent=self._rows_container)
            self._rows_layout.addWidget(row)
            self._rows[step.key] = row
            if step.key in self._pending_details:
                row.set_detail_widget(self._pending_details[step.key])

    def set_step_detail_widget(self, key: str, widget: QWidget | None) -> None:
        """Register a host panel to reveal when step ``key`` is expanded.

        Stored so it survives a later :meth:`set_steps` rebuild; applied to the
        live row immediately when one exists. The panel is re-parented by the
        row on injection.
        """
        if widget is None:
            self._pending_details.pop(key, None)
        else:
            self._pending_details[key] = widget
        row = self._rows.get(key)
        if row is not None:
            row.set_detail_widget(widget)

    def set_step_expanded(self, key: str, expanded: bool) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_expanded(expanded)

    def is_step_expanded(self, key: str) -> bool:
        row = self._rows.get(key)
        return bool(row is not None and row.is_expanded())

    def step_keys(self) -> tuple[str, ...]:
        return tuple(self._rows.keys())

    # ── Internals ──────────────────────────────────────────────────────────

    def _clear_rows(self) -> None:
        # Detach any injected detail panels first so re-parenting survives the
        # row teardown (the host owns those panels; deleting a row must not
        # delete a shared panel it merely hosted).
        for row in self._rows.values():
            row.set_detail_widget(None)
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows.clear()
        self._active_key = None


class TrailSeparator(QFrame):
    """A thin horizontal rule for separating the answer card from the trail."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"color: {tokens.BORDER};")
