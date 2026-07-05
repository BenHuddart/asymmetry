"""Window-agnostic decision-trail widget for the fit wizards.

The trail renders a sequence of :class:`~asymmetry.core.fitting.wizard_narrative.TrailStep`
rows — a numbered state disc, a connector line linking it to its neighbours, and
a plain-sentence headline with an expand chevron. Expanding a step reveals its
``detail_lines`` verbatim and, when the host window has injected a bespoke panel
for that step's ``key`` (via :meth:`set_step_detail_widget`), reveals that
re-parented panel too.

This module holds no analysis logic and imports no window: prose comes entirely
from the core narrative module, and the host is responsible for supplying (and
re-parenting) any deep panel. That keeps the widget reusable by both the
single-spectrum wizard and a future multi-dataset wizard.

Two rendering modes:

* :meth:`set_steps` — the completed trail. Every step is marked done (a ✓ disc)
  and can be expanded. This is the Result-state trail.
* :meth:`stream_placeholders` / :meth:`activate_step` / :meth:`set_status` — the
  Running-state stream. Rows start pending (numbered, greyed); ``activate_step``
  marks earlier rows done, the named row active, and later rows pending. Unknown
  progress text only updates the status line.

Row states
----------

Each row's disc and headline are styled from one of three states, set by
:class:`DecisionTrail` and applied by :meth:`_TrailRow.set_state`:

* ``"pending"`` — not yet reached: dim disc with the row's 1-based number.
* ``"active"`` — the currently-running step during streaming: accent disc,
  still numbered.
* ``"done"`` — finished (either streamed-past or part of the finished trail):
  success-tinted disc showing a check mark.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.gui.styles import tokens

#: Fixed width of the marker column (disc + connector), in px.
_MARKER_COLUMN_WIDTH = 26
#: Disc diameter, in px.
_DISC_DIAMETER = 20
#: Per-row vertical padding so the connector reads as one continuous line
#: across rows even though ``rows_layout`` spacing is 0.
_ROW_VPAD = 4
#: Height of the band the disc centres in, matching the headline row (not the
#: full marker height) — so an expanded row's taller detail area doesn't pull
#: the disc down away from its headline; the connector still runs the full
#: marker height either side of the disc.
_HEADER_BAND_HEIGHT = 24

#: Per-state (background, border colour, border width, text colour) for the
#: disc, and (text colour, font weight) for the headline. Kept as one table so
#: a row's full look is one dict lookup away from drifting out of sync.
_DISC_STYLE = {
    "pending": {
        "bg": tokens.SURFACE_ALT,
        "border": tokens.BORDER,
        "border_width": 1,
        "fg": tokens.TEXT_DIM,
        "bold": False,
    },
    "active": {
        "bg": tokens.ACCENT_SOFT,
        "border": tokens.ACCENT,
        "border_width": 2,
        "fg": tokens.ACCENT,
        "bold": True,
    },
    "done": {
        "bg": tokens.SUCCESS_BG,
        "border": tokens.SUCCESS_BORDER,
        "border_width": 1,
        "fg": tokens.OK,
        "bold": False,
    },
}

_HEADLINE_STYLE = {
    "pending": {"colour": tokens.TEXT_MUTED, "weight": "500"},
    "active": {"colour": tokens.ACCENT, "weight": "700"},
    "done": {"colour": tokens.TEXT, "weight": "600"},
}


class _StepMarker(QWidget):
    """The disc + connector line for one trail row.

    Painted rather than composed from separate line widgets: a single
    ``paintEvent`` draws the connector across the row's full height first, then
    the disc on top, so the line reads as continuous across row boundaries
    regardless of the row's own padding. The first row omits the line above its
    disc, the last omits the line below.
    """

    def __init__(
        self,
        number: int,
        *,
        is_first: bool,
        is_last: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._number = number
        self._is_first = is_first
        self._is_last = is_last
        self._state = "pending"
        self._text = str(number)
        self.setFixedWidth(_MARKER_COLUMN_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_state(self, state: str) -> None:
        self._state = state
        self._text = "✓" if state == "done" else str(self._number)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(_MARKER_COLUMN_WIDTH, _DISC_DIAMETER)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        centre_x = width / 2
        disc_top = (_HEADER_BAND_HEIGHT - _DISC_DIAMETER) / 2
        disc_bottom = disc_top + _DISC_DIAMETER

        line_pen = QPen(tokens.BORDER)
        line_pen.setWidth(1)
        painter.setPen(line_pen)
        if not self._is_first:
            painter.drawLine(int(centre_x), 0, int(centre_x), int(disc_top))
        if not self._is_last:
            painter.drawLine(int(centre_x), int(disc_bottom), int(centre_x), height)

        style = _DISC_STYLE[self._state]
        disc_rect = (
            centre_x - _DISC_DIAMETER / 2,
            disc_top,
            _DISC_DIAMETER,
            _DISC_DIAMETER,
        )
        border_pen = QPen(style["border"])
        border_pen.setWidth(style["border_width"])
        painter.setPen(border_pen)
        painter.setBrush(style["bg"])
        painter.drawEllipse(*disc_rect)

        painter.setPen(style["fg"])
        font = painter.font()
        font.setBold(style["bold"])
        font.setPointSizeF(font.pointSizeF() * 0.85)
        painter.setFont(font)
        painter.drawText(
            int(disc_rect[0]),
            int(disc_rect[1]),
            int(disc_rect[2]),
            int(disc_rect[3]),
            Qt.AlignmentFlag.AlignCenter,
            self._text,
        )


class _TrailRow(QWidget):
    """One trail step: state disc + connector, headline, and inline detail area.

    The detail area holds the step's ``detail_lines`` as a wrapped label and,
    optionally, a host-injected panel appended beneath them. The row never
    creates that panel — it only shows/hides whatever :meth:`set_detail_widget`
    was handed.
    """

    def __init__(
        self,
        step: TrailStep,
        *,
        number: int,
        is_first: bool,
        is_last: bool,
        expandable: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = step.key
        self._number = number
        self._state = "pending"
        self._detail_widget: QWidget | None = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, _ROW_VPAD, 0, _ROW_VPAD)
        outer.setSpacing(0)

        self._marker = _StepMarker(number, is_first=is_first, is_last=is_last, parent=self)
        outer.addWidget(self._marker)

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 0, 0, 0)
        content_layout.setSpacing(2)
        outer.addWidget(content, 1)

        self._header = QToolButton(content)
        self._header.setText(step.headline)
        self._header.setCheckable(expandable)
        self._header.setEnabled(expandable)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.RightArrow if expandable else Qt.ArrowType.NoArrow)
        if expandable:
            self._header.toggled.connect(self._on_toggled)
        content_layout.addWidget(self._header)

        self._detail_area = QWidget(content)
        self._detail_layout = QVBoxLayout(self._detail_area)
        self._detail_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_layout.setSpacing(4)

        self._detail_panel = QFrame(self._detail_area)
        self._detail_panel.setObjectName("trailDetailPanel")
        self._detail_panel.setStyleSheet(
            f"#trailDetailPanel {{ background-color: {tokens.SURFACE_ALT};"
            f" border: 1px solid {tokens.BORDER}; border-radius: 4px; }}"
        )
        panel_layout = QVBoxLayout(self._detail_panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(4)

        self._detail_label = QLabel("\n".join(step.detail_lines), self._detail_panel)
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        panel_layout.addWidget(self._detail_label)
        self._panel_layout = panel_layout

        self._detail_layout.addWidget(self._detail_panel)
        self._detail_area.setVisible(False)
        content_layout.addWidget(self._detail_area)

        self.set_state("pending")

    @property
    def key(self) -> str:
        return self._key

    def set_headline(self, text: str) -> None:
        self._header.setText(text)

    def set_state(self, state: str) -> None:
        """Apply one of ``"pending"``, ``"active"``, ``"done"`` to disc + headline."""
        self._state = state
        self._marker.set_state(state)
        headline = _HEADLINE_STYLE[state]
        self._header.setStyleSheet(
            "QToolButton { border: none; text-align: left;"
            f" font-weight: {headline['weight']}; color: {headline['colour']}; }}"
        )

    @property
    def state(self) -> str:
        return self._state

    def set_detail_widget(self, widget: QWidget | None) -> None:
        """Inject (or clear) the host panel shown when this row is expanded.

        The widget is re-parented into the detail panel, below the muted
        detail-lines label; passing ``None`` removes the current one.
        Visibility follows the row's expanded state.
        """
        if self._detail_widget is widget:
            return
        if self._detail_widget is not None:
            self._panel_layout.removeWidget(self._detail_widget)
            self._detail_widget.setParent(None)
        self._detail_widget = widget
        if widget is not None:
            self._panel_layout.addWidget(widget)
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
    """A vertical list of numbered decision-trail steps.

    Window-agnostic: fed :class:`TrailStep` objects via :meth:`set_steps` (the
    finished trail, every row rendered done/expandable) or driven step-by-step
    during a run via :meth:`stream_placeholders`, :meth:`activate_step`, and
    :meth:`set_status`. Host windows inject deep panels per step key with
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
        self._rows_layout.setSpacing(0)
        self._layout.addWidget(self._rows_container)
        # Absorb any extra height the host layout hands the trail: with every
        # sibling at stretch 0, Qt otherwise shares surplus space between a
        # Preferred-policy trail and the host's trailing spacer, inflating the
        # gaps between rows (observed ~85px per row on the running pages).
        self._layout.addStretch(1)

    # ── Status line ────────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        """Show a free-text status line above the rows (or hide it when empty)."""
        self._status_label.setText(text or "")
        self._status_label.setVisible(bool(text))

    # ── Streaming mode ─────────────────────────────────────────────────────

    def stream_placeholders(self, steps: tuple[TrailStep, ...]) -> None:
        """Show pending, non-expandable placeholder headlines for a run.

        Rows are numbered 1..N and shown pending (dim disc); :meth:`activate_step`
        marks one active as it starts and earlier ones done. Called at run start
        with the known-in-advance step skeleton.
        """
        self._clear_rows()
        last_index = len(steps) - 1
        for index, step in enumerate(steps):
            row = _TrailRow(
                step,
                number=index + 1,
                is_first=(index == 0),
                is_last=(index == last_index),
                expandable=False,
                parent=self._rows_container,
            )
            self._rows_layout.addWidget(row)
            self._rows[step.key] = row

    def activate_step(self, key: str) -> None:
        """Mark the step with ``key`` as the currently-running step.

        Earlier steps (in insertion/build order) are shown as done, the named
        step active, and later steps stay pending. No-op for an unknown key, so
        unmapped progress messages never raise.
        """
        if key not in self._rows:
            return
        self._active_key = key
        seen_active = False
        for row_key, row in self._rows.items():
            if row_key == key:
                row.set_state("active")
                seen_active = True
            elif seen_active:
                row.set_state("pending")
            else:
                row.set_state("done")

    def active_step_key(self) -> str | None:
        """Return the key of the step currently marked active (streaming), if any."""
        return self._active_key

    # ── Finished mode ──────────────────────────────────────────────────────

    def set_steps(self, steps: tuple[TrailStep, ...]) -> None:
        """Render the finished trail: every step done, numbered, and expandable.

        Re-applies any panels previously registered via
        :meth:`set_step_detail_widget` so a host can inject them once, before or
        after the final trail arrives.
        """
        self.set_status("")
        self._clear_rows()
        last_index = len(steps) - 1
        for index, step in enumerate(steps):
            row = _TrailRow(
                step,
                number=index + 1,
                is_first=(index == 0),
                is_last=(index == last_index),
                expandable=True,
                parent=self._rows_container,
            )
            row.set_state("done")
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
