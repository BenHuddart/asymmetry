"""Data browser / logbook panel with dataset grouping support."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QPoint,
    QRect,
    QSignalBlocker,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.nexus import active_series_mean
from asymmetry.core.io.periods import period_count, period_labels
from asymmetry.core.transform.grouping import good_event_count, good_frames
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import header_font
from asymmetry.gui.utils.series_scoring import score_series_path

_GROUP_TEMP_ABS_TOL_K = 5e-3
_GROUP_TEMP_REL_TOL = 2e-3
_GROUP_FIELD_ABS_TOL_G = 1e-3
_LOG_TEMPERATURE_FOREGROUND = QColor(tokens.LOGGED_VALUE_FG)
#: Amber tint + glyph flagging a temperature whose Kelvin label is suspected to
#: be a Celsius value (EMU furnace mislabel; see the loader's
#: ``temperature_unit_suspect`` metadata flag). Distinct from the red log-source
#: tint above; the two cases are mutually exclusive (the suspicion only fires
#: when there is no logged sample thermometer to switch to).
_SUSPECT_TEMPERATURE_FOREGROUND = QColor(tokens.WARN)
_SUSPECT_TEMPERATURE_MARKER = " ⚠"
#: Celsius→Kelvin offset, used only to spell out the conversion in the warning
#: tooltip. The displayed value itself is never converted (see the loader).
_ABSOLUTE_ZERO_CELSIUS = 273.15
_GROUP_FIELD_REL_TOL = 1e-4
_GROUP_HEADER_BACKGROUND = QColor(tokens.GROUP_HEADER_BG)
_GROUP_MEMBER_BACKGROUND = QColor(tokens.GROUP_MEMBER_BG)
#: Item-data role carrying the run comment shown as the Title cell's second line.
_COMMENT_ROLE = Qt.ItemDataRole.UserRole + 1
#: Item-data role carrying a custom column's id on its editable cells, so an edit
#: routes back to the right per-run value regardless of column position.
_CUSTOM_COLUMN_ROLE = Qt.ItemDataRole.UserRole + 2
#: Item-data role carrying a multi-period cue (e.g. "2 periods · Red active")
#: shown as the Title cell's second line, so a buried 2nd period is visible.
_PERIOD_ROLE = Qt.ItemDataRole.UserRole + 3
#: Column index of the two-line Title cell.
_TITLE_COLUMN = 1
# Soft red tint used to mark runs that belong to the active fit series in
# the trend panel.  Red is the FitSeries brand colour (ACCENT_RED_SOFT).
_SERIES_HIGHLIGHT_BACKGROUND = QColor(tokens.ACCENT_RED_SOFT)


def _is_effectively_constant(values: list[float], *, abs_tol: float, rel_tol: float) -> bool:
    """Return True when finite values vary only within tolerance.

    Uses a combined absolute/relative tolerance so low-value groups (e.g.
    0.1 K) and higher-value groups (e.g. 20 K with small drift) are both
    handled robustly.
    """
    if not values:
        return False
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return False

    center = float(np.nanmedian(arr))
    span = float(np.nanmax(arr) - np.nanmin(arr))
    tolerance = max(float(abs_tol), float(rel_tol) * max(abs(center), 1.0))
    return span <= tolerance


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""

    def __init__(self, value: float | int | str):
        super().__init__(str(value))
        self.setFont(mono_font(11.0))
        try:
            self._numeric_value = float(value)
        except (ValueError, TypeError):
            self._numeric_value = 0.0

    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._numeric_value < other._numeric_value

        other_text = other.text() if isinstance(other, QTableWidgetItem) else str(other)
        try:
            other_numeric = float(other_text)
            return self._numeric_value < other_numeric
        except (ValueError, TypeError):
            return self.text() < other_text


@dataclass
class DataGroup:
    group_id: str
    name: str
    member_run_numbers: list[int]
    collapsed: bool = False


#: Discriminators for :class:`ExtraColumn.kind`.
EXTRA_COLUMN_METADATA = "metadata"
EXTRA_COLUMN_CUSTOM = "custom"

#: Dataset-metadata key under which per-run custom-column values are stored,
#: keyed by column id. Living in ``dataset.metadata`` (rather than on the column)
#: lets the values ride the existing per-dataset override persistence and reach
#: the plot-label / trend-x-axis resolvers, which already read dataset metadata.
CUSTOM_FIELDS_METADATA_KEY = "custom_fields"

#: The special "Angle" field is a singleton custom column flagged ``is_angle``:
#: it stores per-run numeric degrees (sample crystallographic axis vs the applied
#: field) and is surfaced as a first-class trend x-axis. Fixed id + label so it is
#: findable across save/reload regardless of creation order.
ANGLE_COLUMN_ID = "angle"
ANGLE_COLUMN_LABEL = "Angle (°)"


@dataclass
class ExtraColumn:
    """A user-visible column beyond the fixed Run/Title/T/B browser columns.

    Two kinds share one model so the header context menu, persistence, plot-label
    and trend-x-axis integrations can treat every extra column uniformly:

    * ``metadata`` — the value is *derived* (read-only) from each dataset's
      metadata at :attr:`source_key` (a dotted path such as
      ``nexus_fields.sample.shape`` or a synthetic ``run_info.*`` key).
      :attr:`label` defaults to a humanised name but is user-renamable, while
      :attr:`source_key` is always retained so the underlying NeXus/metadata
      field the column was built from stays recoverable.
    * ``custom`` — user-entered free-text, empty by default and editable in the
      table. The per-run values are stored in ``dataset.metadata`` (under
      :data:`CUSTOM_FIELDS_METADATA_KEY`, keyed by :attr:`id`), not on the
      column; :attr:`source_key` is ``None``.
    """

    id: str
    label: str
    kind: str = EXTRA_COLUMN_METADATA
    source_key: str | None = None
    #: ``True`` for the special "Angle" field — a custom column (so all the custom
    #: value/edit/persistence plumbing applies) that additionally carries degree
    #: semantics: numeric validation on edit and a first-class trend x-axis.
    is_angle: bool = False

    @property
    def is_custom(self) -> bool:
        return self.kind == EXTRA_COLUMN_CUSTOM

    def to_dict(self) -> dict:
        data: dict[str, object] = {"id": self.id, "label": self.label, "kind": self.kind}
        if self.source_key is not None:
            data["source_key"] = self.source_key
        if self.is_angle:
            data["is_angle"] = True
        return data

    @classmethod
    def from_dict(cls, data: object) -> ExtraColumn | None:
        """Build a column from saved state, tolerating the legacy string form.

        Older projects stored ``extra_columns`` as a bare list of metadata keys
        (strings); each promotes to a metadata column whose ``id``/``source_key``
        is that key. New projects store the full dict form.
        """
        if isinstance(data, str):
            key = data.strip()
            return (
                cls(id=key, label=key, kind=EXTRA_COLUMN_METADATA, source_key=key) if key else None
            )
        if not isinstance(data, dict):
            return None
        # Validate the discriminator at the boundary: an unknown kind (corrupt or
        # forward-version project) is treated as a (read-only) metadata column
        # rather than left to crash later as a non-custom column with no source.
        kind = str(data.get("kind", EXTRA_COLUMN_METADATA))
        if kind not in (EXTRA_COLUMN_METADATA, EXTRA_COLUMN_CUSTOM):
            kind = EXTRA_COLUMN_METADATA
        source_key = data.get("source_key")
        source_key = str(source_key) if source_key is not None else None
        col_id = str(data.get("id", "")).strip()
        if not col_id and kind == EXTRA_COLUMN_METADATA:
            col_id = source_key or ""
        if not col_id:
            return None
        # A metadata column must always carry a source key for value resolution;
        # fall back to its id if a project dropped/omitted source_key.
        if kind == EXTRA_COLUMN_METADATA and source_key is None:
            source_key = col_id
        label = str(data.get("label", col_id))
        # ``is_angle`` is meaningful only for custom columns (degree semantics ride
        # the custom value plumbing); ignore a stray flag on a metadata column.
        is_angle = kind == EXTRA_COLUMN_CUSTOM and bool(data.get("is_angle", False))
        return cls(id=col_id, label=label, kind=kind, source_key=source_key, is_angle=is_angle)


class FilterDialog(QDialog):
    """Excel-style filter dialog with checkboxes for unique values."""

    def __init__(
        self,
        column_name: str,
        unique_values: list[str],
        current_selection: set[str] | None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Filter - {column_name}")
        self.setMinimumWidth(300)
        self.setMinimumHeight(400)

        self._checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)

        self._all_checkbox = QCheckBox("(Select All)")
        self._all_checkbox.setChecked(current_selection is None)
        self._all_checkbox.stateChanged.connect(self._on_all_changed)
        layout.addWidget(self._all_checkbox)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        for value in unique_values:
            checkbox = QCheckBox(value)
            checkbox.setChecked(current_selection is None or value in current_selection)
            checkbox.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        clear_btn = QPushButton("Clear Filter")
        clear_btn.clicked.connect(self._clear_filter)
        button_layout.addWidget(clear_btn)

        layout.addLayout(button_layout)

    def _on_all_changed(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _on_checkbox_changed(self) -> None:
        all_checked = all(cb.isChecked() for cb in self._checkboxes)
        none_checked = not any(cb.isChecked() for cb in self._checkboxes)

        self._all_checkbox.blockSignals(True)
        if all_checked:
            self._all_checkbox.setCheckState(Qt.CheckState.Checked)
        elif none_checked:
            self._all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        else:
            self._all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self._all_checkbox.blockSignals(False)

    def _clear_filter(self) -> None:
        self._all_checkbox.setChecked(True)
        self.done(QDialog.DialogCode.Accepted)

    def get_selected_values(self) -> set[str] | None:
        if all(cb.isChecked() for cb in self._checkboxes):
            return None
        return {checkbox.text() for checkbox in self._checkboxes if checkbox.isChecked()}


class _RowHighlightDelegate(QStyledItemDelegate):
    """Full row-highlight painter implementing the six-state background ladder.

    State                          Background            Left bar (col 0 only)
    ─────────────────────────────  ────────────────────  ──────────────────────
    Group header normal            GROUP_HEADER_BG       –
    Group header selected          GROUP_HEADER_SEL_BG   –
    Group header focused           GROUP_HEADER_FOCUS_BG –  (white text)
    Member / run focused           ACCENT_SOFT2          3 px solid ACCENT
    Member / run selected          ACCENT_SOFT           2 px ACCENT @ 40 % alpha
    Member / run unselected        item bg role          –

    All colours come from ``styles/tokens.py`` (see the class attributes below);
    the names above are the token roles, not literals.

    The delegate strips State_Selected and State_HasFocus from the option copy
    before calling super().paint() so that QSS selection-background-color and
    ::item:focus rules never override the custom backgrounds above.

    The Title column renders two lines when the run carries a comment: the
    title on top, the comment in smaller muted text underneath (replacing the
    old Comment column, which forced horizontal scrolling).
    """

    _SENTINEL = "group:"
    _HEADER_SEL_BG = QColor(tokens.GROUP_HEADER_SEL_BG)
    _HEADER_FOC_BG = QColor(tokens.GROUP_HEADER_FOCUS_BG)
    _MEMBER_FOC_BG = QColor(tokens.ACCENT_SOFT2)
    _MEMBER_SEL_BG = QColor(tokens.ACCENT_SOFT)
    _ACCENT = QColor(tokens.ACCENT)
    _ACCENT_SOFT = QColor(tokens.ACCENT)
    _ACCENT_SOFT.setAlpha(102)  # 40 % accent — selected-member left bar
    _WHITE = QColor(tokens.WHITE)
    _COMMENT_COLOR = QColor(tokens.TEXT_MUTED)
    _CLEAR_FLAGS = QStyle.StateFlag.State_Selected | QStyle.StateFlag.State_HasFocus
    #: Vertical padding around the two text lines of a title+comment cell.
    _TWO_LINE_PAD = 3

    @staticmethod
    def _cell_comment(index) -> str:
        """Return the comment carried by a Title cell ('' elsewhere)."""
        if index.column() != _TITLE_COLUMN:
            return ""
        comment = index.data(_COMMENT_ROLE)
        return str(comment).strip() if isinstance(comment, str) else ""

    @staticmethod
    def _cell_period_note(index) -> str:
        """Return the multi-period cue carried by a Title cell ('' elsewhere)."""
        if index.column() != _TITLE_COLUMN:
            return ""
        note = index.data(_PERIOD_ROLE)
        return str(note).strip() if isinstance(note, str) else ""

    @classmethod
    def _cell_second_line(cls, index) -> str:
        """Return the Title cell's muted second line: comment plus period cue."""
        parts = [part for part in (cls._cell_comment(index), cls._cell_period_note(index)) if part]
        return " · ".join(parts)

    @staticmethod
    def _comment_font(base: QFont) -> QFont:
        """Return the smaller font used for the comment line."""
        font = QFont(base)
        size = font.pointSizeF()
        if size > 0:
            font.setPointSizeF(max(7.0, size * 0.85))
        return font

    def _line_metrics(self, base: QFont) -> tuple[QFont, QFontMetrics, QFontMetrics]:
        """Return (comment font, title metrics, comment metrics), cached per font.

        paint() runs per cell per frame; constructing QFontMetrics there makes
        scrolling measurably jankier, and the base font only changes on a UI
        scale change.
        """
        cache = getattr(self, "_metrics_cache", None)
        if cache is None:
            cache = self._metrics_cache = {}
        key = base.key()
        entry = cache.get(key)
        if entry is None:
            comment_font = self._comment_font(base)
            entry = (comment_font, QFontMetrics(base), QFontMetrics(comment_font))
            cache[key] = entry
        return entry

    def _draw_two_line_cell(self, painter, opt, index) -> None:
        """Draw title over muted smaller-text comment inside ``opt.rect``.

        ``opt`` must already be initStyleOption()-initialised; the background
        is the caller's responsibility.
        """
        comment = self._cell_second_line(index)
        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        rect = opt.rect.adjusted(4, self._TWO_LINE_PAD, -4, -self._TWO_LINE_PAD)

        comment_font, title_fm, comment_fm = self._line_metrics(opt.font)

        title_color = opt.palette.color(QPalette.ColorRole.Text)
        foreground = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(foreground, QBrush):
            title_color = foreground.color()
        elif isinstance(foreground, QColor):
            title_color = foreground

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(title_color)
        painter.drawText(
            QRect(rect.left(), rect.top(), rect.width(), title_fm.height()),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title_fm.elidedText(title, Qt.TextElideMode.ElideRight, rect.width()),
        )
        painter.setFont(comment_font)
        painter.setPen(self._COMMENT_COLOR)
        painter.drawText(
            QRect(
                rect.left(),
                rect.top() + title_fm.height(),
                rect.width(),
                comment_fm.height(),
            ),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            comment_fm.elidedText(comment, Qt.TextElideMode.ElideRight, rect.width()),
        )
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 — Qt override
        hint = super().sizeHint(option, index)
        if self._cell_second_line(index):
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            _, title_fm, comment_fm = self._line_metrics(opt.font)
            hint.setHeight(title_fm.height() + comment_fm.height() + 2 * self._TWO_LINE_PAD)
        return hint

    def paint(self, painter, option, index):
        table = self.parent()
        col0 = table.item(index.row(), 0)
        is_header = (
            col0 is not None
            and isinstance(col0.data(Qt.ItemDataRole.UserRole), str)
            and col0.data(Qt.ItemDataRole.UserRole).startswith(self._SENTINEL)
        )
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        has_comment = bool(self._cell_second_line(index))

        if is_selected:
            is_focused = index.row() == table.currentRow()
            bg = (
                (self._HEADER_FOC_BG if is_focused else self._HEADER_SEL_BG)
                if is_header
                else (self._MEMBER_FOC_BG if is_focused else self._MEMBER_SEL_BG)
            )

            # initStyleOption() re-reads Qt.BackgroundRole from the model and overwrites
            # backgroundBrush — so we call it ourselves first, then clear the brush before
            # passing the option directly to drawControl (bypassing the second initStyleOption
            # call that super().paint() would trigger).
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.state = opt.state & ~self._CLEAR_FLAGS
            opt.backgroundBrush = QBrush()

            painter.fillRect(option.rect, bg)

            if is_header and is_focused:
                pal = QPalette(opt.palette)
                pal.setColor(QPalette.ColorRole.Text, self._WHITE)
                opt.palette = pal

            if has_comment:
                self._draw_two_line_cell(painter, opt, index)
            else:
                table.style().drawControl(
                    QStyle.ControlElement.CE_ItemViewItem, opt, painter, table
                )

            # Left-edge bar on column 0 for member / run rows only
            if not is_header and index.column() == 0:
                bar_color = self._ACCENT if is_focused else self._ACCENT_SOFT
                bar_width = 3 if is_focused else 2
                painter.fillRect(
                    QRect(option.rect.left(), option.rect.top(), bar_width, option.rect.height()),
                    bar_color,
                )
        elif has_comment:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            if opt.backgroundBrush.style() != Qt.BrushStyle.NoBrush:
                painter.fillRect(option.rect, opt.backgroundBrush)
            self._draw_two_line_cell(painter, opt, index)
        else:
            super().paint(painter, option, index)


class DataBrowserPanel(QWidget):
    """Logbook-style run table with grouping, sorting, filtering and co-add."""

    dataset_selected = Signal(int)
    selection_changed = Signal()
    group_selected = Signal(str)
    get_info_requested = Signal(int)
    grouping_requested = Signal(int)
    # Emitted when the set or labels of extra columns change (add/remove/rename),
    # so the host can refresh the custom-column options offered as the plot label
    # and the parameter-trend x-axis.
    extra_columns_changed = Signal()
    # Re-fit a co-added selection: the host combines the runs, fits with the
    # active single-fit model, and records a computed trend row (combined_from).
    refit_coadded_requested = Signal(object)  # list[int] source run numbers

    # The comment rides as the Title cell's second line (see
    # _RowHighlightDelegate) instead of its own column, so long comments never
    # force horizontal scrolling.
    # Plain upright units per the design handoff's browser header (the RTF
    # export re-italicises the physical symbols; see _rtf_header_cell).
    _COLUMNS = ["Run", "Title", "T (K)", "B (G)"]
    _RUN_INFO_FIELD_LABELS = {
        "instrument": "Instrument",
        "run_label": "Run",
        "title": "Title",
        "comment": "Comment",
        "started": "Start",
        "stopped": "End",
        "temperature": "Temperature (K)",
        "field": "Magnetic Field (G)",
        "field_direction": "Field Direction",
        "detector_orientation": "Detector Orientation",
        "period_count": "Periods",
        "run_info.points": "Points",
        "run_info.histograms": "Histograms",
        "run_info.bins": "Bins",
        "run_info.bin_width_us": "Bin Width (us)",
        "run_info.counts_mev": "Counts (MEv)",
        "run_info.good_events_mev": "Good Events (MEv)",
        "run_info.events_per_frame": "Events/frame",
        "run_info.counts_per_detector": "Counts per Detector",
        "nexus_fields.sample.shape": "Orientation",
    }
    _BASE_COLUMN_OVERRIDE_KEYS = {"temperature", "field"}
    _GROUP_ROLE = Qt.ItemDataRole.UserRole
    _GROUP_SENTINEL_PREFIX = "group:"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: dict[int, MuonDataset] = {}
        self._combined_datasets: dict[int, list[int]] = {}
        self._combined_source_datasets: dict[int, list[MuonDataset]] = {}
        # Combine sign per combined id: +1 co-add (default), -1 any subtraction.
        # Absent ⇒ co-add, so existing projects are unaffected.
        self._combined_signs: dict[int, int] = {}
        # Combine method per combined id, only set for the subtractions that need
        # to be distinguished from the default reference path on rebuild/persist:
        # "subtract_signed" ⇒ symmetric N-run signed co-subtract. Absent ⇒ the
        # default for its sign (co-add or reference subtraction).
        self._combined_methods: dict[int, str] = {}
        self._next_combined_id = -1

        self._groups: dict[str, DataGroup] = {}
        self._run_to_group: dict[int, str] = {}
        self._display_order: list[int | str] = []

        self._column_filters: dict[int, set[str]] = {}
        self._extra_columns: list[ExtraColumn] = []
        self._use_temperature_from_log = False
        self._temperature_from_log_overrides: dict[int, bool] = {}
        self._use_field_from_log = False
        self._field_from_log_overrides: dict[int, bool] = {}
        self._current_sort_column: int = -1
        self._current_sort_order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
        self._selection_anchor_row: int | None = None
        self._updating_table = False
        #: True while OUR code resizes columns, so the user-takeover latch
        #: below ignores programmatic section resizes.
        self._auto_sizing_columns = False
        #: Set the first time the user drags a column edge; from then on the
        #: auto-fit stands down for the session and never fights their layout.
        self._user_sized_columns = False
        #: Run numbers to tint as "series members" (set by the trend panel).
        self._highlighted_runs: set[int] = set()
        #: Run numbers handed out for synthetic/degraded runs not yet added.
        self._reserved_run_numbers: set[int] = set()
        #: Depth of nested batch_updates() blocks. While > 0, table rebuilds,
        #: sorts and column auto-fits are deferred to one flush at exit —
        #: without this, adding N datasets rebuilds the table N times (O(n²)).
        self._batch_depth = 0
        self._batch_rebuild_pending = False
        self._batch_sort_pending = False
        self._batch_resize_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._refresh_column_headers()
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.SelectedClicked
        )

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 62)  # run numbers are at most six digits
        header.resizeSection(1, 200)
        header.resizeSection(2, 52)
        header.resizeSection(3, 52)
        # The user-takeover latch: any section resize not made by our own
        # auto-fit (or a table rebuild) counts as the user taking control.
        header.sectionResized.connect(self._on_section_resized)
        self._table.setSortingEnabled(False)
        # The indicator appears once the user actually sorts (see _sort_table);
        # the design-handoff header row has no arrow at rest.
        self._table.horizontalHeader().setSortIndicatorShown(False)
        self._table.horizontalHeader().setSectionsClickable(False)
        self._table.horizontalHeader().viewport().installEventFilter(self)
        self._table.viewport().installEventFilter(self)
        self._table.installEventFilter(self)

        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_table_context_menu)
        self._table.viewport().customContextMenuRequested.connect(self._show_table_context_menu)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        self._table.verticalHeader().hide()

        self._table.setFont(mono_font(11.0))
        self._table.horizontalHeader().setFont(header_font())

        self._row_delegate = _RowHighlightDelegate(self._table)
        self._table.setItemDelegate(self._row_delegate)
        # currentItemChanged doesn't automatically repaint both old and new rows
        # in all Qt versions; an explicit viewport update is cheap and safe.
        self._table.currentItemChanged.connect(lambda *_: self._table.viewport().update())

        # Table sits beside a trailing "add custom field" rail: the "+" perches on
        # a header-toned strip (so it reads as a final header slot) while the rail
        # body below drops to the window tone, giving an "overhanging tab" that
        # recedes into the main window. It is a sibling widget rather than a real
        # table column because Qt paints gridlines over delegate fills, which would
        # otherwise slice the recede into visible cells.
        table_row = QWidget()
        table_row_layout = QHBoxLayout(table_row)
        table_row_layout.setContentsMargins(0, 0, 0, 0)
        table_row_layout.setSpacing(0)
        table_row_layout.addWidget(self._table, 1)
        table_row_layout.addWidget(self._build_add_field_rail(), 0)
        layout.addWidget(table_row)

        # Footer band: the selection hint. The band lives on the container so the
        # hint sits on one surfaceAlt strip with a single top border.
        _add_key = "⌘" if sys.platform == "darwin" else "Ctrl"
        self._footer_hint = QLabel(f"{_add_key}-click adds · shift-click ranges")
        self._footer_hint.setWordWrap(True)
        self._footer_hint.setStyleSheet(
            f"QLabel {{ background: transparent; color: {tokens.TEXT_MUTED}; font-size: 10px; }}"
        )

        footer = QWidget()
        footer.setObjectName("dataBrowserFooter")
        footer.setStyleSheet(
            f"#dataBrowserFooter {{ background-color: {tokens.SURFACE_ALT};"
            f" border-top: 1px solid {tokens.BORDER}; }}"
        )
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 4)
        footer_layout.setSpacing(8)
        footer_layout.addWidget(self._footer_hint, 1)
        layout.addWidget(footer)
        self.setMinimumWidth(250)

        # The "+" strip must line up with the table header; the header's final
        # height is only known after the first layout pass.
        QTimer.singleShot(0, self, self._sync_rail_header_height)

    # ------------------------------------------------------------------
    # Add-custom-field rail
    # ------------------------------------------------------------------

    _RAIL_WIDTH = 28

    def _build_add_field_rail(self) -> QWidget:
        """Build the trailing rail carrying the "add custom field" affordance.

        A header-toned "+" button caps a window-toned filler so the button reads
        as a final header slot while the rail recedes into the main window.
        """
        rail = QWidget()
        rail.setObjectName("addFieldRail")
        rail.setFixedWidth(self._RAIL_WIDTH)
        rail.setStyleSheet(f"#addFieldRail {{ background-color: {tokens.BG}; }}")
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(0)

        self._add_field_btn = QToolButton()
        self._add_field_btn.setText("+")
        self._add_field_btn.setToolTip("Add a custom field")
        self._add_field_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Tab-reachable (so keyboard-only users can still add a field — it is the
        # sole affordance) but not click-focusable, so a mouse click doesn't leave
        # a focus ring lingering on the header strip.
        self._add_field_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._add_field_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Top + bottom borders continue the table's top frame line and the
        # header's bottom line, so the "+" strip joins the header seamlessly
        # (its height is sized to frame + header in _sync_rail_header_height).
        self._add_field_btn.setStyleSheet(
            "QToolButton {"
            f" background-color: {tokens.SURFACE_ALT};"
            f" color: {tokens.ACCENT};"
            " border: none;"
            f" border-top: 1px solid {tokens.BORDER};"
            f" border-bottom: 1px solid {tokens.BORDER};"
            " border-radius: 0;"
            " font-size: 15px; font-weight: 600; }"
            f" QToolButton:hover {{ background-color: {tokens.SURFACE_HI}; }}"
        )
        self._add_field_btn.setFixedWidth(self._RAIL_WIDTH)
        self._add_field_btn.clicked.connect(self._show_add_field_menu)

        filler = QWidget()
        filler.setObjectName("addFieldFiller")
        filler.setStyleSheet(f"#addFieldFiller {{ background-color: {tokens.BG}; }}")

        rail_layout.addWidget(self._add_field_btn, 0)
        rail_layout.addWidget(filler, 1)
        return rail

    def _sync_rail_header_height(self) -> None:
        """Match the "+" strip to the header rect (incl. the table's top frame).

        The header is inset by the table's 1px frame, so the strip spans that
        frame offset *plus* the header height; with its own top and bottom
        borders the strip's lines then land exactly on the table's top frame
        line and the header's bottom line.
        """
        if not hasattr(self, "_add_field_btn"):
            return
        header = self._table.horizontalHeader()
        height = header.height()
        if height > 0:
            frame_offset = header.geometry().top()
            self._add_field_btn.setFixedHeight(frame_offset + height)

    def _build_add_field_menu(self) -> QMenu:
        """Build the rail "+" menu: a free-text custom column or the Angle field.

        The Angle entry is disabled once the singleton Angle field exists.
        """
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        menu.addAction("Custom column…", self._prompt_add_custom_column)
        angle_action = menu.addAction(ANGLE_COLUMN_LABEL, self.add_angle_column)
        if self.has_angle_column():
            angle_action.setEnabled(False)
            angle_action.setToolTip("An Angle field already exists")
        return menu

    def _show_add_field_menu(self) -> None:
        """Pop the add-field menu just below the "+" strip."""
        menu = self._build_add_field_menu()
        below = self._add_field_btn.mapToGlobal(self._add_field_btn.rect().bottomLeft())
        menu.exec(below)

    # ------------------------------------------------------------------
    # Batched updates
    # ------------------------------------------------------------------

    @contextmanager
    def batch_updates(self):
        """Defer table rebuilds while adding or removing many datasets.

        Inside the block, ``_rebuild_table``, ``_sort_table`` and
        ``_resize_columns_to_content`` record that they were requested instead
        of running; the outermost exit replays each at most once. Callers must
        not read table *rows* inside the block (model state — ``_datasets``,
        ``_display_order`` — is always current). Nesting is safe.
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._flush_batch_updates()

    def _flush_batch_updates(self) -> None:
        sort_pending = self._batch_sort_pending
        rebuild_pending = self._batch_rebuild_pending
        resize_pending = self._batch_resize_pending
        self._batch_sort_pending = False
        self._batch_rebuild_pending = False
        self._batch_resize_pending = False
        if sort_pending:
            self._sort_table(rebuild=False)
        if rebuild_pending or sort_pending:
            self._rebuild_table()
        if resize_pending:
            self._resize_columns_to_content()

    # ------------------------------------------------------------------
    # Dataset and grouping CRUD
    # ------------------------------------------------------------------

    def all_datasets(self) -> list[MuonDataset]:
        """Return the browser's datasets in insertion order."""
        return list(self._datasets.values())

    def next_derived_run_number(self) -> int:
        """Reserve a run number for a synthetic or degraded run.

        Numbers start at 90001 and skip anything already loaded or reserved
        this session. Real ISIS run numbers can exceed this (modern runs are
        six digits), so a file loaded *later* could carry the same number —
        ``add_dataset`` keys entries by run number and would replace the
        derived run. Acceptable for session-scoped derived data; revisit if
        derived runs gain a persistent identity.
        """
        number = 90001
        existing = set(self._datasets) | self._reserved_run_numbers
        while number in existing:
            number += 1
        self._reserved_run_numbers.add(number)
        return number

    def release_derived_run_number(self, number: int) -> None:
        """Return a reserved derived run number that was never used.

        Lets a dialog hand back the number it reserved when generation fails,
        so a failed Generate does not leave a permanent gap in the SIM series.
        A number already claimed by a loaded dataset is left untouched.
        """
        try:
            value = int(number)
        except (TypeError, ValueError):
            return
        if value not in self._datasets:
            self._reserved_run_numbers.discard(value)

    def add_dataset(self, dataset: MuonDataset) -> None:
        rn = int(dataset.run_number)
        self._datasets[rn] = dataset
        if rn not in self._display_order and rn not in self._run_to_group:
            self._display_order.append(rn)
        if self._current_sort_column >= 0 and not self._groups:
            self._sort_table(rebuild=False)
        self._rebuild_table()
        self._resize_columns_to_content()

    def create_data_group(
        self,
        run_numbers: list[int],
        name: str | None = None,
        group_id: str | None = None,
        collapsed: bool = False,
    ) -> str | None:
        valid_runs = [rn for rn in run_numbers if rn in self._datasets]
        if len(valid_runs) < 2:
            return None

        gid = group_id or str(uuid.uuid4())
        if gid in self._groups:
            return None

        for rn in valid_runs:
            old_gid = self._run_to_group.get(rn)
            if old_gid is not None:
                self._remove_run_from_group(rn, old_gid)

        if not name:
            name = self._default_group_name(valid_runs)

        first_index = min(self._display_index_for_run(rn) for rn in valid_runs)
        for rn in valid_runs:
            if rn in self._display_order:
                self._display_order.remove(rn)

        self._display_order.insert(first_index, gid)
        self._groups[gid] = DataGroup(
            group_id=gid,
            name=name,
            member_run_numbers=list(valid_runs),
            collapsed=collapsed,
        )
        for rn in valid_runs:
            self._run_to_group[rn] = gid

        self._move_groups_to_top()

        self._rebuild_table()
        return gid

    def ungroup(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return

        insert_index = (
            self._display_order.index(group_id)
            if group_id in self._display_order
            else len(self._display_order)
        )
        if group_id in self._display_order:
            self._display_order.remove(group_id)

        for offset, rn in enumerate(group.member_run_numbers):
            self._run_to_group.pop(rn, None)
            if rn in self._datasets:
                self._display_order.insert(insert_index + offset, rn)

        self._groups.pop(group_id, None)
        self._move_groups_to_top()
        self._rebuild_table()

    def _move_groups_to_top(self) -> None:
        """Keep all group headers above non-grouped rows in display order."""
        groups = [
            entry
            for entry in self._display_order
            if isinstance(entry, str) and entry in self._groups
        ]
        runs = [entry for entry in self._display_order if isinstance(entry, int)]
        self._display_order = groups + runs

    def _remove_run_from_group(self, run_number: int, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        group.member_run_numbers = [rn for rn in group.member_run_numbers if rn != run_number]
        self._run_to_group.pop(run_number, None)
        if len(group.member_run_numbers) == 0:
            self.ungroup(group_id)

    def add_runs_to_group(self, run_numbers: list[int], group_id: str) -> bool:
        """Add existing dataset run rows into an existing group.

        Parameters
        ----------
        run_numbers
            Dataset run numbers to move.
        group_id
            Target group identifier.

        Returns
        -------
        bool
            ``True`` if at least one run was moved.
        """
        group = self._groups.get(group_id)
        if group is None:
            return False

        moved_any = False
        for rn in run_numbers:
            if rn not in self._datasets:
                continue
            if self._run_to_group.get(rn) == group_id:
                continue

            old_gid = self._run_to_group.get(rn)
            if old_gid is not None:
                self._remove_run_from_group(rn, old_gid)

            if rn in self._display_order:
                self._display_order.remove(rn)
            self._run_to_group[rn] = group_id
            group.member_run_numbers.append(rn)
            moved_any = True

        if moved_any:
            self._move_groups_to_top()
            self._rebuild_table()
        return moved_any

    def remove_runs_from_group(self, run_numbers: list[int]) -> bool:
        """Remove selected runs from their current groups and move to top-level list."""
        moved_any = False
        insert_at = len(self._display_order)
        for rn in run_numbers:
            gid = self._run_to_group.get(rn)
            if gid is None or rn not in self._datasets:
                continue
            group_index = (
                self._display_order.index(gid)
                if gid in self._display_order
                else len(self._display_order)
            )
            insert_at = min(insert_at, group_index + 1)
            self._remove_run_from_group(rn, gid)
            if rn not in self._display_order:
                self._display_order.insert(insert_at, rn)
                insert_at += 1
            moved_any = True

        if moved_any:
            self._move_groups_to_top()
            self._rebuild_table()
        return moved_any

    def _default_group_name(self, run_numbers: list[int]) -> str:
        datasets = [self._datasets[rn] for rn in run_numbers if rn in self._datasets]
        if not datasets:
            return f"Group {len(self._groups) + 1}"

        temps = [float(ds.metadata.get("temperature", np.nan)) for ds in datasets]
        fields = [float(ds.metadata.get("field", np.nan)) for ds in datasets]
        if _is_effectively_constant(
            temps,
            abs_tol=_GROUP_TEMP_ABS_TOL_K,
            rel_tol=_GROUP_TEMP_REL_TOL,
        ):
            return f"T = {float(np.nanmedian(temps)):.6g} K"
        if _is_effectively_constant(
            fields,
            abs_tol=_GROUP_FIELD_ABS_TOL_G,
            rel_tol=_GROUP_FIELD_REL_TOL,
        ):
            return f"B = {float(np.nanmedian(fields)):.6g} G"
        return f"Group {len(self._groups) + 1}"

    def _display_index_for_run(self, run_number: int) -> int:
        gid = self._run_to_group.get(run_number)
        if gid is not None and gid in self._display_order:
            return self._display_order.index(gid)
        if run_number in self._display_order:
            return self._display_order.index(run_number)
        return len(self._display_order)

    # ------------------------------------------------------------------
    # Table building
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        if self._batch_depth:
            self._batch_rebuild_pending = True
            return

        selected_keys = self._selected_keys()

        self._updating_table = True
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(0)

            for entry in self._display_order:
                if isinstance(entry, str):
                    self._add_group_header_row(entry)
                    group = self._groups.get(entry)
                    if group is None:
                        continue
                    if not group.collapsed:
                        for rn in group.member_run_numbers:
                            if rn in self._datasets:
                                self._add_dataset_row(self._datasets[rn], indent=True)
                else:
                    dataset = self._datasets.get(entry)
                    if dataset is not None:
                        self._add_dataset_row(dataset, indent=False)

            self._updating_table = False
            self._apply_row_visibility()
            self._restore_selection_by_keys(selected_keys)
        finally:
            self._updating_table = False
            self._table.setUpdatesEnabled(True)

    def _add_group_header_row(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return

        row = self._table.rowCount()
        self._table.insertRow(row)

        prefix = "▸" if group.collapsed else "▾"
        run_item = QTableWidgetItem(f"{prefix} {group.name}")
        run_item.setData(self._GROUP_ROLE, f"{self._GROUP_SENTINEL_PREFIX}{group.group_id}")
        run_item.setFlags(
            (run_item.flags() & ~Qt.ItemFlag.ItemIsEditable) | Qt.ItemFlag.ItemIsSelectable
        )
        font = QFont(self._table.font())
        font.setBold(True)
        run_item.setFont(font)
        run_item.setBackground(_GROUP_HEADER_BACKGROUND)
        self._table.setItem(row, 0, run_item)

        # Cols 1–2 (Title, T): blank
        for col in range(1, min(3, self._table.columnCount())):
            blank = QTableWidgetItem("")
            blank.setFlags(blank.flags() & ~Qt.ItemFlag.ItemIsEditable)
            blank.setBackground(_GROUP_HEADER_BACKGROUND)
            self._table.setItem(row, col, blank)

        # Last base column (B): right-aligned member count in muted mono
        if self._table.columnCount() > 3:
            n = len(group.member_run_numbers)
            count_text = f"{n} run" if n == 1 else f"{n} runs"
            count_item = QTableWidgetItem(count_text)
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            count_item.setFont(mono_font(10.0))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            count_item.setForeground(QColor(tokens.TEXT_MUTED))
            count_item.setBackground(_GROUP_HEADER_BACKGROUND)
            self._table.setItem(row, 3, count_item)

        # Extra columns beyond the fixed four: blank
        for col in range(len(self._COLUMNS), self._table.columnCount()):
            blank = QTableWidgetItem("")
            blank.setFlags(blank.flags() & ~Qt.ItemFlag.ItemIsEditable)
            blank.setBackground(_GROUP_HEADER_BACKGROUND)
            self._table.setItem(row, col, blank)

    @staticmethod
    def _derived_run_tooltip(meta: dict) -> str:
        """Provenance tooltip for synthetic/degraded entries, '' otherwise.

        Reads the in-session metadata written by ``core.simulate`` and, for
        runs reloaded from a saved synthetic NeXus file, the ``simulation``
        group surfaced by the loader under ``metadata["nexus_fields"]`` — so
        a saved-and-reopened synthetic run stays badged.
        """

        def text(value: object) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        sim = meta.get("simulation")
        if meta.get("synthetic") and isinstance(sim, dict):
            return (
                "Synthetic run — model: "
                f"{sim.get('model', '?')}; seed {sim.get('seed', '?')}; "
                f"template run {sim.get('template_run_number', '?')}"
            )
        degraded = meta.get("degraded")
        if isinstance(degraded, dict):
            return (
                f"Degraded ×{degraded.get('factor', '?')} from run "
                f"{degraded.get('source_run_label', degraded.get('source_run_number', '?'))} "
                f"(seed {degraded.get('seed', '?')})"
            )

        nexus_fields = meta.get("nexus_fields")
        file_sim = nexus_fields.get("simulation") if isinstance(nexus_fields, dict) else None
        if isinstance(file_sim, dict):
            factor = file_sim.get("degrade_factor")
            if factor is not None:
                return (
                    f"Degraded ×{text(factor)} from run "
                    f"{text(file_sim.get('degrade_source_run_label', file_sim.get('degrade_source_run_number', '?')))} "
                    f"(seed {text(file_sim.get('degrade_seed', '?'))}; reloaded from file)"
                )
            try:
                is_synthetic = int(np.asarray(file_sim.get("synthetic", 0)).flat[0])
            except (TypeError, ValueError):
                is_synthetic = 0
            if is_synthetic:
                return (
                    "Synthetic run — model: "
                    f"{text(file_sim.get('model', '?'))}; "
                    f"seed {text(file_sim.get('seed', '?'))}; "
                    f"template run {text(file_sim.get('template_run_number', '?'))} "
                    "(reloaded from file)"
                )
        return ""

    @staticmethod
    def _period_state(dataset: MuonDataset) -> tuple[int, str] | None:
        """Return ``(count, active_label)`` for a multi-period run, else ``None``.

        The active label is the period the default reduction uses — the
        Grouping dialog's RG-mode selection when one is recorded, otherwise the
        first period (Red for the two-period red/green case), which is the
        default a defaults-following user silently fits.
        """
        try:
            count = period_count(dataset)
        except (TypeError, ValueError, AttributeError):
            return None
        if count <= 1:
            return None
        labels = period_labels(dataset)
        active = labels[0] if labels else "1"
        grouping = getattr(dataset.run, "grouping", None) if dataset.run is not None else None
        if isinstance(grouping, dict):
            mode = grouping.get("period_mode")
            if mode:
                active = str(mode)
        return count, active.capitalize()

    def _period_note(self, dataset: MuonDataset) -> str | None:
        """Return the compact multi-period cue for the Title cell, else ``None``."""
        state = self._period_state(dataset)
        if state is None:
            return None
        count, active = state
        return f"{count} periods · {active} active"

    def multi_period_run_numbers(self) -> set[int]:
        """Return run numbers of loaded datasets that carry more than one period."""
        return {
            int(rn)
            for rn, dataset in self._datasets.items()
            if self._period_state(dataset) is not None
        }

    def _add_dataset_row(self, dataset: MuonDataset, *, indent: bool) -> None:
        rn = int(dataset.run_number)
        meta = dataset.metadata
        run_display = str(dataset.run_label)
        if rn in self._combined_datasets:
            run_display = self._combined_run_display(rn)
        if indent:
            run_display = f"    {run_display}"

        row = self._table.rowCount()
        self._table.insertRow(row)

        if rn in self._combined_datasets or dataset.run_label != str(rn):
            run_item = QTableWidgetItem(run_display)
        else:
            run_item = NumericTableWidgetItem(run_display)
        run_item.setData(self._GROUP_ROLE, rn)
        run_item.setFlags(run_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, run_item)

        title = str(meta.get("title", ""))
        comment = str(meta.get("comment", ""))
        title_item = QTableWidgetItem(title)
        title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        # The comment and a multi-period cue render as the cell's second line
        # (see _RowHighlightDelegate); a buried 2nd period is otherwise invisible.
        title_item.setData(_COMMENT_ROLE, comment)
        period_note = self._period_note(dataset)
        period_tip = (
            f"{period_note}. Switch periods in Grouping ▸ RG Mode "
            "(Red/Green); the other period is not shown by default."
            if period_note is not None
            else ""
        )
        tooltip_parts = [part for part in (title, comment, period_tip) if part]
        if period_note is not None:
            title_item.setData(_PERIOD_ROLE, period_note)
        if tooltip_parts:
            title_item.setToolTip("\n".join(tooltip_parts))
        self._table.setItem(row, 1, title_item)
        if comment.strip() or period_note is not None:
            self._table.setRowHeight(row, self._two_line_row_height())

        provenance_tip = self._derived_run_tooltip(meta)
        if provenance_tip:
            run_item.setForeground(QColor(tokens.ACCENT))
            run_item.setToolTip(provenance_tip)
            # The provenance tip replaces the title/comment tooltip, but keep the
            # multi-period guidance visible on a derived run rather than losing it.
            title_item.setToolTip(
                f"{provenance_tip}\n{period_tip}" if period_tip else provenance_tip
            )

        temp = self._temperature_for_display(dataset)
        temp_item = NumericTableWidgetItem(f"{temp:.2f}")
        temp_item.setFlags(temp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        temp_item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        if self._temperature_uses_log_for_display(dataset):
            temp_item.setForeground(_LOG_TEMPERATURE_FOREGROUND)
        else:
            # The Kelvin header is suspect for EMU furnace runs that store °C
            # under a "Kelvin" label (the loader flags these without converting,
            # since +273 would corrupt a genuinely-cold run). Mark the cell so a
            # user does not silently read a 350 °C furnace point as 350 K. The
            # marker rides as a text suffix; sorting stays numeric because
            # NumericTableWidgetItem keys off the parsed value, not the text.
            suspect_tip = self._temperature_suspect_tooltip(dataset)
            if suspect_tip is not None:
                temp_item.setForeground(_SUSPECT_TEMPERATURE_FOREGROUND)
                temp_item.setText(f"{temp:.2f}{_SUSPECT_TEMPERATURE_MARKER}")
                temp_item.setToolTip(suspect_tip)
        self._table.setItem(row, 2, temp_item)

        field = self._field_for_display(dataset)
        field_item = NumericTableWidgetItem(f"{field:.1f}")
        field_item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        if self._field_uses_log_for_display(dataset):
            # Log-sourced value: display-only and tinted, like the temperature
            # column (editing a log mean is meaningless).
            field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            field_item.setForeground(_LOG_TEMPERATURE_FOREGROUND)
        else:
            field_item.setFlags(field_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 3, field_item)

        for i, column in enumerate(self._visible_extra_columns(), start=len(self._COLUMNS)):
            value = self._value_for_extra_column(dataset, column)
            item = QTableWidgetItem(value)
            if column.is_custom:
                # Custom columns are user-editable; tag the cell with its column
                # id so _on_item_changed can route the edit back to the right
                # per-run value without relying on column position.
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setData(_CUSTOM_COLUMN_ROLE, column.id)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, i, item)

        # Apply background: series-highlight takes priority over group-member tint.
        if rn in self._highlighted_runs:
            bg = _SERIES_HIGHLIGHT_BACKGROUND
        elif self._run_to_group.get(rn) is not None:
            bg = _GROUP_MEMBER_BACKGROUND
        else:
            bg = None
        if bg is not None:
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is not None:
                    item.setBackground(bg)

    def _selected_keys(self) -> list[int | str]:
        selected: list[int | str] = []
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return selected
        for idx in selection_model.selectedRows():
            item = self._table.item(idx.row(), 0)
            if item is None:
                continue
            key = item.data(self._GROUP_ROLE)
            if isinstance(key, (int, str)):
                selected.append(key)
        return selected

    def _restore_selection_by_keys(self, keys: list[int | str]) -> None:
        if not keys:
            return
        wanted = set(keys)
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        selected_any = False
        with QSignalBlocker(self._table):
            self._table.clearSelection()
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item is None:
                    continue
                key = item.data(self._GROUP_ROLE)
                if key in wanted:
                    idx = self._table.model().index(row, 0)
                    selection_model.select(
                        idx,
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    selected_any = True
        if selected_any:
            self._on_selection_changed()

    def _is_row_visible_for_selection(self, row: int) -> bool:
        return 0 <= row < self._table.rowCount() and not self._table.isRowHidden(row)

    def _next_visible_row(self, start_row: int, direction: int) -> int | None:
        if direction == 0:
            return start_row if self._is_row_visible_for_selection(start_row) else None

        row = start_row + direction
        while 0 <= row < self._table.rowCount():
            if self._is_row_visible_for_selection(row):
                return row
            row += direction
        return None

    def _selection_anchor_for_row(self, fallback_row: int) -> int:
        anchor_row = self._selection_anchor_row
        if anchor_row is not None and self._is_row_visible_for_selection(anchor_row):
            return anchor_row

        current_row = self._table.currentRow()
        if self._is_row_visible_for_selection(current_row):
            return current_row

        return fallback_row

    def _select_visible_row_range(
        self,
        anchor_row: int,
        target_row: int,
        *,
        add_to_selection: bool,
    ) -> bool:
        if not (
            self._is_row_visible_for_selection(anchor_row)
            and self._is_row_visible_for_selection(target_row)
        ):
            return False

        selection_model = self._table.selectionModel()
        if selection_model is None:
            return False

        start_row = min(anchor_row, target_row)
        end_row = max(anchor_row, target_row)
        visible_rows = [
            row for row in range(start_row, end_row + 1) if self._is_row_visible_for_selection(row)
        ]
        if not visible_rows:
            return False

        with QSignalBlocker(self._table):
            for index, row in enumerate(visible_rows):
                row_index = self._table.model().index(row, 0)
                flags = QItemSelectionModel.SelectionFlag.Rows
                if add_to_selection or index > 0:
                    flags |= QItemSelectionModel.SelectionFlag.Select
                else:
                    flags |= QItemSelectionModel.SelectionFlag.ClearAndSelect
                selection_model.select(row_index, flags)

            current_index = self._table.model().index(target_row, 0)
            selection_model.setCurrentIndex(
                current_index,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        self._on_selection_changed()
        return True

    def _two_line_row_height(self) -> int:
        """Row height for a title+comment cell, cached per table font."""
        font = self._table.font()
        font_key = font.key()
        cached = getattr(self, "_two_line_height_cache", None)
        if cached is not None and cached[0] == font_key:
            return cached[1]
        title_h = QFontMetrics(font).height()
        comment_h = QFontMetrics(_RowHighlightDelegate._comment_font(font)).height()
        height = title_h + comment_h + 2 * _RowHighlightDelegate._TWO_LINE_PAD
        self._two_line_height_cache = (font_key, height)
        return height

    #: Below this, forcing the fit would make Title unreadable — let the
    #: horizontal scrollbar appear honestly instead (extra-column overflow).
    _TITLE_FIT_FLOOR = 120

    def _resize_columns_to_content(self) -> None:
        """Content-size and clamp the columns, then fit Title to the viewport.

        Stands down permanently once the user has dragged a column edge
        (_user_sized_columns), so their layout is never stomped by a load.
        """
        if self._batch_depth:
            self._batch_resize_pending = True
            return
        if self._user_sized_columns:
            # The user owns the layout — but a brand-new extra section still
            # arrives at Qt's default width and needs its initial sizing.
            self._auto_sizing_columns = True
            try:
                header = self._table.horizontalHeader()
                default_size = header.defaultSectionSize()
                for col in range(len(self._COLUMNS), self._table.columnCount()):
                    if header.sectionSize(col) == default_size:
                        hint = self._table.sizeHintForColumn(col)
                        header.resizeSection(col, max(120, min(320, hint)))
            finally:
                self._auto_sizing_columns = False
            return
        self._auto_sizing_columns = True
        try:
            self._table.resizeColumnsToContents()
            header = self._table.horizontalHeader()
            minimums = {0: 56, 1: 145, 2: 48, 3: 48}
            maximums = {0: 150, 1: 320, 2: 76, 3: 76}
            for col, min_width in minimums.items():
                size = header.sectionSize(col)
                if size < min_width:
                    header.resizeSection(col, min_width)
                elif size > maximums[col]:
                    header.resizeSection(col, maximums[col])

            for col in range(len(self._COLUMNS), self._table.columnCount()):
                size = header.sectionSize(col)
                if size < 120:
                    header.resizeSection(col, 120)
                elif size > 320:
                    header.resizeSection(col, 320)

            self._fit_title_column()
        finally:
            self._auto_sizing_columns = False

    def _scroll_new_extra_column_into_view(self) -> None:
        """Reveal the just-added rightmost extra column.

        Custom / metadata columns append past a viewport-filling Title column,
        so a freshly added one can land off-screen to the right where the user
        adding it never sees it. Scroll horizontally to the last column once the
        scrollbar range has settled (one event-loop turn after the rebuild +
        resize). The ``self`` context cancels the pending callback if the panel
        is destroyed first.
        """
        QTimer.singleShot(0, self, self._scroll_to_last_column)

    def _scroll_to_last_column(self) -> None:
        """Scroll the table fully right so the last extra column is visible."""
        if self._table.columnCount() <= len(self._COLUMNS):
            return  # no extra column to reveal
        bar = self._table.horizontalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _fit_title_column(self) -> None:
        """Stretch Title so the columns exactly fill the viewport.

        Skipped when that would squeeze Title below _TITLE_FIT_FLOOR (e.g.
        several extra metadata columns): the table then overflows honestly
        into a horizontal scrollbar and Title stays at its content width —
        and, unlike a Qt Stretch section, remains user-draggable throughout.
        """
        header = self._table.horizontalHeader()
        viewport_width = self._table.viewport().width()
        if viewport_width <= 0:
            return
        others = sum(
            header.sectionSize(col)
            for col in range(self._table.columnCount())
            if col != _TITLE_COLUMN
        )
        available = viewport_width - others
        if available >= self._TITLE_FIT_FLOOR:
            header.resizeSection(_TITLE_COLUMN, available)

    def _on_section_resized(self, *_args) -> None:
        """Latch user column drags (programmatic resizes are flag-guarded)."""
        if not self._auto_sizing_columns and not self._updating_table:
            self._user_sized_columns = True

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Keep the columns filling the panel as the dock is resized.

        Deferred one event-loop turn: the table's viewport geometry settles
        via the layout pass *after* this event, so an immediate fit would
        read the previous width.
        """
        super().resizeEvent(event)
        self._sync_rail_header_height()
        if not self._user_sized_columns:
            # The context argument cancels the pending timer if the panel is
            # destroyed first — without it the callback fires against a
            # deleted C++ object (RuntimeError in tests/shutdown).
            QTimer.singleShot(0, self, self._fit_title_column_auto)

    def _fit_title_column_auto(self) -> None:
        """Flag-guarded fit used by the deferred resize hook."""
        if self._user_sized_columns:
            return
        self._auto_sizing_columns = True
        try:
            self._fit_title_column()
        finally:
            self._auto_sizing_columns = False

    def changeEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Re-apply two-line row heights when fonts change.

        The heights are explicit pixels computed from the table font at
        rebuild time; a font change (UI scale) would otherwise leave the
        comment line clipped until the next rebuild.
        """
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.ApplicationFontChange):
            self._two_line_height_cache = None
            self._reapply_two_line_heights()
            self._sync_rail_header_height()

    def _reapply_two_line_heights(self) -> None:
        """Reset the explicit two-line heights from the current table font."""
        if not hasattr(self, "_table"):
            return
        height: int | None = None
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _TITLE_COLUMN)
            if item is None:
                continue
            comment = item.data(_COMMENT_ROLE)
            period_note = item.data(_PERIOD_ROLE)
            has_second_line = (isinstance(comment, str) and comment.strip()) or (
                isinstance(period_note, str) and period_note.strip()
            )
            if has_second_line:
                if height is None:
                    height = self._two_line_row_height()
                self._table.setRowHeight(row, height)

    def _refresh_column_headers(self) -> None:
        """Apply base and dynamic column labels to the table header."""
        labels = list(self._COLUMNS) + [
            self._extra_column_header(column) for column in self._visible_extra_columns()
        ]
        # Column-count changes can emit section-resize signals for the new
        # sections; those are ours, not the user's (see _on_section_resized).
        self._auto_sizing_columns = True
        try:
            self._table.setColumnCount(len(labels))
            self._table.setHorizontalHeaderLabels(labels)
        finally:
            self._auto_sizing_columns = False
        # T and B headers/cells centre-align: right-aligned headers were
        # clipped by the sort-indicator arrow when sorting on these columns.
        for col in (2, 3):
            item = self._table.horizontalHeaderItem(col)
            if item is not None:
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        # Spell out the units the headers abbreviate, and warn that the Kelvin
        # column can carry a mislabelled furnace °C value (flagged per-row with
        # an amber ⚠; hover the cell for the conversion).
        temp_header = self._table.horizontalHeaderItem(2)
        if temp_header is not None:
            temp_header.setToolTip(
                "Sample temperature in kelvin. A ⚠ marks a run whose unit looks "
                "mislabelled (EMU furnace °C stored as 'Kelvin') — hover the cell."
            )
        field_header = self._table.horizontalHeaderItem(3)
        if field_header is not None:
            field_header.setToolTip("Applied magnetic field in gauss.")
        # Tooltip each extra-column header so a renamed metadata column still
        # reveals the NeXus/metadata field it came from, and custom columns read
        # as editable.
        for offset, column in enumerate(self._visible_extra_columns()):
            item = self._table.horizontalHeaderItem(len(self._COLUMNS) + offset)
            if item is None:
                continue
            if column.is_angle:
                item.setToolTip("Sample orientation angle in degrees — double-click a cell to edit")
            elif column.is_custom:
                item.setToolTip("Custom column — double-click a cell to edit")
            elif column.source_key:
                item.setToolTip(f"From metadata field: {column.source_key}")

    def _visible_extra_columns(self) -> list[ExtraColumn]:
        """Return extra columns that should appear beyond the fixed browser columns.

        Metadata columns whose source is a base override (temperature/field —
        surfaced through the fixed T/B columns and the from-log toggles) are
        hidden; custom columns are always visible.
        """
        return [
            column
            for column in self._extra_columns
            if column.is_custom or column.source_key not in self._BASE_COLUMN_OVERRIDE_KEYS
        ]

    def _extra_column_header(self, column: ExtraColumn) -> str:
        """Return the display header for an extra column (its gui-facing label)."""
        return str(column.label).strip()

    def _metadata_column(self, source_key: str, *, label: str | None = None) -> ExtraColumn:
        """Build a metadata-backed column, defaulting its label from the registry."""
        key = str(source_key).strip()
        if label is None:
            label = self._RUN_INFO_FIELD_LABELS.get(key, key)
        return ExtraColumn(id=key, label=str(label), kind=EXTRA_COLUMN_METADATA, source_key=key)

    def _find_extra_column(self, column_id: str) -> ExtraColumn | None:
        for column in self._extra_columns:
            if column.id == column_id:
                return column
        return None

    def _next_custom_column_id(self) -> str:
        """Return a fresh, collision-free id for a new custom column."""
        existing = {column.id for column in self._extra_columns}
        while True:
            candidate = f"custom:{uuid.uuid4().hex[:8]}"
            if candidate not in existing:
                return candidate

    def _parse_saved_extra_columns(self, raw: object) -> list[ExtraColumn]:
        """Rebuild column defs from saved state (legacy string list or dict list)."""
        columns: list[ExtraColumn] = []
        seen_ids: set[str] = set()
        if not isinstance(raw, (list, tuple)):
            return columns
        for entry in raw:
            if isinstance(entry, str):
                key = entry.strip()
                if not key:
                    continue
                column = self._metadata_column(key)
            else:
                column = ExtraColumn.from_dict(entry)
                if column is None:
                    continue
                # A metadata column with no explicit (renamed) label shows its
                # registry name rather than the raw dotted source key.
                if (
                    not column.is_custom
                    and column.source_key
                    and column.label in ("", column.source_key)
                ):
                    column.label = self._RUN_INFO_FIELD_LABELS.get(
                        column.source_key, column.source_key
                    )
            if column.id in seen_ids:
                continue
            seen_ids.add(column.id)
            columns.append(column)
        return columns

    def extra_columns(self) -> list[ExtraColumn]:
        """Return a copy of the current extra-column definitions (custom + metadata)."""
        return list(self._extra_columns)

    def custom_label_fields(self) -> list[tuple[str, str]]:
        """Return ``(label, "custom:<id>")`` pairs for the user's custom columns.

        These are the columns offered as the plot legend label and the parameter
        trend x-axis. Their per-run values live in
        ``dataset.metadata["custom_fields"]`` keyed by the same id, so consumers
        resolve them without reaching back into the browser.
        """
        return [(column.label, column.id) for column in self._extra_columns if column.is_custom]

    def custom_values_by_run(self) -> dict[int, dict[str, str]]:
        """Map each known run number to its live custom-column values.

        The source of truth for custom columns is
        ``dataset.metadata["custom_fields"]`` (keyed by column id). Exposing it
        per run lets trend consumers re-link existing batch results to a column
        that was added or populated *after* the fit completed (the ordering
        trap) without re-running the batch. A run with no custom values maps to
        an empty dict so consumers can clear stale snapshots, not just add.
        """
        out: dict[int, dict[str, str]] = {}
        for source in (self._datasets, self._combined_datasets):
            for run_number, dataset in source.items():
                try:
                    run_key = int(run_number)
                except (TypeError, ValueError):
                    continue
                fields = dataset.metadata.get(CUSTOM_FIELDS_METADATA_KEY)
                if isinstance(fields, dict):
                    out[run_key] = {str(k): str(v) for k, v in fields.items()}
                else:
                    out.setdefault(run_key, {})
        return out

    def angle_x_field(self) -> tuple[str, str] | None:
        """Return ``(label, id)`` for the special Angle field, or None if absent.

        The Angle field is promoted to a first-class trend x-axis rather than
        offered among the generic custom columns, so consumers route it separately.
        """
        column = next((c for c in self._extra_columns if c.is_angle), None)
        return (column.label, column.id) if column is not None else None

    def _notify_extra_columns_changed(self) -> None:
        self.extra_columns_changed.emit()

    def add_extra_column(self, field_key: str) -> None:
        """Add a metadata-backed dynamic column to the browser table."""
        key = str(field_key).strip()
        if key == "temperature":
            self.set_use_temperature_from_log(True)
            return
        if key == "field":
            self.set_use_field_from_log(True)
            return
        if not key or any(c.source_key == key for c in self._extra_columns):
            return
        self._extra_columns.append(self._metadata_column(key))
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()
        self._scroll_new_extra_column_into_view()
        self._notify_extra_columns_changed()

    def add_custom_column(self, label: str) -> ExtraColumn | None:
        """Create an empty, user-editable custom column and return its definition."""
        name = str(label).strip()
        if not name:
            return None
        column = ExtraColumn(
            id=self._next_custom_column_id(),
            label=name,
            kind=EXTRA_COLUMN_CUSTOM,
            source_key=None,
        )
        self._extra_columns.append(column)
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()
        self._scroll_new_extra_column_into_view()
        self._notify_extra_columns_changed()
        return column

    def _prompt_add_custom_column(self) -> None:
        """Ask for a name and append an empty, user-editable custom column."""
        name, ok = QInputDialog.getText(self, "Add custom column", "Column name:")
        if ok:
            self.add_custom_column(name)

    def has_angle_column(self) -> bool:
        """Return ``True`` when the singleton special "Angle" field exists."""
        return any(column.is_angle for column in self._extra_columns)

    def add_angle_column(self) -> ExtraColumn | None:
        """Add the singleton special "Angle (°)" field (numeric degrees per run).

        Returns the existing Angle field if one is already present, so the action
        is idempotent and never produces a duplicate.
        """
        existing = next((column for column in self._extra_columns if column.is_angle), None)
        if existing is not None:
            return existing
        column = ExtraColumn(
            id=ANGLE_COLUMN_ID,
            label=ANGLE_COLUMN_LABEL,
            kind=EXTRA_COLUMN_CUSTOM,
            source_key=None,
            is_angle=True,
        )
        self._extra_columns.append(column)
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()
        self._scroll_new_extra_column_into_view()
        self._notify_extra_columns_changed()
        return column

    def rename_extra_column(self, column_id: str, new_label: str) -> bool:
        """Rename a column's gui-facing label (its source_key is untouched)."""
        name = str(new_label).strip()
        column = self._find_extra_column(column_id)
        if column is None or not name or name == column.label:
            return False
        column.label = name
        self._refresh_column_headers()
        self._resize_columns_to_content()
        self._notify_extra_columns_changed()
        return True

    def remove_extra_column(self, field_key_or_id: str) -> None:
        """Remove an extra column by id or (for metadata columns) source key."""
        if field_key_or_id == "temperature":
            self.set_use_temperature_from_log(False)
            return
        if field_key_or_id == "field":
            self.set_use_field_from_log(False)
            return
        removed = [
            column
            for column in self._extra_columns
            if column.id == field_key_or_id or column.source_key == field_key_or_id
        ]
        if not removed:
            return
        self._extra_columns = [column for column in self._extra_columns if column not in removed]
        # Deleting a custom column deletes its data too: purge the per-run stored
        # values so they can't silently resurrect on re-add. This matters for the
        # Angle field (fixed id, so a re-add would otherwise reinherit old values)
        # and also clears the orphan cruft that uuid-keyed custom columns leak.
        for column in removed:
            if column.is_custom:
                self._purge_custom_column_values(column.id)
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()
        self._notify_extra_columns_changed()

    def _purge_custom_column_values(self, column_id: str) -> None:
        """Remove a custom column's per-run value from every dataset's metadata."""
        for dataset in self._datasets.values():
            existing = dataset.metadata.get(CUSTOM_FIELDS_METADATA_KEY)
            if isinstance(existing, dict) and column_id in existing:
                # Copy-on-write: the dict can be shared across dataset/run clones
                # (mirrors _set_custom_column_value), so rebind rather than mutate.
                fields = dict(existing)
                fields.pop(column_id, None)
                dataset.metadata[CUSTOM_FIELDS_METADATA_KEY] = fields

    def get_extra_columns(self) -> list[str]:
        """Return the metadata source keys currently shown (for inclusion tracking).

        Custom columns have no metadata source and are intentionally excluded —
        this drives the Get Info "Include in Data Browser" checkboxes, which only
        concern NeXus/metadata fields. The from-log pseudo-keys are appended to
        mirror the fixed T/B columns' include state.
        """
        columns = [
            column.source_key for column in self._extra_columns if column.source_key is not None
        ]
        if self._use_temperature_from_log:
            columns.append("temperature")
        if self._use_field_from_log:
            columns.append("field")
        return columns

    def set_use_temperature_from_log(self, enabled: bool) -> None:
        """Set the global temperature-from-log display option."""
        enabled = bool(enabled)
        changed = self._use_temperature_from_log != enabled or bool(
            self._temperature_from_log_overrides
        )
        self._use_temperature_from_log = enabled
        self._temperature_from_log_overrides.clear()
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def use_temperature_from_log(self) -> bool:
        """Return the global temperature-from-log display option."""
        return bool(self._use_temperature_from_log)

    def set_dataset_temperature_from_log(self, run_number: int, enabled: bool) -> None:
        """Override temperature-from-log display for a single dataset."""
        rn = int(run_number)
        enabled = bool(enabled)
        if enabled == self._use_temperature_from_log:
            changed = rn in self._temperature_from_log_overrides
            self._temperature_from_log_overrides.pop(rn, None)
        else:
            changed = self._temperature_from_log_overrides.get(rn) != enabled
            self._temperature_from_log_overrides[rn] = enabled
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def dataset_uses_temperature_from_log(self, run_number: int) -> bool:
        """Return whether one dataset is configured to show log temperature."""
        rn = int(run_number)
        return bool(self._temperature_from_log_overrides.get(rn, self._use_temperature_from_log))

    def set_use_field_from_log(self, enabled: bool) -> None:
        """Set the global field-from-log display option (B from the data log).

        The analogue of :meth:`set_use_temperature_from_log` (WiMDA's
        ``Bfromaveinblog``): show the mean of the magnetic-field log channel in
        the B column instead of the header scalar.
        """
        enabled = bool(enabled)
        changed = self._use_field_from_log != enabled or bool(self._field_from_log_overrides)
        self._use_field_from_log = enabled
        self._field_from_log_overrides.clear()
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def use_field_from_log(self) -> bool:
        """Return the global field-from-log display option."""
        return bool(self._use_field_from_log)

    def set_dataset_field_from_log(self, run_number: int, enabled: bool) -> None:
        """Override field-from-log display for a single dataset."""
        rn = int(run_number)
        enabled = bool(enabled)
        if enabled == self._use_field_from_log:
            changed = rn in self._field_from_log_overrides
            self._field_from_log_overrides.pop(rn, None)
        else:
            changed = self._field_from_log_overrides.get(rn) != enabled
            self._field_from_log_overrides[rn] = enabled
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def dataset_uses_field_from_log(self, run_number: int) -> bool:
        """Return whether one dataset is configured to show log field."""
        rn = int(run_number)
        return bool(self._field_from_log_overrides.get(rn, self._use_field_from_log))

    def _resolve_metadata_path(self, dataset: MuonDataset, field_key: str | None):
        """Resolve a metadata/synthetic key to a value for dynamic columns."""
        if not field_key:
            return None
        if field_key.startswith("run_info."):
            return self._resolve_run_info_value(dataset, field_key)

        if field_key == "temperature":
            return self._temperature_for_display(dataset)

        metadata = dataset.metadata
        current = metadata
        for part in field_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _temperature_for_display(self, dataset: MuonDataset) -> float:
        """Return the temperature shown in the fixed browser temperature column."""
        if self.dataset_uses_temperature_from_log(int(dataset.run_number)):
            log_temperature = self._temperature_from_log_for_display(dataset)
            if log_temperature is not None:
                return float(log_temperature)
        try:
            return float(dataset.metadata.get("temperature", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _temperature_uses_log_for_display(self, dataset: MuonDataset) -> bool:
        """Return whether the displayed temperature value came from a log."""
        return (
            self.dataset_uses_temperature_from_log(int(dataset.run_number))
            and self._temperature_from_log_for_display(dataset) is not None
        )

    def _temperature_suspect_tooltip(self, dataset: MuonDataset) -> str | None:
        """Warning tooltip when the run's Kelvin temperature is a suspected °C value.

        The loader sets ``temperature_unit_suspect`` for EMU furnace runs that
        store Celsius under a ``Kelvin`` label (see
        ``NexusLoader._temperature_unit_suspect``); the value is surfaced, never
        converted. Returns ``None`` for ordinary runs.

        The suspicion only fires when the file carries no logged sample
        thermometer, so the **Options ▸ Use temperature from log** toggle has
        nothing to fall back to here — the tooltip says so and spells out the
        °C→K conversion the user must apply by hand if the run really is a
        furnace point.
        """
        metadata = getattr(dataset, "metadata", None)
        if not isinstance(metadata, dict) or not metadata.get("temperature_unit_suspect"):
            return None
        try:
            value = float(metadata.get("temperature", 0.0))
        except (TypeError, ValueError):
            return None
        reason = str(metadata.get("temperature_unit_suspect_reason", "")).strip()
        lines = [
            "Temperature unit looks mislabelled.",
            (
                f"Shown as {value:.2f} K, but this run has no logged sample "
                "thermometer and its value sits in EMU furnace territory, where "
                "the NeXus header is known to store °C under a 'Kelvin' label."
            ),
            (
                f"If this is °C, the temperature in K is "
                f"{value:.2f} + {_ABSOLUTE_ZERO_CELSIUS:g} = "
                f"{value + _ABSOLUTE_ZERO_CELSIUS:.2f} K."
            ),
            (
                "The value is left unchanged (auto-adding 273 would corrupt a "
                "genuinely-cold run). 'Use temperature from log' cannot fix this "
                "— there is no logged thermometer to switch to. Verify the unit."
            ),
        ]
        if reason:
            lines.append(f"Loader note: {reason}")
        return "\n".join(lines)

    def _temperature_from_log_for_display(self, dataset: MuonDataset) -> float | None:
        """Return the log-derived temperature used by the browser column."""
        source_datasets = self.get_combined_source_datasets(int(dataset.run_number))
        if source_datasets:
            weighted_sum = 0.0
            total_weight = 0.0
            fallback_temperatures: list[float] = []
            for source_dataset in source_datasets:
                source_temperature = self._series_mean_for_field(source_dataset, "temperature")
                if source_temperature is None:
                    continue
                fallback_temperatures.append(float(source_temperature))
                event_count = self._event_count_for_dataset(source_dataset)
                if event_count is None or event_count <= 0.0:
                    continue
                weighted_sum += float(source_temperature) * event_count
                total_weight += event_count
            if total_weight > 0.0:
                return weighted_sum / total_weight
            if fallback_temperatures:
                return float(np.mean(fallback_temperatures))
        return self._series_mean_for_field(dataset, "temperature")

    def _field_for_display(self, dataset: MuonDataset) -> float:
        """Return the field shown in the fixed browser B column."""
        if self.dataset_uses_field_from_log(int(dataset.run_number)):
            log_field = self._field_from_log_for_display(dataset)
            if log_field is not None:
                return float(log_field)
        try:
            return float(dataset.metadata.get("field", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _field_uses_log_for_display(self, dataset: MuonDataset) -> bool:
        """Return whether the displayed field value came from a log."""
        return (
            self.dataset_uses_field_from_log(int(dataset.run_number))
            and self._field_from_log_for_display(dataset) is not None
        )

    def _field_from_log_for_display(self, dataset: MuonDataset) -> float | None:
        """Return the log-derived field used by the browser B column."""
        source_datasets = self.get_combined_source_datasets(int(dataset.run_number))
        if source_datasets:
            weighted_sum = 0.0
            total_weight = 0.0
            fallback_fields: list[float] = []
            for source_dataset in source_datasets:
                source_field = self._series_mean_for_field(source_dataset, "field")
                if source_field is None:
                    continue
                fallback_fields.append(float(source_field))
                event_count = self._event_count_for_dataset(source_dataset)
                if event_count is None or event_count <= 0.0:
                    continue
                weighted_sum += float(source_field) * event_count
                total_weight += event_count
            if total_weight > 0.0:
                return weighted_sum / total_weight
            if fallback_fields:
                return float(np.mean(fallback_fields))
        return self._series_mean_for_field(dataset, "field")

    def displayed_coordinate(self, run_number: int) -> dict[str, float | None]:
        """The ``field``/``temperature`` the browser currently *displays* for a run.

        Honours the log toggles and per-dataset overrides — the same resolution
        as the fixed B/T columns (:meth:`_field_for_display`,
        :meth:`_temperature_for_display`) — so a batch trend and its CSV export
        plot each run at the abscissa the user sees in the table, not the parked
        header setpoint. When the browser shows a logged value it replaces the
        scalar; an axis with no recorded value is ``None`` (off that axis), never
        ``0`` — preserving the trend's "missing coordinate" semantics rather than
        the columns' ``0.0`` fallback.
        """
        dataset = self.get_dataset(int(run_number))
        if dataset is None:
            return {"field": None, "temperature": None}
        rn = int(dataset.run_number)
        return {
            "field": self._displayed_axis_coordinate(
                dataset,
                "field",
                uses_log=self.dataset_uses_field_from_log(rn),
                log_value_source=lambda: self._field_from_log_for_display(dataset),
            ),
            "temperature": self._displayed_axis_coordinate(
                dataset,
                "temperature",
                uses_log=self.dataset_uses_temperature_from_log(rn),
                log_value_source=lambda: self._temperature_from_log_for_display(dataset),
            ),
        }

    def _displayed_axis_coordinate(
        self,
        dataset: MuonDataset,
        metadata_key: str,
        *,
        uses_log: bool,
        log_value_source,
    ) -> float | None:
        """Displayed value for one trend axis: logged value if shown, else scalar.

        ``log_value_source`` is resolved lazily so the (potentially weighted)
        log mean is only computed when that axis is actually displaying the log
        — the common setpoint path stays as cheap as the old metadata read.
        Returns ``None`` (not ``0``) when no finite value is recorded.
        """
        if uses_log:
            log_value = log_value_source()
            if log_value is not None:
                return float(log_value)
        raw = dataset.metadata.get(metadata_key)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _event_count_for_dataset(self, dataset: MuonDataset) -> float | None:
        """Return the total histogram counts used to weight combined-run summaries."""
        run = dataset.run
        if run is None or not run.histograms:
            return None
        try:
            return float(np.sum([np.sum(histogram.counts) for histogram in run.histograms]))
        except (TypeError, ValueError):
            return None

    def _series_mean_for_field(self, dataset: MuonDataset, field_key: str) -> float | None:
        """Return the mean from the time-series log associated with a summary field."""
        series = dataset.metadata.get("nexus_time_series", {})
        if not isinstance(series, dict):
            return None
        scored = [
            (score, series_path)
            for series_path in series
            if (score := self._series_path_score(field_key, series_path, series.get(series_path)))
            > 0
        ]
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        for _, series_path in scored:
            info = series.get(series_path, {})
            if not isinstance(info, dict):
                continue
            # Run-active (t >= 0) mean, shared with the loader so the browser and
            # the scriptable ``sample_temperature_logged`` never disagree.
            value = active_series_mean(info)
            if value is not None:
                return value
        return None

    def _series_path_score(self, field_key: str, series_path: str, info) -> int:
        """Score how well a log series matches a browser summary field.

        Delegates to the shared :func:`score_series_path` so this panel and the
        Run Info dialog rank sensors identically.
        """
        return score_series_path(field_key, series_path, info)

    def _resolve_run_info_value(self, dataset: MuonDataset, field_key: str):
        """Resolve synthetic ``run_info.*`` keys used by Run Info summary rows."""
        key = field_key[len("run_info.") :]
        if key == "points":
            return dataset.n_points

        run = dataset.run
        if run is None or not run.histograms:
            return None

        if key == "histograms":
            return len(run.histograms)

        h0 = run.histograms[0]
        if key == "bins":
            return h0.n_bins
        if key == "bin_width_us":
            return h0.bin_width

        if key in ("good_events_mev", "events_per_frame"):
            grouping = run.grouping if isinstance(getattr(run, "grouping", None), dict) else None
            if grouping is None and isinstance(dataset.metadata.get("grouping"), dict):
                grouping = dataset.metadata["grouping"]
            good = good_event_count(run.histograms, grouping)
            if good is None:
                return None
            if key == "good_events_mev":
                return good / 1.0e6
            # Events per frame: good events over the dead-time frame normaliser.
            frames = good_frames(grouping, 0.0)
            if frames <= 0:
                return None
            return good / frames

        total_counts = float(np.sum([np.sum(h.counts) for h in run.histograms]))
        if key == "counts_mev":
            return total_counts / 1.0e6
        if key == "counts_per_detector":
            return total_counts / max(len(run.histograms), 1)
        return None

    def _format_extra_value(self, value) -> str:
        """Format dynamic-column values into compact table text."""
        if value is None:
            return "—"

        if isinstance(value, dict):
            if "mean" in value:
                try:
                    return f"{float(value['mean']):.6g}"
                except (TypeError, ValueError):
                    return str(value.get("mean", "—"))
            text = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
            return text if len(text) <= 48 else f"{text[:45]}..."

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                return "—"
            if np.issubdtype(arr.dtype, np.number):
                return f"{float(np.nanmean(arr.astype(np.float64))):.6g}"
            text = str(list(value))
            return text if len(text) <= 48 else f"{text[:45]}..."

        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"

        text = str(value)
        return text if text else "—"

    def _value_for_extra_column(self, dataset: MuonDataset, column: ExtraColumn) -> str:
        """Return rendered cell text for an extra column."""
        if column.is_custom:
            return self.custom_column_value(dataset, column.id)
        return self._format_extra_value(self._resolve_metadata_path(dataset, column.source_key))

    def _raw_value_for_column(self, dataset: MuonDataset, column: ExtraColumn):
        """Return the raw (unformatted) value used for sorting an extra column."""
        if column.is_custom:
            text = self.custom_column_value(dataset, column.id)
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return text
        return self._resolve_metadata_path(dataset, column.source_key)

    def custom_column_value(self, dataset: MuonDataset, column_id: str) -> str:
        """Return the stored per-run text for a custom column (``""`` if unset)."""
        fields = dataset.metadata.get(CUSTOM_FIELDS_METADATA_KEY)
        if isinstance(fields, dict):
            value = fields.get(column_id)
            if value is not None:
                return str(value)
        return ""

    def _set_custom_column_value(self, dataset: MuonDataset, column_id: str, text: str) -> None:
        """Write (or clear) a custom column's per-run value in dataset metadata.

        Copy-on-write: a *new* dict is bound rather than mutating the existing one
        in place. The custom-fields dict can be shared by reference across a run's
        ``dataset.metadata``/``run.metadata`` (restore) or with a shallow
        :meth:`MuonDataset.copy` clone (co-add/period combine); rebinding here keeps
        an edit on one from silently bleeding into the other.
        """
        existing = dataset.metadata.get(CUSTOM_FIELDS_METADATA_KEY)
        fields = dict(existing) if isinstance(existing, dict) else {}
        if text:
            fields[column_id] = text
        else:
            fields.pop(column_id, None)
        dataset.metadata[CUSTOM_FIELDS_METADATA_KEY] = fields

    # ------------------------------------------------------------------
    # Row and selection helpers
    # ------------------------------------------------------------------

    def _is_group_key(self, key: object) -> bool:
        return isinstance(key, str) and key.startswith(self._GROUP_SENTINEL_PREFIX)

    def _group_id_from_key(self, key: object) -> str | None:
        if not self._is_group_key(key):
            return None
        return str(key)[len(self._GROUP_SENTINEL_PREFIX) :]

    def _dataset_run_numbers_from_keys(self, keys: list[int | str]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for key in keys:
            if isinstance(key, int) and key in self._datasets and key not in seen:
                out.append(key)
                seen.add(key)
                continue
            gid = self._group_id_from_key(key)
            if gid is None:
                continue
            group = self._groups.get(gid)
            if group is None:
                continue
            for rn in group.member_run_numbers:
                if rn in self._datasets and rn not in seen:
                    out.append(rn)
                    seen.add(rn)
        return out

    def _get_selected_run_numbers(self) -> list[int]:
        return self._dataset_run_numbers_from_keys(self._selected_keys())

    def _get_selected_group_ids(self) -> list[str]:
        ids: list[str] = []
        for key in self._selected_keys():
            gid = self._group_id_from_key(key)
            if gid is not None:
                ids.append(gid)
        return ids

    def get_selected_group_ids(self) -> list[str]:
        return self._get_selected_group_ids()

    def get_current_selection_key(self) -> int | str | None:
        """Return the key for the current row, if any."""
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        if item is None:
            return None
        key = item.data(self._GROUP_ROLE)
        return key if isinstance(key, (int, str)) else None

    def get_current_dataset(self) -> MuonDataset | None:
        """Return dataset on the current table row when that row is a run."""
        key = self.get_current_selection_key()
        if not isinstance(key, int):
            return None
        return self._datasets.get(key)

    def is_single_group_selected(self) -> bool:
        """Return True when the selection contains exactly one group header row."""
        return len(self._selected_keys()) == 1 and len(self._get_selected_group_ids()) == 1

    def get_group_name(self, group_id: str) -> str | None:
        group = self._groups.get(group_id)
        return None if group is None else group.name

    def get_group_id_for_run(self, run_number: int) -> str | None:
        """Return data-group id containing *run_number*, if any."""
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return None
        return self._run_to_group.get(run_key)

    def get_group_member_run_numbers(self, group_id: str) -> list[int]:
        """Return run numbers currently belonging to *group_id*."""
        group = self._groups.get(group_id)
        if group is None:
            return []
        return [int(rn) for rn in group.member_run_numbers]

    def get_dataset(self, run_number: int) -> MuonDataset | None:
        return self._datasets.get(run_number)

    def set_highlighted_runs(self, run_numbers: set[int] | None) -> None:
        """Tint *run_numbers* with the FitSeries membership indicator (red tint).

        Pass an empty set or ``None`` to clear all highlights.  The tint is
        purely decorative — it never alters the table's selection state.
        """
        new_set = set(run_numbers) if run_numbers else set()
        if new_set == self._highlighted_runs:
            return
        self._highlighted_runs = new_set
        # Update backgrounds in-place without a full table rebuild.
        self._apply_series_highlights()

    def _apply_series_highlights(self) -> None:
        """Walk the table and apply/remove the series-highlight tint."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            key = item.data(self._GROUP_ROLE)
            if not isinstance(key, int):
                continue  # Skip group header rows.
            rn = key
            in_group = self._run_to_group.get(rn) is not None
            if rn in self._highlighted_runs:
                bg = _SERIES_HIGHLIGHT_BACKGROUND
            elif in_group:
                bg = _GROUP_MEMBER_BACKGROUND
            else:
                bg = QColor(0, 0, 0, 0)  # transparent / default
            for col in range(self._table.columnCount()):
                cell = self._table.item(row, col)
                if cell is not None:
                    cell.setBackground(bg)

    def select_runs(self, run_numbers: set[int] | list[int]) -> None:
        """Perform a true selection of *run_numbers* in the browser table.

        This is a real selection (drives ``selection_changed``, updates the fit
        panel, etc.), distinct from the decorative red tint applied by
        :meth:`set_highlighted_runs`.  Existing selection is replaced.  Runs
        not currently visible in the table are silently skipped.
        """
        keys = [int(r) for r in run_numbers]
        if not keys:
            self._table.clearSelection()
            return
        self._restore_selection_by_keys(keys)
        # Scroll to the first matched row so the user can see the selection.
        wanted = set(keys)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(self._GROUP_ROLE) in wanted:
                self._table.scrollToItem(item)
                break

    def get_selected_datasets(self) -> list[MuonDataset]:
        selected: list[MuonDataset] = []
        for run_number in self._get_selected_run_numbers():
            dataset = self._datasets.get(run_number)
            if dataset is not None:
                selected.append(dataset)
        return selected

    def get_all_datasets(self) -> list[MuonDataset]:
        """Return all datasets currently present in the browser."""
        return list(self._datasets.values())

    def _combined_run_display(self, run_number: int) -> str:
        """Source-number label for a combined row, with the operator separator.

        Co-add rows join with ``" + "``; reference-subtraction rows with
        ``" − "`` (sample − reference).
        """
        sources = self._combined_datasets.get(run_number, [])
        separator = " − " if self._combined_signs.get(run_number, 1) == -1 else " + "
        return separator.join(map(str, sources))

    def is_combined_dataset(self, run_number: int) -> bool:
        """Return ``True`` when *run_number* refers to a combined row."""
        try:
            return int(run_number) in self._combined_datasets
        except (TypeError, ValueError):
            return False

    def get_combined_source_datasets(self, run_number: int) -> list[MuonDataset]:
        """Return hidden source datasets for a combined row."""
        try:
            combined_rn = int(run_number)
        except (TypeError, ValueError):
            return []
        return list(self._combined_source_datasets.get(combined_rn, []))

    def _combine_builder_for(self, combined_rn: int):
        """Return the rebuild builder matching a combined row's operation."""
        if self._combined_methods.get(combined_rn) == "subtract_signed":
            return self._signed_subtract_datasets
        if self._combined_signs.get(combined_rn, 1) == -1:
            return self._subtract_datasets
        return self._coadd_datasets

    def rebuild_combined_dataset(self, run_number: int) -> MuonDataset | None:
        """Recompute one combined dataset from its hidden source datasets."""
        try:
            combined_rn = int(run_number)
        except (TypeError, ValueError):
            return None

        source_datasets = self._combined_source_datasets.get(combined_rn, [])
        if len(source_datasets) < 2:
            return None

        source_run_numbers = self._combined_datasets.get(
            combined_rn,
            [int(ds.run_number) for ds in source_datasets],
        )
        from asymmetry.core.data.combine import CombineError

        builder = self._combine_builder_for(combined_rn)
        try:
            rebuilt = builder(
                source_datasets,
                source_run_numbers,
                combined_run_number=combined_rn,
                existing_dataset=self._datasets.get(combined_rn),
            )
        except CombineError:
            # Source runs no longer combine (e.g. histograms unavailable after a
            # partial reload); leave the existing combined row untouched.
            return None
        self._datasets[combined_rn] = rebuilt
        return rebuilt

    def _normalize_grouping_value(self, value):
        """Return a deterministic representation for grouping comparisons."""
        if isinstance(value, dict):
            normalized: dict[str, object] = {}
            for key in sorted(value, key=lambda item: str(item)):
                try:
                    norm_key = str(int(key))
                except (TypeError, ValueError):
                    norm_key = str(key)
                normalized[norm_key] = self._normalize_grouping_value(value[key])
            return normalized
        if isinstance(value, (list, tuple)):
            return [self._normalize_grouping_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return [self._normalize_grouping_value(v) for v in value.tolist()]
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (np.floating, float)):
            val = float(value)
            if not np.isfinite(val):
                return str(val)
            return round(val, 12)
        if isinstance(value, str):
            return value.strip()
        return value

    def _grouping_signature(self, dataset: MuonDataset):
        """Return normalized grouping payload for co-add compatibility checks."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if run is None or not isinstance(grouping, dict):
            return None

        groups = grouping.get("groups")
        if not isinstance(groups, dict) or not groups:
            return None

        histograms = getattr(run, "histograms", None) or []
        t0_default = 0
        last_good_default = max(0, dataset.n_points - 1)
        if histograms:
            try:
                t0_default = int(histograms[0].t0_bin)
            except (TypeError, ValueError, IndexError):
                t0_default = 0
            try:
                last_good_default = max(0, len(histograms[0].counts) - 1)
            except (TypeError, ValueError, IndexError):
                last_good_default = max(0, dataset.n_points - 1)

        try:
            t0_bin = int(grouping.get("t0_bin", t0_default))
        except (TypeError, ValueError):
            t0_bin = t0_default

        raw_t_good = grouping.get("t_good_offset")
        if raw_t_good is None:
            try:
                raw_t_good = int(grouping.get("first_good_bin", t0_bin)) - t0_bin
            except (TypeError, ValueError):
                raw_t_good = 0
        try:
            t_good_offset = max(0, int(raw_t_good))
        except (TypeError, ValueError):
            t_good_offset = 0

        first_good_bin = max(0, t0_bin + t_good_offset)
        try:
            last_good_bin = int(grouping.get("last_good_bin", last_good_default))
        except (TypeError, ValueError):
            last_good_bin = last_good_default

        try:
            bin_index_base = 1 if int(grouping.get("bin_index_base", 0)) == 1 else 0
        except (TypeError, ValueError):
            bin_index_base = 0

        try:
            bunching_factor = int(grouping.get("bunching_factor", 1))
        except (TypeError, ValueError):
            bunching_factor = 1
        signature = {
            "groups": groups,
            "forward_group": int(grouping.get("forward_group", 1)),
            "backward_group": int(grouping.get("backward_group", 2)),
            "alpha": float(grouping.get("alpha", 1.0)),
            "alpha_x": grouping.get("alpha_x"),
            "alpha_y": grouping.get("alpha_y"),
            "alpha_z": grouping.get("alpha_z"),
            "vector_axis": grouping.get("vector_axis"),
            "group_names": grouping.get("group_names", {}),
            "t0_bin": t0_bin,
            "t_good_offset": t_good_offset,
            "first_good_bin": first_good_bin,
            "last_good_bin": last_good_bin,
            "bin_index_base": bin_index_base,
            "bunching_factor": bunching_factor,
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "dead_time_us": grouping.get("dead_time_us"),
            "period_mode": grouping.get("period_mode"),
            "period_dead_time_us": grouping.get("period_dead_time_us"),
        }
        return self._normalize_grouping_value(signature)

    def _coadd_compatibility_error(self, datasets: list[MuonDataset]) -> str | None:
        """Return a user-facing error when selected datasets cannot be co-added."""
        if len(datasets) < 2:
            return "Select at least two grouped datasets to co-add."

        signatures = [self._grouping_signature(ds) for ds in datasets]
        if any(signature is None for signature in signatures):
            return (
                "Co-add requires identical grouping on every selected dataset. "
                "Apply grouping to each source run before combining them."
            )

        first_signature = signatures[0]
        if any(signature != first_signature for signature in signatures[1:]):
            return (
                "Co-add requires identical grouping on every selected dataset. "
                "Align groups, alpha, good-bin limits, bunching, and deadtime settings first."
            )
        return None

    def render_logbook_tsv(self) -> tuple[str, int]:
        """Render the TSV logbook to a string, reading widget/grouping state.

        Returns ``(content, exported_rows)``. The render reads the table model
        (columns, grouping, every dataset) so it must run on the GUI thread; the
        caller writes ``content`` to disk (off-thread where it matters). The
        export includes rows hidden by filters or collapsed groups.
        """
        headers = self._active_column_headers()
        sections = self._export_sections()
        exported_rows = 0

        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        for section_index, (section_name, run_numbers) in enumerate(sections):
            writer.writerow(self._group_header_values(len(headers), section_name))
            writer.writerow(headers)

            for run_number in run_numbers:
                dataset = self._datasets.get(run_number)
                if dataset is None:
                    continue
                writer.writerow(self._export_row_values(run_number, dataset))
                exported_rows += 1

            if section_index < len(sections) - 1:
                writer.writerow([])

        return buffer.getvalue(), exported_rows

    def export_logbook_tsv(self, path: str) -> int:
        """Render and write the TSV logbook to *path* (synchronous helper)."""
        content, exported_rows = self.render_logbook_tsv()
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write(content)
        return exported_rows

    def render_logbook_rtf(self) -> tuple[str, int]:
        """Render the RTF logbook to a string, reading widget/grouping state.

        Returns ``(content, exported_rows)``. Like :meth:`render_logbook_tsv`,
        the render reads the table model and must run on the GUI thread; the
        caller writes the bytes. Includes rows hidden by filters/collapsed groups.
        """
        headers = self._active_column_headers()
        sections = self._export_sections()
        exported_rows = 0
        header_cells = [self._rtf_header_cell(header) for header in headers]
        header_line = self._rtf_tabbed_line(header_cells, preescaped=True)

        parts: list[str] = [r"{\rtf1\ansi\deff0\n"]
        for section_index, (section_name, run_numbers) in enumerate(sections):
            group_header = self._group_header_values(len(headers), section_name)
            group_cells = [self._rtf_escape(value) for value in group_header]
            parts.append(self._rtf_tabbed_line(group_cells, preescaped=True))
            parts.append("\n")
            parts.append(header_line)
            parts.append("\n")

            for run_number in run_numbers:
                dataset = self._datasets.get(run_number)
                if dataset is None:
                    continue
                parts.append(self._rtf_tabbed_line(self._export_row_values(run_number, dataset)))
                parts.append("\n")
                exported_rows += 1

            if section_index < len(sections) - 1:
                parts.append(r"\par\n")

        parts.append("}")
        return "".join(parts), exported_rows

    def export_logbook_rtf(self, path: str) -> int:
        """Render and write the RTF logbook to *path* (synchronous helper)."""
        content, exported_rows = self.render_logbook_rtf()
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write(content)
        return exported_rows

    def _rtf_tabbed_line(self, values: list[str], *, preescaped: bool = False) -> str:
        escaped = values if preescaped else [self._rtf_escape(str(value)) for value in values]
        return r"\tab ".join(escaped) + r"\par"

    def _group_header_values(self, column_count: int, section_name: str) -> list[str]:
        """Return a section-header row with the same width as the table."""
        if column_count <= 0:
            return [f"Data Group: {section_name}"]

        row = [""] * column_count
        row[0] = "Data Group"
        if column_count >= 2:
            row[1] = section_name
        else:
            row[0] = f"Data Group: {section_name}"
        return row

    def _rtf_header_cell(self, header: str) -> str:
        """Return RTF-formatted header text for export table cells."""
        if header == "T (K)":
            return r"\i T\i0 (K)"
        if header == "B (G)":
            return r"\i B\i0 (G)"
        return self._rtf_escape(header)

    def _rtf_signed16(self, value: int) -> int:
        return value if value < 0x8000 else value - 0x10000

    def _rtf_escape(self, text: str) -> str:
        sanitized = text.replace("\r", " ").replace("\n", " ")
        if sanitized.isascii():
            return sanitized.translate(
                {
                    ord("\\"): r"\\",
                    ord("{"): r"\{",
                    ord("}"): r"\}",
                    ord("\t"): r"\tab ",
                }
            )

        escaped: list[str] = []
        for ch in sanitized:
            if ch == "\\":
                escaped.append(r"\\")
                continue
            if ch == "{":
                escaped.append(r"\{")
                continue
            if ch == "}":
                escaped.append(r"\}")
                continue
            if ch == "\t":
                escaped.append(r"\tab ")
                continue
            if ch in ("\r", "\n"):
                escaped.append(" ")
                continue

            codepoint = ord(ch)
            if codepoint <= 0x7F:
                escaped.append(ch)
                continue

            if codepoint <= 0xFFFF:
                escaped.append(f"\\u{self._rtf_signed16(codepoint)}?")
                continue

            encoded = codepoint - 0x10000
            high_surrogate = 0xD800 + (encoded >> 10)
            low_surrogate = 0xDC00 + (encoded & 0x3FF)
            escaped.append(f"\\u{self._rtf_signed16(high_surrogate)}?")
            escaped.append(f"\\u{self._rtf_signed16(low_surrogate)}?")

        return "".join(escaped)

    def _active_column_headers(self) -> list[str]:
        """Return logbook-export column headers.

        The browser shows comments inline under the title, but the export
        keeps Comment as its own column after the base columns.
        """
        headers = list(self._COLUMNS) + ["Comment"]
        headers.extend(
            self._extra_column_header(column) for column in self._visible_extra_columns()
        )
        return headers

    def _export_sections(self) -> list[tuple[str, list[int]]]:
        """Build export sections in display order with group headers."""
        sections: list[tuple[str, list[int]]] = []
        ungrouped_runs: list[int] = []

        for entry in self._display_order:
            if isinstance(entry, str):
                group = self._groups.get(entry)
                if group is None:
                    continue
                members = [int(rn) for rn in group.member_run_numbers if int(rn) in self._datasets]
                sections.append((group.name, members))
                continue

            if entry in self._datasets:
                ungrouped_runs.append(int(entry))

        if ungrouped_runs or not sections:
            sections.append(("Ungrouped", ungrouped_runs))

        return sections

    def _export_row_values(self, run_number: int, dataset: MuonDataset) -> list[str]:
        """Return exported row values for one dataset in active-column order."""
        meta = dataset.metadata
        run_display = str(dataset.run_label)
        if run_number in self._combined_datasets:
            run_display = self._combined_run_display(run_number)

        row = [
            run_display,
            str(meta.get("title", "")),
            f"{self._temperature_for_display(dataset):.2f}",
            f"{float(meta.get('field', 0.0)):.1f}",
            str(meta.get("comment", "")),
        ]
        for column in self._visible_extra_columns():
            row.append(self._value_for_extra_column(dataset, column))
        return row

    # ------------------------------------------------------------------
    # Editing and removal
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table:
            return

        custom_column_id = item.data(_CUSTOM_COLUMN_ROLE)
        if isinstance(custom_column_id, str):
            self._on_custom_column_edited(item, custom_column_id)
            return

        if item.column() != 3:
            return

        row = item.row()
        run_item = self._table.item(row, 0)
        if run_item is None:
            return

        run_number = run_item.data(self._GROUP_ROLE)
        if not isinstance(run_number, int):
            return

        dataset = self._datasets.get(run_number)
        if dataset is None:
            return

        text = item.text().strip()
        try:
            field_value = float(text.split()[0]) if text else 0.0
        except ValueError:
            self._updating_table = True
            item.setText(f"{float(dataset.metadata.get('field', 0.0)):.1f}")
            self._updating_table = False
            return

        dataset.metadata["field"] = field_value
        if dataset.run is not None:
            dataset.run.metadata["field"] = field_value

        self._updating_table = True
        item.setText(f"{field_value:.1f}")
        self._updating_table = False

    def _on_custom_column_edited(self, item: QTableWidgetItem, column_id: str) -> None:
        """Persist a user edit to a custom-column cell into the dataset metadata."""
        run_item = self._table.item(item.row(), 0)
        if run_item is None:
            return
        run_number = run_item.data(self._GROUP_ROLE)
        if not isinstance(run_number, int):
            return
        dataset = self._datasets.get(run_number)
        if dataset is None:
            return
        text = item.text().strip()
        # The Angle field is numeric degrees: reject a non-numeric (non-blank)
        # entry, reverting the cell to its stored value with a brief warning.
        # Blank clears; any real number is accepted (wrapping is applied downstream).
        column = self._find_extra_column(column_id)
        if column is not None and column.is_angle and text:
            try:
                valid = math.isfinite(float(text))
            except ValueError:
                valid = False
            if not valid:
                # Reject non-numeric and non-finite (inf/nan) input alike: a
                # non-finite "angle" would poison the trend x-axis downstream.
                with QSignalBlocker(self._table):
                    item.setText(self.custom_column_value(dataset, column_id))
                QToolTip.showText(QCursor.pos(), "Angle must be a finite number (degrees)")
                return
        # Free-text by design (numbers are inferred only where consumed, e.g. the
        # trend x-axis): store the trimmed text verbatim, clearing on empty.
        self._set_custom_column_value(dataset, column_id, text)
        # Notify consumers (plot legend labels, trend x-axis) so an edit made
        # *after* a batch fit re-links live into existing results rather than
        # silently trending as all-NaN ("N/N skipped") until the batch is re-run.
        self._notify_extra_columns_changed()

    def _remove_run_number(self, run_number: int) -> None:
        self._datasets.pop(run_number, None)
        self._combined_datasets.pop(run_number, None)
        self._combined_source_datasets.pop(run_number, None)
        self._combined_signs.pop(run_number, None)
        self._combined_methods.pop(run_number, None)
        self._temperature_from_log_overrides.pop(int(run_number), None)
        self._field_from_log_overrides.pop(int(run_number), None)

        gid = self._run_to_group.get(run_number)
        if gid is not None:
            self._remove_run_from_group(run_number, gid)
        if run_number in self._display_order:
            self._display_order.remove(run_number)

    def _remove_selected_entries(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return

        selected_group_ids = [
            gid for gid in (self._group_id_from_key(k) for k in keys) if gid is not None
        ]
        for gid in selected_group_ids:
            self.ungroup(gid)

        for run_number in self._dataset_run_numbers_from_keys(keys):
            self._remove_run_number(run_number)

        self._rebuild_table()
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _create_table_context_menu(self) -> QMenu | None:
        keys = self._selected_keys()
        if not keys:
            return None

        menu = QMenu(self)
        selected_runs = [k for k in keys if isinstance(k, int)]
        selected_group_ids = [
            gid for gid in (self._group_id_from_key(k) for k in keys) if gid is not None
        ]
        expanded_selected_runs = self._dataset_run_numbers_from_keys(keys)
        grouped_selected_runs = [
            rn for rn in selected_runs if self._run_to_group.get(rn) is not None
        ]

        regular_runs = [rn for rn in expanded_selected_runs if rn not in self._combined_datasets]
        combined_runs = [rn for rn in expanded_selected_runs if rn in self._combined_datasets]

        if len(regular_runs) >= 2 and not combined_runs:
            menu.addAction("Co-add Selected", self._coadd_selected)
            menu.addAction("Subtract Selected (signed)…", self._signed_subtract_selected)
            menu.addAction("Re-fit as Co-added", self._emit_refit_coadded)
        if len(expanded_selected_runs) >= 2 and not selected_group_ids:
            menu.addAction("Form Data Group", self._form_data_group)
        if len(selected_runs) == 1 and not selected_group_ids:
            selected_run = selected_runs[0]
            menu.addAction("Get Info", lambda rn=selected_run: self.get_info_requested.emit(rn))
            dataset = self._datasets.get(selected_run)
            if (
                dataset is not None
                and dataset.run is not None
                and dataset.run.histograms
                and selected_run not in self._combined_datasets
            ):
                menu.addAction(
                    "Degrade Statistics…",
                    lambda rn=selected_run: self._on_degrade_statistics(rn),
                )
                if self._reference_subtraction_candidates(selected_run):
                    menu.addAction(
                        "Subtract Reference Run…",
                        lambda rn=selected_run: self._subtract_reference_run(rn),
                    )

        if combined_runs:
            menu.addAction("Separate Combined", self._separate_combined)

        if selected_runs and self._groups:
            send_menu = menu.addMenu("Send to Group")
            self._populate_send_to_group_menu(send_menu, selected_runs)

        if grouped_selected_runs:
            label = "Remove from Group" if len(grouped_selected_runs) == 1 else "Remove from Groups"
            menu.addAction(label, self._remove_selected_from_group)

        if len(selected_group_ids) == 1 and len(keys) == 1:
            gid = selected_group_ids[0]
            group = self._groups.get(gid)
            if group is not None:
                collapse_text = "Expand Group" if group.collapsed else "Collapse Group"
                menu.addAction(collapse_text, lambda gid=gid: self._toggle_group_collapsed(gid))
                menu.addAction("Rename Group", lambda gid=gid: self._rename_group(gid))
                menu.addAction("Ungroup", lambda gid=gid: self.ungroup(gid))
                menu.addSeparator()

        label = "Remove Entry" if len(keys) == 1 else "Remove Selected Entries"
        menu.addAction(label, self._remove_selected_entries)
        return menu

    def _populate_send_to_group_menu(self, send_menu: QMenu, selected_runs: list[int]) -> None:
        """Populate Send-to-Group submenu with current groups."""
        groups = sorted(self._groups.values(), key=lambda g: g.name.lower())
        if not groups:
            action = send_menu.addAction("(No groups)")
            action.setEnabled(False)
            return

        for group in groups:
            action = send_menu.addAction(group.name)
            action.triggered.connect(
                lambda _checked=False, gid=group.group_id, runs=list(selected_runs): (
                    self.add_runs_to_group(runs, gid)
                )
            )

    def _remove_selected_from_group(self) -> None:
        run_numbers = [
            rn for rn in self._get_selected_run_numbers() if self._run_to_group.get(rn) is not None
        ]
        if not run_numbers:
            return
        self.remove_runs_from_group(run_numbers)

    def _on_degrade_statistics(self, run_number: int) -> None:
        """Prompt for a factor + seed and add the thinned run beside the source."""
        from asymmetry.gui.windows.simulate_dialog import DegradeStatisticsDialog

        dialog = DegradeStatisticsDialog(self)
        if not dialog.exec():
            return
        self.apply_degrade_statistics(run_number, dialog.factor(), dialog.seed())

    def apply_degrade_statistics(
        self, run_number: int, factor: float, seed: int
    ) -> MuonDataset | None:
        """Resample *run_number*'s histograms by *factor*; add the derived run.

        Returns the new browser dataset, or ``None`` when the source has no
        detector histograms or the factor is invalid. The source run is never
        modified (WiMDA degraded in place; see docs/porting/simulate-mode/).
        """
        from asymmetry.core.simulate import degrade_run, reduce_run_to_dataset

        dataset = self._datasets.get(run_number)
        run = dataset.run if dataset is not None else None
        if run is None or not run.histograms:
            QMessageBox.warning(
                self,
                "Degrade Statistics",
                "This entry has no detector histograms to resample.",
            )
            return None
        try:
            derived = degrade_run(run, factor, seed=seed, run_number=self.next_derived_run_number())
            reduced = reduce_run_to_dataset(derived)
        except ValueError as exc:
            QMessageBox.warning(self, "Degrade Statistics", str(exc))
            return None
        self.add_dataset(reduced)
        return reduced

    def _show_table_context_menu(self, position: QPoint) -> None:
        viewport_pos = position
        item = self._table.itemAt(viewport_pos)
        if item is None:
            return

        row = item.row()
        selected_rows = {idx.row() for idx in self._table.selectedIndexes()}
        if row not in selected_rows:
            self._table.selectRow(row)

        menu = self._create_table_context_menu()
        if menu is None:
            return

        global_pos = self._table.viewport().mapToGlobal(viewport_pos)
        menu.popup(global_pos)

    def _form_data_group(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        if len(run_numbers) < 2:
            return

        default_name = self._default_group_name(run_numbers)
        name, ok = QInputDialog.getText(self, "Form Data Group", "Group name:", text=default_name)
        if not ok:
            return
        group_name = name.strip() or default_name
        self.create_data_group(run_numbers, name=group_name)

    def _toggle_group_collapsed(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        group.collapsed = not group.collapsed
        self._rebuild_table()

    def _rename_group(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Data Group", "Group name:", text=group.name)
        if not ok:
            return
        new_name = name.strip()
        if not new_name:
            return
        group.name = new_name
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Event filter, selection, sorting, filtering
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):  # noqa: N802
        header = self._table.horizontalHeader()

        if watched is self._table and event.type() == QEvent.Type.KeyPress:
            if (
                event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and self._table.state() != QAbstractItemView.State.EditingState
            ):
                self._remove_selected_entries()
                return True

            if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                current_row = self._table.currentRow()
                if current_row < 0:
                    return False

                direction = -1 if event.key() == Qt.Key.Key_Up else 1
                target_row = self._next_visible_row(current_row, direction)
                if target_row is None:
                    return True

                add_to_selection = bool(
                    event.modifiers()
                    & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
                )
                anchor_row = self._selection_anchor_for_row(current_row)
                if self._select_visible_row_range(
                    anchor_row,
                    target_row,
                    add_to_selection=add_to_selection,
                ):
                    self._selection_anchor_row = anchor_row
                    return True

        if watched is self._table.viewport():
            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                item = self._table.itemAt(event.position().toPoint())
                if item is not None:
                    gid = self._group_id_from_key(item.data(self._GROUP_ROLE))
                    if gid is not None:
                        _chevron_right = self._table.columnViewportPosition(0) + 20
                        if event.position().toPoint().x() <= _chevron_right:
                            return True  # single-click already toggled; absorb silently
                        self._toggle_group_collapsed(gid)
                        return True

            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                row = self._table.rowAt(event.position().toPoint().y())
                if row >= 0:
                    modifiers = event.modifiers()
                    index = self._table.model().index(row, 0)
                    selection_model = self._table.selectionModel()

                    # Chevron single-click: toggle collapse without changing selection
                    _item = self._table.item(row, 0)
                    if _item is not None:
                        _gid = self._group_id_from_key(_item.data(self._GROUP_ROLE))
                        if _gid is not None:
                            _chevron_right = self._table.columnViewportPosition(0) + 20
                            if event.position().toPoint().x() <= _chevron_right:
                                self._toggle_group_collapsed(_gid)
                                return True

                    if bool(modifiers & Qt.KeyboardModifier.ShiftModifier):
                        anchor_row = self._selection_anchor_for_row(row)

                        if selection_model is not None:
                            add_to_selection = bool(
                                modifiers
                                & (
                                    Qt.KeyboardModifier.ControlModifier
                                    | Qt.KeyboardModifier.MetaModifier
                                )
                            )
                            if self._select_visible_row_range(
                                anchor_row,
                                row,
                                add_to_selection=add_to_selection,
                            ):
                                self._selection_anchor_row = anchor_row
                                return True
                    else:
                        self._selection_anchor_row = row
                        if selection_model is not None:
                            selection_model.setCurrentIndex(
                                index, QItemSelectionModel.SelectionFlag.NoUpdate
                            )

        if watched is header.viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                pos = event.position().toPoint()
                logical_index = header.logicalIndexAt(pos)
                if logical_index < 0:
                    return False
                if event.button() == Qt.MouseButton.LeftButton:
                    self._on_header_clicked(logical_index)
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    QTimer.singleShot(
                        0, lambda ci=logical_index: self._open_header_context_menu(ci)
                    )
                    return True
        return super().eventFilter(watched, event)

    def _open_header_context_menu(self, col_idx: int) -> None:
        """Right-click header menu: filter (base) / remove (extra) / add column."""
        if col_idx < 0:
            return

        menu = QMenu(self)
        if col_idx < len(self._COLUMNS):
            menu.addAction("Filter…", lambda ci=col_idx: self._open_filter_dialog(ci))
        else:
            extra_index = col_idx - len(self._COLUMNS)
            visible_extra_columns = self._visible_extra_columns()
            if 0 <= extra_index < len(visible_extra_columns):
                column = visible_extra_columns[extra_index]
                menu.addAction(
                    "Rename…",
                    lambda cid=column.id: self._prompt_rename_extra_column(cid),
                )
                remove_label = "Delete column" if column.is_custom else "Remove from Data Browser"
                menu.addAction(
                    remove_label,
                    lambda cid=column.id: self.remove_extra_column(cid),
                )

        self._append_add_column_menu(menu)
        if not menu.isEmpty():
            menu.exec(self.cursor().pos())

    def _prompt_rename_extra_column(self, column_id: str) -> None:
        """Ask for a new gui-facing label for an extra column and apply it.

        For a metadata column the underlying NeXus/metadata source key is kept and
        shown so the user always knows which field they renamed.
        """
        column = self._find_extra_column(column_id)
        if column is None:
            return
        prompt = "New column name:"
        if not column.is_custom and column.source_key:
            prompt = f"New name for '{column.source_key}':"
        new_label, ok = QInputDialog.getText(self, "Rename column", prompt, text=column.label)
        if ok:
            self.rename_extra_column(column_id, new_label)

    def _append_add_column_menu(self, menu: QMenu) -> None:
        """Append an "Add column…" submenu of hideable run-quality columns.

        The browser previously had no end-user way to *add* a metadata column
        (only removal via this menu). This lists the ``run_info.*`` run-quality
        fields — including the good-range events and events/frame columns — that
        are not already shown, mirroring the Remove path.
        """
        available = self._addable_run_info_columns()
        if not available:
            return
        if not menu.isEmpty():
            menu.addSeparator()
        submenu = menu.addMenu("Add column…")
        for key in available:
            label = self._RUN_INFO_FIELD_LABELS.get(key, key)
            submenu.addAction(label, lambda fk=key: self.add_extra_column(fk))

    def _addable_run_info_columns(self) -> list[str]:
        """``run_info.*`` run-quality columns not currently shown."""
        shown_source_keys = {c.source_key for c in self._extra_columns}
        return [
            key
            for key in self._RUN_INFO_FIELD_LABELS
            if key.startswith("run_info.") and key not in shown_source_keys
        ]

    def _on_header_clicked(self, logical_index: int) -> None:
        if logical_index == self._current_sort_column:
            self._current_sort_order = (
                Qt.SortOrder.DescendingOrder
                if self._current_sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )
        else:
            self._current_sort_column = logical_index
            self._current_sort_order = Qt.SortOrder.AscendingOrder
        self._sort_table()

    def _sort_table(self, *, rebuild: bool = True) -> None:
        if self._current_sort_column < 0:
            return
        if self._batch_depth:
            self._batch_sort_pending = True
            if rebuild:
                self._batch_rebuild_pending = True
            return

        reverse = self._current_sort_order == Qt.SortOrder.DescendingOrder

        def _sort_key(run_number: int):
            dataset = self._datasets.get(run_number)
            if dataset is None:
                return ""
            meta = dataset.metadata
            if self._current_sort_column == 0:
                return run_number
            if self._current_sort_column == 1:
                return str(meta.get("title", ""))
            if self._current_sort_column == 2:
                return self._temperature_for_display(dataset)
            if self._current_sort_column == 3:
                return float(meta.get("field", 0.0))
            if self._current_sort_column >= len(self._COLUMNS):
                idx = self._current_sort_column - len(self._COLUMNS)
                visible_extra_columns = self._visible_extra_columns()
                if idx < 0 or idx >= len(visible_extra_columns):
                    return ""
                value = self._raw_value_for_column(dataset, visible_extra_columns[idx])
                # Return a type-ranked key so a column with a *mix* of numeric and
                # text/blank values (the norm for a custom column: empty by default
                # with the odd number typed in) never compares float against str —
                # which would raise TypeError mid-sort. Numerics sort first (rank
                # 0), text/blank after (rank 1); the two ranks never cross-compare.
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return (0, float(value))
                if isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value)
                    if arr.size and np.issubdtype(arr.dtype, np.number):
                        return (0, float(np.nanmean(arr.astype(np.float64))))
                return (1, "" if value is None else str(value))
            return str(meta.get("comment", ""))

        runs = [entry for entry in self._display_order if isinstance(entry, int)]
        sorted_runs = sorted(runs, key=_sort_key, reverse=reverse)

        if self._groups:
            groups = [
                entry
                for entry in self._display_order
                if isinstance(entry, str) and entry in self._groups
            ]
            for gid in groups:
                group = self._groups[gid]
                group.member_run_numbers = sorted(
                    group.member_run_numbers, key=_sort_key, reverse=reverse
                )
            self._display_order = groups + sorted_runs
        else:
            self._display_order = sorted_runs

        self._table.horizontalHeader().setSortIndicatorShown(self._current_sort_column >= 0)
        self._table.horizontalHeader().setSortIndicator(
            self._current_sort_column, self._current_sort_order
        )
        if rebuild:
            self._rebuild_table()

    def _open_filter_dialog(self, col_idx: int) -> None:
        unique_values = set()
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            if self._is_group_key(run_item.data(self._GROUP_ROLE)):
                continue
            item = self._table.item(row, col_idx)
            if item:
                unique_values.add(item.text().strip())

        header_item = self._table.horizontalHeaderItem(col_idx)
        column_name = header_item.text() if header_item is not None else str(col_idx)

        dialog = FilterDialog(
            column_name,
            sorted(unique_values),
            self._column_filters.get(col_idx),
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_values = dialog.get_selected_values()
            if selected_values is None:
                self._column_filters.pop(col_idx, None)
            else:
                self._column_filters[col_idx] = selected_values
            self._apply_row_visibility()

    def _row_visible_by_filters(self, row: int) -> bool:
        if not self._column_filters:
            return True
        for col_idx, allowed in self._column_filters.items():
            item = self._table.item(row, col_idx)
            if item and item.text().strip() not in allowed:
                return False
        return True

    def _apply_row_visibility(self) -> None:
        for row in range(self._table.rowCount()):
            self._table.setRowHidden(row, False)

        if not self._column_filters:
            # still need to apply collapsed state
            for row in range(self._table.rowCount()):
                run_item = self._table.item(row, 0)
                if run_item is None:
                    continue
                key = run_item.data(self._GROUP_ROLE)
                if isinstance(key, int):
                    gid = self._run_to_group.get(key)
                    if (
                        gid is not None
                        and self._groups.get(gid) is not None
                        and self._groups[gid].collapsed
                    ):
                        self._table.setRowHidden(row, True)
            self.selection_changed.emit()
            return

        # First pass: hide dataset rows not matching filter or collapsed by group.
        group_has_visible: dict[str, bool] = {gid: False for gid in self._groups}
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            key = run_item.data(self._GROUP_ROLE)
            if self._is_group_key(key):
                continue

            visible = self._row_visible_by_filters(row)
            if isinstance(key, int):
                gid = self._run_to_group.get(key)
                if (
                    gid is not None
                    and self._groups.get(gid) is not None
                    and self._groups[gid].collapsed
                ):
                    visible = False
                if gid is not None and visible:
                    group_has_visible[gid] = True
            self._table.setRowHidden(row, not visible)

        # Second pass: hide group rows when all children filtered out.
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            gid = self._group_id_from_key(run_item.data(self._GROUP_ROLE))
            if gid is None:
                continue
            self._table.setRowHidden(row, not group_has_visible.get(gid, False))

        if not self._is_row_visible_for_selection(self._selection_anchor_row or -1):
            self._selection_anchor_row = None

        self.selection_changed.emit()

    def _on_selection_changed(self) -> None:
        selected_datasets = self.get_selected_datasets()
        selected_group_ids = self._get_selected_group_ids()

        self.selection_changed.emit()

        if len(selected_group_ids) == 1 and len(self._selected_keys()) == 1:
            self.group_selected.emit(selected_group_ids[0])
            return

        if len(selected_datasets) == 1:
            self.dataset_selected.emit(selected_datasets[0].run_number)

    # ------------------------------------------------------------------
    # Co-add and separate
    # ------------------------------------------------------------------

    def _emit_refit_coadded(self) -> None:
        """Ask the host to combine the selection and re-fit it as one run."""
        runs = [rn for rn in self._get_selected_run_numbers() if rn not in self._combined_datasets]
        if len(runs) >= 2:
            self.refit_coadded_requested.emit(list(runs))

    def _coadd_selected(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        if len(run_numbers) < 2:
            return

        datasets_to_combine: list[MuonDataset] = []
        for rn in run_numbers:
            if rn in self._combined_datasets:
                return
            dataset = self._datasets.get(rn)
            if dataset:
                datasets_to_combine.append(dataset)

        if len(datasets_to_combine) < 2:
            return

        incompatibility = self._coadd_compatibility_error(datasets_to_combine)
        if incompatibility is not None:
            QMessageBox.warning(self, "Cannot Co-add Selected Datasets", incompatibility)
            return

        from asymmetry.core.data.combine import CombineError

        insert_index = min(self._display_index_for_run(rn) for rn in run_numbers)
        combined_rn = self._next_combined_id
        source_datasets = [self._datasets[rn] for rn in run_numbers if rn in self._datasets]
        try:
            combined_dataset = self._coadd_datasets(
                source_datasets,
                run_numbers,
                combined_run_number=combined_rn,
            )
        except CombineError as exc:
            QMessageBox.warning(self, "Cannot Co-add Selected Datasets", str(exc))
            return

        self._next_combined_id -= 1
        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = list(run_numbers)
        self._combined_source_datasets[combined_rn] = source_datasets

        for rn in run_numbers:
            self._remove_run_number(rn)

        self._display_order.insert(insert_index, combined_rn)
        self._rebuild_table()
        self.select_runs({combined_rn})

    def _subtract_reference_run(self, sample_rn: int) -> None:
        """Subtract a chosen reference run from *sample_rn* (study RA3/RA4).

        Opens a picker of the other loaded runs; the difference becomes a new
        combined row (sample − reference) hiding both constituents, restorable
        with "Separate Combined".
        """
        sample = self._datasets.get(sample_rn)
        if sample is None or sample.run is None or not sample.run.histograms:
            return

        candidates = self._reference_subtraction_candidates(sample_rn)
        if not candidates:
            QMessageBox.information(
                self,
                "Subtract Reference Run",
                "Load another run (with histograms) to subtract as a reference.",
            )
            return

        reference_rn = self._prompt_reference_run(sample_rn, candidates)
        if reference_rn is None:
            return

        from asymmetry.core.data.combine import CombineError

        run_numbers = [int(sample_rn), int(reference_rn)]
        source_datasets = [self._datasets[sample_rn], self._datasets[reference_rn]]
        insert_index = self._display_index_for_run(sample_rn)
        combined_rn = self._next_combined_id
        try:
            combined_dataset = self._subtract_datasets(
                source_datasets,
                run_numbers,
                combined_run_number=combined_rn,
            )
        except CombineError as exc:
            QMessageBox.warning(self, "Cannot Subtract Reference Run", str(exc))
            return

        self._next_combined_id -= 1
        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = run_numbers
        self._combined_source_datasets[combined_rn] = source_datasets
        self._combined_signs[combined_rn] = -1

        for rn in run_numbers:
            self._remove_run_number(rn)

        self._display_order.insert(insert_index, combined_rn)
        self._rebuild_table()
        self.select_runs({combined_rn})

    def _signed_subtract_selected(self) -> None:
        """Symmetric N-run signed co-subtract of the selected runs (sample − rest).

        Opens a small dialog to pick the sample (positive) run; every other
        selected run is subtracted at unit scale. The difference becomes a
        combined row hiding all constituents, restorable with "Separate
        Combined".
        """
        run_numbers = [rn for rn in self._get_selected_run_numbers()]
        regular = [rn for rn in run_numbers if rn not in self._combined_datasets]
        if len(regular) < 2:
            return
        for rn in regular:
            dataset = self._datasets.get(rn)
            if dataset is None or dataset.run is None or not dataset.run.histograms:
                QMessageBox.warning(
                    self,
                    "Cannot Subtract Selected Runs",
                    "Every selected run needs detector histograms to co-subtract.",
                )
                return

        ordered = self._prompt_signed_subtract(regular)
        if ordered is None:
            return

        from asymmetry.core.data.combine import CombineError

        source_datasets = [self._datasets[rn] for rn in ordered]
        insert_index = min(self._display_index_for_run(rn) for rn in ordered)
        combined_rn = self._next_combined_id
        try:
            combined_dataset = self._signed_subtract_datasets(
                source_datasets,
                ordered,
                combined_run_number=combined_rn,
            )
        except CombineError as exc:
            QMessageBox.warning(self, "Cannot Subtract Selected Runs", str(exc))
            return

        self._next_combined_id -= 1
        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = ordered
        self._combined_source_datasets[combined_rn] = source_datasets
        self._combined_signs[combined_rn] = -1
        self._combined_methods[combined_rn] = "subtract_signed"

        for rn in ordered:
            self._remove_run_number(rn)

        self._display_order.insert(insert_index, combined_rn)
        self._rebuild_table()
        self.select_runs({combined_rn})

    def _prompt_signed_subtract(self, run_numbers: list[int]) -> list[int] | None:
        """Pick the sample (positive) run; return [sample, *others] or None.

        Others keep their ascending order so the displayed formula is stable.
        """
        ordered_runs = sorted(run_numbers)
        dialog = QDialog(self)
        dialog.setWindowTitle("Subtract Selected Runs")
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Symmetric signed co-subtract: the sample run minus every other\n"
                "selected run (unit scale). Counts subtract bin-by-bin; each run's\n"
                "Poisson errors add in quadrature."
            )
        )
        layout.addWidget(QLabel("Sample (positive) run:"))
        combo = QComboBox(dialog)
        for rn in ordered_runs:
            combo.addItem(self._datasets[rn].run_label, rn)
        layout.addWidget(combo)
        preview = QLabel()
        layout.addWidget(preview)

        def _update_preview() -> None:
            sample = int(combo.currentData())
            rest = [rn for rn in ordered_runs if rn != sample]
            labels = [self._datasets[sample].run_label] + [
                self._datasets[r].run_label for r in rest
            ]
            preview.setText("Result:  " + " − ".join(labels))

        combo.currentIndexChanged.connect(_update_preview)
        _update_preview()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        sample = int(combo.currentData())
        return [sample] + [rn for rn in ordered_runs if rn != sample]

    def _reference_subtraction_candidates(self, sample_rn: int) -> list[int]:
        """Loaded, non-combined runs (with histograms) usable as a reference."""
        return [
            rn
            for rn in self._datasets
            if rn != sample_rn
            and rn not in self._combined_datasets
            and self._datasets[rn].run is not None
            and self._datasets[rn].run.histograms
        ]

    def _prompt_reference_run(self, sample_rn: int, candidates: list[int]) -> int | None:
        """Modal picker returning the chosen reference run number (or None)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Subtract Reference Run")
        layout = QVBoxLayout(dialog)
        sample_label = self._datasets[sample_rn].run_label
        layout.addWidget(
            QLabel(
                f"Subtract a frame-scaled reference run from run {sample_label}.\n"
                "Counts are subtracted bin-by-bin; errors add in quadrature."
            )
        )
        combo = QComboBox(dialog)
        for rn in candidates:
            combo.addItem(self._datasets[rn].run_label, rn)
        layout.addWidget(combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return int(combo.currentData())

    def _coadd_datasets(
        self,
        datasets: list[MuonDataset],
        run_numbers: list[int],
        *,
        combined_run_number: int,
        existing_dataset: MuonDataset | None = None,
    ) -> MuonDataset:
        """Co-add source datasets at the raw-count level via ``combine_runs``.

        Replaces the former curve-mean co-add (statistically wrong at low
        counts, and it discarded histograms). The combined dataset now carries
        real summed histograms, so it can be regrouped, deadtime-corrected,
        count-fitted and transformed like any run. Combined results change
        numerically — this is the correctness fix (study RA1/RA2/RA8).
        """
        from asymmetry.core.data.combine import combine_runs, reduce_combined_run

        runs = self._runs_for_combine(datasets)
        combined_run = combine_runs(
            runs,
            sign=1,
            run_number=combined_run_number,
            label=" + ".join(map(str, run_numbers)),
        )
        reduced = reduce_combined_run(combined_run)
        return self._store_combined_reduction(reduced, existing_dataset)

    def _subtract_datasets(
        self,
        datasets: list[MuonDataset],
        run_numbers: list[int],
        *,
        combined_run_number: int,
        existing_dataset: MuonDataset | None = None,
    ) -> MuonDataset:
        """Subtract a frame-scaled reference run from a sample (study RA3/RA4).

        ``datasets`` is ``[sample, reference]``. The reference is resolved and
        frame-scaled through the single reference-run home
        (:func:`asymmetry.core.io.resolve_background_reference`, F9) and the
        per-detector arithmetic runs through ``subtract_scaled_counts`` inside
        ``combine_runs`` — no parallel subtraction path.
        """
        from asymmetry.core.data.combine import combine_runs, reduce_combined_run
        from asymmetry.core.io import resolve_background_reference
        from asymmetry.core.transform.grouping import good_frames

        runs = self._runs_for_combine(datasets)
        sample_run, reference_run = runs
        sample_frames = good_frames(sample_run.grouping, 0.0) or None
        # Route reference resolution + frame scale through the shared home so a
        # reference subtraction uses exactly the background path's exposure
        # scale (sample/reference good frames).
        payload = {
            "run_number": int(reference_run.run_number),
            "source_file": reference_run.source_file,
            "good_frames_reference": good_frames(reference_run.grouping, 0.0) or None,
        }
        try:
            resolved = resolve_background_reference(
                payload,
                sample_good_frames=sample_frames,
                datasets=[datasets[1]],
            )
            scale = float(resolved.scale)
        except (ValueError, OSError):
            # Fall back to the direct frame ratio when resolution is unavailable
            # (both runs are already in hand, so this is always computable).
            ref_frames = good_frames(reference_run.grouping, 0.0)
            scale = (sample_frames / ref_frames) if (sample_frames and ref_frames) else 1.0

        combined_run = combine_runs(
            runs,
            sign=-1,
            scales=[1.0, scale],
            run_number=combined_run_number,
            label=" − ".join(map(str, run_numbers)),
        )
        reduced = reduce_combined_run(combined_run)
        return self._store_combined_reduction(reduced, existing_dataset)

    def _signed_subtract_datasets(
        self,
        datasets: list[MuonDataset],
        run_numbers: list[int],
        *,
        combined_run_number: int,
        existing_dataset: MuonDataset | None = None,
    ) -> MuonDataset:
        """Symmetric N-run signed co-subtract ``runs[0] − Σ runs[k≥1]``.

        ``datasets[0]`` is the sample (positive term); every other selected run
        is subtracted at unit scale, each contributing its own Poisson variance.
        The per-detector arithmetic runs through ``subtract_scaled_counts``
        inside ``combine_runs`` (``subtract_method="signed"``, F9). Unlike the
        reference path this takes two *or more* runs and applies no frame scaling
        — for photo-µSR laser-on/off and background-style differences.
        """
        from asymmetry.core.data.combine import combine_runs, reduce_combined_run

        runs = self._runs_for_combine(datasets)
        combined_run = combine_runs(
            runs,
            sign=-1,
            subtract_method="signed",
            scales=[1.0] * len(runs),
            run_number=combined_run_number,
            label=" − ".join(map(str, run_numbers)),
        )
        reduced = reduce_combined_run(combined_run)
        return self._store_combined_reduction(reduced, existing_dataset)

    def _runs_for_combine(self, datasets: list[MuonDataset]) -> list:
        """Shallow run copies whose metadata reflects the browser's scalars.

        Thin wrapper over :func:`asymmetry.core.data.combine.runs_with_dataset_metadata`
        — the dataset metadata is the browser's source of truth for the displayed
        scalars (field overrides, from-log, …) that may not be on ``run.metadata``.
        """
        from asymmetry.core.data.combine import runs_with_dataset_metadata

        return runs_with_dataset_metadata(datasets)

    def _store_combined_reduction(
        self,
        reduced: MuonDataset,
        existing_dataset: MuonDataset | None,
    ) -> MuonDataset:
        """Place a freshly reduced combined dataset, reusing ``existing_dataset``.

        Rebuild paths (``.asymp`` load, regroup) hold a reference to the
        combined dataset; mutate it in place so those references stay valid.
        """
        if existing_dataset is None:
            return reduced
        existing_dataset.time = reduced.time
        existing_dataset.asymmetry = reduced.asymmetry
        existing_dataset.error = reduced.error
        existing_dataset.metadata = reduced.metadata
        existing_dataset.run = reduced.run
        if hasattr(existing_dataset, "_grouping_source_arrays_cache"):
            delattr(existing_dataset, "_grouping_source_arrays_cache")
        return existing_dataset

    def _separate_combined(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        combined_items = [rn for rn in run_numbers if rn in self._combined_datasets]
        if not combined_items:
            return

        restored_run_numbers: list[int] = []
        for rn in combined_items:
            insert_index = self._display_index_for_run(rn)
            source_datasets = self._combined_source_datasets.get(rn, [])
            restored_run_numbers.extend(int(ds.run_number) for ds in source_datasets)
            group_id = self._run_to_group.get(rn)
            group = self._groups.get(group_id) if group_id is not None else None

            self._datasets.pop(rn, None)
            self._combined_datasets.pop(rn, None)
            self._combined_source_datasets.pop(rn, None)
            self._combined_signs.pop(rn, None)
            self._combined_methods.pop(rn, None)
            if group is not None:
                try:
                    member_index = group.member_run_numbers.index(rn)
                except ValueError:
                    member_index = len(group.member_run_numbers)
                group.member_run_numbers = [
                    member for member in group.member_run_numbers if member != rn
                ]
                self._run_to_group.pop(rn, None)

                for offset, dataset in enumerate(source_datasets):
                    source_rn = int(dataset.run_number)
                    self._datasets[source_rn] = dataset
                    group.member_run_numbers.insert(member_index + offset, source_rn)
                    self._run_to_group[source_rn] = group.group_id
            else:
                self._run_to_group.pop(rn, None)
            if rn in self._display_order:
                self._display_order.remove(rn)

            if group is None:
                for offset, dataset in enumerate(source_datasets):
                    source_rn = int(dataset.run_number)
                    self._datasets[source_rn] = dataset
                    if source_rn not in self._display_order:
                        self._display_order.insert(insert_index + offset, source_rn)

        self._rebuild_table()
        if restored_run_numbers:
            self.select_runs(set(restored_run_numbers))

    # ------------------------------------------------------------------
    # Project state
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._datasets.clear()
        self._combined_datasets.clear()
        self._combined_source_datasets.clear()
        self._combined_signs.clear()
        self._combined_methods.clear()
        self._next_combined_id = -1
        self._groups.clear()
        self._run_to_group.clear()
        self._display_order.clear()
        self._column_filters.clear()
        self._extra_columns.clear()
        self._use_temperature_from_log = False
        self._temperature_from_log_overrides.clear()
        self._use_field_from_log = False
        self._field_from_log_overrides.clear()
        self._current_sort_column = -1
        self._current_sort_order = Qt.SortOrder.AscendingOrder
        # Clear series-highlight state so stale run numbers from the previous
        # project cannot tint rows in the next project.
        self._highlighted_runs = set()
        self._refresh_column_headers()
        self._table.setRowCount(0)

    def add_combined_dataset(
        self,
        source_run_numbers: list[int],
        *,
        sign: int = 1,
        operation: str | None = None,
    ) -> int | None:
        """Recreate a combined row programmatically (``.asymp`` load).

        ``sign=+1`` co-adds; ``sign=-1`` subtracts. ``operation`` disambiguates
        the subtractions: ``"subtract_signed"`` is the symmetric N-run signed
        co-subtract (sample − every other source), any other value (or ``None``
        with ``sign=-1``) is the two-run reference subtraction. Co-add requires
        identical grouping; the subtractions only need the count-level
        invariants (``combine_runs`` checks them), so the stricter grouping gate
        is skipped for them.
        """
        datasets_to_combine = []
        for rn in source_run_numbers:
            ds = self._datasets.get(rn)
            if ds is None:
                return None
            datasets_to_combine.append(ds)

        if len(datasets_to_combine) < 2:
            return None

        signed = sign == -1 and operation == "subtract_signed"
        if sign == 1:
            incompatibility = self._coadd_compatibility_error(datasets_to_combine)
            if incompatibility is not None:
                return None
        elif not signed and len(datasets_to_combine) != 2:
            # The reference subtraction is sample + one reference.
            return None

        from asymmetry.core.data.combine import CombineError

        if sign != -1:
            builder = self._coadd_datasets
        elif signed:
            builder = self._signed_subtract_datasets
        else:
            builder = self._subtract_datasets
        combined_rn = self._next_combined_id
        source_datasets = [self._datasets[rn] for rn in source_run_numbers if rn in self._datasets]
        try:
            combined_dataset = builder(
                source_datasets,
                source_run_numbers,
                combined_run_number=combined_rn,
            )
        except CombineError:
            return None

        self._next_combined_id -= 1
        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = source_run_numbers
        self._combined_source_datasets[combined_rn] = source_datasets
        if sign == -1:
            self._combined_signs[combined_rn] = -1
        if signed:
            self._combined_methods[combined_rn] = "subtract_signed"

        insert_index = min(self._display_index_for_run(rn) for rn in source_run_numbers)
        for rn in source_run_numbers:
            self._remove_run_number(rn)
        self._display_order.insert(insert_index, combined_rn)

        self._rebuild_table()
        return combined_rn

    def get_state(self) -> dict:
        filters = {str(col): sorted(values) for col, values in self._column_filters.items()}
        data_groups = [
            {
                "group_id": group.group_id,
                "name": group.name,
                "member_run_numbers": [int(rn) for rn in group.member_run_numbers],
                "collapsed": bool(group.collapsed),
            }
            for group in self._groups.values()
        ]
        selected_group_ids = self._get_selected_group_ids()
        return {
            # Version 2: the Comment column was removed (comments ride on the
            # Title cell), so indices >= 4 shifted down by one. restore_state
            # migrates version-1 indices.
            "column_layout": 2,
            "sort_column": self._current_sort_column,
            "sort_order": "ascending"
            if self._current_sort_order == Qt.SortOrder.AscendingOrder
            else "descending",
            "filters": filters,
            "selected_run_numbers": self._get_selected_run_numbers(),
            "selected_group_ids": selected_group_ids,
            "data_groups": data_groups,
            "extra_columns": [column.to_dict() for column in self._extra_columns],
            "use_temperature_from_log": bool(self._use_temperature_from_log),
            "temperature_from_log_overrides": {
                str(rn): bool(enabled)
                for rn, enabled in sorted(self._temperature_from_log_overrides.items())
            },
            "use_field_from_log": bool(self._use_field_from_log),
            "field_from_log_overrides": {
                str(rn): bool(enabled)
                for rn, enabled in sorted(self._field_from_log_overrides.items())
            },
        }

    def restore_state(self, state: dict) -> None:
        try:
            layout_version = int(state.get("column_layout", 1))
        except (TypeError, ValueError):
            layout_version = 1
        # Layout v1 had Comment as column 4; v2 removed it, shifting extras
        # down by one. Migrate legacy indices so old projects don't filter or
        # sort against the wrong column (a stale Comment filter would hide
        # every row).
        legacy_comment_col = 4

        self._column_filters = {}
        for col_str, values in state.get("filters", {}).items():
            col = int(col_str)
            if layout_version < 2:
                if col == legacy_comment_col:
                    continue
                if col > legacy_comment_col:
                    col -= 1
            self._column_filters[col] = set(values)

        sort_column = int(state.get("sort_column", -1))
        if layout_version < 2:
            if sort_column == legacy_comment_col:
                sort_column = -1
            elif sort_column > legacy_comment_col:
                sort_column -= 1
        self._current_sort_column = sort_column
        sort_order_str = state.get("sort_order", "ascending")
        self._current_sort_order = (
            Qt.SortOrder.AscendingOrder
            if sort_order_str == "ascending"
            else Qt.SortOrder.DescendingOrder
        )
        parsed_extra_columns = self._parse_saved_extra_columns(state.get("extra_columns", []))
        # The from-log pseudo-keys are *not* real columns; detect a legacy
        # "temperature" entry (a from-log request in old projects) before the
        # base-override filter strips it.
        saved_metadata_keys = {
            column.source_key for column in parsed_extra_columns if column.source_key
        }
        self._use_temperature_from_log = bool(
            state.get("use_temperature_from_log", "temperature" in saved_metadata_keys)
        )
        # Default OFF when the key is absent: unlike "temperature" (always a
        # from-log pseudo-key), older projects could save "field" as an ordinary
        # "Magnetic Field (G)" extra column, so a present "field" must not be
        # read as a request for field-from-log (which would silently switch the
        # B column to the log mean on open).
        self._use_field_from_log = bool(state.get("use_field_from_log", False))
        self._extra_columns = [
            column
            for column in parsed_extra_columns
            if column.is_custom or column.source_key not in self._BASE_COLUMN_OVERRIDE_KEYS
        ]
        self._temperature_from_log_overrides = {}
        for run_number, enabled in state.get("temperature_from_log_overrides", {}).items():
            try:
                rn = int(run_number)
            except (TypeError, ValueError):
                continue
            if rn in self._datasets:
                self._temperature_from_log_overrides[rn] = bool(enabled)
        self._field_from_log_overrides = {}
        for run_number, enabled in state.get("field_from_log_overrides", {}).items():
            try:
                rn = int(run_number)
            except (TypeError, ValueError):
                continue
            if rn in self._datasets:
                self._field_from_log_overrides[rn] = bool(enabled)
        self._refresh_column_headers()

        for group_entry in state.get("data_groups", []):
            if not isinstance(group_entry, dict):
                continue
            group_id = str(group_entry.get("group_id") or "")
            if not group_id:
                continue
            run_numbers = [
                int(v)
                for v in group_entry.get("member_run_numbers", [])
                if int(v) in self._datasets
            ]
            if len(run_numbers) < 2:
                continue
            self.create_data_group(
                run_numbers,
                name=str(group_entry.get("name") or "").strip() or None,
                group_id=group_id,
                collapsed=bool(group_entry.get("collapsed", False)),
            )

        self._sort_table(rebuild=False)
        self._move_groups_to_top()
        self._rebuild_table()

        selected_runs = set(state.get("selected_run_numbers", []))
        selected_group_ids = {str(v) for v in state.get("selected_group_ids", [])}

        keys: list[int | str] = list(selected_runs)
        keys.extend(f"{self._GROUP_SENTINEL_PREFIX}{gid}" for gid in selected_group_ids)
        self._restore_selection_by_keys(keys)

        # Let the host re-offer the restored custom columns as plot labels / trend
        # x-axes once project load has rebuilt them.
        self._notify_extra_columns_changed()
