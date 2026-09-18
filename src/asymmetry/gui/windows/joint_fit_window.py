"""The joint-fit window: compose recorded series, share parameters, run one fit.

A joint fit couples several :class:`~asymmetry.core.representation.series.FitSeries`
that the user has already made in the Batch tab — different models, different
runs — by holding named parameters equal across them
(``docs/plans/joint-fit.md``). This window is the only surface for that: an
undocked, menu-launched window owned by :class:`~asymmetry.gui.mainwindow.MainWindow`,
modelled on :class:`~asymmetry.gui.windows.global_parameter_fit_window.GlobalParameterFitWindow`.

It deliberately holds **no project model**. The host hands it two callables —
one that lists the active representation's series as plain
:class:`JointSeriesEntry` records, one that crops a series' member datasets to
that series' own recipe window — and gets back a launch context plus a result
on :attr:`JointFitWindow.joint_fit_completed`. Everything the window decides on
its own is decidable from the entries: which series may be ticked together (a
run may belong to exactly one member, D3), what the shared table can say (only
a series' Global parameters, D4), and what a shared row is seeded with (the
first contributing series' recipe row, D6).

The window edits **no recipe** (D2): per-series seeds, bounds, roles and fit
ranges belong to the Batch tab, and a joint run resolves them exactly as a solo
run of that series would — through
:func:`~asymmetry.gui.panels.fit.recipe_inputs.build_recipe_engine_inputs`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.joint import (
    JointSeriesProblem,
    SharedParameter,
    fit_joint,
    suggest_shared_parameters,
)
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.joint_fit import JointFit
from asymmetry.core.representation.naming import default_joint_fit_label
from asymmetry.gui.panels.fit.recipe_inputs import build_recipe_engine_inputs
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import apply_param_table_style
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.fit_results_card import FitCardSummary, FitResultsCard
from asymmetry.gui.widgets.fit_run_controls import FitRunControls
from asymmetry.gui.widgets.panel_section import PanelSection

__all__ = ["JointFitLaunch", "JointFitRun", "JointFitWindow", "JointSeriesEntry"]

#: The "no parameter from this series" entry of a shared row's per-series combo.
NO_MEMBER = "—"

#: Fixed columns of the shared table, before the one-per-ticked-series columns.
_SHARED_COLUMNS = ("Shared", "Value", "Min", "Max")

#: Shown in the Series section when the active representation is a frequency
#: one. ``fit_joint`` takes no ``error_oversampling``, and a zero-padded FFT
#: spectrum's samples are correlated — a frequency joint fit would report
#: uncertainties that are simply wrong, so v1 does not offer one.
FREQUENCY_NOTICE = "Joint fits are available for time-domain series only."


@dataclass(frozen=True)
class JointSeriesEntry:
    """One recorded series, as the joint-fit window lists it.

    Built by the host from the :class:`~asymmetry.core.representation.project_model.ProjectModel`
    so the window never holds it. ``members`` is the series' *live* effective
    membership (group members minus its own exclusions), which is what the
    overlap rule must be decided on — a stale snapshot would let two series
    that now share a run be ticked together. ``blocked_reason`` carries the
    reasons the host can see on its own (a detector-group series, a computed
    series with no model); the overlap reason is the window's to compute,
    because it depends on what is ticked.
    """

    batch_id: str
    label: str
    rep_type: RepresentationType
    model_text: str
    members: tuple[int, ...]
    status: str
    blocked_reason: str
    model: CompositeModel | None
    #: ``{parameter: "global"|"local"|"fixed"}`` — this series' roles (D4).
    roles: Mapping[str, str]
    #: This series' :attr:`FitSeries.recipe` (seeds, bounds, window) — D6.
    recipe: Mapping[str, Any]

    def global_params(self) -> list[str]:
        """The parameters this series may contribute to a shared row (D4)."""
        return [name for name, role in self.roles.items() if role == "global"]


@dataclass(frozen=True)
class JointFitLaunch:
    """The launch-time context a joint-fit completion is interpreted against.

    The joint fit lands after an arbitrary delay, by which time the user may
    have ticked a different series, retyped the label or edited the shared
    table. Every completion — the window's own rendering and the host's
    recording — reads what was *fitted* off this frozen record, exactly as
    :class:`~asymmetry.gui.panels.fit.global_tab.FitLaunch` does for a batch.
    """

    #: The joint fit being re-run, or ``None`` when this run creates one.
    joint_id: str | None
    label: str
    rep_type: RepresentationType
    #: Member series, in tick order — which is the seed order D6 refers to.
    member_batch_ids: tuple[str, ...]
    #: The shared table as :attr:`JointFit.shared` rows (name, members, bounds).
    shared_rows: tuple[dict[str, Any], ...]
    models: Mapping[str, CompositeModel]
    global_params: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class JointFitRun:
    """What the joint-fit worker hands back: the result *and* its drawn curves.

    ``fit_joint`` reports parameters, not curves, but a fitted curve is one
    model evaluation per run — and for a helical, Overhauser or dipolar
    component that is seconds over a wide series, which the GUI thread may not
    spend (see AGENTS.md, "Never run long work on the GUI thread"). So the
    worker evaluates them here, on the same cropped axes the fit used, and the
    completion handler only draws.
    """

    result: Any
    #: ``{series key: {run number: (t, y)}}`` — empty when the fit failed.
    curves: dict[str, dict[int, tuple[Any, Any]]]


def run_joint_fit_with_curves(
    problems: Sequence[JointSeriesProblem],
    shared: Sequence[SharedParameter],
    *,
    cancel_callback: Callable[[], bool],
) -> JointFitRun:
    """Fit, then evaluate each member run's fitted curve — all off the GUI thread.

    The whole body of the window's worker task. Kept module-level (rather than
    as a closure) so it is callable and testable on its own, and so the
    ``fit_joint`` it calls is the module global a test can substitute.
    """
    result = fit_joint(problems, shared, strategy="least_squares", cancel_callback=cancel_callback)
    curves: dict[str, dict[int, tuple[Any, Any]]] = {}
    if not result.success:
        # A failed fit records and draws nothing, so its parameters are not
        # worth evaluating — and may not even be finite.
        return JointFitRun(result=result, curves=curves)
    for problem in problems:
        per_run: dict[int, tuple[Any, Any]] = {}
        for dataset in problem.datasets:
            run_number = int(dataset.run_number)
            fit_result = result.series_results[problem.key][run_number]
            values = {p.name: p.value for p in fit_result.parameters}
            per_run[run_number] = (dataset.time, problem.model_fn(dataset.time, **values))
        curves[str(problem.key)] = per_run
    return JointFitRun(result=result, curves=curves)


@dataclass
class _SharedRow:
    """One row of the shared-parameter table, as the window holds it."""

    name: str
    #: ``{batch_id: that series' parameter name}``.
    members: dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    minimum: float = -math.inf
    maximum: float = math.inf
    ticked: bool = False
    #: ``"exact"``/``"candidate"`` from a suggestion, ``"user"`` when added by
    #: hand. With :attr:`user_edited` this decides what survives a re-suggest.
    tier: str = "user"
    user_edited: bool = False

    def survives_suggestion(self) -> bool:
        """Whether a re-run of "Suggest" keeps this row (D7)."""
        return self.tier == "user" or self.user_edited

    def to_record(self) -> dict[str, Any]:
        """This row in the persisted :attr:`JointFit.shared` shape."""
        return {
            "name": self.name,
            "members": dict(self.members),
            "value": float(self.value),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


def _format_bound(value: float) -> str:
    """A limit as table text; an infinite bound reads as the open ``-inf``/``inf``."""
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:g}"


def _parse_bound(text: str, unbounded: float) -> float:
    """Read a limit cell; anything unparseable means "no bound on this side"."""
    try:
        return float(text)
    except ValueError:
        return unbounded


class JointFitWindow(QMainWindow):
    """Compose recorded series into one coupled fit with shared parameters."""

    #: Emitted ``(JointFitLaunch, JointFitResult, curves)`` when a joint fit
    #: converges, where ``curves`` is ``{batch_id: {run: (t, y)}}`` already
    #: evaluated in the worker. The host records and draws it (D8) without
    #: touching a model; a failed fit is shown here and emits nothing.
    joint_fit_completed = Signal(object, object, object)
    #: Emitted (batch_id) from a per-series "Open in Batch tab" button.
    open_series_requested = Signal(str)
    #: Emitted (joint_id) once the user has confirmed "Delete joint fit…".
    joint_fit_delete_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Joint Fit")
        self.resize(900, 700)

        self._series_provider: Callable[[], list[JointSeriesEntry]] = list
        self._datasets_provider: Callable[[str], list[MuonDataset]] = lambda _bid: []

        self._entries: list[JointSeriesEntry] = []
        #: Ticked member ids in tick order — the joint fit's member order (D6).
        self._checked: list[str] = []
        self._shared_rows: list[_SharedRow] = []
        self._joint_id: str | None = None
        self._rep_type: RepresentationType | None = None
        #: Set once the user types in the Label field; until then the label
        #: tracks the ticked series' default (``naming.default_joint_fit_label``).
        self._label_edited = False
        self._launch: JointFitLaunch | None = None
        self._result = None
        #: Why the Series section is empty, when the reason is not "no series".
        #: Owned here (the section renders it) so it can be read back.
        self._series_notice = ""
        #: Set while the tables are repopulated, so the per-cell ``itemChanged``
        #: and combo signals fired by rebuilding are not read as user edits.
        self._populating = False

        self._tasks = TaskRunner(self)
        self._worker = None
        self._busy = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # ── Stale banner (D9) ───────────────────────────────────────────────
        self._stale_banner = QWidget(root)
        stale_layout = QHBoxLayout(self._stale_banner)
        stale_layout.setContentsMargins(8, 4, 8, 4)
        self._stale_label = QLabel("")
        self._stale_label.setWordWrap(True)
        self._refit_btn = QPushButton("Refit")
        self._refit_btn.clicked.connect(self._on_refit_clicked)
        stale_layout.addWidget(self._stale_label, 1)
        stale_layout.addWidget(self._refit_btn, 0)
        self._stale_banner.setStyleSheet(
            f"QWidget {{ background-color: {tokens.WARN_BANNER_BG}; }}"
            f"QLabel {{ color: {tokens.WARN_BANNER_TEXT}; font-weight: bold; }}"
        )
        self._stale_banner.setVisible(False)
        layout.addWidget(self._stale_banner)

        # ── Label ───────────────────────────────────────────────────────────
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label"))
        self._label_edit = QLineEdit(root)
        # textEdited (not textChanged) fires only for typing, so refreshing the
        # default label programmatically never marks it as the user's.
        self._label_edit.textEdited.connect(self._on_label_edited)
        label_row.addWidget(self._label_edit, 1)
        layout.addLayout(label_row)

        # ── Series ──────────────────────────────────────────────────────────
        self._series_section = PanelSection("Series")
        series_section = self._series_section
        self._series_table = QTableWidget(0, 4)
        self._series_table.setHorizontalHeaderLabels(["Series", "Model", "Members", "Status"])
        self._series_table.verticalHeader().setVisible(False)
        self._series_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._series_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        series_header = self._series_table.horizontalHeader()
        series_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            series_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._series_table.itemChanged.connect(self._on_series_item_changed)
        series_section.addWidget(self._series_table)
        layout.addWidget(series_section)

        # ── Shared parameters ───────────────────────────────────────────────
        shared_section = PanelSection("Shared parameters")
        self._shared_table = QTableWidget(0, len(_SHARED_COLUMNS))
        self._shared_table.setHorizontalHeaderLabels(list(_SHARED_COLUMNS))
        apply_param_table_style(self._shared_table)
        self._shared_table.itemChanged.connect(self._on_shared_item_changed)
        shared_section.addWidget(self._shared_table)

        shared_buttons = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest")
        self._suggest_btn.setToolTip(
            "Propose shared parameters from the ticked series' models and roles."
        )
        self._suggest_btn.clicked.connect(self._on_suggest_clicked)
        shared_buttons.addWidget(self._suggest_btn)
        self._add_shared_btn = QPushButton("Add shared parameter…")
        self._add_shared_btn.clicked.connect(self._on_add_shared_clicked)
        shared_buttons.addWidget(self._add_shared_btn)
        self._remove_shared_btn = QPushButton("Remove")
        self._remove_shared_btn.clicked.connect(self._on_remove_shared_clicked)
        shared_buttons.addWidget(self._remove_shared_btn)
        shared_buttons.addStretch()
        shared_section.addLayout(shared_buttons)
        layout.addWidget(shared_section)

        # ── Footer: run controls and results ────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run joint fit")
        self._run_btn.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_btn)
        self._run_controls = FitRunControls(
            button_label="Stop",
            tooltip="Cancel the running joint fit.",
            on_cancel=self._on_stop_clicked,
        )
        run_row.addWidget(self._run_controls.button)
        run_row.addStretch()
        # Delete sits in the footer rather than beside Refit in the stale
        # banner: that banner is hidden while a joint fit is fresh, and a
        # perfectly good record must still be deletable.
        self._delete_btn = QPushButton("Delete joint fit…")
        self._delete_btn.setToolTip(
            "Remove this joint fit. Its member series and their results are kept."
        )
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        run_row.addWidget(self._delete_btn)
        layout.addLayout(run_row)

        self._results_card = FitResultsCard(actions=())
        self._results_card.set_message("No joint fit yet.")
        layout.addWidget(self._results_card)

        self._series_results = QWidget(root)
        self._series_results_layout = QVBoxLayout(self._series_results)
        self._series_results_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._series_results)

        self._update_delete_enabled()

    # ── Host wiring ─────────────────────────────────────────────────────────

    def set_providers(
        self,
        series_provider: Callable[[], list[JointSeriesEntry]],
        datasets_provider: Callable[[str], list[MuonDataset]],
    ) -> None:
        """Bind the host's series listing and per-series dataset crop."""
        self._series_provider = series_provider
        self._datasets_provider = datasets_provider

    # ── Opening ─────────────────────────────────────────────────────────────

    def start_new(self, rep_type: RepresentationType | None) -> None:
        """Open empty on *rep_type*: nothing ticked, no shared rows, no record."""
        self._joint_id = None
        self._rep_type = rep_type
        self._checked = []
        self._shared_rows = []
        self._label_edited = False
        self._result = None
        self._launch = None
        self.setWindowTitle("Joint Fit")
        self.set_stale(False)
        self._results_card.set_message("No joint fit yet.")
        self._clear_series_results()
        self.refresh_series()

    def load_joint_fit(self, joint: JointFit, *, stale_reason: str = "") -> None:
        """Show a recorded joint fit: its members ticked, its shared table loaded (D11)."""
        self._joint_id = joint.joint_id
        self._rep_type = joint.rep_type
        self._label_edited = joint.label is not None
        self._shared_rows = [
            _SharedRow(
                name=str(row["name"]),
                members=dict(row["members"]),
                value=float(row["value"]),
                minimum=float(row["min"]),
                maximum=float(row["max"]),
                ticked=True,
                # A persisted row is the user's table, not a fresh proposal, so
                # "Suggest" must leave it alone (D7).
                tier="user",
            )
            for row in joint.shared
        ]
        self._checked = list(joint.member_batch_ids)
        self.refresh_series()
        if joint.label is not None:
            self._label_edit.setText(joint.label)
        self._render_stored_result(joint)
        self.set_stale(bool(stale_reason), stale_reason)

    def open_joint_id(self) -> str | None:
        """The id of the joint fit on show, or ``None`` for an unrecorded one."""
        return self._joint_id

    def note_recorded(self, joint_id: str) -> None:
        """Adopt the record this run created or updated (the host mints the id).

        The window launches a run knowing only whether it is *re-running* a
        record (``joint_id`` on the launch) — a first run has no id until the
        host allocates one. Taking it back here is what makes the next run an
        update rather than a second record, and what lets the submenu tick the
        entry that is on show.
        """
        self._joint_id = str(joint_id)
        self.set_stale(False)
        self._update_delete_enabled()

    def forget_joint_fit(self) -> None:
        """Detach from a record that no longer exists; the table stays editable.

        Called by the host when the displayed joint fit is deleted out from
        under the window (its last member went). The composition on screen is
        still perfectly runnable — it would simply record a *new* joint fit —
        so the window keeps it and only drops the id it would have updated.
        """
        self._joint_id = None
        self.set_stale(False)
        self.refresh_series()

    # ── Series section ──────────────────────────────────────────────────────

    def refresh_series(self) -> None:
        """Re-read the host's series list and repopulate the picker.

        Eligibility is re-derived here on every call — including after each
        tick — because the overlap rule (D3) is relative to what is ticked: a
        series shares runs with another only while that other one is a member.
        """
        self._entries = list(self._series_provider())
        known = {entry.batch_id for entry in self._entries if not entry.blocked_reason}
        self._checked = [batch_id for batch_id in self._checked if batch_id in known]
        self._refresh_series_notice()
        self._populate_series_table()
        self._refresh_label_default()
        self._populate_shared_table()
        self._update_delete_enabled()

    def _refresh_series_notice(self) -> None:
        """Say why the picker is empty when "no series" is not the reason (v1)."""
        frequency = self._rep_type is not None and self._rep_type.domain == "frequency"
        self._series_notice = FREQUENCY_NOTICE if frequency else ""
        self._series_section.set_hint(self._series_notice or None)

    def series_notice(self) -> str:
        """The line shown under the Series header, or ``""`` when there is none."""
        return self._series_notice

    def _entry(self, batch_id: str) -> JointSeriesEntry | None:
        for entry in self._entries:
            if entry.batch_id == batch_id:
                return entry
        return None

    def checked_series(self) -> list[JointSeriesEntry]:
        """The ticked series, in tick order."""
        return [entry for batch_id in self._checked if (entry := self._entry(batch_id))]

    def _overlap_reason(self, entry: JointSeriesEntry) -> str:
        """Why *entry* may not be ticked alongside what already is (D3), or ``""``."""
        if entry.batch_id in self._checked:
            return ""
        members = set(entry.members)
        for batch_id in self._checked:
            other = self._entry(batch_id)
            if other is not None and members & set(other.members):
                return f"Shares runs with {other.label}"
        return ""

    def _populate_series_table(self) -> None:
        self._populating = True
        try:
            table = self._series_table
            table.clearContents()
            table.setRowCount(len(self._entries))
            for row, entry in enumerate(self._entries):
                reason = entry.blocked_reason or self._overlap_reason(entry)
                name_item = QTableWidgetItem(entry.label)
                name_item.setData(Qt.ItemDataRole.UserRole, entry.batch_id)
                flags = Qt.ItemFlag.ItemIsSelectable
                if not reason:
                    flags |= Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
                name_item.setFlags(flags)
                name_item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.batch_id in self._checked
                    else Qt.CheckState.Unchecked
                )
                cells = [
                    name_item,
                    QTableWidgetItem(entry.model_text),
                    QTableWidgetItem(str(len(entry.members))),
                    QTableWidgetItem(entry.status),
                ]
                for column, item in enumerate(cells):
                    if column:
                        item.setFlags(
                            Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                            if not reason
                            else Qt.ItemFlag.ItemIsSelectable
                        )
                    if reason:
                        item.setToolTip(reason)
                    table.setItem(row, column, item)
        finally:
            self._populating = False

    def _on_series_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != 0:
            return
        batch_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(batch_id, str):
            return
        if item.checkState() == Qt.CheckState.Checked:
            if batch_id not in self._checked:
                self._checked.append(batch_id)
        elif batch_id in self._checked:
            self._checked.remove(batch_id)
        # A tick changes who overlaps whom and which columns the shared table
        # has, so both tables are re-derived rather than patched.
        self._populate_series_table()
        self._refresh_label_default()
        self._populate_shared_table()

    # ── Label ───────────────────────────────────────────────────────────────

    def _on_label_edited(self, _text: str) -> None:
        self._label_edited = True

    def _refresh_label_default(self) -> None:
        if self._label_edited:
            return
        labels = [entry.label for entry in self.checked_series()]
        self._label_edit.setText(default_joint_fit_label(labels) if labels else "")

    def current_label(self) -> str:
        return self._label_edit.text().strip()

    # ── Shared table ────────────────────────────────────────────────────────

    def _row_is_tickable(self, row: _SharedRow) -> bool:
        """A shared row needs two contributing ticked series to mean anything (D4)."""
        return sum(1 for batch_id in self._checked if row.members.get(batch_id)) >= 2

    def _populate_shared_table(self) -> None:
        self._populating = True
        try:
            entries = self.checked_series()
            table = self._shared_table
            table.clearContents()
            table.setColumnCount(len(_SHARED_COLUMNS) + len(entries))
            table.setHorizontalHeaderLabels([*_SHARED_COLUMNS, *(entry.label for entry in entries)])
            table.setRowCount(len(self._shared_rows))
            for row_index, row in enumerate(self._shared_rows):
                tickable = self._row_is_tickable(row)
                if not tickable:
                    row.ticked = False
                name_item = QTableWidgetItem(row.name)
                flags = (
                    Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsEditable
                )
                if tickable:
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
                else:
                    name_item.setToolTip(
                        "A shared parameter must be contributed by at least two series."
                    )
                name_item.setFlags(flags)
                name_item.setCheckState(
                    Qt.CheckState.Checked if row.ticked else Qt.CheckState.Unchecked
                )
                table.setItem(row_index, 0, name_item)
                for column, text in enumerate(
                    (
                        f"{row.value:g}",
                        _format_bound(row.minimum),
                        _format_bound(row.maximum),
                    ),
                    start=1,
                ):
                    table.setItem(row_index, column, QTableWidgetItem(text))
                for offset, entry in enumerate(entries):
                    combo = QComboBox()
                    combo.addItem(NO_MEMBER)
                    combo.addItems(entry.global_params())
                    current = row.members.get(entry.batch_id, NO_MEMBER)
                    combo.setCurrentText(current if combo.findText(current) >= 0 else NO_MEMBER)
                    combo.currentTextChanged.connect(
                        lambda text, r=row_index, b=entry.batch_id: self._on_member_changed(
                            r, b, text
                        )
                    )
                    table.setCellWidget(row_index, len(_SHARED_COLUMNS) + offset, combo)
        finally:
            self._populating = False

    def _on_shared_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating:
            return
        row_index = item.row()
        if not 0 <= row_index < len(self._shared_rows):
            return
        row = self._shared_rows[row_index]
        column = item.column()
        if column == 0:
            row.name = item.text().strip()
            row.ticked = item.checkState() == Qt.CheckState.Checked
        elif column == 1:
            row.value = _parse_bound(item.text(), 0.0)
        elif column == 2:
            row.minimum = _parse_bound(item.text(), -math.inf)
        elif column == 3:
            row.maximum = _parse_bound(item.text(), math.inf)
        else:
            return
        row.user_edited = True

    def _on_member_changed(self, row_index: int, batch_id: str, text: str) -> None:
        if self._populating or not 0 <= row_index < len(self._shared_rows):
            return
        row = self._shared_rows[row_index]
        if text == NO_MEMBER:
            row.members.pop(batch_id, None)
        else:
            row.members[batch_id] = text
        row.user_edited = True
        # Only this row's tick-ability can have changed, and repopulating the
        # whole table from inside a cell widget's own signal would delete the
        # combo that is emitting — so patch the one row in place.
        self._refresh_row_tickability(row_index)

    def _refresh_row_tickability(self, row_index: int) -> None:
        """Re-apply the two-contributor rule (D4) to one shared row's checkbox."""
        row = self._shared_rows[row_index]
        item = self._shared_table.item(row_index, 0)
        tickable = self._row_is_tickable(row)
        if not tickable:
            row.ticked = False
        self._populating = True
        try:
            flags = (
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
            )
            if tickable:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if row.ticked else Qt.CheckState.Unchecked)
            item.setToolTip(
                "" if tickable else "A shared parameter must be contributed by at least two series."
            )
        finally:
            self._populating = False

    def _seed_from_first_contributor(self, row: _SharedRow) -> None:
        """Seed *row*'s value and bounds from its first contributing series (D6).

        "First" is in tick order, which is also the joint fit's member order, so
        the seed a shared row starts from does not depend on how the shared
        table happens to be sorted. Only that one series is consulted: the
        shared value is a new fitted quantity, and averaging two series' seeds
        would invent a number neither of them holds.
        """
        contributors = (
            (self._entry(batch_id), row.members.get(batch_id)) for batch_id in self._checked
        )
        first = next(
            ((entry, pname) for entry, pname in contributors if entry is not None and pname),
            None,
        )
        if first is None:
            return
        entry, pname = first
        for recipe_row in entry.recipe["parameters"]:
            if str(recipe_row["name"]) != pname:
                continue
            row.value = float(recipe_row["value"])
            minimum, _, maximum = str(recipe_row["bounds"]).partition(",")
            row.minimum = _parse_bound(minimum.strip(), -math.inf)
            row.maximum = _parse_bound(maximum.strip(), math.inf)
            return

    def _on_suggest_clicked(self) -> None:
        entries = [entry for entry in self.checked_series() if entry.model is not None]
        if len(entries) < 2:
            self._results_card.set_message(
                "Tick at least two series before suggesting shared parameters.",
                tag="Error",
                tone="error",
            )
            return
        suggestions = suggest_shared_parameters(
            [entry.model for entry in entries],
            [dict(entry.roles) for entry in entries],
        )
        # A user-added or user-edited row is an answer the user has already
        # given; a re-proposal must not take it back (D7).
        kept = [row for row in self._shared_rows if row.survives_suggestion()]
        taken = {row.name for row in kept}
        for suggestion in suggestions:
            if suggestion.name in taken:
                continue
            row = _SharedRow(
                name=suggestion.name,
                members={
                    entries[index].batch_id: pname for index, pname in suggestion.members.items()
                },
                ticked=suggestion.tier == "exact",
                tier=suggestion.tier,
            )
            self._seed_from_first_contributor(row)
            kept.append(row)
            taken.add(row.name)
        self._shared_rows = kept
        self._populate_shared_table()

    def _on_add_shared_clicked(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add shared parameter", "Name:")
        name = name.strip()
        if not accepted or not name:
            return
        self._shared_rows.append(_SharedRow(name=name, tier="user"))
        self._populate_shared_table()

    def _on_remove_shared_clicked(self) -> None:
        rows = sorted({index.row() for index in self._shared_table.selectedIndexes()}, reverse=True)
        for row_index in rows:
            if 0 <= row_index < len(self._shared_rows):
                del self._shared_rows[row_index]
        self._populate_shared_table()

    # ── Running ─────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        self._run_btn.setVisible(not busy)
        self._run_controls.button.setVisible(busy)
        for widget in (
            self._series_table,
            self._shared_table,
            self._suggest_btn,
            self._add_shared_btn,
            self._remove_shared_btn,
            self._label_edit,
        ):
            widget.setEnabled(not busy)
        self._busy = busy
        self._update_delete_enabled()

    def _update_delete_enabled(self) -> None:
        """Delete is offered only for a recorded joint fit, and never mid-run."""
        self._delete_btn.setEnabled(self._joint_id is not None and not self._busy)

    def _on_run_clicked(self) -> None:
        entries = [entry for entry in self.checked_series() if entry.model is not None]
        if len(entries) < 2:
            self._results_card.set_message(
                "A joint fit needs at least two series.", tag="Error", tone="error"
            )
            return
        ticked_ids = {entry.batch_id for entry in entries}
        rows = [row for row in self._shared_rows if row.ticked and self._row_is_tickable(row)]
        if not rows:
            self._results_card.set_message(
                "Tick at least one shared parameter.", tag="Error", tone="error"
            )
            return

        problems: list[JointSeriesProblem] = []
        for entry in entries:
            datasets = list(self._datasets_provider(entry.batch_id))
            inputs = build_recipe_engine_inputs(entry.recipe, entry.model, datasets)
            problems.append(
                JointSeriesProblem(
                    key=entry.batch_id,
                    datasets=datasets,
                    model_fn=entry.model.function,
                    global_params=inputs.global_params,
                    local_params=inputs.local_params,
                    initial_params=inputs.initial_params,
                    t_min=inputs.t_min,
                    t_max=inputs.t_max,
                )
            )
        shared = [
            SharedParameter(
                name=row.name,
                members={
                    batch_id: pname
                    for batch_id, pname in row.members.items()
                    if batch_id in ticked_ids
                },
                value=row.value,
                min=row.minimum,
                max=row.maximum,
            )
            for row in rows
        ]

        launch = JointFitLaunch(
            joint_id=self._joint_id,
            label=self.current_label(),
            # The members are all of one representation type by construction
            # (the host lists only the active one, D3), so the first is it.
            rep_type=entries[0].rep_type,
            member_batch_ids=tuple(entry.batch_id for entry in entries),
            shared_rows=tuple(
                {
                    **row.to_record(),
                    "members": {
                        batch_id: pname
                        for batch_id, pname in row.members.items()
                        if batch_id in ticked_ids
                    },
                }
                for row in rows
            ),
            models={entry.batch_id: entry.model for entry in entries},
            global_params={
                entry.batch_id: tuple(problem.global_params)
                for entry, problem in zip(entries, problems, strict=True)
            },
        )
        self._launch = launch
        self._results_card.set_message("Fitting… coupled over the ticked series.", tag="Fitting")
        self._set_busy(True)
        self._worker = self._tasks.start(
            lambda worker: run_joint_fit_with_curves(
                problems, shared, cancel_callback=worker.is_cancelled
            ),
            on_finished=self._on_fit_finished,
            on_error=self._on_fit_error,
            on_cancelled=self._on_fit_cancelled,
            cancel_exceptions=(FitCancelledError,),
        )

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_refit_clicked(self) -> None:
        """Re-run the displayed joint fit against the live series (D9)."""
        self.refresh_series()
        self._on_run_clicked()

    def _on_delete_clicked(self) -> None:
        """Confirm, then ask the host to drop this joint fit (D10).

        The confirmation says what deleting does *not* do, because that is the
        part a user cannot see: the members and every result they carry stay
        exactly where they are — only the record and its stamps go.
        """
        if self._joint_id is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete joint fit",
            f'Delete the joint fit "{self.current_label()}"?\n\n'
            "Its member series and their results are kept; only the joint fit "
            "and its shared-parameter table are removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.joint_fit_delete_requested.emit(self._joint_id)

    def _on_fit_finished(self, run: JointFitRun) -> None:
        self._set_busy(False)
        self._worker = None
        result = run.result
        if not result.success:
            # Nothing is recorded: a failed joint fit must not overwrite the
            # members' previous results with an unconverged answer (D8).
            self._results_card.set_message(
                result.message or "The joint fit did not converge.", tag="Error", tone="error"
            )
            self._clear_series_results()
            return
        self._result = result
        self._render_result(self._launch, result)
        # The curves travelled back with the result, so the host draws without
        # evaluating a model on the GUI thread.
        self.joint_fit_completed.emit(self._launch, result, run.curves)

    def _on_fit_error(self, message: str) -> None:
        self._set_busy(False)
        self._worker = None
        self._results_card.set_message(message, tag="Error", tone="error")

    def _on_fit_cancelled(self) -> None:
        # The previous result stays on screen, and nothing was recorded, so the
        # members keep whatever they were last fitted with.
        self._set_busy(False)
        self._worker = None
        self._results_card.set_message("Joint fit stopped.", tag="Stopped", tone="warn")

    # ── Results ─────────────────────────────────────────────────────────────

    def has_result(self) -> bool:
        return self._result is not None

    def _clear_series_results(self) -> None:
        while self._series_results_layout.count():
            item = self._series_results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_series_result_row(self, batch_id: str, label: str, chi2r: float | None) -> None:
        row = QWidget(self._series_results)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        text = label if chi2r is None else f"{label} — χ²ᵣ = {chi2r:.4g}"
        row_layout.addWidget(QLabel(text), 1)
        button = QPushButton("Open in Batch tab")
        button.clicked.connect(
            lambda _checked=False, b=batch_id: self.open_series_requested.emit(b)
        )
        row_layout.addWidget(button, 0)
        self._series_results_layout.addWidget(row)

    def _render_result(self, launch: JointFitLaunch, result) -> None:
        shared_text = " · ".join(
            f"{name} = {result.shared_parameters[name].value:.5g}"
            f" ± {result.shared_uncertainties.get(name, float('nan')):.3g}"
            for name in result.shared_parameters.names
        )
        self._results_card.set_summary(
            FitCardSummary(
                tag="Joint ✓",
                tone="ok",
                headline=f"{len(launch.member_batch_ids)} series",
                meta=f"ndof {result.dof}",
                detail_html=f"χ²ᵣ = {result.reduced_chi_squared:.4g} · {shared_text}",
            )
        )
        self._clear_series_results()
        # The members that were *fitted*, off the frozen launch — the tick state
        # may already have moved on while the fit ran.
        for batch_id in launch.member_batch_ids:
            entry = self._entry(batch_id)
            self._add_series_result_row(
                batch_id,
                entry.label if entry is not None else batch_id,
                result.series_reduced_chi_squared.get(batch_id),
            )

    def _render_stored_result(self, joint: JointFit) -> None:
        """Replay a recorded joint fit's summary (no live result behind it)."""
        self._clear_series_results()
        summary = joint.result
        if not summary:
            self._results_card.set_message("No joint fit yet.")
            return
        shared_values = summary.get("shared_values") or {}
        uncertainties = summary.get("shared_uncertainties") or {}
        shared_text = " · ".join(
            f"{name} = {float(value):.5g} ± {float(uncertainties.get(name, float('nan'))):.3g}"
            for name, value in shared_values.items()
        )
        self._results_card.set_message(
            f"χ²ᵣ = {float(summary.get('reduced_chi_squared', float('nan'))):.4g} · {shared_text}",
            tag="Joint ✓",
            tone="ok",
        )
        per_series = summary.get("series_reduced_chi_squared") or {}
        for entry in self.checked_series():
            value = per_series.get(entry.batch_id)
            self._add_series_result_row(
                entry.batch_id, entry.label, None if value is None else float(value)
            )

    # ── Staleness (D9) ──────────────────────────────────────────────────────

    def set_stale(self, stale: bool, reason: str = "") -> None:
        """Show or hide the stale banner with *reason*."""
        self._stale_label.setText(
            f"This joint fit is out of date: {reason}."
            if reason
            else "This joint fit is out of date."
        )
        self._stale_banner.setVisible(bool(stale))

    # ── Persistence ─────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, object]:
        """Only the open joint id: everything else lives on the record itself."""
        return {"joint_id": self._joint_id}

    def restore_state(self, state: object) -> None:
        """Take the saved joint id; the host loads that record into the window."""
        if not isinstance(state, dict):
            return
        joint_id = state.get("joint_id")
        self._joint_id = str(joint_id) if joint_id else None

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        self._tasks.shutdown()
        super().closeEvent(event)
