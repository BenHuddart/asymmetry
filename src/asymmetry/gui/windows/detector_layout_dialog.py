"""Interactive detector layout editor dialog.

This dialog provides a visual interface for assigning individual detector
elements to named groups.  It is opened from a button inside the standard
:class:`~asymmetry.gui.windows.grouping_dialog.GroupingDialog` and writes
back updated group definitions on acceptance.

The dialog has three sections arranged horizontally:

* **Left** — an interactive :class:`~asymmetry.gui.widgets.detector_schematic.DetectorSchematicWidget`
  that shows the detector arrangement for the selected instrument.  Clicking a
  segment toggles its membership in the currently active group.
* **Centre** — eight toggle buttons (Group 1–Group 8) that select the active
  group, each accompanied by an editable name field.
* **Right** — an instrument selector and a preset dropdown with an
  *Apply Grouping* button that populates all groups from a manual-derived
  preset.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.instrument import (
    InstrumentLayout,
    get_instrument_layout,
    instrument_choices_for,
    recommend_grouping_preset,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.detector_schematic import _GROUP_COLOURS, DetectorSchematicWidget

__all__ = ["DetectorLayoutDialog"]

# Maximum number of groups exposed in the UI
_MAX_GROUPS = 8


def _colour_css(group_id: int) -> str:
    """Return a CSS border-color string for the group's colour."""
    r, g, b, _ = _GROUP_COLOURS[(group_id - 1) % len(_GROUP_COLOURS)]
    qc = QColor.fromRgbF(r, g, b)
    return qc.name()


class DetectorLayoutDialog(QDialog):
    """Visual detector grouping editor.

    Parameters
    ----------
    instrument:
        Pre-selected instrument layout.  Users can override this via the
        instrument combo inside the dialog.
    groups:
        Current group definitions mapping group ID (1-based) to list of
        1-based detector IDs.
    group_names:
        Optional mapping from group ID to human-readable group name.
    forward_group:
        Group ID currently designated as the *forward* group.
    backward_group:
        Group ID currently designated as the *backward* group.
    field_direction:
        Applied-field geometry of the run (``"Transverse"`` / ``"Longitudinal"``
        / ``"Zero field"`` / ``None``).  When the run is transverse-field but the
        current preset is not the recommended transverse one, the dialog shows a
        non-blocking hint and pre-selects the recommended preset in the combo
        (the user still clicks *Apply Grouping*).
    parent:
        Parent Qt widget.
    """

    def __init__(
        self,
        instrument: InstrumentLayout,
        groups: dict[int, list[int]],
        group_names: dict[int, str] | None = None,
        initial_preset_name: str | None = None,
        forward_group: int = 1,
        backward_group: int = 2,
        excluded_detectors: list[int] | None = None,
        projections: list[dict] | None = None,
        field_direction: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detector Layout Editor")
        self.resize(1020, 560)

        self._instrument = instrument
        # Applied-field geometry of the loaded run (metadata["field_direction"]):
        # used to nudge transverse-field data off a longitudinal preset.
        self._field_direction = field_direction
        self._forward_group = forward_group
        self._backward_group = backward_group
        # Declared projections (multi-projection presets). Seeded from the
        # incoming grouping so an open→OK without re-applying a preset preserves
        # them; replaced wholesale when a preset is applied.
        self._applied_projections: list[dict] = [dict(p) for p in (projections or [])]

        # Internal group state: gid → set of 1-based detector IDs
        self._groups: dict[int, set[int]] = {gid: set(ids) for gid, ids in groups.items()}
        # Group names
        self._group_names: dict[int, str] = dict(group_names or {})
        self._applied_preset_name: str | None = (
            str(initial_preset_name) if initial_preset_name else None
        )

        self._active_group: int = 1
        self._group_button_scale = 1.0
        self._ui_scale_sync_connected = False

        # Build UI -------------------------------------------------------
        # Root: vertical stack of [panel row] + [button box]
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Horizontal panel row
        root = QHBoxLayout()
        root.setSpacing(8)
        main_layout.addLayout(root, stretch=1)

        # ------------------------------------------------------------------
        # Left: schematic
        # ------------------------------------------------------------------
        self._schematic = DetectorSchematicWidget(self._instrument, parent=self)
        self._schematic.setMinimumWidth(360)
        self._schematic.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._schematic.detector_toggled.connect(self._on_detector_toggled)
        self._schematic.set_excluded_detectors(set(excluded_detectors or []))
        root.addWidget(self._schematic, stretch=6)

        # ------------------------------------------------------------------
        # Centre: group selector panel
        # ------------------------------------------------------------------
        group_box = QGroupBox("Groups")
        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(4)

        self._group_btn_group = QButtonGroup(self)
        self._group_btn_group.setExclusive(True)
        self._group_buttons: dict[int, QPushButton] = {}
        self._group_name_edits: dict[int, QLineEdit] = {}
        self._group_rows: dict[int, QWidget] = {}

        for gid in range(1, _MAX_GROUPS + 1):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            colour = _colour_css(gid)
            swatch.setStyleSheet(
                f"background-color: {colour}; border-radius: 2px;"
                " border: 1px solid rgba(0,0,0,0.15);"
            )
            swatch.setToolTip(f"Group {gid} colour")

            btn = QPushButton(f"Group {gid}")
            btn.setCheckable(True)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            # Preferred (not Fixed) horizontally so setMinimumWidth can grow the
            # button to fit its label; height stays pinned via setFixedHeight.
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self._group_btn_group.addButton(btn, gid)
            self._group_buttons[gid] = btn

            name_edit = QLineEdit()
            name_edit.setPlaceholderText(f"Group {gid}")
            name_edit.setText(self._group_names.get(gid, ""))
            name_edit.textChanged.connect(self._on_group_definition_changed)
            self._group_name_edits[gid] = name_edit

            row.addWidget(swatch)
            row.addWidget(btn)
            row.addWidget(name_edit)
            group_layout.addWidget(row_widget)

            # Hovering a group's row temporarily highlights that group's
            # detectors in the schematic (edge emphasis); purely visual.
            row_widget.installEventFilter(self)
            self._group_rows[gid] = row_widget

        group_layout.addStretch()
        self._group_btn_group.idClicked.connect(self._on_group_button_clicked)

        # Select group 1 initially
        self._group_buttons[1].setChecked(True)

        clear_btn = QPushButton("Clear All Groups")
        clear_btn.setAutoDefault(False)
        clear_btn.setDefault(False)
        clear_btn.clicked.connect(self._on_clear_all)
        group_layout.addWidget(clear_btn)

        self._exclude_mode_btn = QPushButton("Exclude mode")
        self._exclude_mode_btn.setCheckable(True)
        self._exclude_mode_btn.setAutoDefault(False)
        self._exclude_mode_btn.setDefault(False)
        self._exclude_mode_btn.setToolTip(
            "While active, clicking a detector toggles its exclusion (dead/hot "
            "detectors are dropped from every group at reduction time)."
        )
        self._exclude_mode_btn.toggled.connect(self._schematic.set_exclude_mode)
        group_layout.addWidget(self._exclude_mode_btn)

        self._clear_excluded_btn = QPushButton("Clear excluded")
        self._clear_excluded_btn.setAutoDefault(False)
        self._clear_excluded_btn.setDefault(False)
        self._clear_excluded_btn.setToolTip(
            "Remove every detector from the exclusion set (undo all exclude-mode edits)."
        )
        self._clear_excluded_btn.clicked.connect(self._on_clear_excluded)
        group_layout.addWidget(self._clear_excluded_btn)

        root.addWidget(group_box, stretch=3)

        # ------------------------------------------------------------------
        # Right: instrument + preset panel
        # ------------------------------------------------------------------
        preset_box = QGroupBox("Default Groupings")
        preset_layout = QVBoxLayout(preset_box)
        preset_layout.setSpacing(8)

        preset_layout.addWidget(QLabel("Instrument:"))
        self._instrument_combo = QComboBox()
        # Items carry the registry key as itemData; the visible text is the
        # display name. Variant families (GPS BIN vs GPS ROOT sub-detectors)
        # collapse to one "GPS" entry mapping to the variant matching the loaded
        # data, so the user only sees the GPS that fits their file format.
        for display_name, registry_key in instrument_choices_for(self._instrument.name):
            self._instrument_combo.addItem(display_name, registry_key)
        current_idx = self._instrument_combo.findData(self._instrument.name)
        if current_idx >= 0:
            self._instrument_combo.setCurrentIndex(current_idx)
        self._instrument_combo.currentIndexChanged.connect(self._on_instrument_index_changed)
        preset_layout.addWidget(self._instrument_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        preset_layout.addWidget(sep)

        preset_layout.addWidget(QLabel("Preset grouping:"))
        self._preset_combo = QComboBox()
        self._populate_preset_combo()
        preset_layout.addWidget(self._preset_combo)

        self._preset_status_label = QLabel("(Current: Custom)")
        self._preset_status_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        preset_layout.addWidget(self._preset_status_label)

        apply_btn = QPushButton("Apply Grouping")
        apply_btn.setDefault(False)
        apply_btn.setAutoDefault(False)
        apply_btn.clicked.connect(self._on_apply_preset)
        preset_layout.addWidget(apply_btn)

        # Non-blocking transverse-field nudge: shown when a TF run is on a
        # longitudinal (or otherwise non-recommended) preset. Hidden otherwise.
        self._tf_hint_label = QLabel()
        self._tf_hint_label.setWordWrap(True)
        self._tf_hint_label.setStyleSheet(f"color: {tokens.WARN};")
        self._tf_hint_label.setVisible(False)
        preset_layout.addWidget(self._tf_hint_label)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        preset_layout.addWidget(sep2)

        help_label = QLabel(
            "Click detector segments to toggle\n"
            "membership in the active group.\n\n"
            "Detectors can be included in\n"
            "multiple groups."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        preset_layout.addWidget(help_label)
        preset_layout.addStretch()

        root.addWidget(preset_box, stretch=2)

        # ------------------------------------------------------------------
        # Bottom: OK / Cancel
        # ------------------------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Initial schematic state
        self._apply_group_button_metrics()
        self._apply_group_button_styles()
        self._sync_schematic()
        self._update_preset_status_label()
        # Seed the TF nudge once: pre-select the recommended preset in the combo
        # so applying it is one click (no auto-apply).
        self._update_grouping_recommendation(preselect=True)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._ensure_ui_scale_sync()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Highlight a group's detectors in the schematic while its row is hovered."""
        if event.type() == QEvent.Type.Enter:
            for gid, row_widget in self._group_rows.items():
                if watched is row_widget:
                    self._schematic.set_group_highlight(gid)
                    break
        elif event.type() == QEvent.Type.Leave:
            for row_widget in self._group_rows.values():
                if watched is row_widget:
                    self._schematic.set_group_highlight(None)
                    break
        return super().eventFilter(watched, event)

    def _ensure_ui_scale_sync(self) -> None:
        if self._ui_scale_sync_connected:
            return
        parent = self.parentWidget()
        while parent is not None:
            manager = getattr(parent, "_ui_manager", None)
            if manager is not None:
                manager.ui_scale_changed.connect(self._on_ui_scale_changed)
                self._ui_scale_sync_connected = True
                self._on_ui_scale_changed(manager.ui_scale, manager.effective_scale)
                return
            parent = parent.parentWidget()

    def _on_ui_scale_changed(self, _ui_scale: float, effective_scale: float) -> None:
        self._group_button_scale = max(0.8, float(effective_scale))
        self._apply_group_button_metrics()
        self._apply_group_button_styles()

    def _button_text_min_width(self, button: QPushButton) -> int:
        """Return the narrowest width that shows *button*'s full text unclipped.

        Uses the button's font-advance for its current text plus the style's
        contents margins (the same inputs Qt's own ``sizeHint`` combines), so
        this stays consistent whether the text or a UI-scale change is what
        triggered the recompute.
        """
        fm = QFontMetrics(button.font())
        text_width = fm.horizontalAdvance(button.text())
        option = QStyleOptionButton()
        option.initFrom(button)
        option.text = button.text()
        style = button.style() or QApplication.style()
        # sizeFromContents adds the style's frame/padding around the raw text
        # extent (mirrors what setFixedWidth previously ignored).
        content_size = style.sizeFromContents(
            QStyle.ContentsType.CT_PushButton,
            option,
            button.fontMetrics().boundingRect(button.text()).size(),
            button,
        )
        return max(text_width + 16, content_size.width())

    def _apply_group_button_metrics(self) -> None:
        scale = max(0.8, float(self._group_button_scale))
        button_width_floor = max(68, round(76 * scale))
        button_height = max(28, round(28 * scale))
        edit_width = max(100, round(110 * scale))
        for button in self._group_buttons.values():
            button.setMinimumWidth(max(button_width_floor, self._button_text_min_width(button)))
            button.setFixedHeight(button_height)
        for edit in self._group_name_edits.values():
            edit.setFixedWidth(edit_width)

    def _apply_group_button_styles(self) -> None:
        scale = max(0.8, float(self._group_button_scale))
        border_radius = max(12, round(14 * scale))
        padding_v = max(2, round(2 * scale))
        padding_h = max(8, round(10 * scale))
        for gid, button in self._group_buttons.items():
            colour = _colour_css(gid)
            button.setStyleSheet(
                "QPushButton {"
                " border: 1px solid #999;"
                f" border-radius: {border_radius}px;"
                f" padding: {padding_v}px {padding_h}px;"
                f" background-color: {tokens.SURFACE_ALT};"
                "}"
                f"QPushButton:checked {{ border: 2px solid {colour}; "
                f"border-radius: {border_radius}px; "
                f"padding: {padding_v}px {padding_h}px; "
                f"background-color: {tokens.ACCENT_SOFT}; font-weight: bold; }}"
            )

    # ------------------------------------------------------------------
    # Preset combo management
    # ------------------------------------------------------------------

    def _populate_preset_combo(self) -> None:
        """Fill the preset combo from the current instrument's presets."""
        self._preset_combo.clear()
        for name in self._instrument.presets:
            self._preset_combo.addItem(name)

    # ------------------------------------------------------------------
    # Schematic synchronisation
    # ------------------------------------------------------------------

    def _sync_schematic(self) -> None:
        """Push current group state and active group to the schematic."""
        self._schematic.set_all_groups(
            {gid: list(ids) for gid, ids in self._groups.items()},
            self._active_group,
        )
        self._schematic.set_group_names(self._current_group_names_from_edits())
        self._refresh_group_button_labels()

    # Buttons wider than this are elided; pathologically long custom group
    # names should not be allowed to blow out the centre panel's layout.
    # Ordinary labels — including composite preset names like
    # "Top-Bottom Top (18)" — must stay well under this so only genuinely
    # excessive user-entered names ever get elided.
    _MAX_GROUP_BUTTON_TEXT_WIDTH = 320

    def _refresh_group_button_labels(self) -> None:
        """Update each group button's text to show its member count, e.g. "Top (18)"."""
        for gid, btn in self._group_buttons.items():
            name = self._group_name_edits[gid].text().strip()
            base = name if name else f"Group {gid}"
            count = len(self._groups.get(gid, ()))
            full_text = f"{base} ({count})" if count else base
            btn.setToolTip(full_text)

            fm = QFontMetrics(btn.font())
            if fm.horizontalAdvance(full_text) > self._MAX_GROUP_BUTTON_TEXT_WIDTH:
                display_text = fm.elidedText(
                    full_text, Qt.TextElideMode.ElideRight, self._MAX_GROUP_BUTTON_TEXT_WIDTH
                )
            else:
                display_text = full_text
            btn.setText(display_text)

            # Text changed independently of any UI-scale change, so the
            # minimum width needs recomputing here too (not only from
            # _apply_group_button_metrics).
            scale = max(0.8, float(self._group_button_scale))
            button_width_floor = max(68, round(76 * scale))
            btn.setMinimumWidth(max(button_width_floor, self._button_text_min_width(btn)))

    def _current_group_names_from_edits(self) -> dict[int, str]:
        """Return normalized group-name mapping from the editable name fields."""
        names: dict[int, str] = {}
        for gid, edit in self._group_name_edits.items():
            text = edit.text().strip()
            if text:
                names[gid] = text
        return names

    def _state_matches_preset(self, preset_name: str) -> bool:
        """Return True when current groups/names/FB selections match *preset_name*."""
        preset = self._instrument.presets.get(preset_name)
        if preset is None:
            return False

        current_groups = {gid: set(ids) for gid, ids in self._groups.items() if ids}
        preset_groups = {
            gid: set(gdef.detector_ids) for gid, gdef in preset.groups.items() if gdef.detector_ids
        }
        if current_groups != preset_groups:
            return False

        current_names = self._current_group_names_from_edits()
        preset_names = {gid: gdef.name for gid, gdef in preset.groups.items() if gdef.name}
        if current_names != preset_names:
            return False

        if self._forward_group != preset.forward_group:
            return False
        if self._backward_group != preset.backward_group:
            return False
        return True

    def _update_preset_status_label(self) -> None:
        """Update the '(Current: …)' status text under the preset selector."""
        if self._applied_preset_name and self._state_matches_preset(self._applied_preset_name):
            self._preset_status_label.setText(f"(Current: {self._applied_preset_name})")
        else:
            self._applied_preset_name = None
            self._preset_status_label.setText("(Current: Custom)")

    def _update_grouping_recommendation(self, *, preselect: bool = False) -> None:
        """Show or hide the transverse-field nudge for the current layout.

        The nudge appears when the run is transverse-field and the applied preset
        is not the recommended transverse one (see
        :func:`asymmetry.core.instrument.recommend_grouping_preset`).  When
        *preselect* is set, the recommended preset is also pre-selected in the
        combo so applying it is a single click — never auto-applied.
        """
        recommended = recommend_grouping_preset(self._instrument, self._field_direction)
        if recommended is None or recommended == self._applied_preset_name:
            self._tf_hint_label.setVisible(False)
            return

        if preselect:
            idx = self._preset_combo.findText(recommended)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
        self._tf_hint_label.setText(
            f"Transverse-field run: the current grouping washes out the "
            f"precession. ‘{recommended}’ is recommended — click "
            f"Apply Grouping."
        )
        self._tf_hint_label.setVisible(True)

    def _on_group_definition_changed(self, *_args) -> None:
        """Refresh preset status and button labels when detector/name assignments change."""
        self._update_preset_status_label()
        self._refresh_group_button_labels()
        self._schematic.set_group_names(self._current_group_names_from_edits())

    # ------------------------------------------------------------------
    # Slot: group button clicked
    # ------------------------------------------------------------------

    def _on_group_button_clicked(self, gid: int) -> None:
        """Switch the active group."""
        self._active_group = gid
        self._schematic.set_active_group(gid)

    # ------------------------------------------------------------------
    # Slot: detector toggled from schematic
    # ------------------------------------------------------------------

    def _on_detector_toggled(self, det_id: int, included: bool) -> None:
        """Update internal state when the schematic emits a toggle."""
        if included:
            self._groups.setdefault(self._active_group, set()).add(det_id)
        else:
            self._groups.setdefault(self._active_group, set()).discard(det_id)
        self._on_group_definition_changed()

    # ------------------------------------------------------------------
    # Slot: instrument combo changed
    # ------------------------------------------------------------------

    def _on_instrument_index_changed(self, _index: int) -> None:
        """Combo slot: resolve the selected item's registry key, then load it."""
        registry_key = self._instrument_combo.currentData()
        if registry_key:
            self._on_instrument_changed(str(registry_key))

    def _on_instrument_changed(self, name: str) -> None:
        """Load a different instrument layout and rebuild the schematic."""
        try:
            new_layout = get_instrument_layout(name)
        except KeyError:
            return
        self._instrument = new_layout
        self._groups = {}
        self._group_names = {}
        for edit in self._group_name_edits.values():
            edit.clear()
        self._active_group = 1
        self._group_buttons[1].setChecked(True)
        self._applied_preset_name = None
        self._populate_preset_combo()
        self._schematic.set_instrument(new_layout)
        self._sync_schematic()
        self._update_preset_status_label()
        self._update_grouping_recommendation(preselect=True)

    # ------------------------------------------------------------------
    # Slot: apply preset
    # ------------------------------------------------------------------

    def _on_apply_preset(self) -> None:
        """Apply the selected preset grouping to all group slots."""
        preset_name = self._preset_combo.currentText()
        preset = self._instrument.presets.get(preset_name)
        if preset is None:
            return

        # Clear all groups then apply preset
        self._groups = {}
        for gid in range(1, _MAX_GROUPS + 1):
            self._groups[gid] = set()

        self._group_names = {}
        for gid, gdef in preset.groups.items():
            self._groups[gid] = set(gdef.detector_ids)
            self._group_names[gid] = gdef.name
            edit = self._group_name_edits.get(gid)
            if edit is not None:
                edit.setText(gdef.name)

        # Clear name edits for unused group slots
        for gid in range(1, _MAX_GROUPS + 1):
            if gid not in preset.groups:
                self._group_names.pop(gid, None)
                edit = self._group_name_edits.get(gid)
                if edit is not None:
                    edit.clear()

        self._forward_group = preset.forward_group
        self._backward_group = preset.backward_group
        self._applied_preset_name = preset_name
        self._applied_projections = [p.to_payload() for p in preset.projections]

        self._sync_schematic()
        self._update_preset_status_label()
        self._update_grouping_recommendation()

    # ------------------------------------------------------------------
    # Slot: clear all
    # ------------------------------------------------------------------

    def _on_clear_all(self) -> None:
        """Remove all detectors from all groups."""
        confirm = QMessageBox.question(
            self,
            "Clear All Groups",
            "Remove all detectors from all groups?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        for members in self._groups.values():
            members.clear()
        self._sync_schematic()
        self._on_group_definition_changed()

    # ------------------------------------------------------------------
    # Slot: clear excluded
    # ------------------------------------------------------------------

    def _on_clear_excluded(self) -> None:
        """Remove every detector from the exclusion set."""
        self._schematic.set_excluded_detectors(set())

    # ------------------------------------------------------------------
    # Slot: OK
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        """Flush name-edit widgets into ``self._group_names`` then accept."""
        for gid, edit in self._group_name_edits.items():
            text = edit.text().strip()
            if text:
                self._group_names[gid] = text
            else:
                self._group_names.pop(gid, None)
        self.accept()

    # ------------------------------------------------------------------
    # Public result accessor
    # ------------------------------------------------------------------

    def get_result(self) -> dict[str, Any]:
        """Return the editing result as a plain dictionary.

        Returns
        -------
        dict
            Keys:

            ``"groups"``
                Mapping from group ID (int, 1-based) to list of 1-based
                detector IDs.  Only non-empty groups are included.
            ``"group_names"``
                Mapping from group ID (int, 1-based) to name string.
            ``"forward_group"``
                Group ID of the designated forward group.
            ``"backward_group"``
                Group ID of the designated backward group.
            ``"instrument"``
                Instrument name (str).
        """
        non_empty = {gid: sorted(ids) for gid, ids in self._groups.items() if ids}
        # Flush any pending name edits
        for gid, edit in self._group_name_edits.items():
            text = edit.text().strip()
            if text:
                self._group_names[gid] = text
            else:
                self._group_names.pop(gid, None)

        return {
            "groups": non_empty,
            "group_names": dict(self._group_names),
            "forward_group": self._forward_group,
            "backward_group": self._backward_group,
            "instrument": self._instrument.name,
            "grouping_preset": self._applied_preset_name,
            "excluded_detectors": sorted(self._schematic.get_excluded_detectors()),
            "projections": [dict(p) for p in self._applied_projections],
        }
