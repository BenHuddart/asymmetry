"""Model family picker: which components a fit wizard screens, by family.

Renders a :class:`~asymmetry.core.fitting.wizard_scope.ScopeView` and edits a
:class:`~asymmetry.core.fitting.wizard_scope.WizardScope`; it holds no physics.
The window injects the describer (``lambda scope: describe_scope(datasets,
scope)``), applies a direction answer to its runs with
:func:`~asymmetry.core.fitting.wizard_scope.set_user_field_direction`, then
calls :meth:`ModelFamilyPicker.refresh`. Design: ``docs/plans/model-family-picker.md`` (D5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QEnterEvent, QResizeEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass
from asymmetry.core.fitting.wizard_scope import (
    FIELD_DIRECTION_TEXT,
    RecordedGeometry,
    ScopeComponent,
    ScopeFamily,
    ScopeView,
    WizardScope,
)
from asymmetry.gui.styles import metrics, tokens
from asymmetry.gui.styles.typography import header_font
from asymmetry.gui.styles.widgets import (
    build_segmented_button_qss,
    build_segmented_cell_qss,
    build_segmented_container_qss,
    clear_layout,
    make_section_header,
)
from asymmetry.gui.widgets.flow_layout import FlowLayout

__all__ = ["ModelFamilyPicker", "PHYSICS_CHIPS"]

#: Builds the view for a scope over the window's runs.
Describer = Callable[[WizardScope], ScopeView]

#: The "Looking for" chips. Generic relaxation and background are always in scope.
PHYSICS_CHIPS: tuple[tuple[PhysicsClass, str], ...] = (
    (PhysicsClass.MAGNETISM, "Static magnetism"),
    (PhysicsClass.DYNAMICS, "Spin dynamics"),
    (PhysicsClass.SUPERCONDUCTIVITY, "Superconductivity"),
    (PhysicsClass.MOLECULAR, "F–μ–F & nuclear dipoles"),
    (PhysicsClass.MUONIUM, "Muonium"),
)

#: The direction buttons in order; ``None`` is "Not recorded".
_DIRECTIONS: tuple[FieldGeometry | None, ...] = (*FIELD_DIRECTION_TEXT, None)

#: Most cards across the grid, and the narrowest a card may get, in characters.
_MAX_COLUMNS = 3
_CARD_MIN_CHARS = 34
#: Width of the details panel, and the picker width below which it hides, in characters.
_DETAILS_CHARS = 40
_WIDE_CHARS = 140


def _muted(text: str = "", *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
    return label


def _bold(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {tokens.TEXT}; font-weight: 600;")
    return label


def _pill_qss(*, included: bool, hit: bool, dim: bool) -> str:
    """A pill's look: accent when included, a heavy accent border on a search hit."""
    border = (
        f"2px solid {tokens.ACCENT}"
        if hit
        else f"1px solid {tokens.ACCENT if included else tokens.BORDER}"
    )
    text = tokens.TEXT_DIM if dim else tokens.ACCENT if included else tokens.TEXT_MUTED
    tag = tokens.TEXT_DIM if dim else tokens.WARN
    return (
        f"QPushButton {{ border: {border}; border-radius: 10px;"
        f" background-color: {tokens.ACCENT_SOFT if included else tokens.SURFACE}; }}"
        f" QPushButton:hover {{ background-color: "
        f"{tokens.ACCENT_SOFT2 if included else tokens.SURFACE_HI}; }}"
        f" QLabel {{ color: {text}; background: transparent; }}"
        f" QLabel#slowTag {{ color: {tag}; }}"
    )


class _ComponentPill(QPushButton):
    """One component's checkable pill: its label, and a small tag when it is slow."""

    hovered = Signal(str)

    def __init__(self, component: ScopeComponent) -> None:
        super().__init__()
        self.name = component.name
        self.setCheckable(True)
        self.setAccessibleName(component.label)
        row = QHBoxLayout(self)
        row.setContentsMargins(9, 3, 9, 3)
        row.setSpacing(5)
        texts = [QLabel(component.label)]
        if component.slow:
            tag = QLabel("slow")
            tag.setObjectName("slowTag")
            tag.setFont(header_font())
            texts.append(tag)
        for label in texts:
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row.addWidget(label)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return self.layout().sizeHint()

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802 — Qt override
        super().enterEvent(event)
        self.hovered.emit(self.name)


class _FamilyCard(QFrame):
    """One family: a tri-state box, title, count, blurb, pills and what does not apply."""

    def __init__(self, family: ScopeFamily) -> None:
        super().__init__()
        self.setObjectName("familyCard")
        self.setStyleSheet(
            f"QFrame#familyCard {{ background-color: {tokens.SURFACE};"
            f" border: 1px solid {tokens.BORDER}; border-radius: 6px; }}"
        )
        self.box = QCheckBox()
        self.box.setAccessibleName(f"All {family.title} models")
        self.count = _muted()
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.box)
        head.addWidget(_bold(family.title))
        head.addStretch(1)
        head.addWidget(self.count)

        strip = QWidget()
        flow = FlowLayout(strip)
        flow.setContentsMargins(0, 0, 0, 0)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        strip.setSizePolicy(policy)
        self.pills = {c.name: _ComponentPill(c) for c in family.components}
        for pill in self.pills.values():
            flow.addWidget(pill)

        self.not_applicable = _muted(wrap=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(head)
        layout.addWidget(_muted(family.blurb, wrap=True))
        layout.addWidget(strip)
        layout.addWidget(self.not_applicable)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        # The grid fixes the width; the wrapped pills and text decide the height.
        width = self.minimumWidth()
        return QSize(width, self.heightForWidth(width))


class _CardGrid(QWidget):
    """Family cards flowing up to three across, each as wide as its column."""

    def __init__(self) -> None:
        super().__init__()
        self.flow = FlowLayout(self)
        self.flow.setContentsMargins(0, 0, 0, 0)

    def fit_cards(self) -> None:
        """Size every card to its column for the current width."""
        spacing = self.flow.horizontalSpacing()
        columns = max(
            1,
            min(
                _MAX_COLUMNS,
                (self.width() + spacing) // (metrics.char_width(_CARD_MIN_CHARS) + spacing),
            ),
        )
        # FlowLayout wraps an item that ends exactly on the edge, so leave one pixel.
        width = (self.width() - 1 - spacing * (columns - 1)) // columns
        for index in range(self.flow.count()):
            self.flow.itemAt(index).widget().setFixedWidth(width)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self.fit_cards()


class _DetailsPanel(QFrame):
    """What one component is for and why it is in or out of this scope."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("detailsPanel")
        self.setStyleSheet(
            f"QFrame#detailsPanel {{ background-color: {tokens.SURFACE};"
            f" border: 1px solid {tokens.BORDER}; border-radius: 6px; }}"
        )
        self.category = make_section_header("")
        self.label = _bold()
        self.label.setWordWrap(True)
        self.use_when = QLabel()
        self.use_when.setWordWrap(True)
        self.description = _muted(wrap=True)
        description_box = QFrame()
        description_box.setObjectName("descriptionBox")
        description_box.setStyleSheet(
            f"QFrame#descriptionBox {{ background-color: {tokens.SURFACE_ALT};"
            " border-radius: 4px; }"
        )
        QVBoxLayout(description_box).addWidget(self.description)
        description_box.layout().setContentsMargins(8, 6, 8, 6)
        facts = QGridLayout()
        facts.setHorizontalSpacing(10)
        facts.setVerticalSpacing(4)
        facts.setColumnStretch(1, 1)
        self.facts: dict[str, QLabel] = {}
        for row, key in enumerate(("Geometries", "Fitting cost", "This scope")):
            facts.addWidget(_muted(key), row, 0, Qt.AlignmentFlag.AlignTop)
            value = QLabel()
            value.setWordWrap(True)
            facts.addWidget(value, row, 1)
            self.facts[key] = value

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        for widget in (self.category, self.label, self.use_when, description_box):
            layout.addWidget(widget)
        layout.addLayout(facts)

    def show_component(self, family: ScopeFamily, component: ScopeComponent) -> None:
        self.category.setText(family.title.upper())
        self.label.setText(component.label)
        self.use_when.setText(component.use_when)
        self.description.setText(component.description)
        self.facts["Geometries"].setText(
            " · ".join(
                text for g, text in FIELD_DIRECTION_TEXT.items() if g in component.geometries
            )
        )
        self.facts["Fitting cost"].setText(
            "Slow — its fits dominate screening time" if component.slow else "Quick"
        )
        self.facts["This scope"].setText("Included" if component.included else component.reason)


def _recorded_text(geometry: RecordedGeometry) -> str:
    """``Recorded: Longitudinal``, with run counts when the files disagree."""
    if len(geometry.counts) == 1:
        return f"Recorded: {FIELD_DIRECTION_TEXT[geometry.counts[0][0]]}"
    return "Recorded: " + ", ".join(f"{FIELD_DIRECTION_TEXT[g]} ({n})" for g, n in geometry.counts)


def _direction_note(geometry: RecordedGeometry) -> str:
    """Which runs record no direction, or where the user's answer was saved."""
    recorded = sum(n for _, n in geometry.counts)
    total = geometry.unrecorded + recorded
    answered = sum(n for _, n in geometry.answered)
    if not answered:
        if recorded:
            return f"{geometry.unrecorded} of {total} runs record no field direction."
        return f"The run{'s' if total > 1 else ''} record{'' if total > 1 else 's'} no field direction."
    if total == 1:
        return "Set by you — saved on the run, which records none."
    runs = "the run that records" if answered == 1 else f"the {answered} runs that record"
    others = f"; the files record the other {recorded}" if recorded else ""
    return f"Set by you — saved on {runs} none{others}."


class ModelFamilyPicker(QWidget):
    """Pick the model families and components a fit wizard screens.

    :meth:`set_scope` is silent. A user edit re-describes the scope and emits
    :attr:`scope_changed` then :attr:`validity_changed`; :meth:`refresh` emits
    :attr:`validity_changed`, since the runs may have changed under the scope.
    A direction click only emits :attr:`direction_answered` — the window owns
    the runs, applies the answer and calls :meth:`refresh`.
    """

    #: The new :class:`WizardScope`, on every user edit.
    scope_changed = Signal(object)
    #: A :class:`FieldGeometry`, or ``None`` for "Not recorded".
    direction_answered = Signal(object)
    #: Whether a model besides the background constant is included.
    validity_changed = Signal(bool)

    def __init__(self, describe: Describer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._describe = describe
        self._view = describe(WizardScope())
        self._cards: dict[str, _FamilyCard] = {}
        self._detail_name = ""

        # ── Direction row ────────────────────────────────────────────────────
        self._direction_frame = QFrame()
        self._direction_frame.setStyleSheet(build_segmented_container_qss())
        cells = QHBoxLayout(self._direction_frame)
        cells.setContentsMargins(0, 0, 0, 0)
        cells.setSpacing(0)
        self._direction_group = QButtonGroup(self)
        labels = (*FIELD_DIRECTION_TEXT.values(), "Not recorded")
        for index, text in enumerate(labels):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setStyleSheet(
                build_segmented_cell_qss(first=index == 0, last=index == len(labels) - 1)
            )
            self._direction_group.addButton(button, index)
            cells.addWidget(button)
        self._direction_group.idClicked.connect(
            lambda index: self.direction_answered.emit(_DIRECTIONS[index])
        )
        self._recorded = _bold()
        self._direction_note = _muted()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Find a model — Keren, F–μ–F, helix…")
        self._search.setAccessibleName("Find a model")
        self._search.setClearButtonEnabled(True)
        self._search.setMinimumWidth(metrics.field_width_for(30))
        self._search.textChanged.connect(self._render)

        direction_row = QHBoxLayout()
        direction_row.setSpacing(10)
        direction_row.addWidget(_muted("Field direction"))
        direction_row.addWidget(self._direction_frame)
        direction_row.addWidget(self._recorded)
        direction_row.addWidget(self._direction_note)
        direction_row.addStretch(1)
        direction_row.addWidget(self._search)

        # ── Physics chips ────────────────────────────────────────────────────
        chip_strip = QWidget()
        chip_flow = FlowLayout(chip_strip)
        chip_flow.setContentsMargins(0, 0, 0, 0)
        chip_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        chip_policy.setHeightForWidth(True)
        chip_strip.setSizePolicy(chip_policy)
        self._chips: dict[PhysicsClass, QPushButton] = {}
        for physics, text in PHYSICS_CHIPS:
            chip = QPushButton(text.replace("&", "&&"))  # a lone & would be a mnemonic
            chip.setCheckable(True)
            chip.setStyleSheet(build_segmented_button_qss())
            chip.clicked.connect(lambda on, p=physics: self._look_for(p, on))
            chip_flow.addWidget(chip)
            self._chips[physics] = chip
        chip_row = QHBoxLayout()
        chip_row.setSpacing(10)
        chip_row.addWidget(_muted("Looking for (optional)"), 0, Qt.AlignmentFlag.AlignTop)
        chip_row.addWidget(chip_strip, 1)

        # ── Cards and details ────────────────────────────────────────────────
        self._grid = _CardGrid()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._grid)
        self._details = _DetailsPanel()
        self._details.setFixedWidth(metrics.char_width(_DETAILS_CHARS))
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(scroll, 1)
        body.addWidget(self._details, 0, Qt.AlignmentFlag.AlignTop)

        # ── Footer ───────────────────────────────────────────────────────────
        self._screen_count = QLabel()
        self._slow_line = QLabel()
        self._slow_line.setWordWrap(True)
        self._skip_slow = QCheckBox("Leave out slow models")
        self._skip_slow.clicked.connect(lambda on: self._edit(replace(self.scope(), skip_slow=on)))
        self._reset = QPushButton("Reset to suggestions")
        self._reset.setStyleSheet(  # a text-only button that reads as a link
            f"QPushButton {{ border: none; background: transparent; padding: 0;"
            f" color: {tokens.ACCENT}; }}"
            " QPushButton:hover { text-decoration: underline; }"
        )
        self._reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset.setToolTip("Drop your own model choices; keep what you are looking for.")
        self._reset.clicked.connect(
            lambda: self._edit(
                replace(
                    self.scope(), include_components=frozenset(), exclude_components=frozenset()
                )
            )
        )
        counts = QVBoxLayout()
        counts.setSpacing(2)
        counts.addWidget(self._screen_count)
        counts.addWidget(self._slow_line)
        footer = QHBoxLayout()
        footer.setSpacing(16)
        footer.addLayout(counts, 1)
        footer.addWidget(self._reset, 0, Qt.AlignmentFlag.AlignTop)
        footer.addWidget(self._skip_slow, 0, Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(direction_row)
        layout.addLayout(chip_row)
        layout.addLayout(body, 1)
        layout.addLayout(footer)
        self._render()

    # ── Public API ───────────────────────────────────────────────────────────

    def scope(self) -> WizardScope:
        return self._view.scope

    def set_scope(self, scope: WizardScope) -> None:
        """Show *scope* over the current runs; emits nothing."""
        self._view = self._describe(scope)
        self._render()

    def refresh(self) -> None:
        """Re-describe the scope after the runs changed, e.g. a direction answer."""
        self.set_scope(self.scope())
        self.validity_changed.emit(self.is_valid())

    def is_valid(self) -> bool:
        return self._view.screens_a_model

    # ── Edits ────────────────────────────────────────────────────────────────

    def _edit(self, scope: WizardScope) -> None:
        self.set_scope(scope)
        self.scope_changed.emit(scope)
        self.validity_changed.emit(self.is_valid())

    def _look_for(self, physics: PhysicsClass, on: bool) -> None:
        current = self.scope().physics
        self._edit(
            replace(self.scope(), physics=current | {physics} if on else current - {physics})
        )

    def _switch(self, names: frozenset[str], on: bool) -> None:
        """Switch components on or off by overriding the scope's own choice."""
        scope = self.scope()
        include, exclude = scope.include_components, scope.exclude_components
        self._edit(
            replace(
                scope,
                include_components=include | names if on else include - names,
                exclude_components=exclude - names if on else exclude | names,
            )
        )

    def _switch_family(self, title: str) -> None:
        """Switch every applicable component on, or off when all are already on."""
        family = next(f for f in self._view.families if f.title == title)
        applicable = [c for c in family.components if c.applies]
        on = not all(c.included for c in applicable)
        self._switch(frozenset(c.name for c in applicable if c.included != on), on)

    def _click_pill(self, name: str, on: bool) -> None:
        self._detail_name = name
        self._switch(frozenset({name}), on)

    def _show_details(self, name: str) -> None:
        self._detail_name = name
        self._details.show_component(*self._find(name))

    def _find(self, name: str) -> tuple[ScopeFamily, ScopeComponent]:
        return next(
            (family, component)
            for family in self._view.families
            for component in family.components
            if component.name == name
        )

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self) -> None:
        view = self._view
        structure = [(f.title, [c.name for c in f.components]) for f in view.families]
        if structure != [(title, list(card.pills)) for title, card in self._cards.items()]:
            clear_layout(self._grid.flow)
            self._cards = {}
            for family in view.families:
                card = _FamilyCard(family)
                card.box.clicked.connect(lambda _on=False, t=family.title: self._switch_family(t))
                for name, pill in card.pills.items():
                    pill.clicked.connect(lambda on, n=name: self._click_pill(n, on))
                    pill.hovered.connect(self._show_details)
                self._grid.flow.addWidget(card)
                self._cards[family.title] = card
            self._grid.fit_cards()
            self._detail_name = view.families[0].components[0].name

        geometry = view.geometry
        self._direction_frame.setVisible(geometry.editable)
        self._direction_note.setVisible(geometry.editable)
        self._recorded.setVisible(not geometry.editable)
        self._direction_group.button(_DIRECTIONS.index(geometry.answer)).setChecked(True)
        self._direction_note.setText(_direction_note(geometry))
        self._recorded.setText(_recorded_text(geometry))

        for physics, chip in self._chips.items():
            chip.setChecked(physics in view.scope.physics)

        run_geometries = {g for g, _ in geometry.counts + geometry.answered}
        run_geometry_text = "/".join(g.value for g in FieldGeometry if g in run_geometries)
        query = self._search.text().strip().casefold()
        for family in view.families:
            card = self._cards[family.title]
            applicable = [c for c in family.components if c.applies]
            included = sum(c.included for c in applicable)
            card.box.setEnabled(bool(applicable))
            card.box.setCheckState(
                Qt.CheckState.Unchecked
                if included == 0
                else Qt.CheckState.Checked
                if included == len(applicable)
                else Qt.CheckState.PartiallyChecked
            )
            card.count.setText(f"{included} of {len(applicable)}" if applicable else "none apply")
            for component in family.components:
                pill = card.pills[component.name]
                hit = bool(query) and query in (
                    f"{component.label} {component.name} {component.use_when}".casefold()
                )
                pill.setVisible(component.applies)
                pill.setChecked(component.included)
                pill.setToolTip(
                    component.use_when
                    if component.included
                    else f"{component.use_when}\n{component.reason}"
                )
                pill.setStyleSheet(
                    _pill_qss(included=component.included, hit=hit, dim=bool(query) and not hit)
                )
            # What does not apply is either another geometry or, at a known TF field,
            # a muonium form outside its field regime.
            off = [c for c in family.components if not c.applies]
            off_geometry = [c.label for c in off if not c.geometries & run_geometries]
            off_regime = [c.label for c in off if c.geometries & run_geometries]
            lines = [
                f"{heading}: {' · '.join(labels)}"
                for heading, labels in (
                    (f"Not {run_geometry_text} models", off_geometry),
                    ("Outside these runs' field range", off_regime),
                )
                if labels
            ]
            card.not_applicable.setText("\n".join(lines))
            card.not_applicable.setVisible(bool(lines))

        self._details.show_component(*self._find(self._detail_name))

        valid = view.screens_a_model
        self._screen_count.setText(
            f"Will screen {view.included_count} of {view.applicable_count} models"
            if valid
            else "Only the background is included — switch on a model to screen."
        )
        self._screen_count.setStyleSheet(
            f"color: {tokens.TEXT if valid else tokens.ERROR}; font-weight: 600;"
        )
        slow = view.slow_included
        self._slow_line.setText(
            f"{len(slow)} slow: {' · '.join(c.label for c in slow)}"
            if slow
            else "No slow models included."
        )
        self._slow_line.setStyleSheet(f"color: {tokens.WARN if slow else tokens.TEXT_MUTED};")
        self._skip_slow.setChecked(view.scope.skip_slow)
        self._reset.setVisible(bool(view.scope.include_components or view.scope.exclude_components))

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        # Narrow windows lose the details panel; each pill's tooltip still says what it is for.
        self._details.setVisible(event.size().width() >= metrics.char_width(_WIDE_CHARS))
