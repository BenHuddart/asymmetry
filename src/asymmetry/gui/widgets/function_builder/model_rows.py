"""Structured model-row editor over the flat composite structure.

``ModelRowList`` is a structured editor for a composite model expressed as five
parallel lists (the same shape ``parse_composite_expression`` /
``build_component_expression`` speak):

- ``component_names: list[str]``
- ``operators: list[str]`` (length ``len(component_names) - 1``)
- ``open_parentheses: list[int]`` (length ``len(component_names)``)
- ``close_parentheses: list[int]`` (length ``len(component_names)``)
- ``fraction_groups: list[tuple[int, int]]`` (each maps to one paren span)

This flat state is the single source of truth. Rendering derives a container
tree from the parenthesis counts and paints it recursively; every editing
operation mutates the flat state, re-renders, and emits ``structure_changed``.

The fiddly part is index arithmetic under insert/delete: inserting a component
at index ``i`` shifts every later component's operator/paren counts by one and
remaps every ``(start, end)`` fraction-group pair; deleting does the reverse.
These are handled by :meth:`_insert_component_at` / :meth:`_remove_component_at`
and covered exhaustively by ``tests/gui/test_function_builder_rows.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite import build_component_expression
from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.formatting import format_param_label

#: Character budget for the elided parameter summary label. Rows are laid out
#: before the dialog is fully sized, so eliding to a fixed character budget
#: (rather than a measured pixel width) keeps the summary readable without a
#: relayout dependency; :class:`QLabel` still elides further if the row is
#: squeezed narrower than this budget by the layout.
_PARAM_SUMMARY_CHAR_BUDGET = 60

#: Accent palette cycled per fraction group (moved from the old fit builder).
#: Public alias ``FRACTION_GROUP_COLORS`` is the sanctioned import for other
#: modules (e.g. the preview card in ``function_builder/dialog.py``) that
#: need to match a fraction group's row-container accent color; index by the
#: group's position in ``sorted(model.fraction_groups)``, same as here.
_FRACTION_GROUP_COLORS = ["#005A9C", "#A44A00", "#0B6E4F", "#8A1C1C", "#6B4F00"]
FRACTION_GROUP_COLORS = _FRACTION_GROUP_COLORS

#: Internal drag mime type: a row reorder carries its component index.
_ROW_MIME_TYPE = "application/x-asymmetry-model-row"

_DEFAULT_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")


@dataclass
class _Node:
    """A node in the derived container tree.

    A leaf node wraps a single component (``component_index`` set); a container
    node has ``children`` and a ``(start, end)`` component span.
    """

    kind: str  # "leaf" | "container"
    component_index: int | None = None
    start: int = 0
    end: int = 0
    is_fraction: bool = False
    children: list[_Node] = field(default_factory=list)


def _pretty_param_name(name: str) -> str:
    """Return the shared display label for *name*, falling back to the raw name."""
    try:
        label = format_param_label(name)
    except Exception:
        return name
    return label or name


def _build_forest(
    open_parentheses: Sequence[int],
    close_parentheses: Sequence[int],
    fraction_groups: set[tuple[int, int]],
) -> list[_Node]:
    """Derive the container forest from the paren counts.

    Each balanced paren span becomes a container node; bare components become
    leaves. Returns the top-level (depth-0) node list in component order.
    """
    n = len(open_parentheses)
    root_children: list[_Node] = []
    # Stack of (children_list, start_index) for each open paren.
    stack: list[tuple[list[_Node], int]] = []
    current = root_children

    for idx in range(n):
        for _ in range(open_parentheses[idx]):
            new_children: list[_Node] = []
            stack.append((new_children, idx))
            current = new_children

        current.append(_Node(kind="leaf", component_index=idx, start=idx, end=idx))

        for _ in range(close_parentheses[idx]):
            if not stack:
                raise ValueError("Invalid parentheses: closing before opening")
            children, start = stack.pop()
            container = _Node(
                kind="container",
                start=start,
                end=idx,
                is_fraction=(start, idx) in fraction_groups,
                children=children,
            )
            current = stack[-1][0] if stack else root_children
            current.append(container)

    if stack:
        raise ValueError("Invalid parentheses: unbalanced expression")
    return root_children


class _RowWidget(QFrame):
    """A single component row: [operator | name | params | duplicate | delete]."""

    def __init__(
        self,
        owner: ModelRowList,
        component_index: int,
        *,
        first_in_container: bool,
    ) -> None:
        super().__init__()
        self._owner = owner
        self._component_index = component_index
        self.setObjectName("modelRow")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        names = owner._component_names
        operators = owner._operators
        definitions = owner._component_definitions
        name = names[component_index]
        definition = definitions.get(name)
        known = definition is not None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        grip = QLabel("☰")  # trigram-for-heaven "hamburger" grip glyph
        grip.setObjectName("rowGrip")
        grip.setToolTip("Drag to reorder within this group")
        grip.setCursor(Qt.CursorShape.OpenHandCursor)
        # Subtle hover feedback makes the grip read as draggable rather than
        # decorative; requires WA_Hover so QSS :hover fires on a plain QLabel.
        grip.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        grip.setStyleSheet(
            f"#rowGrip {{ color: {tokens.TEXT_DIM}; padding: 1px 3px; border-radius: 3px; }}"
            f"#rowGrip:hover {{ color: {tokens.TEXT}; background: {tokens.SURFACE_HI}; }}"
        )
        self._grip = grip
        layout.addWidget(grip)

        self._operator_combo: QComboBox | None = None
        if first_in_container:
            dash = QLabel("–")  # en-dash placeholder (no leading operator)
            dash.setFixedWidth(38)
            dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dash.setStyleSheet(f"color: {tokens.TEXT_DIM};")
            layout.addWidget(dash)
        else:
            combo = QComboBox()
            combo.addItems(list(owner._operators_available))
            combo.setFixedWidth(52)
            current_op = operators[component_index - 1]
            index = combo.findText(current_op)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.currentTextChanged.connect(self._on_operator_changed)
            self._operator_combo = combo
            layout.addWidget(combo)

        name_label = QLabel(name)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        if not known:
            name_label.setStyleSheet(f"color: {tokens.WARN};")
            name_label.setToolTip("Unknown component (missing user function)")
        layout.addWidget(name_label)

        if known:
            param_names = list(getattr(definition, "param_names", []) or [])
            pretty_names = [_pretty_param_name(p) for p in param_names]
            summary = ", ".join(pretty_names)
        else:
            summary = "unknown"
        summary_label = QLabel()
        summary_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        summary_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        summary_label.setToolTip(summary)
        elided = QFontMetrics(summary_label.font()).elidedText(
            summary,
            Qt.TextElideMode.ElideRight,
            QFontMetrics(summary_label.font()).averageCharWidth() * _PARAM_SUMMARY_CHAR_BUDGET,
        )
        summary_label.setText(elided)
        layout.addWidget(summary_label, 1)

        duplicate = QPushButton("Duplicate")
        duplicate.setToolTip("Duplicate this component")
        duplicate.clicked.connect(lambda: owner.duplicate_row(self._component_index))
        layout.addWidget(duplicate)

        delete = QPushButton("Delete")
        delete.setToolTip("Remove this component")
        delete.clicked.connect(lambda: owner.delete_row(self._component_index))
        layout.addWidget(delete)

        self._refresh_style()

    def _on_operator_changed(self, text: str) -> None:
        self._owner._set_operator(self._component_index, text)

    def _refresh_style(self) -> None:
        selected = self._component_index in self._owner._selected_indices
        if selected:
            self.setStyleSheet(
                f"#modelRow {{ background: {tokens.ACCENT_SOFT}; "
                f"border: 1px solid {tokens.ACCENT}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet(
                f"#modelRow {{ background: {tokens.SURFACE}; "
                f"border: 1px solid {tokens.BORDER}; border-radius: 4px; }}"
            )

    # -- selection + drag ---------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            self._owner._handle_row_click(self._component_index, modifiers)
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        start = getattr(self, "_drag_start", None)
        if start is None:
            return
        if (event.position().toPoint() - start).manhattanLength() < 12:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_ROW_MIME_TYPE, str(self._component_index).encode("ascii"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)


class _DropTarget(QFrame):
    """A thin drop zone between sibling rows in the same container."""

    def __init__(self, owner: ModelRowList, container_span: tuple[int, int], before_index: int):
        super().__init__()
        self._owner = owner
        self._container_span = container_span
        self._before_index = before_index
        self.setAcceptDrops(True)
        self.setFixedHeight(6)
        self.setStyleSheet("background: transparent;")

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_ROW_MIME_TYPE):
            event.acceptProposedAction()
            self.setStyleSheet(f"background: {tokens.ACCENT};")

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setStyleSheet("background: transparent;")

    def dropEvent(self, event) -> None:  # noqa: N802
        self.setStyleSheet("background: transparent;")
        raw = bytes(event.mimeData().data(_ROW_MIME_TYPE)).decode("ascii")
        try:
            source_index = int(raw)
        except ValueError:
            return
        if self._owner._drop_row(source_index, self._container_span, self._before_index):
            event.acceptProposedAction()


class ModelRowList(QWidget):
    """Structured editor over the flat composite structure.

    The five parallel lists are the single source of truth; the widget tree is
    rebuilt from them on every change. Emits :attr:`structure_changed` after any
    mutation and :attr:`selection_changed` when the selected rows change.
    """

    structure_changed = Signal()
    selection_changed = Signal()

    def __init__(
        self,
        component_definitions: Mapping[str, object],
        *,
        operators: Sequence[str] = _DEFAULT_OPERATORS,
        enable_fraction_groups: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._component_definitions: dict[str, object] = dict(component_definitions)
        self._operators_available: tuple[str, ...] = tuple(operators)
        self._enable_fraction_groups = enable_fraction_groups

        self._component_names: list[str] = []
        self._operators: list[str] = []
        self._open_parentheses: list[int] = []
        self._close_parentheses: list[int] = []
        self._fraction_groups: list[tuple[int, int]] = []
        self._selected_indices: set[int] = set()
        self._anchor_index: int | None = None

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        self._body_layout.setSpacing(4)
        self._outer.addWidget(self._body)

        self._empty_label = QLabel("Add a function from the library")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; padding: 24px;")

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._render()

    # ------------------------------------------------------------------ API
    def set_structure(
        self,
        component_names: Sequence[str],
        operators: Sequence[str],
        open_parentheses: Sequence[int],
        close_parentheses: Sequence[int],
        fraction_groups: Sequence[tuple[int, int]],
    ) -> None:
        """Replace the whole structure and re-render (does not emit)."""
        self._component_names = list(component_names)
        self._operators = list(operators)
        self._open_parentheses = list(open_parentheses)
        self._close_parentheses = list(close_parentheses)
        self._fraction_groups = sorted({(int(s), int(e)) for s, e in fraction_groups})
        self._selected_indices.clear()
        self._anchor_index = None
        self._render()

    def structure(
        self,
    ) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
        """Return copies of the five structure lists."""
        return (
            list(self._component_names),
            list(self._operators),
            list(self._open_parentheses),
            list(self._close_parentheses),
            list(self._fraction_groups),
        )

    def expression(self) -> str:
        """Return the canonical builder expression (includes ``{frac}``)."""
        return build_component_expression(
            self._component_names,
            self._operators,
            self._open_parentheses,
            self._close_parentheses,
            self._fraction_groups,
        )

    def clear(self) -> None:
        self._component_names = []
        self._operators = []
        self._open_parentheses = []
        self._close_parentheses = []
        self._fraction_groups = []
        self._selected_indices.clear()
        self._anchor_index = None
        self._render()
        self.structure_changed.emit()

    def append_component(self, name: str) -> None:
        """Append *name* at the end of its context.

        If a single row is selected, insert it right after that row inside the
        same container joined with ``+``; otherwise append at the top level with
        ``+``.
        """
        if self._selected_indices and len(self._selected_indices) == 1:
            after = max(self._selected_indices)
            self._insert_component_at(after + 1, name, operator="+", opens=0, closes=0)
        else:
            insert_at = len(self._component_names)
            self._insert_component_at(insert_at, name, operator="+", opens=0, closes=0)
        self._render()
        self.structure_changed.emit()

    def selected_spans(self) -> list[tuple[int, int]]:
        """Return contiguous selected index runs as ``(start, end)`` spans."""
        if not self._selected_indices:
            return []
        ordered = sorted(self._selected_indices)
        spans: list[tuple[int, int]] = []
        run_start = ordered[0]
        prev = ordered[0]
        for idx in ordered[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            spans.append((run_start, prev))
            run_start = idx
            prev = idx
        spans.append((run_start, prev))
        return spans

    # ----------------------------------------------------- index arithmetic
    def _shift_groups_for_insert(self, at: int) -> None:
        """Remap fraction groups for a component inserted at *at*."""
        self._fraction_groups = sorted(
            ((s + 1 if s >= at else s, e + 1 if e >= at else e) for s, e in self._fraction_groups)
        )

    def _shift_groups_for_delete(self, at: int) -> None:
        """Remap fraction groups for the component removed at *at*.

        A group whose range collapses to a single term (or invalid) is dropped.
        """
        remapped: list[tuple[int, int]] = []
        for s, e in self._fraction_groups:
            new_s = s - 1 if s > at else s
            new_e = e - 1 if e > at else e
            if new_s < new_e:
                remapped.append((new_s, new_e))
        self._fraction_groups = sorted(set(remapped))

    def _insert_component_at(
        self,
        at: int,
        name: str,
        *,
        operator: str,
        opens: int,
        closes: int,
    ) -> None:
        """Insert *name* at component index *at*, fixing all parallel lists.

        The operator joins the new component to its predecessor when ``at > 0``;
        when inserting at index 0 the operator slot for the *old* head is added
        instead so the head still has no leading operator.
        """
        n = len(self._component_names)
        at = max(0, min(at, n))

        self._component_names.insert(at, name)
        self._open_parentheses.insert(at, opens)
        self._close_parentheses.insert(at, closes)

        # operators[i] joins component i to component i-1. Inserting a component
        # at `at` adds one join. When at==0 the new head has no operator, so the
        # join we add sits before the *old* head (operator index 0).
        if n == 0:
            pass  # first component: no operator
        elif at == 0:
            self._operators.insert(0, operator)
        else:
            self._operators.insert(at - 1, operator)

        self._shift_groups_for_insert(at)

    def _remove_component_at(self, at: int) -> None:
        """Remove the component at index *at*, fixing all parallel lists."""
        n = len(self._component_names)
        if not (0 <= at < n):
            return

        # Preserve paren balance: the removed component may carry opens/closes.
        # Reattach a *balanced* net to a surviving neighbour so the expression
        # stays parseable; unbalanced remainder is dropped (dissolves container).
        opens = self._open_parentheses[at]
        closes = self._close_parentheses[at]

        del self._component_names[at]
        del self._open_parentheses[at]
        del self._close_parentheses[at]

        # Remove the operator that joined this component. If it had a leading
        # operator (at > 0) drop operators[at-1]; else drop operators[0] (the
        # join to the new head), if any.
        if self._operators:
            op_index = at - 1 if at > 0 else 0
            op_index = max(0, min(op_index, len(self._operators) - 1))
            del self._operators[op_index]

        # Re-home surplus parens onto a neighbour so nesting survives a middle
        # deletion (e.g. deleting the opener of a container hands its opens to
        # the next surviving member; its closes to the previous member).
        surviving = len(self._component_names)
        if surviving:
            if opens:
                # Hand the removed opener's opens to the next surviving member.
                target = min(at, surviving - 1)
                self._open_parentheses[target] += opens
            if closes:
                # Hand the removed closer's closes to the previous surviving member.
                target = min(max(at - 1, 0), surviving - 1)
                self._close_parentheses[target] += closes

        self._shift_groups_for_delete(at)
        self._rebalance_dropping_unbalanced()

    def _rebalance_dropping_unbalanced(self) -> None:
        """Trim any net-unbalanced parens so the structure stays parseable.

        After deletions the paren counts can become unbalanced; strip the
        surplus from the outermost positions and drop now-invalid groups.
        """
        n = len(self._component_names)
        if n == 0:
            self._open_parentheses = []
            self._close_parentheses = []
            self._fraction_groups = []
            return

        # Drop closes that have no matching open (scan left→right).
        balance = 0
        for idx in range(n):
            balance += self._open_parentheses[idx]
            excess_close = max(self._close_parentheses[idx] - balance, 0)
            if excess_close:
                self._close_parentheses[idx] -= excess_close
            balance -= self._close_parentheses[idx]
        # Drop opens left unmatched at the end.
        if balance > 0:
            remaining = balance
            for idx in range(n - 1, -1, -1):
                if remaining <= 0:
                    break
                take = min(self._open_parentheses[idx], remaining)
                self._open_parentheses[idx] -= take
                remaining -= take

        # Strip redundant parens that now wrap a single component (start==end):
        # a lone-term container carries no grouping meaning and would render as a
        # spurious "Group" frame. Peel one matching open/close per such pair.
        changed = True
        while changed:
            changed = False
            for start, end in sorted(self._parenthesized_spans()):
                if start == end and self._open_parentheses[start] and self._close_parentheses[end]:
                    self._open_parentheses[start] -= 1
                    self._close_parentheses[end] -= 1
                    changed = True
                    break

        # Drop fraction groups that no longer map to a real paren span.
        valid_spans = self._parenthesized_spans()
        self._fraction_groups = sorted(
            g for g in set(self._fraction_groups) if g in valid_spans and g[0] < g[1]
        )

    def _parenthesized_spans(self) -> set[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        stack: list[int] = []
        for idx in range(len(self._component_names)):
            for _ in range(self._open_parentheses[idx]):
                stack.append(idx)
            for _ in range(self._close_parentheses[idx]):
                if stack:
                    spans.add((stack.pop(), idx))
        return spans

    # ------------------------------------------------------------ mutations
    def _set_operator(self, component_index: int, operator: str) -> None:
        if component_index <= 0 or component_index - 1 >= len(self._operators):
            return
        if self._operators[component_index - 1] == operator:
            return
        self._operators[component_index - 1] = operator
        # Changing a join can invalidate a fraction group (needs all '+').
        self._fraction_groups = [
            g
            for g in self._fraction_groups
            if all(op == "+" for op in self._operators[g[0] : g[1]])
        ]
        self._render()
        self.structure_changed.emit()

    def duplicate_row(self, component_index: int) -> None:
        if not (0 <= component_index < len(self._component_names)):
            return
        name = self._component_names[component_index]
        # Same operator as the source; if the source is a container head the
        # duplicate simply joins with '+' inside the same context.
        if component_index > 0:
            operator = self._operators[component_index - 1]
        else:
            operator = self._operators[0] if self._operators else "+"

        insert_at = component_index + 1
        group = self._group_containing(component_index)
        # Was the duplicated row the group's LAST member? If so, the insertion
        # point (insert_at == group end + 1) lands strictly after the group's
        # remapped end, so `_shift_groups_for_insert` (called by
        # `_insert_component_at` below) does NOT extend the group — extending it
        # here is our job. If the duplicated row was any earlier member, the
        # insertion point falls inside the group's remapped span, so the shift
        # itself already grew the group's end by one; calling
        # `_extend_group_end` on top would double-extend it and swallow the
        # next sibling (see tests/gui/test_function_builder_rows.py).
        is_last_member = group is not None and component_index == group[1]
        self._insert_component_at(insert_at, name, operator=operator, opens=0, closes=0)

        if is_last_member:
            self._extend_group_end(group, insert_at)

        self._render()
        self.structure_changed.emit()

    def _group_containing(self, component_index: int) -> tuple[int, int] | None:
        for s, e in self._fraction_groups:
            if s <= component_index <= e:
                return (s, e)
        return None

    def _extend_group_end(self, original_group: tuple[int, int], inserted_at: int) -> None:
        """Grow a fraction group to include a component inserted within it.

        ``_shift_groups_for_insert`` has already run, so ``original_group`` was
        remapped: its start is unchanged (insert was after start) and its end
        moved to ``end + 1`` because ``end >= inserted_at``. We must extend that
        remapped group's end by one so the new component is inside.
        """
        s, e = original_group
        remapped_start = s + 1 if s >= inserted_at else s
        remapped_end = e + 1 if e >= inserted_at else e
        new_groups: list[tuple[int, int]] = []
        for gs, ge in self._fraction_groups:
            if (gs, ge) == (remapped_start, remapped_end):
                new_groups.append((gs, ge + 1))
                # The inserted component's own opens/closes are zero, so it
                # lives inside the existing paren span — extend the container's
                # closing paren to the new member.
                if self._close_parentheses[ge]:
                    self._close_parentheses[ge] -= 1
                    self._close_parentheses[ge + 1] += 1
            else:
                new_groups.append((gs, ge))
        self._fraction_groups = sorted(set(new_groups))

    def delete_row(self, component_index: int) -> None:
        if not (0 <= component_index < len(self._component_names)):
            return
        self._remove_component_at(component_index)
        self._selected_indices.discard(component_index)
        self._selected_indices = {
            (i - 1 if i > component_index else i) for i in self._selected_indices
        }
        self._anchor_index = None
        self._render()
        self.structure_changed.emit()

    def delete_selected(self) -> None:
        if not self._selected_indices:
            return
        for idx in sorted(self._selected_indices, reverse=True):
            self._remove_component_at(idx)
        self._selected_indices.clear()
        self._anchor_index = None
        self._render()
        self.structure_changed.emit()

    def _container_of(self, component_index: int) -> tuple[int, int]:
        """Return the innermost paren span containing *component_index*.

        Falls back to the whole top level ``(0, n-1)`` when the component is not
        inside any parentheses.
        """
        best: tuple[int, int] | None = None
        for s, e in self._parenthesized_spans():
            if s <= component_index <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e)
        if best is not None:
            return best
        return (0, max(len(self._component_names) - 1, 0))

    def _sibling_term_ranges(self, container: tuple[int, int]) -> list[tuple[int, int]]:
        """Return top-level child spans (leaf or sub-container) of *container*.

        Splits the container's inner components at depth-0 operator positions,
        where depth is measured relative to the container.
        """
        start, end = container
        if start > end:
            return []
        inside_container = container in self._parenthesized_spans()
        base = 1 if inside_container else 0

        ranges: list[tuple[int, int]] = []
        depth = self._open_parentheses[start] - base
        term_start = start
        for idx in range(start, end):
            depth_after = depth - self._close_parentheses[idx]
            if depth_after == 0:
                ranges.append((term_start, idx))
                term_start = idx + 1
            depth = depth_after + self._open_parentheses[idx + 1]
        ranges.append((term_start, end))
        return ranges

    def _parent_container_for_move(self, component_index: int) -> tuple[int, int]:
        """Return the container whose siblings include *component_index* as a term.

        Unlike :meth:`_container_of` (which returns the innermost paren span a
        component sits in), this returns the container *one level up* when the
        component is itself a container's opener — so moving a row that heads a
        sub-container reorders that whole container among its siblings. Chooses
        the smallest candidate container in which the index is a term-start.
        """
        top = (0, max(len(self._component_names) - 1, 0))
        candidates = [top, *self._parenthesized_spans()]
        best: tuple[int, int] | None = None
        best_term_end = -1
        for span in candidates:
            if span[0] > span[1]:
                continue
            for term_start, term_end in self._sibling_term_ranges(span):
                if term_start != component_index:
                    continue
                # Prefer the container in which the term starting here spans the
                # most components (moves a whole sub-container as one unit);
                # break ties toward the smaller enclosing container.
                if term_end > best_term_end or (
                    term_end == best_term_end
                    and best is not None
                    and (span[1] - span[0]) < (best[1] - best[0])
                ):
                    best = span
                    best_term_end = term_end
        return best if best is not None else top

    def move_row(self, component_index: int, delta: int) -> None:
        """Move a row (and its sub-container) up/down past a sibling.

        ``delta`` is ``-1`` (up) or ``+1`` (down). Movement is constrained to
        siblings within the same container; a whole sub-container moves as a
        unit. Renders + emits ``structure_changed`` once.
        """
        if not self._move_row_no_render(component_index, delta):
            return
        self._selected_indices.clear()
        self._anchor_index = None
        self._render()
        self.structure_changed.emit()

    def _move_row_no_render(self, component_index: int, delta: int) -> bool:
        """Move a row (and its sub-container) past a sibling (no render, no emit).

        Structural counterpart of :meth:`move_row`, factored out so
        :meth:`_drop_row` can chain several single-step moves and pay for a
        render + ``structure_changed`` emission only once, at the end, rather
        than once per intermediate swap.
        """
        if delta not in (-1, 1):
            return False
        container = self._parent_container_for_move(component_index)
        terms = self._sibling_term_ranges(container)
        pos = next((i for i, (s, e) in enumerate(terms) if s <= component_index <= e), None)
        if pos is None:
            return False
        target = pos + delta
        if not (0 <= target < len(terms)):
            return False
        return self._swap_terms_no_render(terms[pos], terms[target])

    def _swap_terms(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        """Swap two adjacent sibling spans as whole units, then render + emit once."""
        if not self._swap_terms_no_render(left, right):
            return
        self._selected_indices.clear()
        self._anchor_index = None
        self._render()
        self.structure_changed.emit()

    def _swap_terms_no_render(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        """Swap two adjacent sibling spans as whole units (no render, no emit).

        Pure structural-list arithmetic factored out of :meth:`_swap_terms` so
        :meth:`_drop_row` can perform a multi-step reorder as repeated swaps
        without paying for a full re-render + ``structure_changed`` emission on
        every intermediate step. Returns ``True`` when a swap was applied
        (``left``/``right`` were adjacent); ``False`` (no-op) otherwise.
        """
        if left[0] > right[0]:
            left, right = right, left
        # Only adjacent swaps are supported (right must immediately follow left).
        if right[0] != left[1] + 1:
            return False

        def slice_of(lists: list, span: tuple[int, int]) -> list:
            return lists[span[0] : span[1] + 1]

        li0, li1 = left
        ri0, ri1 = right

        names = self._component_names
        opens = self._open_parentheses
        closes = self._close_parentheses

        left_names = slice_of(names, left)
        right_names = slice_of(names, right)
        left_opens = slice_of(opens, left)
        right_opens = slice_of(opens, right)
        left_closes = slice_of(closes, left)
        right_closes = slice_of(closes, right)

        # The operator that joined left→right stays between the swapped spans.
        joining_operator = self._operators[li1]  # operator before right's head

        # Rebuild names / parens with right block first, then left block.
        new_names = names[:li0] + right_names + left_names + names[ri1 + 1 :]
        new_opens = opens[:li0] + right_opens + left_opens + opens[ri1 + 1 :]
        new_closes = closes[:li0] + right_closes + left_closes + closes[ri1 + 1 :]

        # Operators: operators[i] joins component i to i-1. The internal
        # operators of each block move with the block; the join between the two
        # blocks (originally operators[li1]) stays at the new boundary.
        left_len = li1 - li0 + 1
        right_len = ri1 - ri0 + 1
        left_internal = self._operators[li0:li1]  # joins within left block
        right_internal = self._operators[ri0:ri1]  # joins within right block
        before = self._operators[:li0]  # includes join into left's head if any
        after = self._operators[ri1:]  # join after right block onward

        # after[0] (if present) is operators[ri1], the join after right block.
        new_operators = before + right_internal + [joining_operator] + left_internal + after

        self._component_names = new_names
        self._open_parentheses = new_opens
        self._close_parentheses = new_closes
        self._operators = new_operators

        # Remap fraction groups: build an old→new index map for the moved range.
        index_map = {i: i for i in range(len(names))}
        # Right block occupies [li0, li0+right_len)
        for offset in range(right_len):
            index_map[ri0 + offset] = li0 + offset
        # Left block occupies [li0+right_len, li0+right_len+left_len)
        for offset in range(left_len):
            index_map[li0 + offset] = li0 + right_len + offset

        remapped: list[tuple[int, int]] = []
        for gs, ge in self._fraction_groups:
            ns, ne = index_map.get(gs, gs), index_map.get(ge, ge)
            lo, hi = min(ns, ne), max(ns, ne)
            remapped.append((lo, hi))
        self._fraction_groups = sorted(set(remapped))
        return True

    def _drop_row(
        self, source_index: int, container_span: tuple[int, int], before_index: int
    ) -> bool:
        """Reorder *source_index* to sit before *before_index* in a container.

        Cross-container drops are rejected: the dragged row's container (as a
        move unit) must be the drop zone's container. Returns ``True`` when a
        move was applied. Implemented as repeated adjacent swaps toward the
        target, recomputing the moving block's head each step so arithmetic
        stays in the well-tested :meth:`_swap_terms_no_render` path. Each
        intermediate swap uses the no-render/no-emit variants, so a
        multi-position drop renders and emits ``structure_changed`` exactly
        once, at the end, rather than once per swap.
        """
        source_container = self._parent_container_for_move(source_index)
        if source_container != container_span:
            return False
        terms = self._sibling_term_ranges(container_span)
        src_pos = next((i for i, (s, e) in enumerate(terms) if s <= source_index <= e), None)
        if src_pos is None:
            return False
        # Destination position among the sibling terms (before_index is a
        # component index that begins a term, or one past the container end).
        dst_pos = len(terms)
        for i, (s, _e) in enumerate(terms):
            if before_index <= s:
                dst_pos = i
                break
        # Dropping just before or just after the source is a no-op.
        if dst_pos in (src_pos, src_pos + 1):
            return False

        # Track the moving block by its position (index) among the container's
        # sibling terms, stepping one sibling per swap. Position is stable under
        # a single adjacent swap: after moving the block at ``cur_pos`` by
        # ``direction`` it occupies ``cur_pos + direction``. This works for
        # whole sub-container blocks too (the head is re-derived each step).
        target_index = dst_pos if dst_pos < src_pos else dst_pos - 1
        cur_pos = src_pos
        guard = 0
        moved = False
        while cur_pos != target_index and guard < len(self._component_names) + 2:
            guard += 1
            current_terms = self._sibling_term_ranges(container_span)
            if not (0 <= cur_pos < len(current_terms)):
                break
            direction = 1 if target_index > cur_pos else -1
            if self._move_row_no_render(current_terms[cur_pos][0], direction):
                moved = True
            cur_pos += direction

        if moved:
            self._selected_indices.clear()
            self._anchor_index = None
            self._render()
            self.structure_changed.emit()
        return True

    # ------------------------------------------------------------- grouping
    def can_group(self, span: tuple[int, int]) -> bool:
        """Return True if *span* is a groupable contiguous sibling run."""
        start, end = span
        if start >= end:
            return False
        container = self._container_of(start)
        if self._container_of(end) != container:
            return False
        # Every join inside the span must be '+'.
        if any(op != "+" for op in self._operators[start:end]):
            return False
        # The span must align with whole sibling terms of the container.
        terms = self._sibling_term_ranges(container)
        span_starts = {s for s, _e in terms}
        span_ends = {e for _s, e in terms}
        return start in span_starts and end in span_ends

    def group_span(self, span: tuple[int, int]) -> bool:
        """Wrap *span* in parentheses. Returns True on success."""
        if not self.can_group(span):
            return False
        start, end = span
        self._open_parentheses[start] += 1
        self._close_parentheses[end] += 1
        self._render()
        self.structure_changed.emit()
        return True

    def set_fraction(self, span: tuple[int, int], enabled: bool) -> bool:
        """Enable/disable the fraction flag on an existing paren span."""
        if not self._enable_fraction_groups:
            return False
        if span not in self._parenthesized_spans():
            # Group it first if it is a bare eligible run.
            if enabled and self.can_group(span):
                self.group_span(span)
            else:
                return False
        start, end = span
        if any(op != "+" for op in self._operators[start:end]) or (end - start) < 1:
            return False
        current = set(self._fraction_groups)
        if enabled:
            current.add(span)
        else:
            current.discard(span)
        self._fraction_groups = sorted(current)
        self._render()
        self.structure_changed.emit()
        return True

    def ungroup_span(self, span: tuple[int, int]) -> bool:
        """Remove one paren pair (and any fraction flag) for *span*."""
        start, end = span
        if span not in self._parenthesized_spans():
            return False
        if self._open_parentheses[start] > 0:
            self._open_parentheses[start] -= 1
        if self._close_parentheses[end] > 0:
            self._close_parentheses[end] -= 1
        self._fraction_groups = [g for g in self._fraction_groups if g != span]
        self._render()
        self.structure_changed.emit()
        return True

    # ------------------------------------------------------------ selection
    def _handle_row_click(self, component_index: int, modifiers) -> None:
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if shift and self._anchor_index is not None:
            lo, hi = sorted((self._anchor_index, component_index))
            self._selected_indices = set(range(lo, hi + 1))
        elif ctrl:
            if component_index in self._selected_indices:
                self._selected_indices.discard(component_index)
            else:
                self._selected_indices.add(component_index)
            self._anchor_index = component_index
        else:
            self._selected_indices = {component_index}
            self._anchor_index = component_index
        self._refresh_row_styles()
        self.selection_changed.emit()

    def _refresh_row_styles(self) -> None:
        for row in self._row_widgets:
            row._refresh_style()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        modifiers = event.modifiers()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected_indices:
            self.delete_selected()
            return
        if modifiers & Qt.KeyboardModifier.AltModifier and len(self._selected_indices) == 1:
            idx = next(iter(self._selected_indices))
            if key == Qt.Key.Key_Up:
                self.move_row(idx, -1)
                return
            if key == Qt.Key.Key_Down:
                self.move_row(idx, 1)
                return
        super().keyPressEvent(event)

    # -------------------------------------------------------------- rendering
    def _render(self) -> None:
        self._row_widgets: list[_RowWidget] = []
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                # _empty_label is a persistent, reused widget (re-added on the
                # next empty render), so it must survive teardown; every other
                # torn-down child (rows, drop targets, group frames) is
                # rebuilt fresh each render and must be scheduled for deletion
                # here — setParent(None) alone detaches it but does not delete
                # it, and Qt only dispatches a detached widget's queued
                # deleteLater via an explicit DeferredDelete event, so without
                # this call these accumulate for the life of the app (the
                # repo's established leaked-widget pattern; see CLAUDE.md).
                if widget is not self._empty_label:
                    widget.deleteLater()

        if not self._component_names:
            self._body_layout.addWidget(self._empty_label)
            self._empty_label.show()
            return
        self._empty_label.hide()

        fraction_set = set(self._fraction_groups)
        forest = _build_forest(self._open_parentheses, self._close_parentheses, fraction_set)
        self._fraction_group_order = {g: i for i, g in enumerate(self._fraction_groups)}

        top_span = (0, len(self._component_names) - 1)
        self._render_children_with_drops(self._body_layout, forest, top_span)
        self._body_layout.addStretch(1)

    def _render_children_with_drops(
        self, layout, children: list[_Node], container_span: tuple[int, int]
    ) -> None:
        """Add *children* to *layout* with drop zones between siblings."""
        for i, child in enumerate(children):
            layout.addWidget(_DropTarget(self, container_span, child.start))
            widget = self._render_node(child, first_in_container=(i == 0))
            layout.addWidget(widget)
        # A trailing drop zone lets a row drop to the very end of the container.
        if children:
            layout.addWidget(_DropTarget(self, container_span, container_span[1] + 1))

    def _render_node(self, node: _Node, *, first_in_container: bool) -> QWidget:
        if node.kind == "leaf":
            assert node.component_index is not None
            row = _RowWidget(self, node.component_index, first_in_container=first_in_container)
            self._row_widgets.append(row)
            return row
        return self._render_container(node, first_in_container=first_in_container)

    def _render_container(self, node: _Node, *, first_in_container: bool) -> QWidget:
        span = (node.start, node.end)
        frame = QFrame()
        frame.setObjectName("groupFrame")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(8, 6, 8, 8)
        vbox.setSpacing(4)

        header = QHBoxLayout()
        if node.is_fraction:
            color = _FRACTION_GROUP_COLORS[
                self._fraction_group_order.get(span, 0) % len(_FRACTION_GROUP_COLORS)
            ]
            amplitude_n = node.start + 1
            title = QLabel(f"Fraction group · amplitude A_{amplitude_n}")
            title.setStyleSheet(f"color: {color}; font-weight: 600;")
            frame.setStyleSheet(
                f"#groupFrame {{ border: 2px solid {color}; border-radius: 6px; "
                f"background: {tokens.SURFACE_ALT}; }}"
            )
        else:
            color = tokens.BORDER_STRONG
            title = QLabel("Group")
            title.setStyleSheet(f"color: {tokens.TEXT}; font-weight: 600;")
            frame.setStyleSheet(
                f"#groupFrame {{ border: 1px solid {tokens.BORDER_STRONG}; "
                f"border-radius: 6px; background: {tokens.SURFACE_ALT}; }}"
            )
        header.addWidget(title)
        header.addStretch(1)

        if node.is_fraction:
            # A fraction group can be converted back to per-component absolute
            # amplitudes: this keeps the parentheses but drops the {frac} flag.
            if self._enable_fraction_groups:
                toggle = QPushButton("Use absolute amplitudes")
                toggle.clicked.connect(lambda _=False, s=span: self.set_fraction(s, False))
                header.addWidget(toggle)
        elif self._enable_fraction_groups:
            # Non-fraction plain groups can be promoted to fractional amplitudes.
            terms = self._sibling_term_ranges(span)
            eligible = len(terms) >= 2 and all(
                op == "+" for op in self._operators[node.start : node.end]
            )
            if eligible:
                toggle = QPushButton("Use fractional amplitudes")
                toggle.clicked.connect(lambda _=False, s=span: self.set_fraction(s, True))
                header.addWidget(toggle)

        ungroup = QPushButton("Ungroup")
        ungroup.clicked.connect(lambda _=False, s=span: self.ungroup_span(s))
        header.addWidget(ungroup)
        vbox.addLayout(header)

        self._render_children_with_drops(vbox, node.children, span)

        return frame
