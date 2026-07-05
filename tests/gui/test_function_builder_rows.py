"""Structure/index-arithmetic tests for the structured model-row editor."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel, parse_composite_expression
from asymmetry.gui.utils.formatting import format_param_label
from asymmetry.gui.widgets.function_builder.model_rows import ModelRowList


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _rows(enable_fraction_groups: bool = True) -> ModelRowList:
    return ModelRowList(COMPONENTS, enable_fraction_groups=enable_fraction_groups)


def _seed(widget: ModelRowList, expression: str) -> None:
    names, ops, opens, closes, fracs = parse_composite_expression(expression)
    widget.set_structure(names, ops, opens, closes, fracs)


# ----------------------------------------------------------------- round-trip
@pytest.mark.parametrize(
    "expression",
    [
        "Exponential + Constant",
        "Exponential * ( Gaussian + Constant )",
        "( Exponential + Gaussian ){frac} + Constant",
        "( Exponential + Gaussian + Constant ){frac}",
        "( Exponential + Gaussian ){frac} + ( Gaussian + Constant ){frac}",
        "Exponential * ( Gaussian + Constant ) + ( Exponential + Gaussian ){frac}",
    ],
)
def test_roundtrip_structure_to_expression(qapp: QApplication, expression: str) -> None:
    widget = _rows()
    _seed(widget, expression)
    # The emitted expression re-parses to an identical structure (whitespace
    # around parens is canonicalized by build_component_expression, so compare
    # the parsed five-tuple rather than raw text).
    original = parse_composite_expression(expression)
    roundtrip = parse_composite_expression(widget.expression())
    assert roundtrip == original
    assert CompositeModel.from_expression(widget.expression()) is not None


def test_structure_returns_five_lists(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Constant")
    names, ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Constant"]
    assert ops == ["+"]
    assert opens == [0, 0]
    assert closes == [0, 0]
    assert fracs == []


# ------------------------------------------------------------------- append
def test_append_at_top_level(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Constant")
    widget.append_component("Gaussian")
    names, ops, _o, _c, _f = widget.structure()
    assert names == ["Exponential", "Constant", "Gaussian"]
    assert ops == ["+", "+"]


def test_append_after_selected_row(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Constant")
    widget._selected_indices = {0}
    widget.append_component("Gaussian")
    names, ops, _o, _c, _f = widget.structure()
    assert names == ["Exponential", "Gaussian", "Constant"]
    assert ops == ["+", "+"]


def test_append_to_empty(qapp: QApplication) -> None:
    widget = _rows()
    widget.append_component("Exponential")
    names, ops, opens, closes, _f = widget.structure()
    assert names == ["Exponential"]
    assert ops == []
    assert opens == [0]
    assert closes == [0]


# ---------------------------------------------------------------- duplicate
def test_duplicate_row_inserts_after_with_same_operator(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Constant")
    widget.duplicate_row(0)
    names, ops, _o, _c, _f = widget.structure()
    assert names == ["Exponential", "Exponential", "Constant"]
    assert ops == ["+", "+"]


def test_duplicate_inside_fraction_group_extends_it(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac}")
    names, ops, _o, _c, fracs = widget.structure()
    assert fracs == [(0, 1)]
    widget.duplicate_row(1)  # duplicate Gaussian, inside the group
    names, ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Gaussian", "Gaussian"]
    assert fracs == [(0, 2)]
    # The group's closing paren moved to the new last member.
    assert closes == [0, 0, 1]
    assert opens == [1, 0, 0]
    # Still a valid model with a widened group.
    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == [(0, 2)]


def test_duplicate_first_member_of_group_followed_by_sibling(qapp: QApplication) -> None:
    # Regression: duplicating the FIRST member of a group whose insertion point
    # falls strictly inside the group's remapped span must not double-extend
    # the group and swallow the next sibling. See _extend_group_end /
    # duplicate_row in model_rows.py.
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Oscillatory")
    names, ops, opens, closes, fracs = widget.structure()
    assert fracs == [(0, 1)]
    widget.duplicate_row(0)  # duplicate Exponential, the group's first member
    names, ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Exponential", "Gaussian", "Oscillatory"]
    # The group extends by exactly one; Oscillatory stays outside the group.
    assert fracs == [(0, 2)]
    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == [(0, 2)]


def test_duplicate_last_member_of_group_followed_by_sibling(qapp: QApplication) -> None:
    # Existing behavior preserved: duplicating the LAST member of a group
    # extends the group by one (the insertion point lands after the group's
    # remapped end, so _extend_group_end must do the extending).
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Oscillatory")
    widget.duplicate_row(1)  # duplicate Gaussian, the group's last member
    names, ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Gaussian", "Gaussian", "Oscillatory"]
    assert fracs == [(0, 2)]
    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == [(0, 2)]


def test_duplicate_member_of_group_with_no_following_sibling(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac}")
    widget.duplicate_row(0)  # duplicate Exponential; no sibling after the group
    names, ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Exponential", "Gaussian"]
    assert fracs == [(0, 2)]
    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == [(0, 2)]


def test_duplicate_shifts_following_group(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + ( Gaussian + Constant ){frac}")
    # group is (1, 2); duplicate the leading Exponential (index 0)
    widget.duplicate_row(0)
    names, _ops, _o, _c, fracs = widget.structure()
    assert names == ["Exponential", "Exponential", "Gaussian", "Constant"]
    assert fracs == [(2, 3)]


# ------------------------------------------------------------------- delete
def test_delete_middle_row(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    widget.delete_row(1)
    names, ops, _o, _c, _f = widget.structure()
    assert names == ["Exponential", "Constant"]
    assert ops == ["+"]


def test_delete_last_row_in_container_dissolves_it(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential * ( Gaussian + Constant )")
    # Delete Gaussian and Constant, dissolving the parens.
    widget.delete_row(2)  # Constant
    widget.delete_row(1)  # Gaussian
    names, _ops, opens, closes, _f = widget.structure()
    assert names == ["Exponential"]
    assert opens == [0]
    assert closes == [0]


def test_delete_from_two_term_fraction_group_dissolves_group(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Constant")
    assert widget.structure()[4] == [(0, 1)]
    widget.delete_row(1)  # delete Gaussian → group has one term left
    names, _ops, opens, closes, fracs = widget.structure()
    assert names == ["Exponential", "Constant"]
    assert fracs == []  # group dissolved
    # Parens dissolved too → clean additive expression.
    assert opens == [0, 0]
    assert closes == [0, 0]
    assert CompositeModel.from_expression(widget.expression()) is not None


def test_delete_selected(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    widget._selected_indices = {0, 2}
    widget.delete_selected()
    names, _ops, _o, _c, _f = widget.structure()
    assert names == ["Gaussian"]


# --------------------------------------------------------------------- move
def test_move_row_swaps_siblings(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    widget.move_row(0, 1)  # move Exponential down
    names, _ops, _o, _c, _f = widget.structure()
    assert names == ["Gaussian", "Exponential", "Constant"]


def test_move_whole_container_as_unit(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ) + Constant")
    # Move the container (starting at index 0) down past Constant.
    widget.move_row(0, 1)
    names, _ops, opens, closes, _f = widget.structure()
    assert names == ["Constant", "Exponential", "Gaussian"]
    # The paren span moved with the block.
    assert opens == [0, 1, 0]
    assert closes == [0, 0, 1]
    assert CompositeModel.from_expression(widget.expression()) is not None


def test_move_container_carries_fraction_group(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Constant")
    widget.move_row(0, 1)
    names, _ops, _o, _c, fracs = widget.structure()
    assert names == ["Constant", "Exponential", "Gaussian"]
    assert fracs == [(1, 2)]


# -------------------------------------------------------------- group + frac
def test_group_two_additive_siblings(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    assert widget.group_span((0, 1)) is True
    _n, _ops, opens, closes, _f = widget.structure()
    assert opens == [1, 0, 0]
    assert closes == [0, 1, 0]


def test_group_rejects_non_additive_join(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential * Gaussian + Constant")
    assert widget.can_group((0, 1)) is False
    assert widget.group_span((0, 1)) is False


def test_fraction_toggle_on_group(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    assert widget.set_fraction((0, 1), True) is True
    _n, _ops, _o, _c, fracs = widget.structure()
    assert fracs == [(0, 1)]
    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == [(0, 1)]


def test_fraction_toggle_rejected_when_disabled(qapp: QApplication) -> None:
    widget = _rows(enable_fraction_groups=False)
    _seed(widget, "Exponential + Gaussian")
    assert widget.set_fraction((0, 1), True) is False


def test_ungroup_removes_parens_and_fraction(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Constant")
    assert widget.ungroup_span((0, 1)) is True
    _n, _ops, opens, closes, fracs = widget.structure()
    assert fracs == []
    assert opens == [0, 0, 0]
    assert closes == [0, 0, 0]


def test_set_fraction_false_keeps_parens_drops_flag(qapp: QApplication) -> None:
    # "Use absolute amplitudes": disabling the fraction flag on an existing
    # group must keep the parentheses (only ungroup_span removes them) so the
    # plain grouped structure survives, per-component amplitudes return.
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Constant")
    assert widget.set_fraction((0, 1), False) is True
    names, ops, opens, closes, fracs = widget.structure()
    assert fracs == []
    assert names == ["Exponential", "Gaussian", "Constant"]
    assert opens == [1, 0, 0]
    assert closes == [0, 1, 0]
    assert "{frac}" not in widget.expression()
    assert "(Exponential + Gaussian)" in widget.expression()

    model = CompositeModel.from_expression(widget.expression())
    assert model.fraction_groups == []
    assert "A_1" in model.param_names  # per-component amplitude, not f_Exponential
    assert "A_2" in model.param_names


def test_fraction_toggle_roundtrip_fractional_absolute_fractional(qapp: QApplication) -> None:
    # Toggling fractional -> absolute -> fractional must round-trip the
    # structure: parens survive both ways and {frac} disappears/reappears.
    widget = _rows()
    _seed(widget, "( Exponential + Gaussian ){frac} + Constant")
    original_structure = widget.structure()
    assert "{frac}" in widget.expression()

    assert widget.set_fraction((0, 1), False) is True
    assert "{frac}" not in widget.expression()
    absolute_model = CompositeModel.from_expression(widget.expression())
    assert absolute_model.fraction_groups == []
    assert {"A_1", "A_2"}.issubset(set(absolute_model.param_names))

    assert widget.set_fraction((0, 1), True) is True
    assert "{frac}" in widget.expression()
    assert widget.structure() == original_structure
    fractional_model = CompositeModel.from_expression(widget.expression())
    assert fractional_model.fraction_groups == [(0, 1)]
    assert "f_Exponential" in fractional_model.param_names
    # A_2 (Gaussian's own absolute amplitude) is replaced by the derived
    # fraction weight; A_1 is the group's shared amplitude, present either way.
    assert "A_2" not in fractional_model.param_names


# ------------------------------------------------------------------ selection
def test_selected_spans_contiguous_runs(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    widget._selected_indices = {0, 1}
    assert widget.selected_spans() == [(0, 1)]
    widget._selected_indices = {0, 2}
    assert widget.selected_spans() == [(0, 0), (2, 2)]


# -------------------------------------------------------------- drag reorder
def test_drop_row_reorders_to_end(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    assert widget._drop_row(0, (0, 2), 3) is True
    assert widget.structure()[0] == ["Gaussian", "Constant", "Exponential"]


def test_drop_row_reorders_to_front(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    assert widget._drop_row(2, (0, 2), 0) is True
    assert widget.structure()[0] == ["Constant", "Exponential", "Gaussian"]


def test_drop_row_noop_returns_false(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant")
    assert widget._drop_row(1, (0, 2), 2) is False
    assert widget.structure()[0] == ["Exponential", "Gaussian", "Constant"]


def test_drop_row_rejects_cross_container(qapp: QApplication) -> None:
    widget = _rows()
    _seed(widget, "Exponential + ( Gaussian + Constant )")
    # Dropping the top-level Exponential into the inner container is rejected.
    assert widget._drop_row(0, (1, 2), 1) is False


def test_drop_row_multi_position_emits_structure_changed_once(qapp: QApplication) -> None:
    # Regression: _drop_row moved a row via repeated _swap_terms calls, each
    # doing a full re-render + structure_changed emission. A multi-position
    # drop (here: 4 sibling terms, moving index 0 to the very end, three
    # single-step swaps) must render + emit exactly once, not once per swap.
    widget = _rows()
    _seed(widget, "Exponential + Gaussian + Constant + Oscillatory")
    emit_count = 0

    def _on_structure_changed() -> None:
        nonlocal emit_count
        emit_count += 1

    widget.structure_changed.connect(_on_structure_changed)
    assert widget._drop_row(0, (0, 3), 4) is True
    assert widget.structure()[0] == ["Gaussian", "Constant", "Oscillatory", "Exponential"]
    assert emit_count == 1


def test_empty_state(qapp: QApplication) -> None:
    widget = _rows()
    widget.clear()
    assert widget.structure() == ([], [], [], [], [])
    assert widget.expression() == ""


# ---------------------------------------------------------- pretty param labels
def test_row_param_summary_uses_formatted_labels(qapp: QApplication) -> None:
    # Raw internal names ("A, Lambda") must render through the shared
    # format_param_label foundation ("A (%), λ (µs⁻¹)"), matching the fit table.
    widget = _rows()
    _seed(widget, "Exponential")
    assert len(widget._row_widgets) == 1
    row = widget._row_widgets[0]
    # The summary label is the only plain (non-rich) QLabel with muted styling;
    # locate it via its tooltip, which always holds the untruncated summary.
    expected_full = ", ".join(format_param_label(p) for p in ["A", "Lambda"])
    labels_with_tooltip = [
        lbl for lbl in row.findChildren(QLabel) if lbl.toolTip() == expected_full
    ]
    assert len(labels_with_tooltip) == 1
    label = labels_with_tooltip[0]
    # Rendered text is the formatted label (possibly elided, but short names
    # here fit comfortably within the elide budget so it round-trips exactly).
    assert label.text() == expected_full
    assert "λ" in expected_full  # lambda symbol present in the formatted label
