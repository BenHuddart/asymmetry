"""Structure/index-arithmetic tests for the structured model-row editor."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel, parse_composite_expression
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


def test_empty_state(qapp: QApplication) -> None:
    widget = _rows()
    widget.clear()
    assert widget.structure() == ([], [], [], [], [])
    assert widget.expression() == ""
