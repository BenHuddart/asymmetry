"""Tests for the searchable component-library panel."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import COMPONENTS, ComponentDefinition
from asymmetry.gui.widgets.function_builder.library_panel import (
    _CATEGORY_ITEM_TYPE,
    _COMPONENT_ITEM_TYPE,
    _NAME_ROLE,
    ComponentLibraryPanel,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _category_labels(panel: ComponentLibraryPanel) -> list[str]:
    root = panel._tree.invisibleRootItem()
    labels = []
    for index in range(root.childCount()):
        item = root.child(index)
        if item.type() == _CATEGORY_ITEM_TYPE:
            labels.append(item.text(0))
    return labels


def _component_names_under(item) -> list[str]:
    names = []
    for index in range(item.childCount()):
        child = item.child(index)
        if child.type() == _COMPONENT_ITEM_TYPE:
            names.append(child.data(0, _NAME_ROLE))
    return names


def _all_component_names_in_order(panel: ComponentLibraryPanel) -> list[str]:
    return [item.data(0, _NAME_ROLE) for item in panel._component_items_in_display_order()]


def _row_widget(panel: ComponentLibraryPanel, item) -> object:
    """Return the per-item row widget (label + add/info buttons)."""
    return panel._tree.itemWidget(item, 0)


def _row_label_text(panel: ComponentLibraryPanel, item) -> str:
    """Return the rendered rich-text of a row's label (the sole painter)."""
    row = _row_widget(panel, item)
    assert row is not None
    return row.label.text()


def test_construction_with_components(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    assert panel.search_text() == ""
    assert panel.current_component_name() is not None


def test_empty_query_groups_by_canonical_category_order(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    labels = _category_labels(panel)
    canonical_order = [
        "General",
        "Relaxation",
        "Oscillation",
        "Kubo-Toyabe",
        "Muonium",
        "Nuclear dipolar",
        "Background",
        "Frequency Domain",
    ]
    assert labels == [label for label in canonical_order if label in labels]
    # Canonical order: Relaxation before Oscillation before Kubo-Toyabe, etc.
    assert labels.index("Relaxation") < labels.index("Oscillation")
    assert labels.index("Oscillation") < labels.index("Kubo-Toyabe")
    assert labels.index("Kubo-Toyabe") < labels.index("Muonium")

    # Components appear beneath their category.
    root = panel._tree.invisibleRootItem()
    relaxation_item = next(
        root.child(i) for i in range(root.childCount()) if root.child(i).text(0) == "Relaxation"
    )
    relaxation_names = _component_names_under(relaxation_item)
    assert "Exponential" in relaxation_names
    assert "Gaussian" in relaxation_names


def test_typing_query_flattens_and_ranks(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Constant")
    QApplication.processEvents()

    assert _category_labels(panel) == []
    names = _all_component_names_in_order(panel)
    assert names[0] == "Constant"
    assert "ConstantBackground" in names


def test_alias_query_surfaces_kubo_toyabe_family(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("kt")
    QApplication.processEvents()

    names = set(_all_component_names_in_order(panel))
    assert "StaticGKT_ZF" in names
    assert "LongitudinalFieldKT" in names


def test_matched_field_annotation_appears_for_alias_hit(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("kubo")
    QApplication.processEvents()

    item = next(
        item
        for item in panel._component_items_in_display_order()
        if item.data(0, _NAME_ROLE) == "StaticGKT_ZF"
    )
    assert "alias" in _row_label_text(panel, item)


def test_activation_signal_on_double_click(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Constant")
    QApplication.processEvents()

    activated: list[str] = []
    panel.component_activated.connect(activated.append)

    item = panel._component_items_in_display_order()[0]
    panel._on_item_activated(item, 0)

    assert activated == ["Constant"]


def test_activation_signal_on_enter_in_search_box(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Constant")
    QApplication.processEvents()

    activated: list[str] = []
    panel.component_activated.connect(activated.append)

    panel._search_edit.setFocus()
    QTest.keyClick(panel._search_edit, Qt.Key.Key_Return)

    assert activated == ["Constant"]


def test_activation_signal_on_enter_in_tree(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Constant")
    QApplication.processEvents()

    activated: list[str] = []
    panel.component_activated.connect(activated.append)

    panel._tree.setFocus()
    QTest.keyClick(panel._tree, Qt.Key.Key_Return)

    assert activated == ["Constant"]


def test_down_arrow_from_search_box_moves_selection(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.show()
    QApplication.processEvents()
    panel.set_search_text("Constant")
    QApplication.processEvents()

    names = _all_component_names_in_order(panel)
    assert len(names) >= 2

    first = panel.current_component_name()
    assert first == names[0]

    panel._search_edit.setFocus()
    QApplication.processEvents()
    QTest.keyClick(panel._search_edit, Qt.Key.Key_Down)

    assert panel.current_component_name() == names[1]
    # Focus stays in the search box (event filter consumed the key).
    assert panel._search_edit.hasFocus()
    panel.close()


def test_set_components_with_restricted_pool_filters(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    restricted = {"Constant": COMPONENTS["Constant"], "Gaussian": COMPONENTS["Gaussian"]}
    panel.set_components(restricted)
    QApplication.processEvents()

    names = set(_all_component_names_in_order(panel))
    assert names == {"Constant", "Gaussian"}


def test_set_components_preserves_current_query(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Gaussian")
    QApplication.processEvents()

    panel.set_components(dict(COMPONENTS))
    QApplication.processEvents()

    assert panel.search_text() == "Gaussian"
    assert _category_labels(panel) == []


def test_user_badge_shown_for_user_definition(qapp: QApplication) -> None:
    user_component = ComponentDefinition(
        name="MyUserFunc",
        description="A user-registered test function",
        function=COMPONENTS["Constant"].function,
        param_names=["A"],
        param_defaults={"A": 1.0},
        param_info={},
        formula_template="{A}",
        category="General",
        user=True,
    )
    pool = {"MyUserFunc": user_component}
    panel = ComponentLibraryPanel(pool)
    QApplication.processEvents()

    item = panel._component_items_in_display_order()[0]
    assert "user" in _row_label_text(panel, item)


def test_missing_definition_gets_distinct_foreground(qapp: QApplication) -> None:
    normal_component = ComponentDefinition(
        name="NormalFunc",
        description="normal",
        function=COMPONENTS["Constant"].function,
        param_names=["A"],
        param_defaults={"A": 1.0},
        param_info={},
        formula_template="{A}",
        category="General",
    )
    missing_component = ComponentDefinition(
        name="MissingFunc",
        description="a placeholder for an unregistered user function",
        function=COMPONENTS["Constant"].function,
        param_names=["A"],
        param_defaults={"A": 1.0},
        param_info={},
        formula_template="{A}",
        category="General",
        missing=True,
    )
    pool = {"NormalFunc": normal_component, "MissingFunc": missing_component}
    panel = ComponentLibraryPanel(pool)
    QApplication.processEvents()

    items = {item.data(0, _NAME_ROLE): item for item in panel._component_items_in_display_order()}
    normal_text = _row_label_text(panel, items["NormalFunc"])
    missing_text = _row_label_text(panel, items["MissingFunc"])
    assert normal_text != missing_text
    assert "missing" in missing_text


def test_no_matches_shows_empty_state(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.show()
    QApplication.processEvents()
    panel.set_search_text("zzzznonexistentqueryxyz")
    QApplication.processEvents()

    assert panel._empty_label.isVisible()
    assert "No matches" in panel._empty_label.text()
    assert panel.current_component_name() is None
    panel.close()


def test_escape_clears_search(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.set_search_text("Gaussian")
    QApplication.processEvents()

    panel._search_edit.setFocus()
    QTest.keyClick(panel._search_edit, Qt.Key.Key_Escape)

    assert panel.search_text() == ""


def test_component_row_has_no_second_painted_text(qapp: QApplication) -> None:
    """Regression: column 0 must not paint both item.text() and the item widget.

    Previously every row called ``item.setText(0, name)`` *and*
    ``setItemWidget(item, 0, label)`` — Qt painted both on top of each
    other, producing smeared/bold-looking rows. The rich-text row widget
    must be the sole text-painting mechanism; the item's own text stays
    empty.
    """
    panel = ComponentLibraryPanel(COMPONENTS)
    QApplication.processEvents()

    for item in panel._component_items_in_display_order():
        assert item.text(0).strip() == ""


def test_component_row_has_add_and_info_buttons(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.show()
    QApplication.processEvents()

    item = panel._component_items_in_display_order()[0]
    row = _row_widget(panel, item)
    assert row is not None
    assert row.add_button.isVisible()
    assert row.info_button.isVisible()
    panel.close()


def test_add_button_click_emits_component_activated_and_selects_row(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.show()
    QApplication.processEvents()

    items = panel._component_items_in_display_order()
    # Pick a row that isn't already current, so the assertion is meaningful.
    target = next(
        item for item in items if item.data(0, _NAME_ROLE) != panel.current_component_name()
    )
    target_name = target.data(0, _NAME_ROLE)
    row = _row_widget(panel, target)
    assert row is not None

    activated: list[str] = []
    panel.component_activated.connect(activated.append)

    QTest.mouseClick(row.add_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert activated == [target_name]
    assert panel.current_component_name() == target_name
    panel.close()


def test_info_button_click_selects_row(qapp: QApplication) -> None:
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.show()
    QApplication.processEvents()

    items = panel._component_items_in_display_order()
    target = next(
        item for item in items if item.data(0, _NAME_ROLE) != panel.current_component_name()
    )
    target_name = target.data(0, _NAME_ROLE)
    row = _row_widget(panel, target)
    assert row is not None

    # The info dialog is modal; patch it out so the click doesn't block the
    # test on a real dialog exec loop while still exercising the row
    # selection side effect the button is responsible for.
    from asymmetry.gui.widgets.function_builder import library_panel as module

    called: list[object] = []
    original = module.show_component_info_dialog
    module.show_component_info_dialog = lambda parent, definition: called.append(definition)
    try:
        QTest.mouseClick(row.info_button, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
    finally:
        module.show_component_info_dialog = original

    assert panel.current_component_name() == target_name
    assert len(called) == 1
    panel.close()


def test_item_size_hint_width_covers_name_font_metrics() -> None:
    """No-clip proxy: the item's natural size hint must fit its full name.

    The row widget elides on-screen when the column is too narrow (with the
    full name in the tooltip), but the *size hint* itself — the row's
    natural/unclipped width — must be at least as wide as the plain name
    text, so a wide-enough panel never clips or elides unnecessarily.
    """
    app = QApplication.instance() or QApplication([])
    panel = ComponentLibraryPanel(COMPONENTS)
    QApplication.processEvents()

    fm = QFontMetrics(panel._tree.font())
    for item in panel._component_items_in_display_order():
        name = item.data(0, _NAME_ROLE)
        hint = item.sizeHint(0)
        assert hint.width() >= fm.horizontalAdvance(name), name
    del app


def test_long_name_elides_rather_than_clips_at_narrow_width(qapp: QApplication) -> None:
    """A genuinely-too-long name shrinks to fit with an ellipsis, not a hard clip."""
    panel = ComponentLibraryPanel(COMPONENTS)
    panel.resize(220, 500)
    panel.show()
    QApplication.processEvents()
    QApplication.processEvents()

    long_name = "StretchedExponential"
    assert long_name in COMPONENTS
    item = next(
        item
        for item in panel._component_items_in_display_order()
        if item.data(0, _NAME_ROLE) == long_name
    )
    row = _row_widget(panel, item)
    assert row is not None

    fm = QFontMetrics(row.label.font())
    assert fm.horizontalAdvance(long_name) > row.label.width(), (
        "test assumes the name is wider than the label at this panel width"
    )

    # Rendered text must not equal the full, un-elided name...
    assert long_name not in row.label.text()
    # ...but the full name is still reachable via the tooltip.
    assert long_name in row.label.toolTip() or long_name in row.toolTip()
    panel.close()
