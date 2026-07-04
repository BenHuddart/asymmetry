"""Tests for the searchable component-library panel."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
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
    label = panel._tree.itemWidget(item, 0)
    assert label is not None
    assert "alias" in label.text()


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
    label = panel._tree.itemWidget(item, 0)
    assert label is not None
    assert "user" in label.text()


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
    normal_label = panel._tree.itemWidget(items["NormalFunc"], 0)
    missing_label = panel._tree.itemWidget(items["MissingFunc"], 0)
    assert normal_label is not None and missing_label is not None
    assert normal_label.text() != missing_label.text()
    assert "missing" in missing_label.text()


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
