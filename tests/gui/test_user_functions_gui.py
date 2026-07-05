"""GUI behaviour around user functions: placeholder degrade in the fit tabs.

The named-placeholder contract (W1): a saved model referencing a user
component that is not registered must open with its original expression,
never be silently replaced, and refuse to fit with a message naming the
missing components.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _state_with_missing_component() -> dict:
    return {
        "composite_model": {
            "component_names": ["UserGoneDecay", "Constant"],
            "operators": ["+"],
            "open_parentheses": [0, 0],
            "close_parentheses": [0, 0],
            "fraction_groups": [],
        },
        "parameters": [],
    }


def test_single_fit_restore_preserves_model_with_missing_user_component(qapp):
    tab = SingleFitTab()
    tab.restore_state(_state_with_missing_component())

    model = tab._composite_model
    assert model.component_names == ["UserGoneDecay", "Constant"]
    assert model.missing_component_names == ("UserGoneDecay",)
    assert "UserGoneDecay" in tab._result_label.text()
    # Re-saving emits the original names — nothing silently dropped.
    assert tab.get_state()["composite_model"]["component_names"] == [
        "UserGoneDecay",
        "Constant",
    ]


def test_single_fit_blocks_fitting_with_missing_user_component(qapp):
    tab = SingleFitTab()
    tab.restore_state(_state_with_missing_component())
    t = np.linspace(0.0, 4.0, 50)
    tab._current_dataset = MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.4 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 1},
    )

    tab._run_fit()
    assert tab.wait_for_fit()

    text = tab._result_label.text()
    assert "missing user function" in text
    assert "UserGoneDecay" in text


def test_single_fit_restore_still_defaults_on_malformed_model(qapp):
    tab = SingleFitTab()
    tab.restore_state({"composite_model": {"component_names": "not-a-list"}, "parameters": []})
    assert tab._composite_model.component_names == ["Exponential", "Constant"]


def test_global_fit_restore_and_fit_block_with_missing_user_component(qapp):
    tab = GlobalFitTab()
    tab.restore_state(_state_with_missing_component())

    model = tab._composite_model
    assert model.component_names == ["UserGoneDecay", "Constant"]
    assert model.missing_component_names == ("UserGoneDecay",)

    tab._run_global_fit()
    text = tab._result_text.toPlainText()
    assert "missing user function" in text
    assert "UserGoneDecay" in text


# ── provenance badges and the load-report dialog ───────────────────────────


@pytest.fixture
def _registry_snapshot(registry_snapshot):
    """Alias of the shared conftest ``registry_snapshot`` fixture."""
    yield


def _register_badged_component():
    from asymmetry.core.fitting.user_functions import register_component

    return register_component(
        "UserBadgeDecay",
        lambda t, A: A * np.exp(-np.asarray(t, dtype=float)),
        ["A"],
        domain="time",
        description="Badge test component",
        formula_template="{A}*exp(-t)",
        applicability="Badge-test applicability text.",
    )


def test_builder_picker_badges_user_components(qapp, _registry_snapshot):
    from asymmetry.core.fitting.component_search import search_components
    from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

    definition = _register_badged_component()
    dialog = FitFunctionBuilderDialog(domain="time")

    # The rebuilt dialog searches a component library; the user function is in
    # the pool and its definition carries the ``user`` provenance flag that the
    # library renders as a "· user" badge.
    library = dialog._library
    assert library._definitions["UserBadgeDecay"] is definition
    assert getattr(definition, "user", False) is True
    # Selecting it in the library and activating inserts the bare name.
    library.set_search_text("UserBadgeDecay")
    assert library.current_component_name() == "UserBadgeDecay"
    inserted: list[str] = []
    library.component_activated.connect(inserted.append)
    library._activate_current()
    assert inserted == ["UserBadgeDecay"]

    # The user component lands in its own "User" category of the library — the
    # grouping the panel renders from ``search_components("")``.
    names_by_category: dict[str, list[str]] = {}
    for result in search_components("", components=library._definitions):
        component = library._definitions[result.name]
        names_by_category.setdefault(component.category, []).append(result.name)
    assert "UserBadgeDecay" in names_by_category["User"]
    dialog.deleteLater()


def test_component_info_html_carries_user_provenance(qapp, _registry_snapshot):
    from asymmetry.core.fitting.composite import COMPONENTS
    from asymmetry.gui.widgets.component_info_dialog import build_component_info_html

    definition = _register_badged_component()
    html_text = build_component_info_html(definition, render_latex_images=False)
    assert "User function" in html_text
    assert "Badge-test applicability text." in html_text

    builtin_html = build_component_info_html(COMPONENTS["Keren"], render_latex_images=False)
    assert "User function" not in builtin_html


def test_user_functions_dialog_reports_sources(qapp, _registry_snapshot, tmp_path):
    from asymmetry.core.plugins import load_user_functions
    from asymmetry.gui.windows.user_functions_dialog import UserFunctionsDialog, _report_html

    (tmp_path / "good.py").write_text(
        """
import numpy as np
from asymmetry.core.fitting.user_functions import register_component

register_component(
    "UserDialogDecay",
    lambda t, A: A * np.exp(-np.asarray(t, dtype=float)),
    ["A"],
    domain="time",
    description="Dialog test component",
    formula_template="{A}*exp(-t)",
)
""",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text("raise RuntimeError('kaboom')\n", encoding="utf-8")

    report = load_user_functions(tmp_path)
    html_text = _report_html(report)
    assert "UserDialogDecay" in html_text
    assert "broken.py" in html_text
    assert "kaboom" in html_text

    dialog = UserFunctionsDialog()
    dialog.deleteLater()


def test_user_functions_dialog_handles_no_report(qapp):
    from asymmetry.gui.windows.user_functions_dialog import _report_html

    assert "No user functions" in _report_html(None)


def test_user_functions_dialog_open_folder_button(qapp, tmp_path, monkeypatch):
    """The Open folder… button creates the user-functions dir and opens it.

    ``QDesktopServices.openUrl`` is monkeypatched so no real file browser
    window opens; the directory is monkeypatched to ``tmp_path`` and must be
    created on demand so the button works on a fresh install.
    """
    from pathlib import Path

    from PySide6.QtGui import QDesktopServices

    from asymmetry.core import plugins
    from asymmetry.gui.windows.user_functions_dialog import UserFunctionsDialog

    target = tmp_path / "user_functions"
    monkeypatch.setattr(plugins, "USER_FUNCTIONS_DIR", target)
    opened: list = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)

    dialog = UserFunctionsDialog()
    assert dialog._open_folder_button is not None
    assert not target.exists()
    dialog._open_folder_button.click()

    assert target.is_dir()  # created so the file browser has somewhere to land
    assert len(opened) == 1
    assert Path(opened[0].toLocalFile()) == target
    dialog.deleteLater()


def test_user_functions_dialog_lists_gui_authored_function(qapp, _registry_snapshot, tmp_path):
    """A function created via ``create_user_function`` (the New User Function
    dialog's core call) must show up in the Setup -> User Functions... report,
    exactly like a hand-written plugin file discovered at startup."""
    from PySide6.QtWidgets import QTextBrowser

    from asymmetry.core.fitting.user_function_authoring import (
        DraftParameter,
        UserFunctionDraft,
        create_user_function,
    )
    from asymmetry.core.plugins import last_load_report
    from asymmetry.gui.windows.user_functions_dialog import UserFunctionsDialog, _report_html

    draft = UserFunctionDraft(
        kind="component",
        name="UserAuthoredDecay",
        description="GUI-authored test component",
        formula="A*exp(-lam*x)",
        parameters=[DraftParameter("A", 0.2), DraftParameter("lam", 0.5)],
        domain="time",
    )
    created = create_user_function(draft, directory=tmp_path)
    assert created.path.exists()
    assert created.definition.name == "UserAuthoredDecay"

    report = last_load_report()
    assert report is not None
    html_text = _report_html(report)
    assert "UserAuthoredDecay" in html_text
    assert created.path.name in html_text

    dialog = UserFunctionsDialog()
    browser = dialog.findChildren(QTextBrowser)[0]
    assert "UserAuthoredDecay" in browser.toHtml()
    dialog.deleteLater()


def test_multi_group_simulate_seed_with_missing_component_falls_back(qapp):
    """A persisted simulate seed referencing an uninstalled user component
    must not crash the dialog; it falls back to the default model (simulating
    a zero-valued placeholder would silently generate wrong data)."""
    from asymmetry.gui.windows.simulate_dialog import MultiGroupSimulateDialog
    from tests.gui.test_simulate_dialog import _ring_template_run

    seed = {
        "model": {
            "component_names": ["UserGoneDecay"],
            "operators": [],
            "open_parentheses": [0],
            "close_parentheses": [0],
            "fraction_groups": [],
        },
        "base_parameters": {},
        "specs": [],
    }
    dialog = MultiGroupSimulateDialog(_ring_template_run(4), seed=seed)
    assert dialog._model.component_names == ["Oscillatory"]
    dialog.deleteLater()
