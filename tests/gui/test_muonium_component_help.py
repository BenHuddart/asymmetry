"""The displayed intratriplet rate agrees with the component's units and scope."""

import pytest
from PySide6.QtWidgets import QTextBrowser, QWidget

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

pytestmark = pytest.mark.gui


def test_lf_component_help_renders_envelope_and_angular_rate(qapp):
    parent = QWidget()
    definition = COMPONENTS["MuoniumLFRelax"]
    dialog = show_component_info_dialog(parent, definition)
    qapp.processEvents()
    browser = dialog.findChild(QTextBrowser)
    text = browser.toPlainText()
    assert definition.param_info["delta_ex"].unit == "µs⁻¹"
    assert "µs⁻¹" in text
    assert "angular-rate amplitude" in text
    assert "separate repolarization amplitude" in text
    assert "not a general electron spin-exchange model" in text
    assert "data:image/png" in browser.toHtml()
    dialog.close()
    parent.close()
