"""Rendered derived-model help preserves units and readable equation lines."""

import base64
import re

import pytest
from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import QTextBrowser

from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.styles.fonts import set_ui_font_scale, ui_font_scale
from asymmetry.gui.widgets.component_info_dialog import (
    build_component_info_html,
    show_component_info_dialog,
)

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("name,lines", [("RFResonanceMuP", 2), ("Arrhenius", 1)])
@pytest.mark.parametrize("scale", [1.0, 1.2])
def test_rendered_equation_lines_fit_component_dialog(qapp, name, lines, scale):
    component = PARAMETER_MODEL_COMPONENTS[name]
    expression = (
        build_component_info_html(component)
        .split("<h3>Model Expression</h3>")[1]
        .split("<h3>Parameters</h3>")[0]
    )
    images = re.findall(r"src='data:image/png;base64,([^']+)'", expression)
    assert len(images) == lines
    assert expression.count("<p>") == (lines if lines > 1 else 0)
    for data in images:
        assert not QImage.fromData(base64.b64decode(data), "PNG").isNull()

    original_font, original_scale = QFont(qapp.font()), ui_font_scale()
    scaled_font = QFont(original_font)
    scaled_font.setPointSizeF(original_font.pointSizeF() * scale)
    try:
        qapp.setFont(scaled_font)
        set_ui_font_scale(scale)
        dialog = show_component_info_dialog(None, component)
        dialog.resize(820, 600)
        qapp.processEvents()
        qapp.processEvents()
        browser = dialog.findChild(QTextBrowser)
        assert browser.horizontalScrollBar().maximum() == 0
        dialog.close()
    finally:
        qapp.setFont(original_font)
        set_ui_font_scale(original_scale)


def test_parameter_help_uses_local_scientific_conventions():
    rf = build_component_info_html(
        PARAMETER_MODEL_COMPONENTS["RFResonanceMuP"], render_latex_images=False
    )
    assert "No parameter description available" not in rf
    assert "Additive constant baseline" in rf
    assert "combined in quadrature" not in rf
    redfield = build_component_info_html(
        PARAMETER_MODEL_COMPONENTS["Redfield"], render_latex_images=False
    )
    assert "Cyclic frequency scale" in redfield
    assert "r=2πν" in redfield
    sc = build_component_info_html(
        PARAMETER_MODEL_COMPONENTS["SC_NonmonotonicD"], render_latex_images=False
    )
    assert "square root of the second moment" in sc
    assert "default β=0.8 is monotonic" in sc
