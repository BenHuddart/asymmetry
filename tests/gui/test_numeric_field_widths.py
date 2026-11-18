"""Scientific numeric controls retain a readable text area through UI scaling."""

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QStyle, QStyleOptionFrame

from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab
from asymmetry.gui.styles.metrics import char_width

pytestmark = pytest.mark.gui


def settle():
    loop = QEventLoop()
    QTimer.singleShot(30, loop.quit)
    loop.exec()


def text_width(field):
    option = QStyleOptionFrame()
    field.initStyleOption(option)
    area = field.style().subElementRect(QStyle.SubElement.SE_LineEditContents, option, field)
    margins = field.textMargins()
    return area.width() - margins.left() - margins.right() - 4


@pytest.mark.parametrize("initial_scale", [1.0, 1.2])
def test_frequency_and_moment_values_fit_at_launch_and_live_scale(qapp, initial_scale):
    from PySide6.QtCore import QSettings

    QSettings().setValue("ui/scale", initial_scale)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_domain_button_clicked("frequency")
    spin = window._frequency_plot_panel._frequency_reference_spin
    spin.setValue(95.94)
    moments = window._fourier_panel.moments_widget
    fields = (moments._range_min_spin, moments._range_max_spin)
    for field, value in zip(fields, (1918.270, 2213.388), strict=True):
        field.setValue(value)
    fit_limit = window._fit_panel._single_tab._fit_range_min_spin
    fit_limit.setValue(-1000)
    try:
        for scale in (initial_scale, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0):
            window._ui_manager.set_ui_scale(scale)
            settle()
            edit = spin.lineEdit()
            assert edit.text() == "95.94 G"
            assert text_width(edit) >= edit.fontMetrics().horizontalAdvance(edit.text())
            assert spin.font().pointSizeF() == pytest.approx(11 * scale)
            for field in (*fields, fit_limit):
                field.resize(field.minimumWidth(), field.sizeHint().height())
                assert text_width(field) >= field.fontMetrics().horizontalAdvance(field.text())
                assert field.font().pointSizeF() == pytest.approx(11 * scale)
            assert window.width() <= 1305
    finally:
        window.close()


@pytest.mark.parametrize("tab_class,dock_chars", [(SingleFitTab, 40), (GlobalFitTab, 46)])
@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_fit_range_wraps_with_full_precision_under_a_small_application_font(
    qapp, tab_class, dock_chars, domain
):
    original = qapp.font()
    qapp.setFont(QFont("DejaVu Sans", 9))
    tab = tab_class()
    try:
        tab.set_domain(domain)
        tab.set_fit_range_display(-1000, 1000)
        tab.resize(char_width(dock_chars), 1200)
        tab.show()
        settle()
        assert tab.minimumSizeHint().width() <= char_width(dock_chars)
        for field in (tab._fit_range_min_spin, tab._fit_range_max_spin):
            assert field.text() == f"{field.value():.{3 if domain == 'time' else 4}f}"
            assert text_width(field) >= field.fontMetrics().horizontalAdvance(field.text())
        mid, high, unit = (
            tab._fit_range_mid_label,
            tab._fit_range_max_spin,
            tab._fit_range_unit_label,
        )
        assert mid.mapTo(tab, mid.rect().center()).y() == high.mapTo(tab, high.rect().center()).y()
        assert (
            unit.mapTo(tab, unit.rect().center()).y() == high.mapTo(tab, high.rect().center()).y()
        )
    finally:
        tab.close()
        qapp.setFont(original)
