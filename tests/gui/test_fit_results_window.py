"""Standalone unit tests for the per-parameter fit-results window."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent  # type: ignore
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget  # type: ignore

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import VERDICT_CHIP_OBJECT_NAME
from asymmetry.gui.windows.fit_results_window import (
    FitParameterRow,
    FitRangeResults,
    FitResults,
    FitResultsWindow,
)

_GOOD = (tokens.SUCCESS_BG, tokens.SUCCESS_BORDER, tokens.OK)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _range(
    *,
    model: str = "SC_TwoGap_SS",
    chi_squared: str = "χ²ᵣ 0.89",
    bounds: str = "2 – 40",
    parameters: tuple[FitParameterRow, ...] | None = None,
) -> FitRangeResults:
    rows = parameters or (
        FitParameterRow(name="Tc", symbol="T꜀", unit="K", value=35.8, error=0.5, fixed=False),
        FitParameterRow(
            name="sigma_0", symbol="σ₀", unit="µs⁻¹", value=1.24, error=0.03, fixed=False
        ),
    )
    return FitRangeResults(
        model=model,
        chi_squared=chi_squared,
        verdict="good fit (band 0.7–1.3 at 95 %)",
        colours=_GOOD,
        bounds=bounds,
        error_mode="column",
        parameters=rows,
    )


def _results(*ranges: FitRangeResults) -> FitResults:
    return FitResults(
        parameter_name="sigma",
        x_label="Temperature (K)",
        runs="12 / 12 runs · 1 range(s)",
        ranges=ranges or (_range(),),
    )


def test_window_titles_itself_after_the_parameter(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    assert window.windowTitle() == "Fit results — σ (µs⁻¹)"
    window.deleteLater()


def test_window_tables_one_row_per_parameter(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    table = window.findChild(QTableWidget)
    assert table.rowCount() == 2
    assert [table.horizontalHeaderItem(c).text() for c in range(3)] == [
        "Parameter",
        "Value",
        "Unit",
    ]
    assert [table.item(0, column).text() for column in range(3)] == ["T꜀", "35.8(5)", "K"]
    assert [table.item(1, column).text() for column in range(3)] == [
        "σ₀",
        "1.24(3)",
        "µs⁻¹",
    ]
    window.deleteLater()


def test_window_separates_and_greys_the_fixed_parameters(qapp: QApplication) -> None:
    rows = (
        FitParameterRow(name="Tc", symbol="T꜀", unit="K", value=35.8, error=0.5, fixed=False),
        FitParameterRow(name="weight", symbol="weight", unit="", value=0.5, error=None, fixed=True),
    )
    window = FitResultsWindow(_results(_range(parameters=rows)))

    table = window.findChild(QTableWidget)
    assert table.rowCount() == 3
    assert table.item(1, 0).text() == "held fixed"
    assert table.columnSpan(1, 0) == 3
    assert [table.item(2, column).text() for column in range(3)] == ["weight", "0.5", ""]
    assert table.item(2, 0).foreground().color().name() == tokens.TEXT_MUTED
    window.deleteLater()


def test_window_names_the_x_axis_range_and_error_mode(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    texts = [label.text() for label in window.findChildren(QLabel)]
    assert "x: Temperature (K) · fit range 2 – 40 · errors: column" in texts
    assert "12 / 12 runs · 1 range(s)" in texts
    window.deleteLater()


def test_window_gives_each_range_its_own_section_and_table(qapp: QApplication) -> None:
    window = FitResultsWindow(_results(_range(), _range(chi_squared="χ²ᵣ 2.1", bounds="40 – 80")))

    assert len(window.findChildren(QTableWidget)) == 2
    texts = [label.text() for label in window.findChildren(QLabel)]
    assert "Range 1 · 2 – 40 · χ²ᵣ 0.89" in texts
    assert "Range 2 · 40 – 80 · χ²ᵣ 2.1" in texts
    window.deleteLater()


def test_window_header_chip_wears_the_verdict_colours(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    chip = next(
        label
        for label in window.findChildren(QLabel)
        if label.objectName() == VERDICT_CHIP_OBJECT_NAME
    )
    assert chip.text() == "χ²ᵣ 0.89"
    assert tokens.SUCCESS_BG in chip.styleSheet()
    assert chip.toolTip() == "χ²ᵣ 0.89 · good fit (band 0.7–1.3 at 95 %)"
    window.deleteLater()


def test_window_copy_writes_a_tab_separated_table(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    window._copy()

    assert qapp.clipboard().text().splitlines() == [
        "Model: SC_TwoGap_SS",
        "χ²ᵣ 0.89",
        "Tc\t35.80 ± 0.50\tK",
        "sigma_0\t1.240 ± 0.030\tµs⁻¹",
    ]
    window.deleteLater()


def test_window_edit_button_carries_the_parameter_name(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())
    requested: list[str] = []
    window.edit_requested.connect(requested.append)

    edit = next(
        button for button in window.findChildren(QPushButton) if button.text() == "Edit model fit…"
    )
    edit.click()

    assert requested == ["sigma"]
    window.deleteLater()


def test_window_set_results_replaces_the_previous_render(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    window.set_results(
        FitResults(
            parameter_name="Lambda",
            x_label="Field (G)",
            runs="3 / 4 runs · 1 range(s)",
            ranges=(_range(model="Linear", chi_squared="χ²ᵣ 1.1"),),
        )
    )
    # The previous render's widgets are only scheduled for deletion.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert window.windowTitle() == "Fit results — λ (µs⁻¹)"
    assert len(window.findChildren(QTableWidget)) == 1
    texts = [label.text() for label in window.findChildren(QLabel)]
    assert "3 / 4 runs · 1 range(s)" in texts
    window.deleteLater()
