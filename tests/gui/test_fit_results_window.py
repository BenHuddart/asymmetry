"""Standalone unit tests for the per-parameter fit-results window."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent  # type: ignore
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget  # type: ignore

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit.tab_base import fit_results_snapshot
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    FIT_VERDICT_CHIP_COLOURS,
    NEUTRAL_CHIP_COLOURS,
    VERDICT_CHIP_OBJECT_NAME,
)
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
        title="Fit results — σ (µs⁻¹)",
        parameter_name="sigma",
        x_label="Temperature (K)",
        runs="12 / 12 runs · 1 range(s)",
        ranges=ranges or (_range(),),
    )


def test_window_titles_itself_from_the_snapshot(qapp: QApplication) -> None:
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
    window = FitResultsWindow(_results(), editable=True)
    requested: list[str] = []
    window.edit_requested.connect(requested.append)

    edit = next(
        button for button in window.findChildren(QPushButton) if button.text() == "Edit model fit…"
    )
    edit.click()

    assert requested == ["sigma"]
    window.deleteLater()


def test_window_has_no_edit_button_when_the_fit_is_not_editable(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    labels = [button.text() for button in window.findChildren(QPushButton)]
    assert labels == ["Copy", "Close"]
    window.deleteLater()


def test_window_set_results_replaces_the_previous_render(qapp: QApplication) -> None:
    window = FitResultsWindow(_results())

    window.set_results(
        FitResults(
            title="Fit results — λ (µs⁻¹)",
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


# ── fit_results_snapshot: a core FitResult frozen for this window ────────────


def _fit_result(*, chi_squared: float = 595.0, dof: int = 595) -> FitResult:
    """A fabricated run fit: two free parameters and one held fixed."""
    parameters = ParameterSet(
        [
            Parameter("A0", value=0.2512),
            Parameter("baseline", value=0.01, fixed=True),
            Parameter("sigma", value=1.24),
        ]
    )
    return FitResult(
        success=True,
        chi_squared=chi_squared,
        reduced_chi_squared=chi_squared / dof if dof else 0.0,
        dof=dof,
        parameters=parameters,
        uncertainties={"A0": 0.0043, "sigma": 0.03},
    )


def _snapshot(result: FitResult) -> FitResults:
    return fit_results_snapshot(
        result,
        title="Fit results — 3001",
        model="A0*exp(-lambda*t)",
        fit_range="0.10–10.00 µs",
        runs="Run 3001",
    )


def test_snapshot_lists_the_free_parameters_before_the_fixed_ones() -> None:
    solved = _snapshot(_fit_result()).ranges[0]

    assert [row.name for row in solved.parameters] == ["A0", "sigma", "baseline"]
    assert solved.parameters[0] == FitParameterRow(
        name="A0", symbol="A₀", unit="%", value=0.2512, error=0.0043, fixed=False
    )
    assert solved.parameters[2] == FitParameterRow(
        name="baseline", symbol="baseline", unit="%", value=0.01, error=None, fixed=True
    )


def test_snapshot_carries_the_tabs_strings_and_the_chi_squared_chip() -> None:
    snapshot = _snapshot(_fit_result())

    assert snapshot.title == "Fit results — 3001"
    assert snapshot.runs == "Run 3001"
    solved = snapshot.ranges[0]
    assert solved.model == "A0*exp(-lambda*t)"
    assert solved.bounds == "0.10–10.00 µs"
    assert solved.chi_squared == "χ²ᵣ 1"
    # The engine's FitResult records no error mode.
    assert solved.error_mode == ""


def test_snapshot_takes_its_verdict_and_colours_from_the_fit_summary() -> None:
    solved = _snapshot(_fit_result()).ranges[0]

    assert solved.verdict.startswith("good fit (band ")
    assert solved.colours == FIT_VERDICT_CHIP_COLOURS["good"]


def test_snapshot_reads_no_verdict_when_the_fit_supports_none() -> None:
    solved = _snapshot(_fit_result(chi_squared=0.0, dof=0)).ranges[0]

    assert solved.verdict == "no verdict"
    assert solved.colours == NEUTRAL_CHIP_COLOURS


def test_snapshot_renders_in_the_window_without_an_edit_button(qapp: QApplication) -> None:
    window = FitResultsWindow(_snapshot(_fit_result()))

    assert window.windowTitle() == "Fit results — 3001"
    table = window.findChild(QTableWidget)
    # Two free rows, the "held fixed" separator, and the fixed row.
    assert table.rowCount() == 4
    assert [button.text() for button in window.findChildren(QPushButton)] == ["Copy", "Close"]
    window.deleteLater()


def test_window_omits_the_footer_segments_a_run_fit_has_none_of(qapp: QApplication) -> None:
    """A run's asymmetry fit has no trended x axis and no error mode of its own."""
    solved = _range(bounds="0.15 – 12.0 µs")
    run_fit = FitResults(
        title="Fit results — 3001",
        parameter_name="",
        x_label="",
        runs="3001",
        ranges=(FitRangeResults(**{**solved.__dict__, "error_mode": ""}),),
    )
    window = FitResultsWindow(run_fit, editable=False)

    texts = [label.text() for label in window.findChildren(QLabel)]
    assert "fit range 0.15 – 12.0 µs" in texts
    assert not any(text.startswith("x: ") for text in texts)
    assert not any("errors:" in text for text in texts)
    window.deleteLater()
