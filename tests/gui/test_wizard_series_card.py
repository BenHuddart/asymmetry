"""Standalone unit tests for the window-agnostic WizardSeriesCard widget."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.gui.styles.widgets import (
    CONFIDENCE_CHIP_OBJECT_NAME,
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_SUCCESS_STYLE,
)
from asymmetry.gui.widgets.wizard_series_card import (
    SeriesRunTrace,
    SeriesTrend,
    WizardSeriesCard,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _run(label: str, axis_value: float | None, *, fitted: bool = True) -> SeriesRunTrace:
    t = np.linspace(0, 8, 40)
    rate = 0.3 if axis_value is None else 0.2 + 0.1 * axis_value
    y = 0.2 * np.exp(-rate * t) + 0.01 + 0.002 * np.sin(t)
    e = np.full_like(t, 0.01)
    fit_t = t if fitted else None
    fit_y = 0.2 * np.exp(-rate * t) + 0.01 if fitted else None
    return SeriesRunTrace(
        run_label=label,
        axis_value=axis_value,
        time=t,
        asymmetry=y,
        error=e,
        fitted_time=fit_t,
        fitted_curve=fit_y,
    )


def _trend() -> SeriesTrend:
    return SeriesTrend(
        parameter_label="λ (µs⁻¹)",
        axis_label="Field (G)",
        axis_values=(10.0, 20.0, 30.0),
        values=(0.4, 0.6, 0.9),
        errors=(0.02, 0.03, 0.04),
    )


# ── (1) Verdict text, chip presence per tier, frame tint ──────────────────


def test_verdict_high_shows_chip_and_success_frame(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("Exponential — shared λ", "High confidence across the series.", "high")
    assert card._verdict_label.text() == "Exponential — shared λ"
    assert card._confidence_label.text() == "High confidence across the series."
    assert card._chip is not None
    assert card._chip.objectName() == CONFIDENCE_CHIP_OBJECT_NAME
    assert card._frame.styleSheet() == RESULT_BOX_SUCCESS_STYLE


def test_verdict_medium_shows_chip_and_neutral_frame(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("Gaussian — shared σ", "Medium confidence.", "medium")
    assert card._chip is not None
    assert card._frame.styleSheet() == RESULT_BOX_NEUTRAL_STYLE


def test_verdict_none_tier_shows_muted_chip(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("No clear winner", "", "none")
    assert card._chip is not None
    assert card._frame.styleSheet() == RESULT_BOX_NEUTRAL_STYLE
    # Empty confidence text hides the prose line.
    assert card._confidence_label.isVisible() is False


def test_verdict_none_value_shows_no_chip(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("Result", "Some prose.", None)
    assert card._chip is None
    assert card._frame.styleSheet() == RESULT_BOX_NEUTRAL_STYLE


def test_verdict_rebuilds_chip_on_retier(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("A", "high prose", "high")
    first = card._chip
    card.set_verdict("A", "", None)
    assert card._chip is None
    card.set_verdict("A", "medium prose", "medium")
    assert card._chip is not None
    assert card._chip is not first


# ── (2) Series overlay + trend axes count ─────────────────────────────────


def test_set_series_draws_single_axes(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    runs = [_run("R1", 10.0), _run("R2", 20.0, fitted=False), _run("R3", 30.0)]
    card.set_series(runs, "Field (G)")
    assert len(card._figure.axes) == 1


def test_set_trend_adds_second_axes_and_dropping_reverts(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    runs = [_run("R1", 10.0), _run("R2", 20.0), _run("R3", 30.0)]
    card.set_series(runs, "Field (G)")
    assert len(card._figure.axes) == 1
    card.set_trend(_trend())
    assert len(card._figure.axes) == 2
    card.set_trend(None)
    assert len(card._figure.axes) == 1


# ── (3) Alternatives ──────────────────────────────────────────────────────


def test_alternatives_create_click_emits_and_syncs(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_alternatives(
        [
            ("exp", "Exponential", "simpler"),
            ("gauss", "Gaussian", "peer"),
        ]
    )
    assert set(card._alt_buttons) == {"exp", "gauss"}
    emitted: list[str] = []
    card.selection_changed.connect(emitted.append)
    card._alt_buttons["gauss"].click()
    assert emitted == ["gauss"]
    assert card.selected_key() == "gauss"
    assert card._alt_buttons["gauss"].isChecked() is True
    assert card._alt_buttons["exp"].isChecked() is False


def test_set_selected_key_no_reemit_when_unchanged(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_alternatives([("exp", "Exponential", "")])
    card.set_selected_key("exp")
    emitted: list[str] = []
    card.selection_changed.connect(emitted.append)
    card.set_selected_key("exp")  # unchanged → no emit
    assert emitted == []
    card.set_selected_key(None)  # None → no emit but clears checked state
    assert emitted == []
    assert card._alt_buttons["exp"].isChecked() is False


# ── (4) Apply ─────────────────────────────────────────────────────────────


def test_apply_button_emits_and_can_disable(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    emitted: list[int] = []
    card.apply_requested.connect(lambda: emitted.append(1))
    card._apply_btn.click()
    assert emitted == [1]
    card.set_apply_enabled(False)
    assert card._apply_btn.isEnabled() is False
    card._apply_btn.click()
    assert emitted == [1]  # disabled button does not emit


def test_apply_text_default_and_override(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    assert card._apply_btn.text() == "Apply recommended fit"
    card.set_apply_text("Use this global fit")
    assert card._apply_btn.text() == "Use this global fit"


# ── (5) clear() ───────────────────────────────────────────────────────────


def test_clear_resets_then_series_works(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    card.set_verdict("Winner", "High confidence.", "high")
    card.set_alternatives([("exp", "Exponential", "")])
    card.set_series([_run("R1", 10.0), _run("R2", 20.0)], "Field (G)")
    card.set_trend(_trend())

    card.clear()
    assert card._verdict_label.text() == ""
    assert card._confidence_label.text() == ""
    assert card._chip is None
    assert card._alt_buttons == {}
    assert card._frame.styleSheet() == RESULT_BOX_NEUTRAL_STYLE
    assert len(card._figure.axes) == 1  # trend dropped → single overlay axes

    # A subsequent set_series still works.
    card.set_series([_run("R1", 5.0), _run("R2", 15.0), _run("R3", 25.0)], "Temp (K)")
    assert len(card._figure.axes) == 1


# ── (6) Ungraded runs (None axis_value) fall back without crashing ────────


def test_ungraded_runs_use_fallback_palette(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    runs = [_run("R1", None), _run("R2", None), _run("R3", None)]
    card.set_series(runs, "Field (G)")
    assert len(card._figure.axes) == 1


def test_degenerate_axis_range_uses_fallback(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    # All identical axis values → degenerate range → fallback palette, no crash.
    runs = [_run("R1", 10.0), _run("R2", 10.0)]
    card.set_series(runs, "Field (G)")
    assert len(card._figure.axes) == 1


def test_many_runs_suppress_legend(qapp: QApplication) -> None:
    card = WizardSeriesCard()
    runs = [_run(f"R{i}", float(i)) for i in range(10)]  # > 8 → no legend
    card.set_series(runs, "Field (G)")
    assert len(card._figure.axes) == 1
    ax = card._figure.axes[0]
    assert ax.get_legend() is None
