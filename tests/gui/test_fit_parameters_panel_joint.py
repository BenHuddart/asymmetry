"""Parameters panel joint-fit surfacing (``docs/plans/joint-fit.md`` phase 4).

Covers the three phase-4 additions: ``_GroupFitData``'s derived
``joint_fit_id``/``joint_fit_label``/``shared_params`` fields (supplied by
``load_representation_series``, never serialised — mirrors ``phase``/
``short_name``), a shared parameter's "Shared" badge on its card, and the flat
trend line for a series-Global parameter (one line per series, collapsed to
one line across the union of x extents when several loaded joint-fit members
share it). The chip-rail section itself is built in
``MainWindow._trend_panel_sections`` — see
``test_trend_panel_sections_group_joint_fit_members_ahead_of_data_groups`` in
``tests/gui/test_series_workflow.py``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow, _GroupFitData
from tests.gui._trend_panel import axes_for, card, select_params


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _row(run_number: int, field: float, **values: float) -> _FitRow:
    return _FitRow(
        run_number=run_number,
        run_label=str(run_number),
        field=field,
        temperature=10.0,
        values=dict(values),
        errors={name: 0.01 for name in values},
    )


def _stamped_group(
    group_id: str,
    name: str,
    rows: list[_FitRow],
    global_names: list[str],
    shared_params: dict[str, str],
    *,
    joint_fit_id: str = "joint-1",
    joint_fit_label: str = "High-field joint fit",
) -> _GroupFitData:
    return _GroupFitData(
        group_id=group_id,
        group_name=name,
        rows=rows,
        global_params=ParameterSet([Parameter(n, value=rows[0].values[n]) for n in global_names]),
        varying_params=[],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
        joint_fit_id=joint_fit_id,
        joint_fit_label=joint_fit_label,
        shared_params=dict(shared_params),
    )


def _panel_with_joint_members(qapp: QApplication) -> FitParametersPanel:
    """Two joint-fit members: A shares ``A_bg``/``Bg`` → ``A_bg_shared``; A also
    carries ``phase``, an ordinary (unshared) Global-role parameter of its own.
    """
    panel = FitParametersPanel()
    rows_a = [_row(1, 100.0, A_bg=0.05, phase=0.2), _row(2, 150.0, A_bg=0.05, phase=0.2)]
    rows_b = [_row(10, 300.0, Bg=0.05), _row(11, 400.0, Bg=0.05)]
    panel._group_fit_results = {
        "a": _stamped_group("a", "Series A", rows_a, ["A_bg", "phase"], {"A_bg": "A_bg_shared"}),
        "b": _stamped_group("b", "Series B", rows_b, ["Bg"], {"Bg": "A_bg_shared"}),
    }
    panel._active_group_id = "a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    # The flat line's "one line, one legend entry" shape is an Overlay-canvas
    # concept (Subplots draws one figure per card with no cross-series union);
    # the panel defaults to Subplots, so the flat-line tests below switch.
    panel._overlay_button.setChecked(True)
    return panel


# ── _GroupFitData plumbing (load_representation_series) ────────────────────


def test_load_representation_series_threads_joint_fit_fields(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    entries = [
        (
            "a",
            "Series A",
            [
                {
                    "run_number": 1,
                    "run_label": "1",
                    "field": 100.0,
                    "temperature": 10.0,
                    "values": {"A_bg": 0.05},
                    "errors": {"A_bg": 0.01},
                }
            ],
        )
    ]
    panel.load_representation_series(
        entries,
        joint_fit_by_id={"a": ("joint-1", "High-field joint fit")},
        shared_params_by_id={"a": {"A_bg": "A_bg_shared"}},
    )
    group = panel._group_fit_results["a"]
    assert group.joint_fit_id == "joint-1"
    assert group.joint_fit_label == "High-field joint fit"
    assert group.shared_params == {"A_bg": "A_bg_shared"}


def test_load_representation_series_defaults_joint_fields_when_absent(
    qapp: QApplication,
) -> None:
    """An unstamped series (or a caller that has not migrated) gets the empty
    defaults — the field-4 side of "an unstamped series renders exactly as
    before"."""
    panel = FitParametersPanel()
    entries = [
        (
            "p",
            "Plain series",
            [
                {
                    "run_number": 1,
                    "run_label": "1",
                    "field": 100.0,
                    "temperature": 10.0,
                    "values": {"Lambda": 0.1},
                    "errors": {"Lambda": 0.01},
                }
            ],
        )
    ]
    panel.load_representation_series(entries)
    group = panel._group_fit_results["p"]
    assert group.joint_fit_id is None
    assert group.joint_fit_label == ""
    assert group.shared_params == {}


def test_serialize_group_fit_results_excludes_joint_fields(qapp: QApplication) -> None:
    """The joint-fit fields are derived display state, not persisted (D5)."""
    panel = _panel_with_joint_members(qapp)
    payload = panel._serialize_group_fit_results()
    assert "joint_fit_id" not in payload["a"]
    assert "joint_fit_label" not in payload["a"]
    assert "shared_params" not in payload["a"]


# ── "Shared" badge ───────────────────────────────────────────────────────────


def test_shared_parameter_card_shows_badge_and_unshared_does_not(qapp: QApplication) -> None:
    panel = _panel_with_joint_members(qapp)
    select_params(panel, ["A_bg", "phase"])

    shared_card = card(panel, "A_bg")
    assert shared_card.shared_chip is not None
    assert "Shared" in shared_card.shared_chip.text()
    tooltip = shared_card.shared_chip.toolTip()
    assert 'joint fit "High-field joint fit"' in tooltip
    assert "Series B" in tooltip

    unshared_card = card(panel, "phase")
    assert unshared_card.shared_chip is None


def test_unstamped_series_never_offers_a_global_parameter_as_y(qapp: QApplication) -> None:
    """A plain (non-jointed) series' Global-role parameter stays excluded —
    the panel's long-standing "held constant" rule (:meth:`_shared_held_constant_params`),
    untouched by the new joint-fit flat-line mechanism."""
    panel = FitParametersPanel()
    rows = [_row(1, 100.0, A_bg=0.2, Lambda=0.1), _row(2, 200.0, A_bg=0.2, Lambda=0.2)]
    panel._group_fit_results = {
        "p": _GroupFitData(
            group_id="p",
            group_name="Plain series",
            rows=rows,
            global_params=ParameterSet([Parameter("A_bg", value=0.2)]),
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._active_group_id = "p"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["p"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    assert "A_bg" not in panel._display_y_parameters()
    assert "A_bg" not in panel._y_chips
    assert panel._shared_held_constant_params() == ["A_bg"]


# ── Flat trend line ─────────────────────────────────────────────────────────


def _named_lines(ax):
    return [
        line for line in ax.get_lines() if line.get_label() and not line.get_label().startswith("_")
    ]


def test_unshared_global_parameter_on_jointed_series_draws_its_own_flat_line(
    qapp: QApplication,
) -> None:
    panel = _panel_with_joint_members(qapp)
    select_params(panel, ["phase"])

    ax = axes_for(panel, "phase")
    lines = _named_lines(ax)
    assert len(lines) == 1
    xs = lines[0].get_xdata()
    assert min(xs) == pytest.approx(100.0)
    assert max(xs) == pytest.approx(150.0)
    assert lines[0].get_ydata()[0] == pytest.approx(0.2)
    assert "Shared" not in lines[0].get_label()


def test_shared_parameter_draws_one_flat_line_across_union_of_x_extents(
    qapp: QApplication,
) -> None:
    panel = _panel_with_joint_members(qapp)
    panel._set_selected_group_ids(["a", "b"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    select_params(panel, ["A_bg"])

    ax = axes_for(panel, "A_bg")
    lines = _named_lines(ax)
    assert len(lines) == 1
    xs = lines[0].get_xdata()
    assert min(xs) == pytest.approx(100.0)
    assert max(xs) == pytest.approx(400.0)
    assert lines[0].get_ydata()[0] == pytest.approx(0.05)
    assert "Shared" in lines[0].get_label()

    _handles, labels = ax.get_legend_handles_labels()
    assert sum(1 for label in labels if "Shared" in label) == 1
