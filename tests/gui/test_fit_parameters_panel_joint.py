"""Parameters panel joint-fit surfacing (``docs/plans/joint-fit.md`` phase 4).

Covers the two phase-4 additions that survived the 2026-09-18 review:
``_GroupFitData``'s derived ``joint_fit_id``/``joint_fit_label``/
``shared_params`` fields (supplied by ``load_representation_series``, never
serialised — mirrors ``phase``/``short_name``), and the "held constant"
footer hint naming which joint fit a shared parameter is shared across
(Decision A). The first draft of this phase also gave a shared parameter a
card, a "Shared" badge and a flat trend line; Ben withdrew that after
reviewing the feature on a real project, because "Global" means one value
per series everywhere in the app, and a shared parameter is still, in every
contributing series, a Global one — so it gets no chip and no card, exactly
like any other Global parameter. The chip-rail section itself is built in
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
from asymmetry.gui.utils.formatting import format_param_label
from tests.gui._trend_panel import select_params


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
    varying_params: list[str] = (),
    joint_fit_id: str = "joint-1",
    joint_fit_label: str = "High-field joint fit",
) -> _GroupFitData:
    return _GroupFitData(
        group_id=group_id,
        group_name=name,
        rows=rows,
        global_params=ParameterSet([Parameter(n, value=rows[0].values[n]) for n in global_names]),
        varying_params=list(varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
        joint_fit_id=joint_fit_id,
        joint_fit_label=joint_fit_label,
        shared_params=dict(shared_params),
    )


def _panel_with_joint_members(qapp: QApplication) -> FitParametersPanel:
    """Two joint-fit members: A shares ``A_bg``/``Bg`` → ``A_bg_shared``; A also
    carries ``phase`` (an ordinary, unshared Global-role parameter of its own)
    and ``Lambda`` (an everyday per-run varying parameter, for contrast with
    the "held constant" set, which only ever names Global-role parameters).
    """
    panel = FitParametersPanel()
    rows_a = [
        _row(1, 100.0, A_bg=0.05, phase=0.2, Lambda=0.10),
        _row(2, 150.0, A_bg=0.05, phase=0.2, Lambda=0.15),
    ]
    rows_b = [_row(10, 300.0, Bg=0.05), _row(11, 400.0, Bg=0.05)]
    panel._group_fit_results = {
        "a": _stamped_group(
            "a",
            "Series A",
            rows_a,
            ["A_bg", "phase"],
            {"A_bg": "A_bg_shared"},
            varying_params=["Lambda"],
        ),
        "b": _stamped_group("b", "Series B", rows_b, ["Bg"], {"Bg": "A_bg_shared"}),
    }
    panel._active_group_id = "a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
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


# ── Decision A: a shared parameter is still a Global one ───────────────────


def test_shared_parameter_gets_no_chip_or_card(qapp: QApplication) -> None:
    """A shared parameter is, in every contributing series, still a Global
    parameter — it gets no chip and no card, exactly like an unshared one."""
    panel = _panel_with_joint_members(qapp)

    assert "A_bg" not in panel._display_y_parameters()
    assert "A_bg" not in panel._y_chips

    # An attempt to select it (e.g. a stale preference from before a project
    # restore) is simply a no-op — there is no chip to check.
    select_params(panel, ["A_bg", "Lambda"])
    assert "A_bg" not in {c.name for c in panel._card_stack.cards()}
    assert "Lambda" in {c.name for c in panel._card_stack.cards()}


def test_shared_parameter_draws_no_flat_line_when_two_members_are_overlaid(
    qapp: QApplication,
) -> None:
    """The withdrawn first draft drew one flat line across a shared column's
    members on the Overlay canvas; there is nothing left to draw, because the
    parameter never reaches the y rail in the first place."""
    panel = _panel_with_joint_members(qapp)
    panel._overlay_button.setChecked(True)
    panel._set_selected_group_ids(["a", "b"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    assert "A_bg" not in panel._display_y_parameters()
    select_params(panel, ["A_bg"])
    assert panel._card_stack.cards() == []


def test_shared_and_unshared_global_parameters_both_held_constant_with_hint(
    qapp: QApplication,
) -> None:
    """ "Global" means one value per series everywhere in the app, whether or
    not the joint fit shares the column: both "A_bg" (shared) and "phase"
    (an ordinary Global-role parameter of series A alone) stay off the y rail
    and are named by the footer's "held constant" hint. Only "A_bg"'s entry
    in that hint gains the joint-fit suffix."""
    panel = _panel_with_joint_members(qapp)

    assert "phase" not in panel._display_y_parameters()
    assert "A_bg" not in panel._display_y_parameters()
    assert panel._shared_held_constant_params() == ["A_bg", "phase"]

    panel._update_global_param_hint()
    assert not panel._global_param_hint.isHidden()
    text = panel._global_param_hint.text()
    shared_label = format_param_label("A_bg")
    plain_label = format_param_label("phase")
    assert f'{shared_label} — shared across joint fit "High-field joint fit"' in text
    assert plain_label in text
    assert f"{plain_label} — shared" not in text
    assert text.endswith("Global — held constant")


def test_unstamped_series_never_offers_a_global_parameter_as_y(qapp: QApplication) -> None:
    """A plain (non-jointed) series' Global-role parameter stays excluded —
    the panel's long-standing "held constant" rule (:meth:`_shared_held_constant_params`),
    untouched by the joint-fit hint suffix (there is no joint fit to name)."""
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

    panel._update_global_param_hint()
    assert not panel._global_param_hint.isHidden()
    text = panel._global_param_hint.text()
    assert "shared across joint fit" not in text
