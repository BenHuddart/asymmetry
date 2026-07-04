"""Tests for the Phase 5 two-study comparison view.

Covers the pure helpers (:func:`count_free_parameters`,
:func:`information_criteria`), the :class:`GlobalFitCompareDialog` (grid,
per-panel notes, stats/Δ block, union parameter table with discrepancy
highlighting, comparability caveats), and the MainWindow wiring (sidebar
"Compare with…" candidate filtering + the compare handler opening/replacing a
single dialog).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore  # noqa: E402
from PySide6.QtWidgets import QApplication  # type: ignore  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.representation.global_fit_study import GlobalFitStudy  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.windows.global_fit_compare_dialog import (  # noqa: E402
    GlobalFitCompareDialog,
)
from asymmetry.gui.windows.global_fit_window_helpers import (  # noqa: E402
    count_free_parameters,
    information_criteria,
)
from asymmetry.gui.windows.global_parameter_fit_window import (  # noqa: E402
    StudySidebarEntry,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


# ── synthetic builders ──────────────────────────────────────────────────────


def _group(gid: str, gname: str, *, y: float = 0.15) -> ParameterGroupData:
    return ParameterGroupData(
        group_id=gid,
        group_name=gname,
        x=np.array([100.0, 200.0, 300.0], dtype=float),
        y=np.array([y, y, y], dtype=float),
        yerr=np.array([0.01, 0.01, 0.01], dtype=float),
        group_variable_value=10.0,
    )


def _result(
    *,
    chi2: float = 3.0,
    n_points: int = 6,
    global_params: ParameterSet | None = None,
    global_uncertainties: dict[str, float] | None = None,
    local_groups: list[str] | None = None,
    error_mode: str = "column",
) -> CrossGroupFitResult:
    if global_params is None:
        global_params = ParameterSet([Parameter("c", value=0.15)])
    if local_groups is None:
        local_groups = ["g0", "g1"]
    return CrossGroupFitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / max(n_points, 1),
        global_parameters=global_params,
        local_parameters={gid: ParameterSet() for gid in local_groups},
        fixed_parameters=ParameterSet(),
        global_uncertainties=global_uncertainties or {},
        error_mode=error_mode,
        n_points=n_points,
    )


def _study(
    study_id: str,
    name: str,
    groups: list[ParameterGroupData],
    result: CrossGroupFitResult,
    *,
    parameter_name: str = "Lambda",
    x_key: str = "field",
    input_digest: str = "digest",
) -> GlobalFitStudy:
    return GlobalFitStudy(
        study_id=study_id,
        name=name,
        parameter_name=parameter_name,
        x_key=x_key,
        x_label="B (T)",
        group_variable_key="temperature",
        group_variable_label="T (K)",
        created="",
        updated="",
        source_group_ids=[g.group_id for g in groups],
        groups=groups,
        model=ParameterCompositeModel(["Constant"]),
        config={},
        result=result,
        input_digest=input_digest,
    )


# ── helpers: count_free_parameters ───────────────────────────────────────────


def test_count_free_parameters_globals_plus_locals() -> None:
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=1.0), Parameter("b", value=2.0)]),
        local_parameters={
            "g0": ParameterSet(
                [Parameter("x", value=1.0), Parameter("y", value=2.0), Parameter("z", value=3.0)]
            ),
            "g1": ParameterSet(
                [Parameter("x", value=1.0), Parameter("y", value=2.0), Parameter("z", value=3.0)]
            ),
        },
        # Fixed params must NOT count toward k.
        fixed_parameters=ParameterSet([Parameter("f", value=9.0)]),
    )
    # 2 globals + 3 locals × 2 groups = 8.
    assert count_free_parameters(result) == 8


def test_count_free_parameters_none() -> None:
    assert count_free_parameters(None) == 0


# ── helpers: information_criteria ────────────────────────────────────────────


def test_information_criteria_values_match_hand_computation() -> None:
    # k = 2 globals + 0 locals = 2; n = 20; chi2 = 10.
    result = _result(
        chi2=10.0,
        n_points=20,
        global_params=ParameterSet([Parameter("a", value=1.0), Parameter("b", value=2.0)]),
        local_groups=["g0", "g1"],
    )
    ic = information_criteria(result)
    assert ic is not None
    k, n, chi2 = 2, 20, 10.0
    assert ic["k"] == k
    assert ic["n"] == n
    assert ic["aic"] == pytest.approx(chi2 + 2 * k)
    assert ic["aicc"] == pytest.approx(chi2 + 2 * k + (2 * k * (k + 1)) / (n - k - 1))
    assert ic["bic"] == pytest.approx(chi2 + k * math.log(n))


def test_information_criteria_none_when_no_points() -> None:
    assert information_criteria(_result(n_points=0)) is None


def test_information_criteria_aicc_inf_when_overparameterized() -> None:
    # n - k - 1 <= 0 → aicc = inf. k = 2 globals, n = 3 → 3 - 2 - 1 = 0.
    result = _result(
        chi2=1.0,
        n_points=3,
        global_params=ParameterSet([Parameter("a", value=1.0), Parameter("b", value=2.0)]),
    )
    ic = information_criteria(result)
    assert ic is not None
    assert math.isinf(ic["aicc"])


# ── dialog: grid + per-panel notes ───────────────────────────────────────────


def _two_studies_shared_plus_b_only():
    """Study A: g0,g1. Study B: g0,g1,g2 (g2 is B-only)."""
    groups_a = [_group("g0", "G0"), _group("g1", "G1")]
    groups_b = [_group("g0", "G0"), _group("g1", "G1"), _group("g2", "G2")]
    result_a = _result(local_groups=["g0", "g1"], n_points=6)
    result_b = _result(local_groups=["g0", "g1", "g2"], n_points=9)
    study_a = _study("id-a", "Study A", groups_a, result_a)
    study_b = _study("id-b", "Study B", groups_b, result_b)
    return study_a, study_b


def test_dialog_grid_has_union_of_group_panels(qapp: QApplication) -> None:
    study_a, study_b = _two_studies_shared_plus_b_only()
    dialog = GlobalFitCompareDialog(study_a, study_b)
    # 3 groups in the union → 3 axes.
    assert dialog._figure is not None
    assert len(dialog._figure.axes) == 3


def _curve_lines(ax):
    """Return only the study model curves on *ax* (exclude errorbar artifacts).

    A ``capsize`` errorbar contributes extra ``Line2D`` objects (caps); the
    model curves are the ``ax.plot`` lines, identified here by their explicit
    curve colours.
    """
    from asymmetry.gui.windows.global_fit_compare_dialog import (
        _CURVE_COLOR_A,
        _CURVE_COLOR_B,
    )

    curve_colors = {_CURVE_COLOR_A, _CURVE_COLOR_B}
    return [ln for ln in ax.get_lines() if ln.get_color() in curve_colors]


def test_dialog_shared_panel_has_two_curves_one_dashed(qapp: QApplication) -> None:
    study_a, study_b = _two_studies_shared_plus_b_only()
    dialog = GlobalFitCompareDialog(study_a, study_b)
    axes = dialog._figure.axes
    # The first panel is a shared group (g0): two model curves, one dashed.
    shared_ax = axes[0]
    curves = _curve_lines(shared_ax)
    line_styles = [ln.get_linestyle() for ln in curves]
    assert len(curves) == 2
    assert any(ls in ("--", "dashed") for ls in line_styles)
    assert any(ls in ("-", "solid") for ls in line_styles)


def test_dialog_b_only_panel_noted(qapp: QApplication) -> None:
    study_a, study_b = _two_studies_shared_plus_b_only()
    dialog = GlobalFitCompareDialog(study_a, study_b)
    axes = dialog._figure.axes
    # The g2 panel (B-only) is the last one and carries a "B only" note.
    b_only_ax = axes[2]
    texts = [t.get_text() for t in b_only_ax.texts]
    assert any("B only" in t for t in texts)
    # And only one model curve (study B's).
    assert len(_curve_lines(b_only_ax)) == 1


# ── dialog: stats + Δ block ──────────────────────────────────────────────────


def test_dialog_stats_columns_show_both_names(qapp: QApplication) -> None:
    study_a, study_b = _two_studies_shared_plus_b_only()
    dialog = GlobalFitCompareDialog(study_a, study_b)
    headers = [
        dialog._stats_table.horizontalHeaderItem(c).text()
        for c in range(dialog._stats_table.columnCount())
    ]
    assert "Study A" in headers
    assert "Study B" in headers
    assert any("Δ" in h for h in headers)


def test_dialog_delta_row_present(qapp: QApplication) -> None:
    study_a, study_b = _two_studies_shared_plus_b_only()
    dialog = GlobalFitCompareDialog(study_a, study_b)
    table = dialog._stats_table
    # The Δ column (index 3) has at least one numeric (non "—") entry.
    delta_texts = [table.item(r, 3).text() for r in range(table.rowCount())]
    assert any(t not in ("—", "") for t in delta_texts)


# ── dialog: union parameter table + discrepancy highlight ────────────────────


def test_dialog_param_table_union_with_dash_for_absent(qapp: QApplication) -> None:
    # Study A has c and d (global); study B has only c → d absent in B.
    groups = [_group("g0", "G0"), _group("g1", "G1")]
    result_a = _result(
        global_params=ParameterSet([Parameter("c", value=0.15), Parameter("A_bg", value=0.02)]),
        global_uncertainties={"c": 0.01, "A_bg": 0.001},
    )
    result_b = _result(
        global_params=ParameterSet([Parameter("c", value=0.15)]),
        global_uncertainties={"c": 0.01},
    )
    study_a = _study("id-a", "Study A", groups, result_a)
    study_b = _study("id-b", "Study B", groups, result_b)
    dialog = GlobalFitCompareDialog(study_a, study_b)
    table = dialog._param_table
    # Union has 2 param rows (c, A_bg).
    assert table.rowCount() == 2
    # The A_bg row's B cell (absent) is "—".
    b_texts = {table.item(r, 0).text(): table.item(r, 2).text() for r in range(table.rowCount())}
    # A_bg present only in A → its B cell is a dash.
    a_bg_key = next(k for k in b_texts if k.startswith("A_bg") or "A" in k and "bg" in k.lower())
    assert b_texts[a_bg_key] == "—"


def test_dialog_discrepancy_highlight_on_2sigma_pair(qapp: QApplication) -> None:
    # c differs by 0.10 with tiny errors → far more than 2σ → highlighted.
    groups = [_group("g0", "G0"), _group("g1", "G1")]
    result_a = _result(
        global_params=ParameterSet([Parameter("c", value=0.10)]),
        global_uncertainties={"c": 0.001},
    )
    result_b = _result(
        global_params=ParameterSet([Parameter("c", value=0.20)]),
        global_uncertainties={"c": 0.001},
    )
    study_a = _study("id-a", "Study A", groups, result_a)
    study_b = _study("id-b", "Study B", groups, result_b)
    dialog = GlobalFitCompareDialog(study_a, study_b)
    table = dialog._param_table
    item_a = table.item(0, 1)
    item_b = table.item(0, 2)
    # Both value cells carry a (non-default) warning background.
    bg_a = item_a.background()
    bg_b = item_b.background()
    assert bg_a.style() != 0  # a brush was set
    assert bg_b.style() != 0


# ── dialog: comparability caveats ────────────────────────────────────────────


def test_dialog_error_mode_mismatch_shows_caveat_and_suppresses_bold(qapp: QApplication) -> None:
    groups = [_group("g0", "G0"), _group("g1", "G1")]
    result_a = _result(chi2=3.0, n_points=6, error_mode="column")
    result_b = _result(chi2=5.0, n_points=6, error_mode="scatter")
    study_a = _study("id-a", "Study A", groups, result_a)
    study_b = _study("id-b", "Study B", groups, result_b)
    dialog = GlobalFitCompareDialog(study_a, study_b)
    assert dialog._criteria_comparable is False
    # A caveat label with the expected text is present among the dialog's labels.
    from PySide6.QtWidgets import QLabel

    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert any("Criteria not comparable" in t for t in labels)
    # No stats cell is bold (Δ bolding suppressed).
    table = dialog._stats_table
    any_bold = any(
        table.item(r, c) is not None and table.item(r, c).font().bold()
        for r in range(table.rowCount())
        for c in range(1, 3)
    )
    assert not any_bold


def test_dialog_delta_bolds_better_side_when_comparable(qapp: QApplication) -> None:
    groups = [_group("g0", "G0"), _group("g1", "G1")]
    # Same n and error mode → comparable. B has lower chi2 → B better on χ²ᵣ/AIC.
    result_a = _result(chi2=10.0, n_points=8, error_mode="column")
    result_b = _result(chi2=4.0, n_points=8, error_mode="column")
    study_a = _study("id-a", "Study A", groups, result_a)
    study_b = _study("id-b", "Study B", groups, result_b)
    dialog = GlobalFitCompareDialog(study_a, study_b)
    assert dialog._criteria_comparable is True
    table = dialog._stats_table
    # Some column-B cell is bold (B is the better/lower side).
    any_b_bold = any(
        table.item(r, 2) is not None and table.item(r, 2).font().bold()
        for r in range(table.rowCount())
    )
    assert any_b_bold


def test_dialog_snapshot_caveat_on_digest_mismatch(qapp: QApplication) -> None:
    groups = [_group("g0", "G0"), _group("g1", "G1")]
    study_a = _study("id-a", "Study A", groups, _result(chi2=3.0), input_digest="aaa")
    study_b = _study("id-b", "Study B", groups, _result(chi2=5.0), input_digest="bbb")
    dialog = GlobalFitCompareDialog(study_a, study_b)
    from PySide6.QtWidgets import QLabel

    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert any("different data snapshots" in t for t in labels)
    # The generic mode/n caveat does not apply: same error mode, same n.
    assert not any("different data/error mode" in t for t in labels)
    # Digest mismatch alone must suppress the Δ-column bolding: two studies
    # fitted to different data snapshots are not criterion-comparable even
    # when their error mode and point count coincide.
    assert not dialog._criteria_comparable
    table = dialog._stats_table
    any_bold = any(
        table.item(r, c) is not None and table.item(r, c).font().bold()
        for r in range(table.rowCount())
        for c in range(table.columnCount())
    )
    assert not any_bold


# ── window sidebar: Compare with… candidate filtering + signal ───────────────


def test_context_menu_lists_only_same_param_and_x_key(qapp: QApplication) -> None:
    from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow

    window = GlobalParameterFitWindow()
    window.set_studies_list(
        [
            StudySidebarEntry("id-a", "A study", False, "Lambda", "field"),
            StudySidebarEntry("id-b", "B study", False, "Lambda", "field"),
            # Same parameter, different x_key → NOT a candidate.
            StudySidebarEntry("id-c", "C study", False, "Lambda", "temperature"),
            # Different parameter → NOT a candidate.
            StudySidebarEntry("id-d", "D study", False, "D_2D", "field"),
        ]
    )
    # Reproduce the candidate filter the context menu uses.
    this = window._sidebar_entries["id-a"]
    candidates = [
        e.study_id
        for e in window._sidebar_entries.values()
        if e.study_id != "id-a"
        and e.parameter_name == this.parameter_name
        and e.x_key == this.x_key
    ]
    assert candidates == ["id-b"]


def test_compare_signal_emits_right_ids(qapp: QApplication) -> None:
    from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow

    window = GlobalParameterFitWindow()
    window.set_studies_list(
        [
            StudySidebarEntry("id-a", "A study", False, "Lambda", "field"),
            StudySidebarEntry("id-b", "B study", False, "Lambda", "field"),
        ]
    )
    fired: list[tuple[str, str]] = []
    window.study_compare_requested.connect(lambda a, b: fired.append((a, b)))
    # Emit directly (the QMenu.exec path is interactive and cannot be driven
    # headlessly); this exercises the same signal contract the menu uses.
    window.study_compare_requested.emit("id-a", "id-b")
    assert fired == [("id-a", "id-b")]


# ── mainwindow handler: open + replace single dialog ─────────────────────────


def _register_two_studies(mainwindow: MainWindow):
    groups_a = [_group("g0", "G0"), _group("g1", "G1")]
    groups_b = [_group("g0", "G0"), _group("g1", "G1")]
    study_a = _study("id-a", "Study A", groups_a, _result(chi2=3.0), input_digest="aaa")
    study_b = _study("id-b", "Study B", groups_b, _result(chi2=5.0), input_digest="aaa")
    study_c = _study("id-c", "Study C", groups_a, _result(chi2=4.0), input_digest="aaa")
    mainwindow._global_fit_studies["id-a"] = study_a
    mainwindow._global_fit_studies["id-b"] = study_b
    mainwindow._global_fit_studies["id-c"] = study_c
    return study_a, study_b, study_c


def test_mainwindow_compare_handler_opens_dialog(mainwindow: MainWindow) -> None:
    _register_two_studies(mainwindow)
    assert mainwindow._global_fit_compare_dialog is None
    mainwindow._on_global_fit_study_compare_requested("id-a", "id-b")
    dialog = mainwindow._global_fit_compare_dialog
    assert isinstance(dialog, GlobalFitCompareDialog)


def test_mainwindow_compare_handler_replaces_previous(mainwindow: MainWindow) -> None:
    _register_two_studies(mainwindow)
    mainwindow._on_global_fit_study_compare_requested("id-a", "id-b")
    first = mainwindow._global_fit_compare_dialog
    assert first is not None
    mainwindow._on_global_fit_study_compare_requested("id-a", "id-c")
    second = mainwindow._global_fit_compare_dialog
    assert second is not None
    # A new dialog instance replaced the first.
    assert second is not first


def test_mainwindow_compare_handler_ignores_missing_study(mainwindow: MainWindow) -> None:
    _register_two_studies(mainwindow)
    mainwindow._on_global_fit_study_compare_requested("id-a", "does-not-exist")
    assert mainwindow._global_fit_compare_dialog is None
