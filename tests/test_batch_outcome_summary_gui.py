"""Post-batch outcome summary line (Phase 2.2 / F2).

After a batch fit the results block must account for the whole selection —
fitted / failed / flagged — so a silently dropped or garbage member is visible
in the panel, not just the log.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import GlobalFitTab


def _result(*, success=True, value=20.0, err=0.4) -> FitResult:
    ps = ParameterSet([Parameter("A_1", value=value)])
    return FitResult(
        success=success, reduced_chi_squared=1.0, parameters=ps, uncertainties={"A_1": err}
    )


def test_summary_counts_fitted_failed_and_flagged(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = {
        2960: _result(),
        2955: _result(),
        2950: _result(success=False),  # failed
        2949: _result(),  # flagged via engine flags below
    }
    tab._series_seeding_meta = {"member_flags": {2949: ["spurious_reseeded"]}}
    labels = {run: str(run) for run in results}
    line = tab._batch_outcome_summary_line(results, labels)
    assert line.startswith("3/4 fitted")
    assert "1 failed (2950)" in line
    assert "1 flagged (2949)" in line


def test_summary_all_clean_reports_only_fitted(qapp: QApplication) -> None:
    tab = GlobalFitTab(member_kind="runs")
    results = {2960: _result(), 2955: _result()}
    tab._series_seeding_meta = None
    line = tab._batch_outcome_summary_line(results, {2960: "2960", 2955: "2955"})
    assert line == "2/2 fitted"


def test_summary_derives_flags_from_result_when_engine_flags_absent(qapp: QApplication) -> None:
    # No stashed engine flags (e.g. the tied global_fit path): the generic
    # flags are recomputed from each FitResult, so a huge-relative-error member
    # is still counted as flagged.
    tab = GlobalFitTab(member_kind="runs")
    results = {
        2960: _result(),
        2949: _result(value=0.5, err=10.0),  # |σ/value| ≫ 1 → large_rel_err
    }
    tab._series_seeding_meta = None
    line = tab._batch_outcome_summary_line(results, {2960: "2960", 2949: "2949"})
    assert "1 flagged (2949)" in line
