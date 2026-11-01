"""Screen reports preserve unavailable criteria in text and JSON."""

from pathlib import Path

from asymmetry.cli.commands.wizard import _render
from asymmetry.core.workflow.screen import ScreenCandidate, ScreenResult


def test_screen_report_displays_na_and_serializes_reason():
    reason = "n − k − 1 = 0"
    candidate = ScreenCandidate(
        key="small",
        title="Small sample",
        category="General",
        aicc=None,
        aicc_unavailable=reason,
        chi2_red=1.0,
        parameter_count=3,
        is_recommended=False,
        is_comparable=False,
    )
    result = ScreenResult(
        run_number=1,
        geometry="ZF",
        geometry_source="user",
        scope_preset="broad",
        scope_include=[],
        scope_exclude=[],
        scope_note="",
        confidence="low",
        verdict="Inspect diagnostics",
        caveat="",
        summary="",
        recommended_key=None,
        candidates=[candidate],
        narrative="",
        recommendation={},
        recipe=None,
        runner_up=None,
        lines=[],
        time_window=(0.0, 1.0),
    )
    text = _render("work", None, result, Path("wizard.json"), None, None, None)
    assert "n/a" in text
    assert "inf" not in text
    row = result.to_dict()["candidates"][0]
    assert row["aicc"] is None
    assert row["aicc_unavailable"] == reason
