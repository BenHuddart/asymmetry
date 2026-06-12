"""FitLog formatter + provenance enrichment (fit-workflow-diagnostics)."""

from __future__ import annotations

from asymmetry.core.fitting.fit_log import FitLog, enrich_summary_provenance


def _record() -> dict:
    return {
        "success": True,
        "chi_squared": 118.0,
        "reduced_chi_squared": 1.0,
        "parameters": {"A0": 0.19811, "sig": 0.05134},
        "uncertainties": {"A0": 0.00428, "sig": 0.00363},
        "uncertainties_asymmetric": {"sig": [-0.00375, 0.00352]},
        "quality": {
            "verdict": "good",
            "chi2_reduced": 1.0,
            "band_low": 0.74,
            "band_high": 1.30,
            "confidence": 0.95,
            "dof": 118,
        },
    }


def test_enrich_provenance_only_sets_provided_keys():
    summary = {"success": True}
    enrich_summary_provenance(
        summary, model_name="KuboToyabe", fit_range="0–12 µs", timestamp="2026-06-12T10:00:00"
    )
    assert summary["model_name"] == "KuboToyabe"
    assert summary["fit_range"] == "0–12 µs"
    assert summary["timestamp"] == "2026-06-12T10:00:00"
    assert "provenance" not in summary  # None-valued keys are not written
    # A second call does not erase earlier values.
    enrich_summary_provenance(summary, provenance="single")
    assert summary["model_name"] == "KuboToyabe"
    assert summary["provenance"] == "single"


def test_format_record_contains_verdict_and_asymmetric_errors():
    block = FitLog().format_record(_record(), title="Run 1234")
    assert "=== Run 1234 ===" in block
    assert "chi^2/nu = 1.0000" in block
    assert "(nu=118)" in block
    assert "[good" in block and "95%" in block
    # symmetric error rendered; sig also carries the asymmetric overlay.
    assert "A0" in block and "+/-" in block
    assert "+0.00352 / -0.00375" in block


def test_format_record_tolerates_minimal_legacy_record():
    block = FitLog().format_record({"parameters": {"A0": 0.2}})
    assert "A0" in block
    assert "=== Fit ===" in block  # no timestamp -> bare title


def test_format_record_flags_non_convergence():
    block = FitLog().format_record({"success": False, "parameters": {}})
    assert "did not converge" in block


def test_format_report_joins_multiple_records():
    records = [_record(), {"parameters": {"x": 1.0}}]
    report = FitLog().format_report(records, header="Latest fits", titles=["Run A", "Run B"])
    assert report.startswith("Latest fits")
    assert "=== Run A ===" in report
    assert "=== Run B ===" in report
    assert report.endswith("\n")
