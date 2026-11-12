"""Completion audits consume compatible scientific executions."""

import json
from pathlib import Path

from asymmetry.core import computation
from asymmetry.core.workflow.workdir import SCHEMA
from tests.core.test_cli_numbers import _manifest


def test_a_tesla_field_line_without_a_two_line_fit_is_named(tmp_path: Path) -> None:
    from asymmetry.cli.commands.audit import _untested_doublets

    root = tmp_path / "wd"
    for folder in ("spectra", "fits", "series"):
        (root / folder).mkdir(parents=True)
    spectrum = {
        "computation_revision": computation.COMPUTATION_REVISION,
        "axis": "frequency",
        "run": 686,
        "field_gauss": 60000.0,
        "peak_analysis": {"peaks": [{"frequency_mhz": 813.59}]},
    }
    (root / "spectra" / "run-686.json").write_text(json.dumps(spectrum))

    def fit(**frequencies: float) -> str:
        return json.dumps(
            {
                "computation_revision": computation.COMPUTATION_REVISION,
                "expression": "...",
                "fit": {"success": True, "parameters": frequencies},
            }
        )

    (root / "fits" / "one-686.json").write_text(fit(frequency=813.59))
    assert _untested_doublets([root]) == [(root, 686, 813.59)]
    # A second line at the first's harmonic is not a two-line test.
    (root / "fits" / "wizard-686.json").write_text(fit(frequency_1=813.6, frequency_3=1627.2))
    assert _untested_doublets([root]) == [(root, 686, 813.59)]
    (root / "fits" / "two-line-686.json").write_text(fit(frequency_1=813.60, frequency_3=813.54))
    assert _untested_doublets([root]) == []


def test_all_instrument_survey_uses_each_instruments_fit_coverage(tmp_path):
    from asymmetry.cli.commands.audit import unfitted_scans

    roots = [tmp_path / "overview", tmp_path / "other"]
    scan = {
        "instrument": "MUSR",
        "axis": "temperature",
        "temperature": None,
        "field": 0.0,
        "geometry": "ZF",
        "geometry_note": "",
        "runs": [11, 12, 13, 14],
        "values": [1.0, 2.0, 3.0, 4.0],
        "notes": "",
        "samples": [],
    }
    for root, instrument in zip(roots, ("EMU", "MUSR"), strict=True):
        (root / "fits").mkdir(parents=True)
        (root / "manifest.json").write_text(_manifest("same-data", instrument))
    (roots[0] / "survey.json").write_text(
        json.dumps(
            {
                "computation_revision": computation.COMPUTATION_REVISION,
                "schema": SCHEMA,
                "folder": "same-data",
                "runs": [],
                "scans": [scan],
                "notes_scans": [],
            }
        )
    )
    assert unfitted_scans(roots)[0]
    for run in scan["runs"]:
        (roots[0] / "fits" / f"wrong-{run}.json").write_text(
            json.dumps({"computation_revision": computation.COMPUTATION_REVISION, "run": run})
        )
    assert unfitted_scans(roots)[0]  # Same-number fits in the other instrument do not count.
    for run in scan["runs"]:
        (roots[1] / "fits" / f"right-{run}.json").write_text(
            json.dumps({"computation_revision": computation.COMPUTATION_REVISION, "run": run})
        )
    assert unfitted_scans(roots) == ([], [])
