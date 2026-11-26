"""The BiSCCO field comparison keeps the identity of the plotted runs."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from docs.screenshots.scenarios.corpus import cuprate_bscco

pytestmark = pytest.mark.gui


def test_bscco_overlay_exports_run_ids_and_widths(qapp, tmp_path, monkeypatch):
    expected = {
        400: dict(
            zip(
                [*range(1277, 1290), 1276],
                [10, 30, 50, 70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125],
                strict=True,
            )
        ),
        200: dict(
            zip(
                [1292, 1293, 1294, *range(1296, 1302), 1290],
                [30, 50, 70, 85, 90, 95, 100, 105, 110, 120],
                strict=True,
            )
        ),
    }

    def load_run(paths):
        (path,) = paths
        run = int(path.removesuffix(".nxs")[-8:])
        field = 400 if run < 1290 else 200
        return [
            SimpleNamespace(
                run_number=run,
                run_label=str(run),
                temperature=expected[field][run],
                field=field,
                asymmetry=np.array([-23.0]),
            )
        ]

    def fit_run(self, dataset, model, parameters):
        width = dataset.run_number / 1000
        return SimpleNamespace(
            success=True,
            parameters=ParameterSet([Parameter("sigma", width)]),
            uncertainties={"sigma": width / 100},
            chi_squared=100.0,
            reduced_chi_squared=1.0,
        )

    monkeypatch.setattr(cuprate_bscco, "load_corpus_datasets", load_run)
    monkeypatch.setattr(FitEngine, "fit", fit_run)
    output = tmp_path / "linewidths.tsv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(output), ""))

    panel = cuprate_bscco.BsccoFieldCompareScenario().build()
    next(
        action for action in panel._more_menu.actions() if action.text() == "Export TSV…"
    ).trigger()
    with output.open(encoding="utf-8") as stream:
        rows = list(
            csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t")
        )

    assert len(rows) == 24
    assert {
        field: {
            int(row["Run"]): float(row["T (K)"]) for row in rows if float(row["B (G)"]) == field
        }
        for field in expected
    } == expected
    for row in rows:
        assert row["Series"] == f"{float(row['B (G)']):.0f} G — TF scan"
        assert float(row["sigma (µs⁻¹)"]) == pytest.approx(int(row["Run"]) / 1000)
        assert float(row["err_sigma (µs⁻¹)"]) == pytest.approx(int(row["Run"]) / 100000)
