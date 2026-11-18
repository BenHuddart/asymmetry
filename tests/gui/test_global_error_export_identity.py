"""Generated parameter tables retain each error's recorded execution identity."""

from pathlib import Path

import pytest

from asymmetry.core import computation

pytestmark = pytest.mark.gui


def test_tsv_and_gle_keep_mixed_current_historical_and_unknown_revisions(
    qapp, tmp_path, monkeypatch
):
    pytest.importorskip("PySide6")
    from asymmetry.gui.panels.fit_parameters_panel import (
        FitParametersPanel,
        _FitRow,
        _GroupFitData,
        _PlotSeries,
    )

    panel = FitParametersPanel()
    rows = [
        _FitRow(
            run_number=run,
            run_label=str(run),
            field=run,
            temperature=5.0,
            values={"Lambda": 0.1 * run},
            errors={"Lambda": 0.01 * run},
            computation_revision=revision,
        )
        for run, revision in enumerate((computation.COMPUTATION_REVISION, "2", None), 1)
    ]
    other = _FitRow(
        run_number=1,
        run_label="1",
        field=1.0,
        temperature=5.0,
        values={"Lambda": 0.5},
        errors={"Lambda": 0.02},
        computation_revision="1",
    )
    panel._rows = rows
    panel._varying_params = ["Lambda"]
    panel._inferred_x_key = "field"
    panel._rebuild_y_controls()
    panel._active_group_id = "current"
    panel._group_fit_results = {
        "current": _GroupFitData(
            group_id="current",
            group_name="mixed",
            rows=rows,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    monkeypatch.setattr(
        panel,
        "_series_to_plot",
        lambda: [
            _PlotSeries("current", "mixed", "C0", rows, True),
            _PlotSeries("old", "historic", "C1", [other], False),
        ],
    )
    tsv = tmp_path / "parameters.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tsv), "TSV files (*.tsv)"),
    )
    panel._export_tsv()
    text = tsv.read_text()
    for run, revision in enumerate((computation.COMPUTATION_REVISION, "2", "unknown"), 1):
        assert f"# Computation revision ['mixed', run {run}]: {revision}" in text
    assert "# Computation revision ['historic', run 1]: 1" in text
    data = [line.split("\t") for line in text.splitlines() if not line.startswith("#")]
    assert data[0][0] == "Series"
    assert len(data) == 5
    assert [float(line[5]) for line in data[1:]] == pytest.approx([0.01, 0.02, 0.03, 0.02])

    gle = tmp_path / "parameters.dat"
    panel._write_gle_data_file(Path(gle))
    text = gle.read_text()
    for run, revision in enumerate((computation.COMPUTATION_REVISION, "2", "unknown"), 1):
        assert f"! Computation revision ['mixed', run {run}]: {revision}" in text
    assert "historic" not in text
    numeric = [line.split() for line in text.splitlines() if line and not line.startswith("!")]
    assert len(numeric) == 3
    assert [float(line[4]) for line in numeric] == pytest.approx([0.01, 0.02, 0.03])
    assert [row.computation_revision for row in rows] == [
        computation.COMPUTATION_REVISION,
        "2",
        None,
    ]
    panel.close()
