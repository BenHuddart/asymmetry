"""The EuO walkthrough fits the same observations its trend exports."""

from pathlib import Path

import numpy as np
import pytest

from docs.screenshots.scenarios.corpus.euo_ordering import EuoNuTTrendScenario
from examples.euo_order_parameter import EUO_RUNS, fit_euo_export, order_parameter

pytestmark = pytest.mark.gui


def test_displayed_euo_trend_matches_fit_parameters_export(qapp, tmp_path, monkeypatch):
    temperature = np.linspace(1.6, 68.69, len(EUO_RUNS))
    frequency = order_parameter(temperature, 30.0, 69.5, 1.6, 0.4)
    # Small supplied errors must retain their weight even when they differ
    # substantially from the rest of the column.
    errors = np.linspace(0.007, 0.16, len(EUO_RUNS))
    frequency += np.linspace(0.02, -0.03, len(EUO_RUNS))
    rows = [
        {
            "run_number": run,
            "temperature": t,
            "values": {"frequency": value},
            "errors": {"frequency": error},
            "model_name": "Oscillatory * Exponential + Constant",
        }
        for run, t, value, error in zip(EUO_RUNS, temperature, frequency, errors, strict=True)
    ]
    panel = EuoNuTTrendScenario.build_trend_panel(rows)
    out = tmp_path / "eu.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_args: (str(out), "TSV files (*.tsv)"),
    )
    panel._export_tsv()

    exported_rows, reproduced, chi_squared = fit_euo_export(out)
    fit_range = panel._model_fits["frequency"].ranges[0]
    result = fit_range.result
    assert [run for run, *_ in exported_rows] == list(EUO_RUNS)
    exported_temperature, exported_frequency, exported_error = np.array(exported_rows)[:, 1:].T
    np.testing.assert_allclose(
        [exported_temperature, exported_frequency, exported_error],
        [temperature, frequency, errors],
        rtol=5e-6,
    )
    assert (fit_range.x_min, fit_range.x_max) == pytest.approx((temperature[0], temperature[-1]))
    assert result.n_points == len(EUO_RUNS)
    assert result.dof == len(EUO_RUNS) - 4
    expected = [fit_range.parameters[name].value for name in ("y0", "Tc", "alpha", "beta")]
    assert reproduced == pytest.approx(expected, abs=0.001)
    residuals = exported_frequency - order_parameter(exported_temperature, *reproduced)
    assert chi_squared == pytest.approx(np.sum((residuals / exported_error) ** 2))
    assert chi_squared == pytest.approx(result.chi_squared, abs=0.0001)
    panel.close()


def test_euo_example_rejects_missing_frequency_error(tmp_path: Path):
    out = tmp_path / "eu.tsv"
    out.write_text(
        "Run\tT (K)\tfrequency (MHz)\terr_frequency (MHz)\n2960\t1.6\t30.1\t\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Run 2960 is missing a temperature, frequency or frequency error"
    ):
        fit_euo_export(out)
