"""Checks on the Re₆Zr documentation's fitted transverse-field linewidth."""

from math import exp, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.results import FitResult

pytestmark = pytest.mark.gui


def test_re6zr_tf_linewidth_trend_keeps_valid_fit_values_and_errors(qapp, monkeypatch):
    from docs.screenshots.scenarios.corpus import trsb_re6zr

    outcomes = {
        11: (8.0, True, 0.17, 0.02),
        12: (4.0, False, 0.29, 0.03),
        13: (0.3, True, 0.40, 0.01),
        14: (5.0, True, None, 0.04),
        15: (6.0, True, 0.26, None),
        16: (7.0, True, 0.19, np.nan),
        17: (3.0, True, np.nan, 0.06),
        18: (2.0, True, 0.35, 0.0),
        19: (None, True, 0.41, 0.07),
    }
    evaluated_runs = []

    def load_synthetic_run(paths):
        (path,) = paths
        run_number = int(path.rsplit("MUSR000", 1)[1].split(".")[0])
        temp, _, _, _ = outcomes[run_number]
        return [
            SimpleNamespace(
                run_number=run_number,
                run_label=str(run_number),
                field=400.0,
                temperature=temp,
            )
        ]

    def fit_synthetic_run(_, dataset, model, params):
        evaluated_runs.append(dataset.run_number)
        _, success, width, error = outcomes[dataset.run_number]
        assert params.names == ["A_1", "frequency", "phase", "sigma", "A_bg"]

        # Singh et al., PRL 112, 107002, Eq. (1): extract only the sample
        # envelope, evaluated at ten complete cycles and with a nonzero offset.
        time = 2.5
        model_value = model(
            np.array([time]),
            **{"A_1": 8.0, "frequency": 4.0, "phase": 0.0, "sigma": 0.4, "A_bg": 0.5},
        )[0]
        assert (model_value - 0.5) / 8.0 == pytest.approx(exp(-1))
        paper_sigma = sqrt(2) * 0.4
        assert exp(-((paper_sigma * time) ** 2) / 2) == pytest.approx(exp(-1))

        return FitResult(
            success=success,
            objective=Objective.GAUSSIAN,
            parameters=ParameterSet([Parameter("sigma", width)] if width is not None else []),
            uncertainties={"sigma": error} if error is not None else {},
        )

    monkeypatch.setattr(trsb_re6zr, "_TF_RUNS", list(outcomes))
    monkeypatch.setattr(trsb_re6zr, "load_corpus_datasets", load_synthetic_run)
    monkeypatch.setattr(FitEngine, "fit", fit_synthetic_run)
    scenario = trsb_re6zr.TrsbSigmaScTScenario()
    panel = scenario.build()
    scenario.settle(panel)

    assert evaluated_runs == list(outcomes)
    [series] = panel._group_fit_results.values()
    assert series.group_name == "σ_G(T) — Re₆Zr TF 40 mT"
    assert [(row.run_number, row.temperature, row.field) for row in series.rows] == [
        (13, 0.3, 400.0),
        (11, 8.0, 400.0),
    ]
    assert [row.values for row in series.rows] == [{"sigma": 0.4}, {"sigma": 0.17}]
    assert [row.errors for row in series.rows] == [{"sigma": 0.01}, {"sigma": 0.02}]
    for row in series.rows:
        assert row.parameter_sources["sigma"]["component"] == "Gaussian"
        assert row.parameter_sources["sigma"]["widths"] == ["GAUSSIAN_COMPONENT"]

    [axes] = panel._figure.axes
    assert axes.get_title() == "Fitted TF Gaussian linewidth"
    assert axes.get_ylabel() == r"$\sigma_G$ (µs⁻¹)"
    assert panel._effective_x_key() == "temperature"
    assert [label.get_text() for label in axes.get_legend().get_texts()] == ["Per-run fit ± 1σ"]
