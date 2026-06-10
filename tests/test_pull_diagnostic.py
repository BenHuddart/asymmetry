"""Pull-distribution diagnostic core — reproduces the suite's pull check.

The headline verification: on a known case the pulls of every free parameter
centre on 0 with width 1 (exact Poisson error propagation in
compute_asymmetry, PR #35 — no (1+A²)/(1−A²) correction needed). See also
tests/test_nexus_writer.py::TestRefitRecovery::test_pull_distribution_over_seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.pull_diagnostic import ParameterPull, run_pull_distribution

N_BINS = 1500
BIN_WIDTH = 0.016
T0_BIN = 40


def _template() -> Run:
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=T0_BIN,
            good_bin_start=T0_BIN + 5,
            good_bin_end=N_BINS - 10,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN + 5,
        "last_good_bin": N_BINS - 10,
        "good_frames": 18000.0,
    }
    return Run(run_number=4321, histograms=histograms, grouping=grouping)


def _exp_model(t: np.ndarray, a0: float = 20.0, rate: float = 0.5) -> np.ndarray:
    return a0 * np.exp(-rate * t)


def _make_refit(a0_start: float, rate_start: float, t_max: float = 8.0):
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    def refit(dataset):
        params = ParameterSet(
            [
                Parameter(name="a0", value=a0_start, min=0.0, max=100.0),
                Parameter(name="rate", value=rate_start, min=0.0, max=10.0),
            ]
        )
        result = FitEngine().fit(dataset, _exp_model, params, t_max=t_max)
        if not result.success:
            return None
        values = {p.name: p.value for p in result.parameters}
        return values, dict(result.uncertainties)

    return refit


class TestParameterPull:
    def test_statistics_of_a_known_normal_sample(self) -> None:
        rng = np.random.default_rng(0)
        pull = ParameterPull("x", truth=0.0, pulls=rng.normal(0.0, 1.0, 5000))
        assert abs(pull.mean) < 4.0 * pull.mean_uncertainty
        assert abs(pull.width - 1.0) < 4.0 * pull.width_uncertainty
        assert "well-calibrated" in pull.verdict()

    def test_inflated_errors_read_as_over_estimated(self) -> None:
        # Pulls drawn with width 0.5 → reported errors twice too large.
        rng = np.random.default_rng(1)
        pull = ParameterPull("x", truth=0.0, pulls=rng.normal(0.0, 0.5, 4000))
        assert "OVER-estimated" in pull.verdict()


class TestRunPullDistribution:
    def test_pulls_centre_on_standard_normal(self) -> None:
        pytest.importorskip("iminuit")
        truth = {"a0": 21.0, "rate": 0.6}
        n_seeds = 100
        result = run_pull_distribution(
            _template(),
            _exp_model,
            truth,
            _make_refit(a0_start=15.0, rate_start=1.0),
            total_events=2.0e6,
            n_seeds=n_seeds,
            time_range=(None, 8.0),
        )
        assert result.n_converged >= 0.9 * n_seeds
        for name in truth:
            pull = result.parameters[name]
            assert abs(pull.mean) < 4.0 / np.sqrt(pull.n), name
            sigma_var = np.sqrt(2.0 / (pull.n - 1))
            assert abs(pull.width**2 - 1.0) < 4.0 * sigma_var, name

    def test_progress_callback_reports_each_seed(self) -> None:
        pytest.importorskip("iminuit")
        seen: list[tuple[int, int]] = []
        run_pull_distribution(
            _template(),
            _exp_model,
            {"a0": 21.0, "rate": 0.6},
            _make_refit(15.0, 1.0),
            total_events=1.0e6,
            n_seeds=5,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    def test_non_convergence_is_counted_not_fatal(self) -> None:
        result = run_pull_distribution(
            _template(),
            _exp_model,
            {"a0": 21.0, "rate": 0.6},
            lambda dataset: None,  # always "fails"
            total_events=1.0e6,
            n_seeds=4,
        )
        assert result.n_converged == 0
        assert result.parameters["a0"].n == 0
        assert "too few" in result.verdict()
