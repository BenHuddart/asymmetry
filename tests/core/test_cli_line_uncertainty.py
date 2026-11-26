"""Frequency spacing uses the covariance of the two fitted coordinates."""

import json

import numpy as np
import pytest

from asymmetry.cli import main
from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.spectral_lines import fitted_frequency_spacings
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.reduction import ReductionSettings
from asymmetry.core.workflow.workdir import ReducedEntry, WorkDir


@pytest.fixture
def reduced_line_data(tmp_path):
    """Write an independently generated signal for a real CLI fit."""
    time = np.linspace(0.0, 10.0, 501)
    workdir = WorkDir(tmp_path / "work")

    def write(frequencies, damping):
        amplitudes = (9, 4)[: len(frequencies)]
        signal = sum(a * np.cos(2 * np.pi * f * time) for a, f in zip(amplitudes, frequencies))
        noise = np.random.default_rng(316).normal(0, 0.3, size=time.size)
        dataset = MuonDataset(
            time,
            np.exp(-damping * time) * signal + noise,
            np.full(time.size, 0.3),
            computation_revision=computation.COMPUTATION_REVISION,
        )
        workdir.write_reduced(
            dataset,
            ReducedEntry(
                run_number=1,
                digest="synthetic",
                source_file="SIM00000001.nxs",
                n_points=dataset.n_points,
                settings=ReductionSettings(),
                run={"run_number": 1},
                alpha=1.0,
                deadtime_mode="none",
                forward_group=1,
                backward_group=2,
                members=[],
            ),
        )
        return workdir, ["fit", str(tmp_path), "--run", "1", "--workdir", str(workdir.root)]

    return write


@pytest.fixture
def line_fit_command(reduced_line_data):
    """Fit a reduced record with correlated frequencies."""
    workdir, command = reduced_line_data((1.012, 1.0), 0.0)
    recipe = FitRecipe.from_expression("Oscillatory + Oscillatory").with_overrides(
        fix={"A_1": 9.0, "phase_1": 0.0, "A_2": 4.0, "phase_2": 0.0},
        initial={"frequency_1": 1.011, "frequency_2": 1.001},
    )
    workdir.write_recipe("lines", recipe)
    return workdir, [*command, "--recipe", "lines"]


def _mock_fit(monkeypatch, parameters, sigmas, **metadata):
    result = {
        "run": 1,
        "success": True,
        "computation_revision": computation.COMPUTATION_REVISION,
        "parameters": parameters,
        "uncertainties": dict(zip(parameters, sigmas)),
        "free_params": [
            name for name in parameters if name not in metadata.get("fixed_parameters", ())
        ],
        "chi_squared": 1.0,
        "reduced_chi_squared": 1.0,
        "ndof": 1,
        "quality": None,
        "quality_reason": None,
        "quality_flags": [],
        "params_at_bound": [],
        "background_held": False,
        **metadata,
    }
    monkeypatch.setattr("asymmetry.core.workflow.series.fit_one", lambda *args: result)


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize(
    ("correlation", "expected"),
    [(0.9, 0.00447214), (0.0, 0.01414214), (-0.9, 0.01949359)],
)
def test_line_spacings_use_fitted_cross_covariance(
    close, correlation, expected, line_fit_command, monkeypatch, capsys
):
    high = 1.01 if close else 2.5
    variances = 0.01**2
    covariance = {
        "frequency_1": {"frequency_1": variances, "frequency_3": correlation * variances},
        "frequency_3": {"frequency_1": correlation * variances, "frequency_3": variances},
    }
    parameters = {"frequency_1": 1.0, "frequency_3": high}
    (spacing,) = fitted_frequency_spacings(parameters)
    assert spacing.standard_uncertainty_mhz(dict.fromkeys(parameters, 0.01), covariance) == (
        pytest.approx(expected, abs=5e-9)
    )
    _mock_fit(monkeypatch, parameters, (0.01, 0.01), covariance=covariance)
    main(line_fit_command[1])
    assert f"{high - 1:.4g} ± {expected:.2g} MHz" in capsys.readouterr().out


def test_named_signed_frequencies_survive_sorting(line_fit_command, monkeypatch, capsys):
    parameters = {"frequency_1": 3.1, "frequency_3": 1.01, "frequency_5": -1.0}
    matrix = np.array([[9e-4, -5e-5, 2e-5], [-5e-5, 4e-4, 1.5e-4], [2e-5, 1.5e-4, 1e-4]])
    covariance = {a: dict(zip(parameters, row)) for a, row in zip(parameters, matrix)}
    _mock_fit(monkeypatch, parameters, np.sqrt(matrix.diagonal()), covariance=covariance)
    main(line_fit_command[1])
    output = capsys.readouterr().out
    close_jacobian = np.array([0, 1, 1])
    upper_jacobian = np.array([1, -1, 0])
    close = np.sqrt(close_jacobian @ matrix @ close_jacobian)
    upper = np.sqrt(upper_jacobian @ matrix @ upper_jacobian)
    assert (
        f"Line spacings: 1.01 − 1 = 0.01 ± {close:.2g} MHz; 3.1 − 1.01 = 2.09 ± {upper:.2g} MHz"
        in output
    )


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize(
    ("sigmas", "metadata", "expected"),
    [
        ((0.0, 0.4), {}, " MHz (uncertainty unavailable)"),
        ((0.3, 0.4), {"covariance": None}, " MHz (uncertainty unavailable)"),
        (
            (None, 0.4),
            {"covariance": {"frequency_1": {"frequency_3": 0.0}}},
            " MHz (uncertainty unavailable)",
        ),
        (
            (float("nan"), 0.4),
            {"covariance": {"frequency_1": {"frequency_3": 0.0}}},
            " MHz (uncertainty unavailable)",
        ),
        (
            (None, None),
            {
                "covariance": {
                    "frequency_1": {"frequency_1": 0.09},
                    "frequency_3": {"frequency_1": 0.0, "frequency_3": 0.16},
                }
            },
            " ± 0.5 MHz",
        ),
        (
            (0.3, 0.4),
            {
                "covariance": {
                    "frequency_1": {"frequency_1": 0.09},
                    "frequency_3": {"frequency_3": 0.16},
                }
            },
            " MHz (uncertainty unavailable)",
        ),
        ((None, 0.4), {"fixed_parameters": ["frequency_1"]}, " ± 0.4 MHz"),
        ((None, None), {"fixed_parameters": ["frequency_1", "frequency_3"]}, " MHz"),
    ],
)
def test_missing_covariance_and_explicitly_fixed_coordinates(
    close, sigmas, metadata, expected, line_fit_command, monkeypatch, capsys
):
    high = 1.01 if close else 2.5
    _mock_fit(monkeypatch, {"frequency_1": 1.0, "frequency_3": high}, sigmas, **metadata)
    main(line_fit_command[1])
    output = capsys.readouterr().out
    text = f"{high - 1:.4g}{expected}"
    assert f"Line spacings: {high:g} − 1 = {text}\n" in output
    assert "± nan" not in output


def test_frequency_at_zero_requires_an_explicit_fixed_state(line_fit_command, monkeypatch, capsys):
    parameters = {"frequency_1": 0.0, "frequency_3": 1.0}
    sigmas = (0.01, 0.01)
    covariance = {"frequency_1": {"frequency_3": 0.0}}
    _mock_fit(monkeypatch, parameters, sigmas, covariance=covariance)
    main(line_fit_command[1])
    assert "Line spacings: 1 − 0 = 1 MHz (uncertainty unavailable)" in capsys.readouterr().out

    _mock_fit(
        monkeypatch,
        parameters,
        (0.0, 0.01),
        covariance=covariance,
        fixed_parameters=["frequency_1"],
    )
    main(line_fit_command[1])
    assert "Line spacings: 1 − 0 = 1 ± 0.01 MHz" in capsys.readouterr().out


def test_fit_result_covariance_reaches_cli_summary(line_fit_command, capsys):
    workdir, command = line_fit_command
    main(command)
    output = capsys.readouterr().out
    summary = json.loads((workdir.fits_dir / "lines-1.json").read_text())["fit"]
    high, low = "frequency_1", "frequency_2"
    spacing = summary["parameters"][high] - summary["parameters"][low]
    assert spacing == pytest.approx(0.012, abs=0.002)
    covariance = summary["covariance"]
    matrix = np.array([[covariance[a][b] for b in (high, low)] for a in (high, low)])
    jacobian = np.array([1, -1])
    expected = np.sqrt(jacobian @ matrix @ jacobian)
    assert abs(matrix[0, 1]) > 0.4 * np.sqrt(matrix[0, 0] * matrix[1, 1])
    assert summary["fixed_parameters"] == ["A_1", "phase_1", "A_2", "phase_2"]
    assert f"{spacing:.4g} ± {expected:.2g} MHz" in output


@pytest.mark.parametrize(
    ("frequencies", "fixed_frequencies"),
    [
        pytest.param((5.0,), (5.0, 5.04), id="single-line"),
        pytest.param((5.0,), (5.0, 5.0), id="coincident-frequencies"),
        pytest.param((4.6, 5.4), (), id="separated-doublet"),
    ],
)
def test_fit_describes_frequencies_without_preferring_a_line_count(
    frequencies, fixed_frequencies, reduced_line_data, capsys
):
    damping = 0.2
    workdir, command = reduced_line_data(frequencies, damping)
    names = ("frequency_1", "frequency_3")
    component = "Oscillatory * Exponential"
    recipe = FitRecipe.from_expression(f"{component} + {component}").with_overrides(
        fix={
            "phase_1": 0.0,
            "phase_3": 0.0,
            "Lambda_2": damping,
            "Lambda_4": damping,
            **dict(zip(names, fixed_frequencies)),
        },
        initial={"A_1": 8.0, "A_3": 3.0, "frequency_1": 4.62, "frequency_3": 5.38},
    )
    workdir.write_recipe("two", recipe)
    two_line = [*command, "--recipe", "two"]
    main(two_line)
    output = capsys.readouterr().out
    fitted = json.loads((workdir.fits_dir / "two-1.json").read_text())["fit"]
    assert fitted["success"]
    expected_frequencies = fixed_frequencies or frequencies
    assert [fitted["parameters"][name] for name in names] == pytest.approx(
        expected_frequencies, abs=0.003
    )
    assert "Assessing line multiplicity requires a separate model comparison." in output
    assert "compare chi2" not in output and "this fit holds two lines" not in output

    if fixed_frequencies == (5.0, 5.0):
        assert "Line spacings: 5 − 5 = 0 MHz" in output
    elif len(frequencies) == 2:
        assert "Line spacings:" in output
        main([*two_line, "--json"])
        assert json.loads(capsys.readouterr().out)["fit"] == fitted
        assert [fitted["parameters"][name] for name in ("A_1", "A_3")] == pytest.approx(
            (9.0, 4.0), abs=0.1
        )
    else:
        one_recipe = FitRecipe.from_expression(component).with_overrides(
            fix={"frequency": 5.0, "phase": 0.0, "Lambda": damping}
        )
        workdir.write_recipe("one", one_recipe)
        one_line = [*command, "--recipe", "one", "--json"]
        main([*one_line, "--tmax", "6"])
        assert json.loads(capsys.readouterr().out)["fit"]["time_window"] != fitted["time_window"]
        main(two_line)
        assert capsys.readouterr().out == output

        main(one_line)
        comparable = json.loads(capsys.readouterr().out)["fit"]
        assert comparable["time_window"] == fitted["time_window"]
        assert fitted["chi_squared"] < comparable["chi_squared"]
        main(two_line)
        assert capsys.readouterr().out == output
