"""Tests for ``--plot`` on ``reduce``/``wizard``/``fit``/``fit-series``/``trend``.

The fit wizard is the one expensive step here (seconds of fitting), so this
file pays for exactly one wizard build — reusing the session-scoped
``reduced_workdir`` fixture (already-reduced spectra, shared with
``tests/core/test_workflow_screen.py`` and friends) rather than reducing
again, and screening only once, in
:func:`test_wizard_plot_writes_data_and_the_recommended_curve`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asymmetry import __version__, cli
from asymmetry.cli import plots
from asymmetry.cli._output import SCHEMA
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.workdir import WorkDir
from tests.core.conftest import SCAN_RUNS

#: A PNG this small would be a blank or corrupt figure, not a real plot.
_MIN_PNG_BYTES = 1024


def _json_output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _assert_real_png(path: Path) -> None:
    assert path.is_file(), f"{path} was not written"
    assert path.stat().st_size > _MIN_PNG_BYTES, f"{path} is too small to be a real plot"


def _noisy_tail_record(
    n_points: int = 2000, *, break_time: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ``(time, error)`` whose error explodes past *break_time*.

    Flat (informative) below the break, exponentially blown up above it — the
    shape ``frame_for_record`` exists to truncate, and large enough
    (``n_points``) to also exercise the display-bunching path.
    """
    time = np.linspace(0.0, 32.0, n_points)
    error = np.where(time < break_time, 1.0, np.exp((time - break_time) / 1.5))
    return time, np.clip(error, None, 100.0)


# -- framing / bunching (asymmetry.cli.plots internals) ----------------------


def test_frame_for_record_caps_the_window_at_the_effective_end() -> None:
    time, error = _noisy_tail_record()

    window_min, window_max = plots.frame_for_record(time, error, t_min=None, t_max=None)

    assert window_min == time[0]
    # Truncated well before the requested (full) end, and inside the flat
    # (informative) region rather than out in the exploded tail.
    assert window_max < time[-1]
    assert window_max < 20.0
    assert window_max > 10.0


def test_frame_for_record_never_exceeds_an_explicit_t_max() -> None:
    time, error = _noisy_tail_record()

    _, window_max = plots.frame_for_record(time, error, t_min=None, t_max=5.0)

    # The record is not yet noisy at 5 µs, so the caller's own window wins.
    assert window_max == pytest.approx(5.0)


def test_frame_for_record_does_not_truncate_a_flat_error_record() -> None:
    time = np.linspace(0.0, 10.0, 50)
    error = np.full_like(time, 1.0)

    window_min, window_max = plots.frame_for_record(time, error, t_min=None, t_max=None)

    assert window_min == time[0]
    assert window_max == time[-1]


def test_bunch_factor_is_one_for_a_short_record_and_greater_for_a_long_one() -> None:
    assert plots._bunch_factor(100) == 1
    assert plots._bunch_factor(400) == 1
    assert plots._bunch_factor(401) == 2
    assert plots._bunch_factor(2000) == 5
    # And it is genuinely the smallest such factor.
    factor = plots._bunch_factor(1601)
    assert 1601 / factor <= 400
    assert 1601 / (factor - 1) > 400


def test_trend_frame_ignores_a_wildly_flagged_point_and_reports_it_outside() -> None:
    y = np.array([1.0, 1.1, 0.9, 1.05, 18.0])
    y_err = np.array([0.1, 0.1, 0.1, 0.1, 5.0])
    flagged = np.array([False, False, False, False, True])

    y_lo, y_hi, outside = plots._trend_frame(y, y_err, flagged)

    # Framed on the four clean points (~0.9-1.1), not stretched to the outlier.
    assert y_lo > 0.5
    assert y_hi < 1.5
    assert list(outside) == [False, False, False, False, True]


def test_trend_frame_covers_every_point_when_all_are_flagged() -> None:
    y = np.array([1.0, 5.0, 9.0])
    y_err = np.array([0.1, 0.1, 0.1])
    flagged = np.array([True, True, True])

    y_lo, y_hi, outside = plots._trend_frame(y, y_err, flagged)

    assert y_lo <= y.min()
    assert y_hi >= y.max()
    assert not outside.any()


def test_trend_frame_ignores_missing_errors_and_frames_on_the_bare_values() -> None:
    """Every row flagged and every error ``None`` — the wholly-failed series."""
    y = np.array([1.0, 2.0, 3.0])
    y_err = np.array([np.nan, np.nan, np.nan])
    flagged = np.array([True, True, True])

    y_lo, y_hi, outside = plots._trend_frame(y, y_err, flagged)

    assert y_lo == pytest.approx(1.0 - 0.2)
    assert y_hi == pytest.approx(3.0 + 0.2)
    assert not outside.any()


def test_trend_frame_has_no_range_when_no_value_is_finite() -> None:
    y = np.array([np.nan, np.nan])
    y_err = np.array([np.nan, np.nan])
    flagged = np.array([False, False])

    y_lo, y_hi, outside = plots._trend_frame(y, y_err, flagged)

    assert y_lo is None
    assert y_hi is None
    assert not outside.any()


def _trend_rows(values, errors):
    return [
        {"x": float(index), "run": 100 + index, "lam": value, "lam_err": error, "flags": ["bad"]}
        for index, (value, error) in enumerate(zip(values, errors))
    ]


def test_trend_plot_frames_a_series_whose_errors_are_all_missing(tmp_path: Path) -> None:
    """A wholly flagged/failed series used to hand ``set_ylim`` NaN limits."""
    out_path = plots.plot_trend(
        _trend_rows([1.0, 2.0, 3.0], [None, None, None]),
        param_name="lam",
        order_key="temperature",
        out_path=tmp_path / "all-errors-missing.png",
    )
    _assert_real_png(out_path)


def test_trend_plot_still_writes_a_png_when_no_value_is_finite(tmp_path: Path) -> None:
    out_path = plots.plot_trend(
        _trend_rows([None, None, None], [None, None, None]),
        param_name="lam",
        order_key="temperature",
        out_path=tmp_path / "no-values.png",
    )
    _assert_real_png(out_path)


# -- framing/bunching, end to end (plot_fit on a synthetic noisy-tail record) -


def test_fit_plot_on_a_record_with_an_exploding_tail_is_framed_and_bunched(
    tmp_path: Path,
) -> None:
    time, error = _noisy_tail_record()
    rng = np.random.default_rng(0)
    asymmetry = 20.0 * np.exp(-0.2 * time) + rng.normal(scale=0.2, size=time.size)

    out_path = plots.plot_fit(
        time,
        asymmetry,
        error,
        model_function=lambda t, amplitude, rate: amplitude * np.exp(-rate * t),
        parameters={"amplitude": 20.0, "rate": 0.2},
        t_min=None,
        t_max=None,
        run_number=1,
        expression="Exponential",
        out_path=tmp_path / "fit.png",
    )
    _assert_real_png(out_path)

    # The same framing/bunching this call used, so the test does not just
    # trust the drawing code — it checks the numbers behind it too.
    window_min, window_max = plots.frame_for_record(time, error, t_min=None, t_max=None)
    assert window_max < time[-1]
    windowed_points = int(np.count_nonzero((time >= window_min) & (time <= window_max)))
    assert plots._bunch_factor(windowed_points) > 1


def test_integral_scan_plot_writes_data_and_fitted_curve(tmp_path: Path) -> None:
    x = np.linspace(1000.0, 5000.0, 21)
    parameters = {"amplitude": -0.04, "centre": 3300.0, "width": 180.0, "offset": 0.2}

    def model(field, amplitude, centre, width, offset):
        return offset + amplitude / (1.0 + ((field - centre) / width) ** 2)

    value = model(x, **parameters)
    out_path = plots.plot_scan(
        x,
        value,
        np.full_like(x, 0.002),
        expression="test resonance",
        model_function=model,
        parameters=parameters,
        x_label="field / G",
        y_label="integral asymmetry",
        title="ALC scan",
        out_path=tmp_path / "scan.png",
    )

    _assert_real_png(out_path)


def test_fourier_plot_writes_real_and_magnitude_channels(tmp_path: Path) -> None:
    frequency = np.linspace(0.0, 5.0, 101)
    real = np.cos(frequency) * np.exp(-frequency)
    magnitude = np.abs(real)

    out_path = plots.plot_spectrum(
        frequency,
        real,
        magnitude,
        run_number=20721,
        coupling=False,
        out_path=tmp_path / "fourier.png",
    )

    _assert_real_png(out_path)


@pytest.fixture
def fitting_workdir(workflow_folder: Path, tmp_path: Path) -> Path:
    """A work directory with the scan reduced and one expression-built recipe stored.

    Built from an expression rather than the wizard — these tests are about
    ``--plot`` on ``fit``/``fit-series``/``trend``, not about screening, and a
    screening run would cost seconds of fitting to produce a recipe they would
    not otherwise care about (mirrors ``tests/core/test_cli_commands.py``).
    """
    workdir = tmp_path / "wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--workdir",
            str(workdir),
        ]
    )
    stored = WorkDir(workdir)
    stored.write_recipe(
        "relax",
        FitRecipe.from_expression("Exponential + Constant", dataset=stored.reduced(SCAN_RUNS[0])),
    )
    return workdir


# -- reduce -----------------------------------------------------------------


def test_reduce_plot_writes_a_png_per_run_and_lists_them(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    runs = SCAN_RUNS[:2]
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            f"{runs[0]}-{runs[-1]}",
            "--plot",
            "--json",
            "--workdir",
            str(workdir),
        ]
    )
    payload = _json_output(capsys)
    assert payload["schema"] == SCHEMA
    assert len(payload["plots"]) == len(runs)

    for run_number, plot_path in zip(runs, payload["plots"]):
        path = Path(plot_path)
        assert path == workdir / "plots" / f"reduced-{run_number}.png"
        _assert_real_png(path)


def test_reduce_without_plot_writes_no_plots_key_pngs(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--json",
            "--workdir",
            str(workdir),
        ]
    )
    payload = _json_output(capsys)
    assert payload["plots"] == []
    assert not (workdir / "plots").exists() or not list((workdir / "plots").glob("*.png"))


# -- wizard -------------------------------------------------------------


def test_wizard_plot_writes_data_and_the_recommended_curve(
    workflow_folder: Path, reduced_workdir: WorkDir, capsys
) -> None:
    run = SCAN_RUNS[0]
    cli.main(
        [
            "wizard",
            str(workflow_folder),
            "--run",
            str(run),
            "--plot",
            "--json",
            "--workdir",
            str(reduced_workdir.root),
        ]
    )
    payload = _json_output(capsys)
    assert payload["plot_note"] is None
    assert len(payload["plots"]) == 1

    path = Path(payload["plots"][0])
    assert path == reduced_workdir.plots_dir / f"wizard-{run}.png"
    _assert_real_png(path)


# -- fit ------------------------------------------------------------------


def test_fit_plot_writes_a_fit_png_and_lists_it(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--recipe",
            "relax",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    payload = _json_output(capsys)
    assert len(payload["plots"]) == 1
    path = Path(payload["plots"][0])
    assert path == fitting_workdir / "plots" / f"fit-{SCAN_RUNS[0]}.png"
    _assert_real_png(path)


# -- fit-series -------------------------------------------------------------


def test_fit_series_plot_writes_per_run_and_trend_pngs_and_stores_the_recipe(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            "scan",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    payload = _json_output(capsys)
    free_params = payload["series"]["free_params"]
    assert free_params  # the expression has at least one free parameter

    per_run = [Path(fitting_workdir, "plots", "scan", f"{run}.png") for run in SCAN_RUNS]
    trend_pngs = [Path(fitting_workdir, "plots", f"scan-trend-{name}.png") for name in free_params]
    for path in [*per_run, *trend_pngs]:
        _assert_real_png(path)
        assert str(path) in payload["plots"]

    assert len(payload["plots"]) == len(per_run) + len(trend_pngs)

    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert stored["recipe"]["expression"] == "Exponential + Constant"
    # The stored recipe round-trips through FitRecipe, exactly as `trend --plot` reads it.
    recipe = FitRecipe.from_dict(stored["recipe"])
    assert recipe.expression == payload["series"]["expression"]


def test_fit_series_without_plot_still_stores_the_recipe(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--recipe",
            "relax",
            "--order",
            "run",
            "--name",
            "noplot",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    stored = json.loads((fitting_workdir / "series" / "noplot.json").read_text(encoding="utf-8"))
    assert "recipe" in stored
    assert not (fitting_workdir / "plots" / "noplot").exists()


# -- trend ------------------------------------------------------------------


def test_trend_plot_writes_one_png_per_free_parameter(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            "scan",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    capsys.readouterr()

    cli.main(
        [
            "trend",
            str(workflow_folder),
            "--series",
            "scan",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    payload = _json_output(capsys)
    assert payload["asymmetry_version"] == __version__

    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    free_params = stored["free_params"]
    assert free_params

    trend_pngs = [Path(fitting_workdir, "plots", f"scan-trend-{name}.png") for name in free_params]
    assert len(payload["plots"]) == len(trend_pngs) == len(free_params)
    for path in trend_pngs:
        _assert_real_png(path)
        assert str(path) in payload["plots"]


# -- stored products name their own artefacts -------------------------------


def test_a_stored_scan_spectrum_and_global_fit_name_their_own_path_and_plots(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    """What the CLI reports is also what the stored product says about itself.

    The skill sends an agent back to ``scans/``, ``spectra/`` and ``series/``
    for provenance, so a product whose plot and own path live only in the
    command's stdout is a dead end: the JSON on disk has to name them too.
    """
    cli.main(
        [
            "integral-scan",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[2]}",
            "--order",
            "run",
            "--name",
            "alc",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    reported = _json_output(capsys)
    stored = json.loads((fitting_workdir / "scans" / "alc.json").read_text(encoding="utf-8"))
    assert (
        stored["scan_path"] == reported["scan_path"] == str(fitting_workdir / "scans" / "alc.json")
    )
    assert stored["plot"] == reported["plot"]
    _assert_real_png(Path(stored["plot"]))

    cli.main(
        [
            "fourier",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--name",
            "spectrum",
            "--fmax",
            "10",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    reported = _json_output(capsys)
    stored = json.loads(Path(reported["metadata_path"]).read_text(encoding="utf-8"))
    assert stored["plot"] == reported["plot"]
    _assert_real_png(Path(stored["plot"]))

    cli.main(
        [
            "fit-global",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--recipe",
            "relax",
            "--shared",
            "A_bg",
            "--strategy",
            "least_squares",
            "--name",
            "joint",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    reported = _json_output(capsys)["global_fit"]
    stored = json.loads((fitting_workdir / "series" / "joint.json").read_text(encoding="utf-8"))
    assert stored["series_path"] == reported["series_path"]
    assert stored["plots"] == reported["plots"] and len(stored["plots"]) == 2
    for path in stored["plots"]:
        _assert_real_png(Path(path))
