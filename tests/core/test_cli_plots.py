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

import pytest

from asymmetry import __version__, cli
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
