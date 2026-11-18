"""Line readouts retain central separations without inventing unavailable precision."""

import argparse
from types import SimpleNamespace

import pytest

from asymmetry.cli.commands import fit
from asymmetry.core import computation


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize(
    ("sigmas", "suffix"),
    [
        ((0.0, 0.0), ""),
        ((None, None), ""),
        ((None, 0.4), ""),
        ((float("nan"), 0.4), ""),
        ((0.3, 0.4), " ± 0.5"),
        ((0.0, 0.4), " ± 0.4"),
    ],
)
def test_line_readout_preserves_central_spacing_with_available_precision(
    close, sigmas, suffix, monkeypatch, capsys, tmp_path
):
    high = 1.01 if close else 2.5
    result = {
        "run": 1,
        "success": True,
        "computation_revision": computation.COMPUTATION_REVISION,
        "parameters": {"frequency_1": 1.0, "frequency_3": high},
        "uncertainties": {
            name: sigma
            for name, sigma in zip(("frequency_1", "frequency_3"), sigmas)
            if sigma is not None
        },
        "free_params": [],
        "chi_squared": 1.0,
        "reduced_chi_squared": 1.0,
        "quality": None,
        "quality_flags": [],
        "params_at_bound": [],
        "background_held": False,
    }
    recipe = SimpleNamespace(
        expression="two lines", to_dict=lambda: {}, time_window=lambda _: (0.0, 1.0)
    )
    workdir = SimpleNamespace(write_fit=lambda *args: None)
    monkeypatch.setattr(fit, "workdir_for", lambda *args: (workdir, None))
    monkeypatch.setattr(fit, "reduced_datasets", lambda *args: {1: None})
    monkeypatch.setattr(fit, "load_recipe", lambda *args: recipe)
    monkeypatch.setattr(fit, "recipe_with_overrides", lambda *args, **kwargs: recipe)
    monkeypatch.setattr(fit, "window_note", lambda *args: None)
    monkeypatch.setattr("asymmetry.core.workflow.series.fit_one", lambda *args: result)
    fit.run(
        argparse.Namespace(
            plot=False,
            json=False,
            folder=str(tmp_path),
            workdir=None,
            instrument=None,
            run=1,
            recipe="lines",
            fix={},
            free=[],
            tmin=None,
            tmax=None,
        )
    )
    output = capsys.readouterr().out
    assert f"{high - 1:.4g}{suffix} MHz" in output
    assert "± nan" not in output
    if not suffix:
        assert "±" not in output
    if not close:
        assert "Line spacings: 2.5 − 1 = 1.5" in output
