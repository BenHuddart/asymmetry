"""Pulse requests remain editable while historical fit values survive project I/O."""

from __future__ import annotations

import pytest

from asymmetry.core.fourier.spectrum import GroupSpectrumConfig
from asymmetry.core.maxent import MaxEntConfig
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from tests.project.single_fits import open_fit
from tests.project.test_saved_single_fits_migration import _v24_project


def test_historical_free_separation_and_invalid_recipe_survive_project_open(tmp_path):
    project = _v24_project()
    reps = project["datasets"][0]["representations"]
    recorded = {"name": "dpsep", "value": 0.324, "fixed": False, "min": 0.1, "max": 0.5}
    reps["time_fb_asymmetry"]["fit"]["parameters"] = [recorded]
    history = {
        "success": True,
        "values": {"dpsep": 0.324},
        "errors": {"dpsep": 0.012},
        "time": [0.0, 0.1],
        "curve": [100.0, 99.0],
    }
    reps["time_fb_asymmetry"]["fit"]["result"] = history
    reps["freq_fft"]["recipe"] = {
        "fourier_config": {"pulse_compensation": True, "pulse_half_width_us": 0.0}
    }
    path = tmp_path / "history.asymp"
    save_project(project, path)
    loaded = load_project(path)
    model = ProjectModel.from_project_state(loaded)
    slot = model.representation(10, RepresentationType.TIME_FB_ASYMMETRY).fit_set(None).open_fit()
    assert {key: slot.result[key] for key in history} == history
    assert slot.parameters == [recorded]
    saved_slot = open_fit(loaded["datasets"][0]["representations"]["time_fb_asymmetry"])
    assert {key: saved_slot["result"][key] for key in history} == history
    assert saved_slot["parameters"] == [recorded]
    config = GroupSpectrumConfig.from_dict(
        model.representation(10, RepresentationType.FREQ_FFT).fourier_config()
    )
    assert config.pulse_compensation and config.pulse_half_width_us == 0.0


@pytest.mark.parametrize("mode", ["ignore", "single", "double"])
def test_pulse_config_json_keeps_authored_request(mode):
    import json

    maxent = MaxEntConfig(pulse_mode=mode, pulse_half_width_us=0.08, pulse_separation_us=0.324)
    fft = GroupSpectrumConfig(
        pulse_compensation=mode != "ignore",
        pulse_half_width_us=0.08,
        pulse_n_pulses={"ignore": 0, "single": 1, "double": 2}[mode],
    )
    assert (
        MaxEntConfig.from_dict(json.loads(json.dumps(maxent.to_dict()))).to_dict()
        == maxent.to_dict()
    )
    assert (
        GroupSpectrumConfig.from_dict(json.loads(json.dumps(fft.to_dict()))).to_dict()
        == fft.to_dict()
    )


def test_historical_power_fit_survives_current_recomputation(tmp_path, monkeypatch):
    import copy

    import numpy as np

    import asymmetry.core.fourier.grouping_identity as identity
    from tests.core.test_fourier_spectrum import _run

    run = _run()
    current_digest = identity.fourier_grouping_digest(run)
    with monkeypatch.context() as patch:
        patch.setattr(identity, "CONDITIONING_REVISION", "amplitude_gain_all_displays")
        recorded_digest = identity.fourier_grouping_digest(run)
    assert recorded_digest != current_digest
    project = _v24_project()
    rep = project["datasets"][0]["representations"]["freq_fft"]
    config = GroupSpectrumConfig(
        display="Power", pulse_compensation=True, pulse_half_width_us=0.05, selected_group_ids=[1]
    )
    rep["recipe"] = {"fourier_config": config.to_dict(), "grouping_digest": recorded_digest}
    history = {
        "success": True,
        "values": {"A": 1.2},
        "errors": {"A": 0.3},
        "time": [0.0, 0.5],
        "curve": [1.0, 2.0],
    }
    rep["fit"]["result"] = history
    path = tmp_path / "power.asymp"
    save_project(project, path)
    loaded = load_project(path)
    model = ProjectModel.from_project_state(loaded)
    restored = model.representation(10, RepresentationType.FREQ_FFT)
    before = copy.deepcopy(restored.fit_set(None).open_fit().result)
    recomputed = restored.compute(run)[0]
    magnitude = restored.__class__(
        recipe={"fourier_config": {**config.to_dict(), "display": "Magnitude"}}
    ).compute(run)[0]
    np.testing.assert_allclose(recomputed.asymmetry, magnitude.asymmetry**2)
    assert restored.recipe["grouping_digest"] == recorded_digest
    assert restored.fit_set(None).open_fit().result == before
    assert {key: before[key] for key in history} == history
