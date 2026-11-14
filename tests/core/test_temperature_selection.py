"""Explicit thermometer selection without changing acquired metadata."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.workflow import scan_axis, select_logged_temperature


def dataset():
    metadata = {
        "temperature": 100.0,
        "sample_temperature_logged": 80.0,
        "sample_temperature_log_source": "default",
        "nexus_time_series": {
            "logs/sample": {
                "time": [-1, 0, 1, 4],
                "values": [999, 10, 20, 60],
                "units": "K",
                "mean": -123,
            },
            "logs/controller": {"time": [0, 1], "values": [40, 80], "units": "K"},
        },
    }
    return MuonDataset(
        np.array([0.0, 1.0]),
        np.ones(2),
        np.ones(2),
        metadata=deepcopy(metadata),
        run=Run(run_number=7, metadata=deepcopy(metadata)),
    )


def test_selection_agrees_across_consumers_and_retains_input():
    source = dataset()
    original = deepcopy(source.metadata)
    selected = select_logged_temperature(source, "logs/sample")
    expected = (10 + 20 + 60) / 3
    assert selected.temperature == expected
    assert selected.sample_temperature_logged == expected
    assert selected.run.temperature == expected
    assert selected.run.sample_temperature_logged == expected
    assert scan_axis({7: selected}, "temperature").values == {7: expected}
    assert scan_axis({7: selected}, "sample_temperature_logged").values == {7: expected}
    assert f"Temperature : {expected} K" in selected.summary()
    assert source.metadata == original
    assert source.run.metadata == original
    assert selected.metadata["nexus_time_series"] == original["nexus_time_series"]
    assert selected.metadata["temperature_selection"]["original"] == {
        key: original[key]
        for key in ("temperature", "sample_temperature_logged", "sample_temperature_log_source")
    }
    second = select_logged_temperature(selected, "logs/controller")
    assert second.temperature == 60
    assert (
        second.metadata["temperature_selection"]["original"]
        == selected.metadata["temperature_selection"]["original"]
    )
    assert selected.temperature == expected


@pytest.mark.parametrize(
    "unit,values,expected",
    [
        ("mK", [1000, 2000], 1.5),
        ("degC", [0, 20], 283.15),
        ("kelvin", [2, 4], 3),
    ],
)
def test_declared_unit_conversion(unit, values, expected):
    source = dataset()
    source.metadata["nexus_time_series"]["logs/sample"].update(
        time=[0, 1], values=values, units=unit
    )
    assert select_logged_temperature(source, "logs/sample").temperature == pytest.approx(expected)


@pytest.mark.parametrize(
    "update,match",
    [
        ({"values": []}, "non-empty"),
        ({"values": [1, 2]}, "same length"),
        ({"values": [1, 2, float("nan"), 4]}, "finite"),
        ({"values": [1, 2, float("inf"), 4]}, "finite"),
        ({"time": [-4, -3, -2, -1]}, "run-active"),
        ({"time": [0, 1, float("nan"), 4]}, "finite"),
        ({"units": ""}, "unit"),
        ({"units": "G"}, "unit"),
        ({"units": "unknown"}, "unit"),
        ({"values": [1, 0, 2, 3]}, "above 0 K"),
    ],
)
def test_invalid_channel_readings_raise(update, match):
    source = dataset()
    source.metadata["nexus_time_series"]["logs/sample"].update(update)
    with pytest.raises(ValueError, match=match):
        select_logged_temperature(source, "logs/sample")


@pytest.mark.parametrize("channel", ["", "sample", "logs/missing"])
def test_exact_path_required(channel):
    source = dataset()
    source.metadata["nexus_time_series"]["other/sample"] = source.metadata["nexus_time_series"][
        "logs/sample"
    ]
    with pytest.raises(ValueError, match="exact channel path"):
        select_logged_temperature(source, channel)


def test_curve_only_and_missing_original():
    source = dataset()
    source.run = None
    del source.metadata["temperature"]
    selected = select_logged_temperature(source, "logs/sample")
    assert selected.run is None
    assert "temperature" not in selected.metadata["temperature_selection"]["original"]


def test_nonfinite_prerun_is_not_consumed():
    source = dataset()
    source.metadata["nexus_time_series"]["logs/sample"]["values"][0] = float("nan")
    assert select_logged_temperature(source, "logs/sample").temperature == 30
