"""HAL-9500 readout recognition and reductions through the PSI loader."""

import numpy as np
import pytest

from asymmetry.core.instrument import (
    PresetLayoutMismatchError,
    get_instrument_layout,
    preset_grouping_for_run,
    recommend_grouping_preset_for_run,
)
from asymmetry.core.instruments.resolution import UnmappedAcquisition, resolve_instrument
from asymmetry.core.io import load, load_background_run
from asymmetry.core.io.run_preparation import ReplayState
from asymmetry.core.transform.units import PERCENT_PER_FRACTION
from tests.io.test_psi_loader import _write_psi_mdu

pytestmark = pytest.mark.io


_HAL_FORWARD_LABELS = (b"MV", *(f"F{i}".encode() for i in range(1, 9)))
_HAL_FULL_LABELS = (*_HAL_FORWARD_LABELS, *(f"B{i}".encode() for i in range(1, 9)))


def test_hal_forward_ring_loads_transverse_preset_and_reduces_f1_f5(tmp_path) -> None:
    path = tmp_path / "tdc_hifi_00686.mdu"
    _write_psi_mdu(path, run_number=686, labels=_HAL_FORWARD_LABELS)

    dataset = load(path)
    run = dataset.run
    grouping = run.grouping

    assert run.instrument_choice.selection.identity == ("psi.hal9500", "forward_ring")
    assert grouping["grouping_preset"] == "Transverse (opposed pairs)"
    assert grouping["groups"] == {i: [i + 1] for i in range(1, 9)}
    assert grouping["group_names"] == {i: f"F{i}" for i in range(1, 9)}
    assert grouping["included_groups"] == {i: True for i in range(1, 9)}
    assert grouping["forward_group"] == 1
    assert grouping["backward_group"] == 5
    assert grouping["histogram_labels"] == [label.decode() for label in _HAL_FORWARD_LABELS]
    assert dataset.n_points > 0
    assert "correction_error" not in dataset.metadata
    good = slice(grouping["first_good_bin"], grouping["last_good_bin"] + 1)
    f1 = run.histograms[1].counts[good]
    f5 = run.histograms[5].counts[good]
    np.testing.assert_allclose(dataset.asymmetry, PERCENT_PER_FRACTION * (f1 - f5) / (f1 + f5))


@pytest.mark.parametrize("load_mode", ["fresh", "replay", "reference"])
def test_synthetic_hal_full17_keeps_per_octant_preset(tmp_path, load_mode):
    path = tmp_path / "tdc_hifi_00153.mdu"
    _write_psi_mdu(path, run_number=153, labels=_HAL_FULL_LABELS)

    match load_mode:
        case "fresh":
            dataset = load(path)
        case "replay":
            dataset = load(path, replay=ReplayState(run_number=153))
        case "reference":
            dataset = load_background_run({"source_file": str(path), "run_number": 153})

    run = dataset.run
    assert run.instrument_choice.selection.identity == ("psi.hal9500", "full17")
    assert run.grouping["grouping_preset"] == "Per-octant"
    assert run.grouping["groups"] == {i: [i + 1, i + 9] for i in range(1, 9)}
    assert (run.grouping["forward_group"], run.grouping["backward_group"]) == (1, 5)
    assert dataset.n_points > 0


@pytest.mark.parametrize(
    "name, labels",
    [
        ("tdc_hifi_00686.mdu", _HAL_FORWARD_LABELS[:-1]),
        ("tdc_hifi_00686.mdu", (*_HAL_FORWARD_LABELS[:-1], b"B8")),
        ("tdc_hifi_00686.mdu", (*_HAL_FORWARD_LABELS[:-1], b"F7")),
        (
            "tdc_hifi_00686.mdu",
            (_HAL_FORWARD_LABELS[1], _HAL_FORWARD_LABELS[0], *_HAL_FORWARD_LABELS[2:]),
        ),
        ("unknown_00686.mdu", _HAL_FORWARD_LABELS),
        ("tdc_hifi_gps_00686.mdu", _HAL_FORWARD_LABELS),
    ],
    ids=[
        "other-count",
        "other-label",
        "duplicate-label",
        "reordered",
        "unknown-instrument",
        "conflicting-instrument",
    ],
)
def test_hal_forward_ring_preset_requires_matching_labels_and_identity(tmp_path, name, labels):
    path = tmp_path / name
    _write_psi_mdu(path, run_number=686, labels=labels)

    run = load(path).run
    context = resolve_instrument(run)

    assert context.selection is None or isinstance(
        context.selection.acquisition, UnmappedAcquisition
    )
    assert "grouping_preset" not in run.grouping
    assert (
        recommend_grouping_preset_for_run(
            n_histograms=len(run.histograms), metadata=run.metadata, source_file=run.source_file
        )
        is None
    )


def test_hal_full_preset_cannot_supply_the_forward_ring_variant():
    layout = get_instrument_layout("HAL")
    labels = [label.decode() for label in _HAL_FORWARD_LABELS]
    with pytest.raises(PresetLayoutMismatchError, match="HAL-FORWARD"):
        preset_grouping_for_run(layout, "Per-octant", n_histograms=len(labels), labels=labels)
