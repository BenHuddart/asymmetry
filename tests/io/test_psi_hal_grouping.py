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
from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.run_preparation import ReplayState
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, prepare_project
from asymmetry.core.project.sources import sample_load_request, verify_project_sources
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


def test_direct_psi_load_keeps_gps_split_readout_layout_key(tmp_path):
    path = tmp_path / "tdc_gps_00888.mdu"
    labels = (
        b"Forw",
        b"Back",
        b"Up_B",
        b"Up_F",
        b"Down_B",
        b"Down_F",
        b"Right_B",
        b"Right_F",
        b"Left_B",
        b"Left_F",
        b"Mob-RL",
    )
    _write_psi_mdu(path, run_number=888, labels=labels)

    dataset = PsiLoader().load(str(path))

    assert dataset.run.instrument_choice.selection.identity == ("psi.gps", "root_split")
    assert dataset.run.grouping["instrument"] == "GPS-RD"
    assert dataset.n_points > 0


@pytest.mark.parametrize("version", [40, CURRENT_SCHEMA_VERSION], ids=["v40", "migrated-v45"])
@pytest.mark.parametrize(
    "saved_facts, conflict",
    [
        ({"histogram_labels": [label.decode() for label in _HAL_FORWARD_LABELS]}, False),
        ({"detector_t0_bins": [1] * 9}, False),
        ({"histogram_labels": [label.decode() for label in _HAL_FORWARD_LABELS[:-1]]}, True),
        ({"detector_t0_bins": [1] * 8}, True),
        ({"histogram_labels": ["MV", "F2", "F1", *(f"F{i}" for i in range(3, 9))]}, True),
        ({"detector_t0_bins": [1] * 9, "detector_last_good_bins": [5] * 8}, True),
    ],
    ids=[
        "exact-labels",
        "nine-clocks",
        "eight-labels",
        "eight-clocks",
        "swapped-labels",
        "mixed-counts",
    ],
)
def test_legacy_replay_checks_saved_acquired_facts(tmp_path, version, saved_facts, conflict):
    path = tmp_path / "tdc_hifi_00686.mdu"
    _write_psi_mdu(path, run_number=686, labels=_HAL_FORWARD_LABELS)
    project_path = tmp_path / "project.asymp"
    state = {
        "schema_version": 40,
        "datasets": [
            {
                "run_number": 686,
                "source_file": path.name,
                "grouping_overrides": {
                    "instrument": "HAL",
                    "groups": {1: [2, 3], 5: [6]},
                    "forward_group": 1,
                    "backward_group": 5,
                    **saved_facts,
                },
            }
        ],
    }
    if version == CURRENT_SCHEMA_VERSION:
        state = prepare_project(state).state
    assert state["schema_version"] == version
    prepared = prepare_project(state)
    recorded = prepared.instrument_choices[686]
    assert recorded.selection.identity == ("psi.hal9500", UnmappedAcquisition("incompatible"))
    assert recorded.evidence == (("grouping_overrides.instrument", "HAL"),)
    facts = verify_project_sources(prepared, project_path).facts
    _, source, replay = sample_load_request(prepared, facts[0])

    run = load(source, replay=replay).run

    assert run.instrument_choice.selection == recorded.selection
    assert run.instrument_choice.evidence[: len(recorded.evidence)] == recorded.evidence
    assert resolve_instrument(run).requires_choice == conflict
