"""ISIS manual anchors are separate from exact layout compatibility snapshots."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instrument import detect_instrument, get_instrument_layout
from asymmetry.core.instruments.isis_hifi import PROFILE as HIFI
from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.core.project.profiles import ProfileFingerprint, profile_fingerprint_for_run
from asymmetry.core.transform.t0 import source_is_pulsed
from asymmetry.core.workflow.grouping import apply_grouping_preset


@pytest.mark.parametrize("profile,stem", [(HIFI, "hifi"), (MUSR, "musr")])
def test_isis_profile_preserves_complete_existing_layout(profile, stem):
    expected = json.loads(
        (Path(__file__).parents[1] / f"data/instruments/{stem}-layout.json").read_text()
    )
    layout = layout_from_profile(profile, "nexus64")
    assert json.loads(json.dumps(asdict(layout))) == expected
    assert asdict(get_instrument_layout(profile.identity.layout_key)) == asdict(layout)


def test_hifi_manual_rf_memberships_and_overlap_are_literal_anchors():
    # HiFi User Manual §7.1, pp.30–31: lists, not angle-derived hemisphere rules.
    layout = layout_from_profile(HIFI, "nexus64")
    longitudinal = layout.presets["Longitudinal"]
    assert longitudinal.groups[longitudinal.forward_group].detector_ids == tuple(range(1, 33))
    assert longitudinal.groups[longitudinal.backward_group].detector_ids == tuple(range(33, 65))
    preset = layout.presets["Transverse (Vector)"]
    assert {key: group.detector_ids for key, group in preset.groups.items()} == {
        1: (5, 6, 7, 8, 9, 10, 11, 12, 13, 52, 53, 54, 55, 56, 57, 58, 59, 60),
        2: (21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 42, 43, 44),
        3: (13, 14, 15, 16, 17, 18, 19, 20, 21, 44, 45, 46, 47, 48, 49, 50, 51, 52),
        4: (1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 35, 36, 60, 61, 62, 63, 64),
    }
    assert set(preset.groups[1].detector_ids) & set(preset.groups[3].detector_ids) == {13, 52}
    assert [(p.label, p.forward_group, p.backward_group, p.alpha) for p in preset.projections] == [
        ("Left-Right", 1, 2, 1.0),
        ("Top-Bottom", 3, 4, 1.0),
    ]


def test_musr_manual_tf_memberships_and_longitudinal_sign_are_literal_anchors():
    # MuSR Manual §2.6.1 p.12 and Fig.9 p.13: F33–64/B1–32 in LF mode.
    layout = layout_from_profile(MUSR, "nexus64")
    preset = layout.presets["Transverse (Vector)"]
    assert {key: group.detector_ids for key, group in preset.groups.items()} == {
        1: (17, 18, 19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54, 55, 56),
        2: (1, 2, 3, 4, 5, 6, 7, 8, 33, 34, 35, 36, 37, 38, 39, 40),
        3: (9, 10, 11, 12, 13, 14, 15, 16, 57, 58, 59, 60, 61, 62, 63, 64),
        4: (25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44, 45, 46, 47, 48),
    }
    longitudinal = layout.presets["Longitudinal"]
    assert longitudinal.groups[longitudinal.forward_group].detector_ids == tuple(range(33, 65))
    assert longitudinal.groups[longitudinal.backward_group].detector_ids == tuple(range(1, 33))
    assert [(p.label, p.forward_group, p.backward_group, p.alpha) for p in preset.projections] == [
        ("Top-Bottom", 1, 2, 1.0),
        ("Fwd-Back", 3, 4, 1.0),
    ]


@pytest.mark.parametrize("profile,forward,backward", [(HIFI, 528, 1552), (MUSR, 1552, 528)])
def test_synthetic_recognition_and_explicit_preset_reduction_keep_existing_settings(
    profile, forward, backward
):
    key = profile.identity.layout_key
    run = Run(
        run_number=42,
        histograms=[
            Histogram(np.full(8, number, dtype=float), 0.1, 0, 0, 7) for number in range(1, 65)
        ],
        metadata={"instrument": key},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "good_frames": 100000,
            "background_correction": False,
        },
    )
    data = MuonDataset(np.array([]), np.array([]), np.array([]), metadata=run.metadata, run=run)
    assert detect_instrument(64, metadata=run.metadata) == key
    assert profile_fingerprint_for_run(run) == ProfileFingerprint(
        profile.identity.key, 64, "nexus64"
    )
    result = apply_grouping_preset(data, "Longitudinal")
    assert result.run.grouping["groups"] == {1: list(range(1, 33)), 2: list(range(33, 65))}
    np.testing.assert_allclose(
        result.asymmetry, 100 * (forward - backward) / (forward + backward), rtol=0, atol=1e-12
    )
    assert result.run.histograms is run.histograms
    assert run.grouping["groups"] == {1: [1], 2: [2]}
    assert profile.source.pulsed is True
    assert source_is_pulsed(run.metadata) is None
    assert not get_instrument_layout(key).apply_default_preset_on_load


@pytest.mark.parametrize("count", [64, 96])
def test_count_alone_remains_unknown_while_existing_name_and_facility_precedence_survive(count):
    assert detect_instrument(count, metadata={}) is None
    assert detect_instrument(count, metadata={"facility": "ISIS"}) is None
    assert detect_instrument(count, source_file="unknown.nxs") is None
    assert detect_instrument(count, metadata={"instrument": "ISIS HiFi"}) == "HiFi"
    assert detect_instrument(count, metadata={"instrument_name": "MUSR"}) == "MuSR"
    assert detect_instrument(count, source_file="hifi00001.nxs") == "HiFi"
    assert detect_instrument(count, source_file="musr00001.nxs") == "MuSR"
    assert detect_instrument(count, metadata={"facility": "PSI", "instrument": "HIFI"}) == "HAL"
    assert detect_instrument(count, metadata={"facility": "PSI", "instrument": "MuSR"}) is None


def test_isis_profile_recognition_and_grouping_are_frontend_free():
    import subprocess
    import sys
    import textwrap

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib.abc
            import sys
            class BlockFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith(("PySide6", "matplotlib", "asymmetry.gui")):
                        raise AssertionError(f"Core imported {fullname}")
            sys.meta_path.insert(0, BlockFrontend())
            from asymmetry.core.instrument import detect_instrument, preset_grouping_for_run
            from asymmetry.core.instruments.isis_hifi import PROFILE as HIFI
            from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
            from asymmetry.core.instruments.resolve import layout_from_profile
            for profile in (HIFI, MUSR):
                assert detect_instrument(64, metadata={"instrument":profile.identity.name}) == profile.identity.layout_key
                layout = layout_from_profile(profile, "nexus64")
                assert len(layout.active_segments) == 64
                assert preset_grouping_for_run(layout, "Longitudinal", n_histograms=64) == {1:list(range(1,33)),2:list(range(33,65))}
            assert not any(name.startswith(("PySide6", "matplotlib", "asymmetry.gui")) for name in sys.modules)
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
