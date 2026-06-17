"""Tests for MusrRoot/LEM ROOT loading."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

from asymmetry.core.io import RootLoader, load

uproot = pytest.importorskip("uproot")

#: musrfit ships example ROOT files in its own source tree, which Asymmetry does
#: not vendor. Point ``ASYMMETRY_MUSRFIT_DATA`` at ``<musrfit>/doc/examples/data``
#: to run the reader-parity tests below; without it (e.g. in CI) they skip.
_MUSRFIT_DATA_DIR = os.environ.get("ASYMMETRY_MUSRFIT_DATA")


def _musrfit_example_file(name: str) -> Path:
    """Resolve a musrfit example data file, skipping if the corpus is absent."""
    if not _MUSRFIT_DATA_DIR:
        pytest.skip("ASYMMETRY_MUSRFIT_DATA not set; musrfit reader-parity corpus unavailable")
    path = Path(_MUSRFIT_DATA_DIR) / name
    if not path.exists():
        pytest.skip(f"musrfit example file not available: {name}")
    return path


def _write_musrroot_directory(path: Path) -> None:
    edges = np.arange(-0.5, 7.5, 1.0)
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "2468"
        root_file["RunHeader/RunInfo/Run Title"] = "Synthetic MusrRoot"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Instrument"] = "LEM"
        root_file["RunHeader/RunInfo/Muon Source"] = "low energy muon source"
        root_file["RunHeader/RunInfo/Sample Name"] = "Ag"
        root_file["RunHeader/RunInfo/Sample Temperature"] = "12.5 +- 0.1 K"
        root_file["RunHeader/RunInfo/Sample Magnetic Field"] = "0.01 T"
        root_file["RunHeader/RunInfo/Run Start Time"] = "2026-01-01 10:00:00"
        root_file["RunHeader/RunInfo/Run Stop Time"] = "2026-01-01 11:00:00"
        root_file["RunHeader/RunInfo/No of Histos"] = "2"
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        root_file["RunHeader/DetectorInfo/Detector001/Name"] = "Left"
        root_file["RunHeader/DetectorInfo/Detector001/Histo Number"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/Time Zero Bin"] = "2"
        root_file["RunHeader/DetectorInfo/Detector001/First Good Bin"] = "3"
        root_file["RunHeader/DetectorInfo/Detector001/Last Good Bin"] = "6"
        root_file["RunHeader/DetectorInfo/Detector002/Name"] = "Right"
        root_file["RunHeader/DetectorInfo/Detector002/Histo Number"] = "2"
        root_file["RunHeader/DetectorInfo/Detector002/Time Zero Bin"] = "3"
        root_file["RunHeader/DetectorInfo/Detector002/First Good Bin"] = "4"
        root_file["RunHeader/DetectorInfo/Detector002/Last Good Bin"] = "6"
        root_file["histos/DecayAnaModule/hDecay001"] = (
            np.array([0, 10, 20, 30, 40, 50, 60], dtype=np.float64),
            edges,
        )
        root_file["histos/DecayAnaModule/hDecay002"] = (
            np.array([0, 0, 15, 25, 35, 45, 55], dtype=np.float64),
            edges,
        )
        root_file["histos/SCAnaModule/hSampleTemperature"] = (
            np.array([11.8, 12.2, 12.4], dtype=np.float64),
            np.array([0.0, 20.0, 40.0, 60.0], dtype=np.float64),
        )


def _write_red_green_offset_directory(path: Path) -> None:
    edges = np.arange(-0.5, 4.5, 1.0)
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "1357"
        root_file["RunHeader/RunInfo/Instrument"] = "LEM"
        root_file["RunHeader/RunInfo/No of Histos"] = "1"
        root_file["RunHeader/RunInfo/RedGreen Offsets"] = "20"
        root_file["RunHeader/RunInfo/Time Resolution"] = "5 ns"
        root_file["RunHeader/DetectorInfo/Detector001/Name"] = "Offset detector"
        root_file["RunHeader/DetectorInfo/Detector001/Histo Number"] = "21"
        root_file["RunHeader/DetectorInfo/Detector001/Time Zero Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/First Good Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/Last Good Bin"] = "3"
        root_file["histos/DecayAnaModule/hDecay001"] = (
            np.array([100, 100, 100, 100], dtype=np.float64),
            edges,
        )
        root_file["histos/DecayAnaModule/hDecay021"] = (
            np.array([1, 2, 3, 4], dtype=np.float64),
            edges,
        )


def _write_zero_field_title_root_directory(path: Path) -> None:
    edges = np.arange(-0.5, 4.5, 1.0)
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "9753"
        root_file["RunHeader/RunInfo/Run Title"] = "Sample, LF 32G Bz"
        root_file["RunHeader/RunInfo/Comment"] = "Header field intentionally left at zero"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Instrument"] = "FLAME"
        root_file["RunHeader/RunInfo/Sample Magnetic Field"] = "0 G"
        root_file["RunHeader/RunInfo/No of Histos"] = "2"
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        root_file["RunHeader/DetectorInfo/Detector001/Name"] = "Forward"
        root_file["RunHeader/DetectorInfo/Detector001/Histo Number"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/Time Zero Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/First Good Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/Last Good Bin"] = "3"
        root_file["RunHeader/DetectorInfo/Detector002/Name"] = "Backward"
        root_file["RunHeader/DetectorInfo/Detector002/Histo Number"] = "2"
        root_file["RunHeader/DetectorInfo/Detector002/Time Zero Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector002/First Good Bin"] = "1"
        root_file["RunHeader/DetectorInfo/Detector002/Last Good Bin"] = "3"
        root_file["histos/DecayAnaModule/hDecay001"] = (
            np.array([10, 20, 30, 40], dtype=np.float64),
            edges,
        )
        root_file["histos/DecayAnaModule/hDecay002"] = (
            np.array([9, 18, 27, 36], dtype=np.float64),
            edges,
        )


def _write_flame_root_directory(path: Path) -> None:
    edges = np.arange(-0.5, 5.5, 1.0)
    labels = ["Forward", "Backward", "Right", "Left", "R_F", "R_B", "L_F", "L_B"]
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "8642"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Instrument"] = "FLAME"
        root_file["RunHeader/RunInfo/No of Histos"] = str(len(labels))
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        for index, label in enumerate(labels, start=1):
            prefix = f"RunHeader/DetectorInfo/Detector{index:03d}"
            root_file[f"{prefix}/Name"] = label
            root_file[f"{prefix}/Histo Number"] = str(index)
            root_file[f"{prefix}/Time Zero Bin"] = "1"
            root_file[f"{prefix}/First Good Bin"] = "1"
            root_file[f"{prefix}/Last Good Bin"] = "4"
            root_file[f"histos/DecayAnaModule/hDecay{index:03d}"] = (
                np.arange(5, dtype=np.float64) + index,
                edges,
            )
        root_file["RunHeader/RunInfo/Sample Temperature"] = (
            "10.4 +- 0.1 K; SP: 10; SP=SAM_ts_target Sens=SAM_ts_value"
        )
        root_file["RunHeader/RunInfo/Sample Temperature [0]"] = (
            "10.9 +- 0.2 K; Sens=DIL_T_mix_value"
        )
        root_file["histos/SCAnaModule/[12] flamesam0 SAM_ts_value"] = (
            np.array([10.8, 10.6, 10.4], dtype=np.float64),
            np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float64),
        )
        root_file["histos/SCAnaModule/[06] flamedil0 DIL_T_mix_value"] = (
            np.array([11.2, 11.1, 10.9], dtype=np.float64),
            np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float64),
        )


def _write_gps_subdetector_root_directory(path: Path) -> None:
    edges = np.arange(-0.5, 5.5, 1.0)
    # GPS MusrRoot exposes the raw split sub-detectors (verified histogram order).
    labels = [
        "Forw",
        "Back",
        "Up_B",
        "Up_F",
        "Down_B",
        "Down_F",
        "Right_B",
        "Right_F",
        "Left_B",
        "Left_F",
        "Mob-RL",
    ]
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "5848"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Instrument"] = "GPS"
        root_file["RunHeader/RunInfo/No of Histos"] = str(len(labels))
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        for index, label in enumerate(labels, start=1):
            prefix = f"RunHeader/DetectorInfo/Detector{index:03d}"
            root_file[f"{prefix}/Name"] = label
            root_file[f"{prefix}/Histo Number"] = str(index)
            root_file[f"{prefix}/Time Zero Bin"] = "1"
            root_file[f"{prefix}/First Good Bin"] = "1"
            root_file[f"{prefix}/Last Good Bin"] = "4"
            root_file[f"histos/DecayAnaModule/hDecay{index:03d}"] = (
                np.arange(5, dtype=np.float64) + index,
                edges,
            )


def test_root_loader_groups_gps_subdetectors_like_psi_bin(tmp_path) -> None:
    # GPS ROOT (MusrRoot) splits each transverse plate into _B/_F halves; the
    # default grouping must combine them so ROOT matches the six-group PSI-BIN
    # GPS default (Forward, Backward, Up, Down, Left, Right), leaving the
    # ungrouped Mobile detector on its own.
    path = tmp_path / "gps_subdetectors.root"
    _write_gps_subdetector_root_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    groups = ds.run.grouping["groups"]
    names = ds.run.grouping["group_names"]
    assert groups == {
        1: [1],
        2: [2],
        3: [3, 4],
        4: [5, 6],
        5: [7, 8],
        6: [9, 10],
        7: [11],
    }
    assert names == {
        1: "Forw",
        2: "Back",
        3: "Up",
        4: "Down",
        5: "Right",
        6: "Left",
        7: "Mob-RL",
    }
    # Beam Forward/Backward map to the muon-spin analysis pair (as for FLAME).
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1


def _write_labeled_root_directory(path: Path, *, instrument: str, labels: list[str]) -> None:
    edges = np.arange(-0.5, 5.5, 1.0)
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "5849"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Instrument"] = instrument
        root_file["RunHeader/RunInfo/No of Histos"] = str(len(labels))
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        for index, label in enumerate(labels, start=1):
            prefix = f"RunHeader/DetectorInfo/Detector{index:03d}"
            root_file[f"{prefix}/Name"] = label
            root_file[f"{prefix}/Histo Number"] = str(index)
            root_file[f"{prefix}/Time Zero Bin"] = "1"
            root_file[f"{prefix}/First Good Bin"] = "1"
            root_file[f"{prefix}/Last Good Bin"] = "4"
            root_file[f"histos/DecayAnaModule/hDecay{index:03d}"] = (
                np.arange(5, dtype=np.float64) + index,
                edges,
            )


def test_root_loader_merges_paired_subdetectors_regardless_of_instrument(tmp_path) -> None:
    # Real GPS files may not report Instrument == "GPS", so the split-half merge
    # is detected structurally: a transverse base present as BOTH _B and _F halves
    # is the GPS convention and is combined whatever the instrument label reads.
    path = tmp_path / "paired_subdetectors.root"
    _write_labeled_root_directory(
        path,
        instrument="",  # instrument label missing / non-"GPS"
        labels=["Forw", "Back", "Up_B", "Up_F", "Down_B", "Down_F"],
    )

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run.grouping["groups"] == {1: [1], 2: [2], 3: [3, 4], 4: [5, 6]}
    assert ds.run.grouping["group_names"] == {1: "Forw", 2: "Back", 3: "Up", 4: "Down"}


def test_root_loader_does_not_merge_unpaired_subdetectors(tmp_path) -> None:
    # Without a genuine _B/_F pair (here a lone Up_F), nothing is merged — each
    # histogram stays its own group.
    path = tmp_path / "unpaired_subdetectors.root"
    _write_labeled_root_directory(
        path,
        instrument="DOLLY",
        labels=["Forward", "Backward", "Up_F", "Left"],
    )

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run.grouping["groups"] == {1: [1], 2: [2], 3: [3], 4: [4]}
    assert ds.run.grouping["group_names"] == {1: "Forward", 2: "Backward", 3: "Up_F", 4: "Left"}


def test_load_musrroot_directory_reads_header_histograms_and_grouping(tmp_path) -> None:
    path = tmp_path / "lem_synthetic.root"
    _write_musrroot_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 2468
    assert ds.metadata["root_format"] == "musr-root-directory"
    assert ds.metadata["facility"] == "PSI"
    assert ds.metadata["instrument"] == "LEM"
    assert ds.metadata["temperature"] == pytest.approx(12.5)
    assert ds.metadata["field"] == pytest.approx(100.0)
    assert ds.metadata["musrroot_slow_control_log"]["source_format"] == "MusrRoot SCAnaModule"
    assert ds.metadata["musrroot_slow_control_log"]["channels"] == ["Sample Temperature"]
    series = ds.metadata["nexus_time_series"]["musrroot_slow_control/Sample Temperature"]
    assert series["units"] == "K"
    assert series["time"] == pytest.approx([10.0, 30.0, 50.0])
    assert series["values"] == pytest.approx([11.8, 12.2, 12.4])
    assert series["mean"] == pytest.approx(12.1333333333)
    assert len(ds.run.histograms) == 2
    assert ds.run.histograms[0].bin_width == pytest.approx(0.01)
    assert ds.run.grouping["groups"] == {1: [1], 2: [2]}
    assert ds.run.grouping["group_names"] == {1: "Left", 2: "Right"}
    assert ds.run.grouping["forward_group"] == 1
    assert ds.run.grouping["backward_group"] == 2
    assert ds.run.grouping["detector_t0_bins"] == [2, 3]
    assert ds.run.grouping["root_histo_numbers"] == [1, 2]
    assert ds.time[0] == pytest.approx(0.01)


def test_red_green_offsets_select_declared_histogram_numbers(tmp_path) -> None:
    path = tmp_path / "lem_offset.root"
    _write_red_green_offset_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run.grouping["root_histo_numbers"] == [21]
    assert ds.run.grouping["group_names"] == {1: "Offset detector"}
    assert ds.run.grouping["detector_t0_bins"] == [1]
    assert ds.run.histograms[0].counts.tolist() == [1, 2, 3, 4]
    assert ds.run.histograms[0].bin_width == pytest.approx(0.005)


def test_root_loader_marks_title_field_candidate_when_header_field_is_zero(tmp_path) -> None:
    path = tmp_path / "flame_zero_field.root"
    _write_zero_field_title_root_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.metadata["field"] == pytest.approx(0.0)
    assert ds.metadata["field_header"] == pytest.approx(0.0)
    assert ds.metadata["field_comment_candidate"] == pytest.approx(32.0)


def test_root_loader_preserves_flame_instrument_for_layout_detection(tmp_path) -> None:
    path = tmp_path / "flame_synthetic.root"
    _write_flame_root_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.metadata["instrument"] == "FLAME"
    assert ds.run.grouping["instrument"] == "FLAME"
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1
    assert ds.run.grouping["group_names"][5] == "R_F"


def test_root_loader_reads_flame_sensor_named_temperature_logs(tmp_path) -> None:
    path = tmp_path / "flame_synthetic.root"
    _write_flame_root_directory(path)

    ds = RootLoader().load(str(path))

    series = ds.metadata["nexus_time_series"]
    sample_series = series["musrroot_slow_control/flamesam0 SAM ts value"]
    assert sample_series["role"] == "sample_temperature"
    assert sample_series["sensor"] == "SAM_ts_value"
    assert sample_series["primary"] is True
    assert sample_series["units"] == "K"
    assert sample_series["time"] == pytest.approx([15.0, 45.0, 75.0])
    assert sample_series["mean"] == pytest.approx(10.6)

    dil_series = series["musrroot_slow_control/flamedil0 DIL T mix value"]
    assert dil_series["role"] == "sample_temperature"
    assert dil_series["sensor"] == "DIL_T_mix_value"
    assert dil_series["primary"] is False


def test_load_convenience_registers_root_loader(tmp_path) -> None:
    path = tmp_path / "lem_synthetic.root"
    _write_musrroot_directory(path)

    ds = load(str(path))

    assert ds.metadata["root_format"] == "musr-root-directory"


def test_load_musrfit_musrroot_folder_fixture() -> None:
    path = _musrfit_example_file("lem15_his_2994.root")

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 2994
    assert ds.metadata["root_format"] == "musr-root-folder"
    assert ds.metadata["facility"] == "PSI"
    assert ds.metadata["instrument"] == "LEM"
    assert ds.metadata["temperature"] == pytest.approx(45.0)
    assert ds.metadata["field"] == pytest.approx(26.67)
    assert len(ds.run.histograms) == 32
    assert ds.run.histograms[0].bin_width == pytest.approx(0.0001953125)
    assert ds.run.grouping["root_histo_numbers"][:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds.run.grouping["root_histo_numbers"][8:16] == [21, 22, 23, 24, 25, 26, 27, 28]
    assert ds.run.grouping["group_names"][1].startswith("e+ Left")


def test_load_musrfit_musrroot_fixture_used_by_musrfit_histo_test() -> None:
    """Read the ROOT file referenced by musrfit's test-histo-MusrRoot.msr."""
    path = _musrfit_example_file("lem12_his_2466.root")

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 2466
    assert ds.metadata["root_format"] == "musr-root-folder"
    assert ds.metadata["title"].startswith("LSCO x=0.02 (224-227), T=12.00 (K)")
    assert ds.metadata["sample"] == "LSCO x=0.02 (224-227)"
    assert ds.metadata["temperature"] == pytest.approx(11.999)
    assert ds.metadata["field"] == pytest.approx(49.11)
    assert ds.metadata["started"] == "2012-06-03 13:22:00"
    assert ds.metadata["stopped"] == "2012-06-03 14:04:38"
    assert len(ds.run.histograms) == 16
    assert ds.run.histograms[0].n_bins == 66601
    assert ds.run.histograms[0].bin_width == pytest.approx(0.0001953125)
    assert ds.run.grouping["root_histo_numbers"][:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds.run.grouping["root_histo_numbers"][8:16] == [21, 22, 23, 24, 25, 26, 27, 28]
    assert ds.run.grouping["detector_t0_bins"] == [2741] * 16
    assert ds.run.grouping["detector_first_good_bins"] == [2741] * 16
    assert ds.run.grouping["detector_last_good_bins"] == [66600] * 16
    assert ds.run.grouping["group_names"][1] == "e+ Left D(F)"
    assert ds.run.grouping["group_names"][8] == "e+ Bottom U(B)"
    assert [float(np.sum(h.counts)) for h in ds.run.histograms[:8]] == pytest.approx(
        [
            521957.0,
            566162.0,
            540160.0,
            481742.0,
            491464.0,
            505091.0,
            499640.0,
            483329.0,
        ]
    )
