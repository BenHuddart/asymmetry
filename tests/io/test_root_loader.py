"""Tests for MusrRoot/LEM ROOT loading."""

from __future__ import annotations

import itertools
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


def _write_header_strings(root_file, entries: list[tuple[str, str, str]]) -> None:
    """Write ``RunHeader`` TObjStrings with explicit TKey names.

    ``root_file[path] = value`` cannot create the new-format leaves: the
    encoded names contain ``/`` ("N/A", "MeV/c", URLs) and ``:``, which
    uproot's path assignment splits or rejects, so the strings are added to
    each folder directly.
    """
    from uproot.writing.identify import add_to_directory

    directories: dict[str, object] = {}
    streamers: list = []
    for folder, name, payload in entries:
        directory = directories.get(folder)
        if directory is None:
            directory = root_file.mkdir(f"RunHeader/{folder}")
            directories[folder] = directory
        add_to_directory(payload, name, directory, streamers)
    root_file.file._cascading.streamers.update_streamers(root_file.file.sink, streamers)


def _th1(name: str, title: str, counts: np.ndarray, *, xmax: float):
    """A writable TH1D with an explicit ``fTitle``.

    uproot's plain ``(counts, edges)`` assignment cannot set a title, and the
    new-format fixtures need one to reproduce the DAQ trap of writing every
    histogram title with the last character of the filename dropped.
    """
    from uproot.writing.identify import to_TAxis, to_TH1x

    data = np.concatenate([[0.0], np.asarray(counts, dtype=np.float64), [0.0]])
    return to_TH1x(
        fName=name,
        fTitle=title,
        data=data,
        fEntries=float(len(counts)),
        fTsumw=0.0,
        fTsumw2=0.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=np.zeros(len(data)),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=len(counts), fXmin=0.0, fXmax=xmax),
    )


_ENCODED_FLAME_DETECTORS = ["Forward", "Backward", "Right", "Left", "R_F", "R_B", "L_F", "L_B"]

#: Free-text RunSummary lines (some with colons, some without, one blank); the
#: DAQ numbers them "NNNN <text>" with no ``-@type`` suffix.
_ENCODED_RUN_SUMMARY_LINES = [
    "########################################",
    "# Run summary information",
    "########################################",
    "",
    "Run:",
    "  started at: 2026-01-02 10:00:00",
    "Events: 12345678",
]


def _write_encoded_musrroot_directory(path: Path) -> None:
    """Write the new-format (2026 FLAME DAQ) MusrRoot TDirectory layout.

    Every ``RunHeader`` leaf is a TObjString whose TKey name AND payload are
    the identical encoded string ``NNN - Label: Value -@type``, numbered by a
    single global counter that interleaves across subfolders (so per-folder
    numbering is non-contiguous, as in real files). All values are invented.
    """
    n_bins = 102401
    counter = itertools.count()
    entries: list[tuple[str, str, str]] = []

    def put(folder: str, entry: str) -> None:
        encoded = f"{next(counter):03d} - {entry}"
        entries.append((folder, encoded, encoded))

    def put_detector(index: int) -> None:
        folder = f"DetectorInfo/Detector{index:03d}"
        put(folder, f"Name: {_ENCODED_FLAME_DETECTORS[index - 1]} -@0")
        put(folder, f"Histo Number: {index} -@1")
        put(folder, f"Histo Length: {n_bins} -@1")
        put(folder, "Time Zero Bin: 2000.000000 -@2")
        put(folder, "First Good Bin: 2020 -@1")
        put(folder, f"Last Good Bin: {n_bins - 1} -@1")

    for entry in [
        "Version: N/A -@0",
        "Generic Validator URL: https://example.invalid/validator -@0",
        "Specific Validator URL: https://example.invalid/validator -@0",
        "Generator: MuSRrootHeader -@0",
        "Proposal Number: 20990001 -@1",
        "Main Proposer: A. Nonymous -@0",
        "Run Title: CuTest, TF45, synthetic run -@0",
        "Run Number: 4321 -@1",
        "Run Start Time: 2026-01-02 10:00:00 -@0",
        "Run Stop Time: 2026-01-02 11:00:00 -@0",
        "Run Duration: 3600 sec -@3",
        "Laboratory: PSI -@0",
        "Instrument: flame -@0",  # the DAQ writes the instrument in lower case
        "Muon Beam Momentum: 28.1000003815 MeV/c -@3",
        "Muon Source: M -@0",
        "Setup: flame, Sample, TestMag, SampleStick -@0",
        "Comment: n/a -@0",
        "Sample Name: CuTest -@0",
        "Sample Temperature: 150.15 +- 0.01 K -@3",
        "Sample Magnetic Field: 7799.8 +- 0.1 G -@3",
        "No of Histos: 8 -@1",
        "Time Resolution: 0.09765625 ns; SiPM -@3",
        "RedGreen Offsets: 0 -@5",
    ]:
        put("RunInfo", entry)
    put_detector(1)
    put("SampleEnvironmentInfo", "Cryo: SampleStick -@0")
    put("MagneticFieldEnvironmentInfo", "Magnet Name: TestMag -@0")
    put("BeamlineInfo", "Name: piM3.3 -@0")
    for entry in [
        "P-Group: p00000 -@0",
        "Field longitudinal: 7799.8 +- 0.1 G -@3",
        "Field vertical: 0.014 +- 0.008 G -@3",
        "Field horizontal: -0.012 +- 0.007 G -@3",
        "Cryostat Temperature: 149.998 +- 0.008 K -@3",
        "Dilution Mix Temperature: 126.112 +- 0.005 K -@3",
        "Dilution Sorb Temperature: 132 +- 138 K -@3",
    ]:
        put("RunInfo", entry)
    for index in range(2, 9):
        put_detector(index)

    for line_no, text in enumerate(_ENCODED_RUN_SUMMARY_LINES):
        payload = f"{line_no:04d} {text}"
        entries.append(("RunSummary", payload.rstrip(), payload))

    truncated = f"{path.name[:-1]}"  # the DAQ drops the filename's last char
    sc_edges = 6
    with uproot.recreate(path) as root_file:
        _write_header_strings(root_file, entries)
        for index, name in enumerate(_ENCODED_FLAME_DETECTORS, start=1):
            counts = np.zeros(n_bins)
            counts[2000:] = 100.0 + index
            root_file[f"histos/DecayAnaModule/hDecay{index:03d}"] = _th1(
                f"hDecay{index:03d}", f"{name} Run {truncated}", counts, xmax=float(n_bins)
            )
        for name, values in [
            # Zero-padded tail after the recorded range (real trap shape).
            ("Cryostat Sample Temperature", [150.2, 150.1, 150.15, 0.0, 0.0, 0.0]),
            ("Cryostat Temperature", [149.9, 150.0, 150.1, 0.0, 0.0, 0.0]),
            # NaN in the populated range plus the zero-padded tail (real trap).
            ("Dilution Sample Temperature", [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0]),
            # A channel with no finite samples at all.
            ("Dilution Still Temperature", [np.nan] * sc_edges),
        ]:
            root_file[f"histos/SCAnaModule/{name}"] = _th1(
                name, f"{name} Run {truncated}", np.asarray(values), xmax=3600.0
            )


def _write_encoded_gps_directory(path: Path) -> None:
    """GPS variant of the encoded TDirectory layout (split _B/_F halves)."""
    labels = ["Forw", "Back", "Up_B", "Up_F", "Down_B", "Down_F"]
    counter = itertools.count()
    entries: list[tuple[str, str, str]] = []

    def put(folder: str, entry: str) -> None:
        encoded = f"{next(counter):03d} - {entry}"
        entries.append((folder, encoded, encoded))

    for entry in [
        "Run Number: 7654 -@1",
        "Laboratory: PSI -@0",
        "Instrument: gps -@0",
        f"No of Histos: {len(labels)} -@1",
        "Time Resolution: 10 ns -@3",
    ]:
        put("RunInfo", entry)
    for index, label in enumerate(labels, start=1):
        folder = f"DetectorInfo/Detector{index:03d}"
        put(folder, f"Name: {label} -@0")
        put(folder, f"Histo Number: {index} -@1")
        put(folder, "Time Zero Bin: 1.000000 -@2")
        put(folder, "First Good Bin: 1 -@1")
        put(folder, "Last Good Bin: 4 -@1")

    edges = np.arange(-0.5, 5.5, 1.0)
    with uproot.recreate(path) as root_file:
        _write_header_strings(root_file, entries)
        for index in range(1, len(labels) + 1):
            root_file[f"histos/DecayAnaModule/hDecay{index:03d}"] = (
                np.arange(5, dtype=np.float64) + index,
                edges,
            )


def test_load_encoded_musrroot_directory_parses_header(tmp_path) -> None:
    # The filename digits (00001) are deliberately misleading: everything must
    # come from the encoded header, not from filename scraping.
    path = tmp_path / "flame99_his_00001.root"
    _write_encoded_musrroot_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.metadata["root_format"] == "musr-root-directory"
    assert ds.run_number == 4321
    assert ds.metadata["title"] == "CuTest, TF45, synthetic run"
    assert ds.metadata["sample"] == "CuTest"
    assert ds.metadata["temperature"] == pytest.approx(150.15)
    assert ds.metadata["field"] == pytest.approx(7799.8)
    assert ds.metadata["instrument"] == "FLAME"
    assert ds.metadata["facility"] == "PSI"
    assert ds.metadata["beamline"] == "piM3.3"
    assert ds.metadata["muon_source"] == "M"
    assert ds.metadata["comment"] == "n/a"
    assert ds.metadata["started"] == "2026-01-02 10:00:00"
    assert ds.metadata["stopped"] == "2026-01-02 11:00:00"
    # 0.09765625 ns despite the "; SiPM" description on the @3 quantity.
    assert ds.run.histograms[0].bin_width == pytest.approx(9.765625e-5)
    assert ds.metadata["musrroot_run_summary"] == "\n".join(_ENCODED_RUN_SUMMARY_LINES)


def test_encoded_musrroot_header_decodes_labels_and_skips_run_summary(tmp_path) -> None:
    path = tmp_path / "flame99_his_00001.root"
    _write_encoded_musrroot_directory(path)

    with uproot.open(path) as root_file:
        header, kind, run_summary = RootLoader()._read_header(root_file)

    assert kind == "musr-root-directory"
    assert header["RunInfo/Run Title"] == "CuTest, TF45, synthetic run"
    assert header["RunInfo/Instrument"] == "flame"
    # "/" inside a leaf name must not be mistaken for a path separator.
    assert header["RunInfo/Version"] == "N/A"
    assert header["RunInfo/Muon Beam Momentum"] == "28.1000003815 MeV/c"
    assert header["RunInfo/RedGreen Offsets"] == "0"
    assert header["RunInfo/Time Resolution"] == "0.09765625 ns; SiPM"
    assert header["DetectorInfo/Detector001/Name"] == "Forward"
    assert header["DetectorInfo/Detector008/Name"] == "L_B"
    assert header["SampleEnvironmentInfo/Cryo"] == "SampleStick"
    assert header["MagneticFieldEnvironmentInfo/Magnet Name"] == "TestMag"
    assert header["BeamlineInfo/Name"] == "piM3.3"
    assert not any(key.startswith("RunSummary") for key in header)
    assert run_summary == "\n".join(_ENCODED_RUN_SUMMARY_LINES)


def test_encoded_musrroot_directory_grouping_and_good_bins(tmp_path) -> None:
    path = tmp_path / "flame99_his_00001.root"
    _write_encoded_musrroot_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    grouping = ds.run.grouping
    # Labels come from the decoded header Names, not the truncated TH1 titles,
    # and the single-letter R_F/R_B/L_F/L_B detectors are genuinely distinct
    # (never merged into synthetic "R"/"L" groups).
    assert grouping["groups"] == {gid: [gid] for gid in range(1, 9)}
    assert grouping["group_names"] == {
        gid: name for gid, name in enumerate(_ENCODED_FLAME_DETECTORS, start=1)
    }
    assert grouping["forward_group"] == 2
    assert grouping["backward_group"] == 1
    assert grouping["detector_t0_bins"] == [2000] * 8
    assert grouping["detector_first_good_bins"] == [2020] * 8
    assert grouping["detector_last_good_bins"] == [102400] * 8
    assert len(ds.run.histograms) == 8
    assert ds.run.histograms[0].n_bins == 102401


def test_encoded_musrroot_slow_control_roles_and_nan_robustness(tmp_path) -> None:
    path = tmp_path / "flame99_his_00001.root"
    _write_encoded_musrroot_directory(path)

    ds = RootLoader().load(str(path))

    series = ds.metadata["nexus_time_series"]
    primary = series["musrroot_slow_control/Cryostat Sample Temperature"]
    assert primary["role"] == "sample_temperature"
    assert primary["primary"] is True
    assert primary["units"] == "K"

    dilution = series["musrroot_slow_control/Dilution Sample Temperature"]
    assert dilution["role"] == "sample_temperature"
    assert dilution["primary"] is False
    assert np.isfinite(dilution["values"]).all()

    # Cryostat Temperature is a distinct RunInfo entry, not the sample role.
    assert not series["musrroot_slow_control/Cryostat Temperature"].get("primary", False)
    # A channel with no finite samples is dropped rather than crashing.
    assert "musrroot_slow_control/Dilution Still Temperature" not in series


def test_primary_sample_temperature_selection_masks_zero_padding() -> None:
    # Without masking the zero-padded tail, the padded series' mean is ~75.1
    # and the 80 K series would win the closeness comparison against 150.15.
    loader = RootLoader()
    padded = {
        "role": "sample_temperature",
        "primary": False,
        "values": [150.2, 150.1, 150.15, 0.0, 0.0, 0.0],
    }
    offset = {"role": "sample_temperature", "primary": False, "values": [80.0, 80.0]}
    series = {"musrroot_slow_control/padded": padded, "musrroot_slow_control/offset": offset}

    loader._resolve_primary_sample_temperature(
        series, {"RunInfo/Sample Temperature": "150.15 +- 0.01 K"}
    )

    assert padded["primary"] is True
    assert offset["primary"] is False


def test_primary_sample_temperature_promotes_lone_candidate() -> None:
    # A file with a single label-matched temperature channel must still get a
    # primary series (psi.py and series scoring key off the flag).
    loader = RootLoader()
    lone = {"role": "sample_temperature", "primary": False, "values": [150.1, 150.2]}
    series = {"musrroot_slow_control/Cryostat Sample Temperature": lone}

    loader._resolve_primary_sample_temperature(series, {})

    assert lone["primary"] is True


def test_primary_sample_temperature_keeps_sensor_primary_without_scalar() -> None:
    # With no usable header scalar, a Sens=-designated primary must not be
    # demoted in favour of whichever candidate happens to iterate first.
    loader = RootLoader()
    first = {"role": "sample_temperature", "primary": False, "values": [10.9]}
    designated = {"role": "sample_temperature", "primary": True, "values": [10.4]}
    series = {
        "musrroot_slow_control/a_first": first,
        "musrroot_slow_control/b_designated": designated,
    }

    loader._resolve_primary_sample_temperature(series, {})

    assert designated["primary"] is True
    assert first["primary"] is False


def test_encoded_gps_directory_merges_split_halves_like_folder_fixture(tmp_path) -> None:
    # The encoded TDirectory form must group GPS exactly like the clean/folder
    # fixtures: _B/_F transverse halves merge, beam counters stay separate.
    path = tmp_path / "gps_encoded.root"
    _write_encoded_gps_directory(path)

    ds = RootLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 7654
    assert ds.metadata["instrument"] == "GPS"
    assert ds.run.grouping["groups"] == {1: [1], 2: [2], 3: [3, 4], 4: [5, 6]}
    assert ds.run.grouping["group_names"] == {1: "Forw", 2: "Back", 3: "Up", 4: "Down"}
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1


def test_parse_musrroot_string_preserves_internal_hyphens() -> None:
    loader = RootLoader()
    assert loader._parse_musrroot_string("005 - Name: Left/Forward - field off -@0") == (
        "Name",
        "Left/Forward - field off",
    )
    assert loader._parse_musrroot_string("023 - Time Resolution: 0.09765625 ns; SiPM -@3") == (
        "Time Resolution",
        "0.09765625 ns; SiPM",
    )


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
