"""Tests for PSI BIN/MDU raw histogram loading."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.io import load
from asymmetry.core.io.psi import _FE_HEADER, PsiLoader
from asymmetry.core.transform import (
    apply_deadtime_correction,
    apply_grouping_aligned,
    common_t0_for_groups,
    prepare_histograms_with_deadtime,
)

#: musrfit ships example PSI-BIN/MDU files in its own source tree, which
#: Asymmetry does not vendor. Point ``ASYMMETRY_MUSRFIT_DATA`` at
#: ``<musrfit>/doc/examples/data`` to run the reader-parity tests below; without
#: it (e.g. in CI) they skip.
_MUSRFIT_DATA_DIR = os.environ.get("ASYMMETRY_MUSRFIT_DATA")


def _musrfit_example_file(name: str) -> Path:
    """Resolve a musrfit example data file, skipping if the corpus is absent."""
    if not _MUSRFIT_DATA_DIR:
        pytest.skip("ASYMMETRY_MUSRFIT_DATA not set; musrfit reader-parity corpus unavailable")
    path = Path(_MUSRFIT_DATA_DIR) / name
    if not path.exists():
        pytest.skip(f"musrfit example file not available: {name}")
    return path


def _write_psi_bin(
    path,
    labels: list[bytes] | None = None,
    counts: np.ndarray | None = None,
    field: bytes = b"0.1T      ",
    title: bytes = b"PSI BIN test",
) -> None:
    if labels is None:
        labels = [b"Back", b"Forw"]
    if counts is None:
        counts = np.array(
            [
                [0, 10, 20, 30, 40, 50],
                [0, 0, 0, 15, 25, 35],
            ],
            dtype="<i4",
        )
    n_hist = len(labels)
    n_bins = int(counts.shape[1])
    header = bytearray(1024)
    header[0:2] = b"1N"
    struct.pack_into("<h", header, 2, 4)
    struct.pack_into("<h", header, 6, 4321)
    struct.pack_into("<h", header, 28, n_bins)
    struct.pack_into("<h", header, 30, n_hist)
    struct.pack_into("<h", header, 128, n_hist)
    struct.pack_into("<h", header, 130, n_bins)
    struct.pack_into("<h", header, 132, 1)
    struct.pack_into("<h", header, 134, 1)
    header[138:148] = b"Sample    "
    header[148:158] = b"50.0K     "
    header[158:168] = field[:10].ljust(10, b" ")
    header[168:178] = b"TF        "
    header[218:227] = b"01-JAN-26"
    header[227:236] = b"01-JAN-26"
    header[236:244] = b"10:00:00"
    header[244:252] = b"11:00:00"
    header[860:922] = title[:62].ljust(62, b" ")
    for i, label in enumerate(labels):
        header[948 + i * 4 : 952 + i * 4] = label[:4].ljust(4, b" ")
    struct.pack_into("<f", header, 1012, 0.01)
    t0_values = [1 + 2 * i for i in range(n_hist)]
    first_good_values = [min(n_bins - 1, value + 2) for value in t0_values]
    last_good_values = [n_bins - 1 for _ in range(n_hist)]
    for i, value in enumerate(t0_values):
        struct.pack_into("<h", header, 458 + i * 2, value)
    for i, value in enumerate(first_good_values):
        struct.pack_into("<h", header, 490 + i * 2, value)
    for i, value in enumerate(last_good_values):
        struct.pack_into("<h", header, 522 + i * 2, value)

    path.write_bytes(bytes(header) + counts.astype("<i4", copy=False).tobytes())


def _write_psi_mon(path) -> None:
    lines = ["! ignored"] * 7
    lines.extend(
        [
            "! 04-Jul-2011 10:40:23",
            "! Title: Heater Sample Cryostat Shield",
            "00:00:00\\4\\4.9906 5.1805 4.9921 5.1804\\0 0 0 0\\",
            "00:00:10\\4\\5.0000 5.2000 5.1000 5.3000\\0 0 0 0\\",
        ]
    )
    path.write_text("\n".join(lines), encoding="latin-1")


def _write_psi_mon_with_backslash_titles(path) -> None:
    lines = ["! ignored"] * 7
    lines.extend(
        [
            "! 04-Jul-2011 10:40:23",
            "! Title: Cryostat\\Sample",
            "00:00:00\\2\\4.9906\\5.1805\\",
            "00:00:10\\2\\5.0000\\5.2000\\",
        ]
    )
    path.write_text("\n".join(lines), encoding="latin-1")


def _write_flame_mon(path, equipment: str, title: str, rows: list[str]) -> None:
    lines = [
        "! File = run_4321.mon",
        f"! Midas Event ID is: 221  Equipment name: FRAPPY {equipment}",
        "! Record format is:",
        "!",
        "!   <delta_time>\\<n_vals>\\<val1> <val2> ..\\<int1> <int2> ..\\",
        "!    <delta_time> = [dd ]hh:mm:ss",
        "!",
        "! 05-NOV-2025 14:53:19: Start of Run 4321",
        f"! Title: {title}",
    ]
    lines.extend(rows)
    path.write_text("\n".join(lines), encoding="latin-1")


def _tag(
    label: bytes,
    tag_type: bytes,
    *,
    histominb: int = 0,
    histomaxb: int = 5,
    t0b: int = 1,
    tfb: int = 2,
    tlb: int = 5,
) -> bytes:
    data = bytearray(60)
    data[:12] = label[:12].ljust(12, b"\x00")
    data[12:13] = tag_type
    values = [0, 0, 0, 0, 0, 0, histominb, histomaxb, t0b, tfb, tlb]
    struct.pack_into("<11i", data, 16, *values)
    return bytes(data)


def _write_psi_mdu(path) -> None:
    tags = [_tag(b"F1", b"P"), _tag(b"B1", b"P")]
    tags.extend(_tag(b"", b"N", histomaxb=0) for _ in range(30))
    settings_prefix = struct.pack("<13i", 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0)
    settings = settings_prefix + b"".join(tags)
    stats = bytes(336)
    header = _FE_HEADER.pack(
        b"T",
        b"5",
        b"01-JAN-2026\x00",
        b"10:00:00\x00",
        b"01-JAN-2026\x00",
        b"11:00:00\x00",
        2468,
        1,
        b"Sample    25.0K     1.0G      TF       \x00",
        b"PSI MDU test\x00",
        b"\x00" * 20,
        1,
        0,
        6,
        2,
        b"0 1\x00",
        b"25.0\x00",
        b"0.1\x00",
        0,
        _FE_HEADER.size,
        len(settings),
        60,
        len(stats),
    )
    counts_a = np.array([10, 20, 30, 40, 50, 60], dtype="<i4")
    counts_b = np.array([8, 18, 28, 38, 48, 58], dtype="<i4")
    payload = bytearray()
    for idx, tag in enumerate(tags):
        payload.extend(tag)
        if tag[12:13] == b"P":
            payload.extend((counts_a if idx == 0 else counts_b).tobytes())
    path.write_bytes(header + settings + stats + bytes(payload))


def test_load_psi_bin_uses_labels_and_per_detector_t0(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    _write_psi_bin(path)

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.metadata["facility"] == "PSI"
    assert ds.metadata["psi_format"] == "psi-bin"
    assert ds.metadata["temperature"] == pytest.approx(50.0)
    assert ds.metadata["field"] == pytest.approx(1000.0)
    # The "TF" orientation tag in the header is classified as Transverse so the
    # transverse-field grouping nudge can fire on PSI data (B8a).
    assert ds.metadata["field_direction"] == "Transverse"
    assert ds.run.grouping["groups"] == {1: [1], 2: [2]}
    assert ds.run.grouping["group_names"] == {1: "Back", 2: "Forw"}
    assert ds.run.grouping["forward_group"] == 1
    assert ds.run.grouping["backward_group"] == 2
    assert ds.run.grouping["detector_t0_bins"] == [1, 3]
    assert ds.run.grouping["t0_bin"] == 3
    assert ds.n_points == 1
    assert ds.time[0] == pytest.approx(0.02)


def test_load_psi_bin_marks_flame_from_filename(tmp_path) -> None:
    labels = [b"Forw", b"Back", b"Righ", b"Left", b"R_F", b"R_B", b"L_F", b"L_B"]
    counts = np.vstack(
        [np.arange(6, dtype=np.int32) + offset for offset in range(len(labels))]
    ).astype("<i4")
    path = tmp_path / "flame0001.bin"
    _write_psi_bin(path, labels=labels, counts=counts)

    ds = PsiLoader().load(str(path))

    assert ds.metadata["instrument"] == "FLAME"
    assert ds.run is not None
    assert ds.run.grouping["instrument"] == "FLAME"
    assert ds.n_points == 1
    assert ds.time[0] == pytest.approx(0.02)


def test_load_psi_bin_guesses_gps_from_digit_adjacent_filename(tmp_path) -> None:
    # gps2923.bin: the GPS token abuts the run number with no separator. The
    # loader must still stamp the instrument as GPS (regression: unmatched names
    # previously fell through to the generic "PSI", which downstream could
    # mis-resolve to a wrong layout).
    labels = [b"Forw", b"Back", b"Up", b"Down", b"Righ"]
    counts = np.vstack(
        [np.arange(6, dtype=np.int32) + offset for offset in range(len(labels))]
    ).astype("<i4")
    path = tmp_path / "gps2923.bin"
    _write_psi_bin(path, labels=labels, counts=counts)

    ds = PsiLoader().load(str(path))

    assert ds.metadata["instrument"] == "GPS"


def test_load_psi_bin_classic_gps_five_counters_resolves_to_gps_layout(tmp_path) -> None:
    # Classic GPS .bin with only five counters (no Left) and no instrument token
    # in the filename must resolve to the GPS layout, never FLAME.
    from asymmetry.core.instrument import detect_instrument

    labels = [b"Forw", b"Back", b"Up", b"Down", b"Righ"]
    counts = np.vstack(
        [np.arange(6, dtype=np.int32) + offset for offset in range(len(labels))]
    ).astype("<i4")
    path = tmp_path / "run4931.bin"
    _write_psi_bin(path, labels=labels, counts=counts)

    ds = PsiLoader().load(str(path))
    n_hist = len(ds.run.histograms)

    resolved = detect_instrument(n_hist, metadata=ds.metadata, source_file=str(path))
    assert resolved == "GPS"


def test_load_psi_bin_reads_temperature_mon_file(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    mon_path = tmp_path / "run_4321_templs0.mon"
    _write_psi_bin(path)
    _write_psi_mon(mon_path)

    ds = PsiLoader().load(str(path))

    series = ds.metadata["nexus_time_series"]
    heater = series["psi_temperature/Temp_Heater"]
    assert heater["units"] == "K"
    assert heater["time"] == pytest.approx([0.0, 10.0])
    assert heater["values"] == pytest.approx([4.9906, 5.0])
    assert heater["mean"] == pytest.approx((4.9906 + 5.0) / 2.0)
    assert heater["min"] == pytest.approx(4.9906)
    assert heater["max"] == pytest.approx(5.0)
    assert heater["source_file"] == str(mon_path)
    assert heater["reader_provenance"] == "Mantid LoadPSIMuonBin-compatible"
    assert ds.metadata["psi_temperature_log"]["source_file"] == str(mon_path)
    assert ds.metadata["psi_temperature_log"]["source_format"] == "PSI .mon"
    assert ds.metadata["psi_temperature_log"]["start_time"] == "2011-07-04T10:40:23"
    assert ds.metadata["psi_temperature_log"]["channels"] == [
        "Temp_Cryostat",
        "Temp_Heater",
        "Temp_Sample",
        "Temp_Shield",
    ]


def test_load_psi_bin_marks_comment_field_candidate_when_header_field_is_zero(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    _write_psi_bin(path, field=b"0G", title=b"LF 32G Bz")

    ds = PsiLoader().load(str(path))

    assert ds.metadata["field"] == pytest.approx(0.0)
    assert ds.metadata["field_header"] == pytest.approx(0.0)
    assert ds.metadata["field_comment_candidate"] == pytest.approx(32.0)


def test_load_psi_bin_finds_temperature_mon_file_below_data_dir(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    log_dir = tmp_path / "logs" / "temperature"
    log_dir.mkdir(parents=True)
    _write_psi_bin(path)
    _write_psi_mon(log_dir / "temperature_4321.mon")

    ds = PsiLoader().load(str(path))

    assert "psi_temperature/Temp_Sample" in ds.metadata["nexus_time_series"]


def test_psi_temperature_log_search_prefers_exact_run_tokens_in_tlog(tmp_path) -> None:
    path = tmp_path / "deltat_tdc_flame_4321.bin"
    log_dir = tmp_path / "tlog"
    log_dir.mkdir()
    _write_psi_bin(path)
    for name in (
        "run_4321_flamesam0.mon",
        "run_14321_flamesam0.mon",
        "run_43210_flamesam0.mon",
    ):
        (log_dir / name).write_text("not parsed here", encoding="latin-1")

    files = PsiLoader()._find_temperature_log_files(path, 4321)

    assert [file.name for file in files] == ["run_4321_flamesam0.mon"]


def test_load_psi_bin_merges_flame_tlog_files_and_marks_sample_temperature(tmp_path) -> None:
    path = tmp_path / "deltat_tdc_flame_4321.bin"
    log_dir = tmp_path / "tlog"
    log_dir.mkdir()
    _write_psi_bin(path)
    _write_flame_mon(
        log_dir / "run_4321_flamedil0.mon",
        "flamedil0",
        "DIL_T_mix DIL_T_sorb DIL_T_still DIL_T_hx\\DIL_T_mix DIL_T_sorb DIL_T_still DIL_T_hx",
        [
            "00:00:03\\4\\0.031 2.7 0.02 2.5\\0.031 2.7 0.02 2.5\\",
            "00:00:13\\4\\0.033 2.8 0.02 2.6\\0.033 2.8 0.02 2.6\\",
        ],
    )
    _write_flame_mon(
        log_dir / "run_4321_flamesam0.mon",
        "flamesam0",
        "SAM_ts SAM_ts_high SAM_ts_low NONE\\SAM_ts SAM_ts_high SAM_ts_low NONE",
        [
            "00:00:01\\4\\0.101 0.102 0 -1\\0.101 0.102 0 -1\\",
            "00:00:11\\4\\0.103 0.104 0 -1\\0.103 0.104 0 -1\\",
        ],
    )
    _write_flame_mon(
        log_dir / "run_4321_variox0.mon",
        "variox0",
        "Variox Sample\\Variox Sample",
        [
            "00:00:00\\2\\2.30 1.20\\2.30 1.20\\",
            "1 00:00:10\\2\\2.32 1.20\\2.32 1.20\\",
        ],
    )

    ds = PsiLoader().load(str(path))

    series = ds.metadata["nexus_time_series"]
    assert "psi_temperature/flamedil0/DIL_T_mix" in series
    assert "psi_temperature/flamesam0/SAM_ts" in series
    assert "psi_temperature/variox0/Variox" in series
    assert "psi_temperature/variox0/Sample" in series
    assert all("NONE" not in key for key in series)

    sample = series["psi_temperature/flamesam0/SAM_ts"]
    assert sample["mean"] == pytest.approx(0.102)
    assert sample["role"] == "sample_temperature"
    assert sample["sensor"] == "SAM_ts_value"
    assert sample["primary"] is True
    assert sample["source_file"] == str(log_dir / "run_4321_flamesam0.mon")

    dilution = series["psi_temperature/flamedil0/DIL_T_mix"]
    assert dilution["mean"] == pytest.approx(0.032)
    assert dilution["role"] == "sample_temperature"
    assert dilution["primary"] is False

    variox = series["psi_temperature/variox0/Variox"]
    assert variox["mean"] == pytest.approx(2.31)
    assert variox["time"] == pytest.approx([0.0, 86410.0])
    assert variox["role"] == "sample_temperature"
    assert variox["primary"] is False
    assert "role" not in series["psi_temperature/variox0/Sample"]

    log_metadata = ds.metadata["psi_temperature_log"]
    assert log_metadata["source_file"] == str(log_dir / "run_4321_flamesam0.mon")
    assert log_metadata["source_files"] == [
        str(log_dir / "run_4321_flamedil0.mon"),
        str(log_dir / "run_4321_flamesam0.mon"),
        str(log_dir / "run_4321_variox0.mon"),
    ]
    assert log_metadata["channels"] == [
        "flamedil0/DIL_T_hx",
        "flamedil0/DIL_T_mix",
        "flamedil0/DIL_T_sorb",
        "flamedil0/DIL_T_still",
        "flamesam0/SAM_ts",
        "flamesam0/SAM_ts_high",
        "flamesam0/SAM_ts_low",
        "variox0/Sample",
        "variox0/Variox",
    ]


def test_load_psi_bin_reads_backslash_title_temperature_mon_file(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    mon_path = tmp_path / "run_4321_templs0.mon"
    _write_psi_bin(path)
    _write_psi_mon_with_backslash_titles(mon_path)

    ds = PsiLoader().load(str(path))

    series = ds.metadata["nexus_time_series"]
    assert series["psi_temperature/Temp_Cryostat"]["values"] == pytest.approx([4.9906, 5.0])
    assert series["psi_temperature/Temp_Sample"]["values"] == pytest.approx([5.1805, 5.2])


def test_load_psi_mdu_t5(tmp_path) -> None:
    path = tmp_path / "tdc_hifi_2468.mdu"
    _write_psi_mdu(path)

    ds = load(str(path))

    assert ds.run is not None
    assert ds.run_number == 2468
    assert ds.metadata["psi_format"] == "psi-mdu"
    assert ds.metadata["instrument"] == "HIFI"
    assert ds.metadata["field_direction"] == "Transverse"
    assert len(ds.run.histograms) == 2
    assert ds.run.grouping["groups"] == {1: [1], 2: [2]}
    assert ds.run.grouping["group_names"] == {1: "F1", 2: "B1"}
    # Positron detectors are included by default.
    assert ds.run.grouping["included_groups"] == {1: True, 2: True}
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1
    assert ds.run.grouping["t0_bin"] == 1
    assert ds.run.grouping["first_good_bin"] == 2
    assert ds.run.grouping["last_good_bin"] == 5
    assert ds.n_points == 4


def test_non_positron_label_classifier() -> None:
    loader = PsiLoader()
    for label in ("MV", "mv", "FV", "BV", "M1", "M2", "Veto"):
        assert loader._is_non_positron_label(label), label
    for label in ("F1", "B8", "Forw", "Back", "Left", "Right", "R_F"):
        assert not loader._is_non_positron_label(label), label


def test_default_groups_hal_forward_only_ring_excludes_mv_veto() -> None:
    """HAL ``.mdu`` shipping only the forward ring (``MV, F1..F8``) must not pair against MV.

    Regression for the high-TF AFM default-grouping bug: forward=F1 was paired
    against backward=MV (the muon veto, ~0 counts in the good region), pinning
    the F-B asymmetry at +100% with no usable signal. The default must select
    two opposed *positron* detectors instead — F1 vs F5 (180° apart) for the
    eight-detector ring.
    """
    loader = PsiLoader()
    labels = ["MV", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    groups, names, forward_gid, backward_gid = loader._default_groups(labels, len(labels))

    mv_gid = next(gid for gid, name in names.items() if name == "MV")
    assert forward_gid != backward_gid
    assert mv_gid not in (forward_gid, backward_gid)
    assert not loader._is_non_positron_label(names[forward_gid])
    assert not loader._is_non_positron_label(names[backward_gid])
    assert {names[forward_gid], names[backward_gid]} == {"F1", "F5"}
    # The MV histogram is still present as its own (excluded) group.
    assert groups[mv_gid] == [1]


def test_default_groups_hal_full_layout_pairs_forward_and_backward_rings() -> None:
    """The full HAL layout (``MV, F1..F8, B1..B8``) pairs a forward vs a backward ring detector."""
    loader = PsiLoader()
    labels = ["MV"] + [f"F{i}" for i in range(1, 9)] + [f"B{i}" for i in range(1, 9)]
    _groups, names, forward_gid, backward_gid = loader._default_groups(labels, len(labels))

    mv_gid = next(gid for gid, name in names.items() if name == "MV")
    assert mv_gid not in (forward_gid, backward_gid)
    assert forward_gid != backward_gid
    # One half from the forward ring (F#), the other from the backward ring (B#).
    assert {names[forward_gid][0], names[backward_gid][0]} == {"F", "B"}


def test_load_psi_bin_preserves_individual_labeled_groups(tmp_path) -> None:
    path = tmp_path / "deltat_pta_gpd_4321.bin"
    counts = np.vstack([np.arange(12, dtype=np.int32) + i for i in range(6)])
    _write_psi_bin(
        path,
        labels=[b"Forw", b"Back", b"Forw", b"Back", b"Forw", b"Back"],
        counts=counts,
    )

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run.grouping["groups"] == {
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
    }
    assert ds.run.grouping["group_names"] == {
        1: "Forw",
        2: "Back",
        3: "Forw 2",
        4: "Back 2",
        5: "Forw 3",
        6: "Back 3",
    }
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1


def test_musrfit_bin_fixture_matches_musrfit_psi_reader_dump() -> None:
    """Compare a musrfit PSI-BIN example with musrfit's MuSR_td_PSI_bin reader."""
    path = _musrfit_example_file("deltat_pta_gpd_0423.bin")

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 423
    assert ds.metadata["sample"] == "FeSe"
    assert ds.metadata["temperature"] == pytest.approx(5.0)
    assert ds.metadata["field"] == pytest.approx(100.0)
    assert ds.metadata["comment"] == "FeSe 9p4 TF100 p107apr09_sample*1p02"
    assert ds.metadata["field_direction"] == "Transverse"  # from the "TF100" tag
    assert len(ds.run.histograms) == 2
    assert [h.n_bins for h in ds.run.histograms] == [8192, 8192]
    assert ds.run.histograms[0].bin_width == pytest.approx(0.00125)
    assert ds.run.grouping["histogram_labels"] == ["Back", "Forw"]
    assert ds.run.grouping["detector_t0_bins"] == [160, 200]
    assert ds.run.grouping["detector_first_good_bins"] == [165, 205]
    # musrfit reports raw BIN last-good values [8192, 8192]; Asymmetry stores
    # valid zero-based inclusive indices for use with NumPy arrays.
    assert ds.run.grouping["detector_last_good_bins"] == [8191, 8191]
    assert [float(np.sum(h.counts)) for h in ds.run.histograms] == pytest.approx(
        [3014024.0, 3223863.0]
    )


def test_musrfit_bin_multihistogram_fixture_matches_musrfit_psi_reader_dump() -> None:
    """Compare a five-histogram musrfit PSI-BIN example against musrfit output."""
    path = _musrfit_example_file("deltat_pta_gps_3110.bin")

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 3110
    assert ds.metadata["sample"] == "Y-1248"
    assert ds.metadata["temperature"] == pytest.approx(4.5)
    assert ds.metadata["field"] == pytest.approx(150.0)
    assert ds.metadata["comment"] == "Y124 TF150G 4.5K (ab)"
    assert ds.metadata["field_direction"] == "Transverse"  # from the "TF150G" tag
    assert len(ds.run.histograms) == 5
    assert ds.run.histograms[0].bin_width == pytest.approx(0.000625)
    assert ds.run.grouping["histogram_labels"] == ["Forw", "Back", "Up", "Down", "Righ"]
    assert ds.run.grouping["detector_t0_bins"] == [256, 251, 249, 249, 250]
    assert ds.run.grouping["detector_first_good_bins"] == [264, 259, 257, 257, 258]
    assert ds.run.grouping["detector_last_good_bins"] == [8100, 8100, 8191, 8191, 8191]
    assert [float(np.sum(h.counts)) for h in ds.run.histograms] == pytest.approx(
        [2482848.0, 6770250.0, 10047817.0, 9768739.0, 1462619.0]
    )


def test_musrfit_mdu_fixture_matches_musrfit_psi_reader_dump() -> None:
    """Compare the musrfit PSI-MDU example with musrfit's MuSR_td_PSI_bin reader."""
    path = _musrfit_example_file("tdc_hifi_2014_00153.mdu")

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 153
    assert ds.metadata["psi_format"] == "psi-mdu"
    assert ds.metadata["sample"] == "MnSi"
    assert ds.metadata["temperature"] == pytest.approx(50.0)
    assert ds.metadata["field"] == pytest.approx(75000.0)
    assert ds.metadata["comment"] == "MnSi, FLC68.2, 50 K"
    assert len(ds.run.histograms) == 17
    assert ds.run.histograms[0].n_bins == 409600
    assert ds.run.histograms[0].bin_width == pytest.approx(2.44140625e-05)
    assert ds.run.grouping["histogram_labels"] == [
        "MV",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
    ]
    assert ds.run.grouping["detector_t0_bins"] == [
        20097,
        20097,
        20076,
        20045,
        20061,
        20071,
        20056,
        20057,
        20022,
        20050,
        20048,
        20034,
        20036,
        20043,
        20043,
        20041,
        20061,
    ]
    assert ds.run.grouping["detector_first_good_bins"] == [
        20107,
        20107,
        20086,
        20055,
        20071,
        20081,
        20066,
        20067,
        20032,
        20060,
        20058,
        20044,
        20046,
        20053,
        20053,
        20051,
        20071,
    ]
    assert ds.run.grouping["detector_last_good_bins"] == [409190] * 17
    # The muon-veto MV (group 1) is excluded by default; F1-8/B1-8 are included.
    included = ds.run.grouping["included_groups"]
    assert included[1] is False
    assert all(included[gid] for gid in range(2, 18))
    assert [float(np.sum(h.counts)) for h in ds.run.histograms] == pytest.approx(
        [
            1006775.0,
            2055086.0,
            2104929.0,
            2241303.0,
            2251635.0,
            2194705.0,
            1867666.0,
            1878720.0,
            1907234.0,
            2593412.0,
            2612373.0,
            2803099.0,
            3029049.0,
            3048043.0,
            3007903.0,
            2990259.0,
            2750281.0,
        ]
    )


def test_hal_layout_matches_musrfit_mdu_histogram_order() -> None:
    """The HAL-9500 layout's detector labels match the file's histogram order.

    Detector IDs map positionally to histogram indices (detector N ->
    histogram N-1), so the layout labels must line up with the file's
    ``MV, F1..F8, B1..B8`` ordering for grouping to select the right channels.
    """
    from asymmetry.core.instrument import detect_instrument, get_instrument_layout

    path = _musrfit_example_file("tdc_hifi_2014_00153.mdu")

    ds = PsiLoader().load(str(path))
    n_hist = len(ds.run.histograms)
    file_labels = ds.run.grouping["histogram_labels"]

    assert detect_instrument(n_hist, metadata=ds.metadata, source_file=str(path)) == "HAL"

    layout = get_instrument_layout("HAL")
    by_id = {s.detector_id: s for s in layout.all_segments}
    assert layout.n_detectors == n_hist
    # Layout detector N labels histogram index N-1.
    for hist_index, file_label in enumerate(file_labels):
        assert by_id[hist_index + 1].label == file_label


def test_aligned_grouping_preserves_per_detector_t0() -> None:
    histograms = [
        Histogram(np.array([1, 2, 3, 4], dtype=float), bin_width=0.1, t0_bin=1),
        Histogram(np.array([10, 20, 30, 40], dtype=float), bin_width=0.1, t0_bin=2),
    ]
    common_t0 = common_t0_for_groups(histograms, [0], [1])

    assert common_t0 == 2
    assert apply_grouping_aligned(histograms, [0], common_t0_bin=common_t0) == pytest.approx(
        [0, 1, 2, 3, 4]
    )
    assert apply_grouping_aligned(histograms, [1], common_t0_bin=common_t0) == pytest.approx(
        [10, 20, 30, 40]
    )


def test_core_deadtime_formula_uses_good_frames() -> None:
    corrected = apply_deadtime_correction(
        np.array([100.0]),
        tau_us=0.01,
        bin_width_us=0.02,
        num_good_frames=1000.0,
    )
    expected = 100.0 / (1.0 - (100.0 * 0.01 / (0.02 * 1000.0)))
    assert corrected[0] == pytest.approx(expected)


def test_prepare_deadtime_does_not_estimate_when_file_values_are_absent() -> None:
    observed = np.array([100.0, 95.0, 90.0], dtype=float)
    grouping = {}

    corrected, applied = prepare_histograms_with_deadtime(
        [
            Histogram(
                observed.astype(float),
                bin_width=0.02,
                t0_bin=0,
                good_bin_start=0,
                good_bin_end=2,
            )
        ],
        grouping,
        use_deadtime=True,
    )

    assert applied is False
    assert "deadtime_method" not in grouping
    assert "estimated_dead_time_factors" not in grouping
    assert corrected[0].counts == pytest.approx(observed)
