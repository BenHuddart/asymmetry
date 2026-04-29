"""Tests for PSI BIN/MDU raw histogram loading."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.io import load
from asymmetry.core.io.psi import _FE_HEADER, PsiLoader
from asymmetry.core.transform import (
    apply_deadtime_correction,
    apply_grouping_aligned,
    common_t0_for_groups,
    prepare_histograms_with_deadtime,
)

MUSRFIT_EXAMPLE_DATA = Path("/Users/bhuddart/Source/musrfit/doc/examples/data")


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
    assert ds.run.grouping["groups"] == {1: [1], 2: [2]}
    assert ds.run.grouping["group_names"] == {1: "Back", 2: "Forw"}
    assert ds.run.grouping["forward_group"] == 2
    assert ds.run.grouping["backward_group"] == 1
    assert ds.run.grouping["detector_t0_bins"] == [1, 3]
    assert ds.run.grouping["t0_bin"] == 3
    assert ds.n_points == 1
    assert ds.time[0] == pytest.approx(0.02)


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
    assert len(ds.run.histograms) == 2
    assert ds.run.grouping["groups"] == {1: [1], 2: [2]}
    assert ds.run.grouping["group_names"] == {1: "F1", 2: "B1"}
    assert ds.run.grouping["forward_group"] == 1
    assert ds.run.grouping["backward_group"] == 2
    assert ds.run.grouping["t0_bin"] == 1
    assert ds.run.grouping["first_good_bin"] == 2
    assert ds.run.grouping["last_good_bin"] == 5
    assert ds.n_points == 4


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
    assert ds.run.grouping["forward_group"] == 1
    assert ds.run.grouping["backward_group"] == 2


def test_musrfit_bin_fixture_matches_musrfit_psi_reader_dump() -> None:
    """Compare a musrfit PSI-BIN example with musrfit's MuSR_td_PSI_bin reader."""
    path = MUSRFIT_EXAMPLE_DATA / "deltat_pta_gpd_0423.bin"
    if not path.exists():
        pytest.skip("musrfit PSI-BIN fixture not available")

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 423
    assert ds.metadata["sample"] == "FeSe"
    assert ds.metadata["temperature"] == pytest.approx(5.0)
    assert ds.metadata["field"] == pytest.approx(100.0)
    assert ds.metadata["comment"] == "FeSe 9p4 TF100 p107apr09_sample*1p02"
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
    path = MUSRFIT_EXAMPLE_DATA / "deltat_pta_gps_3110.bin"
    if not path.exists():
        pytest.skip("musrfit PSI-BIN fixture not available")

    ds = PsiLoader().load(str(path))

    assert ds.run is not None
    assert ds.run_number == 3110
    assert ds.metadata["sample"] == "Y-1248"
    assert ds.metadata["temperature"] == pytest.approx(4.5)
    assert ds.metadata["field"] == pytest.approx(150.0)
    assert ds.metadata["comment"] == "Y124 TF150G 4.5K (ab)"
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
    path = MUSRFIT_EXAMPLE_DATA / "tdc_hifi_2014_00153.mdu"
    if not path.exists():
        pytest.skip("musrfit PSI-MDU fixture not available")

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
