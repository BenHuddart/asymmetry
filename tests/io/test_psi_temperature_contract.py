"""PSI monitor text and filename boundary contracts."""

from pathlib import Path

import pytest

from asymmetry.core.io.psi import PsiLoader


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("gps", ("GPS", "piM3.2", "continuous surface muon source")),
        ("ltf", ("LTF", "piM3.3", "continuous surface muon source")),
        ("gpd", ("GPD", "muE1", "continuous decay channel muon source")),
        ("dolly", ("DOLLY", "piE1", "continuous surface muon source")),
        ("alc", ("ALC", "piE3", "continuous surface muon source")),
        ("hifi", ("HIFI", "piE3", "continuous surface muon source")),
    ],
)
@pytest.mark.parametrize("form", ["{}_1", "1_{}", "x_{}_1", "{}123", "123{}", "{}123x"])
def test_instrument_filename_tokens(token, expected, form) -> None:
    assert PsiLoader()._guess_psi_instrument(Path(form.format(token.upper()) + ".bin")) == expected


@pytest.mark.parametrize("name", ["gps", "xgpsx", "gps-1", "x123gpsx", "hifi"])
def test_instrument_unknown_filename(name) -> None:
    assert PsiLoader()._guess_psi_instrument(Path("/gps/" + name + ".mdu")) == (
        "PSI",
        "n/a",
        "continuous muon source",
    )


def test_instrument_flame_substring_has_priority() -> None:
    assert PsiLoader()._guess_psi_instrument(Path("gps_123_inflamed.bin")) == (
        "FLAME",
        "PSI",
        "continuous muon source",
    )
    assert PsiLoader()._guess_psi_instrument(Path("gpd_123_gps.bin"))[0] == "GPS"
    assert PsiLoader()._guess_psi_instrument(Path("gpd123gps.bin"))[0] == "GPS"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", ""),
        ("! 01-JAN-95 1:99:99", "2095-01-01T1:99:99"),
        ("! 31-FOO-26 01:02:03", "2026-01-31T01:02:03"),
        ("! 2026-99-99 99:99:99", "2026-99-99T99:99:99"),
        ("! 2026-1-2 3:04:05", "2026-01-02T3:04:05"),
        ("! 1-Jan-069 03:04:05", "2069-01-01T03:04:05"),
    ],
)
def test_monitor_start_text(line, expected) -> None:
    assert PsiLoader()._parse_temperature_start_time(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("! 01-JAN-26 01:02:03", True),
        ("! 1-JAN-26 01:02:03", False),
        ("abcdeJAN", True),
        ("! 2026-1-2 3:04:05", True),
        ("", False),
    ],
)
def test_monitor_date_recognition(line, expected) -> None:
    assert PsiLoader()._looks_like_psi_temperature_date(line) is expected


def test_monitor_header_fields() -> None:
    loader = PsiLoader()
    assert (
        loader._parse_temperature_equipment("! Equipment name: controller flamesam0") == "flamesam0"
    )
    assert loader._parse_temperature_equipment("! equipment name: other") == ""
    assert loader._parse_temperature_equipment("! Equipment name: ") == ""
    assert loader._parse_temperature_title_groups("! Title: \\ A B \\ \\ C") == [["A", "B"], ["C"]]
    assert loader._parse_temperature_title_groups("no colon") == []


@pytest.mark.parametrize(
    ("line", "groups", "equipment", "expected"),
    [
        (
            "00:00:01\\2\\1 2\\1 2\\",
            [["A", "B"], ["A", "B"]],
            "",
            [("Temp_A", 1.0, 1.0), ("Temp_B", 1.0, 2.0)],
        ),
        (
            "1 00:00:01\\2\\1 bad 3\\4 5\\",
            [["A", "B"], ["A", "C"]],
            "sensor",
            [("A", 86401.0, 1.0), ("B", 86401.0, 3.0), ("C", 86401.0, 5.0)],
        ),
        (
            "00:00:00\\2\\1 2\\3 4\\5 6\\",
            [["A", "B"], ["A", "B"], ["C", "D"]],
            "",
            [
                ("Temp_A", 0.0, 1.0),
                ("Temp_B", 0.0, 2.0),
                ("Temp_C", 0.0, 5.0),
                ("Temp_D", 0.0, 6.0),
            ],
        ),
        (
            "00:00:00\\2\\1 2\\1 2\\5 6\\",
            [["A", "B"], ["A", "B"], ["C", "D"]],
            "",
            [("Temp_A", 0.0, 1.0), ("Temp_B", 0.0, 2.0)],
        ),
        ("99:99:99suffix\\2\\1 2\\", [["NONE", "Sample"]], "", [("Temp_Sample", 362439.0, 2.0)]),
        ("00:00:00\\-1\\1 2\\", [["A", "B"]], "", []),
        (
            "00:00:00\\2\\1 2\\3 4",
            [["A", "B"], ["C", "D"]],
            "",
            [("Temp_A", 0.0, 1.0), ("Temp_B", 0.0, 2.0)],
        ),
    ],
)
def test_monitor_data_records(line, groups, equipment, expected) -> None:
    assert PsiLoader()._parse_temperature_data_line(line, groups, equipment) == expected


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("00:00:00 1 2", "not backslash-delimited"),
        ("bad\\1\\2\\", "Invalid PSI temperature log time"),
        ("00:00:00\\bad\\2\\", "invalid literal for int"),
    ],
)
def test_monitor_record_errors(line, message) -> None:
    with pytest.raises(ValueError, match=message):
        PsiLoader()._parse_temperature_data_line(line, [["Sample"]])


def test_monitor_search_depth_symlink_and_sorting(tmp_path) -> None:
    candidates = ["Z/run_0007.mon", "a/b/c/run_7.MON", "a/b/c/d/run_7.mon", "a/run_17.mon"]
    for name in candidates:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    (tmp_path / "Z" / "loop").symlink_to(tmp_path, target_is_directory=True)
    result = PsiLoader()._find_temperature_log_files_recursively(tmp_path / "gps7.bin", 7)
    assert result == [tmp_path / "a/b/c/run_7.MON", tmp_path / "Z/run_0007.mon"]


def test_monitor_file_metadata_and_header_boundary(tmp_path) -> None:
    path = tmp_path / "run_7.mon"
    path.write_text(
        "! Equipment name: crate flamesam0\n! Title: SAM_ts NONE\n! 01-JAN-26 01:02:03\n00:00:01\\2\\1 999\\\n! Title: ignored\n00:00:00\\2\\3 999\\\n",
        encoding="latin-1",
    )
    result = PsiLoader()._parse_temperature_log_file(path)
    assert result.start_time == "2026-01-01T01:02:03"
    assert result.channels == ["flamesam0/SAM_ts"]
    assert result.source_files == [str(path)]
    [series] = result.time_series.values()
    assert series == {
        "path": "psi_temperature/flamesam0/SAM_ts",
        "units": "K",
        "time": [1.0, 0.0],
        "values": [1.0, 3.0],
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
        "source_file": str(path),
        "source_format": "PSI .mon",
        "reader_provenance": "Mantid LoadPSIMuonBin-compatible",
        "equipment": "flamesam0",
        "role": "sample_temperature",
        "sensor": "SAM_ts_value",
        "primary": True,
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "does not define channel titles"),
        ("\n! Title: A\n", "does not define channel titles"),
        ("! Title: A\n", "does not contain numeric data"),
        ("! Title: NONE\n00:00:00\\1\\2\\\n", "does not contain numeric data"),
    ],
)
def test_monitor_file_errors(tmp_path, contents, message) -> None:
    path = tmp_path / "run_7.mon"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        PsiLoader()._parse_temperature_log_file(path)
