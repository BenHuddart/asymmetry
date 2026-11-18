"""Independent SI equivalence checks for permissive NeXus metadata."""

import pytest

from asymmetry.core.io.nexus import (
    _normalize_field_to_gauss,
    _normalize_temperature_to_kelvin,
)


@pytest.mark.parametrize("value,unit,expected", [(100, "uT", 1), (0.5, "kG", 500)])
def test_field_prefixes(value, unit, expected):
    assert _normalize_field_to_gauss(value, unit) == pytest.approx(expected)


def test_millikelvin():
    assert _normalize_temperature_to_kelvin(300, "mK") == pytest.approx(0.3)


@pytest.mark.parametrize("unit", [None, "", "minutes", "unrecognised"])
def test_compatibility_units_keep_numeric_value(unit):
    assert _normalize_temperature_to_kelvin(20, unit) == 20
    assert _normalize_field_to_gauss(100, unit) == 100


def test_loaded_units_report_preserves_log_declarations(tmp_path):
    from asymmetry.core.io import load, unit_report
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "units.nxs"
    _write_v2_file(
        path,
        temp_setpoint=300,
        temp_setpoint_units="mK",
        temp_log_values=(300, 400, 500),
        temp_log_units="mK",
        magnetic_field=0.5,
        magnetic_field_units="kG",
    )
    ds = load(str(path))
    assert ds.temperature == pytest.approx(0.3)
    assert ds.field == pytest.approx(500)
    assert ds.sample_temperature_logged == pytest.approx(0.4)
    rows = {row.path: row for row in unit_report(ds)}
    assert rows["/sample/temperature"].interpreted_value == pytest.approx(0.3)
    log_path = ds.metadata["sample_temperature_log_source"]
    assert rows["/" + log_path + "/value"].declared_unit == "mK"
    assert "/" + log_path + "/time" in rows
    assert ds.metadata["nexus_time_series"][log_path]["values"] == [300, 400, 500]
    del ds.metadata["nexus_unit_policy"]
    assert all(row.interpretation == "unrecorded" for row in unit_report(ds))


def test_info_units_json(tmp_path, capsys):
    import json

    from asymmetry import cli
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "units.nxs"
    _write_v2_file(path, magnetic_field_units="K")
    cli.main(["info", str(path), "--units", "--json"])
    result = json.loads(capsys.readouterr().out)
    field = next(row for row in result["units"] if row["path"] == "/sample/magnetic_field")
    assert field["interpretation"] == "compatibility"
    assert field["declared_unit"] == "K"
    assert field["interpreted_value"] == 150


def test_singleton_unit_attribute_matches_report(tmp_path):
    import h5py
    import numpy as np

    from asymmetry.core.io import load, unit_report
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "array.nxs"
    _write_v2_file(path, temp_setpoint=300)
    with h5py.File(path, "r+") as file:
        file["raw_data_1/sample/temperature"].attrs["units"] = np.array([b"mK"])
    ds = load(str(path))
    row = next(row for row in unit_report(ds) if row.path == "/sample/temperature")
    assert ds.temperature == row.interpreted_value == pytest.approx(0.3)


def test_report_does_not_reject_uninterpretable_stored_header(tmp_path):
    import h5py
    import numpy as np

    from asymmetry.core.io import load, unit_report
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "text.nxs"
    _write_v2_file(path)
    with h5py.File(path, "r+") as file:
        sample = file["raw_data_1/sample"]
        del sample["temperature"]
        sample.create_dataset("temperature", data=np.array([b"unknown"]))
    ds = load(str(path))
    row = next(row for row in unit_report(ds) if row.path == "/sample/temperature")
    assert row.original_value == ["unknown"]
    assert row.interpreted_value is None


def test_singleton_blank_declaration_is_visible(tmp_path):
    import h5py
    import numpy as np

    from asymmetry.core.io import load, unit_report
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "blank.nxs"
    _write_v2_file(path)
    with h5py.File(path, "r+") as file:
        file["raw_data_1/sample/temperature"].attrs["units"] = np.array([b""])
    row = next(row for row in unit_report(load(str(path))) if row.path == "/sample/temperature")
    assert row.declaration == "Unit declaration blank"


def test_long_log_numeric_summary_converts_without_materialising_samples(tmp_path):
    from asymmetry.core.io import load, unit_report
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "long.nxs"
    _write_v2_file(
        path,
        temp_log_values=tuple(range(300, 400)),
        temp_log_times=tuple(range(100)),
        temp_log_units="mK",
    )
    ds = load(str(path))
    row = next(row for row in unit_report(ds) if row.path.endswith("/Temp_Sample/value"))
    assert row.interpreted_value["mean"] == pytest.approx(0.3495)
    assert row.interpreted_value["min"] == pytest.approx(0.3)
    assert row.interpreted_value["max"] == pytest.approx(0.399)


def test_main_sweep_and_geometry_convert_independent_coil_units(tmp_path):
    import h5py

    from asymmetry.core.io import load
    from asymmetry.core.workflow.survey import coil_geometry
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "coils.nxs"
    _write_v2_file(path, magnetic_field=150, magnetic_field_units="G")
    with h5py.File(path, "r+") as file:
        entry = file["raw_data_1"]
        for name, value, unit in [
            ("Field_Main", 1, "T"),
            ("Field_Z", 0.015, "T"),
            ("Field_X", 100, "G"),
            ("Field_Y", 100, "G"),
        ]:
            log = entry.create_group(name)
            log.create_dataset("time", data=[0.0, 1.0])
            log.create_dataset("values", data=[value, value]).attrs["units"] = unit
    ds = load(str(path))
    assert ds.field == pytest.approx(10150)
    assert coil_geometry(ds.metadata) == "LF"


def test_strict_selection_still_accepts_singleton_log_unit(tmp_path):
    import h5py
    import numpy as np

    from asymmetry.core.io import load
    from asymmetry.core.workflow import select_logged_temperature
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "selected-array.nxs"
    _write_v2_file(path, temp_log_values=(300, 400, 500))
    with h5py.File(path, "r+") as file:
        file["raw_data_1/sample/Temp_Sample/value"].attrs["units"] = np.array([b"mK"])
    ds = load(str(path))
    selected = select_logged_temperature(ds, ds.metadata["sample_temperature_log_source"])
    assert selected.temperature == pytest.approx(0.4)
