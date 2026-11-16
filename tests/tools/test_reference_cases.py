"""Malformed provenance must fail before a recorded value can be compared."""

import json
import shutil
from pathlib import Path

import pytest

from tests.reference_cases import load_reference_case

FIXTURE = Path(__file__).parents[1] / "fixtures/reference_cases/constant"


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", {}),
        ("mapping", {}),
        ("inputs", {}),
        ("native_status", ""),
        ("evidence_class", "fresh_execution"),
        ("format_version", 99),
    ],
)
def test_missing_or_misclassified_provenance_fails(tmp_path, field, value):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "reference.json"
    case = json.loads(path.read_text())
    case[field] = value
    path.write_text(json.dumps(case))
    with pytest.raises(AssertionError):
        load_reference_case(path)


@pytest.mark.parametrize("defect", ["input", "binary", "tolerance", "precision", "nonfinite"])
def test_invalid_reference_contract_fails(tmp_path, defect):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "reference.json"
    case = json.loads(path.read_text())
    if defect == "input":
        (tmp_path / "owned.dat").write_text("different observations")
    elif defect == "binary":
        case["identity"]["musrfit"]["executable_sha256"] = "z" * 64
    elif defect == "tolerance":
        case["tolerances"]["musrfit"]["atol"] = -1
    elif defect == "precision":
        case["tolerances"]["musrfit"]["reason"] = ""
    else:
        case["native_output"]["musrfit"]["value"] = float("nan")
    path.write_text(json.dumps(case))
    with pytest.raises((AssertionError, ValueError)):
        load_reference_case(path)


def test_serialized_case_retains_recorded_status_and_native_values():
    case = load_reference_case(FIXTURE / "reference.json")
    restored = json.loads(json.dumps(case, allow_nan=False))
    assert restored == case
    assert restored["evidence_class"] == "recorded_native_execution"
    assert restored["native_output"]["musrfit"]["value"] == 4.44444
    assert "Changes in function value are too small" in restored["native_status"]["Mantid"]


@pytest.mark.parametrize(
    "case_name,field,key,value",
    [
        ("constant", "mapping", "value_scale", 2),
        ("constant", "settings", "selected_indices", [0, 1]),
        ("manual-phase", "settings", "phase_degrees", [12, 37]),
        ("manual-phase", "mapping", "rotation_sign", 1),
        ("manual-phase", "mapping", "unrotated_bins", []),
        (
            "psi_root",
            "settings",
            "selected_bins",
            {"forward": [21, 60], "backward": [23, 63], "upper": "exclusive"},
        ),
        ("psi_root", "settings", "t0", [4, 8]),
    ],
)
def test_metadata_drift_from_consumed_observation_or_mapping_fails(
    tmp_path, case_name, field, key, value
):
    folder = (
        FIXTURE.parent / case_name if case_name != "psi_root" else FIXTURE.parents[1] / "psi_root"
    )
    shutil.copytree(folder, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "reference.json"
    case = json.loads(path.read_text())
    case[field][key] = value
    path.write_text(json.dumps(case))
    with pytest.raises(AssertionError):
        load_reference_case(path)


@pytest.mark.parametrize(
    "folder,missing",
    [
        ("reference_cases/constant", "owned.dat"),
        ("reference_cases/manual-phase", "synthetic.nxs"),
        ("psi_root", "synthetic.bin"),
    ],
)
def test_partial_missing_consumed_input_hash_fails(tmp_path, folder, missing):
    shutil.copytree(FIXTURE.parents[1] / folder, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "reference.json"
    case = json.loads(path.read_text())
    del case["inputs"][missing]
    path.write_text(json.dumps(case))
    with pytest.raises(AssertionError, match="incomplete consumed"):
        load_reference_case(path)
