"""Read owned synthetic, recorded-execution cases at the fixture boundary."""

import hashlib
import json
import math
import re
from pathlib import Path


def load_reference_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    assert type(case["format_version"]) is int and case["format_version"] == 1, (
        "unsupported reference-case format"
    )
    assert case["evidence_class"] == "recorded_native_execution", "wrong evidence class"
    for key in ("case_id", "operation", "anchor", "limits"):
        assert isinstance(case[key], str) and case[key].strip(), f"missing {key}"
    for key in ("identity", "inputs", "settings", "mapping", "native_output", "tolerances"):
        assert isinstance(case[key], dict) and case[key], f"missing {key}"
    for reference in case["identity"].values():
        for key in ("version", "build", "executable_sha256"):
            assert isinstance(reference[key], str) and reference[key], f"missing identity {key}"
        assert re.fullmatch(r"[0-9a-f]{64}", reference["executable_sha256"]), (
            "invalid executable hash"
        )
    assert isinstance(case["mapping"]["description"], str) and case["mapping"]["description"]
    for name, digest in case["inputs"].items():
        assert isinstance(name, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid input hash"
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts, "unsafe input path"
        assert hashlib.sha256((path.parent / relative).read_bytes()).hexdigest() == digest, (
            f"input hash mismatch: {name}"
        )
    for tolerance in case["tolerances"].values():
        assert isinstance(tolerance["reason"], str) and tolerance["reason"], (
            "missing precision rationale"
        )
        for key in ("atol", "rtol"):
            value = tolerance[key]
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0, (
                f"invalid tolerance {key}"
            )
    assert case["native_status"], "missing native execution status"
    # Strict JSON transport must preserve evidence class and finite numerical values.
    assert json.loads(json.dumps(case, allow_nan=False)) == case
    settings, mapping = case["settings"], case["mapping"]
    if case["operation"] == "weighted_constant_fit":
        assert set(case["inputs"]) == {"owned.dat", "owned.msr", "musrfit_startup.xml"}, (
            "incomplete consumed constant input/hash set"
        )
        rows = [
            [float(value) for value in line.split(",")]
            for line in (path.parent / "owned.dat").read_text().splitlines()[3:]
        ]
        assert rows == list(
            map(list, zip(settings["x"], settings["y"], settings["sigma"], strict=True))
        ), "constant observations differ from native input"
        assert settings["selected_indices"] == [0, 1, 2], (
            "constant selection must retain all three points"
        )
        assert settings["packing"] == 1 and settings["units"] == "dimensionless"
        assert settings["musrfit"]["fit_range"] == [-0.5, 2.5]
        assert settings["musrfit"]["type"] == 8
        assert settings["musrfit"]["commands"] == ["MIGRAD", "HESSE", "SAVE"]
        assert settings["musrfit"]["threads"] == 1
        assert (
            settings["objective"]
            == "sum(((y-a)/sigma)^2); supplied absolute errors; no residual rescaling"
        ), "incompatible recorded fit objective/error convention"
        assert settings["Mantid"] == {
            "function": "FlatBackground,A0=3",
            "minimizer": "Levenberg-Marquardt",
            "cost_function": "Least squares",
            "StartX": 0,
            "EndX": 2,
            "EvaluationType": "CentrePoint",
            "MaxIterations": 100,
        }, "incompatible recorded Mantid selection/objective"
        assert mapping["value_scale"] == mapping["standard_error_scale"] == 1, (
            "invalid constant scale"
        )
        assert mapping["Mantid_objective"] == "chi_squared_per_ndf * ndf"
        request = (path.parent / "owned.msr").read_text()
        assert all(
            line in request.splitlines()
            for line in (
                "fittype 8",
                "fit -0.5 2.5",
                "packing 1",
                "MIGRAD",
                "HESSE",
                "SAVE",
            )
        ), "constant request differs from settings"
    elif case["operation"] == "fb_reduction_and_count_packing":
        assert set(case["inputs"]) == {
            "synthetic.bin",
            "legacy-lem.root",
            "musrroot-folder.root",
            "musrroot-directory.root",
            "sum-1.msr",
            "fb-1.msr",
            "sum-3.msr",
            "sum-4.msr",
            "musrfit_startup.xml",
        }, "incomplete consumed PSI input/hash set"
        assert settings["selected_bins"] == {
            "forward": [20, 60],
            "backward": [23, 63],
            "upper": "exclusive",
        }
        assert settings["t0"] == [4, 7] and mapping["forward_t0_bin"] == 4
        assert settings["grouping"] == {"sum": [1, 2], "forward": [1], "backward": [2]}
        assert settings["alpha"] == settings["beta"] == 1
        assert settings["corrections"] == {"background": 0, "deadtime": False}
        assert mapping["asymmetry_scale_to_percent"] == 100 and mapping["time_unit"] == "us"
        assert settings["packing"] == [1, 3, 4]
        assert mapping["packed_native_groups"] == {"1": 40, "3": 13, "4": 9}
        for asymmetry, packing, name in (
            (False, 1, "sum-1.msr"),
            (True, 1, "fb-1.msr"),
            (False, 3, "sum-3.msr"),
            (False, 4, "sum-4.msr"),
        ):
            request = (path.parent / name).read_text().splitlines()
            assert all(
                line in request
                for line in (
                    "t0 4 7",
                    f"packing {packing}",
                    f"fittype {2 if asymmetry else 0}",
                    "data 20 60 23 63" if asymmetry else "data 20 60",
                    "backgr.fix 0 0" if asymmetry else "3 bg 0 0 none",
                    "forward 1" if asymmetry else "forward 1 2",
                )
            ), "incompatible recorded PSI selection/packing/corrections"
    elif case["operation"] == "manual_phase_fft":
        assert set(case["inputs"]) == {
            "synthetic.nxs",
            "phase0.msr",
            "phase37.msr",
            "musrfit_startup.xml",
            "export_canvas.C",
        }, "incomplete consumed phase input/hash set"
        assert settings["phase_degrees"] == [0, 37], "phase pair differs from native requests"
        assert mapping["rotation_sign"] == -1 and mapping["phase_unit"] == "degree"
        assert mapping["normalization_scale"] == 1
        assert mapping["rotated_bins"] == [0, 494] and mapping["unrotated_bins"] == [494]
        zero = (path.parent / "phase0.msr").read_text()
        phased = (path.parent / "phase37.msr").read_text()
        assert phased == zero.replace("phase 0", "phase 37"), "native requests differ beyond phase"
        assert settings["packing"] == 1 and settings["t0"] == 0
        assert settings["grouping"] == {"forward": [1]}
        assert settings["corrections"]["background"] == 200000
        assert settings["corrections"]["deadtime"] is False
        assert settings["window"] == "none"
        assert all(
            line in zero.splitlines()
            for line in (
                "data 3 993",
                "t0 0",
                "packing 1",
                "apodization none",
                "backgr.fit 2",
                "2 bg 200000 0 none",
                "deadtime-cor no",
                "range 0.03 9.93",
                "phase 0",
            )
        ), "phase request differs from settings"
    else:
        raise AssertionError(f"unsupported recorded operation: {case['operation']}")
    return case
