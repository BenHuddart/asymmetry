"""Run delivered scientific acceptance on a fresh native runner without a checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

if __name__ == "__main__":
    sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import desktop_acceptance as desktop  # noqa: E402
import release_artifacts as artifacts  # noqa: E402

TOOLS = (
    "native_acceptance.py",
    "desktop_acceptance.py",
    "release_artifacts.py",
    "release_environment.py",
)
FIXTURES = ("counts.root", "counts.nxs")
NUMERICAL_FIELDS = (
    "time_us",
    "asymmetry_percent",
    "error_percent",
    "fit_percent",
    "fit_error_percent",
    "replay_fraction",
)


def prepare_packet(packet: Path, target: str, commit: str) -> None:
    """Use the reviewed build environment once; transport no application source or venv."""
    from acceptance_fixtures import write_fixtures

    identity = artifacts.expected_identity(target, commit)
    packet.mkdir(parents=True, exist_ok=False)
    write_fixtures(packet / "inputs")
    for name in TOOLS:
        shutil.copyfile(artifacts.ROOT / "tools" / name, packet / name)
    artifacts.write_json(
        packet / "expectation.json",
        {
            "identity": identity,
            "notice_sha256": artifacts.release.digest(artifacts.ROOT / "THIRD_PARTY_LICENSES.md"),
        },
    )
    environment = desktop.payload_environment()
    environment["PYTHONPATH"] = str(artifacts.ROOT / "src")
    git = shutil.which("git")
    if git is None:
        raise ValueError("source comparison requires the authoring Git executable")
    environment["PATH"] = os.pathsep.join((str(Path(git).parent), environment["PATH"]))
    if target == "windows-x64":
        environment["ASYMMETRY_HDF4_DLL_DIR"] = str(artifacts.ROOT / "build/hdf4-dlls")
    with tempfile.TemporaryDirectory(prefix="asymmetry-source-anchor-") as temporary:
        work = Path(temporary) / "invocation"
        try:
            outcome = desktop.run_acceptance(
                [sys.executable, "-B", "-m", "asymmetry.gui.app"],
                packet / "inputs",
                source_identity(identity),
                work,
                environment=environment,
            )
        except RuntimeError:
            if (work / "outcome.json").is_file():
                print(json.dumps(artifacts.read_json(work / "outcome.json"), allow_nan=False))
            raise
    artifacts.write_json(packet / "source-result.json", outcome["report"])


def source_identity(identity: dict) -> dict:
    """The existing source diagnostic distinguishes its entry from packaged targets."""
    return {
        "version": identity["version"],
        "source_commit": identity["source_commit"],
        "target": "source",
    }


def packet_expectation(packet: Path, identity: dict) -> dict:
    """Check bounded test inputs at the transport boundary, without scientific imports."""
    expected = artifacts.read_json(packet / "expectation.json")
    if expected["identity"] != identity:
        raise ValueError("scientific packet identity differs from expected source")
    allowed = {
        *TOOLS,
        "expectation.json",
        "source-result.json",
        "inputs/fixtures.json",
        *(f"inputs/{name}" for name in FIXTURES),
    }
    actual = set()
    for path in packet.rglob("*"):
        if path.is_symlink():
            raise ValueError("scientific packet must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(packet).as_posix())
    if actual != allowed:
        raise ValueError("scientific packet contains missing or unexpected files")
    fixtures = artifacts.read_json(packet / "inputs/fixtures.json")
    if set(fixtures) != set(FIXTURES) or any(
        artifacts.release.digest(packet / "inputs" / name) != fixtures[name] for name in FIXTURES
    ):
        raise ValueError("scientific fixture bytes differ from packet record")
    source = artifacts.read_json(packet / "source-result.json")
    if (
        source["identity"] != source_identity(identity)
        or source["mode"] != "source"
        or source["success"] is not True
        or source["stages"][-1:] != ["complete"]
        or source["fixtures"] != fixtures
        or set(source["results"]) != set(FIXTURES)
    ):
        raise ValueError("scientific source comparison is incomplete or stale")
    json.dumps(source, allow_nan=False)
    return expected


def verify_source_packet(packet: Path, target: str, commit: str) -> None:
    """Bind packet authority and executable tool bytes to the triggering checkout."""
    identity = artifacts.expected_identity(target, commit)
    expected = packet_expectation(packet, identity)
    if expected["notice_sha256"] != artifacts.release.digest(
        artifacts.ROOT / "THIRD_PARTY_LICENSES.md"
    ):
        raise ValueError("scientific packet notice differs from source")
    for name in TOOLS:
        if (packet / name).read_bytes() != (artifacts.ROOT / "tools" / name).read_bytes():
            raise ValueError("scientific packet tool bytes differ from source: " + name)


def clean_host(workspace: Path, runner_temp: Path) -> dict:
    """Require a fresh job, not a checkout with only its working directory changed."""
    for path in (
        workspace / ".git",
        workspace / "src",
        workspace / ".venv",
        runner_temp / "release-replay",
    ):
        if path.exists():
            raise ValueError("native acceptance host contains checkout or build environment")
    names = ("asymmetry", "numpy", "scipy", "iminuit", "uproot", "awkward", "pyhdf")
    if any(importlib.util.find_spec(name) is not None for name in names):
        raise ValueError("native acceptance driver can import developer scientific packages")
    return {
        "checkout": "absent",
        "build_venv": "absent",
        "driver_scientific_packages": "absent",
        "child_search_path": "OS runtime only; frozen scientific origins checked by application",
    }


def compare_results(source: dict, frozen: dict) -> None:
    """Compare the same finite observations and fit with the scenario's declared tolerances."""
    tolerance = source["tolerances"]
    for name in FIXTURES:
        for key in NUMERICAL_FIELDS:
            first, second = source["results"][name][key], frozen["results"][name][key]
            left = first if isinstance(first, list) else [first]
            right = second if isinstance(second, list) else [second]
            absolute = (
                tolerance["fit_atol_percent"] if key == "fit_percent" else tolerance["array_atol"]
            )
            relative = (
                tolerance["fit_error_rtol"]
                if key == "fit_error_percent"
                else tolerance["array_rtol"]
            )
            if len(left) != len(right) or any(
                not math.isfinite(a)
                or not math.isfinite(b)
                or not math.isclose(a, b, rel_tol=relative, abs_tol=absolute)
                for a, b in zip(left, right, strict=True)
            ):
                raise ValueError("source and frozen scientific anchor differ: " + name + ":" + key)
        if (
            source["results"][name]["effective_recipe"]
            != frozen["results"][name]["effective_recipe"]
        ):
            raise ValueError("source and frozen effective reduction recipe differ")


def omission_components(components: list, control: dict, target: str) -> dict:
    """Select actual removable components; archived Python readers are never rewritten."""
    required = {
        "resource": [
            item
            for item in components
            if item.get("sha256") == control["resource_sha256"]
            and Path(item["path"]).name == "logo_256x256.png"
            and item["kind"] == "resource"
        ],
        "backend": [
            item
            for item in components
            if item["path"] == control["origins"]["pyhdf._hdfext"].replace("\\", "/")
            and item["kind"] == "native_library"
        ],
        "native": [
            item
            for item in components
            if item["kind"] == "native_library"
            and Path(item["path"]).name.lower()
            == ("mfhdf.dll" if target == "windows-x64" else "libmfhdf.0.dylib")
        ],
    }
    if any(len(items) != 1 for items in required.values()):
        raise ValueError(
            "delivered layout has no unique native/resource/backend omission component"
        )
    return {name: items[0] for name, items in required.items()}


@contextmanager
def payload_copy(application, work, cases, name):
    """Keep copy cleanup and primary validation failures visible at the owned OS boundary."""
    directory = Path(tempfile.mkdtemp(prefix="payload-copy-", dir=work))
    clone = directory / application.name
    errors = []
    try:
        shutil.copytree(application, clone, symlinks=True)
        yield clone
    except Exception as exc:
        cases.setdefault(name, {})["copy_primary_failure"] = type(exc).__name__
        errors.append(exc)
    try:
        shutil.rmtree(directory)
        if directory.exists():
            raise OSError("owned payload copy remains after cleanup")
        cases.setdefault(name, {})["copy_cleanup"] = {"success": True}
    except OSError as exc:
        cases.setdefault(name, {})["copy_cleanup"] = {
            "success": False,
            "failure_type": type(exc).__name__,
        }
        errors.append(exc)
    cases.setdefault(name, {})["copy_removed"] = not directory.exists()
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("payload copy validation and cleanup failures", errors)


def execute_case(
    command, inputs, identity, work, cases, name, expected_stage=None, loader_component=None
):
    """Only an actual nonzero scientific failure can satisfy an omission control."""
    try:
        desktop.run_acceptance(command, inputs, identity, work)
    except RuntimeError:
        if expected_stage is None:
            raise
    finally:
        if (work / "outcome.json").is_file():
            cases.setdefault(name, {}).update(artifacts.read_json(work / "outcome.json"))
    outcome = cases[name]
    if expected_stage is None:
        if outcome["report"]["mode"] != "frozen":
            raise ValueError("native launcher did not execute a frozen application")
    else:
        if (
            outcome["success"]
            or type(outcome["exit_code"]) is not int
            or outcome["exit_code"] == 0
            or outcome["failure"] == "timeout"
        ):
            raise ValueError("omission did not produce a decisive nonzero child failure")
        if "report" in outcome:
            report = outcome["report"]
            if (
                outcome["failure"] != "scientific stage failed"
                or report["identity"] != identity
                or report["token"] != outcome["token"]
                or report["failure"]["stage"] != expected_stage
            ):
                raise ValueError("omission has no fresh expected scientific-stage failure")
        else:
            if loader_component is None or outcome["failure"] != "missing report":
                raise ValueError("omission has no valid scientific report or native loader failure")
            marker = None
            for filename in ("launcher-error.log", "launcher.log"):
                log = work / filename
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 16384))
                    diagnostic = stream.read().decode("utf-8", errors="replace")
                found = re.search(
                    r"Library not loaded|DLL load failed|cannot open shared object|dyld\[|Error loading Python DLL|Failed to load",
                    diagnostic,
                    re.IGNORECASE,
                )
                if found and loader_component.lower() in diagnostic.lower():
                    marker = found.group().lower()
                    break
            if marker is None:
                raise ValueError(
                    "missing report has no component-specific native loader diagnostic"
                )
            outcome["native_loader_failure"] = {"component": loader_component, "marker": marker}

    return outcome


def run_native(
    packet: Path, directory: Path, target: str, commit: str, version: str, output: Path
) -> None:
    """Use the final media on a clean native host and retain only explicit reports."""
    if output.exists():
        raise ValueError("native scientific report must be fresh")
    report = {"success": False, "cases": {}, "lifecycle": []}
    try:
        report["isolation"] = clean_host(
            Path(os.environ["GITHUB_WORKSPACE"]), Path(os.environ["RUNNER_TEMP"])
        )
        identity = artifacts.read_json(packet / "expectation.json")["identity"]
        if (identity["source_commit"], identity["target"], identity["version"]) != (
            commit,
            target,
            version,
        ):
            raise ValueError("native expected identity differs from triggering run")
        actual_python = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
        }
        if actual_python != identity["python"]:
            raise ValueError("native driver target differs from reviewed interpreter")
        expected = packet_expectation(packet, identity)
        artifacts.verify_downloaded(directory, identity, expected["notice_sha256"])
        final = artifacts.read_json(directory / f"{target}-artifact.json")
        components = artifacts.read_json(directory / f"{target}-build-components.json")
        binary = directory / final["artifact"]["filename"]
        source = artifacts.read_json(packet / "source-result.json")
        report.update(
            identity=identity,
            artifact=final["artifact"],
            fixtures=source["fixtures"],
            source_results=source["results"],
        )

        def journey(application, work):
            baseline = artifacts.bundle_inventory(application)
            launcher = "Asymmetry.exe" if target == "windows-x64" else "Contents/MacOS/Asymmetry"
            control = execute_case(
                [str(application / launcher)],
                packet / "inputs",
                identity,
                work / "control",
                report["cases"],
                "control",
            )
            compare_results(source, control["report"])
            with payload_copy(application, work, report["cases"], "copy-control") as clone:
                copied = execute_case(
                    [str(clone / launcher)],
                    packet / "inputs",
                    identity,
                    work / "copy-control",
                    report["cases"],
                    "copy-control",
                )
                compare_results(source, copied["report"])
            for name, component in omission_components(
                components["components"], control["report"], target
            ).items():
                with payload_copy(application, work, report["cases"], name) as clone:
                    relative = artifacts.portable(component["path"])
                    removed = (clone / relative).resolve(strict=True)
                    if (
                        not removed.is_relative_to(clone.resolve())
                        or artifacts.release.digest(removed) != component["sha256"]
                    ):
                        raise ValueError("omission component differs from delivered payload")
                    removed.unlink()
                    if removed.exists():
                        raise ValueError("omitted component remains available")
                    report["cases"].setdefault(name, {})["omitted_component"] = {
                        "path": relative,
                        "sha256": component["sha256"],
                        "absent_before_child": True,
                    }
                    execute_case(
                        [str(clone / launcher)],
                        packet / "inputs",
                        identity,
                        work / name,
                        report["cases"],
                        name,
                        "resource" if name == "resource" else "counts.nxs:load",
                        Path(relative).name if name == "native" else None,
                    )
            if artifacts.bundle_inventory(application) != baseline:
                raise ValueError("scientific journey changed the intact delivered application")
            report["intact_payload_unchanged"] = True

        assembly = artifacts.inspect_payload(
            binary,
            components["components"],
            target,
            {item["sha256"] for item in components["assembly"]["additional_resources"]},
            journey,
            events=report["lifecycle"],
        )
        if assembly != components["assembly"] or artifacts.file_record(binary) != final["artifact"]:
            raise ValueError("native media or assembly resources changed during acceptance")
        report["success"] = True
    except Exception as exc:
        report["failure"] = {
            "type": type(exc).__name__,
            "cleanup_failures": [event for event in report["lifecycle"] if not event["success"]],
        }
        raise
    finally:
        artifacts.write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "run"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--target", choices=("windows-x64", "macos-arm64"), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare_packet(args.packet, args.target, args.commit)
    elif args.operation == "verify":
        verify_source_packet(args.packet, args.target, args.commit)
    else:
        run_native(args.packet, args.directory, args.target, args.commit, args.version, args.output)


if __name__ == "__main__":
    main()
