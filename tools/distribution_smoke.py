"""Build via sdist and probe a core-only wheel in fresh, outside-checkout venvs.

Run with an outer interpreter containing PyPA build and pip. Target environments
receive only the wheel's declared requirements, restricted by constraints.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

RESOURCE_PATHS = (
    "asymmetry/resources/skills/asymmetry-analysis/SKILL.md",
    "asymmetry/resources/skills/asymmetry-analysis/references/commands.md",
)


def artifact_contract(source: Path, sdist: Path, wheel: Path) -> dict:
    """Check generated metadata and resources against the declared project."""
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    with tarfile.open(sdist) as archive:
        members = archive.getnames()
        pkg_info = next(
            name for name in members if name.count("/") == 1 and name.endswith("/PKG-INFO")
        )
        metadata_file = archive.extractfile(pkg_info)
        assert metadata_file is not None
        sdist_metadata = BytesParser().parsebytes(metadata_file.read())
        for resource in RESOURCE_PATHS:
            assert any(name.endswith("/src/" + resource) for name in members), resource
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_path))
        for field in ("Name", "Version", "Requires-Python", "Classifier", "Requires-Dist"):
            assert sdist_metadata.get_all(field) == wheel_metadata.get_all(field), field
        for field, declared in (
            ("Name", project["name"]),
            ("Version", project["version"]),
            ("Requires-Python", project["requires-python"]),
        ):
            assert wheel_metadata[field] == declared, field
        assert sorted(wheel_metadata.get_all("Classifier")) == sorted(project["classifiers"])
        files = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if not name.endswith("/")
        }
        for resource in RESOURCE_PATHS:
            assert resource in files, resource
        from configparser import ConfigParser

        entrypoints = ConfigParser()
        entrypoints.read_string(
            archive.read(metadata_path.replace("METADATA", "entry_points.txt")).decode()
        )
        assert dict(entrypoints["console_scripts"]) == project["scripts"], "console scripts"
        assert dict(entrypoints["gui_scripts"]) == project["gui-scripts"], "GUI scripts"
    return {
        "metadata": {
            key: [str(value) for value in wheel_metadata.get_all(key)]
            for key in wheel_metadata.keys()
        },
        "files": files,
        "version": project["version"],
    }


def installed_probe(contract: dict, forbidden_roots: list[str]) -> dict:
    """Executed as source text by the target Python; imports no checkout helpers."""
    import hashlib
    import importlib
    import importlib.metadata
    import importlib.resources
    import importlib.util
    import math
    import os
    import site
    import sys
    import sysconfig
    from pathlib import Path

    assert sys.prefix != sys.base_prefix, "target must be a fresh venv"
    assert not site.ENABLE_USER_SITE, "user-site leakage"
    assert not os.environ.get("PYTHONPATH"), "PYTHONPATH leakage"
    target = Path(sysconfig.get_path("purelib")).resolve()
    for entry in sys.path:
        for root in forbidden_roots:
            assert not Path(entry).resolve().is_relative_to(Path(root).resolve()), (
                "source-path leakage"
            )
    distribution = importlib.metadata.distribution("asymmetry")
    assert distribution.version == contract["version"], "installed version"
    assert distribution.metadata["Requires-Python"] == contract["metadata"]["Requires-Python"][0], (
        "installed Python metadata"
    )
    for name, digest in contract["files"].items():
        if name.endswith(".dist-info/RECORD"):
            continue  # Installers rewrite RECORD for generated scripts and direct_url.json.
        installed = Path(distribution.locate_file(name)).resolve()
        assert installed.is_relative_to(target), f"installed path leakage: {installed}"
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == digest, (
            f"artifact bytes: {name}"
        )
    blocked = (
        "PySide6",
        "matplotlib",
        "h5py",
        "pyhdf",
        "uproot",
        "pytest",
        "build",
        "lmfit",
        "gleplot",
    )
    for name in blocked:
        assert importlib.util.find_spec(name) is None, (
            f"unexpected optional/tool dependency: {name}"
        )
    modules = (
        "asymmetry",
        "asymmetry.core.utils.constants",
        "asymmetry.core.data",
        "asymmetry.core.fitting",
        "asymmetry.core.fitting.models",
        "asymmetry.core.workflow",
        "asymmetry.cli",
    )
    for name in modules:
        importlib.import_module(name)
    from asymmetry import __version__, register_component
    from asymmetry.core.fitting.models import exponential_relaxation

    assert callable(register_component)
    assert __version__ == distribution.version, "public version"
    # At zero time exponential relaxation is its amplitude plus baseline.
    value = float(exponential_relaxation(0.0, A0=0.25, Lambda=0.7, baseline=0.05))
    assert math.isclose(value, 0.3, rel_tol=0, abs_tol=1e-14), value
    resources = importlib.resources.files("asymmetry.resources") / "skills" / "asymmetry-analysis"
    for name in ("SKILL.md", "references/commands.md"):
        assert (resources / name).read_text(encoding="utf-8").strip(), name
    from asymmetry import cli

    help_results = []
    for arguments in (
        ["--help"],
        ["fit", "--help"],
        ["skill", "--help"],
        ["project-sources", "--help"],
    ):
        try:
            cli.main(arguments)
        except SystemExit as exc:
            assert exc.code == 0, arguments
        else:
            raise AssertionError(f"help did not exit: {arguments}")
        help_results.append(arguments)
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "asymmetry" or name.startswith("asymmetry."):
            origin = module.__spec__.origin
            if origin is not None:
                path = Path(origin).resolve()
                assert path.is_relative_to(target), f"module origin leakage: {name}: {path}"
                relative = path.relative_to(target).as_posix()
                assert relative in contract["files"], f"module absent from artifact: {relative}"
                origins[name] = str(path)
        assert not any(
            name == extra or name.startswith(extra + ".") for extra in (*blocked, "asymmetry.gui")
        ), name
    dependency_origins = {}
    for name in ("numpy", "scipy", "iminuit"):
        module = importlib.import_module(name)
        origin = Path(module.__spec__.origin).resolve()
        assert origin.is_relative_to(target), f"dependency origin leakage: {name}: {origin}"
        dependency_origins[name] = str(origin)
    for dist in importlib.metadata.distributions():
        assert Path(dist.locate_file("")).resolve().is_relative_to(target), dist.metadata["Name"]
    return {
        "dependency_origins": dependency_origins,
        "venv_config": (Path(sys.prefix) / "pyvenv.cfg").read_text(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "version": sys.version,
        "sys_path": sys.path,
        "dependencies": sorted(
            (dist.metadata["Name"], dist.version, dist.requires)
            for dist in importlib.metadata.distributions()
        ),
        "origins": origins,
        "analytic_value": value,
        "help": help_results,
    }


def clean_environment(root: Path) -> dict[str, str]:
    """Keep build settings while removing interpreter and user-state injection."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", HOME=str(root / "home"))
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        env[key] = str(root / key.lower())
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def run_logged(command: list[str], cwd: Path, env: dict, log: Path) -> subprocess.CompletedProcess:
    """Retain commands/output and fail closed on process errors or timeouts."""
    with log.open("w") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300
        )
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {command}; see {log}")
    return result


def reject_below_minimum(wheel: Path, output: Path, env: dict, python: str) -> dict:
    """Require target-resolution rejection, including with bootstrap pip versions."""
    rejection_command = [
        python,
        "-m",
        "pip",
        "download",
        "--dest",
        str(output / "below-minimum-download"),
        "--no-deps",
        "--no-index",
        "--python-version",
        "3.10",
        "--only-binary=:all:",
        str(wheel),
    ]
    rejection = subprocess.run(
        rejection_command, cwd=output, env=env, capture_output=True, text=True, timeout=60
    )
    (output / "below-minimum.log").write_text(
        json.dumps(rejection_command) + "\n" + rejection.stdout + rejection.stderr
    )
    assert rejection.returncode != 0 and "requires a different Python" in rejection.stderr, (
        "candidate must reject Python 3.10 resolution"
    )
    return {
        "kind": "pip target-resolution only; no Python 3.10 execution",
        "target": "3.10",
        "exit": rejection.returncode,
    }


def smoke(source: Path, output: Path, interpreters: list[str]) -> None:
    """Build once and reuse the exact wheel across independently created targets."""
    source = source.resolve()
    output = output.resolve()
    assert not output.is_relative_to(source), "output must be outside checkout"
    output.mkdir(parents=True, exist_ok=False)
    env = clean_environment(output)
    distributions = output / "dist"
    distributions.mkdir()
    import importlib.metadata

    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached"], cwd=source)
        .decode()
        .split("\0")
    )
    manifest = {
        "source": str(source),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "driver": sys.executable,
        "driver_version": sys.version,
        "build_tools": {
            name: importlib.metadata.version(name) for name in ("build", "pip", "pyproject-hooks")
        },
        "inputs": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in tracked
            if name and (source / name).is_file()
        },
        "targets": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run_logged(
        [
            sys.executable,
            "-m",
            "build",
            "-v",
            "--sdist",
            "--outdir",
            str(distributions),
            str(source),
        ],
        output,
        env,
        output / "sdist.log",
    )
    (sdist,) = distributions.glob("*.tar.gz")
    unpacked = output / "sdist-source"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    (extracted,) = unpacked.iterdir()
    run_logged(
        [
            sys.executable,
            "-m",
            "build",
            "-v",
            "--wheel",
            "--outdir",
            str(distributions),
            str(extracted),
        ],
        output,
        env,
        output / "wheel.log",
    )
    (wheel,) = distributions.glob("*.whl")
    contract = artifact_contract(source, sdist, wheel)
    manifest.update(
        contract=contract,
        artifacts={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (sdist, wheel)
        },
    )
    manifest["below_minimum"] = reject_below_minimum(wheel, output, env, sys.executable)
    forbidden = [str(source)]
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    forbidden.append(str(Path(common).parent))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for index, interpreter in enumerate(interpreters):
        target = output / f"target-{index}"
        run_logged(
            [interpreter, "-I", "-m", "venv", str(target)],
            output,
            env,
            output / f"venv-{index}.log",
        )
        python = target / "bin" / "python"
        run_logged(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "-c",
                str(source / "constraints.txt"),
                str(wheel),
            ],
            output,
            env,
            output / f"install-{index}.log",
        )
        probe = (
            inspect.getsource(installed_probe)
            + "\nimport json\nprint('PROBE_JSON=' + json.dumps(installed_probe("
            + repr(contract)
            + ", "
            + repr(forbidden)
            + ")))\n"
        )
        run_logged([str(python), "-I", "-c", probe], output, env, output / f"probe-{index}.log")
        for label, command in (
            ("console", [str(target / "bin" / "asymmetry"), "--help"]),
            ("module", [str(python), "-I", "-m", "asymmetry", "--help"]),
        ):
            run_logged(command, output, env, output / f"{label}-{index}.log")
        report = json.loads((output / f"probe-{index}.log").read_text().split("\nPROBE_JSON=")[-1])
        manifest["targets"].append(report)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"distribution smoke passed: {output / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", action="append", default=[])
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="asymmetry-distribution-")) / "evidence"
    smoke(args.source, output, args.python or [sys.executable])
