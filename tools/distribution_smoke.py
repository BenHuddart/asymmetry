"""Build via sdist and probe wheel installations in fresh, outside-checkout venvs.

Run with an outer interpreter containing PyPA build and pip. Target environments
receive the requested extras with constrained or ordinary dependency resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default as metadata_policy
from pathlib import Path

RESOURCE_PATHS = (
    "asymmetry/resources/skills/asymmetry-analysis/SKILL.md",
    "asymmetry/resources/skills/asymmetry-analysis/references/commands.md",
)
INSTALL_PROFILES = {
    "core": ("", ()),
    "gui": ("gui,hdf5,root", ("PySide6", "matplotlib", "h5py", "uproot")),
    "agent": ("agent", ("matplotlib", "h5py")),
    "all": ("all", ("PySide6", "matplotlib", "h5py", "pyhdf", "uproot", "gleplot")),
}


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
        sdist_metadata = BytesParser(policy=metadata_policy).parsebytes(metadata_file.read())
        for resource in RESOURCE_PATHS:
            assert any(name.endswith("/src/" + resource) for name in members), resource
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = BytesParser(policy=metadata_policy).parsebytes(archive.read(metadata_path))
        for field in (
            "Name",
            "Version",
            "Summary",
            "Requires-Python",
            "Classifier",
            "Requires-Dist",
            "Provides-Extra",
        ):
            assert sdist_metadata.get_all(field) == wheel_metadata.get_all(field), field
        for field, declared in (
            ("Name", project["name"]),
            ("Version", project["version"]),
            ("Summary", project["description"]),
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


def installed_probe(
    contract: dict, forbidden_roots: list[str], required_modules: tuple[str, ...] = ()
) -> dict:
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
    optional = (
        "PySide6",
        "matplotlib",
        "h5py",
        "pyhdf",
        "uproot",
        "gleplot",
    )
    blocked = (
        "pytest",
        "build",
        "lmfit",
        "ruff",
        "sphinx",
        "readme_renderer",
        "cupy",
    ) + tuple(name for name in optional if name not in required_modules)
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
    for name in required_modules:
        module = importlib.import_module(name)
        origin = Path(module.__spec__.origin).resolve()
        assert origin.is_relative_to(target), f"optional dependency origin leakage: {name}"
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
        assert not any(name == extra or name.startswith(extra + ".") for extra in blocked), name
        if "PySide6" not in required_modules:
            assert not name.startswith("asymmetry.gui"), name
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
        "required_modules": required_modules,
    }


def exercise_optional_features(required_modules: tuple[str, ...]) -> list[str]:
    """Exercise installed file readers, plots and script export with owned counts."""
    from pathlib import Path

    import numpy as np

    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.io import load

    completed = []
    counts = np.arange(20, dtype=np.int32) + 100
    if "h5py" in required_modules:
        from asymmetry.core.io.nexus_writer import write_nexus_v1

        run = Run(
            run_number=734,
            histograms=[Histogram(counts.copy(), 0.01) for _ in range(2)],
            grouping={"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2},
        )
        write_nexus_v1(run, "installed.nxs")
        restored = load("installed.nxs").run
        np.testing.assert_array_equal(restored.histograms[0].counts, counts)
        completed.append("HDF5 NeXus write/load")
    if "uproot" in required_modules:
        import uproot

        with uproot.recreate("installed.root") as root:
            for name, value in {
                "Run Number": "734",
                "No of Histos": "2",
                "Time Resolution": "10 ns",
            }.items():
                root[f"RunHeader/RunInfo/{name}"] = value
            for index in (1, 2):
                prefix = f"RunHeader/DetectorInfo/Detector{index:03d}"
                for name, value in {
                    "Histo Number": str(index),
                    "Time Zero Bin": "0",
                    "First Good Bin": "0",
                    "Last Good Bin": "19",
                }.items():
                    root[f"{prefix}/{name}"] = value
                root[f"histos/DecayAnaModule/hDecay{index:03d}"] = (counts, np.arange(21))
        restored = load("installed.root").run
        np.testing.assert_array_equal(restored.histograms[0].counts[:20], counts)
        completed.append("ROOT load")
    if "pyhdf" in required_modules:
        from pyhdf.SD import SD, SDC

        container = SD("installed.hdf", SDC.WRITE | SDC.CREATE)
        dataset = container.create("counts", SDC.INT32, (20,))
        dataset[:] = counts
        dataset.endaccess()
        container.end()
        container = SD("installed.hdf", SDC.READ)
        dataset = container.select("counts")
        np.testing.assert_array_equal(dataset[:], counts)
        dataset.endaccess()
        container.end()
        completed.append("HDF4 native runtime write/read")
    if "matplotlib" in required_modules:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure()
        FigureCanvasAgg(figure)
        figure.subplots().plot(counts)
        figure.savefig("installed.png")
        assert Path("installed.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        completed.append("headless PNG")
    if "gleplot" in required_modules:
        import gleplot

        from asymmetry.gui.utils.gle_figure import gle_figure

        figure = gle_figure(gleplot)
        figure.add_subplot(111).plot([0, 1, 2], [1, 2, 1])
        figure.savefig("installed.gle")
        assert Path("installed.gle").read_text().strip()
        completed.append("GLE script generation; external compiler not exercised")
    return completed


def clean_environment(root: Path) -> dict[str, str]:
    """Keep build settings while removing interpreter and user-state injection."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "PIP_"))
    }
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", HOME=str(root / "home"))
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    env["QT_QPA_PLATFORM"] = "offscreen"
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


def smoke(
    source: Path,
    output: Path,
    interpreters: list[str],
    profiles: list[str],
    resolution: str,
    artifacts: Path | None = None,
    build_only: bool = False,
) -> None:
    """Build once and reuse the exact wheel across independently created targets."""
    source = source.resolve()
    output = output.resolve()
    assert not output.is_relative_to(source), "output must be outside checkout"
    output.mkdir(parents=True, exist_ok=False)
    env = clean_environment(output)
    distributions = artifacts.resolve() if artifacts is not None else output / "dist"
    if artifacts is None:
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
        "resolution": resolution,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if artifacts is None:
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
    (sdist,) = distributions.glob("*.tar.gz")
    (wheel,) = distributions.glob("*.whl")
    assert set(distributions.iterdir()) == {sdist, wheel}, "unexpected candidate files"
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
    import itertools

    targets = () if build_only else itertools.product(interpreters, profiles)
    for index, (interpreter, profile) in enumerate(targets):
        extras, required_modules = INSTALL_PROFILES[profile]
        work = output / f"{profile}-{index}"
        work.mkdir()
        target = work / "venv"
        run_logged(
            [interpreter, "-I", "-m", "venv", str(target)],
            output,
            env,
            work / "venv.log",
        )
        python = target / "bin" / "python"
        constraints = ["-c", str(source / "constraints.txt")] if resolution == "constrained" else []
        requirement = str(wheel) + (f"[{extras}]" if extras else "")
        run_logged(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                *constraints,
                requirement,
            ],
            output,
            env,
            work / "install.log",
        )
        run_logged([str(python), "-I", "-m", "pip", "check"], work, env, work / "pip-check.log")
        probe = (
            inspect.getsource(installed_probe)
            + "\nimport json\nprint('PROBE_JSON=' + json.dumps(installed_probe("
            + repr(contract)
            + ", "
            + repr(forbidden)
            + ", "
            + repr(required_modules)
            + ")))\n"
        )
        run_logged([str(python), "-I", "-c", probe], work, env, work / "probe.log")
        readme_code = (
            (source / "README.md")
            .read_text(encoding="utf-8")
            .split("```python\n", 1)[1]
            .split("```", 1)[0]
        )
        readme_probe = (
            readme_code
            + "\nassert result.success\nassert abs(result.parameters['Lambda'].value - 0.3) < 0.02\n"
        )
        run_logged([str(python), "-I", "-c", readme_probe], work, env, work / "readme.log")
        if required_modules:
            feature_probe = (
                inspect.getsource(exercise_optional_features)
                + "\nimport json\nprint(json.dumps(exercise_optional_features("
                + repr(required_modules)
                + ")))\n"
            )
            run_logged([str(python), "-I", "-c", feature_probe], work, env, work / "features.log")
        if "PySide6" in required_modules:
            run_logged(
                [str(target / "bin" / "asymmetry-gui"), "--smoke-test"],
                work,
                env,
                work / "gui.log",
            )
        for label, command in (
            ("console", [str(target / "bin" / "asymmetry"), "--help"]),
            ("module", [str(python), "-I", "-m", "asymmetry", "--help"]),
        ):
            run_logged(command, work, env, work / f"{label}.log")
        report = json.loads((work / "probe.log").read_text().split("\nPROBE_JSON=")[-1])
        report["profile"] = profile
        manifest["targets"].append(report)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shutil.rmtree(target)
    print(f"distribution smoke passed: {output / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts", type=Path, help="Probe these existing sdist/wheel bytes")
    parser.add_argument("--build-only", action="store_true", help="Build without target installs")
    parser.add_argument("--python", action="append", default=[])
    parser.add_argument("--profile", choices=INSTALL_PROFILES, action="append", default=[])
    parser.add_argument(
        "--resolution", choices=("constrained", "unconstrained"), default="constrained"
    )
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="asymmetry-distribution-")) / "evidence"
    smoke(
        args.source,
        output,
        args.python or [sys.executable],
        args.profile or ["core"],
        args.resolution,
        args.artifacts,
        args.build_only,
    )
