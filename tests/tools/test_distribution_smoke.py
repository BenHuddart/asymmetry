"""Adversarial artifact/probe fixtures; no real builds in the fast suite."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

from tools.distribution_smoke import (
    RESOURCE_PATHS,
    artifact_contract,
    clean_environment,
    installed_probe,
    reject_below_minimum,
    run_logged,
)

ROOT = Path(__file__).resolve().parents[2]


def test_supported_minimum_matches_required_enum_api() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    supported = SpecifierSet(project["requires-python"])
    assert "3.10" not in supported, "core requires enum.StrEnum (Python 3.11)"
    assert "3.11" in supported
    classifiers = project["classifiers"]
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert config["tool"]["ruff"]["target-version"] == "py311"


@pytest.fixture
def artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An independently authored tiny distribution with the actual resource layout."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="asymmetry"\nversion="1.0"\nrequires-python=">=3.11"\n'
        'classifiers=["Programming Language :: Python :: 3.11"]\n'
        '[project.scripts]\nasymmetry="asymmetry.cli:main"\n'
        '[project.gui-scripts]\nasymmetry-gui="asymmetry.gui.app:main"\n'
    )
    metadata = b"Name: asymmetry\nVersion: 1.0\nRequires-Python: >=3.11\nClassifier: Programming Language :: Python :: 3.11\nSummary: \xce\xbcSR\n"
    members = {
        "asymmetry-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "asymmetry-1.0.dist-info/METADATA": metadata,
        "asymmetry-1.0.dist-info/entry_points.txt": b"[console_scripts]\nasymmetry = asymmetry.cli:main\n[gui_scripts]\nasymmetry-gui = asymmetry.gui.app:main\n",
        **{name: b"fixture resource" for name in RESOURCE_PATHS},
    }
    wheel = tmp_path / "asymmetry-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    sdist = tmp_path / "asymmetry-1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in {
            "PKG-INFO": metadata,
            **{"src/" + path: members[path] for path in RESOURCE_PATHS},
        }.items():
            info = tarfile.TarInfo("asymmetry-1.0/" + name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return source, sdist, wheel


@pytest.mark.parametrize("defect", [None, "resource", "metadata", "entrypoint"])
def test_artifact_contract_rejects_defective_wheel(artifact, defect) -> None:
    source, sdist, wheel = artifact
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if defect == "resource":
        del members[RESOURCE_PATHS[0]]
    elif defect == "metadata":
        path = "asymmetry-1.0.dist-info/METADATA"
        members[path] = members[path].replace(b">=3.11", b">=3.10")
    elif defect == "entrypoint":
        path = "asymmetry-1.0.dist-info/entry_points.txt"
        members[path] = members[path].replace(b"asymmetry.cli:main", b"asymmetry.cli:missing")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    if defect is None:
        contract = artifact_contract(source, sdist, wheel)
        assert contract["version"] == "1.0"
        json.dumps(contract)
    else:
        with pytest.raises(AssertionError):
            artifact_contract(source, sdist, wheel)


@pytest.fixture
def target(tmp_path: Path, artifact) -> tuple[Path, dict]:
    """A stdlib-only synthetic installed package tests isolation and failure paths."""
    _, _, wheel = artifact
    venv = tmp_path / "target"
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)], check=True, timeout=30
    )
    python = venv / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    files = {
        "__init__.py": '__version__="1.0"\ndef register_component(): pass\n',
        "core/__init__.py": "",
        "core/utils/__init__.py": "",
        "core/utils/constants.py": "from enum import StrEnum\n",
        "core/data/__init__.py": "",
        "core/fitting/__init__.py": "",
        "core/fitting/models.py": "import math\ndef exponential_relaxation(t, A0, Lambda, baseline):\n    return A0*math.exp(-Lambda*t)+baseline\n",
        "core/workflow/__init__.py": "",
        "resources/__init__.py": "",
        "cli.py": 'import argparse\ndef main(args):\n    parser=argparse.ArgumentParser()\n    if "--help" in args: parser.parse_args(["--help"])\n',
    }
    for name, content in files.items():
        path = site / "asymmetry" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in ("numpy", "scipy", "iminuit"):
        (site / (name + ".py")).write_text('"""Synthetic dependency-origin fixture."""\n')
    contract = artifact_contract(*artifact)
    contract["files"].update(
        {
            "asymmetry/" + name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        }
    )
    return python, contract


@pytest.mark.parametrize(
    "defect", [None, "core", "resource", "version", "origin", "pythonpath", "bytes", "optional"]
)
def test_installed_probe_fails_closed(tmp_path: Path, target, defect) -> None:
    python, contract = target
    site = next((python.parent.parent / "lib").glob("python*/site-packages"))
    env = clean_environment(tmp_path)
    forbidden = [str(ROOT)]
    if defect == "core":
        path = site / "asymmetry/core/workflow/__init__.py"
        path.write_text("import absent_required_core_dependency\n")
        contract["files"][path.relative_to(site).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif defect == "resource":
        (site / RESOURCE_PATHS[0]).unlink()
    elif defect == "version":
        contract["version"] = "2.0"
    elif defect == "origin":
        # A .pth leak remains visible even when -I ignores PYTHONPATH.
        (site / "source.pth").write_text(str(ROOT) + "\n")
    elif defect == "pythonpath":
        env["PYTHONPATH"] = str(ROOT)
    elif defect == "bytes":
        (site / "asymmetry/__init__.py").write_text('__version__="1.0"\n')
    elif defect == "optional":
        (site / "matplotlib.py").write_text("")
    probe = inspect.getsource(installed_probe) + f"\ninstalled_probe({contract!r}, {forbidden!r})\n"
    result = subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (defect is None), result.stdout + result.stderr
    if defect == "core":
        assert "absent_required_core_dependency" in result.stderr
    if defect == "origin":
        assert "source-path leakage" in result.stderr


def test_broken_console_script_is_not_success(tmp_path: Path) -> None:
    script = tmp_path / "asymmetry"
    script.write_text("raise RuntimeError('broken installed entrypoint')\n")
    with pytest.raises(RuntimeError, match="exit 1"):
        run_logged(
            [sys.executable, "-I", str(script), "--help"],
            tmp_path,
            clean_environment(tmp_path),
            tmp_path / "console.log",
        )
    assert "broken installed entrypoint" in (tmp_path / "console.log").read_text()


def test_clean_environment_discards_source_and_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("PYTHONHOME", str(ROOT))
    monkeypatch.setenv("VIRTUAL_ENV", str(ROOT / ".venv"))
    env = clean_environment(tmp_path)
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()
    assert Path(env["HOME"]).is_relative_to(tmp_path)


@pytest.mark.parametrize("requires_python", [">=3.11", ">=3.10"])
def test_bootstrap_pip_checks_only_candidate_wheel_target(
    artifact, tmp_path, requires_python
) -> None:
    _, _, wheel = artifact
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    path = "asymmetry-1.0.dist-info/METADATA"
    members[path] = members[path].replace(b">=3.11", requires_python.encode())
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    venv = tmp_path / "resolver"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(venv)], check=True, timeout=30)
    python = str(venv / "bin/python")
    env = clean_environment(tmp_path)
    if requires_python == ">=3.11":
        report = reject_below_minimum(wheel, tmp_path, env, python)
        assert report["exit"] != 0
        assert not list((tmp_path / "below-minimum-download").glob("*.whl"))
    else:
        with pytest.raises(AssertionError, match="candidate must reject Python 3.10"):
            reject_below_minimum(wheel, tmp_path, env, python)
        assert (tmp_path / "below-minimum-download" / wheel.name).is_file()
