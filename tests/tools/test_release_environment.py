"""Release replay rejects drift before native packaging starts."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "release_environment", ROOT / "tools/release_environment.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def observation() -> dict:
    return {
        "python": {
            "implementation": "CPython",
            "version": "3.12.10",
            "platform": "win32",
            "machine": "AMD64",
        },
        "distributions": {"pip": "25.3", "gleplot": "6.0.0", "asymmetry": "0.28.0"},
        "origins": {
            "asymmetry": {
                "url": "file:///private/wheels/asymmetry.whl",
                "archive_info": {"hashes": {"sha256": "b" * 64}},
            },
        },
    }


def verify(actual: dict) -> dict:
    return release.verify_environment(
        actual,
        {
            "implementation": "CPython",
            "version": "3.12.10",
            "platform": "win32",
            "machine": "AMD64",
        },
        {"pip": "25.3", "gleplot": "6.0.0", "asymmetry": "0.28.0"},
        {"asymmetry": "b" * 64},
    )


def test_environment_verification_allows_only_observed_public_fields() -> None:
    actual = observation()
    actual["environment"] = {"secret": "private"}
    verified = verify(actual)
    assert verified == {"python": actual["python"], "distributions": actual["distributions"]}
    assert "private" not in str(verified)


@pytest.mark.parametrize("name,value", [("pip", "26.0"), ("gleplot", "1.8.0"), ("extra", "1.0")])
def test_changed_or_unreviewed_distribution_fails(name: str, value: str) -> None:
    actual = observation()
    actual["distributions"][name] = value
    with pytest.raises(ValueError, match="distribution drift"):
        verify(actual)


def test_missing_distribution_fails() -> None:
    actual = observation()
    del actual["distributions"]["gleplot"]
    with pytest.raises(ValueError, match="distribution drift"):
        verify(actual)


@pytest.mark.parametrize("field", ["version", "platform", "machine", "implementation"])
def test_wrong_interpreter_fails(field: str) -> None:
    actual = observation()
    actual["python"][field] = "wrong"
    with pytest.raises(ValueError, match="Python identity"):
        verify(actual)


@pytest.mark.parametrize("name", ["asymmetry"])
def test_wrong_wheel_origin_fails(name: str) -> None:
    actual = observation()
    actual["origins"][name]["archive_info"]["hashes"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="wheel identity"):
        verify(actual)


@pytest.mark.parametrize("name", ["pip", "gleplot"])
def test_unreviewed_direct_origin_fails(name) -> None:
    actual = observation()
    actual["origins"][name] = {"url": "https://user:secret@example.invalid/pkg.whl"}
    with pytest.raises(ValueError, match="direct origins"):
        verify(actual)


def test_replay_requires_same_environments() -> None:
    first = {"environment": verify(observation())}
    second = copy.deepcopy(first)
    release.verify_replays(first, second)
    second["environment"]["distributions"]["gleplot"] = "6.0.1"
    with pytest.raises(ValueError, match="replay"):
        release.verify_replays(first, second)


def test_inno_observation_must_match_reviewed_compiler() -> None:
    release.verify_inno(
        "Inno Setup 6 Command-Line Compiler\nCopyright\nCompiler engine version: Inno Setup 6.4.3\n",
        "6.4.3",
    )
    with pytest.raises(ValueError, match="Inno Setup"):
        release.verify_inno(
            "Inno Setup 6 Command-Line Compiler\nCompiler engine version: Inno Setup 6.5.0\n",
            "6.4.3",
        )


@pytest.mark.parametrize(
    "line",
    [
        "numpy>=1.24",
        "numpy==2.2.6",
        "-e .",
        "git+https://example.invalid/pkg",
        "pip==25.3 --hash=sha256:bad",
    ],
)
def test_release_requirements_reject_uncontrolled_install_inputs(tmp_path: Path, line: str) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact hashed wheel pins"):
        release.reviewed_requirements(requirements)


def test_release_requirements_normalize_and_require_bootstrap(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "pip==25.3 --hash=sha256:"
        + "a" * 64
        + "\nPySide6_Addons==6.10.0 --hash=sha256:"
        + "b" * 64
        + "\n",
        encoding="utf-8",
    )
    assert release.reviewed_requirements(requirements) == {
        "pip": "25.3",
        "pyside6-addons": "6.10.0",
    }
    requirements.write_text("numpy==2.2.6 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bootstrap pip"):
        release.reviewed_requirements(requirements)


@pytest.mark.parametrize(
    "source",
    [
        "file:///home/user/private.whl",
        "file:///C:/Users/user/private.whl",
        "https://user:secret@files.pythonhosted.org/a.whl",
        "https://files.pythonhosted.org/a.whl?token=secret",
        "https://other.invalid/a.whl",
    ],
)
def test_download_record_rejects_local_or_credential_origins(tmp_path: Path, source: str) -> None:
    import json

    report = {
        "install": [
            {
                "metadata": {"name": "pip"},
                "download_info": {"url": source, "archive_info": {"hashes": {"sha256": "a" * 64}}},
            }
        ]
    }
    (tmp_path / "bootstrap-report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="index origin"):
        release.downloaded_wheels(tmp_path)


def test_download_record_allowlists_nested_fields(tmp_path: Path) -> None:
    import json

    report = {
        "environment": {"secret": "private"},
        "install": [
            {
                "metadata": {"name": "pip", "author": "/private"},
                "download_info": {
                    "url": "https://files.pythonhosted.org/packages/pip.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}, "private": "C:\\Users\\user"},
                    "nested": {"secret": "private"},
                },
            }
        ],
    }
    for name in ("bootstrap-report.json", "install-report.json"):
        (tmp_path / name).write_text(json.dumps(report), encoding="utf-8")
    assert release.downloaded_wheels(tmp_path) == {
        "pip": {"filename": "pip.whl", "sha256": "a" * 64}
    }


@pytest.mark.parametrize("missing", ["matplotlib", "h5py", "pyhdf", "uproot", "gleplot"])
def test_selected_extras_are_checked_even_without_activated_metadata(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    names = ["matplotlib", "h5py", "pyhdf", "uproot", "gleplot"]
    requirements = [
        f'{name}>=1; extra == "{extra}"'
        for name, extra in zip(names, ("gui", "hdf5", "hdf4", "root", "gle"), strict=True)
    ]
    monkeypatch.setattr(release.importlib.metadata, "requires", lambda name: requirements)
    monkeypatch.setattr(
        release,
        "observe",
        lambda: {"distributions": {name: "2.0" for name in names if name != missing}},
    )
    with pytest.raises(ValueError, match="desktop requirement"):
        release.verify_selected_extras()


def test_child_environment_removes_injected_paths_and_pip_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PIP_CONSTRAINT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TARGET",
        "PIP_USER",
    ):
        monkeypatch.setenv(key, "unreviewed")
    env = release.clean_environment()
    assert "unreviewed" not in str(env)
    assert env["PIP_CONFIG_FILE"] == release.os.devnull


def test_pre_freeze_verification_rejects_stale_inputs_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tools = {"gle": {"commit": "a" * 40}, "targets": {"windows-x64": {}}}
    (inputs / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (inputs / "windows-archives.json").write_text("[]", encoding="utf-8")
    replay = tmp_path / "replay-1"
    replay.mkdir()
    record = {
        "source_commit": "b" * 40,
        "target": "windows-x64",
        "input_hashes": {
            "tools.json": "c" * 64,
            "windows-x64.txt": "d" * 64,
            "windows-archives.json": "e" * 64,
        },
    }
    (replay / "observed.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(release, "INPUTS", inputs)
    monkeypatch.setattr(release, "verify_source", lambda root: "d" * 40)
    with pytest.raises(ValueError, match="source commit changed"):
        release.verify_build(tmp_path)
    monkeypatch.setattr(release, "verify_source", lambda root: "b" * 40)
    with pytest.raises(ValueError, match="inputs changed"):
        release.verify_build(tmp_path)


def test_source_wheel_build_is_offline_without_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []

    def build(command: list, cwd: Path, *, offline: bool) -> str:
        commands.append((command, offline))
        (cwd / "synthetic-1-py3-none-any.whl").write_bytes(b"wheel")
        return ""

    monkeypatch.setattr(release, "run", build)
    wheel = release.build_wheel(Path("python"), tmp_path / "source", tmp_path / "output")
    command, offline = commands[0]
    assert offline
    assert {"--no-deps", "--no-build-isolation", "--no-cache-dir"} <= set(command)
    assert wheel.read_bytes() == b"wheel"


@pytest.mark.parametrize(
    "filename",
    [
        "src/asymmetry/resources/private.png",
        "src/asymmetry/injected.py",
        "src/asymmetry/injected.so",
        "src/asymmetry/injected.pyd",
    ],
)
def test_source_guard_rejects_untracked_and_ignored_packaged_files(
    tmp_path: Path, filename: str
) -> None:
    import subprocess

    source = tmp_path / "src/asymmetry"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.so\n*.pyd\n__pycache__/\n", encoding="utf-8")
    for arguments in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "source",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True)
    commit = release.verify_source(tmp_path)
    assert len(commit) == 40
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"cache")
    assert release.verify_source(tmp_path) == commit
    added = tmp_path / filename
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_bytes(b"unreviewed source")
    with pytest.raises(ValueError, match="clean committed|unreviewed file"):
        release.verify_source(tmp_path)


@pytest.mark.parametrize("correct", [False, True])
def test_inno_probe_precedes_real_compile_and_records_actual_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, correct: bool
) -> None:
    import hashlib
    import json

    programs = tmp_path / "Programs"
    compiler = programs / "Inno Setup 6/ISCC.exe"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"compiler executable")
    (compiler.parent / "ISCmplr.dll").write_bytes(b"compiler engine")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "tools.json").write_text(json.dumps({"inno": "6.4.3"}), encoding="utf-8")
    (tmp_path / "build").mkdir()
    monkeypatch.setenv("ProgramFiles(x86)", str(programs))
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "INPUTS", inputs)
    calls = []

    def compile(command: list, cwd: Path) -> str:
        calls.append(command)
        assert command[0] == str(compiler)
        if len(calls) == 1:
            assert "Output=no" in Path(command[1]).read_text(encoding="utf-8")
            return "Compiler engine version: Inno Setup " + ("6.4.3" if correct else "6.5.0") + "\n"
        assert command[1:] == ["real-installer.iss"]
        return "compiled"

    monkeypatch.setattr(release, "run", compile)
    if not correct:
        with pytest.raises(ValueError, match="Inno Setup"):
            release.inno_compile(["real-installer.iss"])
        assert len(calls) == 1
        assert not (tmp_path / "build/inno-observed.json").exists()
    else:
        release.inno_compile(["real-installer.iss"])
        assert len(calls) == 2
        record = json.loads((tmp_path / "build/inno-observed.json").read_text(encoding="utf-8"))
        assert record["components"] == {
            "ISCC.exe": hashlib.sha256(b"compiler executable").hexdigest(),
            "ISCmplr.dll": hashlib.sha256(b"compiler engine").hexdigest(),
        }


def test_mutable_observation_cannot_bless_dependency_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    tools = json.loads((ROOT / "packaging/release/tools.json").read_text(encoding="utf-8"))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (inputs / "windows-archives.json").write_text("[]", encoding="utf-8")
    requirements = inputs / "windows-x64.txt"
    requirements.write_text("pip==25.3 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.28.0"\n', encoding="utf-8")
    work = tmp_path / "replay-1"
    work.mkdir()
    actual = observation()
    actual["distributions"]["pip"] = "26.0"
    record = {
        "target": "windows-x64",
        "source_commit": "b" * 40,
        "input_hashes": {path.name: release.digest(path) for path in inputs.iterdir()},
        "environment": {"python": actual["python"], "distributions": actual["distributions"]},
        "wheels": {},
    }
    (work / "observed.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "INPUTS", inputs)
    monkeypatch.setattr(release, "verify_source", lambda root: "b" * 40)
    monkeypatch.setattr(release, "observe", lambda: actual)
    with pytest.raises(ValueError, match="observation differs"):
        release.verify_build(tmp_path)
    record["input_hashes"] = {}
    (work / "observed.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="input record is incomplete"):
        release.verify_build(tmp_path)


def test_reviewed_input_bytes_survive_autocrlf_checkout(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "source"
    source.mkdir()
    (source / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    relative_paths = [path.relative_to(ROOT) for path in (ROOT / "packaging/release").iterdir()]
    relative_paths.append(Path("THIRD_PARTY_LICENSES.md"))
    for relative in relative_paths:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    for arguments in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "inputs",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=source, check=True, capture_output=True)
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(source), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"], cwd=checkout, check=True, capture_output=True
    )
    subprocess.run(["git", "checkout", "HEAD"], cwd=checkout, check=True, capture_output=True)
    for relative in relative_paths:
        assert release.digest(checkout / relative) == release.digest(ROOT / relative)
        assert b"\r\n" not in (checkout / relative).read_bytes()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_packaged_smoke_cli_waits_and_propagates_failure(tmp_path: Path, exit_code: int) -> None:
    import os
    import subprocess
    import sys

    marker = tmp_path / "finished"
    script = tmp_path / "delayed.py"
    script.write_text(
        "import os, pathlib, sys, time\n"
        "assert 'PYTHONPATH' not in os.environ\n"
        "assert 'PYTHONHOME' not in os.environ\n"
        "assert pathlib.Path.cwd() == pathlib.Path(sys.argv[1])\n"
        "time.sleep(0.2)\n"
        "pathlib.Path(sys.argv[2]).write_text('finished')\n"
        "sys.exit(int(sys.argv[3]))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/release_environment.py"),
            "--smoke",
            sys.executable,
            "-I",
            str(script),
            str(tmp_path),
            str(marker),
            str(exit_code),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "unreviewed", "PYTHONHOME": "unreviewed"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert marker.read_text() == "finished"
    assert (result.returncode == 0) is (exit_code == 0)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_release_command_rejects_unbounded_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        release.run([sys.executable, "-c", "raise AssertionError"], tmp_path, timeout=timeout)


def test_release_command_timeout_kills_and_reaps_owned_child(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        release.run([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, timeout=0.1)
    cleanup = json.loads(caught.value.__notes__[0].split(": ", 1)[1])
    assert cleanup == {"child_reaped": True, "tree_cleanup_exit": 0}


def test_release_command_nonzero_status_remains_failure(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        release.run([sys.executable, "-c", "print('failure');raise SystemExit(7)"], tmp_path)
    assert caught.value.returncode == 7
    assert caught.value.output.strip() == "failure"
