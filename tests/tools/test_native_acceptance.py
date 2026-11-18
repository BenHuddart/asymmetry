"""Delivered native acceptance shares the existing final-media lifecycle."""

import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.test_release_artifacts import build as payload_build
from tools import native_acceptance as native
from tools import release_artifacts as artifacts

build = payload_build


def test_final_media_visitor_executes_actual_installed_payload_and_checked_cleanup(
    build, tmp_path, monkeypatch
):
    root, bundle, binary, _, _, _ = build
    calls = []
    original = artifacts.release.run

    def native(command, cwd, **kwargs):
        calls.append(Path(command[0]).name)
        if Path(command[0]).name == "unins000.exe":
            shutil.rmtree(Path(command[0]).parent)
            return ""
        return original(command, cwd, **kwargs)

    monkeypatch.setattr(artifacts.release, "run", native)
    record = artifacts.read_json(root / "dist/windows-x64-build-components.json")

    def journey(application, work):
        assert application != bundle
        assert (application / "Asymmetry.exe").read_bytes() == (
            bundle / "Asymmetry.exe"
        ).read_bytes()
        calls.append("scientific journey")

    result = artifacts.inspect_payload(
        binary,
        record["components"],
        "windows-x64",
        {item["sha256"] for item in record["assembly"]["additional_resources"]},
        journey,
    )
    assert result == record["assembly"]
    assert calls[-2:] == ["scientific journey", "unins000.exe"]


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def packet(build, tmp_path, monkeypatch):
    root, _, _, _, _, _ = build
    monkeypatch.setattr(native, "artifacts", artifacts)
    (root / "tools").mkdir()
    for name in native.TOOLS:
        shutil.copyfile(ROOT / "tools" / name, root / "tools" / name)
    subprocess.run(["git", "add", "tools"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "driver tools",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    identity = artifacts.expected_identity("windows-x64", commit)
    directory = tmp_path / "packet"
    (directory / "inputs").mkdir(parents=True)
    for name in native.TOOLS:
        shutil.copyfile(root / "tools" / name, directory / name)
    fixtures = {}
    for name in native.FIXTURES:
        (directory / "inputs" / name).write_bytes(b"synthetic boundary bytes")
        fixtures[name] = artifacts.release.digest(directory / "inputs" / name)
    artifacts.write_json(directory / "inputs/fixtures.json", fixtures)
    artifacts.write_json(
        directory / "expectation.json",
        {
            "identity": identity,
            "notice_sha256": artifacts.release.digest(root / "THIRD_PARTY_LICENSES.md"),
        },
    )
    source = {
        "identity": native.source_identity(identity),
        "mode": "source",
        "success": True,
        "stages": ["complete"],
        "fixtures": fixtures,
        "results": {name: {} for name in native.FIXTURES},
    }
    artifacts.write_json(directory / "source-result.json", source)
    return directory, identity, commit


def test_source_verifier_binds_packet_tools_and_expected_identity(packet):
    directory, _, commit = packet
    native.verify_source_packet(directory, "windows-x64", commit)


@pytest.mark.parametrize(
    "change", ["tool", "identity", "notice", "source", "fixture", "extra", "symlink"]
)
def test_source_verifier_rejects_transport_or_authority_changes(packet, change):
    directory, _, commit = packet
    if change == "tool":
        (directory / native.TOOLS[0]).write_text("unreviewed driver")
    elif change in {"identity", "notice"}:
        record = artifacts.read_json(directory / "expectation.json")
        if change == "identity":
            record["identity"]["source_commit"] = "0" * 40
        else:
            record["notice_sha256"] = "0" * 64
        artifacts.write_json(directory / "expectation.json", record)
    elif change == "source":
        record = artifacts.read_json(directory / "source-result.json")
        record["success"] = False
        artifacts.write_json(directory / "source-result.json", record)
    elif change == "fixture":
        (directory / "inputs/counts.nxs").write_bytes(b"changed")
    elif change == "extra":
        (directory / "rescue.py").touch()
    else:
        (directory / "rescue").symlink_to(directory / native.TOOLS[0])
    with pytest.raises(ValueError):
        native.verify_source_packet(directory, "windows-x64", commit)


def test_download_verifier_accepts_source_bound_expectation_without_checkout(build, monkeypatch):
    root, _, _, commit, _, _ = build
    identity = artifacts.expected_identity("windows-x64", commit)
    notice = artifacts.release.digest(root / "THIRD_PARTY_LICENSES.md")
    monkeypatch.setattr(artifacts, "ROOT", Path("absent-source"))
    assert artifacts.verify_downloaded(root / "dist", identity, notice)


def test_driver_clean_host_probe_runs_without_scientific_dependencies(tmp_path):
    workspace, temporary = tmp_path / "workspace", tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()
    script = f"import sys;sys.path.insert(0,{str(ROOT / 'tools')!r});from native_acceptance import clean_host;from pathlib import Path;print(clean_host(Path({str(workspace)!r}),Path({str(temporary)!r})))"
    result = subprocess.run(
        [sys._base_executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "absent" in result.stdout


@pytest.mark.parametrize("owner", [".git", "src", ".venv", "release-replay"])
def test_clean_host_rejects_checkout_or_build_environment(tmp_path, owner):
    workspace, temporary = tmp_path / "workspace", tmp_path / "temporary"
    workspace.mkdir()
    temporary.mkdir()
    ((temporary if owner == "release-replay" else workspace) / owner).mkdir()
    with pytest.raises(ValueError, match="checkout or build"):
        native.clean_host(workspace, temporary)


@pytest.mark.parametrize("failure", ["science", "cleanup", "both"])
def test_payload_failure_preserves_primary_and_native_cleanup(build, monkeypatch, failure):
    root, _, binary, _, _, _ = build
    record = artifacts.read_json(root / "dist/windows-x64-build-components.json")
    original = artifacts.release.run
    calls = []

    def command(args, cwd, **kwargs):
        calls.append(args)
        if Path(args[0]).name == "unins000.exe" and failure in {"cleanup", "both"}:
            raise OSError("synthetic cleanup failure")
        return original(args, cwd, **kwargs)

    def visit(application, work):
        if failure in {"science", "both"}:
            raise RuntimeError("synthetic scientific failure")

    monkeypatch.setattr(artifacts.release, "run", command)
    events = []
    with pytest.raises(
        ExceptionGroup if failure == "both" else (RuntimeError if failure == "science" else OSError)
    ) as caught:
        artifacts.inspect_payload(
            binary,
            record["components"],
            "windows-x64",
            {record["assembly"]["additional_resources"][0]["sha256"]},
            visit,
            events=events,
        )
    assert Path(calls[-1][0]).name == "unins000.exe"
    assert events[-1] == {"stage": "temporary cleanup", "success": True}
    if failure == "both":
        assert [type(exc) for exc in caught.value.exceptions] == [RuntimeError, OSError]


@pytest.mark.parametrize("failure", ["attach", "visitor", "detach", "both"])
def test_shared_dmg_lifecycle_detaches_after_partial_attach_and_scientific_timeout(
    tmp_path, monkeypatch, failure
):
    mounted = set()
    calls = []
    monkeypatch.setattr(Path, "is_mount", lambda path: path in mounted)

    def command(args, cwd, **kwargs):
        calls.append(args[1])
        if args[1] == "info":
            return plistlib.dumps(
                {
                    "images": [
                        {
                            "image-path": str(tmp_path / "app.dmg"),
                            "system-entities": [{"dev-entry": "/dev/disk77"}],
                        }
                    ]
                    if mounted
                    else []
                }
            ).decode()
        if args[1] == "attach":
            mount = Path(args[args.index("-mountpoint") + 1])
            mounted.add(mount)
            (mount / "Asymmetry.app").mkdir()
            (mount / "Applications").symlink_to("/Applications")
            if failure == "attach":
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        else:
            if failure in {"detach", "both"}:
                raise OSError("detach failed")
            assert args[2] == "/dev/disk77"
            mounted.clear()
        return ""

    def visit(application, work):
        if failure in {"visitor", "both"}:
            raise subprocess.TimeoutExpired(["scientific launcher"], 60)

    monkeypatch.setattr(artifacts.release, "run", command)
    events = []
    expected = ExceptionGroup if failure in {"detach", "both"} else subprocess.TimeoutExpired
    with pytest.raises(expected):
        artifacts.inspect_payload(
            tmp_path / "app.dmg", [], "macos-arm64", set(), visit, events=events
        )
    assert [name for name in calls if name != "info"] == ["attach", "detach"]
    assert events[-2]["stage"] == "native cleanup"
    assert events[-2]["success"] == (failure not in {"detach", "both"})


@pytest.mark.parametrize(
    "failure", ["timeout", "zero", "stage", "silent", "loader", "expected", "stale", "identity"]
)
def test_omission_requires_actual_nonzero_expected_stage_or_specific_loader_diagnostic(
    tmp_path, monkeypatch, failure
):
    outcome = {"success": False, "exit_code": 9, "failure": "missing report", "token": "fresh"}
    if failure == "timeout":
        outcome.update(exit_code=-9, failure="timeout")
    elif failure == "zero":
        outcome["exit_code"] = 0
    if failure in {"stage", "expected", "stale", "identity"}:
        outcome["failure"] = (
            "stale or mismatched report" if failure == "stale" else "scientific stage failed"
        )
        outcome["report"] = {
            "token": "old" if failure == "stale" else "fresh",
            "identity": {"wrong": True} if failure == "identity" else {},
            "failure": {"stage": "wrong" if failure == "stage" else "counts.nxs:load"},
        }
    elif failure != "silent":
        outcome["launcher_output_tail"] = "unrelated startup failure"

    def child(*args):
        tmp_path.mkdir(exist_ok=True)
        artifacts.write_json(tmp_path / "outcome.json", outcome)
        (tmp_path / "launcher-error.log").write_text(
            "dyld[123]: Library not loaded: /private/path/libmfhdf.0.dylib"
        )
        (tmp_path / "launcher.log").touch()
        raise RuntimeError("child failed")

    monkeypatch.setattr(native.desktop, "run_acceptance", child)
    arguments = (
        [],
        tmp_path,
        {},
        tmp_path,
        {},
        "omission",
        "counts.nxs:load",
        "libmfhdf.0.dylib" if failure == "loader" else None,
    )
    if failure in {"loader", "expected"}:
        result = native.execute_case(*arguments)
        if failure == "loader":
            assert result["native_loader_failure"] == {
                "component": "libmfhdf.0.dylib",
                "marker": "dyld[",
            }
            assert "private" not in str(result)
    else:
        with pytest.raises(ValueError):
            native.execute_case(*arguments)


@pytest.mark.parametrize("mounted", [False, True])
def test_partial_attach_cleans_actual_device_and_checks_absence(tmp_path, monkeypatch, mounted):
    binary = tmp_path / "app.dmg"
    attached = []
    calls = []
    monkeypatch.setattr(Path, "is_mount", lambda path: bool(attached) and mounted)

    def command(args, cwd, **kwargs):
        calls.append(args)
        if args[1] == "info":
            return plistlib.dumps({"images": attached}).decode()
        if args[1] == "attach":
            attached.append(
                {
                    "image-path": str(binary),
                    "system-entities": [{"dev-entry": "/dev/disk77"}, {"dev-entry": "/dev/disk78"}],
                }
            )
            raise subprocess.TimeoutExpired(args, 60)
        assert args[1:] == ["detach", "/dev/disk77"]
        attached.clear()
        return ""

    monkeypatch.setattr(artifacts.release, "run", command)
    events = []
    with pytest.raises(subprocess.TimeoutExpired):
        artifacts.inspect_payload(binary, [], "macos-arm64", set(), events=events)
    assert not attached
    assert sum(args[1] == "detach" for args in calls) == 1
    assert events[-2:] == [
        {"stage": "native cleanup", "success": True},
        {"stage": "temporary cleanup", "success": True},
    ]


def test_preexisting_image_attachment_is_never_detached(tmp_path, monkeypatch):
    binary = tmp_path / "app.dmg"

    def command(args, cwd, **kwargs):
        assert args == ["hdiutil", "info", "-plist"]
        return plistlib.dumps(
            {
                "images": [
                    {"image-path": str(binary), "system-entities": [{"dev-entry": "/dev/disk77"}]}
                ]
            }
        ).decode()

    monkeypatch.setattr(artifacts.release, "run", command)
    with pytest.raises(ValueError, match="pre-existing"):
        artifacts.inspect_payload(binary, [], "macos-arm64", set())


@pytest.mark.parametrize("failure", [None, "validation", "cleanup", "both"])
def test_payload_copy_keeps_primary_failure_and_cleanup_disposition(tmp_path, monkeypatch, failure):
    application = tmp_path / "application"
    application.mkdir()
    (application / "reader.bin").write_bytes(b"original delivered bytes")
    (application / "reader-link").symlink_to("reader.bin")
    work = tmp_path / "work"
    work.mkdir()
    cases = {}
    original = shutil.rmtree

    def remove(path, *args, **kwargs):
        if Path(path).parent == work and failure in {"cleanup", "both"}:
            raise OSError("owned copy cleanup failed")
        return original(path, *args, **kwargs)

    def journey():
        with native.payload_copy(application, work, cases, "copy-control") as clone:
            assert (clone / "reader-link").is_symlink()
            assert (clone / "reader-link").read_bytes() == b"original delivered bytes"
            if failure in {"validation", "both"}:
                raise ValueError("primary numerical mismatch")

    monkeypatch.setattr(shutil, "rmtree", remove)
    try:
        if failure is None:
            journey()
        else:
            with pytest.raises(
                ExceptionGroup
                if failure == "both"
                else (ValueError if failure == "validation" else OSError)
            ) as caught:
                journey()
            if failure == "both":
                assert [type(exc) for exc in caught.value.exceptions] == [ValueError, OSError]
        record = cases["copy-control"]
        assert record["copy_removed"] == (failure not in {"cleanup", "both"})
        assert record["copy_cleanup"]["success"] == record["copy_removed"]
        if failure in {"validation", "both"}:
            assert record["copy_primary_failure"] == "ValueError"
        assert (application / "reader.bin").read_bytes() == b"original delivered bytes"
    finally:
        original(work)


@pytest.fixture
def source_preparation(monkeypatch):
    pytest.importorskip("pyhdf.SD", exc_type=ImportError)
    pytest.importorskip("uproot")
    from asymmetry.gui.scientific_acceptance import runtime_identity

    identity = {**runtime_identity(), "target": "macos-arm64"}
    monkeypatch.setattr(native.artifacts, "expected_identity", lambda *args: identity)
    return identity


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX executable shim; native workflow checks Windows Git"
)
def test_source_packet_executes_parent_selected_git_before_artifact_isolation(
    tmp_path, monkeypatch, source_preparation
):
    selected = tmp_path / "selected-git"
    selected.mkdir()
    executable = shutil.which("git")
    marker = tmp_path / "source-git-observed"
    git = selected / "git"
    git.write_text(
        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('source authoring only')\n"
        f"os.execv({executable!r},[{executable!r},*sys.argv[1:]])\n"
    )
    git.chmod(0o755)
    monkeypatch.setenv("PATH", str(selected) + os.pathsep + os.environ["PATH"])
    directory = tmp_path / "packet"
    native.prepare_packet(directory, "macos-arm64", source_preparation["source_commit"])
    assert marker.read_text() == "source authoring only"
    report = native.artifacts.read_json(directory / "source-result.json")
    assert report["success"] and report["identity"] == native.source_identity(source_preparation)
    assert str(selected) not in native.desktop.payload_environment()["PATH"]


def test_source_packet_failure_retains_sanitized_real_entry_outcome_before_cleanup(
    tmp_path, source_preparation, capsys
):
    source_preparation["source_commit"] = "0" * 40
    with pytest.raises(RuntimeError):
        native.prepare_packet(tmp_path / "packet", "macos-arm64", "0" * 40)
    output = capsys.readouterr().out
    outcome = json.loads(output)
    assert not outcome["success"]
    assert outcome["report"]["failure"]["stage"] == "identity"
    assert outcome["report"]["failure"]["type"] == "ValueError"
    assert str(tmp_path) not in output
