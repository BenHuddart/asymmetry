"""Production release record generation and verification with synthetic bundles."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import release_artifacts as artifacts

release = artifacts.release
INSPECT_ARCHIVE = artifacts.inspect_python_archive

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build(tmp_path, monkeypatch):
    root = tmp_path / "source"
    inputs = root / "packaging/release"
    inputs.mkdir(parents=True)
    for name in ("tools.json", "windows-x64.txt", "windows-archives.json"):
        (inputs / name).write_bytes((ROOT / "packaging/release" / name).read_bytes())
    (root / ".gitignore").write_text("build/\ndist/\n")
    (root / "pyproject.toml").write_text('[project]\nversion="0.28.0"\n')
    (root / "THIRD_PARTY_LICENSES.md").write_bytes(b"synthetic notices")
    (inputs / "windows-archives.json").write_text("[]")
    for args in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(artifacts, "ROOT", root)
    monkeypatch.setattr(release, "ROOT", root)
    monkeypatch.setattr(release, "INPUTS", inputs)
    monkeypatch.setattr(release, "verify_build", lambda output: None)
    identity = artifacts.expected_identity("windows-x64", commit)
    replay = tmp_path / "replay"
    (replay / "replay-1").mkdir(parents=True)
    tools = json.loads((inputs / "tools.json").read_text())
    wheels = {}
    for name, directory, version in (
        ("asymmetry", "app-wheel", "0.28.0"),
        ("gleplot", "gle-wheel", "1.7.0"),
    ):
        location = replay / "replay-1" / directory
        location.mkdir()
        wheel = location / f"{name}-{version}-py3-none-any.whl"
        wheel.write_bytes(name.encode())
        wheels[name] = {"filename": wheel.name, "sha256": release.digest(wheel)}
    monkeypatch.setattr(release, "native_identity", lambda: {"os_version": "10.0.26100"})
    observation = {
        "source_commit": commit,
        "target": "windows-x64",
        "input_hashes": identity["input_hashes"],
        "environment": {
            "python": identity["python"],
            "distributions": {
                **release.reviewed_requirements(inputs / "windows-x64.txt"),
                "asymmetry": "0.28.0",
                "gleplot": "1.7.0",
            },
        },
        "gle_commit": tools["gle"]["commit"],
        "wheels": wheels,
        "downloaded_wheels": {
            name: {
                "filename": f"{name}-{item['version']}-py3-none-any.whl",
                "sha256": item["sha256"],
            }
            for name, item in release.reviewed_wheels(inputs / "windows-x64.txt").items()
        },
        "native": {"os_version": "10.0.26100"},
    }
    reports = [
        {
            "metadata": {"name": name},
            "download_info": {
                "url": "https://files.pythonhosted.org/packages/" + item["filename"],
                "archive_info": {"hashes": {"sha256": item["sha256"]}},
            },
        }
        for name, item in observation["downloaded_wheels"].items()
    ]
    (replay / "replay-1/bootstrap-report.json").write_text(json.dumps({"install": []}))
    (replay / "replay-1/install-report.json").write_text(json.dumps({"install": reports}))
    (replay / "replay-1/observed.json").write_text(json.dumps(observation))
    output = root / "build"
    output.mkdir()
    artifacts.prepare(replay, "windows-x64", commit)
    bundle = root / "dist/Asymmetry"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "Asymmetry.exe").write_bytes(b"synthetic exe")
    (bundle / "_internal/build-identity.json").write_bytes(
        (output / "build-identity.json").read_bytes()
    )
    (bundle / "_internal/THIRD_PARTY_LICENSES.md").write_bytes(b"synthetic notices")
    artifacts.write_analysis([], [], [])
    (output / "inno-observed.json").write_text(
        json.dumps(
            {
                "version": tools["inno"],
                "components": {"ISCC.exe": "c" * 64, "ISCmplr.dll": "d" * 64},
            }
        )
    )
    monkeypatch.setattr(
        artifacts,
        "inspect_python_archive",
        lambda executable, modules: {
            "PYZ": modules,
            "CArchive_modules": [],
            "CArchive_scripts": [],
        },
    )
    # No staged external libraries in this synthetic fixture.
    monkeypatch.setattr(artifacts, "external_archives", lambda bundle: [])
    artifacts.collect(bundle, "windows-x64", commit)
    binary = root / "dist/Asymmetry-0.28.0-windows-x64-setup.exe"
    binary.write_bytes(b"final installer")
    (root / "build/icons").mkdir()
    (root / "build/icons/Asymmetry.ico").write_bytes(b"installer icon")
    original_run = release.run

    def native_runner(command, cwd, **kwargs):
        if command[0] == str(binary.resolve()):
            destination = Path(
                next(arg.removeprefix("/DIR=") for arg in command if arg.startswith("/DIR="))
            )
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            shutil.copyfile(root / "build/icons/Asymmetry.ico", destination / "asymmetry.ico")
            (destination / "unins000.exe").write_bytes(b"generated uninstaller")
            (destination / "unins000.dat").write_bytes(b"generated installation metadata")
            payload = binary.read_bytes()
            if payload == b"missing identity":
                (destination / "_internal/build-identity.json").unlink()
            elif payload == b"changed notice":
                (destination / "_internal/THIRD_PARTY_LICENSES.md").write_bytes(b"wrong")
            elif payload == b"changed icon":
                (destination / "asymmetry.ico").write_bytes(b"wrong")
            elif payload == b"stale installer":
                (destination / "Asymmetry.exe").write_bytes(b"stale executable")
            return ""
        if Path(command[0]).name == "unins000.exe":
            return ""
        return original_run(command, cwd, **kwargs)

    monkeypatch.setattr(release, "run", native_runner)
    artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")
    return root, bundle, binary, commit, observation, replay


def test_generate_verify_and_checksum_share_authority(build):
    root, _, binary, commit, _, _ = build
    selected = artifacts.verify_download(root / "dist", "windows-x64", commit)
    assert binary in selected
    record = json.loads((root / "dist/windows-x64-artifact.json").read_text())
    assert record["artifact"]["sha256"] == release.digest(binary)
    assert (
        (root / "dist/windows-x64-SHA256SUMS").read_text().startswith(record["artifact"]["sha256"])
    )


@pytest.mark.parametrize(
    "change", ["source_commit", "version", "target", "input_hashes", "python", "missing"]
)
def test_verifier_rejects_stale_or_swapped_identity(build, change):
    root, _, _, commit, _, _ = build
    path = root / "dist/windows-x64-artifact.json"
    record = json.loads(path.read_text())
    if change == "missing":
        del record["identity"]
    else:
        record["identity"][change] = "stale"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        artifacts.verify_download(root / "dist", "windows-x64", commit)


@pytest.mark.parametrize(
    "name", ["Asymmetry.exe", "_internal/THIRD_PARTY_LICENSES.md", "_internal/build-identity.json"]
)
def test_finalizer_rejects_bundle_changes(build, name):
    _, bundle, binary, commit, _, _ = build
    (bundle / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bundle"):
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")


@pytest.mark.parametrize("change", ["binary", "component", "missing", "checksum"])
def test_download_corruption_fails(build, change):
    root, _, binary, commit, _, _ = build
    path = {
        "binary": binary,
        "component": root / "dist/windows-x64-build-components.json",
        "missing": root / "dist/windows-x64-artifact.json",
        "checksum": root / "dist/windows-x64-SHA256SUMS",
    }[change]
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        artifacts.verify_download(root / "dist", "windows-x64", commit)


@pytest.mark.parametrize(
    "private",
    [
        "/home/user/private",
        "C:\\Users\\user\\secret",
        "file:///tmp/local",
        "https://user:password@example.org/a?token=secret",
    ],
)
def test_allowlisted_observation_drops_nested_private_metadata(build, private):
    root, _, _, commit, observation, replay = build
    observation["extra"] = {"url": private, "nested": [{"origin": private}]}
    observation["wheels"]["asymmetry"]["origin"] = {"url": private}
    (replay / "replay-1/observed.json").write_text(json.dumps(observation))
    artifacts.prepare(replay, "windows-x64", commit)
    public = (root / "build/build-environment.json").read_text()
    assert json.dumps(private) not in public
    assert "6a270dfa5f49c907fdfe9e4d9ec2ae5e1000172c" in public
    assert "https://github.com/BenHuddart/gleplot.git" in public


def test_bundle_symlinks_are_relative_and_bounded(tmp_path):
    (tmp_path / "Versions/A").mkdir(parents=True)
    (tmp_path / "Versions/A/lib").write_bytes(b"native")
    (tmp_path / "Versions/Current").symlink_to("A")
    inventory = artifacts.bundle_inventory(tmp_path)
    assert any(item.get("link") == "A" for item in inventory)
    (tmp_path / "escape").symlink_to("/tmp")
    with pytest.raises(ValueError, match="symlink"):
        artifacts.bundle_inventory(tmp_path)


def test_collection_rejects_missing_embedded_identity_and_notice(build):
    _, bundle, _, commit, _, _ = build
    embedded = bundle / "_internal/build-identity.json"
    saved = embedded.read_bytes()
    embedded.unlink()
    with pytest.raises(ValueError, match="packaged build identity"):
        artifacts.collect(bundle, "windows-x64", commit)
    embedded.write_bytes(saved)
    (bundle / "_internal/THIRD_PARTY_LICENSES.md").unlink()
    with pytest.raises(ValueError, match="notices"):
        artifacts.collect(bundle, "windows-x64", commit)


@pytest.mark.parametrize(
    "private",
    [
        "/home/user/private",
        "C:\\Users\\user\\secret",
        "file:///tmp/local",
        "https://user:password@github.com/x/y.git?token=secret",
    ],
)
def test_required_vcs_identity_cannot_be_dropped_or_leak_private_url(build, private):
    root, _, _, commit, _, replay = build
    path = root / "packaging/release/tools.json"
    tools = json.loads(path.read_text())
    tools["gle"]["url"] = private
    path.write_text(json.dumps(tools))
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "unsafe fixture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    observed_path = replay / "replay-1/observed.json"
    observed = json.loads(observed_path.read_text())
    observed["source_commit"] = commit
    observed["input_hashes"] = artifacts.expected_identity("windows-x64", commit)["input_hashes"]
    observed_path.write_text(json.dumps(observed))
    with pytest.raises(ValueError, match="immutable VCS identity"):
        artifacts.prepare(replay, "windows-x64", commit)


def test_expected_commit_is_checked_against_checkout(build):
    root, _, _, _, _, _ = build
    with pytest.raises(ValueError, match="triggering checkout"):
        artifacts.verify_download(root / "dist", "windows-x64", "f" * 40)


@pytest.mark.parametrize(
    "private", ["/home/user/secret", "C:\\Users\\user\\secret", "file:///tmp/local"]
)
def test_compiler_hash_fields_cannot_publish_private_paths(build, private):
    root, bundle, binary, commit, _, _ = build
    path = root / "build/inno-observed.json"
    compiler = json.loads(path.read_text())
    compiler["components"]["ISCC.exe"] = private
    path.write_text(json.dumps(compiler))
    with pytest.raises(ValueError, match="public SHA256 identity"):
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")


@pytest.mark.parametrize("field", ["index_wheels", "native_tools", "source_wheels"])
def test_preparation_rejects_stale_public_observations(build, field):
    _, _, _, commit, observation, replay = build
    if field == "index_wheels":
        observation["downloaded_wheels"] = {"pip": {"filename": "pip.whl", "sha256": "f" * 64}}
    elif field == "native_tools":
        observation["native"]["os_version"] = "stale"
    else:
        observation["wheels"]["asymmetry"]["filename"] = "stale.whl"
    (replay / "replay-1/observed.json").write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="observation|wheel"):
        artifacts.prepare(replay, "windows-x64", commit)


@pytest.mark.parametrize(
    "payload", [b"missing identity", b"changed notice", b"changed icon", b"stale installer"]
)
def test_finalizer_rejects_wrong_actual_installer_payload(build, payload):
    _, bundle, binary, commit, _, _ = build
    binary.write_bytes(payload)
    with pytest.raises(ValueError, match="payload|resource"):
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")


@pytest.mark.parametrize("in_pyz", [False, True])
def test_script_name_does_not_substitute_for_collected_module(monkeypatch, in_pyz):
    output = "Contents of 'app' (PKG/CArchive):\n 0, 12, 12, 0, 's', 'required_module'\n"
    output += "Contents of 'PYZ.pyz' (PYZ):\n typecode, position, length, name\n"
    if in_pyz:
        output += " 0, 17, 42, 'required_module'\n"
    monkeypatch.setattr(release, "run", lambda *args, **kwargs: output)
    if in_pyz:
        assert artifacts.inspect_python_archive(Path("app"), ["required_module"])["PYZ"] == [
            "required_module"
        ]
    else:
        with pytest.raises(ValueError, match="missing Analysis modules"):
            artifacts.inspect_python_archive(Path("app"), ["required_module"])


@pytest.mark.parametrize("tamper", [None, "identity", "notice", "background", "link"])
def test_native_dmg_inspection_binds_payload_and_always_detaches(build, monkeypatch, tamper):
    root, bundle, _, _, _, _ = build
    (root / "build/dmg-bg.png").write_bytes(b"background pixels")
    commands = []

    def native_runner(command, cwd, **kwargs):
        commands.append(command)
        if command[1] == "attach":
            mount = Path(command[command.index("-mountpoint") + 1])
            shutil.copytree(bundle, mount / "Asymmetry.app")
            (mount / "Applications").symlink_to("/wrong" if tamper == "link" else "/Applications")
            (mount / ".background").mkdir()
            (mount / ".background/picture.png").write_bytes(
                b"wrong" if tamper == "background" else b"background pixels"
            )
            if tamper == "identity":
                (mount / "Asymmetry.app/_internal/build-identity.json").unlink()
            elif tamper == "notice":
                (mount / "Asymmetry.app/_internal/THIRD_PARTY_LICENSES.md").write_bytes(b"wrong")
        return ""

    monkeypatch.setattr(release, "run", native_runner)
    if tamper:
        with pytest.raises(ValueError, match="payload|resource|system link"):
            artifacts.inspect_assembled_artifact(root / "final.dmg", bundle, "macos-arm64")
    else:
        record = artifacts.inspect_assembled_artifact(root / "final.dmg", bundle, "macos-arm64")
        assert record["additional_resources"][0]["path"] == ".background/picture.png"
        assert str(root) not in json.dumps(record)
    assert commands[0][1] == "attach"
    assert "-readonly" in commands[0] and "-nobrowse" in commands[0]
    assert commands[-1][1] == "detach"


def test_windows_payload_failure_still_checks_uninstaller(build, monkeypatch):
    _, bundle, binary, _, _, _ = build
    binary.write_bytes(b"missing identity")
    commands = []
    native_runner = release.run

    def recording_runner(command, cwd, **kwargs):
        commands.append(command)
        return native_runner(command, cwd, **kwargs)

    monkeypatch.setattr(release, "run", recording_runner)
    with pytest.raises(ValueError, match="payload"):
        artifacts.inspect_assembled_artifact(binary, bundle, "windows-x64")
    assert "/NORESTART" in commands[0] and "/NOCLOSEAPPLICATIONS" in commands[0]
    assert "/NORESTARTAPPLICATIONS" in commands[0] and "/TASKS=" in commands[0]
    assert Path(commands[-1][0]).name == "unins000.exe"


def test_final_bytes_cannot_change_during_payload_inspection(build, monkeypatch):
    _, bundle, binary, commit, _, _ = build
    native_runner = release.run

    def changing_runner(command, cwd, **kwargs):
        result = native_runner(command, cwd, **kwargs)
        if command[0] == str(binary.resolve()):
            binary.write_bytes(b"changed after payload read")
        return result

    monkeypatch.setattr(release, "run", changing_runner)
    with pytest.raises(ValueError, match="changed during payload inspection"):
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")


def test_cleanup_failure_cannot_be_reported_as_finalization_success(build, monkeypatch):
    _, bundle, binary, commit, _, _ = build
    native_runner = release.run

    def failing_cleanup(command, cwd, **kwargs):
        if Path(command[0]).name == "unins000.exe":
            raise subprocess.CalledProcessError(7, command)
        return native_runner(command, cwd, **kwargs)

    monkeypatch.setattr(release, "run", failing_cleanup)
    with pytest.raises(subprocess.CalledProcessError):
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")


def test_download_requires_final_payload_observation(build):
    root, _, _, commit, _, _ = build
    component = root / "dist/windows-x64-build-components.json"
    record = json.loads(component.read_text())
    del record["assembly"]
    artifacts.write_json(component, record)
    final_path = root / "dist/windows-x64-artifact.json"
    final = json.loads(final_path.read_text())
    final["build_components"] = artifacts.file_record(component)
    artifacts.write_json(final_path, final)
    (root / "dist/windows-x64-SHA256SUMS").write_text(artifacts.checksum_text(final))
    with pytest.raises(ValueError, match="payload inspection record"):
        artifacts.verify_download(root / "dist", "windows-x64", commit)


def test_public_wheel_filename_must_match_allowlisted_download_report(build):
    _, _, _, commit, observation, replay = build
    observation["downloaded_wheels"]["pip"]["filename"] = "stale.whl"
    (replay / "replay-1/observed.json").write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="wheel observation"):
        artifacts.prepare(replay, "windows-x64", commit)


@pytest.mark.parametrize("stage", ["collect", "finalize"])
@pytest.mark.parametrize(
    "private",
    [
        "/home/user/private",
        "C:\\Users\\user\\secret",
        "https://user:password@example.org/a?token=secret",
    ],
)
def test_every_public_write_projects_mutable_build_metadata(build, stage, private):
    root, bundle, binary, commit, _, _ = build
    source = root / (
        "build/build-environment.json"
        if stage == "collect"
        else "dist/windows-x64-build-components.json"
    )
    record = json.loads(source.read_text())
    record["extra"] = {"url": private, "nested": [{"path": private}]}
    record["source_wheels"]["asymmetry"]["origin"] = {"url": private}
    record["direct_sources"]["gleplot"]["metadata"] = {"url": private}
    record["native_tools"]["extra"] = {"path": private}
    artifacts.write_json(source, record)
    analysis = root / "build/bundle-analysis.json"
    value = json.loads(analysis.read_text())
    value["extra"] = {"origin": private}
    artifacts.write_json(analysis, value)
    if stage == "collect":
        artifacts.collect(bundle, "windows-x64", commit)
    else:
        record["analysis"]["extra"] = {"origin": private}
        artifacts.write_json(source, record)
        artifacts.finalize(binary, bundle, "windows-x64", commit, "0.28.0")
    public = (root / "dist/windows-x64-build-components.json").read_text()
    assert json.dumps(private) not in public
    assert "6a270dfa5f49c907fdfe9e4d9ec2ae5e1000172c" in public
    assert "https://github.com/BenHuddart/gleplot.git" in public


def test_required_identity_in_mutable_snapshot_is_rejected(build):
    root, bundle, _, commit, _, _ = build
    path = root / "build/build-environment.json"
    record = json.loads(path.read_text())
    record["direct_sources"]["gleplot"]["url"] = "file:///home/user/private"
    artifacts.write_json(path, record)
    with pytest.raises(ValueError, match="immutable.*identity"):
        artifacts.collect(bundle, "windows-x64", commit)


def test_collector_records_real_carchive_bootstrap_modules(build, monkeypatch):
    root, bundle, _, commit, _, _ = build
    artifacts.write_analysis([("struct", "ignored-host-origin", "PYMODULE")], [], [])
    monkeypatch.setattr(artifacts, "inspect_python_archive", INSPECT_ARCHIVE)
    native_runner = release.run

    def archive_runner(command, cwd, **kwargs):
        if command[0] == "pyi-archive_viewer":
            return "Contents of 'app' (PKG/CArchive):\n 0, 233, 289, 1, 'm', 'struct'\n 233, 100, 200, 1, 's', 'entry'\nContents of 'PYZ.pyz' (PYZ):\n 0, 17, 42, 'fractions'\n"
        return native_runner(command, cwd, **kwargs)

    monkeypatch.setattr(release, "run", archive_runner)
    artifacts.collect(bundle, "windows-x64", commit)
    record = json.loads((root / "dist/windows-x64-build-components.json").read_text())
    assert record["analysis"]["python_archives"] == {
        "PYZ": ["fractions"],
        "CArchive_modules": ["struct"],
        "CArchive_scripts": ["entry"],
    }
