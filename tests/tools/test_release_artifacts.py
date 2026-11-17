"""Production release record generation and verification with synthetic bundles."""

import json
import subprocess
from pathlib import Path

import pytest

from tools import release_artifacts as artifacts

release = artifacts.release

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
        "wheels": {
            "asymmetry": {"filename": "asymmetry-0.28.0-py3-none-any.whl", "sha256": "a" * 64},
            "gleplot": {"filename": "gleplot-1.7.0-py3-none-any.whl", "sha256": "b" * 64},
        },
        "downloaded_wheels": {},
        "native": {"os_version": "10.0.26100"},
    }
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
    monkeypatch.setattr(artifacts, "inspect_python_archive", lambda executable, modules: None)
    # No staged external libraries in this synthetic fixture.
    monkeypatch.setattr(artifacts, "external_archives", lambda bundle: [])
    artifacts.collect(bundle, "windows-x64", commit)
    binary = root / "dist/Asymmetry-0.28.0-windows-x64-setup.exe"
    binary.write_bytes(b"final installer")
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
    assert private not in public
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
