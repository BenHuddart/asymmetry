"""Publication rejects substituted artifacts, missing matrix evidence, and broad trust."""

import hashlib
import itertools
import json
import tarfile
import zipfile
from io import BytesIO

import pytest

from tests.tools.test_distribution_smoke import artifact as artifact
from tools import pypi_release as publication

COMMIT = "a" * 40


@pytest.fixture
def candidate(artifact, tmp_path, monkeypatch):
    source, sdist, wheel = artifact
    readme = "# Asymmetry\n\nSynthetic μSR documentation.\n"
    (source / "README.md").write_text(readme)
    packet = tmp_path / "candidate"
    dist = packet / "dist"
    dist.mkdir(parents=True)
    for path in (sdist, wheel):
        if path == wheel:
            with zipfile.ZipFile(path) as archive:
                files = {n: archive.read(n) for n in archive.namelist()}
            key = "asymmetry-1.0.dist-info/METADATA"
            files[key] += b"Description-Content-Type: text/markdown\n\n" + readme.encode()
            with zipfile.ZipFile(dist / path.name, "w") as archive:
                for name, content in files.items():
                    archive.writestr(name, content)
        else:
            with tarfile.open(path) as archive:
                files = {n: archive.extractfile(n).read() for n in archive.getnames()}
            key = "asymmetry-1.0/PKG-INFO"
            files[key] += b"Description-Content-Type: text/markdown\n\n" + readme.encode()
            with tarfile.open(dist / path.name, "w:gz") as archive:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, BytesIO(content))
    monkeypatch.setattr(publication, "verify_source", lambda root: COMMIT)
    publication.record_candidate(source, packet, COMMIT)
    return source, packet


@pytest.mark.parametrize(
    "defect", [None, "commit", "bytes", "extra", "readme", "record", "checksums"]
)
def test_candidate_identity_rejects_substitution(candidate, defect):
    source, packet = candidate
    commit = COMMIT
    if defect == "commit":
        commit = "b" * 40
    elif defect == "bytes":
        (wheel,) = (packet / "dist").glob("*.whl")
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("asymmetry/substitution.py", "unreviewed = True")
    elif defect == "extra":
        (packet / "dist/extra.whl.metadata").touch()
    elif defect == "readme":
        (source / "README.md").write_text("changed description")
    elif defect == "record":
        (packet / "python-release.json").write_text("{}")
    elif defect == "checksums":
        (packet / "SHA256SUMS").write_text("changed checksums")
    if defect is None:
        identity = publication.verify_candidate(source, packet, commit)
        assert identity["source_commit"] == COMMIT
        assert "μSR" in (packet / "README.html").read_text()
        assert (
            identity["readme_sha256"]
            == hashlib.sha256((source / "README.md").read_bytes()).hexdigest()
        )
    else:
        with pytest.raises((ValueError, AssertionError)):
            publication.verify_candidate(source, packet, commit)


@pytest.mark.parametrize(
    "defect", [None, "commit", "artifact", "resolution", "profile", "python", "modules", "missing"]
)
def test_matrix_requires_same_bytes_and_all_supported_targets(candidate, tmp_path, defect):
    source, packet = candidate
    identity = publication.verify_candidate(source, packet, COMMIT)
    reports = tmp_path / "reports"
    for python, resolution in itertools.product(publication.PYTHONS, publication.RESOLUTIONS):
        report = reports / f"python-installs-{python}-{resolution}"
        report.mkdir(parents=True)
        manifest = {
            "source_head": COMMIT,
            "artifacts": identity["artifacts"],
            "resolution": resolution,
            "contract": {"version": "1.0"},
            "targets": [
                {"profile": name, "version": python + ".1", "required_modules": modules}
                for name, (_, modules) in publication.INSTALL_PROFILES.items()
            ],
        }
        if python == "3.13" and resolution == "unconstrained":
            if defect == "commit":
                manifest["source_head"] = "b" * 40
            elif defect == "artifact":
                manifest["artifacts"] = {}
            elif defect == "resolution":
                manifest["resolution"] = "constrained"
            elif defect == "profile":
                manifest["targets"].pop()
            elif defect == "python":
                manifest["targets"][0]["version"] = "3.12.1"
            elif defect == "modules":
                manifest["targets"][1]["required_modules"] = []
            elif defect == "missing":
                continue
        (report / "manifest.json").write_text(json.dumps(manifest))
    if defect is None:
        publication.verify_candidate(source, packet, COMMIT, reports)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            publication.verify_candidate(source, packet, COMMIT, reports)


@pytest.mark.parametrize(
    "defect", [None, "name", "unrestricted", "reviewer", "branch", "additional", "truncated"]
)
def test_environment_requires_maintainer_review_and_only_release_tags(defect):
    environment = {
        "name": "pypi",
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True},
        "protection_rules": [
            {
                "type": "required_reviewers",
                "reviewers": [{"type": "User", "reviewer": {"login": "BenHuddart"}}],
            }
        ],
    }
    policies = {"total_count": 1, "branch_policies": [{"name": "v*", "type": "tag"}]}
    if defect == "name":
        environment["name"] = "other"
    elif defect == "unrestricted":
        environment["deployment_branch_policy"] = None
    elif defect == "reviewer":
        environment["protection_rules"] = []
    elif defect == "branch":
        policies["branch_policies"][0]["type"] = "branch"
    elif defect == "additional":
        policies["branch_policies"].append({"name": "main", "type": "branch"})
    elif defect == "truncated":
        policies["total_count"] = 2
    if defect is None:
        publication.verify_publisher_environment(environment, policies)
    else:
        with pytest.raises(ValueError):
            publication.verify_publisher_environment(environment, policies)
