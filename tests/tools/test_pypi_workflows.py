"""Release authority and byte checks remain closed to non-release paths."""

import hashlib
import os
import subprocess

import pytest

from tests.tools.test_release_workflows import _triggers, _workflow


def test_oidc_is_confined_to_reviewed_tag_upload_without_source_execution():
    workflow = _workflow("release.yml")
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {"contents": "read"}
    assert [
        name for name, job in jobs.items() if job.get("permissions", {}).get("id-token") == "write"
    ] == ["publish-pypi"]
    job = jobs["publish-pypi"]
    assert job["permissions"] == {"id-token": "write"}
    assert (
        job["if"]
        == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') && github.repository == 'BenHuddart/asymmetry'"
    )
    assert job["environment"]["name"] == "pypi"
    assert set(job["needs"]) == {
        "verify-version",
        "validate",
        "verify-artifacts",
        "accept-native",
        "verify-python",
    }
    steps = job["steps"]
    assert not any("checkout" in step.get("uses", "") for step in steps)
    assert [step["name"] for step in steps if "run" in step] == [
        "Check delivered publication bytes"
    ]
    action = steps[-1]
    assert action["uses"] == "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086"
    assert action["with"] == {
        "packages-dir": "candidate/dist/",
        "attestations": True,
        "skip-existing": False,
    }
    preflight = next(
        s
        for s in jobs["verify-version"]["steps"]
        if s["name"] == "Verify publishing environment protections"
    )
    assert preflight["if"] == "github.event_name == 'push'"
    assert "pypi_release.py environment" in preflight["run"]


def test_release_checks_same_candidate_matrix_and_dispatches_docs_after_publication():
    jobs = _workflow("release.yml")["jobs"]
    installs = jobs["install-python"]
    assert installs["strategy"]["matrix"] == {
        "python": ["3.11", "3.12", "3.13"],
        "resolution": ["constrained", "unconstrained"],
    }
    assert installs["needs"] == "build-python" and "if" not in installs
    run = next(s["run"] for s in installs["steps"] if "--artifacts" in s.get("run", ""))
    assert "--profile core --profile gui --profile agent --profile all" in run
    assert set(jobs["verify-python"]["needs"]) == {"build-python", "install-python"}
    assert "if" not in jobs["verify-python"]
    assert "publish-pypi" in jobs["release"]["needs"]
    steps = jobs["release"]["steps"]
    upload = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("softprops/"))
    dispatch = next(
        i
        for i, s in enumerate(steps)
        if s.get("name") == "Dispatch documentation after publication"
    )
    cleanup = next(i for i, s in enumerate(steps) if s.get("name") == "Delete workflow artifacts")
    assert upload < dispatch < cleanup
    assert steps[dispatch]["run"] == 'gh workflow run docs-pages.yml --ref "$GITHUB_REF_NAME"'
    assert "tags" not in _triggers(_workflow("docs-pages.yml"))["push"]


def test_failed_build_retains_diagnostics_without_publishing_partial_candidate():
    steps = _workflow("release.yml")["jobs"]["build-python"]["steps"]
    diagnostics = next(s for s in steps if s.get("name") == "Retain Python build diagnostics")
    candidate = next(s for s in steps if s.get("name") == "Retain Python candidate")
    assert diagnostics["if"] == "always()"
    assert "*.log" in diagnostics["with"]["path"]
    assert "manifest.json" in diagnostics["with"]["path"]
    assert "if" not in candidate and "continue-on-error" not in candidate
    assert diagnostics["with"]["name"] != candidate["with"]["name"]


@pytest.mark.parametrize("defect", [None, "checksum-file", "wheel", "extra", "symlink", "missing"])
def test_upload_byte_gate_rejects_changed_or_extra_payload(tmp_path, defect):
    candidate = tmp_path / "candidate"
    dist = candidate / "dist"
    dist.mkdir(parents=True)
    checksums = ""
    for name in ("asymmetry-1.0.whl", "asymmetry-1.0.tar.gz"):
        path = dist / name
        path.write_bytes(b"delivered candidate bytes")
        checksums += f"{hashlib.sha256(path.read_bytes()).hexdigest()}  dist/{name}\n"
    receipt = candidate / "SHA256SUMS"
    receipt.write_text(checksums)
    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if defect == "checksum-file":
        receipt.write_text("")
    elif defect == "wheel":
        (dist / "asymmetry-1.0.whl").write_text("substituted")
    elif defect == "extra":
        (dist / "extra.txt").touch()
    elif defect == "symlink":
        path = dist / "asymmetry-1.0.whl"
        path.unlink()
        (tmp_path / "substitute").write_bytes(b"delivered candidate bytes")
        path.symlink_to(tmp_path / "substitute")
    elif defect == "missing":
        (dist / "asymmetry-1.0.whl").unlink()
    script = next(
        s["run"] for s in _workflow("release.yml")["jobs"]["publish-pypi"]["steps"] if "run" in s
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=dict(os.environ, EXPECTED_CHECKSUMS=expected),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is (defect is None), result.stdout + result.stderr
