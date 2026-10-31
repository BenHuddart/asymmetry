from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True, {}))


def test_publish_needs_validation_of_the_triggering_commit() -> None:
    jobs = _workflow("release.yml")["jobs"]
    assert "validate" in jobs["release"]["needs"]
    assert jobs["validate"]["uses"] == "./.github/workflows/full-suite.yml"
    assert not jobs["validate"].get("with")


def test_every_release_job_gates_publication() -> None:
    jobs = _workflow("release.yml")["jobs"]
    pending = list(jobs["release"]["needs"])
    ancestors = set()
    while pending:
        name = pending.pop()
        assert name in jobs
        if name in ancestors:
            continue
        ancestors.add(name)
        needs = jobs[name].get("needs", [])
        pending.extend([needs] if isinstance(needs, str) else needs)
    assert ancestors == set(jobs) - {"release"}


@pytest.mark.parametrize("filename", ["release.yml", "full-suite.yml"])
def test_no_status_function_bypasses_the_gate(filename: str) -> None:
    for job in _workflow(filename)["jobs"].values():
        for entry in [job, *job.get("steps", [])]:
            assert "continue-on-error" not in entry
            assert not re.search(r"\b(always|cancelled|failure)\s*\(", str(entry.get("if", "")))


def test_validation_cannot_be_skipped_or_hang() -> None:
    job = _workflow("full-suite.yml")["jobs"]["full-suite"]
    assert "if" not in job
    assert 0 < job["timeout-minutes"] <= 60


def test_validation_runs_full_tier_when_called() -> None:
    workflow = _workflow("full-suite.yml")
    assert not _triggers(workflow)["workflow_call"]
    job = workflow["jobs"]["full-suite"]
    run = next(step["run"] for step in job["steps"] if step["name"] == "Run full test suite")
    assert run == "python tools/harness.py test --tier=${{ inputs.tier || 'full' }}"
    assert "github.event.inputs" not in yaml.safe_dump(workflow)
    assert _triggers(workflow)["workflow_dispatch"]["inputs"]["tier"]["default"] == "full"


@pytest.mark.parametrize("filename", ["release.yml", "full-suite.yml"])
def test_checkouts_use_the_triggering_commit(filename: str) -> None:
    for job in _workflow(filename)["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert not {"ref", "repository"} & step.get("with", {}).keys()


def test_only_tag_push_publishes() -> None:
    workflow = _workflow("release.yml")
    assert workflow["jobs"]["release"]["if"] == "github.event_name == 'push'"
    assert _triggers(workflow)["push"] == {"tags": ["v*"]}
    assert "workflow_dispatch" in _triggers(workflow)
    assert "push" not in _triggers(_workflow("full-suite.yml"))
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_builds_use_verified_version_in_rehearsals() -> None:
    jobs = _workflow("release.yml")["jobs"]
    assert jobs["verify-version"]["outputs"]["version"] == "${{ steps.version.outputs.version }}"
    for name in ("build-macos-arm", "build-windows"):
        steps = jobs[name]["steps"]
        assert not any(step.get("name") == "Extract version" for step in steps)
        assert "needs.verify-version.outputs.version" in yaml.safe_dump(steps)
        assert "GITHUB_REF_NAME" not in yaml.safe_dump(steps)


def test_workflow_edits_run_standard_ci() -> None:
    steps = _workflow("ci.yml")["jobs"]["changes"]["steps"]
    filters = yaml.safe_load(
        next(step for step in steps if step.get("id") == "filter")["with"]["filters"]
    )
    assert ".github/workflows/**" in filters["code"]


def test_cut_release_tag_push_triggers_release() -> None:
    workflow = _workflow("cut-release.yml")
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["token"] == "${{ secrets.RELEASE_TOKEN }}"
    assert any("git push --atomic" in step.get("run", "") for step in steps)


@pytest.mark.parametrize(
    ("ref_type", "ref_name", "version", "rolled", "succeeds"),
    [
        ("tag", "v1.2.3", "1.2.3", True, True),
        ("tag", "v1.2.4", "1.2.3", True, False),
        ("branch", "fix/rehearsal", "1.2.3", True, True),
        ("tag", "v1.2.3", "1.2.3", False, False),
        ("tag", "v1.2.3-dev", "1.2.3-dev", False, True),
    ],
)
def test_version_guard_in_tag_and_rehearsal_contexts(
    tmp_path: Path, ref_type: str, ref_name: str, version: str, rolled: bool, succeeds: bool
) -> None:
    steps = _workflow("release.yml")["jobs"]["verify-version"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "version")
    (tmp_path / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        f"## [{version}] - 2000-01-01\n" if rolled else "## [Unreleased]\n", encoding="utf-8"
    )
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=dict(
            os.environ,
            GITHUB_REF_TYPE=ref_type,
            GITHUB_REF_NAME=ref_name,
            GITHUB_OUTPUT=str(output),
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
    if succeeds:
        assert output.read_text(encoding="utf-8").strip() == f"version={version}"
