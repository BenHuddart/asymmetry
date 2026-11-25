from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
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
    for job_name, job in _workflow(filename)["jobs"].items():
        for entry in [job, *job.get("steps", [])]:
            assert "continue-on-error" not in entry
            if entry.get("name") == "Report memory counters":
                assert filename == "full-suite.yml" and entry is not job
                assert entry["if"] == "always()"
                assert entry["run"] == "awk '$1 == \"oom_kill\" { print }' /proc/vmstat\nfree -h\n"
                continue
            if entry.get("name") == "Retain native scientific inputs":
                targets = {"build-macos-arm": "macos-arm64", "build-windows": "windows-x64"}
                assert filename == "release.yml" and job_name in targets
                assert entry["if"] == "always()" and entry["uses"] == "actions/upload-artifact@v7"
                assert entry["with"] == {
                    "name": "release-science-inputs-" + targets[job_name],
                    "path": "${{ runner.temp }}/release-science-inputs-" + targets[job_name],
                    "if-no-files-found": "error",
                    "retention-days": 1,
                }
                continue
            if entry.get("name") == "Retain native scientific report":
                assert filename == "release.yml" and job_name == "accept-native"
                assert entry["if"] == "always()" and entry["uses"] == "actions/upload-artifact@v7"
                assert entry["with"]["path"] == "${{ runner.temp }}/native-science-result.json"
                continue
            if entry.get("name") == "Retain Python build diagnostics":
                assert filename == "release.yml" and job_name == "build-python"
                assert entry["if"] == "always()" and entry["uses"] == "actions/upload-artifact@v7"
                assert entry["with"]["name"] == "python-build-evidence"
                assert entry["with"]["path"].splitlines() == [
                    "${{ runner.temp }}/python-build/*.log",
                    "${{ runner.temp }}/python-build/manifest.json",
                    "${{ runner.temp }}/python-build/README.html",
                ]
                continue
            if entry.get("name") == "Retain candidate installation evidence":
                assert filename == "release.yml" and job_name == "install-python"
                assert entry["if"] == "always()" and entry["uses"] == "actions/upload-artifact@v7"
                assert entry["with"]["if-no-files-found"] == "error"
                continue
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
    assert (
        run.partition(" -- ")[0]
        == "python tools/harness.py test --tier=${{ inputs.tier || 'full' }}"
    )
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
    assert jobs["build-macos-arm"]["runs-on"] == "macos-15"
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
    assert {".github/workflows/**", ".github/actions/**", "packaging/**"} <= set(filters["code"])


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


@pytest.mark.io
def test_full_tier_reports_a_worker_crash_without_restarting(tmp_path: Path) -> None:
    case = tmp_path / "test_worker_crash.py"
    case.write_text("import os\ndef test_requested_worker_exit():\n    os._exit(9)\n")
    job = _workflow("full-suite.yml")["jobs"]["full-suite"]
    run = next(step["run"] for step in job["steps"] if step["name"] == "Run full test suite")
    pytest_args = shlex.split(run.partition(" -- ")[2])
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/harness.py"),
            "test",
            "--tier=full",
            "--",
            *pytest_args,
            str(case),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "test_requested_worker_exit" in output.partition("node down:")[0]
    assert "replacing crashed worker" not in output


def test_full_tier_records_memory_counters_around_the_tests() -> None:
    steps = _workflow("full-suite.yml")["jobs"]["full-suite"]["steps"]
    names = [step["name"] for step in steps]
    before = names.index("Record memory counters")
    tests = names.index("Run full test suite")
    after = names.index("Report memory counters")
    assert before < tests < after
    assert "if" not in steps[before]
    assert steps[before]["run"] == steps[after]["run"]
    assert steps[after]["if"] == "always()"


def test_standard_ci_uses_the_native_per_test_watchdog() -> None:
    steps = _workflow("ci.yml")["jobs"]["test"]["steps"]
    run = next(step["run"] for step in steps if step.get("name") == "Run test suite")
    options = shlex.split(run.partition(" -- ")[2])
    assert ("-o", "faulthandler_timeout=90") in tuple(zip(options, options[1:]))


def test_pages_uses_supported_timeout_and_preserves_failed_artifacts() -> None:
    job = _workflow("docs-pages.yml")["jobs"]["deploy-pages"]
    deploy = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/deploy-pages@")
    )
    timeout_ms = int(deploy.get("with", {}).get("timeout", 600000))
    assert 0 < timeout_ms <= 600000
    assert timeout_ms < job["timeout-minutes"] * 60000 <= 30 * 60000
    cleanup = next(step for step in job["steps"] if step["name"] == "Delete github-pages artifact")
    assert cleanup["if"] == "success() && steps.deployment.outcome == 'success'"
    upload = next(
        step
        for step in _workflow("docs-pages.yml")["jobs"]["build-docs"]["steps"]
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert 1 <= upload["with"]["retention-days"] <= 7


def test_docs_modes_share_gate_and_failed_build_cannot_upload() -> None:
    workflow = _workflow("docs-pages.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "docs-pages-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    smoke = workflow["jobs"]["docs-smoke"]
    assert "permissions" not in smoke
    assert "secrets." not in str(smoke)
    smoke_runs = [step.get("run", "") for step in smoke["steps"]]
    assert "python3 tools/harness.py docs --mode source" in smoke_runs
    assert all("--out docs/_generated/screenshots" not in run for run in smoke_runs)
    steps = workflow["jobs"]["build-docs"]["steps"]
    gate_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("run") == "python3 tools/harness.py docs --mode publish"
    )
    upload_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert gate_index < upload_index
    assert not steps[gate_index].get("continue-on-error", False)
    assert "if" not in steps[upload_index]  # GitHub's default success() applies.
    assert "continue-on-error" not in steps[upload_index]
    assert workflow["jobs"]["deploy-pages"]["needs"] == "build-docs"
    for event in ("push", "pull_request"):
        paths = _triggers(workflow)[event]["paths"]
        for changed in (
            "tools/docs_build.py",
            "tools/docs_policy.py",
            "tools/harness.py",
            "docs/conf.py",
            "tests/tools/test_docs_build.py",
            "src/asymmetry/core/transform/asymmetry.py",
        ):
            from fnmatch import fnmatch

            assert any(fnmatch(changed, pattern) for pattern in paths), changed
    filters = str(_workflow("ci.yml")["jobs"]["changes"]["steps"])
    assert "docs/conf.py" in filters and "docs/_ext/**" in filters


def test_docs_smoke_installs_qt_import_runtime_before_sphinx() -> None:
    steps = _workflow("docs-pages.yml")["jobs"]["docs-smoke"]["steps"]
    gate = next(
        i
        for i, step in enumerate(steps)
        if step.get("run") == "python3 tools/harness.py docs --mode source"
    )
    install = next(
        (step for step in steps[:gate] if "apt-get install" in step.get("run", "")), None
    )
    assert install is not None, "GUI autodoc imports require Linux Qt runtime libraries"
    assert "libegl1" in install["run"]
    assert not install.get("continue-on-error", False)


def test_wheel_job_is_always_required_and_separate_from_editable_tests() -> None:
    workflow = _workflow("ci.yml")
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    job = workflow["jobs"]["install-smoke"]
    assert "if" not in job
    assert job["strategy"]["matrix"]["python"] == ["3.11", "3.12", "3.13"]
    assert job["strategy"]["matrix"]["resolution"] == ["constrained", "unconstrained"]
    assert not job.get("continue-on-error", False)
    runs = [step["run"] for step in job["steps"] if "run" in step]
    assert runs == [
        "uv pip install build pip",
        "sudo apt-get update -q && sudo apt-get install -y libegl1",
        'python3 tools/distribution_smoke.py --output "$RUNNER_TEMP/distribution-smoke" '
        "--profile core --profile gui --profile agent --profile all "
        "--resolution ${{ matrix.resolution }}",
    ]
    assert "install-smoke" in workflow["jobs"]["ci-success"]["needs"]
    assert workflow["jobs"]["ci-success"]["if"] == "always()"
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["path"].splitlines() == [
        "${{ runner.temp }}/distribution-smoke/**/*.log",
        "${{ runner.temp }}/distribution-smoke/manifest.json",
    ]
    assert job["timeout-minutes"] <= 15


@pytest.mark.parametrize("smoke_result", ["success", "failure", "cancelled", "skipped"])
def test_required_ci_rejects_missing_core_wheel_success(tmp_path: Path, smoke_result: str) -> None:
    job = _workflow("ci.yml")["jobs"]["ci-success"]
    script = job["steps"][0]["run"]
    for name, result in {
        "changes": "success",
        "checks": "success",
        "test": "skipped",
        "install-smoke": smoke_result,
    }.items():
        script = script.replace("${{ needs." + name + ".result }}", result)
    assert "${{" not in script
    result = subprocess.run(
        ["bash", "-e", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert (result.returncode == 0) is (smoke_result == "success"), result.stdout + result.stderr


@pytest.mark.parametrize(
    "filename", ["release.yml", "preview-windows.yml", "preview-macos-arm.yml"]
)
def test_native_consumers_share_replay_and_verify_before_freezing(filename: str) -> None:
    jobs = _workflow(filename)["jobs"]
    for name, job in jobs.items():
        if name not in ("build-macos-arm", "build-windows"):
            continue
        steps = job["steps"]
        replay = next(
            step for step in steps if step.get("uses") == "./.github/actions/release-environment"
        )
        assert replay["with"]["target"] == ("windows-x64" if "windows" in name else "macos-arm64")
        verification = next(
            i
            for i, step in enumerate(steps)
            if "release_artifacts.py prepare" in step.get("run", "")
        )
        freeze = next(
            i
            for i, step in enumerate(steps)
            if step.get("name") == "Build application with PyInstaller"
        )
        assert verification < freeze
        smoke = next(step for step in steps if step.get("name") == "Smoke-test packaged app")
        assert smoke["working-directory"] == "${{ runner.temp }}"
        assert "GITHUB_WORKSPACE" in smoke["run"]
        assert 'tools/release_environment.py" --smoke' in smoke["run"]
        assert "--smoke-test --smoke-test-root" in smoke["run"]
        if "macos" in name:
            assert job["runs-on"] == "macos-15"
        assert not any("pip install" in step.get("run", "") for step in steps)
        assert not any("python-version" in step.get("with", {}) for step in steps)


def test_shared_bootstrap_selects_reviewed_python_and_compiler() -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/release-environment/action.yml").read_text(encoding="utf-8")
    )
    steps = action["runs"]["steps"]
    setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "${{ steps.tools.outputs.python }}"
    install = next(step for step in steps if step.get("name") == "Install reviewed Inno Setup")
    assert "--version=${{ steps.tools.outputs.inno }}" in install["run"]
    assert "--force" not in shlex.split(install["run"])
    assert "--allow-downgrade" in shlex.split(install["run"])
    assert install["if"] == "inputs.target == 'windows-x64'"


def test_downloaded_artifacts_are_verified_on_dispatch_and_before_publication() -> None:
    jobs = _workflow("release.yml")["jobs"]
    verifier = jobs["verify-artifacts"]
    assert "if" not in verifier
    assert set(verifier["needs"]) == {"verify-version", "build-windows", "build-macos-arm"}
    assert verifier["permissions"] == {"contents": "read", "actions": "read"}
    assert "verify-artifacts" in jobs["release"]["needs"]
    for job in (verifier, jobs["release"]):
        steps = job["steps"]
        downloads = [
            s
            for s in steps
            if s.get("uses", "").startswith("actions/download-artifact@")
            and s.get("with", {}).get("pattern") == "asymmetry-*"
        ]
        assert len(downloads) == 1
        assert downloads[0]["with"]["pattern"] == "asymmetry-*"
        verify = next(s for s in steps if s.get("name") == "Verify downloaded publication assets")
        assert "tools/release_artifacts.py verify" in verify["run"]
        assert '--commit "$GITHUB_SHA"' in verify["run"]
        assert "macos-arm64 windows-x64" in verify["run"]
        assert steps.index(downloads[0]) < steps.index(verify)
    release_steps = jobs["release"]["steps"]
    upload = next(s for s in release_steps if s.get("uses", "").startswith("softprops/"))
    verify = next(
        s for s in release_steps if s.get("name") == "Verify downloaded publication assets"
    )
    assert release_steps.index(verify) < release_steps.index(upload)
    assert "**/*" not in upload["with"]["files"]
    assert "build-components.json" in upload["with"]["files"]
    assert "artifact.json" in upload["with"]["files"]


@pytest.mark.parametrize(
    "filename", ["release.yml", "preview-macos-arm.yml", "preview-windows.yml"]
)
def test_native_records_follow_assembly_and_share_packaging_owner(filename: str) -> None:
    for name, job in _workflow(filename)["jobs"].items():
        if name not in ("build-macos-arm", "build-windows"):
            continue
        steps = job["steps"]
        commands = [s.get("run", "") for s in steps]
        prepare = next(
            i for i, command in enumerate(commands) if "release_artifacts.py prepare" in command
        )
        collect = next(
            i for i, command in enumerate(commands) if "release_artifacts.py collect" in command
        )
        finalize = next(
            i for i, command in enumerate(commands) if "release_artifacts.py finalize" in command
        )
        freeze = next(
            i for i, s in enumerate(steps) if s.get("name") == "Build application with PyInstaller"
        )
        assembly = next(
            i for i, s in enumerate(steps) if s.get("name") in ("Build DMG", "Build EXE installer")
        )
        upload = next(
            i
            for i, s in enumerate(steps)
            if s.get("uses", "").startswith("actions/upload-artifact@")
            and "build-components.json" in s.get("with", {}).get("path", "")
        )
        assert prepare < freeze < collect < assembly < finalize < upload
        assert "build-components.json" in steps[upload]["with"]["path"]
        assert "artifact.json" in steps[upload]["with"]["path"]
        assert "SHA256SUMS" in steps[upload]["with"]["path"]


def test_clean_native_science_jobs_gate_dispatch_and_publication_without_checkout():
    jobs = _workflow("release.yml")["jobs"]
    job = jobs["accept-native"]
    assert "accept-native" in jobs["release"]["needs"]
    assert set(job["needs"]) == {"verify-artifacts", "verify-version"}
    assert "if" not in job and job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["include"] == [
        {"target": "windows-x64", "os": "windows-2025", "architecture": "x64"},
        {"target": "macos-arm64", "os": "macos-15", "architecture": "arm64"},
    ]
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    assert not any("checkout" in step.get("uses", "") for step in job["steps"])
    run = next(
        step["run"]
        for step in job["steps"]
        if step["name"] == "Execute delivered scientific journey and actual omissions"
    )
    assert "venv --without-pip" in run and "-I -B" in run
    assert 'native_acceptance.py" run' in run and '--commit "$GITHUB_SHA"' in run
    verification = next(
        step["run"]
        for step in jobs["verify-artifacts"]["steps"]
        if step["name"] == "Verify downloaded publication assets"
    )
    assert "tools/native_acceptance.py verify" in verification
    for name in ("build-macos-arm", "build-windows"):
        steps = jobs[name]["steps"]
        source = next(
            i
            for i, step in enumerate(steps)
            if "tools/native_acceptance.py prepare" in step.get("run", "")
        )
        identity = next(
            i
            for i, step in enumerate(steps)
            if step.get("name") == "Prepare packaged build identity"
        )
        freeze = next(
            i
            for i, step in enumerate(steps)
            if step.get("name") == "Build application with PyInstaller"
        )
        assert identity < source < freeze
        packet = next(step for step in steps if step["name"] == "Retain native scientific inputs")
        assert packet["if"] == "always()"
        assert packet["with"]["name"].startswith("release-science-inputs-")
        assert packet["with"]["retention-days"] == 1
    report = next(
        step for step in job["steps"] if step["name"] == "Retain native scientific report"
    )
    assert report["with"]["retention-days"] == 1
    assert report["with"]["name"].startswith("release-science-results-")
