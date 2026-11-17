"""Immutable CI event-base selection and actual workflow integration."""

import json
from pathlib import Path

import pytest
import yaml

from tests._harness import _load_harness

SHA = "a" * 40


@pytest.mark.parametrize(
    "name,event",
    [
        ("pull_request", {"pull_request": {"base": {"sha": SHA}}, "after": "b" * 40}),
        ("push", {"before": SHA, "after": "b" * 40}),
        ("workflow_dispatch", {"inputs": {"structural_base": SHA}}),
    ],
)
def test_actual_ci_mode_selects_immutable_event_base(tmp_path, monkeypatch, name, event):
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", name)
    args = _load_harness().build_parser().parse_args(["structural", "--ci"])
    rules = _load_harness()._maintenance
    assert rules.requested_base(args.base, args.ci, args.snapshot) == SHA


@pytest.mark.parametrize(
    "name,event",
    [
        ("push", {"before": "0" * 40}),
        ("workflow_dispatch", {}),
        ("pull_request", {}),
        ("other", {}),
        ("push", {"before": "HEAD"}),
    ],
)
def test_missing_or_invalid_ci_base_fails(name, event):
    with pytest.raises(ValueError):
        _load_harness()._maintenance.ci_base(name, event)


def test_local_environment_and_explicit_snapshot(monkeypatch):
    rules = _load_harness()._maintenance
    monkeypatch.setenv("ASYMMETRY_STRUCTURAL_BASE", SHA)
    assert rules.requested_base(None, False, False) == SHA
    assert rules.requested_base("explicit", False, False) == "explicit"
    assert rules.requested_base(None, False, True) is None
    with pytest.raises(ValueError):
        rules.requested_base(None, True, True)


def test_actual_workflow_supplies_history_and_requires_event_mode():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    checks = workflow["jobs"]["checks"]
    checkout = next(
        step for step in checks["steps"] if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0
    structural = next(step for step in checks["steps"] if step.get("name") == "Structural checks")
    assert structural["run"] == "python3 tools/harness.py structural --ci"
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["pull-requests"] == "read"
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["workflow_dispatch"]["inputs"]["structural_base"]["required"] is True
    assert not checks.get("if")


def test_workflow_base_fetch_uses_the_validated_selector(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["checks"]["steps"]
    fetch = next(step for step in steps if step.get("name") == "Fetch structural event base")
    assert fetch["run"].splitlines() == [
        "base=$(python3 tools/harness_maintenance.py)",
        'git fetch --no-tags origin "$base"',
    ]
    assert steps.index(fetch) < next(
        i for i, s in enumerate(steps) if s.get("name") == "Structural checks"
    )
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": SHA}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    result = subprocess.run(
        [sys.executable, str(root / "tools/harness_maintenance.py")],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0
    assert result.stdout == SHA + "\n"
    event.write_text("{}")
    result = subprocess.run(
        [sys.executable, str(root / "tools/harness_maintenance.py")],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode != 0
    assert not result.stdout


@pytest.mark.parametrize("ci,snapshot", [(True, False), (False, True)])
def test_empty_explicit_base_cannot_be_combined_with_another_mode(ci, snapshot):
    with pytest.raises(ValueError):
        _load_harness()._maintenance.requested_base("", ci, snapshot)
