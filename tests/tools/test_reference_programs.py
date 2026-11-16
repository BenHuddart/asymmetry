"""Optional references skip only before an executable has been selected."""

import hashlib
import json
import subprocess

import pytest

from tests import mantid_pulse, reference_programs


def test_unconfigured_musrfit_skips(monkeypatch):
    monkeypatch.delenv("ASYMMETRY_MUSRFIT", raising=False)
    monkeypatch.setattr(reference_programs.shutil, "which", lambda _: None)
    with pytest.raises(pytest.skip.Exception, match="not installed"):
        reference_programs.musrfit_executable()


@pytest.mark.parametrize("version", ["", "1.12.10 708f777b", "1.12.1 708f777b ROOT 6.38"])
def test_selected_musrfit_identity_mismatch_fails(monkeypatch, version):
    monkeypatch.setenv("ASYMMETRY_MUSRFIT", "owned-musrfit")
    monkeypatch.setattr(
        reference_programs.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, version, ""),
    )
    with pytest.raises(AssertionError, match="identity mismatch"):
        reference_programs.musrfit_executable()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("configured program missing"),
        subprocess.TimeoutExpired("owned", 15),
        subprocess.CalledProcessError(2, "owned"),
    ],
)
def test_selected_musrfit_command_failures_propagate(monkeypatch, error):
    monkeypatch.setenv("ASYMMETRY_MUSRFIT", "owned-musrfit")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(reference_programs.subprocess, "run", fail)
    with pytest.raises(type(error)):
        reference_programs.musrfit_executable()


@pytest.mark.parametrize(
    "defect", ["missing", "hash", "checkout", "import", "timeout", "empty", "parser"]
)
def test_configured_pulse_source_failures_are_not_absence(tmp_path, monkeypatch, defect):
    source = tmp_path / "source"
    source.mkdir()
    module = source / "owned.py"
    module.write_text("raise ImportError('owned broken reference')\n")
    identity = json.loads(mantid_pulse.PULSE_IDENTITY.read_text())
    identity["module"] = module.name
    identity["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    manifest = tmp_path / "identity.json"
    manifest.write_text(json.dumps(identity))
    monkeypatch.setattr(mantid_pulse, "PULSE_IDENTITY", manifest)
    if defect == "missing":
        module.unlink()
    if defect == "hash":
        module.write_text("different module")
    real_run = subprocess.run

    def run(command, **kwargs):
        if command[0] == "git":
            commit = "different checkout" if defect == "checkout" else identity["commit"]
            return subprocess.CompletedProcess(command, 0, commit, "")
        if defect == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if defect in {"empty", "parser"}:
            return subprocess.CompletedProcess(command, 0, "[]" if defect == "empty" else "bad", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(mantid_pulse.subprocess, "run", run)
    expected = {
        "missing": FileNotFoundError,
        "hash": AssertionError,
        "checkout": AssertionError,
        "import": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
        "empty": AssertionError,
        "parser": json.JSONDecodeError,
    }
    with pytest.raises(expected[defect]):
        mantid_pulse.run_start(source, 1)
