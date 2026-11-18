"""The real desktop entry must dispatch diagnostics before normal startup."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.acceptance_fixtures import write_fixtures
from tools.desktop_acceptance import payload_environment, run_acceptance

ROOT = Path(__file__).resolve().parents[2]


def test_acceptance_requires_explicit_inputs_before_startup(tmp_path):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "src"), QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "asymmetry.gui.app", "--scientific-acceptance"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--acceptance-inputs" in result.stderr


@pytest.fixture
def inputs(tmp_path):
    pytest.importorskip("pyhdf.SD", exc_type=ImportError)
    pytest.importorskip("uproot")
    directory = tmp_path / "fixtures"
    write_fixtures(directory)
    return directory


def source_environment(source=ROOT / "src"):
    environment = payload_environment()
    environment["PYTHONPATH"] = str(source)
    return environment


def identity():
    from asymmetry.gui.scientific_acceptance import runtime_identity

    return runtime_identity()


def test_production_entry_nonzero_anchor_export_and_project(tmp_path, inputs):
    outcome = run_acceptance(
        [sys.executable, "-B", "-m", "asymmetry.gui.app"],
        inputs,
        identity(),
        tmp_path / "run",
        environment=source_environment(),
    )
    report = outcome["report"]
    assert report["mode"] == "source"
    assert report["identity"]["version"] == "0.28.0"
    assert report["fixtures"] == json.loads((inputs / "fixtures.json").read_text())
    assert str(tmp_path) not in json.dumps(outcome)
    for result in report["results"].values():
        assert result["asymmetry_percent"] == pytest.approx([20] * 16)
        assert result["replay_fraction"] == pytest.approx([0.2] * 16)
        assert result["time_us"] == pytest.approx([i / 8 for i in range(1, 17)])
        assert result["fit_percent"] == pytest.approx(20, abs=1e-5)
        # sigma_i^2 = 48/(i+1), sum(i+1) over raw bins3..18 =184.
        assert result["fit_error_percent"] == pytest.approx((48 / 184) ** 0.5, rel=1e-4)
        assert (tmp_path / "run/journey" / result["export"]).is_file()
        state = json.loads((tmp_path / "run/journey" / result["project"]).read_text())
        assert len(state["datasets"]) == 1
        assert state["datasets"][0]["source_binding"]["expected"]
    assert (
        report["results"]["counts.root"]["error_percent"]
        == report["results"]["counts.nxs"]["error_percent"]
    )


@pytest.mark.parametrize(
    "failure,stage",
    [
        ("identity", "identity"),
        ("missing", "fixtures"),
        ("hash", "fixtures"),
        ("reader", "counts.nxs:load"),
        ("reduce", "counts.root:reduce"),
        ("fit", "counts.root:fit"),
        ("export", "counts.root:export"),
        ("reload", "counts.root:reload"),
    ],
)
def test_production_entry_rejects_required_stage_failures(tmp_path, inputs, failure, stage):
    expected = identity()
    injection = ""
    if failure == "identity":
        expected["source_commit"] = "0" * 40
    elif failure == "missing":
        (inputs / "counts.root").unlink()
    elif failure == "hash":
        (inputs / "counts.root").write_bytes(b"broken")
    elif failure == "reader":
        injection = """
import importlib.abc
class OmitHDF4(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('pyhdf'):
            raise ImportError('omitted test reader')
sys.meta_path.insert(0, OmitHDF4())
"""
    elif failure == "reduce":
        injection = """
from asymmetry.core.workflow import reduction
original = reduction.reduce_run
def bad_reduction(*args, **kwargs):
    value = original(*args, **kwargs)
    value.asymmetry[0] += 1
    return value
reduction.reduce_run = bad_reduction
"""
    elif failure == "fit":
        injection = """
from dataclasses import replace
from asymmetry.core.fitting.engine import FitEngine
original = FitEngine.fit
def bad_fit(*args, **kwargs):
    return replace(original(*args, **kwargs), success=False)
FitEngine.fit = bad_fit
"""
    elif failure == "export":
        injection = """
from asymmetry.gui.utils.plot_export_data import PlotExportData
def broken_export(self, path, payload):
    path.write_text('')
PlotExportData._write_data_file = broken_export
"""
    elif failure == "reload":
        injection = """
from asymmetry.core import project
def broken_reload(*args, **kwargs):
    raise ValueError('project unavailable')
project.load_prepared_project = broken_reload
"""
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "import sys\n" + injection + "\nfrom asymmetry.gui.app import main\nmain()\n"
    )
    work = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run_acceptance(
            [sys.executable, "-B", str(bootstrap)],
            inputs,
            expected,
            work,
            environment=source_environment(),
        )
    outcome = json.loads((work / "outcome.json").read_text())
    assert not outcome["success"]
    assert outcome["exit_code"] != 0
    assert outcome["report"]["failure"]["stage"] == stage
    assert str(tmp_path) not in json.dumps(outcome)


def test_actual_missing_used_resource_fails_before_analysis(tmp_path, inputs):
    source = tmp_path / "source"
    shutil.copytree(
        ROOT / "src/asymmetry", source / "asymmetry", ignore=shutil.ignore_patterns("__pycache__")
    )
    (source / "asymmetry/resources/logo_256x256.png").unlink()
    # The copied source has no .git; source identity is deliberately fixed for this omission control.
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "from asymmetry.gui import scientific_acceptance\n"
        f"scientific_acceptance.runtime_identity = lambda: {identity()!r}\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    with pytest.raises(RuntimeError):
        run_acceptance(
            [sys.executable, "-B", str(bootstrap)],
            inputs,
            identity(),
            tmp_path / "run",
            environment=source_environment(source),
        )
    report = json.loads((tmp_path / "run/journey/result.json").read_text())
    assert report["failure"]["stage"] == "resource"


def test_early_dispatch_owns_settings_app_data_and_bypasses_discovery(tmp_path, inputs):
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text("""
import sys
from pathlib import Path
from asymmetry.gui import scientific_acceptance
original = scientific_acceptance.establish_state
def isolated(root):
    result = original(root)
    from PySide6.QtCore import QSettings
    from asymmetry.gui.utils import app_data, fit_times
    from asymmetry.core import plugins
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        assert Path(QSettings(QSettings.Format.IniFormat, scope, 'Asymmetry', 'Asymmetry').fileName()).is_relative_to(root)
    assert app_data.app_data_dir().is_relative_to(root)
    assert fit_times.fit_times_path().is_relative_to(root)
    assert plugins.USER_FUNCTIONS_DIR.is_relative_to(root)
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary plugin discovery was invoked')
    plugins.load_user_functions = forbidden
    return result
scientific_acceptance.establish_state = isolated
from asymmetry.gui.app import main
main()
""")
    result = run_acceptance(
        [sys.executable, "-B", str(bootstrap)],
        inputs,
        identity(),
        tmp_path / "run",
        environment=source_environment(),
    )
    assert result["success"]


@pytest.mark.parametrize("mode", ["exit", "missing", "stale", "timeout"])
def test_waiter_rejects_process_failure_timeout_and_stale_or_missing_report(tmp_path, mode):
    child = tmp_path / "child.py"
    child.write_text("""
import json, pathlib, sys, time
mode = sys.argv[1]
if mode == 'exit':
    sys.exit(7)
if mode == 'timeout':
    time.sleep(30)
if mode == 'stale':
    root = pathlib.Path(sys.argv[sys.argv.index('--acceptance-root') + 1])
    root.mkdir()
    (root / 'result.json').write_text(json.dumps({'success': True, 'token': 'old', 'identity': {}, 'stages': ['complete']}))
""")
    with pytest.raises(RuntimeError):
        run_acceptance(
            [sys.executable, "-B", str(child), mode],
            tmp_path,
            {},
            tmp_path / "run",
            timeout=0.2 if mode == "timeout" else 10,
        )
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert not outcome["success"]
    if mode == "timeout":
        assert outcome["failure"] == "timeout"
        assert outcome["tree_cleanup_exit"] == 0


def test_waiter_does_not_reuse_an_existing_invocation(tmp_path):
    work = tmp_path / "previous"
    work.mkdir()
    with pytest.raises(FileExistsError):
        run_acceptance([sys.executable], tmp_path, {}, work)


def test_missing_launcher_retains_failure(tmp_path):
    with pytest.raises(RuntimeError, match="launcher failed"):
        run_acceptance([str(tmp_path / "missing")], tmp_path, {}, tmp_path / "run")
    assert json.loads((tmp_path / "run/outcome.json").read_text())["failure"] == "launcher failed"


def test_payload_environment_removes_developer_overrides(monkeypatch):
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "ASYMMETRY_HDF4_DLL_DIR",
        "QT_PLUGIN_PATH",
        "VIRTUAL_ENV",
    ):
        monkeypatch.setenv(key, "developer-rescue")
    environment = payload_environment()
    assert "developer-rescue" not in environment.values()
    assert environment["PATH"] != os.environ["PATH"]
    assert environment.get("HOME") == os.environ.get("HOME")
