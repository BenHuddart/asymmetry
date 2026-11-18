"""The real desktop entry must dispatch diagnostics before normal startup."""

import json
import os
import shutil
import subprocess
import sys
import time
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
    assert report["identity"]["version"] == identity()["version"]
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


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -float("inf"), 0])
def test_nonfinite_or_zero_budget_never_launches(tmp_path, monkeypatch, budget):
    import tools.desktop_acceptance as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budget reached child launch")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        run_acceptance(["never"], tmp_path, {}, tmp_path / "run", timeout=budget)


@pytest.mark.parametrize("stages", [None, 1, {}, [None], "complete"])
def test_malformed_matching_report_fails_with_retained_outcome(tmp_path, stages):
    child = tmp_path / "child.py"
    child.write_text(
        "import json,pathlib,sys\n"
        "a=sys.argv\n"
        "root=pathlib.Path(a[a.index('--acceptance-root')+1]); root.mkdir()\n"
        "token=a[a.index('--acceptance-token')+1]\n"
        f"(root/'result.json').write_text(json.dumps(dict(token=token,identity={{}},success=True,stages={stages!r})))\n"
    )
    with pytest.raises(RuntimeError):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run")
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert not outcome["success"]
    assert outcome["failure"] == "scientific stage failed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX owned process group")
def test_tree_cleanup_error_preserves_primary_timeout(tmp_path, monkeypatch):
    import tools.desktop_acceptance as runner

    child = tmp_path / "sleep.py"
    child.write_text("import time; time.sleep(30)")

    def denied(*args):
        raise PermissionError("synthetic group cleanup error")

    monkeypatch.setattr(runner.os, "killpg", denied)
    with pytest.raises(RuntimeError, match="timeout"):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run", timeout=0.1)
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert outcome["failure"] == "timeout"
    assert outcome["tree_cleanup_failure"] == "PermissionError"
    assert outcome["child_reaped"]


def test_launcher_diagnostic_is_bounded_and_paths_sanitized(tmp_path):
    child = tmp_path / "failure.py"
    child.write_text(
        "import sys\nprint('x'*5000)\nprint(__file__,file=sys.stderr)\nprint('native loader failed',file=sys.stderr)\nsys.exit(9)"
    )
    with pytest.raises(RuntimeError):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run")
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert len(outcome["launcher_output_tail"]) <= 4096
    assert "native loader failed" in outcome["launcher_output_tail"]
    assert str(tmp_path) not in json.dumps(outcome)


def test_nonfinite_report_is_rejected_at_report_boundary(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import json,pathlib,sys\na=sys.argv\n"
        "root=pathlib.Path(a[a.index('--acceptance-root')+1]); root.mkdir()\n"
        "token=a[a.index('--acceptance-token')+1]\n"
        "(root/'result.json').write_text(json.dumps(dict(token=token,identity={},success=True,stages=['complete'],value=float('nan'))))\n"
    )
    with pytest.raises(RuntimeError):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run")
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert outcome["failure"] == "unreadable report"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-state inspection")
def test_timeout_kills_the_owned_child_tree(tmp_path):
    child = tmp_path / "parent.py"
    pid_file = tmp_path / "grandchild-pid"
    child.write_text(
        "import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(RuntimeError, match="timeout"):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run", timeout=1)
    status = Path("/proc") / pid_file.read_text() / "status"
    # Group SIGKILL delivery can finish after the direct child's wait returns.
    deadline = time.monotonic() + 2
    while True:
        try:
            state = status.read_text()
        except FileNotFoundError:
            break
        if "State:\tZ" in state:
            break
        assert time.monotonic() < deadline, state
        time.sleep(0.01)
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert outcome["child_reaped"]
    assert outcome["tree_cleanup_exit"] == 0


def test_path_redaction_cannot_expand_diagnostic_budget(tmp_path):
    child = tmp_path / "failure.py"
    child.write_text("import sys\nprint('/a '*1500,file=sys.stderr)\nsys.exit(9)")
    with pytest.raises(RuntimeError):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run")
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert len(outcome["launcher_output_tail"]) <= 4096
    assert "/a" not in outcome["launcher_output_tail"]


def test_zero_launch_status_is_required_even_with_fresh_success_report(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import json,pathlib,sys\na=sys.argv\n"
        "root=pathlib.Path(a[a.index('--acceptance-root')+1]); root.mkdir()\n"
        "token=a[a.index('--acceptance-token')+1]\n"
        "(root/'result.json').write_text(json.dumps(dict(token=token,identity={},success=True,stages=['complete'])))\n"
        "sys.exit(7)\n"
    )
    with pytest.raises(RuntimeError, match="process failed"):
        run_acceptance([sys.executable, str(child)], tmp_path, {}, tmp_path / "run")
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert outcome["exit_code"] == 7
    assert outcome["report"]["success"]
    assert not outcome["success"]


def test_windows_environment_lookup_uses_case_insensitive_owner(monkeypatch):
    from types import SimpleNamespace

    import tools.desktop_acceptance as runner

    class WindowsEnvironment(dict):
        def __getitem__(self, key):
            return super().__getitem__(key.upper())

    monkeypatch.setattr(
        runner,
        "os",
        SimpleNamespace(
            name="nt",
            environ=WindowsEnvironment(SYSTEMROOT="C:/Windows", PATH="developer-rescue"),
            pathsep=";",
        ),
    )
    result = runner.payload_environment()
    assert "developer-rescue" not in result["PATH"]
    assert "System32" in result["PATH"]
    assert "Windows" in result["PATH"]


@pytest.mark.parametrize("escape", [False, True])
def test_entry_accepts_macos_bundle_resource_symlinks_and_rejects_escape(tmp_path, inputs, escape):
    app = tmp_path / "Asymmetry.app"
    frameworks = app / "Contents/Frameworks"
    resources = app / "Contents/Resources"
    executable = app / "Contents/MacOS/Asymmetry"
    executable.parent.mkdir(parents=True)
    executable.touch()
    frameworks.mkdir()
    resources.mkdir()
    (resources / "asymmetry/resources").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src/asymmetry/resources/logo_256x256.png",
        resources / "asymmetry/resources/logo_256x256.png",
    )
    # The delivered macOS inventory places pure-Python package data in Resources.
    (frameworks / "asymmetry").symlink_to("../Resources/asymmetry", target_is_directory=True)
    expected = identity()
    expected["target"] = "layout-regression"
    (frameworks / "build-identity.json").write_text(json.dumps(expected))
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "import importlib,sys\nfrom pathlib import Path\n"
        "from asymmetry.gui.scientific_acceptance import scientific_journey\n"
        "from asymmetry.gui.utils.plot_export_data import PlotExportData\n"
        "import PySide6.QtGui, PySide6.QtCore\n"
        "from asymmetry.core.project import load_prepared_project\n"
        "from asymmetry.core.representation import FitSlot\n"
        "from asymmetry.core.workflow.reduction import reduce_run\n"
        "import pyhdf.SD, pyhdf.hdfext, iminuit._core, awkward_cpp, uproot, scipy\n"
        f"root=Path({str(frameworks)!r})\n"
        "names=('asymmetry','asymmetry.core.io.root','asymmetry.core.io.hdf4','numpy','scipy','uproot','awkward','awkward_cpp','pyhdf.SD','pyhdf.hdfext','pyhdf._hdfext','iminuit','iminuit._core')\n"
        "for name in names:\n"
        "    module=importlib.import_module(name)\n"
        "    module.__file__=str(root/Path(*name.split('.'))/'__init__.py') if name=='asymmetry' else str(root/Path(*name.split('.')).with_suffix('.py'))\n"
        + (
            f"import numpy; numpy.__file__={str(tmp_path / 'developer/numpy.py')!r}\n"
            if escape
            else ""
        )
        + f"sys.executable={str(executable)!r}\n"
        "sys.frozen=True\nsys._MEIPASS=str(root)\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    if escape:
        with pytest.raises(RuntimeError):
            run_acceptance(
                [sys.executable, "-B", str(bootstrap)],
                inputs,
                expected,
                tmp_path / "run",
                environment=source_environment(),
            )
        report = json.loads((tmp_path / "run/outcome.json").read_text())["report"]
        assert report["failure"]["stage"] == "origins"
        return
    result = run_acceptance(
        [sys.executable, "-B", str(bootstrap)],
        inputs,
        expected,
        tmp_path / "run",
        environment=source_environment(),
    )
    assert result["report"]["origins"]["asymmetry"].startswith("Contents/Resources/")


@pytest.mark.parametrize("available", [True, False])
def test_entry_imports_lazy_required_dependencies_before_checking_origins(
    tmp_path, inputs, available
):
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "import sys\nfrom asymmetry.gui import scientific_acceptance as acceptance\n"
        "original=acceptance.scientific_journey\n"
        "def journey(*args):\n"
        "    result=original(*args)\n"
        + (
            "    sys.modules.pop('awkward_cpp',None)\n"
            if available
            else "    sys.modules['awkward_cpp']=None\n"
        )
        + "    return result\n"
        "acceptance.scientific_journey=journey\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    args = ([sys.executable, "-B", str(bootstrap)], inputs, identity(), tmp_path / "run")
    if available:
        outcome = run_acceptance(*args, environment=source_environment())
        assert outcome["report"]["origins"]["awkward_cpp"] == "awkward_cpp"
    else:
        with pytest.raises(RuntimeError):
            run_acceptance(*args, environment=source_environment())
        outcome = json.loads((tmp_path / "run/outcome.json").read_text())
        assert outcome["report"]["failure"] == {"stage": "origins", "type": "ModuleNotFoundError"}
    for name in ("counts.root", "counts.nxs"):
        assert outcome["report"]["results"][name]["fit_percent"] == pytest.approx(20, abs=1e-5)
        assert outcome["report"]["results"][name]["replay_fraction"] == pytest.approx([0.2] * 16)


def test_entry_retains_project_failure_message_with_owned_spaced_root_redacted(tmp_path, inputs):
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "import asymmetry.core.project as persistence\n"
        "def reject(project):\n"
        "    raise ValueError(f'deliberate project replay diagnostic at {project}')\n"
        "persistence.load_prepared_project=reject\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    work = tmp_path / "private root" / "run"
    with pytest.raises(RuntimeError):
        run_acceptance(
            [sys.executable, "-B", str(bootstrap)],
            inputs,
            identity(),
            work,
            environment=source_environment(),
        )
    outcome = json.loads((work / "outcome.json").read_text())
    assert outcome["report"]["failure"] == {"stage": "counts.root:reload", "type": "ValueError"}
    assert outcome["launcher_output_tail"].splitlines() == [
        "ValueError: deliberate project replay diagnostic at <path>"
    ]
    assert str(tmp_path) not in json.dumps(outcome)


def test_entry_retains_authoritative_source_verification_failure_facts(tmp_path, inputs):
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "from dataclasses import replace\n"
        "import asymmetry.core.project as persistence\n"
        "verify=persistence.verify_project_sources\n"
        "def unreadable(*args,**kwargs):\n"
        "    result=verify(*args,**kwargs)\n"
        "    return replace(result,facts=tuple(replace(f,status='unreadable',observed=None,diagnostic='deliberate verification fault') for f in result.facts))\n"
        "persistence.verify_project_sources=unreadable\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    with pytest.raises(RuntimeError):
        run_acceptance(
            [sys.executable, "-B", str(bootstrap)],
            inputs,
            identity(),
            tmp_path / "run",
            environment=source_environment(),
        )
    outcome = json.loads((tmp_path / "run/outcome.json").read_text())
    assert outcome["report"]["failure"] == {"stage": "counts.root:reload", "type": "ValueError"}
    tail = outcome["launcher_output_tail"]
    assert "unreadable" in tail and "deliberate verification fault" in tail
    assert "active=True" in tail and "observed=unknown" in tail
    assert "expected=" + outcome["report"]["fixtures"]["counts.root"] in tail
    assert str(tmp_path) not in json.dumps(outcome)


def test_entry_project_replay_uses_consistent_descriptor_timestamp_semantics(tmp_path, inputs):
    bootstrap = tmp_path / "entry.py"
    bootstrap.write_text(
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        "import asymmetry.core.utils.content as content\n"
        "class BirthTimePath(type(Path())):\n"
        "    def stat(self,*args,**kwargs):\n"
        "        value=super().stat(*args,**kwargs)\n"
        "        return SimpleNamespace(st_dev=value.st_dev,st_ino=value.st_ino,st_size=value.st_size,st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns-1000)\n"
        "content.Path=BirthTimePath\n"
        "from asymmetry.gui.app import main\nmain()\n"
    )
    outcome = run_acceptance(
        [sys.executable, "-B", str(bootstrap)],
        inputs,
        identity(),
        tmp_path / "run",
        environment=source_environment(),
    )
    assert outcome["report"]["success"]
    for name in ("counts.root", "counts.nxs"):
        result = outcome["report"]["results"][name]
        assert result["fit_percent"] == pytest.approx(20, abs=1e-5)
        assert result["replay_fraction"] == pytest.approx([0.2] * 16)
