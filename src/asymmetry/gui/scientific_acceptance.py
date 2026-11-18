"""Bounded desktop diagnostic orchestration over the existing scientific owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES = ("counts.root", "counts.nxs")
TOLERANCES = {
    "array_atol": 1e-12,
    "array_rtol": 1e-12,
    "fit_atol_percent": 1e-5,
    "fit_error_rtol": 1e-4,
    "export_rtol": 1e-9,
}


def establish_state(root: Path) -> dict:
    """Redirect actual settings/app-data owners before any normal startup consumer."""
    for name in ("config", "cache", "app-data", "plugins", "matplotlib", "inputs"):
        (root / name).mkdir()
    for key, name in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_DATA_HOME", "app-data"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        os.environ[key] = str(root / name)
    sys.dont_write_bytecode = True
    from PySide6.QtCore import QCoreApplication, QSettings

    from asymmetry.core import plugins
    from asymmetry.gui.utils import app_data

    QCoreApplication.setOrganizationName("Asymmetry")
    QCoreApplication.setApplicationName("Asymmetry")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(root / "config"))
    app_data.app_data_dir = lambda: root / "app-data"
    plugins.USER_FUNCTIONS_DIR = root / "plugins"
    settings = QSettings()
    if not Path(settings.fileName()).is_relative_to(root):
        raise ValueError("diagnostic settings escaped owned root")
    settings.setValue("acceptance/owned", True)
    settings.sync()
    if settings.status() != QSettings.Status.NoError:
        raise OSError("diagnostic settings could not be written")
    return {
        "settings": str(Path(settings.fileName()).relative_to(root)),
        "app_data": "app-data",
        "plugins": "plugins",
        "discovery": "bypassed",
    }


def runtime_identity() -> dict:
    """Read the existing packaged identity; source execution is explicitly distinct."""
    from asymmetry import __version__

    if getattr(sys, "frozen", False):
        value = json.loads((Path(sys._MEIPASS) / "build-identity.json").read_text(encoding="utf-8"))
        if value["version"] != __version__:
            raise ValueError("packaged version differs from embedded identity")
        return value
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return {"version": __version__, "source_commit": commit, "target": "source"}


def frozen_payload_root() -> Path:
    """Bound resolved imports to the delivered app, including macOS resource links."""
    root = Path(sys.executable).resolve().parent
    if root.name == "MacOS" and root.parent.name == "Contents":
        root = root.parent.parent
    if not Path(sys._MEIPASS).resolve().is_relative_to(root):
        raise ValueError("frozen loader directory escaped delivered application")
    return root


def check_array(actual, expected, *, export=False):
    """Compare declared observations without accepting broadcasting or nonfinite values."""
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("acceptance observations have wrong shape or nonfinite values")
    if not np.allclose(
        actual,
        expected,
        atol=TOLERANCES["array_atol"],
        rtol=TOLERANCES["export_rtol" if export else "array_rtol"],
    ):
        raise ValueError("acceptance observations differ from analytic anchor")


def scientific_journey(path: Path, root: Path, stage) -> dict:
    """Load, reduce, fit, export and replay a current project through owning APIs."""
    import numpy as np

    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.result_summary import fit_result_summary
    from asymmetry.core.io import load
    from asymmetry.core.project import (
        CURRENT_SCHEMA_VERSION,
        apply_source_decisions,
        load_prepared_project,
        save_project,
        verify_project_sources,
    )
    from asymmetry.core.representation import FitSlot, RepresentationType
    from asymmetry.core.representation.project_model import ProjectModel
    from asymmetry.core.workflow.reduction import (
        ReductionSettings,
        reduce_run,
        resolve_reduction_grouping,
    )
    from asymmetry.gui.utils.plot_export_data import PlotExportData

    stage("load")
    loaded = load(str(path))
    run = loaded.run
    if run is None or len(run.histograms) != 2:
        raise ValueError("fixture requires two detector histograms")
    for histogram, scale in zip(run.histograms, (120, 80), strict=True):
        check_array(histogram.counts, scale * np.arange(1, 21))
        if (
            histogram.bin_width,
            histogram.t0_bin,
            histogram.t0_time_us,
            histogram.good_bin_start,
            histogram.good_bin_end,
        ) != (0.125, 2, 0.3125, 3, 18):
            raise ValueError("fixture timing or selected raw bounds differ")
    if run.grouping["groups"] != {1: [1], 2: [2]}:
        raise ValueError("fixture detector membership differs")
    settings = ReductionSettings(
        alpha=1.0, alpha_source="user", deadtime="off", background="none", pair=("1", "2"), rebin=1
    )
    stage("reduce")
    reduced = reduce_run(run, settings)
    grouping = resolve_reduction_grouping(run, settings)
    # The independent-count ratio has A=.2 and variance 4FB/(F+B)^3.
    indices = np.arange(3, 19)
    forward, backward = 120 * (indices + 1), 80 * (indices + 1)
    time = (indices - 2) * 0.125
    asymmetry = np.full(16, 20.0)
    error = 100 * np.sqrt(4 * forward * backward / (forward + backward) ** 3)
    check_array(reduced.time, time)
    check_array(reduced.asymmetry, asymmetry)
    check_array(reduced.error, error)
    stage("fit")
    model = CompositeModel.from_expression("Constant")
    fit = FitEngine().fit(
        reduced,
        model.function,
        ParameterSet([Parameter("A_bg", value=10.0)]),
        migrad_kwargs={"ncall": 200},
    )
    value = fit.parameters["A_bg"].value
    uncertainty = fit.uncertainties["A_bg"]
    expected_uncertainty = float(1 / np.sqrt(np.sum(1 / error**2)))
    if (
        not fit.success
        or not np.isfinite([value, uncertainty]).all()
        or uncertainty <= 0
        or abs(value - 20) > TOLERANCES["fit_atol_percent"]
        or not np.isclose(uncertainty, expected_uncertainty, rtol=TOLERANCES["fit_error_rtol"])
    ):
        raise ValueError("constant fit differs from weighted-mean anchor")
    stage("export")
    export = root / f"{path.stem}-{path.suffix[1:]}.dat"
    PlotExportData()._write_data_file(
        export,
        {
            "data": {"t": reduced.time, "y": reduced.asymmetry, "err": reduced.error},
            "grouping": grouping,
            "run_number": run.run_number,
            "effective_recipe": reduced.effective_recipe.to_dict(),
        },
    )
    check_array(
        np.loadtxt(export, comments="!"), np.column_stack((time, asymmetry, error)), export=True
    )
    if "!  Ylabel: % Asymmetry" not in export.read_text(encoding="utf-8"):
        raise ValueError("export lost percent units")
    stage("save")
    project_model = ProjectModel()
    rep_type = RepresentationType.TIME_FB_ASYMMETRY
    representation = project_model.ensure_dataset(run.run_number).ensure(rep_type)
    representation.recipe = {"grouping_ref": grouping}
    representation.cache_datasets([reduced])
    representation.fit = FitSlot(
        model=model.to_dict(),
        parameters=[{"name": "A_bg", "value": value, "fixed": False}],
        result=fit_result_summary(fit),
        provenance="single",
        fit_id=f"constant-{run.run_number}",
    )
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {
                "run_number": run.run_number,
                "source_file": str(path),
                "source_binding": run.source_binding.to_dict(),
                "grouping_overrides": grouping,
            }
        ],
    }
    project_model.write_to_project_state(state)
    project = export.with_suffix(".asymp")
    save_project(state, project)
    stage("reload")
    prepared = load_prepared_project(project)
    verified = verify_project_sources(prepared, project)
    if not verified.facts or any(fact.status != "matched" for fact in verified.facts):
        raise ValueError("saved project source content did not verify")
    application = apply_source_decisions(verified)
    reloaded = load(application.dataset_paths()[run.run_number])
    sample = application.input_for(("datasets", 0))
    application.check_capture(sample, reloaded.run.source_binding.observed)
    restored = prepared.model.representation(run.run_number, rep_type)
    if prepared.groupings[run.run_number].payload != json.loads(
        json.dumps(grouping)
    ) or restored.recipe != json.loads(json.dumps(representation.recipe)):
        raise ValueError("project lost grouping or reduction recipe")
    if (
        restored.fit.parameters != representation.fit.parameters
        or restored.fit.result != representation.fit.result
    ):
        raise ValueError("project lost fit parameters or result")
    replay = restored.compute(reloaded.run)[0]
    check_array(replay.time, time)
    check_array(replay.asymmetry * 100, asymmetry)
    check_array(replay.error * 100, error)
    restored_model = CompositeModel.from_dict(restored.fit.model)
    check_array(
        restored_model.function(time, **{p["name"]: p["value"] for p in restored.fit.parameters}),
        np.full(16, value),
    )
    return {
        "effective_recipe": reduced.effective_recipe.publishable(),
        "selected_forward_counts": forward.tolist(),
        "selected_backward_counts": backward.tolist(),
        "time_us": reduced.time.tolist(),
        "asymmetry_percent": reduced.asymmetry.tolist(),
        "error_percent": reduced.error.tolist(),
        "fit_percent": value,
        "fit_error_percent": uncertainty,
        "expected_fit_error_percent": expected_uncertainty,
        "export": export.name,
        "project": project.name,
        "replay_fraction": replay.asymmetry.tolist(),
    }


def main(arguments: list[str]) -> int:
    """Run an explicitly requested diagnostic; every required stage fails closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-inputs", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--acceptance-identity", required=True, type=json.loads)
    parser.add_argument("--acceptance-token", required=True)
    args = parser.parse_args(arguments)
    root = args.acceptance_root.resolve()
    root.mkdir(parents=False, exist_ok=False)
    report = {
        "token": args.acceptance_token,
        "success": False,
        "stages": [],
        "stage_outcomes": {},
        "tolerances": TOLERANCES,
        "results": {},
    }

    report["declared_settings"] = {
        "groups": {"1": [1], "2": [2]},
        "pair": [1, 2],
        "raw_bounds": [3, 18],
        "bin_width_us": 0.125,
        "t0_time_us": 0.3125,
        "alpha": 1.0,
        "background": "none",
        "deadtime": "off",
        "rebin": 1,
    }

    def stage(name):
        if report["stages"]:
            report["stage_outcomes"][report["stages"][-1]] = "passed"
        report["stage_outcomes"][name] = "passed" if name == "complete" else "running"
        report["stages"].append(name)
        (root / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    try:
        stage("isolation")
        report["state"] = establish_state(root)
        stage("identity")
        report["identity"] = runtime_identity()
        if report["identity"] != args.acceptance_identity:
            raise ValueError("runtime identity differs from expected identity")
        stage("resource")
        from PySide6.QtGui import QImage

        import asymmetry

        resource = Path(asymmetry.__file__).parent / "resources/logo_256x256.png"
        if QImage(str(resource)).isNull():
            raise ValueError("packaged logo cannot be decoded")
        report["resource_sha256"] = hashlib.sha256(resource.read_bytes()).hexdigest()
        stage("fixtures")
        manifest = json.loads(
            (args.acceptance_inputs / "fixtures.json").read_text(encoding="utf-8")
        )
        if set(manifest) != set(FIXTURES):
            raise ValueError("required fixture set differs")
        report["fixtures"] = {}
        for name in FIXTURES:
            destination = root / "inputs" / name
            shutil.copyfile(args.acceptance_inputs / name, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != manifest[name]:
                raise ValueError("fixture bytes differ from declared hash")
            report["fixtures"][name] = digest
            report["results"][name] = scientific_journey(
                destination, root, lambda name, fixture=name: stage(f"{fixture}:{name}")
            )
        check_array(
            report["results"][FIXTURES[0]]["asymmetry_percent"],
            report["results"][FIXTURES[1]]["asymmetry_percent"],
        )
        stage("origins")
        names = (
            "asymmetry",
            "asymmetry.core.io.root",
            "asymmetry.core.io.hdf4",
            "numpy",
            "scipy",
            "uproot",
            "awkward",
            "awkward_cpp",
            "pyhdf.SD",
            "pyhdf.hdfext",
            "pyhdf._hdfext",
            "iminuit",
            "iminuit._core",
        )
        origins = {}
        payload = frozen_payload_root() if getattr(sys, "frozen", False) else None
        for name in names:
            origin = Path(sys.modules[name].__file__).resolve()
            if payload is not None and not origin.is_relative_to(payload.resolve()):
                raise ValueError("scientific import escaped delivered payload")
            origins[name] = str(origin.relative_to(payload.resolve())) if payload else name
        report["origins"] = origins
        report["mode"] = "frozen" if payload else "source"
        report["success"] = True
        stage("complete")
    except Exception as exc:
        report["stage_outcomes"][report["stages"][-1]] = "failed"
        report["failure"] = {"stage": report["stages"][-1], "type": type(exc).__name__}
        (root / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return 1
    return 0
