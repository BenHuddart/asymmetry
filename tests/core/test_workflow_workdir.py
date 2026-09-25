"""Tests for :mod:`asymmetry.core.workflow.workdir`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from asymmetry import __version__
from asymmetry.core.io import load
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.reduction import (
    ReductionSettings,
    reduce_run,
    resolve_reduction_grouping,
)
from asymmetry.core.workflow.survey import build_run_row, precession_evidence
from asymmetry.core.workflow.workdir import (
    SCHEMA,
    WORKDIR_NAME,
    ReducedEntry,
    RunSelection,
    WorkDir,
    WorkDirMismatchError,
    file_fingerprint,
    reduction_digest,
    safe_name,
)
from tests.core.conftest import SCAN_RUNS


@pytest.fixture
def reduced(workflow_folder: Path, tmp_path: Path):
    """A reduced run plus everything needed to store it in a fresh work dir."""
    run_number = SCAN_RUNS[0]
    path = workflow_folder / f"SIM{run_number:08d}.nxs"
    result = load(str(path))
    dataset_in = result[0] if isinstance(result, list) else result
    settings = ReductionSettings()
    grouping = resolve_reduction_grouping(dataset_in.run, settings)
    dataset = reduce_run(dataset_in.run, settings)
    digest = reduction_digest(source_file=path, grouping=grouping, settings=settings)
    entry = ReducedEntry(
        run_number=run_number,
        digest=digest,
        source_file=str(path),
        n_points=dataset.n_points,
        settings=settings,
        run=build_run_row(
            dataset_in,
            path=path,
            prefix="SIM",
            run_number=run_number,
            precession=precession_evidence(dataset, dataset_in.field),
        ).to_dict(),
        alpha=float(grouping["alpha"]),
        deadtime_mode=str(grouping["deadtime_mode"]),
        forward_group=int(grouping["forward_group"]),
        backward_group=int(grouping["backward_group"]),
    )
    return WorkDir(tmp_path / "wd"), dataset, entry, path, grouping, settings


def test_the_default_is_in_the_current_directory_not_the_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    assert WorkDir.default().root == project / WORKDIR_NAME
    assert WorkDir.default(project / "elsewhere").root == project / "elsewhere"


def test_the_default_work_directory_is_not_hidden() -> None:
    """An analyst has to find the plots, the recipes and the stale sessions."""
    assert not WORKDIR_NAME.startswith(".")


def test_a_work_directory_binds_to_the_folder_its_manifest_names(reduced, tmp_path: Path) -> None:
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    folder = tmp_path / "runs"
    folder.mkdir()

    assert workdir.selection is None
    workdir.write_manifest(RunSelection(folder, None), settings=settings, runs=[101])

    assert workdir.selection == RunSelection(folder.resolve(), None)
    assert workdir.bind(folder, None) == RunSelection(folder.resolve(), None)


def test_a_bound_work_directory_accepts_its_folder_named_any_way(
    reduced, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path, an absolute one and a detour through ``..`` are one folder."""
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    folder = tmp_path / "runs"
    folder.mkdir()
    workdir.write_manifest(RunSelection(folder, None), settings=settings, runs=[])

    monkeypatch.chdir(tmp_path)
    workdir.bind("runs", None)
    workdir.bind(folder, None)
    workdir.bind(tmp_path / "runs" / ".." / "runs", None)


def test_a_second_data_folder_cannot_share_a_bound_work_directory(reduced, tmp_path: Path) -> None:
    """Everything stored is keyed on run number alone, so two folders would collide."""
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workdir.write_manifest(RunSelection(first, None), settings=settings, runs=[101])

    with pytest.raises(WorkDirMismatchError) as exc:
        workdir.bind(second, None)

    assert str(first.resolve()) in str(exc.value)
    assert str(second.resolve()) in str(exc.value)
    assert f"--workdir {WORKDIR_NAME}-<name>" in str(exc.value)


def test_the_manifest_records_the_instrument_and_refuses_a_second_one(
    reduced, tmp_path: Path
) -> None:
    """Two instruments in one folder can share run numbers, so they cannot share a session."""
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    workdir.write_manifest(RunSelection(tmp_path, "EMU"), settings=settings, runs=[101])

    assert workdir.read_manifest()["instrument"] == "EMU"
    assert workdir.bind(tmp_path, "EMU") == RunSelection(tmp_path.resolve(), "EMU")
    with pytest.raises(WorkDirMismatchError) as exc:
        workdir.bind(tmp_path, "MUSR")

    assert "(EMU)" in str(exc.value)
    assert "(MUSR)" in str(exc.value)


def test_binding_takes_the_narrower_of_the_stored_and_the_named_instrument(
    reduced, tmp_path: Path
) -> None:
    """A command naming no instrument reads the session's; a whole-folder session takes one on."""
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    assert workdir.bind(tmp_path, None) == RunSelection(tmp_path.resolve(), None)

    workdir.write_manifest(RunSelection(tmp_path, None))
    assert workdir.bind(tmp_path, "EMU") == RunSelection(tmp_path.resolve(), "EMU")

    workdir.write_manifest(RunSelection(tmp_path, "EMU"))
    assert workdir.bind(tmp_path, None) == RunSelection(tmp_path.resolve(), "EMU")


def test_a_schema_3_manifest_holds_every_run_in_its_folder(reduced, tmp_path: Path) -> None:
    """Schema 3 predates instruments; its session was the whole folder."""
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    workdir.write_manifest(RunSelection(tmp_path, None))
    manifest = workdir.read_manifest()
    del manifest["instrument"]
    manifest["schema"] = 3
    workdir.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert workdir.selection == RunSelection(tmp_path.resolve(), None)
    assert workdir.bind(tmp_path, "EMU") == RunSelection(tmp_path.resolve(), "EMU")


def test_a_selection_matches_its_instrument_in_any_case(tmp_path: Path) -> None:
    emu = RunSelection(tmp_path, "EMU")
    assert emu.matches("EMU")
    assert emu.matches("emu")
    assert not emu.matches("MUSR")
    assert RunSelection(tmp_path, None).matches("MUSR")


def test_a_selection_naming_no_file_in_the_folder_lists_what_is_there(tmp_path: Path) -> None:
    for name in ("EMU00000001.nxs", "emu00000002.nxs", "MUSR00000001.nxs"):
        (tmp_path / name).touch()

    assert [path.name for _p, _r, path in RunSelection(tmp_path, "EMU").scan().entries] == [
        "EMU00000001.nxs",
        "emu00000002.nxs",
    ]
    with pytest.raises(
        ValueError, match=r"no HIFI run files; it holds EMU \(2 files\), MUSR \(1 file\)"
    ):
        RunSelection(tmp_path, "HIFI").scan()


def test_reduced_round_trips_through_the_work_directory(reduced) -> None:
    workdir, dataset, entry, _path, _grouping, _settings = reduced
    workdir.write_reduced(dataset, entry)

    restored = workdir.reduced(entry.run_number)
    assert np.array_equal(restored.time, dataset.time)
    assert np.array_equal(restored.asymmetry, dataset.asymmetry)
    assert np.array_equal(restored.error, dataset.error)
    assert restored.run_number == entry.run_number
    assert restored.temperature == dataset.temperature


def test_entry_round_trips_through_its_dict(reduced) -> None:
    workdir, dataset, entry, _path, _grouping, _settings = reduced
    workdir.write_reduced(dataset, entry)

    stored = workdir.entry(entry.run_number)
    assert stored == entry
    assert stored.settings == entry.settings


def test_reduced_raises_key_error_for_an_unreduced_run(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    with pytest.raises(KeyError):
        workdir.reduced(999999)
    with pytest.raises(KeyError):
        workdir.entry(999999)


def test_is_current_tracks_the_digest(reduced) -> None:
    workdir, dataset, entry, path, grouping, settings = reduced
    assert not workdir.is_current(entry.run_number, entry.digest)

    workdir.write_reduced(dataset, entry)
    assert workdir.is_current(entry.run_number, entry.digest)

    changed = ReductionSettings(alpha=1.3, alpha_source="user")
    changed_digest = reduction_digest(source_file=path, grouping=grouping, settings=changed)
    assert changed_digest != entry.digest
    assert not workdir.is_current(entry.run_number, changed_digest)


def test_digest_changes_when_the_grouping_changes(reduced) -> None:
    _workdir, _dataset, _entry, path, grouping, settings = reduced
    baseline = reduction_digest(source_file=path, grouping=grouping, settings=settings)
    altered = dict(grouping)
    altered["first_good_bin"] = int(altered["first_good_bin"]) + 1
    assert reduction_digest(source_file=path, grouping=altered, settings=settings) != baseline


def test_digest_changes_when_the_file_changes(reduced, tmp_path: Path) -> None:
    _workdir, _dataset, _entry, path, grouping, settings = reduced
    baseline = reduction_digest(source_file=path, grouping=grouping, settings=settings)

    copy = tmp_path / "copy.nxs"
    copy.write_bytes(path.read_bytes() + b"\0")
    assert reduction_digest(source_file=copy, grouping=grouping, settings=settings) != baseline


def test_file_fingerprint_reports_size_mtime_and_hash(reduced) -> None:
    _workdir, _dataset, _entry, path, _grouping, _settings = reduced
    fingerprint = file_fingerprint(path)
    assert fingerprint["size"] == path.stat().st_size
    assert fingerprint["mtime_ns"] == path.stat().st_mtime_ns
    assert len(fingerprint["sha256"]) == 64


def test_file_fingerprint_sees_a_byte_changed_near_the_end(reduced, tmp_path: Path) -> None:
    """The whole file is hashed, not a leading slice.

    An edit past the first mebibyte with the size and mtime restored is
    exactly the case a head-only hash would serve from cache.
    """
    _workdir, _dataset, _entry, source, _grouping, _settings = reduced
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    stat = path.stat()
    before = file_fingerprint(path)

    with path.open("r+b") as handle:
        handle.seek(-8, 2)
        tail = handle.read(1)
        handle.seek(-8, 2)
        handle.write(bytes([tail[0] ^ 0xFF]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    after = file_fingerprint(path)
    assert after["size"] == before["size"]
    assert after["mtime_ns"] == before["mtime_ns"]
    assert after["sha256"] != before["sha256"]


@pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", "", ".", "..", ".hidden"])
def test_safe_name_rejects_anything_that_is_not_one_path_component(name: str) -> None:
    with pytest.raises(ValueError):
        safe_name(name)


@pytest.mark.parametrize("name", ["scan", "series-wizard-102", "a.b", "T_300K"])
def test_safe_name_accepts_an_ordinary_name(name: str) -> None:
    assert safe_name(name) == name


@pytest.mark.parametrize("name", ["../escape", "a/b", ""])
def test_no_work_directory_path_can_be_built_from_an_unsafe_name(reduced, name: str) -> None:
    """Every path a caller's name reaches goes through :func:`safe_name`."""
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    recipe = FitRecipe.from_expression("Exponential + Constant")

    for call in (
        lambda: workdir.recipe_path(name),
        lambda: workdir.series_path(name),
        lambda: workdir.series_plot_dir(name),
        lambda: workdir.trend_plot_path(name, "lambda"),
        lambda: workdir.write_recipe(name, recipe),
        lambda: workdir.read_recipe(name),
        lambda: workdir.write_series(name, {"name": name}),
        lambda: workdir.read_series(name),
    ):
        with pytest.raises(ValueError):
            call()

    assert list(workdir.root.parent.glob("*.json")) == []


def test_manifest_records_version_folder_settings_and_runs(reduced, tmp_path: Path) -> None:
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    workdir.write_manifest(RunSelection(tmp_path, None), settings=settings, runs=[101, 102])

    manifest = workdir.read_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["asymmetry_version"] == __version__
    assert manifest["instrument"] is None
    # Absolute and resolved: the folder is what binds the directory, and it is
    # compared against paths typed in later commands from other directories.
    assert manifest["folder"] == str(tmp_path.resolve())
    assert manifest["settings"] == settings.to_dict()
    assert manifest["runs"] == [101, 102]
    assert manifest["updated"]


def test_a_manifest_written_without_settings_keeps_what_the_reduction_recorded(
    reduced, tmp_path: Path
) -> None:
    """``survey`` claims a directory; it must not erase ``reduce``'s provenance."""
    workdir, _dataset, _entry, _path, _grouping, settings = reduced
    workdir.write_manifest(RunSelection(tmp_path, None), settings=settings, runs=[101, 102])

    workdir.write_manifest(RunSelection(tmp_path, None))

    manifest = workdir.read_manifest()
    assert manifest["settings"] == settings.to_dict()
    assert manifest["runs"] == [101, 102]


def test_survey_payload_round_trips(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    workdir.write_survey({"folder": "somewhere", "runs": []})

    stored = workdir.read_survey()
    assert stored["schema"] == SCHEMA
    assert stored["asymmetry_version"] == __version__
    assert stored["folder"] == "somewhere"


def test_reduced_runs_lists_what_is_stored(reduced) -> None:
    workdir, dataset, entry, _path, _grouping, _settings = reduced
    assert workdir.reduced_runs() == []
    workdir.write_reduced(dataset, entry)
    assert workdir.reduced_runs() == [entry.run_number]


# -- screening, recipes and series ------------------------------------------


def test_a_wizard_payload_is_stamped_and_read_back(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    assert workdir.screened_runs() == []

    workdir.write_wizard(102, {"recommended_key": "exp_constant"})

    stored = workdir.read_wizard(102)
    assert stored["schema"] == SCHEMA
    assert stored["asymmetry_version"] == __version__
    assert stored["recommended_key"] == "exp_constant"
    assert workdir.screened_runs() == [102]
    with pytest.raises(KeyError, match="has not been screened"):
        workdir.read_wizard(999)


def test_a_recipe_round_trips_through_the_work_directory(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    recipe = FitRecipe.from_expression("Exponential + Constant")

    path = workdir.write_recipe("relax", recipe)

    assert path == workdir.recipes_dir / "relax.json"
    assert workdir.read_recipe("relax") == recipe
    assert workdir.recipe_names() == ["relax"]
    with pytest.raises(KeyError, match="No recipe"):
        workdir.read_recipe("missing")


def test_a_series_payload_is_stamped_and_read_back(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    assert workdir.series_names() == []

    workdir.write_series("scan", {"name": "scan", "results": []})

    stored = workdir.read_series("scan")
    assert stored["schema"] == SCHEMA
    assert stored["name"] == "scan"
    assert workdir.series_names() == ["scan"]
    with pytest.raises(KeyError, match="No series"):
        workdir.read_series("missing")


def _restamp(path: Path, schema: int) -> None:
    """Rewrite a stored JSON document as if an older asymmetry had written it."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema"] = schema
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_reduced_run_from_an_older_schema_is_refused_until_reduced_again(reduced) -> None:
    workdir, dataset, entry, _path, _grouping, _settings = reduced
    workdir.write_reduced(dataset, entry)
    _restamp(workdir.reduced_dir / f"{entry.run_number}.json", SCHEMA - 1)

    with pytest.raises(KeyError, match="reduced .* by an older asymmetry"):
        workdir.reduced(entry.run_number)


def test_the_digest_folds_in_the_schema(reduced, monkeypatch) -> None:
    from asymmetry.core.workflow import workdir as workdir_module

    _workdir, _dataset, entry, path, grouping, settings = reduced
    monkeypatch.setattr(workdir_module, "SCHEMA", SCHEMA - 1)
    older = reduction_digest(source_file=path, grouping=grouping, settings=settings)
    assert older != entry.digest


def test_a_series_from_an_older_schema_is_refused(reduced) -> None:
    workdir, _dataset, _entry, _path, _grouping, _settings = reduced
    workdir.write_series("scan", {"name": "scan", "results": []})
    _restamp(workdir.series_path("scan"), SCHEMA - 1)

    with pytest.raises(KeyError, match="Series 'scan' was written by an older asymmetry"):
        workdir.read_series("scan")


def test_a_reduced_run_records_the_logged_sample_temperature_beside_the_setpoint(
    reduced,
) -> None:
    _workdir, _dataset, entry, _path, _grouping, _settings = reduced
    # The simulated files log no sample temperature; the field is present and empty.
    assert entry.run["sample_temperature_logged"] is None
    assert entry.run["temperature"] == 10.0
