"""Tests for :mod:`asymmetry.core.workflow.workdir`."""

from __future__ import annotations

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
    WorkDir,
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


def test_for_folder_defaults_to_the_dot_asymmetry_directory(tmp_path: Path) -> None:
    assert WorkDir.for_folder(tmp_path).root == tmp_path / WORKDIR_NAME
    assert WorkDir.for_folder(tmp_path, tmp_path / "elsewhere").root == tmp_path / "elsewhere"


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
    workdir.write_manifest(folder=tmp_path, settings=settings, runs=[101, 102])

    manifest = workdir.read_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["asymmetry_version"] == __version__
    assert manifest["folder"] == str(tmp_path)
    assert manifest["settings"] == settings.to_dict()
    assert manifest["runs"] == [101, 102]
    assert manifest["updated"]


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
