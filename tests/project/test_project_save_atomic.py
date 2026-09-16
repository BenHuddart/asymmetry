"""Tests for :func:`save_project`'s crash-safe atomic write (D9).

Pure-core: no GUI. ``save_project`` writes through a temp file in the
destination directory and ``os.replace``s it into place, keeping the file it
overwrote as ``<path>.bak`` (one generation). A failure during serialisation
must never touch the target file and must leave no temp file behind.
"""

from __future__ import annotations

import json

import pytest

from asymmetry.core.project import schema
from asymmetry.core.project.schema import load_project, save_project


def _state(marker: str) -> dict:
    return {"schema_version": schema.CURRENT_SCHEMA_VERSION, "datasets": [], "marker": marker}


def test_first_save_creates_no_backup(tmp_path):
    path = tmp_path / "project.asymp"

    save_project(_state("first"), path)

    assert path.exists()
    assert not path.with_name(path.name + ".bak").exists()
    assert not any(tmp_path.glob("*.tmp"))
    loaded = load_project(path)
    assert loaded["marker"] == "first"


def test_backup_false_overwrites_without_keeping_a_generation(tmp_path):
    """An autosave snapshot is itself a backup, so it keeps no ``.bak`` of its own."""
    target = tmp_path / "proj.autosave.asymp"
    save_project(_state("one"), target, backup=False)
    save_project(_state("two"), target, backup=False)
    assert json.loads(target.read_text())["marker"] == "two"
    assert not (tmp_path / "proj.autosave.asymp.bak").exists()
    assert list(tmp_path.iterdir()) == [target]


def test_second_save_backs_up_previous_contents(tmp_path):
    path = tmp_path / "project.asymp"

    save_project(_state("first"), path)
    previous_bytes = path.read_bytes()
    save_project(_state("second"), path)

    backup = path.with_name(path.name + ".bak")
    assert backup.exists()
    assert backup.read_bytes() == previous_bytes

    loaded = load_project(path)
    assert loaded["marker"] == "second"
    assert not any(tmp_path.glob("*.tmp"))


def test_third_save_replaces_backup_with_only_the_most_recent_generation(tmp_path):
    path = tmp_path / "project.asymp"

    save_project(_state("first"), path)
    save_project(_state("second"), path)
    second_bytes = path.read_bytes()
    save_project(_state("third"), path)

    backup = path.with_name(path.name + ".bak")
    assert backup.read_bytes() == second_bytes
    assert load_project(path)["marker"] == "third"


def test_failed_serialization_leaves_original_file_and_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "project.asymp"
    save_project(_state("first"), path)
    original_bytes = path.read_bytes()

    def _boom(*args, **kwargs):
        raise ValueError("synthetic serialisation failure")

    monkeypatch.setattr(schema.json, "dumps", _boom)

    with pytest.raises(ValueError, match="synthetic serialisation failure"):
        save_project(_state("second"), path)

    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + ".bak").exists()
    assert not any(tmp_path.glob("*.tmp*"))


def test_failed_write_after_serialization_leaves_original_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "project.asymp"
    save_project(_state("first"), path)
    original_bytes = path.read_bytes()

    def _boom(*args, **kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(schema.os, "fsync", _boom)

    with pytest.raises(OSError, match="synthetic disk failure"):
        save_project(_state("second"), path)

    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + ".bak").exists()
    assert not any(tmp_path.glob("*.tmp*"))


def test_saved_bytes_are_indent_2_json_matching_prior_format(tmp_path):
    path = tmp_path / "project.asymp"
    state = _state("format")

    save_project(state, path)

    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(state, indent=2)
