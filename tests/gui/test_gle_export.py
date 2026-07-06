"""Tests for the shared GLE export orchestration (``gui/utils/gle_export.py``).

Pins the ``run_gle_export`` contract the three export surfaces rely on:
GUI-thread build, worker-thread compile, dialog/editor seams called with the
right arguments on each terminal path. Dialog seams are monkeypatched — their
internal pytest suppression is thereby bypassed, which is the intended way
for tests to observe the flow.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
# The builder callbacks in these tests drive the real gleplot API.
pytest.importorskip("gleplot", reason="gle extra not installed")
from PySide6.QtWidgets import QApplication, QFileDialog, QWidget  # noqa: E402

import asymmetry.gui.utils.gle_export as gle_export  # noqa: E402
from asymmetry.gui.tasks import TaskRunner  # noqa: E402
from asymmetry.gui.utils.gle_export import (  # noqa: E402
    GleExportBuild,
    dedup_export_token,
    extract_gle_data_dependencies,
    prune_stale_sidecars,
    run_gle_export,
    safe_file_token,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def harness(qapp, tmp_path, monkeypatch):
    """A parent widget + TaskRunner + recorded dialog seams + canned save path."""
    parent = QWidget()
    tasks = TaskRunner(parent)

    rec: dict[str, list] = {"result": [], "warning": [], "info": [], "view": []}
    monkeypatch.setattr(
        gle_export,
        "show_export_result_dialog",
        lambda p, title, summary, details: rec["result"].append((summary, details)),
    )
    monkeypatch.setattr(
        gle_export, "show_warning", lambda p, title, msg: rec["warning"].append((title, msg))
    )
    monkeypatch.setattr(
        gle_export, "show_info", lambda p, title, msg: rec["info"].append((title, msg))
    )
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: rec["view"].append(Path(gp)))
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "figure.gleplot"), "")),
    )

    yield parent, tasks, rec, tmp_path
    tasks.shutdown()
    parent.close()


def _wait(qapp, predicate, timeout_s=10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _build(glp, gle_path: Path, export_dir: Path) -> GleExportBuild:
    fig = glp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1])
    fig.savefig(str(gle_path))
    return GleExportBuild(files=[gle_path])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_safe_file_token():
    assert safe_file_token("T = 1.5 K (ZF)") == "T_1_5_K_ZF"
    assert safe_file_token("  ") == "dataset"
    assert safe_file_token("", fallback="group") == "group"


def test_dedup_export_token():
    used: set[str] = {"a"}
    assert dedup_export_token("a", used) == "a_2"
    assert dedup_export_token("a", used) == "a_3"
    assert dedup_export_token("b", None) == "b"


def test_extract_gle_data_dependencies(tmp_path):
    gle = tmp_path / "f.gle"
    gle.write_text('begin graph\n  data "a.dat" d1=c1,c2\n  data b.fit\nend graph\n')
    assert extract_gle_data_dependencies(gle) == ["a.dat", "b.fit"]


# ---------------------------------------------------------------------------
# prune_stale_sidecars
# ---------------------------------------------------------------------------
def test_prune_stale_sidecars_removes_unreferenced_dat_and_fit(tmp_path):
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text('begin graph\n  data "used.dat"\nend graph\n')
    (export_dir / "used.dat").write_text("1 2\n")
    (export_dir / "orphan.dat").write_text("3 4\n")
    (export_dir / "orphan.fit").write_text("fit\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert set(removed) == {"orphan.dat", "orphan.fit"}
    assert (export_dir / "used.dat").exists()
    assert not (export_dir / "orphan.dat").exists()
    assert not (export_dir / "orphan.fit").exists()


def test_prune_stale_sidecars_keeps_files_referenced_via_kept_list(tmp_path):
    """A sidecar not mentioned by a ``data`` command but reported through
    ``GleExportBuild.files`` (e.g. an extra fit sidecar) must survive."""
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    kept_sidecar = export_dir / "extra.fit"
    kept_sidecar.write_text("fit\n")
    (export_dir / "orphan.dat").write_text("3 4\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path, kept_sidecar])

    assert removed == ["orphan.dat"]
    assert kept_sidecar.exists()


def test_prune_stale_sidecars_never_touches_non_sidecar_suffixes(tmp_path):
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    (export_dir / "notes.txt").write_text("keep me\n")
    (export_dir / "figure.pdf").write_bytes(b"%PDF-fake")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    assert (export_dir / "notes.txt").exists()
    assert (export_dir / "figure.pdf").exists()


def test_prune_stale_sidecars_keeps_compiled_outputs_of_other_formats(tmp_path):
    """A re-export to a different output format must not delete last time's
    compiled output — compiled outputs of any known GLE format for the same
    stem are treated as legitimate, not stale."""
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    for fmt in ("pdf", "eps", "png", "jpg", "svg"):
        (export_dir / f"figure.{fmt}").write_bytes(b"data")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    for fmt in ("pdf", "eps", "png", "jpg", "svg"):
        assert (export_dir / f"figure.{fmt}").exists()


def test_prune_stale_sidecars_refuses_non_gleplot_directory(tmp_path):
    """Safety rail: only folders this machinery owns (``*.gleplot``) are
    ever cleaned — anything else is left untouched even if it happens to
    hold ``.dat``/``.fit`` files."""
    export_dir = tmp_path / "not_a_gleplot_folder"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    (export_dir / "orphan.dat").write_text("3 4\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    assert (export_dir / "orphan.dat").exists()


# ---------------------------------------------------------------------------
# run_gle_export terminal paths
# ---------------------------------------------------------------------------
def test_export_success_path(qapp, harness, monkeypatch):
    parent, tasks, rec, tmp_path = harness
    compiled = []
    monkeypatch.setattr(
        gle_export,
        "compile_gle",
        lambda exe, gle_file, fmt, *, cwd, **kw: compiled.append((Path(gle_file), fmt)),
    )
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_build,
    )

    assert _wait(qapp, lambda: rec["result"] or rec["warning"])
    assert not rec["warning"]
    gle_path = tmp_path / "figure.gleplot" / "figure.gle"
    assert gle_path.exists()
    assert compiled == [(gle_path, "pdf")]
    summary, details = rec["result"][0]
    assert str(gle_path) in summary and "figure.pdf" in summary
    assert str(gle_path) in details
    assert rec["view"] == [gle_path]


def test_export_compile_error_surfaces_stderr(qapp, harness, monkeypatch):
    parent, tasks, rec, tmp_path = harness

    def _boom(exe, gle_file, fmt, *, cwd, **kw):
        raise subprocess.CalledProcessError(1, ["gle"], stderr="fonts exploded")

    monkeypatch.setattr(gle_export, "compile_gle", _boom)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_build,
    )

    assert _wait(qapp, lambda: rec["warning"])
    title, msg = rec["warning"][0]
    assert title == "GLE compilation failed"
    assert "fonts exploded" in msg
    assert not rec["result"]
    # The exported script still opens for editing after a failed compile.
    assert rec["view"] == [tmp_path / "figure.gleplot" / "figure.gle"]


def test_export_with_unrunnable_gle_binary_names_the_path(qapp, harness, monkeypatch):
    """A configured path that exists but cannot execute (lost x-bit, not a
    binary) must surface as a clear warning naming the path — not a raw
    traceback out of the click handler (the pre-refactor behavior)."""
    parent, tasks, rec, tmp_path = harness

    def _denied(exe, gle_file, fmt, *, cwd, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gle_export, "compile_gle", _denied)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/opt/broken/gle")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_build,
    )

    assert _wait(qapp, lambda: rec["warning"])
    title, msg = rec["warning"][0]
    assert title == "GLE compilation failed"
    assert "/opt/broken/gle" in msg
    assert "GLE Setup" in msg
    # The export itself succeeded; the script still opens for editing.
    assert rec["view"] == [tmp_path / "figure.gleplot" / "figure.gle"]


def test_export_without_gle_binary(qapp, harness, monkeypatch):
    parent, tasks, rec, tmp_path = harness
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    monkeypatch.setattr(
        gle_export,
        "compile_gle",
        lambda *a, **k: pytest.fail("must not compile without a GLE binary"),
    )

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="eps",
        build=_build,
    )

    # Synchronous path: no task is started.
    assert rec["info"] and "GLE script saved" in rec["info"][0][1]
    assert rec["view"] == [tmp_path / "figure.gleplot" / "figure.gle"]
    assert not rec["result"] and not rec["warning"]


def test_build_returning_none_aborts_silently(qapp, harness, monkeypatch):
    parent, tasks, rec, _tmp = harness
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=lambda glp, gle_path, export_dir: None,
    )

    qapp.processEvents()
    assert not any(rec.values())


def test_build_typeerror_reports_gleplot_update(qapp, harness):
    parent, tasks, rec, _tmp = harness

    def _old_api(glp, gle_path, export_dir):
        raise TypeError("savefig() got an unexpected keyword argument")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_old_api,
    )

    assert rec["warning"] and rec["warning"][0][0] == "gleplot update required"
    assert not rec["view"]


def test_build_failure_shows_export_failed_dialog(qapp, harness):
    """A non-TypeError builder failure surfaces as an 'Export failed' warning
    (parity with the pre-refactor catch-alls), not an uncaught slot exception."""
    parent, tasks, rec, _tmp = harness

    def _broken(glp, gle_path, export_dir):
        raise RuntimeError("disk full while writing sidecars")

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_broken,
    )

    assert rec["warning"] and rec["warning"][0][0] == "Export failed"
    assert "disk full" in rec["warning"][0][1]
    assert not rec["view"]


def test_export_prunes_stale_sidecars_from_previous_larger_export(qapp, harness, monkeypatch):
    """Re-exporting to the same ``.gleplot`` folder must clean up sidecars
    left over from a previous, larger export (e.g. 10 datasets, then a
    3-dataset re-export to the same name) — a stale ``orphan.dat`` must be
    gone afterward, while the file the new export actually uses survives.
    """
    parent, tasks, rec, tmp_path = harness
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")
    monkeypatch.setattr(gle_export, "compile_gle", lambda exe, gle_file, fmt, *, cwd, **kw: None)

    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    orphan = export_dir / "orphan.dat"
    orphan.write_text("stale from a previous, larger export\n")

    def _build(glp, gle_path: Path, export_dir: Path) -> GleExportBuild:
        used = export_dir / "used.dat"
        used.write_text("1 2\n")
        gle_path.write_text('begin graph\n  data "used.dat"\nend graph\n')
        return GleExportBuild(files=[gle_path, used])

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=_build,
    )

    assert not orphan.exists()
    assert (export_dir / "used.dat").exists()


def test_cancelled_save_dialog_does_nothing(qapp, harness, monkeypatch):
    parent, tasks, rec, _tmp = harness
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    built = []

    run_gle_export(
        parent,
        tasks=tasks,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        build=lambda *a: built.append(True),
    )

    assert not built and not any(rec.values())
