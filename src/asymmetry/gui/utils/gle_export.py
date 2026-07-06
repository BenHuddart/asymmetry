"""Shared GLE figure-export orchestration.

Every GLE export surface (main plot, Fit Parameters, Global Parameter Fit
window) runs the same sequence: ask for a ``.gleplot`` folder → import
gleplot → resolve paths → build the figure and sidecars → ``fig.savefig`` →
compile with the GLE binary → report → open the result in the gleplot editor
(static preview fallback). :func:`run_gle_export` owns that sequence once;
surfaces supply only what genuinely differs — the save-dialog wording, the
output format, and a ``build`` callback that writes their figure + sidecars.

Threading: the save dialog and the ``build`` callback run on the GUI thread
(builders read live widget state; figure building is fast). Only the GLE
compile — an external process of unbounded duration — runs on a worker via
the surface's :class:`~asymmetry.gui.tasks.TaskRunner`, with the result
dialogs and editor launch marshalled back to the GUI thread by the runner's
relay. This closes the long-standing invariant violation where exports froze
the GUI for the length of a ``gle`` run.

Test mode: with ``PYTEST_CURRENT_TEST`` set, every user-facing dialog and the
post-export editor/preview step are suppressed (the export itself — files,
compile — still runs). Tests observe behavior by monkeypatching the
module-level seams (:func:`show_export_result_dialog`, :func:`post_export_view`,
:func:`show_warning`, :func:`show_info`).
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.export_paths import (
    default_export_path,
    remember_export_path,
    resolve_gle_export_paths,
)
from asymmetry.gui.gle_settings import get_gle_executable
from asymmetry.gui.styles.metrics import dialog_width, row_height
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.utils.export import compile_gle
from asymmetry.gui.utils.gle_editor import launch_gle_editor

_GLE_EXPORT_FILTER = "GLE export folders (*.gleplot)"
_log = logging.getLogger(__name__)


def _test_mode() -> bool:
    """True under pytest: suppress dialogs and the post-export editor/preview."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


# ---------------------------------------------------------------------------
# Filename helpers (previously triplicated across the export surfaces)
# ---------------------------------------------------------------------------
def safe_file_token(value: str, fallback: str = "dataset") -> str:
    """Sanitize a string for use in an exported filename."""
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    token = "_".join(part for part in token.split("_") if part)
    return token or fallback


def dedup_export_token(base: str, used: set[str] | None) -> str:
    """Return a token unique within *used*, suffixing ``_2``, ``_3`` on clash.

    Guards against two exported series whose labels sanitize to the same
    token silently overwriting each other's ``.dat``/``.fit`` files.
    """
    if used is None:
        return base
    token = base
    n = 2
    while token in used:
        token = f"{base}_{n}"
        n += 1
    used.add(token)
    return token


_KNOWN_COMPILED_SUFFIXES = ("pdf", "eps", "png", "jpg", "svg")
_STALE_SIDECAR_SUFFIXES = {".dat", ".fit"}


def prune_stale_sidecars(export_dir: Path, gle_path: Path, kept: list[Path]) -> list[str]:
    """Remove ``.dat``/``.fit`` files left over from a previous, larger export.

    Re-exporting to the same ``<name>.gleplot`` folder never removed sidecars
    from an earlier export that wrote more of them (e.g. 10 datasets, then a
    3-dataset re-export to the same name) — the ``.gle`` script stays correct
    since it only references the files it needs, but the folder silently
    accumulates orphaned ``.dat``/``.fit`` files that the gleplot editor then
    lists alongside the live ones.

    The referenced set — files this export still needs — is the union of:
    the ``data <file>`` names read back out of *gle_path*, every path in
    *kept* (the builder's own ``GleExportBuild.files``, covering sidecars a
    builder writes but does not reference by a ``data`` command), the script
    itself, and its compiled outputs across every known GLE output format
    (conservative: a re-export in a different format must not delete last
    time's compiled output of another format).

    Only plain files directly inside *export_dir* (no recursion) with suffix
    ``.dat`` or ``.fit`` and a name outside that referenced set are removed.
    As a safety rail against ever running on a folder this machinery does not
    own, nothing is touched unless *export_dir* itself is a ``.gleplot``
    folder. Per-file deletion is best-effort (``OSError`` is swallowed).

    Returns the names removed, for logging by the caller.
    """
    if export_dir.suffix != ".gleplot":
        return []

    referenced = set(extract_gle_data_dependencies(gle_path))
    referenced.update(p.name for p in kept)
    referenced.add(gle_path.name)
    referenced.update(gle_path.with_suffix(f".{fmt}").name for fmt in _KNOWN_COMPILED_SUFFIXES)

    removed: list[str] = []
    for entry in export_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix not in _STALE_SIDECAR_SUFFIXES:
            continue
        if entry.name in referenced:
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        removed.append(entry.name)

    if removed:
        _log.info(
            "Removed %d stale sidecar file(s) from %s: %s",
            len(removed),
            export_dir,
            ", ".join(removed),
        )

    return removed


def extract_gle_data_dependencies(gle_path: Path) -> list[str]:
    """Return data-file names referenced by ``data <file>`` commands."""
    try:
        text = gle_path.read_text(encoding="utf-8")
    except OSError:
        return []

    seen: set[str] = set()
    deps: list[str] = []
    pattern = r"^\s*data\s+(?:\"([^\"]+)\"|(\S+))"
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        token = (match.group(1) or match.group(2) or "").strip()
        name = Path(token).name
        if name and name not in seen:
            seen.add(name)
            deps.append(name)
    return deps


# ---------------------------------------------------------------------------
# Dialog seams (module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------
def show_warning(parent: QWidget, title: str, message: str) -> None:
    if _test_mode():
        return
    QMessageBox.warning(parent, title, message)


def show_info(parent: QWidget, title: str, message: str) -> None:
    if _test_mode():
        return
    QMessageBox.information(parent, title, message)


def show_export_result_dialog(parent: QWidget, title: str, summary: str, details: str) -> None:
    """Show export results with scrollable details and a fixed bottom button."""
    if _test_mode():
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(dialog_width(105), row_height() * 16)

    layout = QVBoxLayout(dialog)
    summary_label = QLabel(summary)
    summary_label.setWordWrap(True)
    layout.addWidget(summary_label)

    details_view = QTextEdit()
    details_view.setReadOnly(True)
    details_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    details_view.setPlainText(details)
    details_view.setMinimumHeight(row_height() * 6)
    details_view.setMaximumHeight(row_height() * 10)
    layout.addWidget(details_view)

    button_row = QHBoxLayout()
    button_row.addStretch()
    close_btn = QPushButton("OK")
    close_btn.clicked.connect(dialog.accept)
    button_row.addWidget(close_btn)
    layout.addLayout(button_row)

    dialog.exec()


def show_static_gle_preview(parent: QWidget, gle_path: Path) -> None:
    """Legacy read-only preview: compile a PNG in a temp copy and display it.

    Fallback for gleplot installs without the editor embedding API
    (< 1.6). Copies the whole export folder to a temp directory (the
    ``.gleplot`` folder is self-contained: script + sidecars) so the
    preview compile never dirties the user's export. Best-effort: any
    failure leaves the "Preview unavailable" label — the export itself
    has already succeeded.
    """
    if _test_mode():
        return
    if not gle_path.exists():
        return
    gle_executable = get_gle_executable()
    if gle_executable is None:
        return

    try:
        dialog = QDialog(parent)
        dialog.setWindowTitle("GLE Plot Preview")
        dialog.resize(dialog_width(118), row_height() * 22)
        layout = QVBoxLayout(dialog)

        image_label = QLabel("Preview unavailable")
        layout.addWidget(image_label)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_export = Path(tmpdir) / gle_path.parent.name
            shutil.copytree(gle_path.parent, tmp_export)
            tmp_gle = tmp_export / gle_path.name
            preview_png = tmp_gle.with_suffix(".png")

            compile_gle(gle_executable, tmp_gle, "png", cwd=tmp_export)

            pixmap = QPixmap(str(preview_png))
            if not pixmap.isNull():
                image_label.setPixmap(pixmap)
                image_label.setText("")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()
    except Exception:
        return


def post_export_view(parent: QWidget, gle_path: Path) -> None:
    """Open the exported figure: gleplot editor, else the static preview."""
    if _test_mode():
        return
    if not launch_gle_editor(gle_path):
        show_static_gle_preview(parent, gle_path)


# ---------------------------------------------------------------------------
# The shared export sequence
# ---------------------------------------------------------------------------
@dataclass
class GleExportBuild:
    """What a surface's ``build`` callback reports back.

    ``files`` lists everything written (script, data, fit sidecars) for the
    result dialog's details pane; ``summary_extra`` appends surface-specific
    lines to the summary (e.g. dataset counts).
    """

    files: list[Path] = field(default_factory=list)
    summary_extra: str = ""


def run_gle_export(
    parent: QWidget,
    *,
    tasks: TaskRunner,
    dialog_title: str,
    default_name: str,
    output_format: str,
    build: Callable[[Any, Path, Path], GleExportBuild | None],
) -> None:
    """Run the full GLE export sequence for one surface.

    Parameters
    ----------
    parent
        Widget owning the dialogs (and, transitively, the export's lifetime).
    tasks
        The surface's :class:`TaskRunner`; the GLE compile runs on it so the
        GUI thread never blocks on the external process.
    dialog_title, default_name
        Save-dialog title and suggested ``<name>.gleplot`` filename.
    output_format
        GLE device for the compiled output (``"pdf"``, ``"eps"``, ``"png"``).
    build
        ``build(glp, gle_path, export_dir) -> GleExportBuild | None`` —
        runs on the GUI thread; writes sidecars, builds the gleplot figure,
        and calls ``fig.savefig(str(gle_path))``. Return ``None`` to abort
        silently (after showing its own message, e.g. "no data to export").
        A ``TypeError`` escaping the builder is treated as an outdated
        gleplot API and reported as such.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, dialog_title, default_export_path(default_name), _GLE_EXPORT_FILTER
    )
    if not path:
        return
    remember_export_path(path)

    try:
        glp = importlib.import_module("gleplot")
    except Exception:
        show_warning(parent, "gleplot not available", "Install gleplot to export GLE plots.")
        return

    gle_path, export_dir = resolve_gle_export_paths(Path(path), folder=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = build(glp, gle_path, export_dir)
    except TypeError as exc:
        # An old gleplot API surface raises TypeError from savefig/plot calls.
        # The detail is included so a genuine builder bug that lands here is
        # visible rather than silently masked as a version problem.
        show_warning(
            parent,
            "gleplot update required",
            "The installed gleplot version does not support this export. "
            f"Update the gleplot package and try again.\n\nDetail: {exc}",
        )
        return
    except Exception as exc:  # noqa: BLE001 - builder bugs must not vanish
        # A failed build must reach the user as a dialog (the pre-refactor
        # surfaces all had an "Export failed" catch-all), and the traceback
        # must still reach the log for diagnosis.
        _log.exception("GLE export build failed")
        show_warning(parent, "Export failed", str(exc))
        return
    if result is None:
        return

    prune_stale_sidecars(export_dir, gle_path, result.files)

    def _finish(compiled: bool, error: str | None = None) -> None:
        output_path = gle_path.with_suffix(f".{output_format}")
        if error is not None:
            show_warning(parent, "GLE compilation failed", error)
        elif compiled:
            files_text = "\n".join(str(p) for p in result.files)
            summary = f"GLE plot exported.\n\nGLE script: {gle_path}\nOutput: {output_path}"
            if result.summary_extra:
                summary += f"\n{result.summary_extra}"
            show_export_result_dialog(parent, "Export Successful", summary, files_text)
        else:
            show_info(
                parent,
                "GLE Not Installed",
                f"GLE script saved to {gle_path}.\n"
                f"Install GLE to compile to {output_format.upper()}.",
            )
        post_export_view(parent, gle_path)

    gle_executable = get_gle_executable()
    if gle_executable is None:
        _finish(compiled=False)
        return

    def _compile_task(_worker: object) -> None:
        try:
            compile_gle(gle_executable, gle_path, output_format, cwd=export_dir)
        except subprocess.CalledProcessError as exc:
            # str(CalledProcessError) drops stderr — surface GLE's actual
            # diagnostics to the error callback instead.
            raise RuntimeError(exc.stderr or str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"GLE did not finish within {exc.timeout:.0f}s and was stopped. "
                f"The script is saved at {gle_path}; compile it manually with "
                f"`gle -d {output_format} {gle_path.name}`."
            ) from exc
        except OSError as exc:
            # E.g. the configured path exists but lost its execute bit, or
            # points at a non-binary. Name the path so Setup ▸ GLE Setup… is
            # the obvious next step.
            raise RuntimeError(
                f"Could not run the GLE executable ({gle_executable}): {exc}\n"
                "Check Setup ▸ GLE Setup…"
            ) from exc

    tasks.start(
        _compile_task,
        on_finished=lambda _result: _finish(compiled=True),
        on_error=lambda message: _finish(compiled=True, error=message),
    )
