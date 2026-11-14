"""Shared GLE export generation and vector preview orchestration."""

from __future__ import annotations

import functools
import importlib
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
from asymmetry.gui.utils.export import compile_gle
from asymmetry.gui.windows.gle_preview import GleExportResult, GlePlotPreview

_GLE_EXPORT_FILTER = "GLE export folders (*.gleplot)"
_log = logging.getLogger(__name__)


def _test_mode() -> bool:
    """True under pytest: suppress warning and information dialogs."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


@functools.lru_cache(maxsize=1)
def gle_offset_supported() -> bool:
    """Return True when the installed gleplot applies a per-series ``offset``.

    gleplot ≥ 1.7 accepts an ``offset=`` keyword on ``plot``/``errorbar``/
    ``fill_between`` and emits it as a GLE ``let`` so a waterfall/overlay stack
    is applied at plot time, leaving the exported ``.dat``/``.fit`` values raw
    on disk). Detected by signature so a dev or
    git install reports its true capability regardless of version string. When
    False (older gleplot), the export bakes the offset into the written data
    instead, matching the on-screen stack.
    """
    try:
        import inspect

        from gleplot.axes import Axes
    except Exception:
        return False
    try:
        return "offset" in inspect.signature(Axes.plot).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Shared export filenames
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
    accumulates orphaned ``.dat``/``.fit`` files alongside the live ones.

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


# ---------------------------------------------------------------------------
# The shared export sequence
# ---------------------------------------------------------------------------
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


@dataclass
class GleExportBuild:
    """What a surface's ``build`` callback reports back.

    ``files`` lists everything written (script, data, fit sidecars) for the
    preview's details pane; ``summary_extra`` appends surface-specific
    lines to the summary (e.g. dataset counts).
    """

    files: list[Path] = field(default_factory=list)
    summary_extra: str = ""


def run_gle_export(
    parent: QWidget,
    *,
    dialog_title: str,
    default_name: str,
    output_format: str,
    prepare: Callable[[], Callable[..., GleExportBuild | None]],
) -> None:
    """Capture widget state, then generate and compile entirely on a worker.

    ``prepare`` runs on the GUI thread and returns a bound data-only builder.
    The returned callable must retain no widgets, artists or panel callbacks.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, dialog_title, default_export_path(default_name), _GLE_EXPORT_FILTER
    )
    if not path:
        return
    remember_export_path(path)
    gle_path, export_dir = resolve_gle_export_paths(Path(path), folder=True)
    gle_path, export_dir = gle_path.absolute(), export_dir.absolute()
    destination = export_dir.resolve()
    for active in QApplication.topLevelWidgets():
        if (
            isinstance(active, GlePlotPreview)
            and active.destination == destination
            and active.tasks.active_count
        ):
            active.show()
            active.raise_()
            active.activateWindow()
            return
    owner = parent
    while not isinstance(owner, QMainWindow) and owner.parentWidget() is not None:
        owner = owner.parentWidget()
    preview = GlePlotPreview(owner, destination=destination)
    preview.details.setPlainText(f"GLE script: {gle_path}")
    preview.show()
    build = prepare()
    executable = get_gle_executable()

    def generate(worker):
        from asymmetry.gui.tasks import check_task_cancelled

        start = time.perf_counter()
        try:
            glp = importlib.import_module("gleplot")
        except ImportError as exc:
            raise RuntimeError("Install gleplot to export GLE plots.") from exc
        export_dir.mkdir(parents=True, exist_ok=True)
        check_task_cancelled(worker)
        result = build(glp, gle_path, export_dir, cancelled=worker.is_cancelled)
        if result is None:
            return None
        check_task_cancelled(worker)
        files = list(
            dict.fromkeys(
                [
                    gle_path,
                    *result.files,
                    *(export_dir / name for name in extract_gle_data_dependencies(gle_path)),
                ]
            )
        )
        prune_stale_sidecars(export_dir, gle_path, files)
        generation_seconds = time.perf_counter() - start
        if executable is None:
            return GleExportResult(
                tuple(files),
                None,
                "GLE script and data saved. Install GLE to compile the plot.",
                generation_seconds,
                0.0,
            )
        worker.progress.emit(0, 0, "Compiling plot…")
        devices = (output_format, "pdf") if output_format != "pdf" else ("pdf",)
        outputs = [gle_path.with_suffix(f".{device}") for device in devices]
        for output in outputs:
            output.unlink(missing_ok=True)
        start = time.perf_counter()
        try:
            compile_gle(
                executable, gle_path, devices, cwd=export_dir, cancelled=worker.is_cancelled
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr or exc.stdout or str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"GLE did not finish within {exc.timeout:g}s and was stopped. Script saved at {gle_path}."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not run GLE ({executable}): {exc}. Check Setup ▸ GLE Setup…"
            ) from exc
        check_task_cancelled(worker)
        for output in outputs:
            if not output.is_file():
                raise RuntimeError(f"GLE did not create {output}.")
        pdf = gle_path.with_suffix(".pdf").read_bytes()
        compilation_seconds = time.perf_counter() - start
        summary = f"GLE plot exported to {export_dir}"
        if result.summary_extra:
            summary += f"\n{result.summary_extra}"
        _log.debug(
            "GLE generation %.3fs; compilation %.3fs", generation_seconds, compilation_seconds
        )
        return GleExportResult(
            tuple(dict.fromkeys([*files, *outputs])),
            pdf,
            summary,
            generation_seconds,
            compilation_seconds,
        )

    preview.tasks.start(
        generate,
        on_finished=preview.export_finished,
        on_error=preview.export_failed,
        on_progress=preview.export_progress,
    )
