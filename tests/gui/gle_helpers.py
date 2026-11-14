"""Recording factories and worker completion for export behaviour tests."""

from pathlib import Path

from PySide6.QtWidgets import QApplication

from asymmetry.gui.utils import global_gle, parameter_gle, plot_gle
from asymmetry.gui.windows.gle_preview import GlePlotPreview
from tests._qt_helpers import wait_for


def recording_factories(monkeypatch):
    """Keep tests of axis commands independent of gleplot configuration."""
    for module in (plot_gle, parameter_gle, global_gle):
        monkeypatch.setattr(module, "gle_figure", lambda glp, **kwargs: glp.figure(**kwargs))
    for module in (plot_gle, parameter_gle, global_gle):
        monkeypatch.setattr(module, "gle_subplots", recording_subplots)


def recording_subplots(glp, *, nrows, ncols, figsize, sharex=False):
    if hasattr(glp, "subplots"):
        return glp.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)
    figure = glp.figure(figsize=figsize)
    axes = []
    for index in range(nrows * ncols):
        axes.append(
            figure.add_subplot(nrows, ncols, index + 1, sharex=axes[0] if sharex and axes else None)
        )
    return figure, axes


def wait_for_gle(owner):
    previews = owner.window().findChildren(GlePlotPreview)
    wait_for(lambda: all(not p.tasks.active_count for p in previews), QApplication.instance())
    return previews[-1] if previews else None


def write_compile_outputs(script, formats):
    """Write a real one-page PDF so UI tests can inspect successful previews."""
    parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 150] /Contents 4 0 R >>",
            b"<< /Length 0 >>\nstream\n\nendstream",
        ],
        start=1,
    ):
        offsets.append(sum(map(len, parts)))
        parts.append(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = sum(map(len, parts))
    parts.append(b"xref\n0 5\n0000000000 65535 f \n")
    parts.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    parts.append(f"trailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n{xref}\n%%EOF\n".encode())
    for fmt in (formats,) if isinstance(formats, str) else formats:
        Path(script).with_suffix("." + fmt).write_bytes(
            b"".join(parts) if fmt == "pdf" else b"output"
        )
