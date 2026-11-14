"""Vector preview lifecycle and editor retirement regressions."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from asymmetry.gui.windows.gle_preview import GlePlotPreview  # noqa: E402

pytestmark = pytest.mark.gui


def test_invalid_pdf_reports_failure(qapp):
    parent = QWidget()
    preview = GlePlotPreview(parent)
    preview.set_pdf(b"invalid")
    assert "Preview unavailable" in preview.status.text()
    preview.close()


def test_preview_is_nonmodal(qapp):
    parent = QWidget()
    preview = GlePlotPreview(parent)
    assert not preview.isModal()
    assert preview.windowTitle() == "GLE Plot Preview"
    preview.close()


def test_pdf_snapshot_survives_source_deletion(qapp, tmp_path):
    from PySide6.QtGui import QPainter, QPdfWriter

    path = tmp_path / "plot.pdf"
    writer = QPdfWriter(str(path))
    painter = QPainter(writer)
    painter.drawText(10, 100, "Vector plot")
    painter.end()
    del writer
    parent = QWidget()
    preview = GlePlotPreview(parent)
    preview.set_pdf(path.read_bytes())
    path.unlink()
    assert preview.document.pageCount() == 1
    preview.zoom_in()
    preview.fit()
    assert preview.document.pageCount() == 1
    preview.close()


def test_editor_action_is_removed(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    try:
        labels = {
            action.text()
            for menu_action in window.menuBar().actions()
            if menu_action.menu() is not None
            for action in menu_action.menu().actions()
        }
        assert "GLE Figure &Editor…" not in labels
    finally:
        window.close()
