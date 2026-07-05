"""Load-report dialog for user functions (Setup → User functions…).

Read-only view of the most recent plugin discovery pass: which files and
entry points were scanned, what each registered, and the full error text for
anything that failed — so a user wondering why their function is missing has
one place to look, any time after startup.
"""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.plugins import (
    USER_FUNCTIONS_DIR,
    UserFunctionLoadReport,
    last_load_report,
)

_KIND_LABELS = {
    "component": "fit component",
    "parameter_component": "parameter-trend component",
}


def _report_html(report: UserFunctionLoadReport | None) -> str:
    directory = report.directory if report is not None else str(USER_FUNCTIONS_DIR)
    parts = [
        "<h2>User functions</h2>",
        f"<p>Plugin directory: <code>{html.escape(directory)}</code><br>"
        "Files are loaded once at startup; restart Asymmetry after adding or "
        "editing plugins. Packaged plugins register through the "
        "<code>asymmetry.user_functions</code> entry-point group.</p>",
        "<p><i>User functions are ordinary Python run with full interpreter "
        "privileges. Only install files you trust.</i></p>",
    ]

    if report is None or not report.sources:
        parts.append("<p><b>No user functions were found at the last scan.</b></p>")
        return "".join(parts)

    parts.append(f"<p><b>{html.escape(report.summary())}</b></p>")
    for source in report.sources:
        origin = "file" if source.kind == "file" else "entry point"
        title = f"{html.escape(source.name)} <i>({origin})</i>"
        if source.ok:
            registered = (
                "; ".join(
                    f"<code>{html.escape(name)}</code> ({_KIND_LABELS.get(kind, kind)})"
                    for kind, name in source.registered
                )
                or "nothing registered"
            )
            parts.append(f"<h3>✓ {title}</h3><p>{registered}</p>")
        else:
            parts.append(
                f"<h3>✗ {title}</h3>"
                f"<p><b>{html.escape(source.error or 'failed')}</b></p>"
                + (
                    f"<pre style='font-size: 11px;'>{html.escape(source.detail)}</pre>"
                    if source.detail
                    else ""
                )
            )
    return "".join(parts)


class UserFunctionsDialog(QDialog):
    """Show the most recent user-function load report."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("User functions")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        report = last_load_report()
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        browser.setHtml(_report_html(report))
        layout.addWidget(browser)

        hint = QLabel(
            "Write plugins with asymmetry.register_component / "
            "register_parameter_component — see the user guide chapter "
            '"User functions". Deleting a function\'s file removes it at the '
            "next start; projects that reference it still open, with the "
            "function shown as missing until the file returns."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._open_folder_button = button_box.addButton(
            "Open folder…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._open_folder_button.clicked.connect(self._open_user_functions_folder)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _open_user_functions_folder(self) -> None:
        """Open the user-functions directory in the system file browser.

        The directory is created first so the button works on a fresh install,
        and ``USER_FUNCTIONS_DIR`` is resolved at click time so tests (and any
        future relocation) see the current value rather than an import-time
        snapshot.
        """
        from asymmetry.core import plugins

        directory = Path(plugins.USER_FUNCTIONS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
