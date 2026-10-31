from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from asymmetry.core import plugins
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.utils import fit_times
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui


def test_settings_and_plugins_use_test_locations(qapp, tmp_path: Path) -> None:
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        settings = QSettings(QSettings.Format.IniFormat, scope, "Asymmetry", "Asymmetry")
        assert Path(settings.fileName()).is_relative_to(tmp_path)
    assert Path(QSettings().fileName()).is_relative_to(tmp_path)
    assert plugins.USER_FUNCTIONS_DIR.is_relative_to(tmp_path)


@pytest.mark.parametrize("sentinel", ["first", "second"])
def test_settings_do_not_leak_between_tests(qapp, sentinel: str) -> None:
    settings = QSettings()
    assert not settings.contains("isolation/sentinel")
    settings.setValue("isolation/sentinel", sentinel)
    settings.sync()


def test_clean_window_close_does_not_create_app_data(qapp, tmp_path: Path) -> None:
    from asymmetry.gui.utils import app_data

    directory = app_data.app_data_dir()
    assert directory.is_relative_to(tmp_path)
    assert fit_times.fit_times_path().is_relative_to(directory)
    window = MainWindow()
    assert Path(window._autosave_path()).is_relative_to(directory)
    window.close()
    assert not directory.exists()


def test_autosave_round_trip_uses_test_app_data(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    path = Path(window._autosave_path())
    assert path.is_relative_to(tmp_path)
    window._mark_dirty()
    window._on_autosave_timeout()
    wait_for(lambda: not window._project_save_active, qapp)
    assert path.is_file()
    window._clear_dirty()
    window.close()
    assert not path.exists()


def test_unwritable_app_data_logs_autosave_failure(qapp, tmp_path: Path, monkeypatch) -> None:
    from asymmetry.gui.utils import app_data

    window = MainWindow()
    messages: list[str] = []
    monkeypatch.setattr(window._log_panel, "log", messages.append)
    blocker = tmp_path / "regular-file"
    blocker.write_text("blocked", encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(app_data, "app_data_dir", lambda: blocker / "data")
        window._mark_dirty()
        window._on_autosave_timeout()
        wait_for(lambda: not window._project_save_active, qapp)
    assert any(message.startswith("Autosave failed:") for message in messages)
    window._clear_dirty()
    window.close()
