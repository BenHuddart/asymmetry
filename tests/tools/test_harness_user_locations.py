from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = runpy.run_path(str(ROOT / "tools" / "harness.py"))


def test_user_locations_have_one_owner() -> None:
    assert HARNESS["find_user_location_violations"](ROOT) == []


@pytest.mark.parametrize(
    "constructor",
    [
        'QSettings("Org", "App")',
        'QSettings("Org")',
        'QSettings(organization="Org", application="App")',
        'QSettings(QSettings.Format.NativeFormat, scope, "Org", "App")',
    ],
)
def test_direct_standard_paths_and_native_settings_are_rejected(
    tmp_path: Path, constructor: str
) -> None:
    gui = tmp_path / "src/asymmetry/gui"
    gui.mkdir(parents=True)
    (gui / "panel.py").write_text("from PySide6.QtCore import QStandardPaths as Paths\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_panel.py").write_text(f"settings = {constructor}\n")

    failures = HARNESS["find_user_location_violations"](tmp_path)

    assert {failure.path.name for failure in failures} == {"panel.py", "test_panel.py"}


def test_explicit_settings_files_and_the_app_data_owner_are_allowed(tmp_path: Path) -> None:
    gui = tmp_path / "src/asymmetry/gui/utils"
    gui.mkdir(parents=True)
    (gui / "app_data.py").write_text("from PySide6.QtCore import QStandardPaths\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_panel.py").write_text(
        'settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)\n'
    )

    assert HARNESS["find_user_location_violations"](tmp_path) == []


def test_gui_marker_is_required_outside_the_gui_test_directory(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    path = tools / "test_example.py"
    path.write_text("def test_widget(qapp): pass\n", encoding="utf-8")

    assert [f.path for f in HARNESS["find_missing_gui_marker_violations"](tmp_path)] == [path]
