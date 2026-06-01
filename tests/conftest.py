from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path: Path) -> Iterator[None]:
    """Redirect ``QSettings`` to a per-test temp location.

    The application uses default-named ``QSettings()``, which otherwise share a
    single backend file across every test *and* every ``pytest-xdist`` worker.
    Under parallel runs that shared state leaks between tests (e.g. the UI-scale
    persistence assertions intermittently reading another test's value), so we
    point each test at its own temporary ini file. No-op if PySide6 is absent.
    """
    try:
        from PySide6.QtCore import QSettings
    except Exception:
        yield
        return

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests that carry no explicit type marker as 'unit'."""
    type_markers = {"unit", "gui", "io"}
    for item in items:
        if not (set(m.name for m in item.iter_markers()) & type_markers):
            item.add_marker(pytest.mark.unit)
