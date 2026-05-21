from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests that carry no explicit type marker as 'unit'."""
    type_markers = {"unit", "gui", "io"}
    for item in items:
        if not (set(m.name for m in item.iter_markers()) & type_markers):
            item.add_marker(pytest.mark.unit)
