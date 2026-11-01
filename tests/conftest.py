from __future__ import annotations

import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--shard K/N`` for splitting a selection across CI runners.

    CI splits the standard tier across several runners. The split is by a
    stable hash of each test's node id rather than collection order, so the N
    independent shard processes still cover every selected test exactly once,
    with no overlap or gap.
    """
    parser.addoption(
        "--shard",
        action="store",
        default=None,
        metavar="K/N",
        help="Run only shard K of N (e.g. 1/3) of the selected tests, partitioned "
        "by a hash of each test's node id.",
    )


@pytest.fixture(scope="session")
def qapp() -> object:
    """Return the process-wide ``QApplication``, creating it once per session.

    GUI tests request this fixture; core tests never import Qt through it.
    """
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")

    app = qt_widgets.QApplication.instance()
    if app is None:
        app = qt_widgets.QApplication([])
    return app


@pytest.fixture
def _bypass_unsaved_changes_prompt(request: pytest.FixtureRequest) -> Iterator[None]:
    """Suppress ``MainWindow``'s unsaved-changes close prompt by default.

    ``closeEvent`` now pops a modal Save/Discard/Cancel dialog when the session
    has unsaved work (F4 / P0-2). Many GUI tests force-close a dirty window
    in-body (``window.close()`` in a ``finally:``), where that modal would block
    the offscreen run forever. We therefore stub ``_maybe_save`` to "proceed"
    for every test except those explicitly marked ``real_save_guard``, which
    exercise the guard itself (with their own ``QMessageBox`` stubs).

    The GUI fixture gate requests this only for gui-marked tests.
    """
    if "real_save_guard" in request.keywords:
        yield
        return
    from asymmetry.gui.mainwindow import MainWindow

    original = MainWindow._maybe_save
    MainWindow._maybe_save = lambda self, *a, **k: True  # type: ignore[assignment]
    try:
        yield
    finally:
        MainWindow._maybe_save = original  # type: ignore[assignment]


@pytest.fixture
def _cleanup_qt_widgets(qapp) -> Iterator[None]:
    """Close widgets and flush deferred deletion; lifetime failures belong to this test."""
    from tests._qt_helpers import dispose_qt_widgets

    yield
    dispose_qt_widgets(qapp)


@pytest.fixture(autouse=True)
def _isolate_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from discovering user-authored functions from the real home."""
    from asymmetry.core import plugins

    monkeypatch.setattr(plugins, "USER_FUNCTIONS_DIR", tmp_path / "user_functions")


@pytest.fixture(autouse=True)
def _gui_state_gate(request: pytest.FixtureRequest) -> None:
    """Only gui-marked tests load Qt fixtures and window cleanup."""
    if request.node.get_closest_marker("gui") is not None:
        request.getfixturevalue("_isolated_gui_state")
        request.getfixturevalue("_bypass_unsaved_changes_prompt")
        request.getfixturevalue("_cleanup_qt_widgets")


@pytest.fixture
def _isolated_gui_state(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep preferences, app data, and cached fit times inside this test's directory."""
    from PySide6.QtCore import QCoreApplication, QSettings

    from asymmetry.gui.styles import fonts
    from asymmetry.gui.utils import app_data, fit_times

    QCoreApplication.setOrganizationName("Asymmetry")
    QCoreApplication.setApplicationName("Asymmetry")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(tmp_path / "qt-settings"))
    monkeypatch.setattr(app_data, "app_data_dir", lambda: tmp_path / "app-data")
    fit_times.shared_fit_time_store.cache_clear()
    fonts.set_ui_font_scale(1.0)
    yield
    fit_times.shared_fit_time_store.cache_clear()
    fonts.set_ui_font_scale(1.0)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark unmarked tests as 'unit', then apply ``--shard`` if given."""
    type_markers = {"unit", "gui", "io"}
    for item in items:
        if not (set(m.name for m in item.iter_markers()) & type_markers):
            item.add_marker(pytest.mark.unit)

    shard = config.getoption("--shard")
    if not shard:
        return

    try:
        index_str, count_str = str(shard).split("/")
        index, count = int(index_str), int(count_str)
    except ValueError as exc:
        raise pytest.UsageError(f"--shard must be K/N (e.g. 1/3), got {shard!r}") from exc
    if count < 1 or not (1 <= index <= count):
        raise pytest.UsageError(f"--shard K/N requires N >= 1 and 1 <= K <= N, got {shard!r}")

    # A stable node-id hash gives disjoint slices independent of collection order.
    kept, deselected = [], []
    for item in items:
        bucket = zlib.crc32(item.nodeid.encode("utf-8")) % count
        (kept if bucket == index - 1 else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept


@pytest.fixture
def registry_snapshot() -> Iterator[None]:
    """Snapshot and restore every registry a user-function registration mutates.

    Opt-in fixture for tests that call ``register_component`` /
    ``register_parameter_component`` (directly or via plugin discovery), so
    user registrations never leak into other tests' view of the fit-function
    registries or the component-documentation dicts.
    """
    from asymmetry.core.fitting import component_docs
    from asymmetry.core.fitting.composite import COMPONENTS
    from asymmetry.core.fitting.models import MODELS
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    doc_dict_names = (
        "FIT_COMPONENT_APPLICABILITY",
        "FIT_COMPONENT_REFERENCES",
        "PARAMETER_MODEL_APPLICABILITY",
        "PARAMETER_MODEL_REFERENCES",
    )
    registries = (COMPONENTS, MODELS, PARAMETER_MODEL_COMPONENTS)
    saved = [dict(registry) for registry in registries]
    saved_docs = {name: dict(getattr(component_docs, name)) for name in doc_dict_names}
    yield
    for registry, snapshot in zip(registries, saved, strict=True):
        registry.clear()
        registry.update(snapshot)
    for name, snapshot in saved_docs.items():
        live = getattr(component_docs, name)
        live.clear()
        live.update(snapshot)
