from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def qapp() -> object:
    """Return the process-wide ``QApplication``, creating it once per session.

    A single application instance is reused across every GUI test. Defining it
    here (session scope) decouples the application's lifetime from the xdist
    distribution strategy: workers no longer rebuild the app per file. Test
    modules that still declare their own module-scoped ``qapp`` fixture simply
    shadow this one harmlessly. Returns ``None`` if PySide6 is unavailable.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _bypass_unsaved_changes_prompt(request: pytest.FixtureRequest) -> Iterator[None]:
    """Suppress ``MainWindow``'s unsaved-changes close prompt by default.

    ``closeEvent`` now pops a modal Save/Discard/Cancel dialog when the session
    has unsaved work (F4 / P0-2). Many GUI tests force-close a dirty window
    in-body (``window.close()`` in a ``finally:``), where that modal would block
    the offscreen run forever. We therefore stub ``_maybe_save`` to "proceed"
    for every test except those explicitly marked ``real_save_guard``, which
    exercise the guard itself (with their own ``QMessageBox`` stubs).

    Imported lazily so non-GUI test sessions (no PySide6) pay nothing.
    """
    if "real_save_guard" in request.keywords:
        yield
        return
    try:
        from asymmetry.gui.mainwindow import MainWindow
    except Exception:
        yield
        return
    original = MainWindow._maybe_save
    MainWindow._maybe_save = lambda self, *a, **k: True  # type: ignore[assignment]
    try:
        yield
    finally:
        MainWindow._maybe_save = original  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _cleanup_qt_widgets() -> Iterator[None]:
    """Tear down all Qt state after every test to prevent cross-test bleed.

    GUI tests historically created ``MainWindow`` (and other widgets) without
    ever closing them. Under a shared ``QApplication`` the windows and their
    parentless child widgets accumulated across a file (~58 top-level widgets
    per ``MainWindow``). Because several construction steps scan every live
    top-level widget, per-test setup time grew linearly with the number of
    leaked windows — turning the file into an O(n^2) crawl (observed ~1s ->
    45s ``MainWindow`` setup) and leaking state that made tests order-dependent.

    The non-obvious part of the fix: ``deleteLater`` posts a ``DeferredDelete``
    event that a bare ``processEvents()`` does NOT dispatch when there is no
    running event loop (the test runner has none). Without explicitly flushing
    those events the widgets are never actually destroyed. We therefore close
    each top-level widget (firing real ``closeEvent`` handlers, where
    ``MainWindow`` joins its MaxEnt thread), schedule deletion, then force the
    ``DeferredDelete`` events through with ``sendPostedEvents`` — which drops
    the live top-level count back to zero between tests and keeps setup flat.

    The fixture depends on no other fixture (it queries
    ``QApplication.instance()`` directly) so module-local ``qapp`` fixtures
    cannot shadow it, and it is a no-op when PySide6 is absent or no
    application has been created (pure-unit tests).
    """
    try:
        from PySide6.QtCore import QEvent, QThread
        from PySide6.QtWidgets import QApplication
    except Exception:
        yield
        return

    app = QApplication.instance()
    if app is None:
        yield
        return

    yield

    _deferred_delete = QEvent.Type.DeferredDelete.value

    def _drain() -> None:
        # Force-dispatch deferred deletions; a bare processEvents() will not
        # flush DeferredDelete events outside a running event loop.
        try:
            app.sendPostedEvents(None, _deferred_delete)
            app.processEvents()
            app.sendPostedEvents(None, _deferred_delete)
        except Exception:
            pass

    # Stop any QThread still running (e.g. a MaxEnt worker whose window was not
    # closed by the test). Bounded wait; pytest-timeout is the hard ceiling.
    try:
        for thread in app.findChildren(QThread):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except Exception:
                pass
    except Exception:
        pass

    # Close every top-level widget (runs closeEvent handlers), then schedule
    # and flush deletion so the C++ objects are gone before the next test.
    # MainWindow's closeEvent now guards unsaved work with a modal Save/Discard/
    # Cancel prompt (F4/P0-2). In an offscreen teardown that dialog would block
    # forever, so force-clear the dirty flag first — this is a non-interactive
    # force-close, the production guard is exercised by its own tests.
    for widget in list(app.topLevelWidgets()):
        clear_dirty = getattr(widget, "_clear_dirty", None)
        if callable(clear_dirty):
            try:
                clear_dirty()
            except Exception:
                pass
        try:
            widget.close()
        except Exception:
            pass
    for widget in list(app.topLevelWidgets()):
        try:
            widget.deleteLater()
        except Exception:
            pass
    _drain()

    # Release any matplotlib figures embedded in the closed widgets.
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


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


@pytest.fixture(autouse=True)
def _reset_ui_font_scale() -> Iterator[None]:
    """Reset the process-wide UI font scale to 1.0 around every test.

    ``ui_manager.UIManager`` publishes the active scale to the font builders via
    ``fonts.set_ui_font_scale`` (so chrome/value fonts built after a scale change
    are born scaled). That module global persists across tests in a worker
    process; without a reset, a test that constructs ``MainWindow`` (applying the
    0.9 default scale) would leak a non-unity scale into a later test that asserts
    design-size fonts. No-op when PySide6 is unavailable.
    """
    try:
        from asymmetry.gui.styles import fonts
    except Exception:
        yield
        return

    fonts.set_ui_font_scale(1.0)
    yield
    fonts.set_ui_font_scale(1.0)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests that carry no explicit type marker as 'unit'."""
    type_markers = {"unit", "gui", "io"}
    for item in items:
        if not (set(m.name for m in item.iter_markers()) & type_markers):
            item.add_marker(pytest.mark.unit)


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
