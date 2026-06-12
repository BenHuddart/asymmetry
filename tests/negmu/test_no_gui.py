"""No-GUI and no-registry isolation guards for core/negmu (verification-plan §4).

These tests ensure the μ⁻ package is invisible to the GUI fit builders and does
not import count_domain or Qt. They execute after the other negmu imports so the
modules are already in sys.modules and the assertions are fast.
"""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# Qt-free imports
# ---------------------------------------------------------------------------


_QT_KEYS = ("PySide6.QtCore", "PySide6.QtWidgets")


def _qt_loaded_by(import_stmt: str) -> set[str]:
    """Return the set of Qt keys added to sys.modules by executing import_stmt."""
    before = {k for k in _QT_KEYS if k in sys.modules}
    exec(import_stmt, {})  # noqa: S102
    after = {k for k in _QT_KEYS if k in sys.modules}
    return after - before


def test_lifetimes_no_qt():
    new = _qt_loaded_by("import asymmetry.core.negmu.lifetimes")
    assert not new, f"negmu.lifetimes triggered Qt load: {new}"


def test_model_no_qt():
    new = _qt_loaded_by("import asymmetry.core.negmu.model")
    assert not new, f"negmu.model triggered Qt load: {new}"


def test_fit_no_qt():
    new = _qt_loaded_by("import asymmetry.core.negmu.fit")
    assert not new, f"negmu.fit triggered Qt load: {new}"


def test_ratio_no_qt():
    new = _qt_loaded_by("import asymmetry.core.negmu.ratio")
    assert not new, f"negmu.ratio triggered Qt load: {new}"


def test_simulate_capture_no_qt():
    new = _qt_loaded_by("import asymmetry.core.simulate")
    assert not new, f"core.simulate triggered Qt load: {new}"


# ---------------------------------------------------------------------------
# count_domain not imported by negmu
# ---------------------------------------------------------------------------


def test_negmu_does_not_import_count_domain():
    """core/negmu must not import count_domain (private helpers off-limits)."""
    # If count_domain was imported by negmu it would appear here;
    # it may be present from other tests but that's OK — what matters
    # is negmu doesn't trigger the import on its own (covered by order).
    # The definitive check is source-level: grep count_domain in negmu/*.py.
    import pathlib

    import asymmetry.core.negmu.fit  # noqa: F401
    import asymmetry.core.negmu.lifetimes  # noqa: F401
    import asymmetry.core.negmu.model  # noqa: F401
    import asymmetry.core.negmu.ratio  # noqa: F401

    negmu_dir = pathlib.Path(asymmetry.core.negmu.fit.__file__).parent
    for src in negmu_dir.glob("*.py"):
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # provenance comments are fine
            assert "count_domain" not in stripped, (
                f"{src.name}:{lineno} imports or references count_domain — forbidden"
            )


# ---------------------------------------------------------------------------
# No-GUI registry check: negmu labels absent from COMPONENTS and MODELS
# ---------------------------------------------------------------------------


def test_negmu_labels_not_in_components():
    from asymmetry.core.fitting.composite import COMPONENTS

    negmu_labels = ["CaptureComponent", "CaptureModelSpec", "capture_count_model"]
    for label in negmu_labels:
        assert label not in COMPONENTS, f"{label!r} found in COMPONENTS registry"


def test_negmu_labels_not_in_models():
    from asymmetry.core.fitting.models import MODELS

    negmu_labels = ["CaptureComponent", "CaptureModelSpec", "capture_count_model"]
    for label in negmu_labels:
        assert label not in MODELS, f"{label!r} found in MODELS registry"


def test_negmu_init_not_imported_by_gui_init():
    """asymmetry.gui does not eagerly import core.negmu (no accidental registration)."""
    # We can only probe what's in sys.modules at the time of this test;
    # GUI tests load Qt so we just check negmu is not listed as a dependency
    # of asymmetry.gui via __init__ inspection.
    import inspect

    try:
        import asymmetry.gui

        src = inspect.getsource(asymmetry.gui)
    except (ImportError, TypeError, OSError):
        pytest.skip("GUI not importable in this environment")
    assert "negmu" not in src, "asymmetry.gui.__init__ must not reference negmu"
