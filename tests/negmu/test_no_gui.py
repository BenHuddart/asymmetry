"""No-GUI and no-registry isolation guards for core/negmu (verification-plan §4).

These tests ensure the μ⁻ package is invisible to the GUI fit builders and does
not import count_domain or Qt.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Qt-free imports (subprocess: the only reliable probe)
# ---------------------------------------------------------------------------

# An in-process sys.modules diff is a NO-OP here: the autouse _cleanup_qt_widgets
# fixture in tests/conftest.py imports PySide6 before every test body, so Qt is
# always already loaded by the time a test runs. A fresh interpreter is the only
# way to verify that importing a negmu module pulls in no Qt / GUI.

_NEGMU_MODULES = [
    "asymmetry.core.negmu",
    "asymmetry.core.negmu.lifetimes",
    "asymmetry.core.negmu.model",
    "asymmetry.core.negmu.fit",
    "asymmetry.core.negmu.ratio",
    "asymmetry.core.negmu.background",
    "asymmetry.core.negmu.polarisation",
    "asymmetry.core.simulate",
]


def _gui_modules_pulled_in_by(module: str) -> tuple[int, str]:
    """Import ``module`` in a fresh interpreter; report any Qt/GUI modules loaded.

    Returns ``(returncode, leaked)`` where ``leaked`` is a ``;``-joined list of
    offending ``sys.modules`` keys (empty when clean).
    """
    code = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401

        leaked = sorted(
            name for name in sys.modules
            if name == "PySide6" or name.startswith("PySide6.")
            or name == "asymmetry.gui" or name.startswith("asymmetry.gui.")
        )
        if leaked:
            print(";".join(leaked))
            sys.exit(1)
        sys.exit(0)
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


@pytest.mark.parametrize("module", _NEGMU_MODULES)
def test_negmu_module_imports_without_qt_or_gui(module):
    """Importing any negmu module in a clean interpreter pulls in no Qt / GUI."""
    returncode, leaked = _gui_modules_pulled_in_by(module)
    assert returncode == 0, f"{module} pulled in Qt/GUI modules: {leaked or '(import failed)'}"


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

    import asymmetry.core.negmu.background  # noqa: F401
    import asymmetry.core.negmu.fit  # noqa: F401
    import asymmetry.core.negmu.lifetimes  # noqa: F401
    import asymmetry.core.negmu.model  # noqa: F401
    import asymmetry.core.negmu.polarisation  # noqa: F401
    import asymmetry.core.negmu.ratio  # noqa: F401

    negmu_dir = pathlib.Path(asymmetry.core.negmu.fit.__file__).parent
    for src in negmu_dir.glob("*.py"):
        for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
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


def test_no_registry_entry_resolves_to_negmu():
    """No COMPONENTS/MODELS entry's function is defined in asymmetry.core.negmu.

    Stronger than the label checks above: it catches an accidental registration
    under *any* label (e.g. 'NegativeMuonCapture') by inspecting where each
    registered function actually lives, not just the keys.
    """
    from asymmetry.core.fitting.composite import COMPONENTS
    from asymmetry.core.fitting.models import MODELS

    offenders: list[str] = []
    for registry_name, registry in (("COMPONENTS", COMPONENTS), ("MODELS", MODELS)):
        for key, definition in registry.items():
            fn = getattr(definition, "function", None)
            module = getattr(fn, "__module__", "") or ""
            if module == "asymmetry.core.negmu" or module.startswith("asymmetry.core.negmu."):
                offenders.append(f"{registry_name}[{key!r}] → {module}")
    assert not offenders, f"negmu functions leaked into GUI registries: {offenders}"


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
