"""Tests for the shared matplotlib canvas factory (gui/widgets/mpl_canvas.py)."""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from asymmetry.gui.widgets.mpl_canvas import create_canvas  # noqa: E402


def test_importing_module_does_not_eagerly_import_matplotlib() -> None:
    """The lazy-import contract: importing mpl_canvas must not pull matplotlib.

    Several panels guard canvas creation behind ``try/except ImportError`` and
    rely on being able to import this module in a no-matplotlib context.
    """
    # A fresh subprocess is the only way to assert this cleanly, since the test
    # session itself has already imported matplotlib elsewhere.
    code = (
        "import sys; import asymmetry.gui.widgets.mpl_canvas as m; "
        "print('matplotlib' in sys.modules)"
    )
    import subprocess

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_default_returns_figure_and_canvas_with_tight_layout(qapp: object) -> None:
    figure, canvas = create_canvas()
    assert canvas.figure is figure
    assert figure.get_tight_layout()


def test_constrained_layout(qapp: object) -> None:
    figure, _canvas = create_canvas(layout="constrained")
    assert figure.get_constrained_layout()


@pytest.mark.parametrize("layout", ["none", None])
def test_plain_layout_has_neither_engine(qapp: object, layout: str | None) -> None:
    figure, _canvas = create_canvas(layout=layout)
    assert not figure.get_tight_layout()
    assert not figure.get_constrained_layout()


def test_toolbar_true_returns_navigation_toolbar(qapp: object) -> None:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    figure, canvas, toolbar = create_canvas(toolbar=True)
    assert canvas.figure is figure
    assert isinstance(toolbar, NavigationToolbar2QT)


def test_figsize_is_forwarded(qapp: object) -> None:
    figure, _canvas = create_canvas(figsize=(4.0, 3.0))
    assert tuple(figure.get_size_inches()) == (4.0, 3.0)


def test_unknown_layout_raises_valueerror(qapp: object) -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        create_canvas(layout="bogus")
