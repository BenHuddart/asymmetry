"""RED target for branch ``fix/freq-global-fit-result-fields``.

Follow-up flagged in PR #108: the **frequency-domain** ``GlobalFitTab`` rebuilds
each member ``FitResult`` field-by-field (``fit_panel.py`` ~L5132, to append the
field-derived parameters) but silently **drops ``dof`` and ``minos_errors``** —
they reset to their defaults (``dof=0``, ``minos_errors=None``). Any other field
added to ``FitResult`` in future would be dropped the same way.

Fix: rebuild with ``dataclasses.replace(result, parameters=…, uncertainties=…)``
so every unrelated field carries through, overriding only the two that change.

This test runs that frequency-domain emit path and asserts ``dof`` and
``minos_errors`` survive into the emitted result. RED today (both are dropped).
"""

from __future__ import annotations

import dataclasses
import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _freq_result() -> FitResult:
    # A frequency-domain spectral fit carrying both fields the rebuild drops, plus
    # a spread of other non-overridden fields used by the carry-through guard.
    return FitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=1.05,
        parameters=ParameterSet([Parameter("nu0", 5.42), Parameter("A_1", 9.0)]),
        uncertainties={"nu0": 0.01, "A_1": 0.1},
        message="Global fit successful",
        function_calls=321,
        edm=1.2e-5,
        covariance_accurate=True,
        dof=1977,
        minos_errors={"nu0": (-0.012, 0.013)},
        warnings=["Fixed-frequency trap: …"],
    )


def _emit(tab, monkeypatch, result: FitResult) -> FitResult:
    """Drive the frequency-domain emit path and return the single emitted result.

    Isolates the rebuild: skip HTML rendering and curve generation, and let the
    emitted results pass through unchanged so we can inspect them off the signal.
    """
    tab._domain = "frequency"
    monkeypatch.setattr(tab, "_render_global_fit_success", lambda **kwargs: None)
    monkeypatch.setattr(tab, "_results_with_curves", lambda model, results: results)

    captured: dict[int, FitResult] = {}
    tab.global_fit_completed.connect(lambda results, glob: captured.update(results))

    tab._emit_global_fit_success(
        model=object(),
        results_dict={10: result},
        fitted_global=ParameterSet(),
        global_param_names=[],
    )
    return captured[10]


def test_frequency_emit_preserves_dof_and_minos(mw, monkeypatch):
    emitted = _emit(mw._fit_panel._global_tab, monkeypatch, _freq_result())
    assert emitted.dof == 1977, "frequency-domain rebuild dropped dof"
    assert emitted.minos_errors == {"nu0": (-0.012, 0.013)}, (
        "frequency-domain rebuild dropped minos_errors"
    )


def test_frequency_emit_carries_through_all_unrelated_fields(mw, monkeypatch):
    """Only parameters/uncertainties change; every other field survives the rebuild.

    Locks the ``dataclasses.replace`` contract so a future hand-rebuild that drops
    a field is caught broadly, not just for dof/minos_errors.
    """
    source = _freq_result()
    emitted = _emit(mw._fit_panel._global_tab, monkeypatch, source)

    # The two overridden fields gained the derived field parameters (B0/Bwid) in a
    # fresh container, while still carrying the originals.
    assert "B0" in emitted.parameters
    assert emitted.uncertainties is not source.uncertainties
    assert {"nu0", "A_1", "B0"} <= set(emitted.uncertainties)

    # Everything else carries through by reference — dataclasses.replace copies
    # non-overridden fields verbatim, so identity is the exact contract (and it
    # sidesteps the ambiguous truth value of comparing array fields with ==).
    for field in dataclasses.fields(FitResult):
        if field.name in ("parameters", "uncertainties"):
            continue
        assert getattr(emitted, field.name) is getattr(source, field.name), (
            f"frequency-domain rebuild dropped/altered {field.name}"
        )
