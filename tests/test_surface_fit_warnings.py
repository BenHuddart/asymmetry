"""RED target for branch ``fix/surface-fit-warnings``.

#100/#101 added discoverability warnings emitted from ``FitEngine.fit``
(``AsymmetryScaleWarning``, ``FixedFrequencyFieldMismatchWarning``). But the GUI
does **not** capture Python warnings (confirmed by grep: the fit panel / mainwindow
have no ``warnings.catch_warnings`` / ``showwarning`` hook), so they only reach
stderr/logs — the user never sees them in the fit panel. This defeats the whole
"point the user at the fix" purpose.

Desired behaviour: the fit worker captures warnings emitted during the fit and the
fit panel surfaces them (results box and/or log) alongside the result.

This is design-led (where the warning is captured + how it is carried to the panel
is the implementer's call). The test pins the *contract* — a completed fit that
warned must carry its warning text to a place the GUI renders. xfail(strict) until
implemented; remove the marker once the surfacing path exists.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fourier.units import gauss_to_mhz

pytest.importorskip("iminuit")


def _tf_dataset(field_gauss: float = 400.0) -> MuonDataset:
    t = np.linspace(0.05, 8.0, 400)
    f_true = float(gauss_to_mhz(field_gauss))
    asym = 9.0 * np.cos(2 * np.pi * f_true * t) * np.exp(-((1.2 * t) ** 2)) - 23.0
    return MuonDataset(t, asym, np.full_like(t, 0.3), {"field": field_gauss, "run_number": 1277})


def _fixed_freq_params(model) -> ParameterSet:
    params: list[Parameter] = []
    for name in model.param_names:
        low = name.lower()
        is_freq = "freq" in low
        value = 6.0 if is_freq else {"sigma": 1.0}.get(low, 9.0 if low.startswith("a_1") else
                 (-23.0 if "bg" in low else float(model.param_defaults.get(name, 0.0))))
        params.append(Parameter(name, value, fixed=is_freq))
    return ParameterSet(params)


def test_engine_emits_warning_baseline() -> None:
    """Sanity: the engine really does warn here (so the GUI has something to surface)."""
    ds = _tf_dataset(400.0)
    model = CompositeModel.from_expression("Oscillatory * Gaussian + Constant").to_model_definition()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitEngine().fit(ds, model.function, _fixed_freq_params(model), 0.05, 8.0)
    assert any("freq" in str(w.message).lower() or "field" in str(w.message).lower() for w in caught)


@pytest.mark.xfail(reason="fix/surface-fit-warnings not yet implemented", strict=True)
def test_fit_result_carries_warnings_for_the_gui() -> None:
    """Contract: a warned fit carries its warning text for the panel to display.

    Implementer: capture warnings in the fit worker and expose them (e.g. a
    ``FitResult.warnings`` list the results box renders, or an explicit signal).
    Adjust this assertion to the chosen seam, then drop the xfail.
    """
    ds = _tf_dataset(400.0)
    model = CompositeModel.from_expression("Oscillatory * Gaussian + Constant").to_model_definition()
    result = FitEngine().fit(ds, model.function, _fixed_freq_params(model), 0.05, 8.0)

    surfaced = getattr(result, "warnings", None)
    assert surfaced, "FitResult exposes no warnings for the GUI to surface"
    assert any("freq" in str(w).lower() or "field" in str(w).lower() for w in surfaced)
