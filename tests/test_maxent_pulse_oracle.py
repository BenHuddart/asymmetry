"""Oracle check: our pulse-shape kernel vs Mantid's ``MaxentTools.start.START``.

Mantid's pure-numpy ``start.py`` builds the same ISIS proton-pulse / pion-decay
response we fold into the forward model.  It imports standalone (no Mantid
framework), so when a local checkout is present we compare the two kernels at a
documented tolerance.  Nothing is copied — this is a read-only verification.

The checkout lives outside the repo (``~/Source/mantid``), so the test skips
cleanly when it is absent (always the case in CI).  Constants differ slightly
(Mantid truncates τ_µ = 2.19704 vs our CODATA 2.1969811), so the double-pulse
case agrees to ~1e-3 rather than bit-exact; the single-pulse case has no τ_µ
dependence (the tanh interference weight vanishes) and agrees to machine
precision.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.maxent import pulse_response
from asymmetry.core.utils.constants import PION_LIFETIME_US

_START_PATH = Path.home() / "Source" / "mantid" / "scripts" / "Muon" / "MaxentTools" / "start.py"


def _load_start():
    """Load Mantid's standalone ``START``, or return None if unavailable."""
    if not _START_PATH.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_mantid_maxent_start", _START_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.START
    except Exception:  # noqa: BLE001 — any import failure means "no oracle"
        return None


_START = _load_start()


class _SilentLog:
    def notice(self, *_args, **_kwargs) -> None:
        pass


@pytest.mark.skipif(_START is None, reason="Mantid MaxentTools.start not importable")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # start.py's own DC div/0 (it patches gw[0])
@pytest.mark.parametrize("n_pulses", [1, 2])
def test_pulse_kernel_matches_mantid_start(n_pulses: int) -> None:
    res = 0.016  # µs per time channel
    npts = 1000
    n_spectrum = 256
    # Mantid's frequency grid: ν_k = k / (res·npts·2).
    fperchan = 1.0 / (res * float(npts) * 2.0)
    frequencies = np.arange(n_spectrum) * fperchan

    # TZERO_fine = −Tpion cancels START's internal exp(i(TZERO+Tpion)ω) time
    # shift, leaving the raw convolution we want to compare against.
    _detect, convol = _START(
        npts,
        n_pulses,
        res,
        n_spectrum,
        -PION_LIFETIME_US,
        _SilentLog(),
    )
    # START hardcodes ISIS half-width 0.05 µs and separation 0.324 µs.
    p_cos, p_sin = pulse_response(
        frequencies, half_width_us=0.05, separation_us=0.324, n_pulses=n_pulses
    )

    # P(ω) = convolr + i·convoli = P_cos − i·P_sin (our sign convention).
    tol = dict(rtol=1.0e-3, atol=1.0e-6) if n_pulses == 2 else dict(rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(p_cos, np.real(convol), **tol)
    np.testing.assert_allclose(p_sin, -np.imag(convol), **tol)
    # DC point is exact either way.
    assert p_cos[0] == pytest.approx(1.0)
    assert p_sin[0] == pytest.approx(0.0)
    assert math.isclose(float(np.real(convol)[0]), 1.0, abs_tol=1e-12)
