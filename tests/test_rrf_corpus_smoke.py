"""Corpus smoke test: RRF on a real 8 T HAL-9500 run (verification item 6).

The corpus lives outside the repository (``~/Documents/WiMDA muon school``),
so this module skips entirely when it is absent — same pattern as
``test_maxent_corpus_smoke.py``.

Run: κ-ET crystal, 8 T, 100 K (``tdc_hifi_2020_00739.mdu``), the canonical
RRF regime — ν_μ ≈ 1085 MHz, ~10⁴ cycles in the window. Two corpus facts the
test design rests on (recorded in the study's test-data.md):

- The file holds the forward octagon only (MV + F1–F8); the loader's default
  grouping pairs F1 against the monitor. The per-detector phases at 8 T span
  the full circle, so any detector-summed FB asymmetry washes out — the
  textbook reason per-detector (Rainford) RRF exists (out of scope here).
  The test therefore builds the asymmetry from the near-opposite pair F2/F7.
- The line sits at ≈ 1084.95 MHz, a few hundred ppm above γ_μ·8 T.

The check: fitting the *raw* pair asymmetry through the rotating-frame
offset wrapper and demodulating at the fitted lab frequency must give a
display envelope that decays at the fitted relaxation rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.rrf_offset import apply_rrf_offsets, rrf_offset_model
from asymmetry.core.transform import compute_asymmetry
from asymmetry.core.transform.rrf import rrf_demodulate

_CORPUS = Path.home() / "Documents" / "WiMDA muon school"
_RUN = _CORPUS / "Magnetism" / "AFM transition in high TF" / "data" / "tdc_hifi_2020_00739.mdu"

pytestmark = pytest.mark.skipif(not _RUN.is_file(), reason="WiMDA muon school corpus not present")

_NU_FRAME = 1084.95  # MHz — the measured line; the frame for the offset fit.
_BUNCH = 8  # 24.4 ps → 195 ps bins: 4.7 samples/cycle, ~80 counts/bin.


def _pair_asymmetry():
    from asymmetry.core.io import load

    run = load(str(_RUN)).run
    grouping = run.grouping
    h = run.histograms
    bin_width = float(h[0].bin_width)
    t0 = int(grouping["t0_bin"])
    first_good = int(grouping["first_good_bin"])
    f2 = np.asarray(h[2].counts, dtype=float)
    f7 = np.asarray(h[7].counts, dtype=float)
    n = min(f2.size, f7.size, int(grouping["last_good_bin"]))
    n_out = (n - first_good) // _BUNCH
    sel = slice(first_good, first_good + n_out * _BUNCH)
    forward = f2[sel].reshape(n_out, _BUNCH).sum(axis=1)
    backward = f7[sel].reshape(n_out, _BUNCH).sum(axis=1)
    time = (
        np.arange(first_good, first_good + n_out * _BUNCH).reshape(n_out, _BUNCH).mean(axis=1) - t0
    ) * bin_width
    asym, err = compute_asymmetry(forward, backward, alpha=1.0)
    return time, 100.0 * asym, 100.0 * err


def test_demodulated_envelope_matches_fitted_relaxation():
    time, asym, err = _pair_asymmetry()
    ok = np.isfinite(asym) & np.isfinite(err) & (err > 0)
    assert ok.sum() > 10_000

    model = CompositeModel.from_expression("Oscillatory * Exponential")
    wrapped = rrf_offset_model(model, _NU_FRAME)
    params = ParameterSet()
    params.add(Parameter(name="A_1", value=10.0, min=0.0, max=60.0))
    params.add(Parameter(name="Lambda", value=0.2, min=0.0, max=20.0))
    params.add(Parameter(name="frequency", value=0.0, min=-30.0, max=30.0))
    params.add(Parameter(name="phase", value=1.4, min=-7.0, max=7.0))
    result = FitEngine().fit(
        MuonDataset(time=time[ok], asymmetry=asym[ok], error=err[ok], metadata={}),
        wrapped,
        params,
    )
    assert result.success
    values = {p.name: p.value for p in result.parameters}
    # The rotating-frame frequency is a small offset, not a GHz number.
    assert abs(values["frequency"]) < 2.0
    assert 5.0 < values["A_1"] < 60.0
    assert 0.05 < values["Lambda"] < 2.0

    lab = apply_rrf_offsets(values, wrapped.rrf_offsets)
    curve = rrf_demodulate(
        time,
        asym,
        err,
        frequency_mhz=lab["frequency"],
        phase_deg=float(np.degrees(values["phase"])),
        bandwidth_mhz=10.0,
    )

    def window_mean(lo: float, hi: float) -> float:
        sel = curve.valid & (time >= lo) & (time < hi)
        assert sel.sum() > 100
        return float(np.mean(curve.real[sel]))

    # The in-phase envelope must decay at the fitted rate. Stay inside
    # [0.1, 4] µs: beyond that a second close line in κ-ET beats slowly
    # against the fit's single-frequency compromise.
    m_early = window_mean(0.5, 1.5)
    m_late = window_mean(2.5, 3.5)
    assert m_early > m_late > 0.0
    lambda_envelope = np.log(m_early / m_late) / 2.0
    assert lambda_envelope == pytest.approx(values["Lambda"], rel=0.25, abs=0.05)

    # And the absolute scale matches the fitted amplitude where the filter
    # is settled.
    predicted = values["A_1"] * np.exp(-values["Lambda"] * 1.0)
    assert window_mean(0.5, 1.5) == pytest.approx(predicted, rel=0.15)
