"""The exact rotating-frame projection: conventions, errors, record and estimators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.projections import reduce_run_projections
from asymmetry.core.transform.rotating_frame import (
    FRAME_FIELDS,
    B1Axis,
    PeriodBaseline,
    Provenance,
    RotatingFrame,
    estimate_frame,
    period_weights,
    rotate_transverse,
)
from asymmetry.core.utils.constants import PeriodMode
from tests.core.vector_synthetic import synthetic_vector_run

NU, W1, PHI = 1.49, 2.0 * math.pi * 0.12, 35.0
#: Polarisation amplitudes (fraction) of P_x, P_y, P_z and the transverse baselines (%):
#: the driven period's, then a second period's that differ from them.
AMPLITUDES, BASELINES, OTHER_BASELINES = (0.10, 0.09, 0.20), (0.4, -0.3), (0.35, -0.22)


def _offsets(baselines):
    """Polarisation offsets whose asymmetry is the given baselines (%)."""
    return [baselines[0] / (100 * AMPLITUDES[0]), baselines[1] / (100 * AMPLITUDES[1])]


def _nutation(axis: B1Axis, sense: int):
    """Measured (P_x, P_y, P_z) of an on-resonance nutation about B₁ (rotating-wave limit).

    In the frame, a spin leaving +z turns about B₁ into ẑ × B̂₁: +y′ for x, −x′
    for y. The lab frame turns it back with e^{−i(2πνt + φ)} (negative
    precession), and the labels see the physical y as −s·P_y.
    """

    def polarisation(t):
        transverse = np.sin(W1 * t) * (1j if axis is B1Axis.X else -1.0)
        lab = transverse * np.exp(-1j * (2.0 * math.pi * NU * t + math.radians(PHI)))
        offsets = _offsets(BASELINES)
        return np.stack([lab.real + offsets[0], -sense * lab.imag + offsets[1], np.cos(W1 * t)])

    return polarisation


def _pair(axis: B1Axis, sense: int, *, seed: int | None, run_number: int = 900):
    run = synthetic_vector_run(
        [_nutation(axis, sense)],
        amplitudes=AMPLITUDES,
        rate=4.0e5,
        n_bins=600,
        run_number=run_number,
        seed=seed,
    )
    reduced = reduce_run_projections(run, 0)
    return reduced["P_x"], reduced["P_y"]


def _frame(axis: B1Axis, sense: int, periods: int = 1) -> RotatingFrame:
    frame = RotatingFrame.typed_frequency(NU, periods).with_values(
        Provenance.TYPED,
        b1_axis=axis,
        rf_phase_deg=PHI,
        sense=sense,
        gain=AMPLITUDES[1] / AMPLITUDES[0],
    )
    for period, baselines in zip(range(periods), (BASELINES, OTHER_BASELINES), strict=False):
        frame = frame.with_baseline(period, Provenance.TYPED, *baselines)
    return frame


@pytest.mark.parametrize("sense", [-1, 1])
@pytest.mark.parametrize("axis", [B1Axis.X, B1Axis.Y])
def test_a_nutation_lands_on_the_axis_perpendicular_to_b1_for_either_sense(axis, sense):
    px, py = _pair(axis, sense, seed=None)
    rx, ry = rotate_transverse(px, py, _frame(axis, sense), weights=(1.0,))
    expected = 100 * AMPLITUDES[0] * np.sin(W1 * px.time)
    along, across = (rx, ry) if axis is B1Axis.X else (ry, rx)
    sign = 1.0 if axis is B1Axis.X else -1.0
    np.testing.assert_allclose(across.asymmetry, sign * expected, atol=0.02)
    np.testing.assert_allclose(along.asymmetry, 0.0, atol=0.02)
    assert (rx.metadata["projection"], ry.metadata["projection"]) == ("P′_x", "P′_y")


def test_rotated_errors_match_monte_carlo():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 2.0, 5)
    sx, sy = np.full(5, 0.2), np.full(5, 0.5)
    frame = (
        _frame(B1Axis.X, -1)
        .with_values(Provenance.TYPED, gain=1.3)
        .with_baseline(0, Provenance.TYPED, 0.0, 0.0)
    )
    draws = [
        rotate_transverse(
            MuonDataset(time=t, asymmetry=rng.normal(0.0, sx), error=sx),
            MuonDataset(time=t, asymmetry=rng.normal(0.0, sy), error=sy),
            frame,
            weights=(1.0,),
        )
        for _ in range(20000)
    ]
    first = draws[0]
    for index in (0, 1):
        values = np.array([pair[index].asymmetry for pair in draws])
        np.testing.assert_allclose(values.std(axis=0), first[index].error, rtol=0.03)


def test_the_pair_must_share_one_time_axis():
    px, py = _pair(B1Axis.X, -1, seed=None)
    shifted = MuonDataset(time=py.time + 0.01, asymmetry=py.asymmetry, error=py.error)
    with pytest.raises(ValueError, match="one time axis"):
        rotate_transverse(px, shifted, _frame(B1Axis.X, -1), weights=(1.0,))


def test_the_frame_records_where_each_value_came_from():
    frame = RotatingFrame.typed_frequency(1.5, 2)
    assert frame.provenance["frequency_mhz"] is Provenance.TYPED
    assert {frame.provenance[name] for name in FRAME_FIELDS[1:]} == {Provenance.DEFAULT}
    assert frame.baselines == (PeriodBaseline(0.0, 0.0, Provenance.DEFAULT),) * 2
    estimated = frame.with_values(Provenance.ESTIMATED, rf_phase_deg=40.0, gain=0.95)
    estimated = estimated.with_baseline(1, Provenance.ESTIMATED, 0.2, -0.1)
    assert estimated.provenance["rf_phase_deg"] is Provenance.ESTIMATED
    assert estimated.provenance["sense"] is Provenance.DEFAULT
    assert estimated.baselines[1] == PeriodBaseline(0.2, -0.1, Provenance.ESTIMATED)
    assert estimated.baselines[0].provenance is Provenance.DEFAULT
    assert RotatingFrame.from_dict(estimated.to_dict()) == estimated


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"frequency_mhz": 0.0}, "ν_RF"),
        ({"sense": 0}, "sense"),
        ({"gain": -1.0}, "Gain"),
        ({"rf_phase_deg": float("nan")}, "φ_RF"),
    ],
)
def test_bad_values_are_refused_at_construction(change, message):
    with pytest.raises(ValueError, match=message):
        RotatingFrame.typed_frequency(1.5, 1).with_values(Provenance.TYPED, **change)
    with pytest.raises(ValueError, match="Unknown"):
        RotatingFrame.typed_frequency(1.5, 1).with_values(Provenance.TYPED, frequency=2.0)


@pytest.mark.parametrize("sense", [-1, 1])
@pytest.mark.parametrize("axis", [B1Axis.X, B1Axis.Y])
def test_auto_detect_recovers_the_frame_from_noisy_runs(axis, sense):
    runs = [(n, [_pair(axis, sense, seed=n, run_number=n)]) for n in (901, 902)]
    estimate = estimate_frame(runs, frequency_mhz=NU, b1_axis=axis)
    assert estimate.sense == sense
    assert abs((estimate.rf_phase_deg - PHI + 180.0) % 360.0 - 180.0) < 4.0
    assert estimate.contrast > 10.0
    for run in estimate.runs:
        assert run.gain == pytest.approx(AMPLITUDES[1] / AMPLITUDES[0], rel=0.05)
        ((bx, by),) = run.baselines
        assert bx == pytest.approx(BASELINES[0], abs=0.05)
        assert by == pytest.approx(BASELINES[1], abs=0.05)


def test_auto_detect_reports_low_contrast_without_a_transverse_signal():
    def still(t):
        return np.stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])

    run = synthetic_vector_run([still], rate=4.0e5, n_bins=600, seed=5)
    reduced = reduce_run_projections(run, 0)
    estimate = estimate_frame(
        [(900, [(reduced["P_x"], reduced["P_y"])])], frequency_mhz=NU, b1_axis=B1Axis.X
    )
    assert estimate.contrast < 3.0


def _still_with_other_baselines(t):
    offsets = _offsets(OTHER_BASELINES)
    return np.stack([np.full_like(t, offsets[0]), np.full_like(t, offsets[1]), np.ones_like(t)])


def test_baselines_are_fitted_beside_the_signal_not_pulled_by_its_early_turns():
    # Noise-free counts: the bin errors still weight the early, strong turns most,
    # which biased a plain inverse-variance mean; a mean over whole turns still
    # leaked a nutation's ν ± ν₁ sidebands.
    px, py = _pair(B1Axis.X, -1, seed=None)
    run = estimate_frame([(900, [(px, py)])], frequency_mhz=NU, b1_axis=B1Axis.X).runs[0]
    ((bx, by),) = run.baselines
    assert bx == pytest.approx(BASELINES[0], abs=0.005)
    assert by == pytest.approx(BASELINES[1], abs=0.005)


def _two_period_run(seed):
    return synthetic_vector_run(
        [_nutation(B1Axis.X, -1), _still_with_other_baselines],
        amplitudes=AMPLITUDES,
        rate=4.0e5,
        n_bins=600,
        seed=seed,
    )


def test_each_period_gets_its_own_baselines():
    periods = [reduce_run_projections(_two_period_run(21), index) for index in (0, 1)]
    estimate = estimate_frame(
        [(900, [(p["P_x"], p["P_y"]) for p in periods])], frequency_mhz=NU, b1_axis=B1Axis.X
    )
    assert estimate.sense == -1
    assert abs((estimate.rf_phase_deg - PHI + 180.0) % 360.0 - 180.0) < 4.0
    for (bx, by), truth in zip(
        estimate.runs[0].baselines, (BASELINES, OTHER_BASELINES), strict=True
    ):
        assert bx == pytest.approx(truth[0], abs=0.05)
        assert by == pytest.approx(truth[1], abs=0.05)


def test_a_period_combination_removes_the_same_combination_of_baselines():
    assert {str(mode) for mode in PeriodMode} == {
        "red",
        "green",
        "green_minus_red",
        "green_plus_red",
    }
    assert period_weights(str(PeriodMode.GREEN_MINUS_RED), 2) == (-1.0, 1.0)
    assert period_weights(str(PeriodMode.GREEN), 1) == (1.0,)
    run = _two_period_run(None)
    red, green = (reduce_run_projections(run, index) for index in (0, 1))
    frame = _frame(B1Axis.X, -1, periods=2)
    for mode, sign in ((PeriodMode.GREEN_MINUS_RED, -1.0), (PeriodMode.GREEN_PLUS_RED, 1.0)):
        combined = [
            MuonDataset(
                time=red[label].time,
                asymmetry=green[label].asymmetry + sign * red[label].asymmetry,
                error=np.hypot(green[label].error, red[label].error),
            )
            for label in ("P_x", "P_y")
        ]
        got = rotate_transverse(*combined, frame, weights=period_weights(str(mode), 2))
        alone = [
            rotate_transverse(p["P_x"], p["P_y"], frame, weights=w)
            for p, w in ((red, (1.0, 0.0)), (green, (0.0, 1.0)))
        ]
        for index in (0, 1):
            np.testing.assert_allclose(
                got[index].asymmetry,
                alone[1][index].asymmetry + sign * alone[0][index].asymmetry,
                atol=1e-9,
            )
    with pytest.raises(ValueError, match="period weights"):
        rotate_transverse(red["P_x"], red["P_y"], frame, weights=(1.0,))


def test_a_window_shorter_than_two_turns_is_refused():
    px, py = _pair(B1Axis.X, -1, seed=None)
    short = [d.time_range(0.0, 1.0 / NU) for d in (px, py)]
    with pytest.raises(ValueError, match="two turns"):
        estimate_frame([(900, [tuple(short)])], frequency_mhz=NU, b1_axis=B1Axis.X)
