"""Independent rational count-loss limits for fitted detector deadtime."""

import numpy as np
import pytest

from asymmetry.core.fitting.count_domain import _apply_deadtime
from asymmetry.core.transform.deadtime import apply_deadtime_correction


def test_forward_inverts_reduction_scalar():
    measured = _apply_deadtime(
        np.array([100.0]),
        0.2,
        frame_norm=100.0,
        detector_counts=np.array([[250 / 3]]),
    )
    np.testing.assert_allclose(measured, [250 / 3], rtol=1e-12)
    restored = apply_deadtime_correction(measured, 0.2, 1.0, num_good_frames=100.0)
    np.testing.assert_allclose(restored, [100.0], rtol=1e-12)


@pytest.mark.parametrize("occupancy", [1e-4, 0.1, 0.5, 0.9, 4.0, -0.05, -0.5])
def test_forward_inverse_over_occupancy(occupancy):
    expected = 100.0 / (1.0 + occupancy)
    measured = _apply_deadtime(
        np.array([100.0]),
        occupancy,
        frame_norm=100.0,
        detector_counts=np.array([[expected]]),
    )
    np.testing.assert_allclose(measured, [expected], rtol=1e-12)
    restored = apply_deadtime_correction(measured, occupancy, 1.0, num_good_frames=100.0)
    np.testing.assert_allclose(restored, [100.0], rtol=1e-12)


def test_zero_coefficient_is_identity():
    counts = np.array([0.0, 123.5, 1e6])
    assert _apply_deadtime(counts, 0.0, frame_norm=100.0, detector_counts=None) is counts


def test_forward_composes_additively():
    first = _apply_deadtime(
        np.array([100.0]), 0.1, frame_norm=100.0, detector_counts=np.array([[100 / 1.1]])
    )
    result = _apply_deadtime(
        first, 0.05, frame_norm=100.0, detector_counts=np.array([[100 / 1.15]])
    )
    np.testing.assert_allclose(result, [86.95652173913044], rtol=1e-12)


@pytest.mark.parametrize("cost", ["gaussian", "poisson"])
def test_trial_beyond_forward_pole_is_infinite_cost(cost):
    from asymmetry.core.fitting.count_domain import _cost_value

    model = _apply_deadtime(
        np.array([100.0]), -1.1, frame_norm=100.0, detector_counts=np.array([[20.0]])
    )
    from asymmetry.core.fitting.likelihood import CountSource, GaussianObservations, PoissonCounts

    observations = (
        GaussianObservations([0.0], [20.0], [np.sqrt(20)])
        if cost == "gaussian"
        else PoissonCounts([0.0], [20.0], CountSource.RAW)
    )
    assert _cost_value(observations, model) == float("inf")


def _polarisation(time, A=20.0):  # noqa: N803
    return A * np.cos(2 * np.pi * time)


def _rational_dataset(shares_f, shares_b, *, occupancy=0.225, bunch=1, seed=None, tau=0.02):
    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
    from asymmetry.core.utils.constants import MUON_LIFETIME_US

    width, frames = 0.004, 1e6
    time = np.arange(2500) * width
    n0 = occupancy * width * frames / abs(tau)
    histograms = []
    rng = np.random.default_rng(seed)
    for shares, sign in [(shares_f, 1), (shares_b, -1)]:
        group = n0 * np.exp(-time / MUON_LIFETIME_US) * (1 + sign * _polarisation(time) / 100)
        for share in shares:
            true = share * group
            measured = true / (1 + tau * true / (width * frames))
            if seed is not None:
                measured = rng.poisson(measured).astype(float)
            histograms.append(Histogram(counts=measured, bin_width=width))
    run = Run(
        histograms=histograms,
        grouping={
            "groups": {
                1: list(range(1, len(shares_f) + 1)),
                2: list(range(len(shares_f) + 1, len(histograms) + 1)),
            },
            "good_frames": frames,
            "bunching_factor": bunch,
        },
    )
    return MuonDataset(time=np.array([]), asymmetry=np.array([]), error=np.array([]), run=run), n0


def _fit_deadtime(dataset, n0, *, fb=False, cost="gaussian"):
    from asymmetry.core.fitting.count_domain import fit_fb_alpha, fit_single_histogram
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    params = ParameterSet(
        [
            Parameter("N0", n0, min=0),
            Parameter("background", 0, fixed=True),
            Parameter("A", 20, fixed=True),
            Parameter("DT0", 0.015),
        ]
    )
    if fb:
        params.add(Parameter("alpha", 1.0, min=0.1, max=10))
        params.add(Parameter("background_b", 0, fixed=True))
        return fit_fb_alpha(dataset, 1, 2, _polarisation, params, cost=cost)
    return fit_single_histogram(dataset, 1, _polarisation, params, cost=cost)


@pytest.mark.parametrize("occupancy", [0.075, 0.225])
@pytest.mark.parametrize("shares", [(0.5, 0.5), (0.8, 0.2), (0.9, 0.1)])
def test_dt0_recovered_rational_data(occupancy, shares):
    ds, n0 = _rational_dataset(shares, (0.5, 0.5), occupancy=occupancy)
    result = _fit_deadtime(ds, n0)
    assert result.success
    assert result.parameters["DT0"].value == pytest.approx(0.02, rel=1e-4)


def test_dt0_unequal_forward_backward_shares():
    ds, n0 = _rational_dataset((0.7, 0.3), (0.5, 0.5))
    result = _fit_deadtime(ds, n0, fb=True)
    assert result.success
    assert result.shared_parameters["DT0"].value == pytest.approx(0.02, rel=1e-4)


@pytest.mark.parametrize("bunch", [1, 2, 4])
def test_dt0_bunched(bunch):
    ds, n0 = _rational_dataset((0.8, 0.2), (0.5, 0.5), bunch=bunch)
    result = _fit_deadtime(ds, n0 * bunch)
    assert result.success
    assert result.parameters["DT0"].value == pytest.approx(0.02, rel=2e-3)


@pytest.mark.parametrize("frames", [None, 0, -1, float("nan"), True])
def test_dt0_requires_good_frames(frames):
    ds, n0 = _rational_dataset((0.8, 0.2), (0.5, 0.5))
    ds.run.grouping["good_frames"] = frames
    with pytest.raises(ValueError, match="good frames must be finite"):
        _fit_deadtime(ds, n0)


def test_seeded_poisson_recovery():
    ds, n0 = _rational_dataset((0.8, 0.2), (0.5, 0.5), seed=20261101)
    result = _fit_deadtime(ds, n0, cost="poisson")
    assert result.success
    assert abs(result.parameters["DT0"].value - 0.02) < 3 * result.uncertainties["DT0"]


def test_short_trace_records_actual_bin_width():
    from asymmetry.core.fitting.grouped_time_domain import build_count_group

    ds, _ = _rational_dataset((0.8, 0.2), (0.5, 0.5), bunch=5000)
    group = build_count_group(ds, 1, lifetime_corrected=False)
    assert len(group.time) == 2500
    assert group.metadata["grouped_time_domain_bunch_factor"] == 1


def test_detector_traces_share_alignment_crop_and_fit_mask():
    from asymmetry.core.fitting.grouped_time_domain import build_count_group
    from asymmetry.core.transform.grouping import apply_grouping_aligned

    ds, _ = _rational_dataset((0.8, 0.2), (0.5, 0.5), bunch=4)
    ds.run.histograms[0].t0_bin = 2
    ds.run.histograms[1].counts = ds.run.histograms[1].counts[:-10]
    group = build_count_group(ds, 1, lifetime_corrected=False, t_min=0.2, t_max=8, exclude=(2, 3))
    np.testing.assert_allclose(group.detector_counts.sum(axis=0), group.counts)
    assert np.all((group.time < 2) | (group.time > 3))
    aligned = apply_grouping_aligned(ds.run.histograms, [0, 1], common_t0_bin=2)
    expected = aligned[: (len(aligned) // 4) * 4].reshape(-1, 4).sum(axis=1)
    time = (np.arange(len(aligned)) - 2) * 0.004
    time = time[: len(expected) * 4].reshape(-1, 4).mean(axis=1)
    mask = (time >= 0.2) & (time <= 8) & ((time < 2) | (time > 3))
    np.testing.assert_allclose(group.counts, expected[mask])


def test_signed_dt0_recovery():
    ds, n0 = _rational_dataset((0.8, 0.2), (0.5, 0.5), tau=-0.02)
    result = _fit_deadtime(ds, n0)
    assert result.success
    assert result.parameters["DT0"].value == pytest.approx(-0.02, rel=1e-4)


@pytest.mark.parametrize("width", [0, -1, float("inf"), float("nan")])
def test_dt0_requires_valid_bin_width(width):
    from asymmetry.core.fitting.count_domain import _deadtime_frame_norm
    from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup

    ds, _ = _rational_dataset((0.8, 0.2), (0.5, 0.5))
    ds.run.histograms[0].bin_width = width
    group = GroupedTimeDomainGroup(
        1,
        "Forward",
        np.array([]),
        np.array([]),
        np.array([]),
        metadata={"grouped_time_domain_bunch_factor": 1},
    )
    with pytest.raises(ValueError, match="bin width.*must be finite and > 0"):
        _deadtime_frame_norm(ds, group)
