"""Core reduction of one period of a run to its declared projections."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.transform.projections import projection_alphas, reduce_run_projections
from tests.core.vector_synthetic import synthetic_vector_run

AMPLITUDES = (0.12, 0.1, 0.2)


def _precessing(nu: float, sense: int):
    def polarisation(t):
        angle = sense * 2.0 * math.pi * nu * t
        return np.stack([np.cos(angle), np.sin(angle), 0.5 * np.ones_like(t)])

    return polarisation


def _static(t):
    return np.stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])


def test_each_projection_is_its_pair_reduced_on_the_chosen_period():
    run = synthetic_vector_run([_precessing(1.5, -1), _static], amplitudes=AMPLITUDES, seed=None)
    for index, polarisation in enumerate((_precessing(1.5, -1), _static)):
        reduced = reduce_run_projections(run, index)
        assert list(reduced) == ["P_x", "P_y", "P_z"]
        for row, (label, dataset) in enumerate(reduced.items()):
            expected = 100.0 * AMPLITUDES[row] * polarisation(dataset.time)[row]
            np.testing.assert_allclose(dataset.asymmetry, expected, atol=1e-9)
            assert dataset.metadata["projection"] == label
            assert dataset.metadata["period_index"] == index


def test_canonical_axes_reduce_with_their_own_alpha():
    run = synthetic_vector_run([_static], alphas=(1.0, 1.0, 1.25), seed=None)
    assert projection_alphas(run.grouping) == {"P_x": 1.0, "P_y": 1.0, "P_z": 1.25}
    pz = reduce_run_projections(run, 0)["P_z"]
    forward, backward = 1.0 + AMPLITUDES[2], 1.0 - AMPLITUDES[2]
    assert pz.asymmetry[0] == pytest.approx(
        100.0 * (forward - 1.25 * backward) / (forward + 1.25 * backward)
    )


def test_a_period_outside_the_run_or_a_grouping_without_projections_is_refused():
    run = synthetic_vector_run([_static], seed=None)
    with pytest.raises(ValueError, match="1 period"):
        reduce_run_projections(run, 1)
    run.grouping.pop("projections")
    run.grouping["group_names"] = {}
    with pytest.raises(ValueError, match="no asymmetry projections"):
        reduce_run_projections(run, 0)
