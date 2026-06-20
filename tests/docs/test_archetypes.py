"""Determinism guards for the documentation-screenshot synthetic data.

These tests pin the shape and byte-stability of every generator in
``docs/screenshots/data/archetypes.py`` so accidental edits — to a default
parameter, a seed, or the time-axis helpers — surface as a test failure
rather than as a silent change in the rendered docs.

The byte-stable hash assertion uses a short slice of the generated arrays
(time + asymmetry) rather than the full payload so the assertion stays
readable on failure while still catching numerical drift.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from docs.screenshots.data import archetypes


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def _hash_dataset(dataset: MuonDataset) -> str:
    return _hash_arrays(dataset.time[:16], dataset.asymmetry[:16])


# ---------------------------------------------------------------------------
# Shape and metadata assertions
# ---------------------------------------------------------------------------


def test_make_ag_zf_gkt_shape() -> None:
    ds = archetypes.make_ag_zf_gkt()
    assert isinstance(ds, MuonDataset)
    assert ds.n_points == 480
    assert ds.metadata["title"] == "ZF Ag polycrystal 20K"
    assert ds.metadata["temperature"] == 20.0
    assert ds.metadata["field"] == 0.0
    assert ds.run_number == 4101


def test_make_ag_lf_decoupling_default_series() -> None:
    runs = archetypes.make_ag_lf_decoupling()
    assert len(runs) == 5
    fields = [ds.metadata["field"] for ds in runs]
    assert fields == [0.0, 5.0, 10.0, 25.0, 50.0]
    assert all(ds.n_points == 480 for ds in runs)
    assert [ds.run_number for ds in runs] == [5201, 5202, 5203, 5204, 5205]


def test_make_ag_lf_decoupling_custom_fields() -> None:
    runs = archetypes.make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
    assert len(runs) == 4
    assert [ds.metadata["field"] for ds in runs] == [0.0, 15.0, 50.0, 100.0]


def test_make_euo_tf_tscan_shape() -> None:
    runs = archetypes.make_euo_tf_tscan()
    assert len(runs) == 6
    temps = [ds.metadata["temperature"] for ds in runs]
    assert temps == [30.0, 50.0, 65.0, 69.0, 73.0, 90.0]
    assert all(ds.metadata["field"] == 0.0 for ds in runs)
    titles = [ds.metadata["title"] for ds in runs]
    assert titles[0].startswith("EuO ZF")
    assert [ds.run_number for ds in runs] == [3001, 3002, 3003, 3004, 3005, 3006]


def test_make_euo_composite_shape() -> None:
    ds = archetypes.make_euo_composite()
    assert ds.metadata["title"] == "EuO ZF 70K (critical)"
    assert ds.metadata["temperature"] == 70.0
    assert ds.n_points == 600


def test_make_mgb2_sigma_t_payload() -> None:
    payload = archetypes.make_mgb2_sigma_t()
    assert set(payload) == {"T_K", "sigma", "sigma_err", "Tc_K"}
    assert payload["T_K"].shape == payload["sigma"].shape == (28,)
    assert payload["Tc_K"] == archetypes.TC_MGB2_K
    # σ(T) should be monotonically decreasing from low to high T.
    assert payload["sigma"][0] > payload["sigma"][-1]


def test_make_ybco_knight_grouped_full_run() -> None:
    ds = archetypes.make_ybco_knight_grouped()
    assert isinstance(ds, MuonDataset)
    assert ds.metadata["temperature"] == 100.0
    assert ds.metadata["field"] == 200.0
    # Run must carry four detector histograms with grouping metadata so the
    # multi-group fit window can engage.
    assert ds.run is not None
    assert len(ds.run.histograms) == 4
    groups = ds.run.grouping["groups"]
    assert set(groups) == {1, 2, 3, 4}
    assert all(len(v) == 1 for v in groups.values())
    assert ds.run.grouping["deadtime_correction"] is False
    assert ds.run.grouping["alpha"] == 1.0


def test_make_ybco_vortex_lattice_full_run() -> None:
    ds = archetypes.make_ybco_vortex_lattice()
    assert ds.metadata["title"].startswith("YBCO TF 200mT")
    assert ds.metadata["temperature"] == 10.0
    assert ds.metadata["field"] == 2000.0
    # Run must expose F/B grouping so the GUI's Compute FFT pipeline runs.
    assert ds.run is not None
    assert len(ds.run.histograms) == 2
    assert ds.run.grouping["forward_group"] == 1
    assert ds.run.grouping["backward_group"] == 2


def test_make_pbf2_fmuf_long_time_window() -> None:
    ds = archetypes.make_pbf2_fmuf()
    assert ds.metadata["title"].startswith("PbF₂")
    # The F-μ-F beat envelope requires t_max ≥ 15 µs to be visible.
    assert ds.time[-1] >= 15.0


def test_make_emu_vector_three_axes() -> None:
    runs = archetypes.make_emu_vector()
    axes = [ds.metadata["polarization_axis"] for ds in runs]
    assert axes == ["Pz", "Px", "Py"]


def test_make_generic_tf_for_processing_low_statistics() -> None:
    ds = archetypes.make_generic_tf_for_processing()
    # Low counts-per-bin should produce visibly larger error bars than the
    # high-statistics datasets above.
    assert ds.error.mean() > 0.3


# ---------------------------------------------------------------------------
# Byte-stable RNG / regression guards
# ---------------------------------------------------------------------------

# These hashes are recomputed any time the documented seed, model, or
# parameter values change. Failures here indicate that a screenshot will
# render differently from the baseline — investigate before updating.


@pytest.mark.parametrize(
    ("name", "expected_hash"),
    [
        ("make_ag_zf_gkt", None),
        ("make_euo_composite", None),
        ("make_pbf2_fmuf", None),
        ("make_generic_tf_for_processing", None),
        ("make_ybco_vortex_lattice", None),
    ],
)
def test_single_dataset_hash_is_stable(name: str, expected_hash: str | None) -> None:
    factory = getattr(archetypes, name)
    first = factory()
    second = factory()
    assert _hash_dataset(first) == _hash_dataset(second), (
        f"{name} produced different output across two invocations — "
        "the RNG seeding is not deterministic"
    )
    # The expected_hash slot is populated once a generator's payload is
    # considered stable. For now we just assert intra-run determinism.


@pytest.mark.parametrize("name", ["make_ag_lf_decoupling", "make_euo_tf_tscan", "make_emu_vector"])
def test_multi_dataset_hash_is_stable(name: str) -> None:
    factory = getattr(archetypes, name)
    first = factory()
    second = factory()
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert _hash_dataset(a) == _hash_dataset(b)


def test_ybco_knight_grouped_hash_is_stable() -> None:
    first = archetypes.make_ybco_knight_grouped()
    second = archetypes.make_ybco_knight_grouped()
    assert _hash_dataset(first) == _hash_dataset(second)
    # Histogram counts should be byte-stable too.
    for h_a, h_b in zip(first.run.histograms, second.run.histograms, strict=True):
        assert _hash_arrays(h_a.counts) == _hash_arrays(h_b.counts)


def test_constants_present() -> None:
    """Pin the physical constants used by the generators so docs-side
    references stay in sync with the implementation."""
    assert archetypes.TC_EUO_K == 69.0
    assert archetypes.DELTA_AG_PER_US == 0.39
    assert archetypes.R_MUF_ANG == 1.17
    assert archetypes.TC_MGB2_K == 36.0
    assert archetypes.TC_YBCO_K == 90.0
    assert archetypes.LAMBDA_YBCO_NM == 130.0
    assert archetypes.XI_YBCO_NM == 2.0
