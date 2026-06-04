"""Unit tests for the core representation model (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)
from asymmetry.core.representation import (
    DatasetRepresentations,
    FitSeries,
    FitSlot,
    FrequencyFFT,
    FrequencyMaxEnt,
    RepresentationType,
    TimeFBAsymmetry,
    TimeGroups,
    make_representation,
    representation_from_dict,
)
from asymmetry.core.representation.factory import REPRESENTATION_REGISTRY
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.grouping import apply_grouping_aligned


def _run() -> Run:
    return Run(
        run_number=7,
        histograms=[
            Histogram(counts=np.array([100.0, 80.0, 60.0, 40.0, 20.0]), bin_width=0.1, t0_bin=0),
            Histogram(counts=np.array([50.0, 45.0, 40.0, 35.0, 30.0]), bin_width=0.1, t0_bin=0),
        ],
        metadata={"field": 120.0, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Forward", 2: "Backward"},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.1,
            "first_good_bin": 0,
            "last_good_bin": 4,
            "bunching_factor": 1,
        },
    )


# ── RepresentationType / domain ────────────────────────────────────────────


def test_domain_of_each_type():
    assert RepresentationType.TIME_FB_ASYMMETRY.domain == "time"
    assert RepresentationType.TIME_GROUPS.domain == "time"
    assert RepresentationType.FREQ_FFT.domain == "frequency"
    assert RepresentationType.FREQ_MAXENT.domain == "frequency"


def test_registry_covers_all_types():
    assert set(REPRESENTATION_REGISTRY) == set(RepresentationType)


# ── FitSlot ────────────────────────────────────────────────────────────────


def test_fit_slot_round_trip_and_provenance_guard():
    slot = FitSlot(
        model={"component_names": ["Exponential"], "operators": []},
        parameters=[{"name": "A", "value": 1.0}],
        result={"chi_squared": 2.0},
        provenance="batch",
        batch_id="b1",
        diverged=True,
        include_in_trend=False,
    )
    restored = FitSlot.from_dict(slot.to_dict())
    assert restored == slot
    assert not restored.is_empty()
    # Unknown provenance is coerced to "none".
    assert FitSlot.from_dict({"provenance": "bogus"}).provenance == "none"
    assert FitSlot().is_empty()


# ── TimeFBAsymmetry ────────────────────────────────────────────────────────


def test_fb_asymmetry_matches_core_pipeline():
    run = _run()
    rep = TimeFBAsymmetry()
    curves = rep.compute(run)
    assert len(curves) == 1
    ds = curves[0]

    forward = apply_grouping_aligned(run.histograms, [0], common_t0_bin=0)
    backward = apply_grouping_aligned(run.histograms, [1], common_t0_bin=0)
    expected_asym, expected_err = compute_asymmetry(forward, backward, 1.1)

    np.testing.assert_allclose(ds.asymmetry, expected_asym)
    np.testing.assert_allclose(ds.error, expected_err)
    np.testing.assert_allclose(ds.time, np.arange(5) * 0.1)
    assert ds.metadata["plot_domain"] == "time"
    assert ds.run is run


def test_fb_asymmetry_requires_grouping():
    run = Run(run_number=1, histograms=[Histogram(np.array([1.0, 2.0]), 0.1)], grouping={})
    with pytest.raises(ValueError):
        TimeFBAsymmetry().compute(run)


def test_representation_domain_property():
    assert TimeFBAsymmetry().domain == "time"
    assert FrequencyFFT().domain == "frequency"


# ── TimeGroups ─────────────────────────────────────────────────────────────


def test_groups_matches_build_grouped_datasets():
    run = _run()
    rep = TimeGroups()
    curves = rep.compute(run)

    source = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    expected = build_grouped_time_domain_datasets(source)

    assert [c.run_label for c in curves] == [e.run_label for e in expected]
    for c, e in zip(curves, expected, strict=True):
        np.testing.assert_allclose(c.asymmetry, e.asymmetry)
    assert all(c.metadata.get("plot_domain") == "time" for c in curves)


# ── FrequencyFFT ───────────────────────────────────────────────────────────


def test_fft_matches_core_average_pipeline():
    run = _run()
    rep = FrequencyFFT()
    spectra = rep.compute(run)
    # FFT yields a single averaged spectrum across enabled groups.
    assert len(spectra) == 1
    ds = spectra[0]

    expected = compute_average_group_spectrum(run, GroupSpectrumConfig())
    assert expected is not None
    np.testing.assert_allclose(ds.time, expected.time)
    np.testing.assert_allclose(ds.asymmetry, expected.asymmetry)
    assert ds.metadata["plot_domain"] == "frequency"
    assert ds.metadata["fourier_display"] == "(Power)^1/2"
    assert ds.metadata["fourier_group_output"] == "average"


def test_fft_respects_group_enabled_table():
    run = _run()
    rep = FrequencyFFT(recipe={"fourier_config": {"group_enabled_table": {2: False}}})
    spectra = rep.compute(run)
    assert len(spectra) == 1
    # Only group 1 averaged in.
    assert spectra[0].metadata["group_ids"] == [1]


# ── FrequencyMaxEnt ────────────────────────────────────────────────────────


def test_maxent_compute_raises():
    with pytest.raises(NotImplementedError):
        FrequencyMaxEnt().compute(_run())


# ── caching / invalidation ─────────────────────────────────────────────────


def test_ensure_computed_caches_until_invalidated():
    run = _run()
    rep = TimeFBAsymmetry()
    first = rep.ensure_computed(run)
    assert rep.ensure_computed(run) is first
    rep.invalidate()
    assert rep.ensure_computed(run) is not first


# ── persistence (recipe + fit + trend only) ────────────────────────────────


def test_representation_to_dict_excludes_arrays_and_round_trips():
    run = _run()
    rep = make_representation(
        RepresentationType.FREQ_FFT,
        recipe={"fourier_config": {"display": "Cos", "padding": 2}},
    )
    rep.fit = FitSlot(
        model=CompositeModel(["GaussianPeak", "ConstantBackground"]).to_dict(),
        provenance="single",
    )
    rep.trend_state = {"x_key": "field"}
    rep.ensure_computed(run)  # populate transient arrays

    data = rep.to_dict()
    assert set(data) == {"rep_type", "recipe", "fit", "trend_state"}
    assert "datasets" not in data and "_datasets" not in data

    restored = representation_from_dict(data)
    assert isinstance(restored, FrequencyFFT)
    assert restored.recipe == rep.recipe
    assert restored.fit.provenance == "single"
    assert restored.trend_state == {"x_key": "field"}
    assert restored.primary is None  # arrays not restored


# ── DatasetRepresentations container ───────────────────────────────────────


def test_dataset_representations_ensure_and_round_trip():
    container = DatasetRepresentations(run_number=7)
    assert container.get(RepresentationType.TIME_FB_ASYMMETRY) is None
    rep = container.ensure(RepresentationType.TIME_FB_ASYMMETRY)
    assert isinstance(rep, TimeFBAsymmetry)
    # ensure is idempotent
    assert container.ensure(RepresentationType.TIME_FB_ASYMMETRY) is rep

    container.ensure(RepresentationType.FREQ_FFT)
    restored = DatasetRepresentations.from_dict(container.to_dict())
    assert restored.run_number == 7
    assert set(restored.by_type) == {
        RepresentationType.TIME_FB_ASYMMETRY,
        RepresentationType.FREQ_FFT,
    }


def test_make_representation_accepts_string_type():
    rep = make_representation("freq_fft")
    assert isinstance(rep, FrequencyFFT)


def test_fit_series_importable_from_package():
    # Smoke check that the package surface includes FitSeries.
    assert FitSeries("b1", RepresentationType.FREQ_FFT).batch_id == "b1"
