"""Tests for the RRF fit-layer frequency offset (core/fitting/rrf_offset.py).

Verification-plan item 5: fitting raw data through the offset wrapper must
reproduce the direct lab-frame fit exactly, with fitted frequencies reading
as rotating-frame offsets δν.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.rrf_offset import (
    RRF_ROTATION_COMPONENTS,
    UnsupportedRRFComponentError,
    apply_rrf_offsets,
    rrf_frequency_offsets,
    rrf_offset_model,
)
from asymmetry.core.fourier.units import gauss_to_mhz, mhz_to_gauss

NU_LAB = 30.0  # MHz
NU_FRAME = 29.2  # MHz
LAM = 0.4  # 1/us


def _dataset(seed: int = 11) -> MuonDataset:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 8.0, 0.004)
    truth = 20.0 * np.exp(-LAM * t) * np.cos(2.0 * np.pi * NU_LAB * t)
    err = np.full_like(t, 0.4)
    return MuonDataset(
        time=t,
        asymmetry=truth + rng.normal(0.0, 0.4, t.size),
        error=err,
        metadata={},
    )


def _params(frequency_value: float, frequency_name: str = "frequency") -> ParameterSet:
    params = ParameterSet()
    params.add(Parameter(name="A_1", value=18.0, min=0.0, max=50.0))
    params.add(Parameter(name="Lambda", value=0.3, min=0.0, max=10.0))
    params.add(Parameter(name=frequency_name, value=frequency_value, min=-500.0, max=5000.0))
    params.add(Parameter(name="phase", value=0.0, min=-3.2, max=3.2))
    return params


class TestOffsets:
    def test_offsets_resolve_unique_names(self):
        model = CompositeModel.from_expression("Oscillatory * Exponential")
        offsets = rrf_frequency_offsets(model, 25.0)
        assert offsets == {"frequency": 25.0}

    def test_field_component_offsets_in_gauss(self):
        model = CompositeModel.from_expression("OscillatoryField * Exponential")
        offsets = rrf_frequency_offsets(model, 25.0)
        assert set(offsets) == {"field"}
        assert offsets["field"] == pytest.approx(float(mhz_to_gauss(25.0)))
        # Round-trip: the Gauss offset is exactly nu0 in frequency terms.
        assert float(gauss_to_mhz(offsets["field"])) == pytest.approx(25.0)

    def test_unsupported_oscillating_component_raises(self):
        model = CompositeModel.from_expression("MuoniumTF + Exponential")
        with pytest.raises(UnsupportedRRFComponentError, match="MuoniumTF"):
            rrf_frequency_offsets(model, 25.0)

    def test_bessel_is_not_a_rotation(self):
        assert "Bessel" not in RRF_ROTATION_COMPONENTS
        model = CompositeModel.from_expression("Bessel * Exponential")
        with pytest.raises(UnsupportedRRFComponentError, match="Bessel"):
            rrf_frequency_offsets(model, 25.0)

    def test_no_rotation_component_raises(self):
        model = CompositeModel.from_expression("Exponential")
        with pytest.raises(UnsupportedRRFComponentError):
            rrf_frequency_offsets(model, 25.0)

    def test_invalid_frequency_raises(self):
        model = CompositeModel.from_expression("Oscillatory * Exponential")
        with pytest.raises(ValueError, match="frequency_mhz"):
            rrf_frequency_offsets(model, -1.0)

    def test_wrapper_rejects_missing_parameter(self):
        model = CompositeModel.from_expression("Oscillatory * Exponential")
        wrapped = rrf_offset_model(model, 25.0)
        with pytest.raises(ValueError, match="frequency"):
            wrapped(np.linspace(0, 1, 8), A_1=1.0, Lambda=0.5, phase=0.0)

    def test_apply_rrf_offsets_reports_lab_frame(self):
        lab = apply_rrf_offsets({"frequency": 0.8, "A_1": 20.0}, {"frequency": 29.2})
        assert lab == {"frequency": 30.0, "A_1": 20.0}


class TestFitEquivalence:
    """Direct lab-frame fit == offset-wrapper fit with δν semantics."""

    def test_oscillatory_route(self):
        dataset = _dataset()
        model = CompositeModel.from_expression("Oscillatory * Exponential")
        engine = FitEngine()

        direct = engine.fit(dataset, model.function, _params(NU_LAB + 0.05))
        assert direct.success

        wrapped = rrf_offset_model(model, NU_FRAME)
        rotating = engine.fit(dataset, wrapped, _params(NU_LAB + 0.05 - NU_FRAME))
        assert rotating.success

        assert rotating.chi_squared == pytest.approx(direct.chi_squared, rel=1e-6)
        lab = apply_rrf_offsets({p.name: p.value for p in rotating.parameters}, wrapped.rrf_offsets)
        direct_values = {p.name: p.value for p in direct.parameters}
        assert lab["frequency"] == pytest.approx(direct_values["frequency"], abs=1e-6)
        assert rotating.parameters["frequency"].value == pytest.approx(NU_LAB - NU_FRAME, abs=1e-3)
        for name in ("A_1", "Lambda", "phase"):
            assert lab[name] == pytest.approx(direct_values[name], abs=1e-5)

    def test_oscillatory_field_route(self):
        dataset = _dataset(seed=12)
        model = CompositeModel.from_expression("OscillatoryField * Exponential")
        engine = FitEngine()

        field_lab = float(mhz_to_gauss(NU_LAB))
        direct = engine.fit(dataset, model.function, _params(field_lab + 0.5, "field"))
        assert direct.success

        wrapped = rrf_offset_model(model, NU_FRAME)
        delta_seed = field_lab + 0.5 - float(mhz_to_gauss(NU_FRAME))
        rotating = engine.fit(dataset, wrapped, _params(delta_seed, "field"))
        assert rotating.success

        assert rotating.chi_squared == pytest.approx(direct.chi_squared, rel=1e-6)
        lab = apply_rrf_offsets({p.name: p.value for p in rotating.parameters}, wrapped.rrf_offsets)
        assert lab["field"] == pytest.approx(
            {p.name: p.value for p in direct.parameters}["field"], abs=1e-4
        )
