"""Tests for the domain-filtered fit-function library."""

from __future__ import annotations

import pytest

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.domain_library import (
    DOMAINS,
    components_for_domain,
    default_model_for_domain,
    models_for_domain,
)
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.spectral import FREQUENCY_COMPONENT_NAMES


def test_domains_are_time_and_frequency():
    assert DOMAINS == ("time", "frequency")


def test_frequency_components_match_canonical_list():
    freq = components_for_domain("frequency")
    assert set(freq) == set(FREQUENCY_COMPONENT_NAMES)


def test_time_components_exclude_frequency_peaks():
    time = components_for_domain("time")
    for name in FREQUENCY_COMPONENT_NAMES:
        assert name not in time
    # Sanity: a core time-domain component is present.
    assert "Exponential" in time


def test_components_partition_by_domain():
    time = components_for_domain("time")
    freq = components_for_domain("frequency")
    assert set(time).isdisjoint(freq)
    assert set(time) | set(freq) == set(COMPONENTS)


def test_all_builtin_models_are_time_domain():
    assert set(models_for_domain("time")) == set(MODELS)
    assert models_for_domain("frequency") == {}


def test_default_model_for_time_is_exponential_plus_constant():
    model = default_model_for_domain("time")
    assert model.component_names == ["Exponential", "Constant"]


def test_default_model_for_frequency_is_peak_plus_background():
    model = default_model_for_domain("frequency")
    assert model.component_names == ["GaussianPeak", "ConstantBackground"]


def test_unknown_domain_raises():
    with pytest.raises(ValueError):
        components_for_domain("spatial")
    with pytest.raises(ValueError):
        default_model_for_domain("")
