"""Tests for core/negmu/model.py (WP1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.negmu.model import (
    CaptureComponent,
    build_capture_count_model,
    evaluate_capture_model,
)


@pytest.fixture
def two_comp():
    return [
        CaptureComponent(label="C", tau_us=2.030),
        CaptureComponent(label="Fe", tau_us=0.206),
    ]


def test_model_matches_numpy(two_comp):
    t = np.linspace(0.0, 10.0, 200)
    amp_C, amp_Fe, bg = 5000.0, 3000.0, 50.0
    params = {"amp_C": amp_C, "amp_Fe": amp_Fe, "background": bg}
    model_fn = build_capture_count_model(two_comp)
    result = model_fn(t, **params)

    expected = amp_C * np.exp(-t / 2.030) + amp_Fe * np.exp(-t / 0.206) + bg
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_tau_override(two_comp):
    t = np.linspace(0.0, 5.0, 100)
    amp_C = 4000.0
    # Override tau_C to a different value
    params_default = {"amp_C": amp_C, "amp_Fe": 0.0, "background": 0.0}
    params_override = {"amp_C": amp_C, "amp_Fe": 0.0, "background": 0.0, "tau_C": 1.0}
    model_fn = build_capture_count_model(two_comp)
    y_default = model_fn(t, **params_default)
    y_override = model_fn(t, **params_override)
    # Overriding tau changes the curve at t > 0
    assert not np.allclose(y_default[1:], y_override[1:])
    # At t=0 both are amp_C
    assert y_default[0] == pytest.approx(amp_C, rel=1e-10)
    assert y_override[0] == pytest.approx(amp_C, rel=1e-10)


def test_unknown_params_ignored(two_comp):
    t = np.array([0.0, 1.0, 2.0])
    model_fn = build_capture_count_model(two_comp)
    result = model_fn(t, amp_C=100.0, amp_Fe=50.0, background=5.0, unknown_param=99.9)
    expected = 100.0 * np.exp(-t / 2.030) + 50.0 * np.exp(-t / 0.206) + 5.0
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_zero_amplitude():
    comps = [CaptureComponent(label="C", tau_us=2.030)]
    model_fn = build_capture_count_model(comps)
    t = np.array([0.0, 1.0])
    result = model_fn(t, amp_C=0.0, background=10.0)
    np.testing.assert_array_equal(result, [10.0, 10.0])


def test_background_default_zero(two_comp):
    """When background param is absent it defaults to zero."""
    t = np.array([0.0])
    model_fn = build_capture_count_model(two_comp)
    # Omit background entirely
    result = model_fn(t, amp_C=100.0, amp_Fe=50.0)
    assert result[0] == pytest.approx(150.0, rel=1e-10)


def test_evaluate_capture_model_convenience(two_comp):
    t = np.linspace(0.0, 5.0, 50)
    params = {"amp_C": 2000.0, "amp_Fe": 1000.0, "background": 20.0}
    result = evaluate_capture_model(two_comp, params, t)
    model_fn = build_capture_count_model(two_comp)
    expected = model_fn(t, **params)
    np.testing.assert_array_equal(result, expected)


def test_single_component():
    comps = [CaptureComponent(label="Pb", tau_us=0.0747)]
    t = np.array([0.0, 0.0747, 2 * 0.0747])
    model_fn = build_capture_count_model(comps)
    amp = 1000.0
    result = model_fn(t, amp_Pb=amp, background=0.0)
    expected = amp * np.exp(-t / 0.0747)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_no_qt_import():
    import sys

    had_gui = "asymmetry.gui" in sys.modules
    import asymmetry.core.negmu.model  # noqa: F401

    added_gui = (not had_gui) and ("asymmetry.gui" in sys.modules)
    assert not added_gui, "importing negmu.model triggered asymmetry.gui load"
