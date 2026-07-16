"""Tests for GUI-free per-axis trend transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.fitting.axis_transforms import (
    CUSTOM,
    IDENTITY,
    LOG,
    RECIPROCAL,
    SQUARE,
    AxisTransform,
    validate_axis_expression,
)


class TestApply:
    def test_identity_passthrough(self):
        t = AxisTransform.identity()
        vals = np.array([1.0, 2.0, np.nan])
        errs = np.array([0.1, 0.2, np.nan])
        out_v, out_e = t.apply(vals, errs)
        np.testing.assert_array_equal(out_v, vals)
        np.testing.assert_array_equal(out_e, errs)

    def test_reciprocal_values_and_error_propagation(self):
        t = AxisTransform.preset(RECIPROCAL)
        # d(1/x) = -1/x^2 dx  ->  sigma_y = sigma_x / x^2
        out_v, out_e = t.apply([2.0, 4.0], [0.1, 0.2])
        np.testing.assert_allclose(out_v, [0.5, 0.25])
        np.testing.assert_allclose(out_e, [0.1 / 4.0, 0.2 / 16.0])

    def test_square_error_propagation(self):
        t = AxisTransform.preset(SQUARE)
        # d(x^2) = 2x dx
        out_v, out_e = t.apply([3.0], [0.1])
        np.testing.assert_allclose(out_v, [9.0])
        np.testing.assert_allclose(out_e, [2 * 3.0 * 0.1])

    def test_log_error_propagation(self):
        t = AxisTransform.preset(LOG)
        # d(ln x) = dx / x
        out_v, out_e = t.apply([10.0], [0.5])
        np.testing.assert_allclose(out_v, [math.log(10.0)])
        np.testing.assert_allclose(out_e, [0.5 / 10.0])

    def test_undefined_maps_to_nan(self):
        # 1/0 and log of a non-positive value are undefined -> NaN, point dropped.
        recip = AxisTransform.preset(RECIPROCAL)
        out_v, out_e = recip.apply([0.0, 2.0], [0.1, 0.1])
        assert math.isnan(out_v[0]) and math.isnan(out_e[0])
        assert out_v[1] == 0.5

        log = AxisTransform.preset(LOG)
        out_v, _ = log.apply([-1.0, math.e], [0.1, 0.1])
        assert math.isnan(out_v[0])
        np.testing.assert_allclose(out_v[1], 1.0)

    def test_missing_uncertainty_yields_nan_error_but_keeps_value(self):
        t = AxisTransform.preset(RECIPROCAL)
        out_v, out_e = t.apply([2.0, 4.0], [np.nan, 0.0])
        np.testing.assert_allclose(out_v, [0.5, 0.25])
        assert math.isnan(out_e[0])  # NaN error in
        assert math.isnan(out_e[1])  # non-positive error in

    def test_errors_default_to_nan_when_omitted(self):
        t = AxisTransform.preset(SQUARE)
        out_v, out_e = t.apply([2.0, 3.0])
        np.testing.assert_allclose(out_v, [4.0, 9.0])
        assert np.all(np.isnan(out_e))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            AxisTransform.preset(SQUARE).apply([1.0, 2.0], [0.1])

    def test_custom_expression_matches_preset(self):
        custom = AxisTransform.custom("1000/x")
        out_v, out_e = custom.apply([250.0], [5.0])
        np.testing.assert_allclose(out_v, [4.0])
        # d(1000/x) = -1000/x^2 dx
        np.testing.assert_allclose(out_e, [1000.0 / 250.0**2 * 5.0])


class TestDescribe:
    @pytest.mark.parametrize(
        "kind, base, expected",
        [
            (IDENTITY, "λ", "λ"),
            (RECIPROCAL, "λ", "1/λ"),
            (SQUARE, "B", "B²"),
            (LOG, "λ", "ln λ"),
        ],
    )
    def test_atomic_symbol_labels(self, kind, base, expected):
        assert AxisTransform.preset(kind).describe(base) == expected

    def test_non_atomic_base_is_bracketed(self):
        assert AxisTransform.preset(RECIPROCAL).describe("B (G)") == "1/(B (G))"
        assert AxisTransform.preset(SQUARE).describe("B (G)") == "(B (G))²"

    def test_custom_splices_base_for_variable(self):
        # The axis variable is replaced; function names containing the letter
        # are left intact.
        assert AxisTransform.custom("1000/x").describe("T") == "1000/T"
        assert AxisTransform.custom("exp(x)").describe("T") == "exp(T)"

    def test_identity_describe_returns_base(self):
        assert AxisTransform.identity().describe("anything") == "anything"


class TestDescribeWithUnit:
    @pytest.mark.parametrize(
        "transform, symbol, unit, expected",
        [
            (AxisTransform.preset(RECIPROCAL), "λ", "µs⁻¹", "1/λ (µs)"),
            (AxisTransform.preset(SQUARE), "B", "G", "B² (G²)"),
            (AxisTransform.preset(LOG), "λ", "µs⁻¹", "ln[λ (µs⁻¹)]"),
            (AxisTransform.preset(RECIPROCAL), "T", "K", "1/T (K⁻¹)"),
            (AxisTransform.custom("1000/x"), "T", "K", "1000/[T (K)]"),
        ],
    )
    def test_reviewer_cases(self, transform, symbol, unit, expected):
        assert transform.describe_with_unit(symbol, unit) == expected

    def test_square_of_inverse_unit(self):
        assert AxisTransform.preset(SQUARE).describe_with_unit("λ", "µs⁻¹") == "λ² (µs⁻²)"

    def test_no_unit_matches_describe(self):
        t = AxisTransform.preset(RECIPROCAL)
        assert t.describe_with_unit("λ") == t.describe("λ")
        assert t.describe_with_unit("λ", "") == t.describe("λ")

    def test_identity_keeps_unit(self):
        assert AxisTransform.identity().describe_with_unit("B", "G") == "B (G)"

    def test_non_simple_unit_brackets(self):
        # A compound unit is not invertible token-wise, so bracket the whole.
        label = AxisTransform.preset(RECIPROCAL).describe_with_unit("D", "cm² s⁻¹")
        assert label == "1/[D (cm² s⁻¹)]"


class TestValidation:
    def test_valid_expression(self):
        ok, msg = validate_axis_expression("1/x + 2")
        assert ok and msg is None

    def test_empty_rejected(self):
        ok, msg = validate_axis_expression("   ")
        assert not ok and msg

    def test_unknown_symbol_rejected(self):
        ok, msg = validate_axis_expression("1/y")
        assert not ok
        assert "y" in msg

    def test_expression_must_reference_variable(self):
        ok, msg = validate_axis_expression("2 + 3")
        assert not ok

    def test_custom_transform_validate_delegates(self):
        assert AxisTransform.custom("x**2").validate() == (True, None)
        ok, _ = AxisTransform.custom("q").validate()
        assert not ok

    def test_preset_always_valid(self):
        assert AxisTransform.preset(SQUARE).validate() == (True, None)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            AxisTransform("bogus")


class TestSerialisation:
    def test_identity_serialises_empty(self):
        assert AxisTransform.identity().to_dict() == {}

    def test_preset_roundtrip(self):
        t = AxisTransform.preset(RECIPROCAL)
        assert AxisTransform.from_dict(t.to_dict()) == t

    def test_custom_roundtrip(self):
        t = AxisTransform.custom("1000/x")
        restored = AxisTransform.from_dict(t.to_dict())
        assert restored == t
        assert restored.kind == CUSTOM
        assert restored.expression == "1000/x"

    def test_from_dict_tolerates_garbage(self):
        assert AxisTransform.from_dict(None).is_identity
        assert AxisTransform.from_dict({}).is_identity
        assert AxisTransform.from_dict({"kind": "nonsense"}).is_identity

    def test_from_dict_none_expression(self):
        # A custom entry with a missing expression degrades gracefully.
        restored = AxisTransform.from_dict({"kind": CUSTOM})
        assert restored.kind == CUSTOM
        assert restored.expression == ""


class TestRedfieldAndArrhenius:
    """The two headline linearisations end-to-end."""

    def test_redfield_linearises_lorentzian_field_dependence(self):
        # lambda(B) = A / (1 + (gamma B)^2) with a T2-like tail -> 1/lambda is
        # linear in B^2 with slope (gamma^2)/A and intercept 1/A.
        a = 5.0
        gamma = 0.5
        b = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
        lam = a / (1.0 + (gamma * b) ** 2)

        inv_lam, _ = AxisTransform.preset(RECIPROCAL).apply(lam)
        b_squared, _ = AxisTransform.preset(SQUARE).apply(b)

        slope, intercept = np.polyfit(b_squared, inv_lam, 1)
        np.testing.assert_allclose(slope, gamma**2 / a)
        np.testing.assert_allclose(intercept, 1.0 / a)

    def test_arrhenius_linearises_activated_rate(self):
        # rate(T) = A exp(-Ea/(k T)) -> ln(rate) linear in 1/T, slope -Ea/k.
        a = 1e3
        ea_over_k = 1500.0  # kelvin
        temp = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
        rate = a * np.exp(-ea_over_k / temp)

        ln_rate, _ = AxisTransform.preset(LOG).apply(rate)
        inv_t, _ = AxisTransform.preset(RECIPROCAL).apply(temp)

        slope, intercept = np.polyfit(inv_t, ln_rate, 1)
        np.testing.assert_allclose(slope, -ea_over_k)
        np.testing.assert_allclose(intercept, math.log(a))
