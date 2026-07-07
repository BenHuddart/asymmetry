"""Tests for the core perf-timing helper (``asymmetry.core.utils.perf``).

Covers the enabled/disabled toggle (explicit override vs. env-var fallback),
the logged message shape (static detail before late ``.detail()`` fields),
exception propagation, and one integration check that a real hot-path
function (``reduce_grouped_asymmetry``) emits a PERF record when enabled.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.core.utils import perf as perf_module
from asymmetry.core.utils.perf import perf_logging_enabled, perf_timer, set_perf_logging

_ENV_VAR = "ASYMMETRY_PERF_LOGGING"


@pytest.fixture(autouse=True)
def _reset_perf_state():
    """Isolate each test from any override/env-var state the others set."""
    previous_override = perf_module._override
    previous_env = os.environ.pop(_ENV_VAR, None)
    set_perf_logging(None)
    yield
    set_perf_logging(previous_override)
    if previous_env is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = previous_env


def test_disabled_by_default_emits_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    with perf_timer("test.event"):
        pass

    assert caplog.records == []


@pytest.mark.parametrize("value", ["1", "true", "YES", "Yes"])
def test_env_var_truthy_values_enable_logging(value: str) -> None:
    os.environ[_ENV_VAR] = value
    assert perf_logging_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "bogus"])
def test_env_var_falsy_or_unrecognised_values_disable_logging(value: str) -> None:
    os.environ[_ENV_VAR] = value
    assert perf_logging_enabled() is False


def test_explicit_override_wins_over_env_var() -> None:
    os.environ[_ENV_VAR] = "1"
    set_perf_logging(False)
    assert perf_logging_enabled() is False

    os.environ[_ENV_VAR] = "0"
    set_perf_logging(True)
    assert perf_logging_enabled() is True


def test_override_none_resets_to_env_var_fallback() -> None:
    set_perf_logging(True)
    assert perf_logging_enabled() is True

    set_perf_logging(None)
    os.environ[_ENV_VAR] = "1"
    assert perf_logging_enabled() is True

    os.environ[_ENV_VAR] = "0"
    assert perf_logging_enabled() is False


def test_enabled_logs_expected_message_shape(caplog: pytest.LogCaptureFixture) -> None:
    set_perf_logging(True)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    with perf_timer("test.shape", static_a=1, static_b="x") as recorder:
        recorder.detail(late_c=3, late_d=None)

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("PERF test.shape: ")
    assert " ms" in message
    # Static detail precedes late detail, in call order; a ``None`` value is
    # dropped rather than rendered as the literal text "None".
    _, _, detail_text = message.partition("ms")
    assert detail_text.strip() == "static_a=1 static_b=x late_c=3"


def test_disabled_logs_nothing_even_with_detail(caplog: pytest.LogCaptureFixture) -> None:
    set_perf_logging(False)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    with perf_timer("test.disabled", n=5) as recorder:
        recorder.detail(m=10)

    assert caplog.records == []


def test_exception_propagates_and_is_logged_with_failed_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_perf_logging(True)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    with pytest.raises(ValueError, match="boom"):
        with perf_timer("test.failure"):
            raise ValueError("boom")

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("PERF test.failure: ")
    assert "failed=True" in message


def test_exception_when_disabled_still_propagates_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_perf_logging(False)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    with pytest.raises(ValueError, match="boom"):
        with perf_timer("test.failure_disabled"):
            raise ValueError("boom")

    assert caplog.records == []


def _tiny_run_histograms() -> tuple[list[Histogram], dict]:
    """A minimal two-detector histogram set for a direct reduction call."""
    forward = Histogram(
        counts=np.array([120.0, 90.0, 70.0, 55.0, 44.0, 36.0], dtype=float),
        bin_width=0.016,
        t0_bin=0,
    )
    backward = Histogram(
        counts=np.array([80.0, 62.0, 50.0, 40.0, 33.0, 27.0], dtype=float),
        bin_width=0.016,
        t0_bin=0,
    )
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": forward.counts.size - 1,
        "bunching_factor": 1,
    }
    return [forward, backward], grouping


def test_reduce_grouped_asymmetry_emits_perf_record_when_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_perf_logging(True)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    histograms, grouping = _tiny_run_histograms()
    result = reduce_grouped_asymmetry(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
        facility="TESTINST",
    )

    assert result.time.size > 0
    matching = [r for r in caplog.records if "core.reduce.grouped_asymmetry" in r.message]
    assert len(matching) == 1
    assert matching[0].message.startswith("PERF core.reduce.grouped_asymmetry: ")
    assert "n_forward=1" in matching[0].message
    assert "n_backward=1" in matching[0].message


def test_reduce_grouped_asymmetry_emits_nothing_when_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_perf_logging(False)
    caplog.set_level(logging.INFO, logger="asymmetry.perf")

    histograms, grouping = _tiny_run_histograms()
    reduce_grouped_asymmetry(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
        facility="TESTINST",
    )

    assert caplog.records == []
