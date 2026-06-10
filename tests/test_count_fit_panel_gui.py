"""GUI wiring for the count-domain fit-target selector in the Multi-Group window.

These tests exercise the routing only — the count-fit numerics are covered by
``tests/test_count_domain_fits.py``. The detector-balance α is recovered here
even with the default model because it is fixed by the forward/backward count
ratio, independent of the oscillation model.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.simulate import build_builtin_template, simulate_run
from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow


def _tf(t, A=20.0, f=1.5, phi=0.0):  # noqa: N803 (A is the conventional asymmetry symbol)
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


@pytest.fixture
def fb_dataset() -> MuonDataset:
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.5, "phi": 0.3}, total_events=40e6, alpha=1.25, seed=1
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def test_target_selector_pushes_mode_to_both_tabs(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # Forward + Backward (free α)
    assert window._single_fit_tab._count_fit_mode == "fb"
    assert window._batch_fit_tab._count_fit_mode == "fb"
    # Cost selector is enabled for count modes, disabled for All groups.
    assert window._cost_combo.isEnabled()
    window._target_combo.setCurrentIndex(0)
    assert window._single_fit_tab._count_fit_mode == "all"
    assert not window._cost_combo.isEnabled()


def test_fb_count_fit_runs_and_recovers_alpha(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb

    captured = []
    window.count_fit_completed.connect(lambda dataset, result: captured.append(result))
    window._single_fit_tab._run_count_domain_fit()

    assert len(captured) == 1
    result = captured[0]
    assert result.success
    alpha = result.group_results[1].parameters["alpha"].value
    assert alpha == pytest.approx(1.25, abs=0.05)


def test_single_count_fit_runs_and_emits(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # Single group
    window._side_combo.setCurrentIndex(0)  # Forward
    assert window._single_fit_tab._count_single_side == "forward"
    assert window._side_combo.isEnabled()

    captured = []
    window.count_fit_completed.connect(lambda dataset, result: captured.append(result))
    window._single_fit_tab._run_count_domain_fit()
    assert len(captured) == 1
    assert "N0" in captured[0].parameters.names


def test_all_groups_mode_leaves_existing_path(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    # Default is All groups; the count-domain routing must not intercept it.
    assert window._single_fit_tab._count_fit_mode == "all"
