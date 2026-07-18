"""Tests for the inline α-calibration section (Corrections panel).

The estimate runs on a TaskRunner worker thread; ``_wait_until`` pumps the event
loop until it lands before asserting (same idiom as the other grouping GUI
tests).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.project.profiles import AlphaPolicy
from asymmetry.gui.windows.grouping.alpha_section import AlphaSectionWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_ms: int = 30_000) -> None:
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(timeout_ms)
    loop.exec()
    check.stop()
    guard.stop()
    assert predicate(), "timed out waiting for the alpha estimate"


def _run(run_number: int, *, ratio: float, metadata: dict | None = None) -> MuonDataset:
    forward = np.full(4, 100.0)
    backward = forward / ratio
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=forward, bin_width=0.01),
            Histogram(counts=backward, bin_width=0.01),
        ],
        metadata={"run_number": run_number, **(metadata or {})},
        grouping={"first_good_bin": 0, "last_good_bin": 3},
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _context() -> dict:
    return {
        "groups": {1: [0], 2: [1]},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [],
        "correction_provider": None,
        "reference_resolver": None,
        "facility": "",
    }


def _configured(section: AlphaSectionWidget, datasets, *, method="ratio", selected=None) -> None:
    section.configure(
        datasets=datasets,
        method=method,
        selected_run_number=selected,
        context_provider=_context,
    )


def test_configure_lists_runs_and_tf_highlight(qapp: QApplication) -> None:
    section = AlphaSectionWidget()
    tf = _run(2, ratio=2.0, metadata={"field_direction": "Transverse", "field": 100.0})
    plain = _run(1, ratio=2.0, metadata={"field_direction": "Longitudinal", "field": 3000.0})
    _configured(section, [plain, tf])
    run_numbers = {section._run_combo.itemData(i) for i in range(section._run_combo.count())}
    assert run_numbers == {1, 2}
    assert section._run_combo.currentData() == 2  # the TF candidate is auto-selected
    section.shutdown()


def test_estimate_emits_calibrated_policy(qapp: QApplication) -> None:
    section = AlphaSectionWidget()
    _configured(section, [_run(5, ratio=2.0)], selected=5)
    policies: list[AlphaPolicy] = []
    section.alpha_estimated.connect(lambda p: policies.append(p))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert policies, "no estimate emitted"
    policy = policies[-1]
    assert policy.mode == "calibrated"
    assert policy.value == pytest.approx(2.0)
    assert policy.method == "ratio"
    assert policy.source_run == 5
    assert "α =" in section._result_label.text()
    section.shutdown()


def test_estimate_reads_context_fresh(qapp: QApplication) -> None:
    """A group/pair change between estimates is honoured (context pulled at run)."""
    section = AlphaSectionWidget()
    # Four detectors: forward {1,2}, backward {3,4}; det 2 is hot.
    forward_hot = np.full(4, 900.0)
    run = Run(
        run_number=7,
        histograms=[
            Histogram(counts=np.full(4, 100.0), bin_width=0.01),
            Histogram(counts=forward_hot, bin_width=0.01),
            Histogram(counts=np.full(4, 50.0), bin_width=0.01),
            Histogram(counts=np.full(4, 50.0), bin_width=0.01),
        ],
        grouping={"first_good_bin": 0, "last_good_bin": 3},
    )
    t = np.arange(4) * 0.01
    ds = MuonDataset(
        time=t, asymmetry=np.zeros_like(t), error=np.ones_like(t), metadata={}, run=run
    )
    excluded: list[int] = []
    ctx = {
        "groups": {1: [0, 1], 2: [2, 3]},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": excluded,
        "correction_provider": None,
        "reference_resolver": None,
        "facility": "",
    }
    section.configure(
        datasets=[ds], method="ratio", selected_run_number=7, context_provider=lambda: ctx
    )
    policies: list[AlphaPolicy] = []
    section.alpha_estimated.connect(lambda p: policies.append(p))

    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0 and policies)
    with_hot = policies[-1].value

    excluded.append(2)  # exclude the hot detector; context is read fresh
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0 and len(policies) == 2)
    without_hot = policies[-1].value

    assert with_hot == pytest.approx(1000.0 / 100.0)
    assert without_hot == pytest.approx(100.0 / 100.0)
    section.shutdown()
