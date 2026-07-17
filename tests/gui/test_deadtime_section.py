"""Tests for the inline deadtime-configuration section (Corrections panel).

Covers the embeddable body that replaced the retired ``DeadtimeDialog``: mode
seeding, the ``changed`` signal on edits, Fill-all, and the Estimate action
(``estimate_deadtime_from_histograms`` monkeypatched).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram
from asymmetry.gui.windows.grouping import deadtime_section as section_module
from asymmetry.gui.windows.grouping.deadtime_section import (
    DeadtimeSectionWidget,
    DeadtimeSourceRun,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _source_runs() -> list[DeadtimeSourceRun]:
    hists = [Histogram(counts=np.full(50, 100.0), bin_width=0.016) for _ in range(2)]
    return [
        DeadtimeSourceRun(run_number=4101, label="Run 4101", histograms=hists, good_frames=1000.0)
    ]


def _configured(mode: str = "off", **overrides) -> DeadtimeSectionWidget:
    section = DeadtimeSectionWidget()
    kwargs = dict(
        n_detectors=2,
        mode=mode,
        file_values_us=[],
        manual_values_us=[],
        manual_method="manual",
        estimated_us=None,
        source_run=None,
        source_runs=_source_runs(),
        reference_run_number=4101,
        peak_rates_per_us=[500.0, 500.0],
        bin_width_us=0.016,
        good_frames=1000.0,
    )
    kwargs.update(overrides)
    section.configure(**kwargs)
    return section


def test_configure_seeds_mode_without_emitting(qapp: QApplication) -> None:
    section = DeadtimeSectionWidget()
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section.configure(
        n_detectors=2,
        mode="manual",
        file_values_us=[],
        manual_values_us=[0.02, 0.02],
        manual_method="manual",
        estimated_us=None,
        source_run=None,
        source_runs=[],
        reference_run_number=None,
        peak_rates_per_us=[],
        bin_width_us=0.016,
        good_frames=1000.0,
    )
    assert section._current_mode() == "manual"
    assert section.get_policy().values == pytest.approx([0.02, 0.02])
    assert not fired  # seeding never emits


def test_mode_click_emits_changed(qapp: QApplication) -> None:
    section = _configured(mode="off")
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section._mode_buttons["manual"].click()
    assert section._current_mode() == "manual"
    assert fired


def test_fill_all_updates_values_and_emits(qapp: QApplication) -> None:
    section = _configured(mode="manual", manual_values_us=[0.01, 0.01])
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section._fill_all_spin.setValue(30.0)  # 30 ns
    section._fill_all_btn.click()
    policy = section.get_policy()
    assert policy.values == pytest.approx([0.03, 0.03])  # ns → µs
    assert fired


def test_estimate_fills_table_and_emits(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(section_module, "estimate_deadtime_from_histograms", lambda *a, **k: 0.05)
    section = _configured(mode="estimate")
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section._estimate_btn.click()
    policy = section.get_policy()
    assert policy.mode == "estimate"
    assert policy.estimated_us == pytest.approx(0.05)
    assert policy.source_run == 4101
    assert fired


def test_off_mode_policy(qapp: QApplication) -> None:
    section = _configured(mode="off")
    assert section.get_policy().mode == "off"


# -- adaptive per-mode layout ------------------------------------------------


def test_off_mode_hides_table_controls_estimate_and_summary(qapp: QApplication) -> None:
    section = _configured(mode="off")
    assert not section._table.isVisibleTo(section)
    assert not section._table_controls_widget.isVisibleTo(section)
    assert not section._estimate_row_widget.isVisibleTo(section)
    assert not section._summary_row_widget.isVisibleTo(section)
    assert not section._disclosure_btn.isVisibleTo(section)
    assert section._file_hint.text() == "Deadtime correction is disabled."


def test_file_mode_shows_summary_and_collapsed_disclosure(qapp: QApplication) -> None:
    section = _configured(mode="file", file_values_us=[0.010, 0.010])
    # Summary reports the mean value and detector count.
    assert "10.000 ns" in section._summary_label.text()
    assert "2 detectors" in section._summary_label.text()
    # Table controls and the estimate row belong to other modes.
    assert not section._table_controls_widget.isVisibleTo(section)
    assert not section._estimate_row_widget.isVisibleTo(section)
    # Disclosure starts collapsed with the read-only table hidden; toggling reveals it.
    assert section._disclosure_btn.isVisibleTo(section)
    assert not section._disclosure_btn.isChecked()
    assert not section._table.isVisibleTo(section)
    section._disclosure_btn.setChecked(True)
    assert section._table.isVisibleTo(section)
    section._disclosure_btn.setChecked(False)
    assert not section._table.isVisibleTo(section)


def test_estimate_mode_shows_source_row_summary_and_disclosure(qapp: QApplication) -> None:
    section = _configured(mode="estimate")
    assert section._estimate_row_widget.isVisibleTo(section)
    assert section._summary_row_widget.isVisibleTo(section)
    assert section._disclosure_btn.isVisibleTo(section)
    assert not section._disclosure_btn.isChecked()
    assert not section._table.isVisibleTo(section)
    assert not section._table_controls_widget.isVisibleTo(section)
    section._disclosure_btn.setChecked(True)
    assert section._table.isVisibleTo(section)


def test_manual_mode_shows_capped_table_and_controls(qapp: QApplication) -> None:
    section = _configured(mode="manual", manual_values_us=[0.01, 0.01])
    assert section._table.isVisibleTo(section)
    assert section._table_controls_widget.isVisibleTo(section)
    assert not section._estimate_row_widget.isVisibleTo(section)
    assert not section._disclosure_btn.isVisibleTo(section)


def test_disclosure_toggle_does_not_emit_changed(qapp: QApplication) -> None:
    section = _configured(mode="file", file_values_us=[0.010, 0.010])
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section._disclosure_btn.setChecked(True)
    assert not fired  # view-only reveal never mutates the policy


def test_table_height_capped_with_many_detectors(qapp: QApplication) -> None:
    n = 64
    section = _configured(
        mode="manual",
        n_detectors=n,
        manual_values_us=[0.01] * n,
        peak_rates_per_us=[500.0] * n,
    )
    section.resize(420, 480)
    section.show()
    qapp.processEvents()
    table = section._table
    assert table.rowCount() == n
    # Cap = 6 data rows + header + frame — far short of all 64 rows.
    row_h = table.verticalHeader().defaultSectionSize()
    assert 0 < table.maximumHeight() < row_h * n
    # The rows beyond the cap scroll inside the table, not the tab.
    assert table.verticalScrollBar().maximum() > 0
    section.close()
