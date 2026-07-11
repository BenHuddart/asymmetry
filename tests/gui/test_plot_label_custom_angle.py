"""Custom-column legend labels resolve on the frequency (FFT) view.

The special **Angle** column uses the bare id ``"angle"`` (not ``custom:<hash>``)
and its per-run value lives in ``dataset.metadata["custom_fields"]["angle"]``.
Two things broke labelling it on an averaged FFT spectrum:

* the bare id was not recognised as a custom column, so it fell through to the
  generic top-level-metadata branch and rendered the baked ``"<run> Average"``
  run label; and
* an averaged spectrum copies its metadata from a per-group source that carries
  no ``custom_fields``, so even once recognised it had no inline value to read —
  it must fall back to the host-provided per-run map.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.plot_panel import PlotPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _select(panel: PlotPanel, key: str) -> None:
    idx = panel._label_field_combo.findData(key)
    assert idx >= 0, f"label field {key!r} not in combo"
    panel._label_field_combo.setCurrentIndex(idx)


def _fft_dataset() -> MuonDataset:
    """A spectrum shaped like ``compute_average_group_spectrum``'s output."""
    f = np.linspace(0.0, 10.0, 40)
    return MuonDataset(
        time=f,
        asymmetry=np.exp(-0.1 * f),
        error=np.zeros_like(f),
        metadata={
            "run_number": 100,
            "run_label": "100 Average",
            "plot_domain": "frequency",
        },
    )


def test_bare_angle_id_resolves_from_inline_custom_fields(qapp):
    # A time-domain dataset carries the Angle value inline; the bare "angle" id
    # must be recognised as a custom column, not fall through to the generic
    # metadata branch (which reads top-level metadata and misses custom_fields).
    panel = PlotPanel(domain="time")
    panel.set_custom_label_fields([("Angle (°)", "angle")])
    _select(panel, "angle")

    t = np.linspace(0.0, 5.0, 20)
    ds = MuonDataset(
        time=t,
        asymmetry=np.exp(-0.1 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 101, "run_label": "101", "custom_fields": {"angle": "45.0"}},
    )
    assert panel._is_valid_label_field("angle")
    assert panel._dataset_label_for(ds) == "45.0"


def test_angle_resolves_on_fft_spectrum_via_run_map(qapp):
    # The averaged FFT spectrum carries no inline custom_fields, so the Angle
    # label must resolve from the host-provided run map instead of falling back
    # to "<run> Average".
    panel = PlotPanel(domain="frequency")
    panel.set_custom_label_fields([("Angle (°)", "angle")])
    panel.set_custom_values_by_run({100: {"angle": "12.5"}})
    _select(panel, "angle")

    ds = _fft_dataset()
    assert "custom_fields" not in ds.metadata
    assert panel._dataset_label_for(ds) == "12.5"


def test_fft_angle_without_value_falls_back_to_run_label(qapp):
    # A run absent from the map (and with no inline value) degrades gracefully to
    # the baked run label rather than erroring.
    panel = PlotPanel(domain="frequency")
    panel.set_custom_label_fields([("Angle (°)", "angle")])
    panel.set_custom_values_by_run({999: {"angle": "10"}})  # different run
    _select(panel, "angle")

    assert panel._dataset_label_for(_fft_dataset()) == "100 Average"


def test_saved_bare_angle_id_survives_restore_before_columns_offered(qapp):
    # Restore can run before the host pushes the custom columns back in; a saved
    # bare "angle" selection must be preserved as the intent, not reset to "run",
    # and get selected once the column is offered.
    panel = PlotPanel(domain="frequency")
    panel.restore_state(
        {"default_label_field": "angle", "label_field": "angle", "frequency_plot_state": {}}
    )
    assert panel._default_label_field == "angle"
    panel.set_custom_label_fields([("Angle (°)", "angle")])
    assert panel._label_field_combo.currentData() == "angle"
