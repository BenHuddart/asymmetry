"""RED target for branch ``fix/fft-compute-discoverability``.

Round-2 + 2026-06-15 live finding (`_findings/windows-gui/CdS_MaxEnt.md`,
`GUI_LORE.md`): FFT is **not** auto-compute — there is a "Compute FFT" button at
the *bottom* of the Fourier panel, below the fold. An experienced tester missed it
twice and filed a false "FFT never renders" bug. The Fourier spectrum is the only
compute-on-demand view with **no on-canvas prompt** (the time-domain view has an
empty state via ``plot_panel._render_empty_plot_state``; MaxEnt reconstruction
shows "Run MaxEnt for this run…"; the ALC view shows "Build a scan to see the ALC
curve"). Worse, entering the frequency view with nothing computed currently
*falls back to the time view* (mainwindow ~L5848/10954), so the user gets no cue.

Desired behaviour: when the active run has no computed FFT, draw a centred message
**over the Fourier plot area** directing the user to configure the FFT parameters
and click **Compute FFT** — reusing the existing empty-state machinery. It must
clear once the FFT is computed (the not-computed marker, mainwindow ~L5598,
already clears on recompute).

xfail(strict) until implemented: today no such on-canvas prompt exists, so the
assertion fails; when the overlay lands it passes and strict-xfail forces removing
the marker.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _tf_dataset(run_number: int = 20711, field: float = 100.0) -> MuonDataset:
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": field},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number},
        run,
    )


def _plot_text_blob(plot_panel) -> str:
    """Concatenate all text drawn on the plot panel's figure, lowercased."""
    figure = getattr(plot_panel, "_figure", None)
    if figure is None:
        return ""
    chunks: list[str] = []
    for ax in figure.axes:
        chunks.extend(t.get_text() for t in ax.texts)
        chunks.append(ax.get_title())
    return " ".join(chunks).lower()


@pytest.mark.xfail(reason="fix/fft-compute-discoverability not yet implemented", strict=True)
def test_uncomputed_fft_view_prompts_to_compute(mw):
    mw._data_browser.add_dataset(_tf_dataset(20711))
    mw._on_dataset_selected(20711)

    # Enter the Fourier view without computing an FFT.
    mw._on_fourier()

    blob = _plot_text_blob(mw._plot_panel)
    assert "compute fft" in blob, (
        "no on-canvas prompt: the uncomputed Fourier view must tell the user to "
        "configure parameters and click Compute FFT (instead of an empty plot or "
        "a silent fall-back to the time view)."
    )
