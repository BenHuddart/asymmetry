"""Render-only MaxEnt smoke tests against the local WiMDA muon-school corpus.

These exercise the full load → MaxEnt → reconstruction-overlay path on real
runs (one TF, one ZF).  They are **not numerical oracles** — no trusted
reference spectrum exists for these files — only that the overlay renders and
ZF/LF mode constrains the fit without crashing.

The corpus lives outside the repository (``~/Documents/WiMDA muon school``), so
every test here skips cleanly when the files are absent (always the case in CI).
Run them locally to confirm the GUI surfaces survive real data.

Marked ``slow``: the TF reconstruction takes ~24s per parametrization, which
would dominate a standard-tier worker. Run explicitly (``test --
tests/test_maxent_corpus_smoke.py``) or via ``--tier full`` after MaxEnt or
reconstruction-overlay changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore

from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    reconstruct_group_signals,
    run_cycles,
)
from asymmetry.core.representation import build_maxent_reconstruction_datasets
from asymmetry.gui.panels.plot_panel import PlotPanel

_CORPUS = Path.home() / "Documents" / "WiMDA muon school"


def _first_match(subdir: str, pattern: str) -> Path | None:
    """Return the first corpus file matching *pattern* under *subdir*, or None."""
    root = _CORPUS / subdir
    if not root.exists():
        return None
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


# A transverse-field PSI HiFi run (60 kG, multi-group) and a zero-field ISIS
# EMU F/B run — both confirmed loadable.  Resolved by glob so a corpus reshuffle
# (the files live under a per-experiment ``data/`` subdir) does not break them.
_TF_RUN = _first_match("Magnetism/AFM transition in high TF", "tdc_hifi_*.mdu")
_ZF_RUN = _first_match("Magnetism/Ferromagnetic nickel", "emu*.nxs")


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _load_run(path: Path):
    from asymmetry.core.io import load

    dataset = load(str(path))
    run = getattr(dataset, "run", None)
    return run if run is not None else dataset


@pytest.mark.skipif(_TF_RUN is None, reason="TF corpus run not present")
@pytest.mark.parametrize("combined", [False, True])
def test_tf_run_reconstruction_overlay_renders(qapp: QApplication, combined: bool) -> None:
    run = _load_run(_TF_RUN)
    config = MaxEntConfig(
        n_spectrum_points=64,
        auto_window=True,
        outer_cycles=2,
        inner_iterations=2,
    )
    maxent_input = build_maxent_input(run, config)
    result = run_cycles(maxent_input, config)
    datasets = build_maxent_reconstruction_datasets(
        reconstruct_group_signals(maxent_input, result.state), run
    )
    assert datasets  # at least one group reconstructed

    panel = PlotPanel()
    try:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_maxent_reconstruction(datasets, combined=combined)
        expected_axes = 2 if combined else 2 * len(datasets)
        assert len(panel._figure.axes) == expected_axes
    finally:
        panel.close()
        panel.deleteLater()


@pytest.mark.skipif(_ZF_RUN is None, reason="ZF corpus run not present")
def test_zf_run_constrains_to_two_groups_with_pinned_phases(qapp: QApplication) -> None:
    run = _load_run(_ZF_RUN)
    grouping = run.grouping
    pair = sorted(int(g) for g in grouping.get("groups", {}))[:2]
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.0,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=2,
        inner_iterations=2,
        mode="zf_lf",
        selected_group_ids=pair,
    )
    # ZF/LF mode hard-constrains the fit: exactly two F/B groups, phases pinned
    # 0/180, amplitudes tied through α.
    maxent_input = build_maxent_input(run, config)
    assert len(maxent_input.groups) == 2
    assert [g.phase_degrees for g in maxent_input.groups] == [0.0, 180.0]
    assert maxent_input.zf_lf_alpha is not None

    result = run_cycles(maxent_input, config)
    datasets = build_maxent_reconstruction_datasets(
        reconstruct_group_signals(maxent_input, result.state), run
    )
    panel = PlotPanel()
    try:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_maxent_reconstruction(datasets)
        assert len(panel._figure.axes) == 2 * len(datasets)
    finally:
        panel.close()
        panel.deleteLater()
