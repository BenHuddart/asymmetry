"""Tests for the shared GLE export orchestration (``gui/utils/gle_export.py``).

Pins the ``run_gle_export`` contract the three export surfaces rely on:
GUI-thread build, worker-thread compile, dialog/editor seams called with the
right arguments on each terminal path. Dialog seams are monkeypatched — their
internal pytest suppression is thereby bypassed, which is the intended way
for tests to observe the flow.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
# The builder callbacks in these tests drive the real gleplot API.
pytest.importorskip("gleplot", reason="gle extra not installed")
from PySide6.QtWidgets import QFileDialog, QWidget  # noqa: E402

import asymmetry.gui.utils.gle_export as gle_export  # noqa: E402
from asymmetry.gui.utils.gle_export import (  # noqa: E402
    GleExportBuild,
    dedup_export_token,
    extract_gle_data_dependencies,
    prune_stale_sidecars,
    run_gle_export,
    safe_file_token,
)


def _wait(qapp, predicate, timeout_s=10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _build(glp, gle_path: Path, export_dir: Path, **kwargs) -> GleExportBuild:
    fig = glp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1])
    fig.savefig(str(gle_path))
    return GleExportBuild(files=[gle_path])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_safe_file_token():
    assert safe_file_token("T = 1.5 K (ZF)") == "T_1_5_K_ZF"
    assert safe_file_token("  ") == "dataset"
    assert safe_file_token("", fallback="group") == "group"


def test_dedup_export_token():
    used: set[str] = {"a"}
    assert dedup_export_token("a", used) == "a_2"
    assert dedup_export_token("a", used) == "a_3"
    assert dedup_export_token("b", None) == "b"


def test_extract_gle_data_dependencies(tmp_path):
    gle = tmp_path / "f.gle"
    gle.write_text('begin graph\n  data "a.dat" d1=c1,c2\n  data b.fit\nend graph\n')
    assert extract_gle_data_dependencies(gle) == ["a.dat", "b.fit"]


# ---------------------------------------------------------------------------
# prune_stale_sidecars
# ---------------------------------------------------------------------------
def test_prune_stale_sidecars_removes_unreferenced_dat_and_fit(tmp_path):
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text('begin graph\n  data "used.dat"\nend graph\n')
    (export_dir / "used.dat").write_text("1 2\n")
    (export_dir / "orphan.dat").write_text("3 4\n")
    (export_dir / "orphan.fit").write_text("fit\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert set(removed) == {"orphan.dat", "orphan.fit"}
    assert (export_dir / "used.dat").exists()
    assert not (export_dir / "orphan.dat").exists()
    assert not (export_dir / "orphan.fit").exists()


def test_prune_stale_sidecars_keeps_files_referenced_via_kept_list(tmp_path):
    """A sidecar not mentioned by a ``data`` command but reported through
    ``GleExportBuild.files`` (e.g. an extra fit sidecar) must survive."""
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    kept_sidecar = export_dir / "extra.fit"
    kept_sidecar.write_text("fit\n")
    (export_dir / "orphan.dat").write_text("3 4\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path, kept_sidecar])

    assert removed == ["orphan.dat"]
    assert kept_sidecar.exists()


def test_prune_stale_sidecars_never_touches_non_sidecar_suffixes(tmp_path):
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    (export_dir / "notes.txt").write_text("keep me\n")
    (export_dir / "figure.pdf").write_bytes(b"%PDF-fake")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    assert (export_dir / "notes.txt").exists()
    assert (export_dir / "figure.pdf").exists()


def test_prune_stale_sidecars_keeps_compiled_outputs_of_other_formats(tmp_path):
    """A re-export to a different output format must not delete last time's
    compiled output — compiled outputs of any known GLE format for the same
    stem are treated as legitimate, not stale."""
    export_dir = tmp_path / "figure.gleplot"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    for fmt in ("pdf", "eps", "png", "jpg", "svg"):
        (export_dir / f"figure.{fmt}").write_bytes(b"data")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    for fmt in ("pdf", "eps", "png", "jpg", "svg"):
        assert (export_dir / f"figure.{fmt}").exists()


def test_prune_stale_sidecars_refuses_non_gleplot_directory(tmp_path):
    """Safety rail: only folders this machinery owns (``*.gleplot``) are
    ever cleaned — anything else is left untouched even if it happens to
    hold ``.dat``/``.fit`` files."""
    export_dir = tmp_path / "not_a_gleplot_folder"
    export_dir.mkdir()
    gle_path = export_dir / "figure.gle"
    gle_path.write_text("begin graph\nend graph\n")
    (export_dir / "orphan.dat").write_text("3 4\n")

    removed = prune_stale_sidecars(export_dir, gle_path, kept=[gle_path])

    assert removed == []
    assert (export_dir / "orphan.dat").exists()


# ---------------------------------------------------------------------------
# run_gle_export terminal paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["pdf", "eps"])
def test_export_preview_compiles_once(qapp, tmp_path, monkeypatch, device):
    from asymmetry.gui.windows.gle_preview import GlePlotPreview

    parent = QWidget()
    calls = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "figure.gleplot"), "")
    )
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "gle")

    def compile(exe, script, devices, **kwargs):
        calls.append(devices)

        for device in devices:
            Path(script).with_suffix("." + device).write_bytes(b"invalid PDF")

    monkeypatch.setattr(gle_export, "compile_gle", compile)
    run_gle_export(
        parent,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format=device,
        prepare=lambda: _build,
    )
    preview = parent.findChild(GlePlotPreview)
    assert preview is not None
    assert _wait(qapp, lambda: not preview.tasks.active_count)
    assert calls == ([("pdf",)] if device == "pdf" else [("eps", "pdf")])
    assert "Preview unavailable" in preview.status.text()
    preview.zoom_in()
    preview.fit()
    assert len(calls) == 1
    preview.close()
    parent.close()


def test_export_without_compiler_saves_script(qapp, tmp_path, monkeypatch):
    from asymmetry.gui.windows.gle_preview import GlePlotPreview

    parent = QWidget()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "figure.gleplot"), "")
    )
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    run_gle_export(
        parent,
        dialog_title="Export",
        default_name="figure.gleplot",
        output_format="pdf",
        prepare=lambda: _build,
    )
    preview = parent.findChild(GlePlotPreview)
    assert _wait(qapp, lambda: not preview.tasks.active_count)
    assert "Install GLE" in preview.status.text()
    assert (tmp_path / "figure.gleplot" / "figure.gle").exists()
    preview.close()
    parent.close()


import numpy as np  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.gui.panels.plot_panel import PlotPanel  # noqa: E402


def _overlay_dataset(run_number: int) -> MuonDataset:
    t = np.linspace(0.0, 10.0, 120)
    a = 0.2 * np.exp(-0.3 * t) * np.cos(2.0 * np.pi * t)
    e = np.full_like(t, 0.005)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": run_number})


def test_export_payloads_carry_waterfall_offset(qapp):
    panel = PlotPanel(domain="time")
    if not panel._has_mpl:
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    try:
        datasets = [_overlay_dataset(1), _overlay_dataset(2), _overlay_dataset(3)]
        panel.set_overlay_enabled(True)
        panel.set_waterfall_enabled(True)
        panel.plot_datasets(datasets)

        delta = panel._waterfall_resolved_delta
        assert delta > 0.0
        payloads = panel.get_current_plot_export_data()
        assert payloads is not None and len(payloads) == 3
        for i, payload in enumerate(payloads):
            assert payload["waterfall_offset"] == pytest.approx(i * delta)
    finally:
        panel.close()
        panel.deleteLater()


def test_export_payloads_have_no_offset_when_waterfall_off(qapp):
    panel = PlotPanel(domain="time")
    if not panel._has_mpl:
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    try:
        datasets = [_overlay_dataset(1), _overlay_dataset(2)]
        panel.set_overlay_enabled(True)
        panel.plot_datasets(datasets)
        payloads = panel.get_current_plot_export_data()
        assert payloads is not None
        assert all("waterfall_offset" not in p for p in payloads)
    finally:
        panel.close()
        panel.deleteLater()


def test_dat_file_offsets_data_and_records_header(qapp, tmp_path):
    panel = PlotPanel(domain="time")
    if not panel._has_mpl:
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    try:
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([0.10, 0.20, 0.30])
        err = np.array([0.01, 0.01, 0.01])
        payload = {
            "run_number": 7,
            "label": "trace",
            "data": {"t": t, "y": y, "err": err},
            "waterfall_offset": 5.0,
        }
        dat_path = tmp_path / "trace.dat"
        panel._write_data_file(dat_path, payload)
        text = dat_path.read_text(encoding="utf-8")

        assert "! START OF RUN INFORMATION" in text
        assert "!  waterfall offset: 5" in text
        # Data rows carry the offset; the source array is untouched.
        rows = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
        values = [float(ln.split()[1]) for ln in rows]
        np.testing.assert_allclose(values, y + 5.0)
        np.testing.assert_allclose(payload["data"]["y"], y)
    finally:
        panel.close()
        panel.deleteLater()


def test_fit_file_offsets_curve_and_records_header(qapp, tmp_path):
    panel = PlotPanel(domain="time")
    if not panel._has_mpl:
        panel.deleteLater()
        pytest.skip("matplotlib not available")
    try:
        t_fit = np.array([0.0, 1.0, 2.0])
        y_fit = np.array([0.15, 0.10, 0.05])
        payload = {
            "run_number": 7,
            "label": "trace",
            "fit": {"t": t_fit, "y": y_fit, "label": "Fit"},
            "waterfall_offset": 2.0,
        }
        fit_path = tmp_path / "trace.fit"
        panel._write_fit_file(fit_path, payload)
        text = fit_path.read_text(encoding="utf-8")

        assert "! waterfall offset: 2" in text
        rows = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
        values = [float(ln.split()[1]) for ln in rows]
        np.testing.assert_allclose(values, y_fit + 2.0)
    finally:
        panel.close()
        panel.deleteLater()
