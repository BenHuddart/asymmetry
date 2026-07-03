"""Tests for grouping dialog alpha-estimation workflow."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

import asymmetry.gui.windows.grouping.dialog as grouping_dialog_dialog_module
import asymmetry.gui.windows.grouping_dialog as grouping_dialog_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset_with_histograms() -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=4001,
        histograms=[h1, h2],
        metadata={"run_number": 4001, "title": "Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 4001},
        run=run,
    )


def _vector_dataset_with_histograms(run_number: int = 4010) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Vector Grouping Test"},
        grouping={
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "alpha_x": 1.1,
            "alpha_y": 1.2,
            "alpha_z": 1.3,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dataset_with_deadtime_profile(run_number: int, tau_us: float) -> MuonDataset:
    amplitude = 120.0
    bin_width = 0.01
    num_good_frames = 1000.0
    lifetime_us = 2.1969811
    times = (np.arange(12, dtype=float) + 1.0) * bin_width
    frame_scale = num_good_frames * bin_width
    true_counts = amplitude * np.exp(-times / lifetime_us)
    observed = true_counts * (
        1.0 - (true_counts / frame_scale) * lifetime_us * (1.0 - np.exp(-tau_us / lifetime_us))
    )
    histograms = [Histogram(observed.copy(), bin_width=bin_width) for _ in range(4)]
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "title": f"Deadtime Run {run_number}"},
        grouping={
            "groups": {1: [1, 2], 2: [3, 4]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 11,
            "good_frames": num_good_frames,
        },
    )
    return MuonDataset(
        time=np.arange(12, dtype=float) * bin_width,
        asymmetry=np.zeros(12, dtype=float),
        error=np.full(12, 0.01, dtype=float),
        metadata={"run_number": run_number},
        run=run,
    )


def _vector_dataset_with_ratio(run_number: int, ratio: float) -> MuonDataset:
    forward = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)
    backward = forward / ratio
    h1 = Histogram(counts=forward, bin_width=0.01)
    h2 = Histogram(counts=backward, bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": f"Vector Run {run_number}"},
        grouping={
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dataset_with_ratio(run_number: int, ratio: float) -> MuonDataset:
    forward = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)
    backward = forward / ratio
    h1 = Histogram(counts=forward, bin_width=0.01)
    h2 = Histogram(counts=backward, bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": f"Run {run_number}"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _two_period_dataset(run_number: int = 6001) -> MuonDataset:
    red = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    green = Histogram(counts=np.array([120.0, 120.0, 120.0, 120.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[red],
        metadata={"run_number": run_number, "period_count": 2},
        grouping={
            "groups": {1: [1], 2: [1]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "period_mode": str(PeriodMode.GREEN),
            "period_histograms": [[red], [green]],
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "period_count": 2},
        run=run,
    )


def test_estimate_alpha_updates_spinbox(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    assert dialog._alpha_spin.value() == pytest.approx(2.0)


def test_get_grouping_result_contains_required_keys(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.metadata["facility"] = "PSI"
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["dead_time_us"] = [0.01, 0.01]
    dialog = GroupingDialog([dataset])
    dialog._deadtime_checkbox.setChecked(True)
    idx = dialog._background_mode_combo.findData("range")
    dialog._background_mode_combo.setCurrentIndex(idx)
    dialog._bunch_spin.setValue(1234)
    result = dialog.get_grouping_result()
    assert result is not None
    assert "forward_indices" in result
    assert "backward_indices" in result
    assert "alpha" in result
    assert "included_groups" in result
    assert result["included_groups"] == {1: True, 2: True}
    assert result["deadtime_correction"] is True
    assert result["background_correction"] is True
    assert result["background_mode"] == "range"
    assert result["bunching_factor"] == 1234


def test_grouping_result_respects_group_include_checkbox(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    include_item = dialog._group_table.item(1, 1)
    assert include_item is not None
    include_item.setCheckState(Qt.CheckState.Unchecked)

    result = dialog.get_grouping_result()

    assert result is not None
    assert result["included_groups"] == {1: True, 2: False}


def test_grouping_dialog_does_not_show_bunching_rules(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    labels = {label.text() for label in dialog.findChildren(QLabel)}

    assert "Bunching Rules" not in labels
    assert dialog._bunch_spin.toolTip() == "Set any bunching factor >= 1."


def test_deadtime_modes_available_without_file_deadtime(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    payload = dialog._current_grouping_payload()

    assert dialog._deadtime_checkbox.isEnabled()
    assert dialog._deadtime_checkbox.isChecked() is False
    assert dialog._deadtime_mode_buttons["file"].isChecked()
    assert "load" not in dialog._deadtime_mode_buttons
    assert payload["deadtime_correction"] is False
    dialog._deadtime_checkbox.setChecked(True)
    assert dialog._deadtime_mode_buttons["file"].isEnabled()
    assert dialog._deadtime_mode_buttons["file"].isChecked()
    assert dialog._deadtime_mode_buttons["manual"].isEnabled()
    assert dialog._deadtime_mode_buttons["estimate"].isEnabled()


def test_manual_deadtime_payload_resolves_uniform_values(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("manual")
    dialog._update_deadtime_controls()
    dialog._deadtime_value_combo.setCurrentIndex(0)
    dialog._deadtime_value_combo.setEditText("25.0")
    dialog._on_deadtime_value_edited()
    dialog._deadtime_value_combo.setCurrentIndex(1)
    dialog._deadtime_value_combo.setEditText("25.0")
    dialog._on_deadtime_value_edited()

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_correction"] is True
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "manual"
    assert payload["dead_time_us"] == pytest.approx([0.025, 0.025])


def test_file_deadtime_updates_detector_value_combo(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["dead_time_us"] = [0.011, 0.022]
    dialog = GroupingDialog([dataset])

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("file")
    dialog._update_deadtime_controls()

    assert dialog._deadtime_value_combo.count() == 2
    assert dialog._deadtime_value_combo.itemText(0) == "H1: 11.000 ns"
    assert dialog._deadtime_value_combo.itemText(1) == "H2: 22.000 ns"


def test_estimate_deadtime_uses_reference_run_only(qapp: QApplication) -> None:
    reference = _dataset_with_deadtime_profile(4101, 0.02)
    other = _dataset_with_deadtime_profile(4102, 0.04)
    dialog = GroupingDialog(
        [reference, other],
        selected_run_number=4101,
        selected_run_numbers=[4101, 4102],
    )

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("estimate")

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_mode"] == "estimate"
    assert payload["deadtime_method"] == "estimate"
    assert payload["deadtime_reference_run"] == 4101
    assert payload["run_numbers"] == [4101, 4102]
    assert payload["dead_time_us"] == pytest.approx([0.02, 0.02, 0.02, 0.02], rel=1e-2, abs=5e-4)
    assert dialog._deadtime_value_combo.itemText(0).startswith("H1: 20.000")


def test_calibrate_deadtime_populates_explicit_table(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    monkeypatch.setattr(
        grouping_dialog_dialog_module,
        "calibrate_deadtime_from_histograms",
        lambda *args, **kwargs: [0.011, 0.022],
    )

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("manual")
    dialog._update_deadtime_controls()
    assert dialog._deadtime_calibrate_btn.isEnabled()
    assert "Fit one deadtime value per detector" in dialog._deadtime_calibrate_btn.toolTip()
    dialog._calibrate_deadtime_from_reference()
    payload = dialog.get_grouping_result()

    assert payload is not None
    assert dialog._deadtime_mode_buttons["manual"].isChecked()
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "calibrate"
    assert payload["dead_time_us"] == pytest.approx([0.011, 0.022])
    assert payload["deadtime_reference_run"] == 4001
    assert dialog._deadtime_value_combo.itemText(0) == "H1: 11.000 ns"


def test_background_range_mode_disabled_for_non_psi_data(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    idx = dialog._background_mode_combo.findData("range")
    item = dialog._background_mode_combo.model().item(idx)
    assert item is not None and not item.isEnabled()
    # Tail fit and background-run modes stay available on pulsed data.
    for mode in ("tail_fit", "reference_run"):
        idx = dialog._background_mode_combo.findData(mode)
        assert dialog._background_mode_combo.model().item(idx).isEnabled()
    assert dialog._current_grouping_payload()["background_correction"] is False
    assert dialog._current_grouping_payload()["background_mode"] == "none"


def _tf_dataset_with_histograms(run_number: int = 4020) -> MuonDataset:
    """A dataset grouped with a transverse-field dual-grouping (non-canonical)
    preset declaring two projections — the MuSR ``Transverse (Vector)`` shape."""
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "TF Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
            "group_names": {
                1: "Top-Bottom Top",
                2: "Top-Bottom Bottom",
                3: "Fwd-Back Forward",
                4: "Fwd-Back Backward",
            },
            "projections": [
                {"label": "Top-Bottom", "forward_group": 1, "backward_group": 2},
                {"label": "Fwd-Back", "forward_group": 3, "backward_group": 4},
            ],
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _wep_dataset_with_histograms(run_number: int = 4030) -> MuonDataset:
    """A non-canonical preset whose projections declare their own alpha — the
    GPS ``WEP`` shape (FB alpha 0.75 / UD alpha 1.0)."""
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "WEP Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
            "group_names": {
                1: "FB Forward",
                2: "FB Backward",
                3: "UD Up",
                4: "UD Down",
            },
            "projections": [
                {"label": "FB", "forward_group": 1, "backward_group": 2, "alpha": 0.75},
                {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.0},
            ],
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def test_vector_mode_shows_per_axis_alpha_controls(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])

    assert dialog._vector_axis_pairs
    assert not dialog._vector_alpha_widget.isHidden()
    assert dialog._single_alpha_widget.isHidden()


def test_transverse_field_preset_shows_per_projection_alpha_controls(
    qapp: QApplication,
) -> None:
    """A non-canonical (transverse-field) projection set shows the generalized
    per-projection alpha table — one row per declared projection — instead of
    the single-alpha control, so users can recalibrate each projection's alpha."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])

    # The TF projections are detected (they drive the plot chip bar) ...
    assert set(dialog._vector_axis_pairs) == {"Top-Bottom", "Fwd-Back"}
    # ... and each gets a row in the per-projection alpha table.
    assert set(dialog._vector_alpha_spins) == {"Top-Bottom", "Fwd-Back"}
    assert not dialog._vector_alpha_widget.isHidden()
    assert dialog._single_alpha_widget.isHidden()
    # No canonical P_x/P_y/P_z rows leak into a non-canonical table.
    assert "P_z" not in dialog._vector_alpha_spins


def test_transverse_field_payload_persists_per_projection_alpha(
    qapp: QApplication,
) -> None:
    """Editing a non-canonical projection's α writes it into that projection's
    ``alpha`` in the projections payload, while the base alpha stays the single
    spin and no canonical per-axis alpha keys leak."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._alpha_spin.setValue(1.42)
    dialog._vector_alpha_spins["Top-Bottom"].setValue(0.83)
    dialog._vector_alpha_spins["Fwd-Back"].setValue(1.17)

    payload = dialog._current_grouping_payload()

    # Base alpha is the single spin (the projections' fallback), not a stale
    # canonical P_z spin.
    assert payload["alpha"] == pytest.approx(1.42)
    # No canonical per-axis alpha keys leak into a non-canonical grouping.
    assert "alpha_x" not in payload
    assert "alpha_z" not in payload
    # The edited per-projection alphas land inside the projections payload.
    by_label = {p["label"]: p for p in payload["projections"]}
    assert set(by_label) == {"Top-Bottom", "Fwd-Back"}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(0.83)
    assert by_label["Fwd-Back"]["alpha"] == pytest.approx(1.17)


def test_per_projection_alpha_seeds_from_declared_projection_alpha(
    qapp: QApplication,
) -> None:
    """Each non-canonical row seeds from its projection's declared alpha (e.g.
    GPS WEP's FB = 0.75 / UD = 1.0), so the table opens on the real values."""
    dialog = GroupingDialog([_wep_dataset_with_histograms()])

    assert set(dialog._vector_alpha_spins) == {"FB", "UD"}
    assert dialog._vector_alpha_spins["FB"].value() == pytest.approx(0.75)
    assert dialog._vector_alpha_spins["UD"].value() == pytest.approx(1.0)


def test_per_projection_alpha_estimate_updates_and_persists(
    qapp: QApplication,
) -> None:
    """Estimating a non-canonical projection updates its spin and the estimated
    value persists into the projection payload."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("Top-Bottom")
    estimated = dialog._vector_alpha_spins["Top-Bottom"].value()

    payload = dialog._current_grouping_payload()
    by_label = {p["label"]: p for p in payload["projections"]}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(estimated)


def test_per_projection_alpha_estimate_persists_provenance(qapp: QApplication) -> None:
    """A non-canonical projection's estimate carries its reference-run provenance
    into the projection payload, and a manual edit invalidates it — mirroring the
    canonical per-axis provenance."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("Top-Bottom")
    estimated = dialog._vector_alpha_spins["Top-Bottom"].value()

    payload = dialog._current_grouping_payload()
    by_label = {p["label"]: p for p in payload["projections"]}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(estimated)
    assert "alpha_reference_run" in by_label["Top-Bottom"]
    # Fwd-Back was not estimated, so it carries a value but no provenance.
    assert "alpha_reference_run" not in by_label["Fwd-Back"]

    # A manual edit invalidates the provenance for that projection.
    dialog._vector_alpha_spins["Top-Bottom"].setValue(estimated + 0.5)
    payload2 = dialog._current_grouping_payload()
    by_label2 = {p["label"]: p for p in payload2["projections"]}
    assert "alpha_reference_run" not in by_label2["Top-Bottom"]


def test_per_projection_alpha_edit_survives_detector_layout_accept(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsaved per-projection alpha edit is preserved when the detector-layout
    editor is accepted (its result carries the preset-declared alpha, which must
    not clobber the user's edit for a surviving projection label)."""
    dialog = GroupingDialog([_wep_dataset_with_histograms()])
    dialog._vector_alpha_spins["FB"].setValue(0.83)  # unsaved table edit

    result = {
        "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
        "group_names": {1: "FB Forward", 2: "FB Backward", 3: "UD Up", 4: "UD Down"},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [],
        # The editor returns the preset-declared alphas (FB = 0.75), not the edit.
        "projections": [
            {"label": "FB", "forward_group": 1, "backward_group": 2, "alpha": 0.75},
            {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.0},
        ],
        "grouping_preset": "WEP",
        "instrument": "GPS",
    }

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return result

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog._on_detector_layout()

    by_label = {p["label"]: p for p in dialog._projection_specs}
    assert by_label["FB"]["alpha"] == pytest.approx(0.83)  # edit preserved
    assert by_label["UD"]["alpha"] == pytest.approx(1.0)  # untouched projection intact
    # The rebuilt table reflects the preserved edit.
    assert dialog._vector_alpha_spins["FB"].value() == pytest.approx(0.83)


def test_grp_round_trip_preserves_projections() -> None:
    """serialize_grp/parse_grp round-trip the projections list (label, groups,
    per-projection alpha, tint) so non-canonical per-projection alpha survives a
    .grp save/load."""
    payload = {
        "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "projections": [
            {
                "label": "FB",
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 0.75,
                "tint": "#2E8B74",
            },
            {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.0},
        ],
    }
    text = GroupingDialog.serialize_grp(payload)
    assert "projection.0=" in text
    parsed = GroupingDialog.parse_grp(text)

    projs = parsed["projections"]
    assert [p["label"] for p in projs] == ["FB", "UD"]
    assert projs[0]["forward_group"] == 1
    assert projs[0]["backward_group"] == 2
    assert projs[0]["alpha"] == pytest.approx(0.75)
    assert projs[0]["tint"] == "#2E8B74"
    assert projs[1]["alpha"] == pytest.approx(1.0)
    assert "tint" not in projs[1]


def test_vector_payload_contains_per_axis_alpha_values(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._vector_alpha_spins["P_x"].setValue(1.11)
    dialog._vector_alpha_spins["P_y"].setValue(1.22)
    dialog._vector_alpha_spins["P_z"].setValue(1.33)

    payload = dialog._current_grouping_payload()

    assert payload["alpha_x"] == pytest.approx(1.11)
    assert payload["alpha_y"] == pytest.approx(1.22)
    assert payload["alpha_z"] == pytest.approx(1.33)
    assert payload["alpha"] == pytest.approx(1.33)


def test_vector_payload_records_per_axis_alpha_provenance(qapp: QApplication) -> None:
    """After estimating an axis, the payload carries that axis's error and
    reference run (per-axis provenance, skipped in Phase 1)."""
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("P_x")
    estimated = dialog._vector_alpha_spins["P_x"].value()

    payload = dialog._current_grouping_payload()

    assert payload["alpha_x"] == pytest.approx(estimated)
    assert "alpha_x_reference_run" in payload
    # P_y was not estimated, so it carries a value but no provenance.
    assert "alpha_y_reference_run" not in payload
    # A manual edit invalidates the provenance for that axis.
    dialog._vector_alpha_spins["P_x"].setValue(estimated + 0.5)
    payload2 = dialog._current_grouping_payload()
    assert "alpha_x_reference_run" not in payload2


def test_vector_estimate_alpha_for_axis_updates_axis_spin(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("P_x")

    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(2.0)


def test_vector_estimate_alpha_uses_selected_reference_run(qapp: QApplication) -> None:
    ds_a = _vector_dataset_with_ratio(5201, ratio=2.0)
    ds_b = _vector_dataset_with_ratio(5202, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5202)
    dialog._estimate_alpha_for_axis("P_x")

    # Must use selected reference run (5202 => alpha=4), not an average of runs.
    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(4.0)


def test_estimate_alpha_uses_reference_run_only(qapp: QApplication) -> None:
    ds_a = _dataset_with_ratio(5001, ratio=2.0)
    ds_b = _dataset_with_ratio(5002, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5002)
    dialog._estimate_alpha()

    # Must use selected reference run (5002 => alpha=4), not an average of runs.
    assert dialog._alpha_spin.value() == pytest.approx(4.0)


def test_preselected_run_numbers_set_dataset_tickboxes(qapp: QApplication) -> None:
    ds_a = _dataset_with_ratio(5101, ratio=2.0)
    ds_b = _dataset_with_ratio(5102, ratio=3.0)

    dialog = GroupingDialog(
        [ds_a, ds_b],
        selected_run_number=5101,
        selected_run_numbers=[5102],
    )

    assert dialog._checked_run_numbers() == [5102]


def test_pressing_enter_on_bunch_factor_does_not_estimate_alpha(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    assert dialog._alpha_spin.value() == pytest.approx(1.0)

    dialog._bunch_spin.setFocus()
    dialog._bunch_spin.setValue(7)
    QTest.keyClick(dialog._bunch_spin, Qt.Key.Key_Return)

    # Enter on bunching should not trigger the Estimate alpha action.
    assert dialog._alpha_spin.value() == pytest.approx(1.0)


def test_pressing_enter_on_bunch_factor_does_not_trigger_save_grp(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    save_called = {"value": False}

    def _stub_get_save_file_name(*_args, **_kwargs):
        save_called["value"] = True
        return "", ""

    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.dialog.QFileDialog.getSaveFileName",
        _stub_get_save_file_name,
    )

    dialog._bunch_spin.setFocus()
    dialog._bunch_spin.setValue(9)
    QTest.keyClick(dialog._bunch_spin, Qt.Key.Key_Return)

    assert save_called["value"] is False


def test_grp_round_trip_parser_and_serializer() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.2345,
        "t0_bin": 2,
        "t_good_offset": 7,
        "first_good_bin": 9,
        "last_good_bin": 2048,
        "bunching_factor": 5,
        "deadtime_correction": True,
        "background_correction": True,
        "period_mode": str(PeriodMode.GREEN_PLUS_RED),
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["groups"][1] == [1, 2]
    assert parsed["groups"][2] == [3, 4]
    assert parsed["forward_group"] == 1
    assert parsed["backward_group"] == 2
    assert parsed["alpha"] == pytest.approx(1.2345)
    assert parsed["t0_bin"] == 2
    assert parsed["t_good_offset"] == 7
    assert parsed["first_good_bin"] == 9
    assert parsed["last_good_bin"] == 2048
    assert parsed["bunching_factor"] == 5
    assert parsed["deadtime_correction"] is True
    assert parsed["background_correction"] is True
    assert parsed["period_mode"] == str(PeriodMode.GREEN_PLUS_RED)


def test_grp_round_trip_preserves_deadtime_mode_metadata() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "deadtime_correction": True,
        "deadtime_mode": "manual",
        "deadtime_method": "manual",
        "deadtime_manual_us": 0.025,
        "dead_time_us": [0.025, 0.025],
    }

    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["deadtime_correction"] is True
    assert parsed["deadtime_mode"] == "manual"
    assert parsed["deadtime_method"] == "manual"
    assert parsed["deadtime_manual_us"] == pytest.approx(0.025)
    assert parsed["dead_time_us"] == pytest.approx([0.025, 0.025])


def test_grp_round_trip_parser_and_serializer_with_vector_alphas() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.2345,
        "alpha_x": 1.1,
        "alpha_y": 1.2,
        "alpha_z": 1.3,
        "first_good_bin": 9,
        "last_good_bin": 2048,
        "bunching_factor": 5,
        "deadtime_correction": True,
        "period_mode": str(PeriodMode.GREEN_PLUS_RED),
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["alpha_x"] == pytest.approx(1.1)
    assert parsed["alpha_y"] == pytest.approx(1.2)
    assert parsed["alpha_z"] == pytest.approx(1.3)


def test_parse_grp_legacy_first_good_derives_t_good_offset() -> None:
    text = "\n".join(
        [
            "forward_group=1",
            "backward_group=2",
            "t0_bin=3",
            "first_good_bin=11",
            "last_good_bin=50",
            "group.1=1",
            "group.2=2",
        ]
    )
    parsed = GroupingDialog.parse_grp(text)
    assert parsed["t0_bin"] == 3
    assert parsed["first_good_bin"] == 11
    assert parsed["t_good_offset"] == 8


def test_current_payload_uses_t0_and_t_good_offset(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._t0_spin.setValue(1)
    dialog._t_good_offset_spin.setValue(2)
    dialog._last_good_spin.setValue(3)

    payload = dialog._current_grouping_payload()

    assert payload["t0_bin"] == 1
    assert payload["t_good_offset"] == 2
    assert payload["first_good_bin"] == 3
    assert payload["last_good_bin"] == 3


def test_one_based_bin_index_base_displays_file_facing_t0(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.histograms[0].t0_bin = 1
    dataset.run.grouping["t0_bin"] = 1
    dataset.run.grouping["first_good_bin"] = 3
    dataset.run.grouping["last_good_bin"] = 3
    dataset.run.grouping["bin_index_base"] = 1

    dialog = GroupingDialog([dataset])

    # Display should match one-based file metadata while payload stays internal.
    assert dialog._t0_spin.value() == 2
    payload = dialog._current_grouping_payload()
    assert payload["t0_bin"] == 1
    assert payload["bin_index_base"] == 1


def test_period_mode_row_visible_for_two_period_reference(qapp: QApplication) -> None:
    dialog = GroupingDialog([_two_period_dataset()])
    assert not dialog._period_mode_label.isHidden()
    assert not dialog._period_mode_widget.isHidden()
    assert dialog._current_period_mode() == str(PeriodMode.GREEN)


# ---------------------------------------------------------------------------
# group_names in payload
# ---------------------------------------------------------------------------


def test_current_grouping_payload_contains_group_names_key(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    payload = dialog._current_grouping_payload()
    assert "group_names" in payload


def test_current_grouping_payload_group_names_reflects_state(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._group_names = {1: "Forward", 2: "Backward"}
    payload = dialog._current_grouping_payload()
    assert payload["group_names"] == {1: "Forward", 2: "Backward"}


def test_current_grouping_payload_contains_grouping_preset(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._grouping_preset_name = "Longitudinal"
    payload = dialog._current_grouping_payload()
    assert payload["grouping_preset"] == "Longitudinal"


def test_current_grouping_payload_contains_instrument(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "MuSR"

    dialog = GroupingDialog([dataset])
    payload = dialog._current_grouping_payload()

    assert payload["instrument"] == "MuSR"


def test_psi_detector_convention_defaults_are_swapped_in_grouping_dropdowns(
    qapp: QApplication,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 1
    dataset.run.grouping["backward_group"] = 2

    dialog = GroupingDialog([dataset])

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


def test_psi_already_swapped_grouping_dropdowns_are_preserved(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 2
    dataset.run.grouping["backward_group"] = 1

    dialog = GroupingDialog([dataset])

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


def test_beam_direction_label_recognises_single_letter_and_compound_names() -> None:
    """Defence in depth: the swap classifier reads F/B and compound spin-rotator
    names, not only the spelled-out Forward/Backward group names."""
    fn = GroupingDialog._beam_direction_label
    assert fn("Forward") == "forward"
    assert fn("Backward") == "backward"
    assert fn("F") == "forward"
    assert fn("B") == "backward"
    assert fn("F+D") == "forward"  # compound: leading beam-axis letter wins
    assert fn("B+U") == "backward"
    # Non-beam directions and unrelated names never classify as F/B.
    assert fn("Up") is None
    assert fn("Down") is None
    assert fn("Left") is None
    assert fn("Right") is None
    assert fn("") is None


def test_applying_fixed_gps_preset_is_not_re_swapped(qapp: QApplication) -> None:
    """Exactly-once: a GPS preset already declares its analysis slots (the
    Backward-named group in the analysis-forward slot), so the PSI beam->analysis
    swap in the dialog must NOT fire again when that preset is applied on a PSI
    reference. Applying the preset's declared pair leaves it unchanged."""
    from asymmetry.core.instrument import get_instrument_layout

    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"

    dialog = GroupingDialog([dataset])
    assert dialog._reference_is_psi()

    layout = get_instrument_layout("GPS")
    lon = layout.presets["Longitudinal"]
    # Mirror the detector-layout editor's group_names into the dialog so the swap
    # classifier sees the physical F/B names the preset carries.
    dialog._group_names = {gid: gdef.name for gid, gdef in lon.groups.items()}

    # The preset already declares analysis-forward = the Backward-named group.
    assert lon.groups[lon.forward_group].name == "Backward"

    # Applying the preset's declared pair (the payload path at _on_apply calls
    # _analysis_pair_for_reference on result["forward_group"]/["backward_group"]).
    result_fwd, result_bwd = lon.forward_group, lon.backward_group
    once_fwd, once_bwd = dialog._analysis_pair_for_reference(result_fwd, result_bwd)
    # No swap: the forward slot does not hold a forward-NAMED group.
    assert (once_fwd, once_bwd) == (result_fwd, result_bwd)

    # Idempotent under a second pass (proves it cannot drift on re-entry).
    twice = dialog._analysis_pair_for_reference(once_fwd, once_bwd)
    assert twice == (once_fwd, once_bwd)


def test_detector_layout_prefers_saved_instrument(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "MuSR"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.dialog.detect_instrument",
        lambda *_args, **_kwargs: "EMU",
    )
    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.dialog.get_instrument_layout",
        lambda name: type("_Layout", (), {"name": name})(),
    )
    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "MuSR"


def test_detector_layout_detects_flame_from_psi_metadata(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "FLAME"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "FLAME"


def test_detector_layout_retries_detection_when_saved_instrument_is_generic_psi(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "PSI"
    dataset.run.grouping["histogram_labels"] = [
        "Forw",
        "Back",
        "Righ",
        "Left",
        "R_F",
        "R_B",
        "L_F",
        "L_B",
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "PSI"
    dataset.run.histograms = dataset.run.histograms * 4

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "FLAME"


def test_detector_layout_corrects_gps_variant_to_histogram_count(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    # Stored as the 6-detector BIN variant, but the data is an 11-histogram ROOT
    # run: the layout editor must open on the matching GPS-RD variant, not the
    # stale stored "GPS".
    dataset.run.grouping["instrument"] = "GPS"
    dataset.run.grouping["histogram_labels"] = [
        "Forw",
        "Back",
        "Up_B",
        "Up_F",
        "Down_B",
        "Down_F",
        "Right_B",
        "Right_F",
        "Left_B",
        "Left_F",
        "Mob-RL",
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "LMU_BULKMUSR_GPS"
    dataset.run.histograms = [dataset.run.histograms[0]] * 11

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "GPS-RD"


def test_detector_layout_resolves_psi_hifi_to_hal(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The PSI loader stores instrument="HIFI" for HAL-9500 runs; this string
    # canonicalises to the unrelated ISIS HiFi layout, so it must be re-detected
    # to HAL for PSI data. Uses the real detector/layout functions.
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["psi_format"] = "psi-mdu"
    dataset.run.metadata["instrument"] = "HIFI"
    dataset.run.grouping["instrument"] = "HIFI"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 1  # Accepted

        def get_result(self):
            return {"instrument": captured["instrument"], "groups": {}, "group_names": {}}

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    # The editor is seeded with the HAL layout (not the unrelated ISIS HiFi),
    assert captured["instrument"] == "HAL"
    # and on Accept the corrected layout name is committed, so it does not revert
    # to the raw "HIFI" string on subsequent opens.
    assert dialog._detector_layout_instrument_name == "HAL"


def test_detector_layout_cancel_does_not_change_stored_instrument(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cancelling the editor must be a no-op for the stored instrument selection.
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["psi_format"] = "psi-mdu"
    dataset.run.metadata["instrument"] = "HIFI"
    dataset.run.grouping["instrument"] = "HIFI"

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 0  # Rejected / cancelled

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    before = dialog._detector_layout_instrument_name
    dialog._on_detector_layout()
    assert dialog._detector_layout_instrument_name == before


def test_psi_detector_layout_result_is_swapped_for_analysis_dropdowns(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "FLAME"
    dataset.run.grouping["instrument"] = "FLAME"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 1
    dataset.run.grouping["backward_group"] = 2

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return {
                "groups": {1: [1], 2: [2]},
                "group_names": {1: "Forward", 2: "Backward"},
                "forward_group": 1,
                "backward_group": 2,
                "instrument": "FLAME",
                "grouping_preset": "Longitudinal",
            }

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


# ---------------------------------------------------------------------------
# .grp format: group_name round-trip
# ---------------------------------------------------------------------------


def test_grp_round_trip_with_group_names() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "included_groups": {1: True, 2: False},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 2048,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "group_names": {1: "Forward", 2: "Backward"},
    }
    text = GroupingDialog.serialize_grp(payload)
    assert "group_name.1=Forward" in text
    assert "group_name.2=Backward" in text
    assert "group_include.1=1" in text
    assert "group_include.2=0" in text
    parsed = GroupingDialog.parse_grp(text)
    assert parsed.get("group_names", {}).get(1) == "Forward"
    assert parsed.get("group_names", {}).get(2) == "Backward"
    assert parsed.get("included_groups", {}).get(1) is True
    assert parsed.get("included_groups", {}).get(2) is False


def test_grp_round_trip_without_group_names_backwards_compat() -> None:
    """Old .grp files without group_name lines must still parse cleanly."""
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 2048,
        "bunching_factor": 1,
        "deadtime_correction": False,
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)
    # Should produce empty dict, not raise
    assert parsed.get("group_names", {}) == {}


def test_serialize_grp_no_group_names_no_spurious_lines() -> None:
    """serialize_grp with empty group_names must not emit group_name lines."""
    payload = {
        "groups": {1: [1]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 10,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "group_names": {},
    }
    text = GroupingDialog.serialize_grp(payload)
    assert "group_name" not in text


# ---------------------------------------------------------------------------
# "Detector Layout…" button exists in the GroupingDialog UI
# ---------------------------------------------------------------------------


def test_detector_layout_button_exists(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    # The button is connected to _on_detector_layout; verify it is present.
    from PySide6.QtWidgets import QPushButton

    buttons = dialog.findChildren(QPushButton)
    labels = [b.text() for b in buttons]
    assert any("Detector Layout" in lbl for lbl in labels)


def test_group_table_uses_scrollable_capped_height(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["groups"] = {idx: [idx] for idx in range(1, 11)}
    dialog = GroupingDialog([dataset])

    assert dialog._group_table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert dialog._group_table.maximumHeight() > 0


# ---------------------------------------------------------------------------
# Alpha estimation method picker + provenance (data-reduction-parity Phase 1)
# ---------------------------------------------------------------------------


def test_alpha_method_combo_defaults_to_diamagnetic(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    assert dialog._current_alpha_method() == "diamagnetic"
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "diamagnetic"


def test_alpha_method_round_trips_through_payload_and_reload(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("ratio")
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "ratio"

    dataset.run.grouping["alpha_method"] = "general"
    dialog2 = GroupingDialog([dataset])
    assert dialog2._current_alpha_method() == "general"


def test_estimate_records_provenance_in_payload(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    assert dialog._alpha_spin.value() == pytest.approx(2.0)
    assert dialog._alpha_result_label.text() != ""
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "diamagnetic"
    assert result["alpha_reference_run"] == 4001
    # Bootstrap error from flat 100/50 counts is small but present.
    assert result.get("alpha_error") is None or result["alpha_error"] >= 0.0


def test_manual_alpha_edit_invalidates_estimate_provenance(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    dialog._alpha_spin.setValue(dialog._alpha_spin.value() + 0.5)
    result = dialog.get_grouping_result()
    assert "alpha_error" not in result
    assert "alpha_reference_run" not in result
    assert result["alpha_method"] == "diamagnetic"


def test_estimate_failure_warns_and_leaves_alpha(qapp: QApplication, monkeypatch) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("general")  # flat 4-bin data: no relaxation contrast
    warnings: list[str] = []
    monkeypatch.setattr(
        grouping_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    before = dialog._alpha_spin.value()
    dialog._estimate_alpha()
    assert warnings, "expected a warning dialog for the failed estimate"
    assert dialog._alpha_spin.value() == pytest.approx(before)


def test_format_value_with_uncertainty() -> None:
    fmt = grouping_dialog_module._format_value_with_uncertainty
    assert fmt(1.2345, 0.0067) == "1.2345(67)"
    assert fmt(1.37, None) == "1.3700"
    assert fmt(0.9876, 0.05) == "0.988(50)"
    assert fmt(1.3, 0.0995) == "1.30(10)"


def test_tail_fit_mode_shows_preview_status(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    # Long histograms so the tail fit has a usable window.
    rng = np.random.default_rng(0)
    counts = rng.poisson(np.full(400, 50.0)).astype(float)
    dataset.run.histograms = [
        Histogram(counts=counts, bin_width=0.016),
        Histogram(counts=counts * 0.8, bin_width=0.016),
    ]
    dataset.run.grouping["last_good_bin"] = 399
    dialog = GroupingDialog([dataset])
    idx = dialog._background_mode_combo.findData("tail_fit")
    dialog._background_mode_combo.setCurrentIndex(idx)
    assert "Tail-fit background" in dialog._background_status_label.text()
    result = dialog.get_grouping_result()
    assert result["background_mode"] == "tail_fit"
    assert result["background_correction"] is True


def test_background_run_payload_round_trips(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["background_correction"] = True
    dataset.run.grouping["background_mode"] = "reference_run"
    dataset.run.grouping["background_run"] = {"run_number": 9001, "source_file": "/tmp/x.nxs"}
    dialog = GroupingDialog([dataset])
    assert dialog._current_background_mode() == "reference_run"
    result = dialog.get_grouping_result()
    assert result["background_run"]["run_number"] == 9001
    assert "9001" in dialog._background_status_label.text()


# ---------------------------------------------------------------------------
# Binning modes, Find t0, detector exclusion (data-reduction-parity Phase 3)
# ---------------------------------------------------------------------------


def test_binning_mode_round_trips_through_payload(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    assert dialog._current_binning_mode() == "fixed"
    assert "binning_mode" not in dialog.get_grouping_result()

    dialog._set_binning_mode("variable")
    dialog._bin0_spin.setValue(0.1)
    dialog._bin10_spin.setValue(0.5)
    result = dialog.get_grouping_result()
    assert result["binning_mode"] == "variable"
    assert result["bin0_us"] == pytest.approx(0.1)
    assert result["bin10_us"] == pytest.approx(0.5)
    assert not dialog._bunch_spin.isEnabled()

    dataset.run.grouping.update(result)
    dialog2 = GroupingDialog([dataset])
    assert dialog2._current_binning_mode() == "variable"
    assert dialog2._bin0_spin.value() == pytest.approx(0.1)


def test_constant_error_mode_hides_bin10(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._set_binning_mode("constant_error")
    result = dialog.get_grouping_result()
    assert result["binning_mode"] == "constant_error"
    assert "bin10_us" not in result
    assert not dialog._bin10_spin.isEnabled()


def test_find_t0_fills_spinner_without_applying(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    rng = np.random.default_rng(0)
    counts = np.full(200, 5.0)
    counts[37] = 5000.0
    counts[38:] = 100.0
    dataset.run.histograms = [
        Histogram(counts=rng.poisson(counts).astype(float), bin_width=0.00125),
        Histogram(counts=rng.poisson(counts).astype(float), bin_width=0.00125),
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.metadata["facility"] = "PSI"
    dataset.run.grouping["last_good_bin"] = 199
    dialog = GroupingDialog([dataset])
    dialog._on_find_t0()
    assert dialog._t0_spin.value() == 37 + dialog._bin_index_base()
    assert "t0" in dialog._alpha_result_label.text()


def test_exclusion_field_round_trips_and_validates(qapp: QApplication, monkeypatch) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    dataset.run.histograms = [Histogram(counts=np.full(4, 100.0), bin_width=0.01) for _ in range(4)]
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("2")
    result = dialog.get_grouping_result()
    assert result["excluded_detectors"] == [2]

    dialog._exclude_edit.setText("")
    # The key is always present: an empty list explicitly clears exclusions
    # (a missing key would leave the apply path falling back to stale state).
    assert dialog.get_grouping_result()["excluded_detectors"] == []

    warnings: list[str] = []
    monkeypatch.setattr(
        grouping_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dialog._exclude_edit.setText("nonsense")
    assert dialog._current_excluded_detectors() is None
    assert warnings


def test_apply_blocks_when_exclusion_empties_a_group(qapp: QApplication, monkeypatch) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("1")  # forward group is exactly detector 1
    warnings: list[str] = []
    monkeypatch.setattr(
        grouping_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dialog._on_apply()
    assert any("no detectors left" in w for w in warnings)


def test_estimate_alpha_respects_detector_exclusion(qapp: QApplication) -> None:
    """Review fix: the estimate is computed on the same detector set the
    reduction will use."""
    dataset = _dataset_with_histograms()
    dataset.run.histograms = [
        Histogram(counts=np.full(4, 100.0), bin_width=0.01),  # det 1 (forward)
        Histogram(counts=np.full(4, 900.0), bin_width=0.01),  # det 2 (forward, hot)
        Histogram(counts=np.full(4, 50.0), bin_width=0.01),  # det 3 (backward)
        Histogram(counts=np.full(4, 50.0), bin_width=0.01),  # det 4 (backward)
    ]
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("ratio")

    dialog._estimate_alpha()
    with_hot = dialog._alpha_spin.value()
    dialog._exclude_edit.setText("2")
    dialog._estimate_alpha()
    without_hot = dialog._alpha_spin.value()

    assert with_hot == pytest.approx(1000.0 / 100.0)
    assert without_hot == pytest.approx(100.0 / 100.0)


def test_find_t0_skips_excluded_detectors(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    good = np.full(200, 5.0)
    good[40] = 5000.0
    bad = np.full(200, 5.0)
    bad[120] = 5000.0  # excluded detector with a bogus peak
    dataset.run.histograms = [
        Histogram(counts=good.copy(), bin_width=0.00125),
        Histogram(counts=bad.copy(), bin_width=0.00125),
        Histogram(counts=good.copy(), bin_width=0.00125),
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.metadata["facility"] = "PSI"
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3]}
    dataset.run.grouping["last_good_bin"] = 199
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("2")
    dialog._on_find_t0()
    assert dialog._t0_spin.value() == 40 + dialog._bin_index_base()


def test_format_value_with_uncertainty_large_errors() -> None:
    """Review fix: uncertainties >= ~100 must not crash the formatter."""
    fmt = grouping_dialog_module._format_value_with_uncertainty
    assert fmt(1.2345, 150.0) == "1(150)"
    assert fmt(1.2, 99.6) == "1(100)"
    assert fmt(2.4, 1234.0) == "2(1234)"


def _gps_dataset(field_direction: str | None) -> MuonDataset:
    """A six-histogram PSI GPS run defaulting to the Longitudinal grouping."""
    histograms = [Histogram(counts=np.full(4, 100.0), bin_width=0.01) for _ in range(6)]
    metadata = {"run_number": 5001, "facility": "PSI", "instrument": "GPS"}
    if field_direction is not None:
        metadata["field_direction"] = field_direction
    run = Run(
        run_number=5001,
        histograms=histograms,
        metadata=dict(metadata),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata=dict(metadata),
        run=run,
    )


def test_transverse_field_gps_run_shows_grouping_nudge(qapp) -> None:
    """B8a: a TF GPS run on the longitudinal default nudges toward spin-rotated."""
    dialog = GroupingDialog([_gps_dataset("Transverse")])
    assert dialog._tf_hint_label.isVisibleTo(dialog)
    assert "Spin-rotated (B+U/F+D)" in dialog._tf_hint_label.text()


def test_gps_run_without_field_direction_does_not_nudge(qapp) -> None:
    dialog = GroupingDialog([_gps_dataset(None)])
    assert not dialog._tf_hint_label.isVisibleTo(dialog)
