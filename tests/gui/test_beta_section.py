"""Tests for the β (asymmetry balance) card in the grouping Corrections editor.

Pins the GUI items of ``docs/porting/beta-correction/verification-plan.md``:
card wiring + payload round-trip (emit-only-when-≠1), dirty marking, the
β compare stage (availability, payload invariance, pager stop), and the
scalar-only vector-mode behaviour (card hidden, payload omits the key).
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

import asymmetry.gui.windows.grouping.beta_section as beta_section_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.beta_calibration import BetaEstimate
from asymmetry.core.project.profiles import AlphaPolicy, BetaPolicy
from asymmetry.gui.styles import tokens
from asymmetry.gui.windows.grouping.beta_section import (
    BETA_PROTOCOL_ITEMS,
    BetaSectionWidget,
    beta_status_text,
    format_value_with_pm,
)
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


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
    assert predicate(), "timed out waiting for the beta estimate"


def _fake_estimate(**overrides) -> BetaEstimate:
    defaults = dict(
        beta=0.8732,
        beta_error=0.0041,
        alpha=1.1520,
        alpha_error=0.0081,
        alpha_beta_correlation=-0.3,
        method="count_fit",
        n_bins_used=100,
        reduced_chi2=1.1,
        ok=True,
        message="",
    )
    defaults.update(overrides)
    return BetaEstimate(**defaults)


def _patch_estimate(monkeypatch, estimate: BetaEstimate) -> None:
    """Patch ``estimate_beta_detailed`` where the β worker looks it up."""
    monkeypatch.setattr(beta_section_module, "estimate_beta_detailed", lambda *a, **k: estimate)


def _calib_run(run_number: int = 4101) -> MuonDataset:
    forward = np.full(4, 100.0)
    backward = np.full(4, 50.0)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=forward, bin_width=0.01),
            Histogram(counts=backward, bin_width=0.01),
        ],
        metadata={"run_number": run_number, "title": "TF calib", "field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.arange(4) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _beta_context() -> dict:
    return {
        "groups": {1: [0], 2: [1]},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [],
        "correction_provider": None,
        "reference_resolver": None,
        "facility": "",
    }


def _configure_beta(section: BetaSectionWidget, datasets, *, applied_alpha=1.0) -> None:
    section.configure(
        datasets=datasets,
        selected_run_number=datasets[0].run_number,
        context_provider=_beta_context,
        current_alpha_provider=lambda: applied_alpha,
    )


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset(*, beta: float | None = None, run_number: int = 4101) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    grouping: dict = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 3,
    }
    if beta is not None:
        grouping["beta"] = beta
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Beta Test"},
        grouping=grouping,
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _vector_dataset(run_number: int = 4110) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Vector Beta Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2], 5: [1], 6: [2]},
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


# --------------------------------------------------------------------------- #
# Widget unit behaviour
# --------------------------------------------------------------------------- #


def test_beta_widget_value_roundtrip_and_leniency(qapp: QApplication) -> None:
    widget = BetaSectionWidget()
    assert widget.value() == 1.0
    assert not widget.is_active()
    widget.set_value(0.9)
    assert widget.value() == pytest.approx(0.9)
    assert widget.is_active()
    for degenerate in (float("nan"), float("inf"), 0.0, -3.0, "x", None):
        widget.set_value(degenerate)
        assert widget.value() == 1.0


def test_beta_status_text_format() -> None:
    assert beta_status_text(1.0) == "β = 1.0000"
    assert beta_status_text(0.87654321) == "β = 0.8765"


# --------------------------------------------------------------------------- #
# Dialog wiring
# --------------------------------------------------------------------------- #


def test_beta_card_registered_and_default_payload_omits_beta(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    assert "beta" in dialog._correction_cards
    assert dialog._correction_cards["beta"] is dialog._beta_card
    # Do-nothing default: the payload stays byte-identical to a pre-β one.
    assert "beta" not in dialog._current_grouping_payload()
    assert not dialog._correction_stage_active("beta")


def test_beta_edit_reaches_payload_and_marks_dirty(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    dialog._clear_dirty()
    dialog._beta_section.set_value(0.9)
    payload = dialog._current_grouping_payload()
    assert payload["beta"] == pytest.approx(0.9)
    assert dialog._draft_dirty
    assert dialog._correction_stage_active("beta")
    # Back to the default removes the key again.
    dialog._beta_section.set_value(1.0)
    assert "beta" not in dialog._current_grouping_payload()


def test_beta_seeded_from_run_grouping(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=1.2)])
    assert dialog._beta_section.value() == pytest.approx(1.2)
    assert dialog._current_grouping_payload()["beta"] == pytest.approx(1.2)


# --------------------------------------------------------------------------- #
# Compare stage
# --------------------------------------------------------------------------- #


def test_beta_compare_available_only_when_active(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset()])
    assert not dialog._compare_stage_available("beta")
    dialog._beta_section.set_value(0.9)
    assert dialog._compare_stage_available("beta")
    # β ≠ 1 alone also makes the compound "vs raw" compare meaningful.
    assert dialog._compare_stage_available("raw")


def test_beta_compare_never_touches_the_persisted_payload(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=0.9)])
    before = dialog._current_grouping_payload()
    dialog._set_compare_stage("beta")
    assert dialog._compare_stage == "beta"
    assert dialog._current_grouping_payload() == before
    dialog._set_compare_stage(None)
    assert dialog._current_grouping_payload() == before


def test_beta_is_a_pager_stop_in_pipeline_order(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset(beta=0.9)])
    dialog._set_compare_stage(None)
    seen: list[str | None] = []
    for _ in range(6):
        dialog._step_compare(1)
        seen.append(dialog._compare_stage)
    assert "beta" in seen
    # β follows α (which is unavailable here at α = 1), i.e. β precedes "raw".
    assert seen.index("beta") < seen.index("raw")


# --------------------------------------------------------------------------- #
# Vector mode (scalar-only)
# --------------------------------------------------------------------------- #


def test_beta_hidden_and_omitted_in_vector_mode(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset()])
    assert bool(dialog._vector_axis_pairs)
    dialog._beta_section.set_value(0.9)
    assert dialog._beta_card.isHidden()
    assert not dialog._compare_stage_available("beta")
    assert not dialog._correction_stage_active("beta")
    assert "beta" not in dialog._current_grouping_payload()


# --------------------------------------------------------------------------- #
# Measure-from-run section
# --------------------------------------------------------------------------- #


def test_protocol_combo_maps_labels_to_method_ids(qapp: QApplication) -> None:
    section = BetaSectionWidget()
    keys = [section._protocol_combo.itemData(i) for i in range(section._protocol_combo.count())]
    assert keys == ["count_fit", "single_histogram"]
    assert section._current_method() == "count_fit"  # count fit is the default
    labels = [label for label, _, _ in BETA_PROTOCOL_ITEMS]
    assert labels == ["Count fit (recommended)", "Single-histogram ratio"]
    section.shutdown()


def test_format_value_with_pm() -> None:
    assert format_value_with_pm(0.8732, 0.0041) == "0.8732 ± 0.0041"
    assert format_value_with_pm(0.87, None) == "0.8700"
    assert format_value_with_pm(0.87, 0.0) == "0.8700"


def test_estimate_flow_shows_result_and_enables_apply(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BetaSectionWidget()
    _configure_beta(section, [_calib_run(5)])
    _patch_estimate(monkeypatch, _fake_estimate())
    assert not section._apply_btn.isEnabled()  # nothing to apply yet
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    text = section._result_label.text()
    assert "β = " in text and "0.8732 ± 0.0041" in text
    assert "fitted α = " in text and "1.1520 ± 0.0081" in text
    assert section._apply_btn.isEnabled()
    section.shutdown()


def test_apply_emits_calibrated_beta_policy(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BetaSectionWidget()
    _configure_beta(section, [_calib_run(7)])
    _patch_estimate(monkeypatch, _fake_estimate())
    betas: list[BetaPolicy] = []
    alphas: list[AlphaPolicy] = []
    section.beta_estimated.connect(lambda p: betas.append(p))
    section.alpha_estimated.connect(lambda p: alphas.append(p))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)

    section._on_apply()
    assert len(betas) == 1
    policy = betas[-1]
    assert policy.mode == "calibrated"
    assert policy.value == pytest.approx(0.8732)
    assert policy.error == pytest.approx(0.0041)
    assert policy.method == "count_fit"
    assert policy.source_run == 7
    assert not alphas  # β only unless "also update α" is ticked
    section.shutdown()


def test_also_update_alpha_emits_count_fit_alpha_policy_for_both_protocols(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    for method in ("count_fit", "single_histogram"):
        section = BetaSectionWidget()
        _configure_beta(section, [_calib_run(9)])
        idx = section._protocol_combo.findData(method)
        section._protocol_combo.setCurrentIndex(idx)
        # single_histogram carries no α–β correlation; the α is still fitted.
        _patch_estimate(monkeypatch, _fake_estimate(method=method))
        alphas: list[AlphaPolicy] = []
        section.alpha_estimated.connect(lambda p: alphas.append(p))
        section._on_estimate()
        _wait_until(lambda: section._tasks.active_count == 0)
        section._also_alpha_check.setChecked(True)
        section._on_apply()
        assert len(alphas) == 1
        assert alphas[-1].mode == "calibrated"
        assert alphas[-1].value == pytest.approx(1.1520)
        # Both protocols stamp "count_fit" — "single_histogram" is not in α's vocab.
        assert alphas[-1].method == "count_fit"
        section.shutdown()


def test_failed_estimate_shows_message_and_no_number(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BetaSectionWidget()
    _configure_beta(section, [_calib_run(3)])
    _patch_estimate(
        monkeypatch,
        _fake_estimate(ok=False, message="No precession found — β is degenerate."),
    )
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    text = section._result_label.text()
    assert text.startswith("Estimate failed:")
    assert "degenerate" in text
    assert "β = " not in text
    assert not section._apply_btn.isEnabled()
    section.shutdown()


def test_beta_outside_typical_range_is_flagged(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BetaSectionWidget()
    _configure_beta(section, [_calib_run(11)])
    _patch_estimate(monkeypatch, _fake_estimate(beta=0.30, beta_error=0.01))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert "typical 0.5–1.5 range" in section._note_label.text()
    # The β value is warn-tinted in the result row.
    assert tokens.WARN in section._result_label.text()
    section.shutdown()


def test_alpha_consistency_warns_beyond_three_sigma(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BetaSectionWidget()
    # Applied α = 1.0, fitted α = 1.152 ± 0.008 → 19σ away → warn.
    _configure_beta(section, [_calib_run(13)], applied_alpha=1.0)
    _patch_estimate(monkeypatch, _fake_estimate(alpha=1.152, alpha_error=0.008))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert "Fitted α disagrees with the applied α" in section._note_label.text()

    # Applied α close to fitted α → no disagreement hint.
    section2 = BetaSectionWidget()
    _configure_beta(section2, [_calib_run(14)], applied_alpha=1.150)
    _patch_estimate(monkeypatch, _fake_estimate(alpha=1.152, alpha_error=0.008))
    section2._on_estimate()
    _wait_until(lambda: section2._tasks.active_count == 0)
    assert "disagrees" not in section2._note_label.text()
    section.shutdown()
    section2.shutdown()
