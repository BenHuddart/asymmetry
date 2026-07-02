"""Tests for the fit-wizard scope selector widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.gui.styles import tokens  # noqa: E402
from asymmetry.gui.widgets.wizard_scope_selector import (  # noqa: E402
    PRESET_CHOICES,
    WizardScopeSelector,
    build_scope_payload,
    parse_scope_payload,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FakeResolver:
    """Records calls and returns canned resolutions keyed by preset id.

    Reflects overrides back into the estimate and note so tests can observe the
    label updating, but leaves ``families`` as the fixed baseline (the widget
    ignores the override-resolution's families by contract).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _families(self, preset: str) -> list[dict]:
        return [
            {
                "key": "precession",
                "title": "Precession",
                "components": [
                    {"name": "TFCos", "included": True, "reason": "", "cost": "1 fit"},
                    {"name": "TFGauss", "included": True, "reason": "", "cost": "1 fit"},
                ],
            },
            {
                "key": "relaxation",
                "title": "Relaxation",
                "components": [
                    {"name": "Exponential", "included": True, "reason": "", "cost": "1 fit"},
                    {
                        "name": "Muonium",
                        "included": False,
                        "reason": "Not indicated by run metadata",
                        "cost": "2 fits",
                    },
                ],
            },
        ]

    def __call__(self, preset_id: str, overrides: dict) -> dict:
        self.calls.append((preset_id, dict(overrides)))
        included = 3 + len(overrides.get("include", [])) - len(overrides.get("exclude", []))
        has_overrides = bool(overrides.get("include") or overrides.get("exclude"))
        note = (
            f"Custom scope (from {preset_id})" if has_overrides else f"{preset_id} — baseline note"
        )
        return {
            "effective_preset": "tf-knight-precession" if preset_id == "auto" else preset_id,
            "note": note,
            "families": self._families(preset_id),
            "estimate": [max(included, 0), max(included, 0) * 2],
        }


def _make(qapp: QApplication) -> tuple[WizardScopeSelector, FakeResolver]:
    resolver = FakeResolver()
    widget = WizardScopeSelector()
    widget.set_resolver(resolver)
    widget.refresh_from_context()
    return widget, resolver


def _leaf(widget: WizardScopeSelector, name: str):
    for leaf in widget._iter_leaves():
        if leaf.data(0, Qt.ItemDataRole.UserRole) == name:
            return leaf
    raise AssertionError(f"leaf {name!r} not found")


# ── Payload helpers ──────────────────────────────────────────────────────────


def test_build_and_parse_payload_round_trip() -> None:
    payload = build_scope_payload("auto", {"b", "a"}, {"z"})
    assert payload == {"version": 1, "preset": "auto", "include": ["a", "b"], "exclude": ["z"]}
    assert parse_scope_payload(payload) == ("auto", {"a", "b"}, {"z"})


def test_parse_payload_none_and_unknown_fall_back_to_auto() -> None:
    assert parse_scope_payload(None) == ("auto", set(), set())
    assert parse_scope_payload({"preset": "nonsense"}) == ("auto", set(), set())


# ── Initial refresh ──────────────────────────────────────────────────────────


def test_initial_refresh_populates_combo_tree_and_labels(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)

    assert widget._preset_combo.count() == len(PRESET_CHOICES)
    assert widget._preset_combo.currentData() == "auto"
    # Two families, four components.
    assert widget._family_tree.topLevelItemCount() == 2
    assert widget._included_leaf_count() == 3
    assert "baseline note" in widget._metadata_label.text()
    assert "3 candidates / 6 screening fits" in widget._estimate_label.text()
    assert widget.is_valid() is True


# ── Preset change ────────────────────────────────────────────────────────────


def test_preset_change_repopulates_and_emits_scope_changed_once(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    emitted: list[object] = []
    widget.scope_changed.connect(emitted.append)

    idx = widget._preset_combo.findData("lf-dynamics")
    widget._preset_combo.setCurrentIndex(idx)

    assert len(emitted) == 1
    assert emitted[0]["preset"] == "lf-dynamics"
    assert emitted[0]["include"] == []
    assert emitted[0]["exclude"] == []
    assert widget._preset_id == "lf-dynamics"


# ── Uncheck an included component → Custom + override round-trips ─────────────


def test_uncheck_included_records_exclude_and_shows_custom(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    emitted: list[object] = []
    widget.scope_changed.connect(emitted.append)

    _leaf(widget, "TFCos").setCheckState(0, Qt.CheckState.Unchecked)

    assert len(emitted) == 1
    payload = widget.current_scope()
    assert payload["exclude"] == ["TFCos"]
    assert payload["include"] == []
    # Transient "Custom" item present and selected.
    assert widget._preset_combo.currentText() == "Custom"
    assert widget._preset_combo.currentData() is None

    # Round-trip through set_scope onto a fresh widget.
    other = WizardScopeSelector()
    other.set_resolver(resolver)
    other.set_scope(payload)
    assert other.current_scope() == payload
    assert _leaf(other, "TFCos").checkState(0) == Qt.CheckState.Unchecked


# ── Re-select a real preset clears overrides and drops Custom ─────────────────


def test_reselect_real_preset_clears_overrides_and_removes_custom(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    _leaf(widget, "TFCos").setCheckState(0, Qt.CheckState.Unchecked)
    assert widget._preset_combo.currentData() is None  # Custom active

    idx = widget._preset_combo.findData("all")
    widget._preset_combo.setCurrentIndex(idx)

    assert widget.current_scope()["exclude"] == []
    assert widget._preset_combo.findData(None) == -1  # Custom removed
    assert widget._preset_combo.currentData() == "all"


# ── Uncheck everything → invalid + warning ───────────────────────────────────


def test_uncheck_all_is_invalid_and_shows_warning(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    validity: list[bool] = []
    widget.validity_changed.connect(validity.append)

    for name in ("TFCos", "TFGauss", "Exponential"):
        _leaf(widget, name).setCheckState(0, Qt.CheckState.Unchecked)

    assert widget.is_valid() is False
    assert validity[-1] is False
    assert "No components included" in widget._estimate_label.text()
    assert tokens.ERROR in widget._estimate_label.styleSheet()


# ── Check a resolver-excluded component → include override ───────────────────


def test_check_excluded_component_records_include(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)

    _leaf(widget, "Muonium").setCheckState(0, Qt.CheckState.Checked)

    payload = widget.current_scope()
    assert payload["include"] == ["Muonium"]
    assert payload["exclude"] == []
    assert widget._preset_combo.currentText() == "Custom"


# ── set_scope resets / falls back, emits nothing ─────────────────────────────


def test_set_scope_none_resets_to_auto_without_signals(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    _leaf(widget, "TFCos").setCheckState(0, Qt.CheckState.Unchecked)

    scope_events: list[object] = []
    validity_events: list[bool] = []
    widget.scope_changed.connect(scope_events.append)
    widget.validity_changed.connect(validity_events.append)

    widget.set_scope(None)

    assert widget.current_scope() == build_scope_payload("auto", set(), set())
    assert widget._preset_combo.currentData() == "auto"
    assert scope_events == []
    assert validity_events == []


def test_set_scope_unknown_preset_falls_back_to_auto(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    scope_events: list[object] = []
    widget.scope_changed.connect(scope_events.append)

    widget.set_scope({"version": 1, "preset": "does-not-exist", "include": [], "exclude": []})

    assert widget._preset_id == "auto"
    assert widget._preset_combo.currentData() == "auto"
    assert scope_events == []


# ── Family tri-state parent toggles all children ─────────────────────────────


def test_family_parent_toggle_excludes_all_children(qapp: QApplication) -> None:
    widget, resolver = _make(qapp)
    emitted: list[object] = []
    widget.scope_changed.connect(emitted.append)

    # Uncheck the "Precession" family parent — both children go unchecked.
    parent = widget._family_tree.topLevelItem(0)
    parent.setCheckState(0, Qt.CheckState.Unchecked)

    payload = widget.current_scope()
    assert set(payload["exclude"]) == {"TFCos", "TFGauss"}
    assert payload["include"] == []
    assert emitted, "at least one scope_changed emitted for the parent toggle"
