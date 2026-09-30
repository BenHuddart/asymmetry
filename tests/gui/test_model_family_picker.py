"""Tests for the model family picker, driven by the real ``describe_scope``."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset, Run  # noqa: E402
from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass  # noqa: E402
from asymmetry.core.fitting.wizard_scope import (  # noqa: E402
    FitTimeEstimates,
    WizardScope,
    describe_scope,
    set_user_field_direction,
)
from asymmetry.gui.styles import tokens  # noqa: E402
from asymmetry.gui.widgets.model_family_picker import ModelFamilyPicker  # noqa: E402

NOT_RECORDED = 3  # the "Not recorded" button's id


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _run(field: float, **metadata: str) -> MuonDataset:
    """A dataset whose run carries its own copy of the metadata, as a loader builds it."""
    meta = {"field": field, **metadata}
    return MuonDataset(
        time=np.linspace(0.0, 8.0, 4),
        asymmetry=np.zeros(4),
        error=np.ones(4),
        metadata=dict(meta),
        run=Run(run_number=1, metadata=dict(meta)),
    )


def _series(*directions: str, field: float = 100.0) -> list[MuonDataset]:
    return [_run(field, field_direction=d) if d else _run(field) for d in directions]


def _picker(datasets: list[MuonDataset], scope: WizardScope = WizardScope()) -> ModelFamilyPicker:
    picker = ModelFamilyPicker(lambda s: describe_scope(datasets, s))
    picker.set_scope(scope)
    return picker


def _pill(picker: ModelFamilyPicker, name: str):
    return next(card.pills[name] for card in picker._cards.values() if name in card.pills)


def _record(signal) -> list:
    received: list = []
    signal.connect(received.append)
    return received


# --- direction row ------------------------------------------------------------


def test_a_series_recording_no_direction_offers_the_answer(qapp):
    picker = _picker(_series("", "", ""))
    assert not picker._direction_frame.isHidden()
    assert picker._recorded.isHidden()
    assert picker._direction_group.checkedId() == NOT_RECORDED
    assert picker._direction_note.text() == "The runs record no field direction."


def test_a_single_silent_run_is_named_in_the_singular(qapp):
    picker = _picker(_series(""))
    assert picker._direction_note.text() == "The run records no field direction."


def test_a_partly_recorded_series_counts_the_silent_runs(qapp):
    picker = _picker(_series("Longitudinal", "Longitudinal", ""))
    assert not picker._direction_frame.isHidden()
    assert picker._direction_note.text() == "1 of 3 runs record no field direction."


def test_clicking_a_direction_emits_it_and_leaves_the_runs_alone(qapp):
    datasets = _series("", "")
    picker = _picker(datasets)
    answers = _record(picker.direction_answered)
    scopes = _record(picker.scope_changed)
    picker._direction_group.button(2).click()
    picker._direction_group.button(NOT_RECORDED).click()
    assert answers == [FieldGeometry.TF, None]
    assert scopes == []
    assert all("field_direction" not in d.metadata for d in datasets)


def test_the_applied_answer_is_checked_after_refresh(qapp):
    datasets = _series("", "")
    picker = _picker(datasets)
    validity = _record(picker.validity_changed)
    set_user_field_direction(datasets, FieldGeometry.LF)
    picker.refresh()
    assert picker._direction_group.checkedButton().text() == "Longitudinal"
    assert validity == [True]
    # The answer narrows what applies: no TF-only component has a pill any more.
    assert _pill(picker, "VortexLattice").isHidden()


def test_the_note_says_where_an_answer_was_saved(qapp):
    datasets = _series("", "", "", "")
    picker = _picker(datasets)
    set_user_field_direction(datasets, FieldGeometry.LF)
    picker.refresh()
    assert picker._direction_note.text() == "Set by you — saved on the 4 runs that record none."
    set_user_field_direction(datasets, None)
    picker.refresh()
    assert picker._direction_note.text() == "The runs record no field direction."


def test_the_answer_note_names_the_runs_the_files_record(qapp):
    datasets = _series("Longitudinal", "Longitudinal", "")
    picker = _picker(datasets)
    set_user_field_direction(datasets, FieldGeometry.LF)
    picker.refresh()
    assert picker._direction_note.text() == (
        "Set by you — saved on the run that records none; the files record the other 2."
    )


def test_the_answer_note_on_a_single_run(qapp):
    datasets = _series("")
    picker = _picker(datasets)
    set_user_field_direction(datasets, FieldGeometry.ZF)
    picker.refresh()
    assert picker._direction_note.text() == "Set by you — saved on the run, which records none."


def test_a_fully_recorded_series_shows_its_direction_read_only(qapp):
    picker = _picker(_series("Transverse", "Transverse"))
    assert picker._direction_frame.isHidden()
    assert picker._direction_note.isHidden()
    assert not picker._recorded.isHidden()
    assert picker._recorded.text() == "Recorded: Transverse"


def test_a_mixed_recorded_series_lists_each_direction_with_its_count(qapp):
    picker = _picker(_series("Zero field", "Transverse", "Zero field"))
    assert picker._recorded.text() == "Recorded: Zero field (2), Transverse (1)"


# --- physics chips -------------------------------------------------------------


def test_physics_chips_combine_into_the_scope(qapp):
    picker = _picker(_series("Longitudinal"))
    scopes = _record(picker.scope_changed)
    picker._chips[PhysicsClass.DYNAMICS].click()
    picker._chips[PhysicsClass.MUONIUM].click()
    assert scopes[-1].physics == {PhysicsClass.DYNAMICS, PhysicsClass.MUONIUM}
    picker._chips[PhysicsClass.DYNAMICS].click()
    assert picker.scope().physics == {PhysicsClass.MUONIUM}
    assert [scope.physics for scope in scopes] == [
        {PhysicsClass.DYNAMICS},
        {PhysicsClass.DYNAMICS, PhysicsClass.MUONIUM},
        {PhysicsClass.MUONIUM},
    ]


def test_set_scope_checks_the_chips_without_emitting(qapp):
    picker = _picker(_series("Longitudinal"))
    scopes = _record(picker.scope_changed)
    validity = _record(picker.validity_changed)
    picker.set_scope(WizardScope(physics=frozenset({PhysicsClass.MOLECULAR}), skip_slow=True))
    assert picker._chips[PhysicsClass.MOLECULAR].isChecked()
    assert not picker._chips[PhysicsClass.MAGNETISM].isChecked()
    assert picker._skip_slow.isChecked()
    assert scopes == [] and validity == []


# --- pills ----------------------------------------------------------------------


def test_switching_off_an_included_pill_excludes_it(qapp):
    picker = _picker(_series("Longitudinal"), WizardScope(include_components=frozenset({"Keren"})))
    pill = _pill(picker, "Keren")
    assert pill.isChecked()
    pill.click()
    assert picker.scope().exclude_components == {"Keren"}
    assert picker.scope().include_components == frozenset()
    assert not _pill(picker, "Keren").isChecked()


def test_switching_on_an_excluded_pill_includes_it(qapp):
    picker = _picker(
        _series("Zero field"),
        WizardScope(
            physics=frozenset({PhysicsClass.MOLECULAR}),
            exclude_components=frozenset({"Oscillatory"}),
        ),
    )
    pill = _pill(picker, "Oscillatory")
    assert not pill.isChecked()
    pill.click()
    assert picker.scope().include_components == {"Oscillatory"}
    assert picker.scope().exclude_components == frozenset()
    assert _pill(picker, "Oscillatory").isChecked()


def test_a_pill_click_shows_its_details(qapp):
    picker = _picker(_series("Longitudinal"))
    _pill(picker, "DynamicGaussianKT").click()
    details = picker._details
    assert details.label.text() == "Dynamic Gaussian KT"
    assert details.use_when.text() == "Fluctuating Gaussian fields (strong collision)"
    assert details.category.text() == "KUBO–TOYABE"
    assert details.facts["Geometries"].text() == "Zero field · Longitudinal"
    assert details.facts["Fitting cost"].text() == "Slow (not yet timed on this computer)"
    assert details.facts["This scope"].text() == "switched off by you"


def test_hovering_a_pill_shows_its_details(qapp):
    picker = _picker(_series("Longitudinal"))
    _pill(picker, "Keren").hovered.emit("Keren")
    assert picker._details.label.text() == "Keren"
    assert picker._details.facts["This scope"].text() == "Included"
    assert picker._details.facts["Fitting cost"].text() == "Quick"


def _slow_tag(picker: ModelFamilyPicker, name: str) -> bool:
    return not _pill(picker, name).slow_tag.isHidden()


def test_timed_models_show_their_estimate_and_are_tagged_by_it(qapp):
    datasets = _series("Longitudinal")
    timed = FitTimeEstimates({"DynamicGaussianKT": 0.4, "Keren": 12.3, "Oscillatory": 123.4})
    picker = ModelFamilyPicker(lambda s: describe_scope(datasets, s, timed))
    cost = picker._details.facts["Fitting cost"]
    for name, text in (
        ("DynamicGaussianKT", "Quick — ≈ 0.4 s per run on this computer"),
        ("Keren", "Slow — ≈ 12 s per run on this computer"),
        ("Oscillatory", "Slow — ≈ 123 s per run on this computer"),
        ("DynamicLorentzianKT", "Slow (not yet timed on this computer)"),
        ("Exponential", "Quick"),
    ):
        _pill(picker, name).hovered.emit(name)
        assert cost.text() == text, name
    assert not _slow_tag(picker, "DynamicGaussianKT")
    assert _slow_tag(picker, "Keren")
    assert _slow_tag(picker, "DynamicLorentzianKT")


def test_a_refresh_retags_the_pills_when_the_fit_times_change(qapp):
    datasets = _series("Longitudinal")
    judgement = {"fit_times": FitTimeEstimates({})}
    picker = ModelFamilyPicker(lambda s: describe_scope(datasets, s, judgement["fit_times"]))
    assert _slow_tag(picker, "DynamicGaussianKT")
    judgement["fit_times"] = FitTimeEstimates({"DynamicGaussianKT": 0.3})
    picker.refresh()
    assert not _slow_tag(picker, "DynamicGaussianKT")


def test_an_excluded_pill_says_why_in_its_tooltip(qapp):
    picker = _picker(
        _series("Zero field"), WizardScope(physics=frozenset({PhysicsClass.MOLECULAR}))
    )
    tooltip = _pill(picker, "Oscillatory").toolTip()
    assert tooltip.startswith("One well-defined local field\n")
    assert "magnetism" in tooltip


# --- family boxes ---------------------------------------------------------------


def test_the_family_box_is_tri_state_over_the_applicable_components(qapp):
    picker = _picker(_series("Longitudinal"))
    card = picker._cards["Kubo–Toyabe"]
    assert card.box.checkState() == Qt.CheckState.Checked
    assert card.count.text() == "4 of 4"

    _pill(picker, "DynamicGaussianKT").click()
    assert card.box.checkState() == Qt.CheckState.PartiallyChecked
    assert card.count.text() == "3 of 4"

    card.box.click()  # partial → every applicable component on
    assert card.box.checkState() == Qt.CheckState.Checked
    assert "DynamicGaussianKT" in picker.scope().include_components
    assert picker.scope().exclude_components == frozenset()

    card.box.click()  # all on → all off
    assert card.box.checkState() == Qt.CheckState.Unchecked
    assert card.count.text() == "0 of 4"
    assert picker.scope().exclude_components == {
        "LongitudinalFieldKT",
        "DynamicGaussianKT",
        "DynamicLorentzianKT",
        "GaussianBroadenedKT",
    }


def test_a_family_with_nothing_applicable_is_disabled(qapp):
    picker = _picker(_series("Transverse"))
    card = picker._cards["Kubo–Toyabe"]
    assert not card.box.isEnabled()
    assert card.count.text() == "none apply"
    assert all(pill.isHidden() for pill in card.pills.values())


def test_cards_carry_the_family_titles_and_blurbs(qapp):
    picker = _picker(_series("Zero field"))
    assert list(picker._cards)[:2] == ["Relaxation", "Kubo–Toyabe"]
    texts = {label.text() for label in picker._cards["Relaxation"].findChildren(QLabel)}
    assert "Monotonic loss of polarisation — the usual first guess." in texts


# --- what does not apply ----------------------------------------------------------


def test_components_of_another_geometry_are_listed_not_hidden_away(qapp):
    picker = _picker(_series("Longitudinal"))
    card = picker._cards["Oscillation"]
    assert not card.not_applicable.isHidden()
    assert card.not_applicable.text().startswith("Not LF models: Cosine precession · ")
    assert "Vortex lattice" in card.not_applicable.text()
    assert _pill(picker, "VortexLattice").isHidden()


def test_muonium_outside_its_field_regime_is_listed_apart(qapp):
    picker = _picker(_series("Transverse", field=20.0))
    text = picker._cards["Muonium"].not_applicable.text()
    assert "Not TF models: Muonium, ZF · Muonium LF relaxation" in text
    assert (
        "Outside these runs' field range: Muonium, high TF · Anisotropic muonium, high TF" in text
    )


def test_nothing_is_listed_when_everything_applies(qapp):
    picker = _picker(_series(""))
    assert all(card.not_applicable.isHidden() for card in picker._cards.values())


# --- search ------------------------------------------------------------------------


def test_search_highlights_hits_and_dims_the_rest(qapp):
    picker = _picker(_series("Longitudinal"))
    picker._search.setText("KEREN")
    assert f"2px solid {tokens.ACCENT}" in _pill(picker, "Keren").styleSheet()
    assert tokens.TEXT_DIM in _pill(picker, "Exponential").styleSheet()
    assert tokens.TEXT_DIM not in _pill(picker, "Keren").styleSheet()
    picker._search.setText("helix")
    assert f"2px solid {tokens.ACCENT}" not in _pill(picker, "Keren").styleSheet()
    picker._search.clear()
    assert tokens.TEXT_DIM not in _pill(picker, "Exponential").styleSheet()
    assert f"2px solid {tokens.ACCENT}" not in _pill(picker, "Keren").styleSheet()


def test_search_matches_the_use_when_line(qapp):
    picker = _picker(_series("Zero field"))
    picker._search.setText("helix")
    assert f"2px solid {tokens.ACCENT}" in _pill(picker, "HelicalPowder").styleSheet()
    assert f"2px solid {tokens.ACCENT}" in _pill(picker, "HelicalCrystal").styleSheet()


# --- footer ----------------------------------------------------------------------


def test_the_footer_counts_models_and_names_the_slow_ones(qapp):
    datasets = _series("Longitudinal")
    picker = _picker(datasets)
    view = describe_scope(datasets, WizardScope())
    assert picker._screen_count.text() == (
        f"Will screen {view.included_count} of {view.applicable_count} models"
    )
    slow = " · ".join(c.label for c in view.slow_included)
    assert picker._slow_line.text() == f"{len(view.slow_included)} slow: {slow}"
    assert tokens.WARN in picker._slow_line.styleSheet()


def test_leaving_out_slow_models_sets_skip_slow(qapp):
    datasets = _series("Longitudinal")
    picker = _picker(datasets)
    scopes = _record(picker.scope_changed)
    picker._skip_slow.click()
    assert scopes == [WizardScope(skip_slow=True)]
    view = describe_scope(datasets, WizardScope(skip_slow=True))
    assert picker._screen_count.text() == (
        f"Will screen {view.included_count} of {view.applicable_count} models"
    )
    assert picker._slow_line.text() == "No slow models included."
    assert tokens.TEXT_MUTED in picker._slow_line.styleSheet()


def test_reset_to_suggestions_shows_only_with_overrides(qapp):
    picker = _picker(_series("Longitudinal"))
    assert picker._reset.isHidden()
    _pill(picker, "Keren").click()
    assert not picker._reset.isHidden()
    _pill(picker, "Keren").click()  # back on: an include override is still an override
    assert picker.scope().include_components == {"Keren"}
    assert not picker._reset.isHidden()


def test_reset_to_suggestions_drops_overrides_and_keeps_the_rest(qapp):
    looking_for = frozenset({PhysicsClass.DYNAMICS})
    picker = _picker(
        _series("Longitudinal"),
        WizardScope(
            physics=looking_for,
            include_components=frozenset({"Oscillatory"}),
            exclude_components=frozenset({"Keren"}),
            skip_slow=True,
        ),
    )
    scopes = _record(picker.scope_changed)
    picker._reset.click()
    assert scopes == [WizardScope(physics=looking_for, skip_slow=True)]
    assert picker.scope() == WizardScope(physics=looking_for, skip_slow=True)
    assert picker._reset.isHidden()
    assert _pill(picker, "Keren").isChecked()


# --- validity ------------------------------------------------------------------------


def test_only_the_background_is_not_a_valid_scope(qapp):
    picker = _picker(_series("Transverse"))
    validity = _record(picker.validity_changed)
    for title, card in picker._cards.items():
        if title != "Background" and card.box.isEnabled():
            card.box.click()
    assert picker.scope().exclude_components
    assert not picker.is_valid()
    assert validity[-1] is False and validity[0] is True
    assert "Only the background is included" in picker._screen_count.text()
    assert tokens.ERROR in picker._screen_count.styleSheet()

    _pill(picker, "Exponential").click()
    assert picker.is_valid()
    assert validity[-1] is True


# --- narrow layout ------------------------------------------------------------------


def test_the_details_panel_hides_in_a_narrow_picker(qapp):
    picker = _picker(_series("Longitudinal"))
    picker.resize(1600, 700)
    picker.show()
    qapp.processEvents()
    assert picker._details.isVisible()
    picker.resize(700, 700)
    qapp.processEvents()
    assert not picker._details.isVisible()
    picker.close()


def test_the_summary_names_the_direction_and_the_models_screened(qapp):
    picker = _picker(_series("Longitudinal", "Longitudinal"))
    direction, count = picker.summary().split(" · ")
    assert direction == "Longitudinal"
    assert count == f"{picker._view.included_count} models"
    assert _picker(_series("", "")).summary().startswith("No direction · ")
