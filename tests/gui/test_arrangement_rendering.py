"""Generated arrangements retain interactive ordinals and minimal beam-only cues."""

import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.core.instruments.geometry import BeamProjection
from asymmetry.gui.utils.detector_drawing import draw_beam
from asymmetry.gui.widgets.detector_schematic import DetectorSchematicWidget
from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("name,ordinal", (("GPS-RD", 11), ("GPS-RD15", 15), ("HAL", 1)))
def test_auxiliary_click_keeps_other_memberships_and_exclusion_rules(qapp, name, ordinal):
    layout = get_instrument_layout(name)
    widget = DetectorSchematicWidget(layout)
    widget.set_all_groups({1: {ordinal}, 3: {ordinal, 2}}, active_group=1)
    widget.set_active_group(1)
    widget.resize(1000, 650)
    widget.show()
    qapp.processEvents()
    segment = layout.auxiliary[0]
    x, y = widget._auxiliary_axes.transData.transform((segment.x_center, segment.y_center))
    event = MouseEvent("button_press_event", widget._canvas, x, y, button=1)
    assert widget._hit_test_event(event) is segment
    changes = []
    widget.detector_toggled.connect(lambda key, included: changes.append((key, included)))
    widget._on_click(event)
    assert changes == [(ordinal, False)]
    assert widget._groups[1] == set() and widget._groups[3] == {ordinal, 2}
    widget._on_click(event)
    assert changes[-1] == (ordinal, True)
    assert widget._groups[1] == {ordinal} and widget._groups[3] == {ordinal, 2}
    widget.set_exclude_mode(True)
    widget._on_click(event)
    assert widget.get_excluded_detectors() == {ordinal}
    assert widget._groups[1] == {ordinal} and widget._groups[3] == {ordinal, 2}
    widget.close()


@pytest.mark.parametrize("name", ("GPS", "GPS-RD", "GPS-RD15", "FLAME", "HAL", "EMU"))
def test_editor_open_ok_preserves_custom_groups_and_exact_pair(qapp, name):
    layout = get_instrument_layout(name)
    groups = {1: [layout.n_detectors, 1], 2: [2], 3: [1]}
    dialog = DetectorLayoutDialog(
        layout,
        groups,
        forward_group=2,
        backward_group=1,
        group_names={1: "Custom one", 2: "Custom two", 3: "Shared"},
    )
    result = dialog.get_result()
    assert result["groups"] == {1: [1, layout.n_detectors], 2: [2], 3: [1]}
    assert (result["forward_group"], result["backward_group"]) == (2, 1)
    assert result["group_names"] == {1: "Custom one", 2: "Custom two", 3: "Shared"}
    assert set(dialog._schematic._patches) == set(range(1, layout.n_detectors + 1))
    assert len(dialog._schematic._axes) == len(layout.banks)
    dialog.close()


@pytest.mark.parametrize(
    "projection,tooltip,arrows,circles",
    (
        (BeamProjection((1, 0), None), "Muon beam direction", 1, 0),
        (BeamProjection((-1, 0), 0), "Muon beam direction", 1, 0),
        (BeamProjection((0, 0), 1), "Muon beam out of the screen", 0, 1),
        (BeamProjection((0, 0), -1), "Muon beam into the screen", 0, 1),
        (
            BeamProjection((0, 0), None),
            "Muon beam direction\nView along beam axis; viewing sense unspecified",
            0,
            0,
        ),
    ),
)
def test_shared_beam_rendering_does_not_infer_axial_sense(projection, tooltip, arrows, circles):
    from matplotlib.patches import Circle, FancyArrowPatch

    axes = Figure().subplots()
    artists = draw_beam(axes, projection)
    assert {text.get_text() for text in axes.texts} <= {"Beam", "·", "×"}
    assert "Beam" in [text.get_text() for text in axes.texts]
    assert {text for _, text in artists} == {tooltip}
    assert sum(isinstance(p, FancyArrowPatch) for p in axes.patches) == arrows
    assert sum(isinstance(p, Circle) for p in axes.patches) == circles
    if arrows:
        start, end = axes.patches[0]._posA_posB
        assert (end[0] - start[0]) * projection.screen[0] > 0


def test_missing_beam_is_absent_and_no_legacy_field_or_spin_cues_are_added(qapp):
    assert draw_beam(Figure().subplots(), None) == ()
    for name in ("GPS", "GPS-RD", "GPS-RD15", "FLAME", "HAL"):
        widget = DetectorSchematicWidget(get_instrument_layout(name))
        texts = [text.get_text() for axes in widget._axes for text in axes.texts]
        assert texts.count("Beam") == len(widget._axes)
        assert not any(
            "spin" in text.lower() or "field" in text.lower() or "+z" in text for text in texts
        )
        widget.close()
    unchanged = get_instrument_layout("EMU")
    widget = DetectorSchematicWidget(unchanged)
    assert widget._beam_artists == []
    assert unchanged.auxiliary == ()
    assert all(bank.beam is None for bank in unchanged.banks)
    widget.close()


def test_row_captions_are_drawn_once_and_hover_uses_declared_beam_tooltip(qapp, monkeypatch):
    from matplotlib.text import Text
    from PySide6.QtWidgets import QToolTip

    widget = DetectorSchematicWidget(get_instrument_layout("GPS-RD15"))
    widget.resize(1000, 650)
    widget.show()
    qapp.processEvents()
    for axes in widget._axes:
        texts = [text.get_text() for text in axes.texts]
        assert texts.count("combined") == texts.count("split") == 2
        assert not any("\ncombined" in text or "\nsplit" in text for text in texts)
    messages = []
    monkeypatch.setattr(QToolTip, "showText", lambda point, text, canvas: messages.append(text))
    label = next(artist for artist, _ in widget._beam_artists if isinstance(artist, Text))
    bbox = label.get_window_extent(widget._canvas.get_renderer())
    event = MouseEvent("motion_notify_event", widget._canvas, *bbox.get_points().mean(axis=0))
    widget._on_hover(event)
    assert messages == ["Muon beam direction"]
    widget.close()


@pytest.mark.parametrize("components", ((1, 2), (-2, 1), (0.7, -0.4)))
def test_oblique_beam_angle_survives_nonsquare_axes_and_resize(components):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import FancyArrowPatch

    figure = Figure(figsize=(8, 2))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    draw_beam(axes, BeamProjection(components, None))
    arrow = next(p for p in axes.patches if isinstance(p, FancyArrowPatch))
    for size in ((8, 2), (2, 8)):
        figure.set_size_inches(*size)
        figure.canvas.draw()
        start, end = arrow.get_transform().transform(arrow._posA_posB)
        delta = end - start
        expected = np.array(components) / np.linalg.norm(components)
        assert delta / np.linalg.norm(delta) == pytest.approx(expected)


@pytest.mark.parametrize("direction,x", (("left", -3), ("right", 3)))
def test_vertical_readout_row_has_one_caption_at_its_declared_anchor(qapp, direction, x):
    from asymmetry.core.instruments.geometry import BankRow, Directional, Occurrence, View
    from asymmetry.core.instruments.resolve import layout_from_profile
    from asymmetry.core.instruments.schema import Acquisition, Channel
    from tests._instrument_profiles import synthetic_profile

    row = BankRow(
        tuple(Occurrence(channel=key) for key in ("a", "b", "c")), (1, 1, 1), label="split"
    )
    profile = synthetic_profile(
        (
            Acquisition(
                "readout",
                "Synthetic readout",
                tuple(Channel(key, i + 1) for i, key in enumerate(("a", "b", "c"))),
                (
                    View(
                        "view",
                        "View",
                        "plan",
                        "Synthetic ordered bank",
                        Directional(direction, (row,), distance=3),
                    ),
                ),
            ),
        )
    )
    widget = DetectorSchematicWidget(layout_from_profile(profile, "readout"))
    labels = [text for text in widget._axes[0].texts if text.get_text() == "split"]
    assert len(labels) == 1
    assert labels[0].get_position() == pytest.approx((x, 1.77))
    widget.close()
