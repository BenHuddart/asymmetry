"""Synthetic profile cases use the actual detector editor renderer and hit tests."""

from types import SimpleNamespace

import pytest
from PySide6.QtTest import QSignalSpy

from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.gui.widgets.detector_schematic import DetectorSchematicWidget
from tests._instrument_profiles import synthetic_profile, synthetic_profiles

pytestmark = pytest.mark.gui


@pytest.mark.parametrize(
    "case", ["concentric", "grid", "ring_rectangles", "explicit", "multiview", "combined_split"]
)
def test_synthetic_declarations_render_actual_selectable_channels(qapp, case):
    profile = synthetic_profiles()[case]
    for config in profile.detectors.acquisitions:
        layout = layout_from_profile(profile, config.key)
        widget = DetectorSchematicWidget(layout)
        widget.resize(1000, 600)
        widget.show()
        qapp.processEvents()
        widget._canvas.draw()
        assert set(widget._patches) == {
            channel.ordinal for channel in config.channels if channel.selectable
        }
        assert None not in widget._patches
        assert not widget.grab().isNull()
        widget.close()
        widget.deleteLater()


def test_hardware_context_has_no_histogram_number_color_or_pick_path(qapp):
    profile = synthetic_profiles()["multiview"]
    layout = layout_from_profile(profile, "test")
    widget = DetectorSchematicWidget(layout)
    widget.set_all_groups({1: [1], 2: [2]}, active_group=1)
    signals = QSignalSpy(widget.detector_toggled)
    widget._on_click(SimpleNamespace(button=1, inaxes=widget._axes[0], xdata=0.0, ydata=2.0))
    assert signals.count() == 0
    assert all(number is not None for number, patch in widget._readonly_patches)
    widget._on_click(SimpleNamespace(button=1, inaxes=widget._axes[0], xdata=-1.0, ydata=0.0))
    assert signals.count() == 1
    assert signals.at(0)[0] == 1


def test_metadata_only_profile_renders_generic_acquired_channels(qapp):
    widget = DetectorSchematicWidget(layout_from_profile(synthetic_profile(), n_histograms=5))
    assert set(widget._patches) == {1, 2, 3, 4, 5}
    widget._canvas.draw()


@pytest.mark.parametrize(
    "case", ["radial_rectangle", "radial_wedge", "plan_rotation", "plan_context"]
)
def test_resolved_primitive_extents_are_visible_in_actual_axes(qapp, case):
    from matplotlib.patches import Circle

    from asymmetry.core.instruments.geometry import (
        EndOn,
        Explicit,
        Occurrence,
        Placement,
        Rectangle,
        View,
        Wedge,
    )
    from asymmetry.core.instruments.schema import Acquisition, Channel, PhysicalElement

    a, b = Occurrence(channel="a"), Occurrence(channel="b")
    if case == "radial_rectangle":
        mode = "radial"
        placements = (
            Placement(a, Rectangle(0.5, 0.5), 3.0, 0.0),
            Placement(b, Rectangle(0.5, 0.5)),
        )
    elif case == "radial_wedge":
        mode = "radial"
        placements = (
            Placement(a, Wedge(2.0, 3.0, 20.0), angle=90.0),
            Placement(b, Wedge(0.5, 1.0, 20.0), angle=270.0),
        )
    else:
        mode = "plan"
        placements = (
            Placement(a, Rectangle(0.2, 4.0, 90.0), 0.0, 1.0),
            Placement(b, Rectangle(0.4, 0.4), 0.5, -0.5),
        )
        if case == "plan_context":
            placements = (
                Placement(a, Rectangle(0.4, 0.4), -0.5, 0.0),
                Placement(b, Rectangle(0.4, 0.4), 0.5, 0.0),
                Placement(Occurrence(hardware="h", editable=False), EndOn(radius=0.75), 4.0, 3.0),
            )
    profile = synthetic_profile(
        (
            Acquisition(
                "test",
                "Visibility",
                (Channel("a", 1), Channel("b", 2)),
                (View("test", "View", mode, "Synthetic visibility case", Explicit(placements)),),
            ),
        ),
        elements=(PhysicalElement("h", "Context"),),
    )
    widget = DetectorSchematicWidget(layout_from_profile(profile, "test"))
    widget.resize(1000, 600)
    widget.show()
    qapp.processEvents()
    widget._canvas.draw()
    axes = widget._axes[0]
    primitives = list(widget._patches.values())
    if case == "plan_context":
        primitives.append(
            next(
                patch
                for patch in axes.patches
                if isinstance(patch, Circle) and patch.center == (4.0, 3.0)
            )
        )
    title = axes.title.get_window_extent()
    assert widget._fig.bbox.contains(title.xmin, title.ymin)
    assert widget._fig.bbox.contains(title.xmax, title.ymax)
    for patch in primitives:
        bounds = patch.get_window_extent()
        assert axes.bbox.contains(bounds.xmin, bounds.ymin)
        assert axes.bbox.contains(bounds.xmax, bounds.ymax)
    if mode == "plan":
        sample = next(
            patch for patch in axes.patches if isinstance(patch, Circle) and patch.center == (0, 0)
        )
        for label in (text for text in axes.texts if text.get_text() in {"1", "2"}):
            assert not sample.contains_point(axes.transData.transform(label.get_position()))
    for size in ((600, 600), (1200, 800)):
        widget.resize(*size)
        qapp.processEvents()
        widget._canvas.draw()
        title = axes.title.get_window_extent()
        assert widget._fig.bbox.contains(title.xmin, title.ymin)
        assert widget._fig.bbox.contains(title.xmax, title.ymax)


def test_emu_retains_its_existing_radial_display_limits(qapp):
    from asymmetry.core.instruments.isis_emu import PROFILE

    widget = DetectorSchematicWidget(layout_from_profile(PROFILE, "nexus96"))
    assert [(axes.get_xlim(), axes.get_ylim()) for axes in widget._axes] == [
        ((-1.25, 1.25), (-1.25, 1.25))
    ] * 2


@pytest.mark.parametrize("acquisition", ("bin_combined", "root_split", "root_combined_split"))
def test_gps_profile_context_uses_existing_marker_radius_and_one_pick_identity(qapp, acquisition):
    from matplotlib.patches import Circle

    from asymmetry.core.instruments.psi_gps import PROFILE

    layout = layout_from_profile(PROFILE, acquisition)
    widget = DetectorSchematicWidget(layout)
    widget.resize(1100, 680)
    widget.show()
    qapp.processEvents()
    widget._canvas.draw()
    assert set(widget._patches) == set(range(1, layout.n_detectors + 1))
    for bank, axes in zip(layout.banks, widget._axes, strict=True):
        for segment in bank.segments:
            if segment.shape.startswith("endon_"):
                markers = [
                    patch
                    for patch in axes.patches
                    if isinstance(patch, Circle)
                    and patch.center == (segment.x_center, segment.y_center)
                ]
                assert any(patch.radius == 0.16 for patch in markers)
                assert segment.label in {text.get_text() for text in axes.texts}
                assert segment.read_only
                assert not widget._point_in_segment(segment.x_center, segment.y_center, segment)
    widget.close()
    widget.deleteLater()


def test_flame_profile_transfers_actual_direction_arrow_labels(qapp):
    from asymmetry.core.instruments.psi_flame import PROFILE

    widget = DetectorSchematicWidget(layout_from_profile(PROFILE, "eight_channel"))
    widget.resize(1100, 680)
    widget.show()
    qapp.processEvents()
    widget._canvas.draw()
    axes = widget._axes[0]
    labels = {text.get_text() for text in axes.texts}
    assert {"beam, main magnetic field", "initial muon spin"} <= labels
    for arrow in PROFILE.detectors.acquisitions[0].reference_arrows:
        assert axes.get_xlim()[0] <= min(arrow.start[0], arrow.end[0])
        assert axes.get_xlim()[1] >= max(arrow.start[0], arrow.end[0])
    widget.close()
    widget.deleteLater()
