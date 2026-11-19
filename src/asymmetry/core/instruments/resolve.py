"""Convert resolved profile data into the shared layout consumer interface."""

from asymmetry.core.instruments.geometry import Circular, EndOn, Rectangle, Wedge, expand_view
from asymmetry.core.instruments.layout import (
    BankLayout,
    DetectorSegment,
    GroupDefinition,
    InstrumentLayout,
    PresetGrouping,
    generic_layout,
)
from asymmetry.core.instruments.schema import InstrumentProfile


def layout_from_profile(
    profile: InstrumentProfile, acquisition: str | None = None, *, n_histograms: int | None = None
) -> InstrumentLayout:
    """Resolve one declared acquisition without changing its analysis ordinals."""
    if acquisition is None:
        if n_histograms is None or n_histograms < 1:
            raise ValueError("Unresolved acquisition requires a positive acquired histogram count")
        return generic_layout(n_histograms)
    config = next(
        (config for config in profile.detectors.acquisitions if config.key == acquisition), None
    )
    if config is None:
        raise ValueError(f"Unknown acquisition {acquisition!r} for {profile.identity.key}")
    channels = {channel.key: channel for channel in config.channels}
    hardware = {element.key: element for element in profile.detectors.elements}
    banks = []
    for view in config.views:
        segments = []
        for element in expand_view(view):
            target, shape = element.target, element.shape
            channel = channels[target.channel] if target.channel is not None else None
            label = (
                target.label
                if target.label is not None
                else channel.label
                if channel
                else hardware[target.hardware].label
            )
            radial = isinstance(view.arrangement, Circular) and view.arrangement.radial_rotation
            segments.append(
                DetectorSegment(
                    detector_id=channel.ordinal if channel else None,
                    sector_index=element.sector,
                    ring_index=element.ring,
                    angle_center_deg=element.angle,
                    angle_half_width_deg=shape.half_angle if isinstance(shape, Wedge) else 0.0,
                    r_inner=shape.inner if isinstance(shape, Wedge) else 0.0,
                    r_outer=shape.outer if isinstance(shape, Wedge) else 0.0,
                    shape="wedge"
                    if isinstance(shape, Wedge)
                    else "rectangle"
                    if isinstance(shape, Rectangle)
                    else "endon_out"
                    if shape.toward
                    else "endon_in",
                    label=label,
                    x_center=element.x,
                    y_center=element.y,
                    width=shape.width
                    if isinstance(shape, Rectangle)
                    else 2 * shape.radius
                    if isinstance(shape, EndOn)
                    else 0.0,
                    height=shape.height
                    if isinstance(shape, Rectangle)
                    else 2 * shape.radius
                    if isinstance(shape, EndOn)
                    else 0.0,
                    rotation_deg=shape.rotation
                    + (
                        element.angle
                        if radial or not isinstance(view.arrangement, Circular)
                        else 0.0
                    )
                    if isinstance(shape, Rectangle)
                    else 0.0,
                    read_only=not target.editable,
                )
            )
        banks.append(BankLayout(view.label, tuple(segments)))
    entry = next(entry for entry in profile.presets if entry.acquisition == acquisition)
    presets = {
        preset.name: PresetGrouping(
            name=preset.name,
            groups={
                group.id: GroupDefinition(
                    group.name, tuple(channels[key].ordinal for key in group.channels)
                )
                for group in preset.groups
            },
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
            projections=preset.projections,
        )
        for preset in entry.presets
    }
    layout_key = config.layout_key or profile.identity.layout_key
    return InstrumentLayout(
        layout_key,
        max(channel.ordinal for channel in config.channels),
        tuple(banks),
        presets,
        view=config.views[0].mode,
        display_name=None
        if profile.identity.display_name == layout_key
        else profile.identity.display_name,
        reference_arrows=config.reference_arrows,
        apply_default_preset_on_load=profile.defaults.apply_default_preset_on_load,
    )
