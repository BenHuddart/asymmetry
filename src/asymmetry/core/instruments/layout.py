"""Shared legacy layout records; detector IDs remain acquired channel ordinals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class DetectorSegment:
    """Geometric description of a single detector element.

    Parameters
    ----------
    detector_id:
        1-based acquired-channel ordinal; None only for read-only hardware context.
    sector_index:
        Azimuthal sector index within the bank (0-based, 0 = first sector).
    ring_index:
        Radial ring index within the sector (0-based, 0 = innermost).
        For single-ring instruments this is always 0.
    angle_center_deg:
        Central azimuthal angle of the segment in degrees.  ``0°`` points
        right and positive rotation is counter-clockwise.
    angle_half_width_deg:
        Half the angular width of the segment in degrees.
    r_inner:
        Inner radius normalised to the outer disc radius (0–1).
    r_outer:
        Outer radius normalised to the outer disc radius (0–1).
    shape:
        Rendering primitive. ``"wedge"`` is used for circular ISIS detector
        banks; ``"rectangle"`` is used for top-view detector plates such as
        PSI FLAME.  ``"endon_out"`` / ``"endon_in"`` draw a detector seen
        end-on — pointing toward (``⊙``) or away from (``⊗``) the viewer — used
        in multi-panel plan layouts to show a detector that is perpendicular to
        the current view (and therefore editable in the other panel).
    label:
        Optional short detector label displayed alongside the detector ID.
    x_center, y_center, width, height, rotation_deg:
        Rectangle geometry in an arbitrary plan-view coordinate system.
    read_only:
        When ``True`` the segment is drawn for spatial context only — it is not
        clickable and never takes a group colour.  A detector that appears in
        more than one plan panel is *active* (clickable) in exactly one panel
        and ``read_only`` in the others.
    """

    detector_id: int | None
    sector_index: int
    ring_index: int
    angle_center_deg: float
    angle_half_width_deg: float
    r_inner: float
    r_outer: float
    shape: str = "wedge"
    label: str | None = None
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation_deg: float = 0.0
    read_only: bool = False

    def __post_init__(self):
        if self.detector_id is None and not self.read_only:
            raise ValueError("Hardware-only context must be read-only")

    @property
    def angle_start_deg(self) -> float:
        """Lower bound of the segment arc in degrees (counter-clockwise measured)."""
        return self.angle_center_deg - self.angle_half_width_deg

    @property
    def angle_end_deg(self) -> float:
        """Upper bound of the segment arc in degrees (counter-clockwise measured)."""
        return self.angle_center_deg + self.angle_half_width_deg


@dataclass(frozen=True)
class BankLayout:
    """Collection of detector segments forming one detector bank.

    Parameters
    ----------
    name:
        Human-readable bank name (e.g. ``"Forward"``).
    segments:
        All detector segments that belong to this bank.
    """

    name: str
    segments: tuple[DetectorSegment, ...]


@dataclass(frozen=True)
class GroupDefinition:
    """A named set of detectors forming one group within a preset.

    Parameters
    ----------
    name:
        Display name for the group (e.g. ``"Forward"``, ``"Pz Backward"``).
    detector_ids:
        Tuple of 1-based detector IDs belonging to this group.
    """

    name: str
    detector_ids: tuple[int, ...]


@dataclass(frozen=True)
class AsymmetryProjection:
    """One named asymmetry projection within a multi-projection preset.

    Ordered detector-group legs define the recorded output. A quantitative
    physical-component interpretation depends on detector response and calibration.

    Parameters
    ----------
    label:
        Display/identity label, e.g. ``"P_x"``, ``"P_z"``, ``"Top-Bottom"``.
    forward_group:
        Group ID supplying the forward histogram of this projection's pair.
    backward_group:
        Group ID supplying the backward histogram of this projection's pair.
    alpha:
        Default per-projection alpha balance.
    tint:
        Fixed semantic frame colour (hex) used for the chip and subplot frame.
        This is *projection identity* and is deliberately distinct from a data
        trace colour (which encodes run identity in RG mode).
    """

    label: str
    forward_group: int
    backward_group: int
    alpha: float = 1.0
    tint: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the plain-dict form persisted in a dataset grouping payload."""
        payload: dict[str, object] = {
            "label": self.label,
            "forward_group": int(self.forward_group),
            "backward_group": int(self.backward_group),
            "alpha": float(self.alpha),
        }
        if self.tint is not None:
            payload["tint"] = self.tint
        return payload


PROJECTION_TINTS: Final[dict[str, str]] = {
    "P_x": "#534AB7",  # purple
    "P_y": "#BA7517",  # amber
    "P_z": "#0F6E56",  # teal
}

TRANSVERSE_PROJECTION_TINTS: Final[dict[str, str]] = {
    "Top-Bottom": "#B0436E",  # rose
    "Fwd-Back": "#2E8B74",  # jade
    "Left-Right": "#3F6FA0",  # steel blue
}


@dataclass(frozen=True)
class PresetGrouping:
    """A named, complete grouping assignment for an instrument.

    Parameters
    ----------
    name:
        Human-readable preset name (e.g. ``"Longitudinal"``).
    groups:
        Mapping from group ID (1-based, matching the Group 1–8 buttons) to
        :class:`GroupDefinition`.  Unused group IDs should not appear.
    forward_group:
        Group ID of the default forward group.
    backward_group:
        Group ID of the default backward group.
    projections:
        Ordered projections exposed by this preset, when it is a multi-projection
        (vector / dual-grouping) preset.  Empty for ordinary single-pair presets.
    """

    name: str
    groups: dict[int, GroupDefinition]
    forward_group: int
    backward_group: int
    projections: tuple[AsymmetryProjection, ...] = ()


@dataclass(frozen=True)
class ReferenceArrow:
    """A labelled direction marker rendered on plan-view schematics."""

    label: str
    start: tuple[float, float]
    end: tuple[float, float]
    color: str = "#333333"


@dataclass(frozen=True)
class InstrumentLayout:
    """Complete detector layout description for one muon instrument.

    Parameters
    ----------
    name:
        Canonical instrument name (``"HiFi"``, ``"MuSR"``, or ``"EMU"``).
    n_detectors:
        Total number of detectors / histograms.
    banks:
        Ordered tuple of :class:`BankLayout` objects.
    presets:
        Ordered dict of preset names to :class:`PresetGrouping`. The first entry is the default preset.
    view:
        Schematic rendering mode, either ``"radial"`` or ``"plan"``.
    reference_arrows:
        Optional labelled arrows used by plan-view layouts.
    display_name:
        Optional user-facing name shown in the instrument dropdown.  Defaults to
        :attr:`name`.  Layout *variants* of one physical instrument (e.g. the
        6-detector PSI GPS BIN layout ``"GPS"``, the 11-detector ROOT
        sub-detector layout ``"GPS-RD"`` and the 15-detector combined+sub-detector
        ROOT layout ``"GPS-RD15"``) share a display name so the user only
        ever sees the variant matching their loaded data, while ``name`` stays a
        distinct registry key for detection and persistence.
    apply_default_preset_on_load:
        When ``True``, a loader applies this instrument's default preset (the
        first entry in :attr:`presets`) to a freshly loaded run automatically,
        instead of leaving the loader's generic label-based grouping in place.
        Used for instruments whose useful grouping is not the plain
        forward/backward split the labels imply — HAL-9500, whose per-octant
        angle-resolved grouping should be live the moment the data loads, without
        the user having to open the grouping window and click Apply.  Off by
        default so every other instrument keeps its loader grouping untouched.
    """

    name: str
    n_detectors: int
    banks: tuple[BankLayout, ...]
    presets: dict[str, PresetGrouping]
    view: str = "radial"
    reference_arrows: tuple[ReferenceArrow, ...] = ()
    display_name: str | None = None
    apply_default_preset_on_load: bool = False

    @property
    def display(self) -> str:
        """User-facing name (``display_name`` if set, else ``name``)."""
        return self.display_name or self.name

    @property
    def all_segments(self) -> list[DetectorSegment]:
        """All segments from all banks, in bank order.

        In a multi-panel plan layout a detector may appear in more than one bank
        (active in its home panel, ``read_only`` elsewhere), so this can contain
        several segments per ``detector_id``.  Use :attr:`active_segments` for one
        segment per selectable acquired channel.
        """
        segs: list[DetectorSegment] = []
        for bank in self.banks:
            segs.extend(bank.segments)
        return segs

    @property
    def active_segments(self) -> list[DetectorSegment]:
        """One clickable segment per detector (the non-``read_only`` segments)."""
        return [seg for seg in self.all_segments if not seg.read_only]

    @property
    def default_preset_name(self) -> str:
        """Name of the first (default) preset."""
        return next(iter(self.presets))


GENERIC_INSTRUMENT: Final = "Generic"


def generic_layout(n_detectors: int) -> InstrumentLayout:
    """Return a preset-free ring of *n_detectors* equal wedges.

    For instruments Asymmetry has no layout for: the drawing makes no claim
    about geometry, only gives each detector a clickable segment. Detector 1
    sits at the top and numbering runs clockwise.
    """
    segments = tuple(
        DetectorSegment(
            detector_id=d,
            sector_index=d - 1,
            ring_index=0,
            angle_center_deg=(90.0 - (d - 1) * 360.0 / n_detectors) % 360.0,
            angle_half_width_deg=180.0 / n_detectors,
            r_inner=0.28,
            r_outer=1.00,
        )
        for d in range(1, n_detectors + 1)
    )
    return InstrumentLayout(
        name=GENERIC_INSTRUMENT,
        n_detectors=n_detectors,
        banks=(BankLayout(name="Detectors", segments=segments),),
        presets={},
        display_name=f"Generic ({n_detectors} detectors)",
    )
