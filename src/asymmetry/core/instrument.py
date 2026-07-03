"""Instrument layout definitions for muon spectrometers.

This module provides static geometric descriptions of the detector arrangements
for HiFi, EMU, MuSR, PSI FLAME, PSI HAL-9500, and PSI GPS (in two variants — the
6-detector PSI-BIN layout ``GPS`` and the 11-detector ROOT sub-detector layout
``GPS-RD``, both shown to the user as "GPS"), along with standard grouping
presets.  The data here is used by the interactive detector layout editor
(:class:`~asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog`)
but has no GUI dependencies and can be used independently.

Some grouping presets follow musrfit's instrument definitions
(``musredit_qt5/musrWiz/instrument_defs/instrument_def_psi.xml``) so that GPS
analyses match the conventions PSI users already know — see the GPS ``WEP``
preset in :func:`_build_gps`.

Detector IDs are always **1-based** in this module, matching the instrument
manual conventions.  Conversion to 0-based indices for internal computation
is the responsibility of the caller.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "DetectorSegment",
    "BankLayout",
    "GroupDefinition",
    "AsymmetryProjection",
    "PresetGrouping",
    "ReferenceArrow",
    "InstrumentLayout",
    "INSTRUMENT_NAMES",
    "CANONICAL_VECTOR_AXES",
    "PROJECTION_TINTS",
    "TRANSVERSE_PROJECTION_TINTS",
    "derive_projection_pairs",
    "get_instrument_layout",
    "recommend_grouping_preset",
    "instrument_display_name",
    "instrument_choices_for",
    "variant_for_histograms",
    "detect_instrument",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorSegment:
    """Geometric description of a single detector element.

    Parameters
    ----------
    detector_id:
        1-based detector number matching instrument-manual convention.
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

    detector_id: int
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

    A forward/backward asymmetry *is* the muon polarization projected onto the
    axis joining that detector pair, so a preset that exposes several such pairs
    (EMU vector polarization, transverse-field dual grouping) is a set of
    projections.  Declaring them explicitly here replaces inferring them from
    canonical group-name strings.

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


#: Fixed semantic frame tints for the canonical EMU vector projections.  Chosen
#: away from the common run-colour palette so they read as chrome, not data.
PROJECTION_TINTS: Final[dict[str, str]] = {
    "P_x": "#534AB7",  # purple
    "P_y": "#BA7517",  # amber
    "P_z": "#0F6E56",  # teal
}

#: Fixed semantic frame tints for the transverse-field dual-grouping projections
#: (MuSR / HiFi).  Like :data:`PROJECTION_TINTS` these are muted and chosen away
#: from both the run-colour palette and the EMU vector tints, since the two
#: instruments are never shown together.  ``Top-Bottom`` keeps the same rose tint
#: on MuSR and HiFi for cross-instrument consistency.
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
        Ordered dict of preset names to :class:`PresetGrouping`.
        The first entry is the default preset.
    view:
        Schematic rendering mode, either ``"radial"`` or ``"plan"``.
    reference_arrows:
        Optional labelled arrows used by plan-view layouts.
    display_name:
        Optional user-facing name shown in the instrument dropdown.  Defaults to
        :attr:`name`.  Layout *variants* of one physical instrument (e.g. the
        6-detector PSI GPS BIN layout ``"GPS"`` and the 11-detector ROOT
        sub-detector layout ``"GPS-RD"``) share a display name so the user only
        ever sees the variant matching their loaded data, while ``name`` stays a
        distinct registry key for detection and persistence.
    """

    name: str
    n_detectors: int
    banks: tuple[BankLayout, ...]
    presets: dict[str, PresetGrouping]
    view: str = "radial"
    reference_arrows: tuple[ReferenceArrow, ...] = ()
    display_name: str | None = None

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
        segment per physical detector.
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


# ---------------------------------------------------------------------------
# Projection derivation
# ---------------------------------------------------------------------------

#: Canonical EMU vector-polarization axes, in payload order. The single source
#: of truth shared by the grouping dialog (per-projection alpha table ordering
#: and canonical detection) and the main window's alpha resolver, so the axis
#: set never drifts between them. The per-axis alpha table displays them P_z
#: first (``reversed``).
CANONICAL_VECTOR_AXES: Final[tuple[str, ...]] = ("P_x", "P_y", "P_z")

#: Legacy canonical vector group names → axis label, used to infer projections
#: for grouping payloads saved before projections were declared explicitly.
_LEGACY_VECTOR_AXIS_NAMES: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "P_x": (("px left",), ("px right",)),
    "P_y": (("py top", "py up"), ("py bottom", "py down")),
    "P_z": (("pz forward",), ("pz backward",)),
}


def derive_projection_pairs(
    groups: dict | None,
    group_names: dict | None = None,
    projections: object = None,
) -> dict[str, tuple[int, int]]:
    """Return ordered ``{label: (forward_gid, backward_gid)}`` for a grouping.

    This is the single source of truth for resolving a multi-projection grouping
    into its forward/backward pairs, shared by the grouping dialog and the main
    window.

    Resolution order:

    1. An explicit ``projections`` declaration — a sequence of
       :class:`AsymmetryProjection` or plain
       ``{"label", "forward_group", "backward_group"}`` dicts.  Only projections
       whose group IDs exist and are non-empty in ``groups`` are returned, in
       declaration order.
    2. Fallback: the legacy canonical EMU vector group names in ``group_names``,
       so groupings saved before projections were declared still resolve.  This
       path is all-or-nothing (it returns ``{}`` unless all three axes resolve),
       matching the original behaviour.
    """
    if not isinstance(groups, dict) or not groups:
        return {}

    norm_groups: dict[int, object] = {}
    for gid, members in groups.items():
        try:
            norm_groups[int(gid)] = members
        except (TypeError, ValueError):
            continue

    def _has(gid: int) -> bool:
        return gid in norm_groups and bool(norm_groups.get(gid))

    if projections:
        pairs: dict[str, tuple[int, int]] = {}
        for proj in projections:
            if isinstance(proj, AsymmetryProjection):
                label, fwd, bwd = proj.label, proj.forward_group, proj.backward_group
            elif isinstance(proj, dict):
                label = proj.get("label")
                fwd = proj.get("forward_group")
                bwd = proj.get("backward_group")
            else:
                continue
            try:
                fwd_i, bwd_i = int(fwd), int(bwd)
            except (TypeError, ValueError):
                continue
            if label and _has(fwd_i) and _has(bwd_i):
                pairs[str(label)] = (fwd_i, bwd_i)
        return pairs

    names = group_names if isinstance(group_names, dict) else {}
    by_name: dict[str, int] = {}
    for gid, name in names.items():
        try:
            by_name[str(name).strip().lower()] = int(gid)
        except (TypeError, ValueError):
            continue

    def _find(candidates: tuple[str, ...]) -> int | None:
        for cand in candidates:
            gid = by_name.get(cand)
            if gid is not None and _has(gid):
                return gid
        return None

    legacy_pairs: dict[str, tuple[int, int]] = {}
    for label, (fwd_names, bwd_names) in _LEGACY_VECTOR_AXIS_NAMES.items():
        fwd = _find(fwd_names)
        bwd = _find(bwd_names)
        if fwd is None or bwd is None:
            return {}
        legacy_pairs[label] = (fwd, bwd)
    return legacy_pairs


# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------

#: Registry keys of all supported instrument layouts.  ``"GPS"`` (6-detector
#: PSI-BIN) and ``"GPS-RD"`` (11-detector ROOT sub-detectors) are two variants of
#: the one physical GPS instrument; they share the display name "GPS" and are
#: collapsed to a single dropdown entry by :func:`instrument_choices_for`.
INSTRUMENT_NAMES: Final[tuple[str, ...]] = (
    "HiFi",
    "MuSR",
    "EMU",
    "FLAME",
    "HAL",
    "GPS",
    "GPS-RD",
)


# ---------------------------------------------------------------------------
# HiFi layout builder
# ---------------------------------------------------------------------------


def _build_hifi() -> InstrumentLayout:
    """Build the HiFi detector layout.

    HiFi has 64 detectors arranged in two circular banks of 32 each.
    Source: *HiFi User Manual*, pages 30–31.

    * Forward bank (1–32): detector *d* centred at
      ``(270 + (d-1) × 11.25) mod 360°``, numbering counter-clockwise.
    * Backward bank (33–64): detector *d* centred at
      ``(264.375 − (d−33) × 11.25) mod 360°``, numbering clockwise
      (the 0.5-sector offset is inferred from the manual figure).
    """
    pitch = 11.25  # degrees per sector (360/32)
    hw = pitch / 2.0  # half-width

    # Radial bounds for the single ring (normalised)
    r_inner = 0.28  # central beam hole
    r_outer = 1.00

    # --- Forward bank (detectors 1–32) ---
    fwd_segs: list[DetectorSegment] = []
    for d in range(1, 33):
        angle = (270.0 + (d - 1) * pitch) % 360.0
        fwd_segs.append(
            DetectorSegment(
                detector_id=d,
                sector_index=d - 1,
                ring_index=0,
                angle_center_deg=angle,
                angle_half_width_deg=hw,
                r_inner=r_inner,
                r_outer=r_outer,
            )
        )

    # --- Backward bank (detectors 33–64) ---
    bwd_segs: list[DetectorSegment] = []
    for d in range(33, 65):
        angle = (264.375 - (d - 33) * pitch) % 360.0
        bwd_segs.append(
            DetectorSegment(
                detector_id=d,
                sector_index=d - 33,
                ring_index=0,
                angle_center_deg=angle,
                angle_half_width_deg=hw,
                r_inner=r_inner,
                r_outer=r_outer,
            )
        )

    banks = (
        BankLayout(name="Forward", segments=tuple(fwd_segs)),
        BankLayout(name="Backward", segments=tuple(bwd_segs)),
    )

    # --- Presets ---
    presets: dict[str, PresetGrouping] = {}

    # Longitudinal (default)
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", tuple(range(1, 33))),
            2: GroupDefinition("Backward", tuple(range(33, 65))),
        },
        forward_group=1,
        backward_group=2,
    )

    # Transverse (Vector): both transverse detector pairs exposed as two
    # asymmetry projections of the same run, mirroring the EMU vector model.
    # Four distinct group IDs so the Left-Right and Top-Bottom pairs coexist
    # (the legacy split presets reused IDs 1/2 and were mutually exclusive).
    #   Left-Right:  Left  = 5–13 (forward) + 52–60 (backward)
    #                Right = 21–29 (forward) + 36–44 (backward)
    #   Top-Bottom:  Top    = 13–21 (forward) + 44–52 (backward)
    #                Bottom = 1–5, 29–36 (forward) + 60–64 (backward)
    presets["Transverse (Vector)"] = PresetGrouping(
        name="Transverse (Vector)",
        groups={
            1: GroupDefinition("Left-Right Left", tuple(range(5, 14)) + tuple(range(52, 61))),
            2: GroupDefinition("Left-Right Right", tuple(range(21, 30)) + tuple(range(36, 45))),
            3: GroupDefinition("Top-Bottom Top", tuple(range(13, 22)) + tuple(range(44, 53))),
            4: GroupDefinition(
                "Top-Bottom Bottom",
                tuple(range(1, 6)) + tuple(range(29, 37)) + tuple(range(60, 65)),
            ),
        },
        forward_group=1,
        backward_group=2,
        projections=(
            AsymmetryProjection("Left-Right", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Left-Right"]),
            AsymmetryProjection("Top-Bottom", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]),
        ),
    )

    return InstrumentLayout(
        name="HiFi",
        n_detectors=64,
        banks=banks,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# MuSR layout builder
# ---------------------------------------------------------------------------


def _build_musr() -> InstrumentLayout:
    """Build the MuSR detector layout.

    MuSR has 64 detectors arranged in two concentric rings of 32 each.
    Source: *MuSR Manual*, pages 12–13.

    * Backward ring (1–32): detector *d* centred at
      ``(230.625 + (d−1) × 11.25) mod 360°``, numbering counter-clockwise.
      Detectors 1–8 occupy the bottom quadrant.
    * Forward ring (33–64): detector *d* centred at
      ``(309.375 − (d−33) × 11.25) mod 360°``, numbering clockwise.
      Detectors 33–40 occupy the bottom quadrant, mirroring the backward ring.
    """
    pitch = 11.25  # degrees per sector
    hw = pitch / 2.0

    r_inner = 0.28
    r_outer = 1.00

    # --- Backward ring (detectors 1–32) ---
    bwd_segs: list[DetectorSegment] = []
    for d in range(1, 33):
        angle = (230.625 + (d - 1) * pitch) % 360.0
        bwd_segs.append(
            DetectorSegment(
                detector_id=d,
                sector_index=d - 1,
                ring_index=0,
                angle_center_deg=angle,
                angle_half_width_deg=hw,
                r_inner=r_inner,
                r_outer=r_outer,
            )
        )

    # --- Forward ring (detectors 33–64) ---
    fwd_segs: list[DetectorSegment] = []
    for d in range(33, 65):
        angle = (309.375 - (d - 33) * pitch) % 360.0
        fwd_segs.append(
            DetectorSegment(
                detector_id=d,
                sector_index=d - 33,
                ring_index=0,
                angle_center_deg=angle,
                angle_half_width_deg=hw,
                r_inner=r_inner,
                r_outer=r_outer,
            )
        )

    # MuSR manual labels the rings "Backward" and "Forward"; the backward ring
    # detectors (1-32) form the *backward* group in longitudinal mode.
    banks = (
        BankLayout(name="Backward", segments=tuple(bwd_segs)),
        BankLayout(name="Forward", segments=tuple(fwd_segs)),
    )

    # --- Presets ---
    presets: dict[str, PresetGrouping] = {}

    # Longitudinal: backward ring = backward group, forward ring = forward group
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Backward", tuple(range(1, 33))),
            2: GroupDefinition("Forward", tuple(range(33, 65))),
        },
        forward_group=2,
        backward_group=1,
    )

    # Transverse (Vector): both transverse detector pairs exposed as two
    # asymmetry projections of the same run, mirroring the EMU vector model.
    # Four distinct group IDs so the Top-Bottom and Fwd-Back pairs coexist
    # (the legacy split presets reused IDs 1/2 and were mutually exclusive).
    #   Top-Bottom:  Top    = 17–24 (backward ring) + 49–56 (forward ring)
    #                Bottom = 1–8   (backward ring) + 33–40 (forward ring)
    #   Fwd-Back:    Forward  = 9–16  (backward ring) + 57–64 (forward ring)
    #                Backward = 25–32 (backward ring) + 41–48 (forward ring)
    presets["Transverse (Vector)"] = PresetGrouping(
        name="Transverse (Vector)",
        groups={
            1: GroupDefinition("Top-Bottom Top", tuple(range(17, 25)) + tuple(range(49, 57))),
            2: GroupDefinition("Top-Bottom Bottom", tuple(range(1, 9)) + tuple(range(33, 41))),
            3: GroupDefinition("Fwd-Back Forward", tuple(range(9, 17)) + tuple(range(57, 65))),
            4: GroupDefinition("Fwd-Back Backward", tuple(range(25, 33)) + tuple(range(41, 49))),
        },
        forward_group=1,
        backward_group=2,
        projections=(
            AsymmetryProjection("Top-Bottom", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]),
            AsymmetryProjection("Fwd-Back", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Fwd-Back"]),
        ),
    )

    return InstrumentLayout(
        name="MuSR",
        n_detectors=64,
        banks=banks,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# EMU layout builder
# ---------------------------------------------------------------------------


def _build_emu() -> InstrumentLayout:
    """Build the EMU detector layout.

    EMU has 96 detectors arranged in two circular banks of 48 each.
    Each bank is divided into 16 azimuthal sectors, each containing three
    radial rings (inner, middle, outer). Numbering is sector-major triplets
    (inner/middle/outer of one sector before moving to the next), sector 0 is
    at 12 o'clock, and numbers increase clockwise as viewed looking upstream
    from downstream — matching the *EMU User Guide*, Section 8.1, and the
    Mantid EMU instrument definition file (detector 1 at azimuth 90°).

    Numbering formula for azimuth sector *s* (0–15):

    * Forward bank: inner = ``1 + 3s``, middle = ``2 + 3s``, outer = ``3 + 3s``.
    * Backward bank: inner = ``49 + 3s``, middle = ``50 + 3s``, outer = ``51 + 3s``.

    Angular convention: looking into the instrument from downstream.
    Sector 0 is at 12 o'clock (90°); numbers increase clockwise.
    Sector *s* centre: ``(90 − 22.5 × s) mod 360°``.

    **The three radial rings exist for stray-count rejection, not for the**
    **"Vector Polarization" preset below.** EMU's inner/middle/outer split lets
    an analysis discard counts from detectors more prone to stray-muon and
    frame-overlap background — see Giblin *et al.*, *Nucl. Instrum. Methods*
    *Phys. Res. A* **751**, 70 (2014). The facility does not document a
    "vector polarization" detector grouping for EMU: the ``Vector
    Polarization`` preset defined below (Px/Py/Pz octant composition) is an
    **Asymmetry construct**, not a facility-defined or published EMU
    convention. It is verified internally consistent against this module's
    own sector geometry (see ``tests/core/test_instrument.py::
    TestEMUVectorPolarizationGeometry``) but has no external oracle to check
    it against.
    """
    n_sectors = 16
    sector_pitch = 22.5  # degrees (360 / 16)
    hw = sector_pitch / 2.0

    # Three radial bands per sector (normalised to outer disc radius)
    radial_bounds = [
        (0.28, 0.52),  # ring 0 = inner (closest to sample)
        (0.52, 0.76),  # ring 1 = middle
        (0.76, 1.00),  # ring 2 = outer (furthest from sample)
    ]

    fwd_segs: list[DetectorSegment] = []
    bwd_segs: list[DetectorSegment] = []

    for s in range(n_sectors):
        angle_center = (90.0 - sector_pitch * s) % 360.0
        for ring_idx, (ri, ro) in enumerate(radial_bounds):
            # Forward bank
            det_fwd = 1 + 3 * s + ring_idx
            fwd_segs.append(
                DetectorSegment(
                    detector_id=det_fwd,
                    sector_index=s,
                    ring_index=ring_idx,
                    angle_center_deg=angle_center,
                    angle_half_width_deg=hw,
                    r_inner=ri,
                    r_outer=ro,
                )
            )
            # Backward bank
            det_bwd = 49 + 3 * s + ring_idx
            bwd_segs.append(
                DetectorSegment(
                    detector_id=det_bwd,
                    sector_index=s,
                    ring_index=ring_idx,
                    angle_center_deg=angle_center,
                    angle_half_width_deg=hw,
                    r_inner=ri,
                    r_outer=ro,
                )
            )

    banks = (
        BankLayout(name="Forward", segments=tuple(fwd_segs)),
        BankLayout(name="Backward", segments=tuple(bwd_segs)),
    )

    # --- Presets ---
    presets: dict[str, PresetGrouping] = {}

    # Longitudinal (default)
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", tuple(range(1, 49))),
            2: GroupDefinition("Backward", tuple(range(49, 97))),
        },
        forward_group=1,
        backward_group=2,
    )

    # --- Vector polarization: 6 groups across Px, Py, Pz axes ---
    #
    # This is an in-house (non-facility-documented) grouping: each bank is
    # split into four 12-detector quadrants (4 sectors x 3 rings) by the
    # geometric half-plane of the sector centre (sector s at angle
    # (90 - 22.5*s) mod 360, matplotlib polar convention: 0 deg = east,
    # 90 deg = north/12 o'clock, CCW positive):
    #   sectors 13-3 (13,14,15,0,1,2,3) = upper half  (y > 0), Py Top
    #   sectors 5-11  (5,6,7,8,9,10,11) = lower half  (y < 0), Py Bottom
    #   sectors 9-15  (9,10,11,12,13,14,15) = left half (x < 0), Px Left
    #   sectors 1-7   (1,2,3,4,5,6,7)   = right half (x > 0), Px Right
    # The four sectors centred exactly on an axis (0 = due north/top,
    # 4 = due east/right, 8 = due south/bottom, 12 = due west/left) are
    # boundary cases with y or x == 0 and are assigned to the neighbouring
    # half that keeps each quadrant an equal 8 sectors: sector 0 -> Right,
    # sector 4 -> Bottom, sector 8 -> Left, sector 12 -> Top. This partition
    # is verified against the layout's own DetectorSegment angles in
    # tests/core/test_instrument.py::TestEMUVectorPolarizationGeometry.
    #
    # Composite channel definitions:
    #   Pz from forward/backward bank sums
    #   Py from top/bottom half sums (across both banks)
    #   Px from left/right half sums (across both banks)
    #
    # This intentionally creates overlap between Pz groups and transverse groups.

    def _emu_sector_ids(sectors: list[int], bank_offset: int) -> tuple[int, ...]:
        """Return all detector IDs (inner+middle+outer) for ``sectors`` in one bank."""
        ids: list[int] = []
        for s in sectors:
            for ring_idx in range(3):
                ids.append(bank_offset + 3 * s + ring_idx)
        return tuple(sorted(ids))

    fwd_offset = 1  # Forward bank: 1 + 3*s + ring_idx
    bwd_offset = 49  # Backward bank: 49 + 3*s + ring_idx

    top_sectors = [12, 13, 14, 15, 0, 1, 2, 3]
    bottom_sectors = [4, 5, 6, 7, 8, 9, 10, 11]
    left_sectors = [8, 9, 10, 11, 12, 13, 14, 15]
    right_sectors = [0, 1, 2, 3, 4, 5, 6, 7]

    presets["Vector Polarization"] = PresetGrouping(
        name="Vector Polarization",
        groups={
            1: GroupDefinition("Pz Forward", tuple(range(1, 49))),
            2: GroupDefinition("Pz Backward", tuple(range(49, 97))),
            3: GroupDefinition(
                "Py Top",
                _emu_sector_ids(top_sectors, fwd_offset) + _emu_sector_ids(top_sectors, bwd_offset),
            ),
            4: GroupDefinition(
                "Py Bottom",
                _emu_sector_ids(bottom_sectors, fwd_offset)
                + _emu_sector_ids(bottom_sectors, bwd_offset),
            ),
            5: GroupDefinition(
                "Px Left",
                _emu_sector_ids(left_sectors, fwd_offset)
                + _emu_sector_ids(left_sectors, bwd_offset),
            ),
            6: GroupDefinition(
                "Px Right",
                _emu_sector_ids(right_sectors, fwd_offset)
                + _emu_sector_ids(right_sectors, bwd_offset),
            ),
        },
        forward_group=1,
        backward_group=2,
        projections=(
            AsymmetryProjection("P_x", 5, 6, tint=PROJECTION_TINTS["P_x"]),
            AsymmetryProjection("P_y", 3, 4, tint=PROJECTION_TINTS["P_y"]),
            AsymmetryProjection("P_z", 1, 2, tint=PROJECTION_TINTS["P_z"]),
        ),
    )

    return InstrumentLayout(
        name="EMU",
        n_detectors=96,
        banks=banks,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# PSI FLAME layout builder
# ---------------------------------------------------------------------------


def _plan_rectangle(
    detector_id: int,
    name: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    read_only: bool = False,
) -> DetectorSegment:
    """Return one rectangular detector plate for a plan-view layout (FLAME, GPS)."""
    return DetectorSegment(
        detector_id=detector_id,
        sector_index=detector_id - 1,
        ring_index=0,
        angle_center_deg=0.0,
        angle_half_width_deg=0.0,
        r_inner=0.0,
        r_outer=0.0,
        shape="rectangle",
        label=name,
        x_center=x,
        y_center=y,
        width=width,
        height=height,
        read_only=read_only,
    )


def _build_flame() -> InstrumentLayout:
    """Build the PSI FLAME detector layout.

    FLAME uses eight rectangular detector plates.  The plan-view geometry below
    follows the published detector-name / histogram-number convention:

    * 1 = Forward, 2 = Backward
    * 3 = Right, 4 = Left
    * 5 = R_F, 6 = R_B, 7 = L_F, 8 = L_B

    The schematic is drawn from above, with the beam and main magnetic field
    along +z and transverse left/right along y.
    """
    segments = (
        _plan_rectangle(1, "Forward", 3.55, 0.0, 0.74, 1.72),
        _plan_rectangle(2, "Backward", -3.55, 0.0, 0.74, 1.72),
        _plan_rectangle(3, "Right", 0.0, -2.18, 2.18, 0.82),
        _plan_rectangle(4, "Left", 0.0, 2.18, 2.18, 0.82),
        _plan_rectangle(5, "R_F", 1.55, -2.18, 0.82, 0.82),
        _plan_rectangle(6, "R_B", -1.55, -2.18, 0.82, 0.82),
        _plan_rectangle(7, "L_F", 1.55, 2.18, 0.82, 0.82),
        _plan_rectangle(8, "L_B", -1.55, 2.18, 0.82, 0.82),
    )
    banks = (BankLayout(name="FLAME top view", segments=segments),)

    presets: dict[str, PresetGrouping] = {}
    # FLAME is a PSI instrument: Forward/Backward name the *beam* directions, and
    # for surface muons the spin is antiparallel to the momentum so the initial
    # polarization points toward the Backward detector.  Declaring the
    # analysis-forward slot as the Backward-named group (2) makes the preset match
    # the PSI convention A = (B − αF)/(B + αF) standalone (headless core, no GUI
    # swap to compensate) — the same treatment applied to the GPS presets.
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", (1,)),
            2: GroupDefinition("Backward", (2,)),
        },
        forward_group=2,  # analysis-forward = beam-Backward group (sees polarization)
        backward_group=1,  # analysis-backward = beam-Forward group
    )
    # Transverse: Right vs Left. musrfit ships no FLAME transverse definition, so
    # there is no reference leg order to match; the two are a transverse pair
    # whose order only sets the phase convention. Left as declared.
    presets["Transverse"] = PresetGrouping(
        name="Transverse",
        groups={
            1: GroupDefinition("Right", (3, 6, 5)),
            2: GroupDefinition("Left", (4, 8, 7)),
        },
        forward_group=1,
        backward_group=2,
    )

    return InstrumentLayout(
        name="FLAME",
        n_detectors=8,
        banks=banks,
        presets=presets,
        view="plan",
        reference_arrows=(
            ReferenceArrow(
                "beam, main magnetic field",
                (-2.55, -0.82),
                (2.55, -0.82),
                "#202020",
            ),
            ReferenceArrow("initial muon spin", (0.0, 0.25), (-2.75, 0.25), "#5b2ea6"),
        ),
    )


# ---------------------------------------------------------------------------
# PSI GPS layout builder
# ---------------------------------------------------------------------------


def _gps_endon(detector_id: int, name: str, x: float, y: float, *, into: bool) -> DetectorSegment:
    """Return a GPS detector seen end-on (⊙ toward / ⊗ away from the viewer).

    End-on segments are always ``read_only`` context: the detector points
    perpendicular to the current panel, so it is edited in the panel where it
    lies in-plane.
    """
    return DetectorSegment(
        detector_id=detector_id,
        sector_index=detector_id - 1,
        ring_index=0,
        angle_center_deg=0.0,
        angle_half_width_deg=0.0,
        r_inner=0.0,
        r_outer=0.0,
        shape="endon_in" if into else "endon_out",
        label=name,
        x_center=x,
        y_center=y,
        read_only=True,
    )


#: Longitudinal preset shared by both GPS variants.
def _gps_longitudinal_preset() -> PresetGrouping:
    """Return the GPS Longitudinal preset in the PSI *analysis* convention.

    PSI names the two beam-axis detectors by **beam** direction (Forward = +z
    downstream, Backward = −z upstream), so the group *names* stay "Forward"/
    "Backward" (physical convention).  For surface muons the spin is antiparallel
    to the momentum, so the initial polarization points toward the **Backward**
    detector — that is the group that sees the polarization and must occupy the
    analysis-forward slot.  musrfit's GPS ZF/LF definitions encode exactly this
    with ``logic_asym_detector FB forward=2(B) backward=1(F)``
    (``instrument_def_psi.xml``), giving A = (B − αF)/(B + αF) as in the GPS
    instrument paper (Amato *et al.*, Rev. Sci. Instrum. 88, 093301 (2017),
    Eq. 2).  We therefore declare ``forward_group`` = the Backward-named group
    (2) and ``backward_group`` = the Forward-named group (1) so the preset is
    correct standalone (headless core, with no GUI swap to compensate).
    """
    return PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", (1,)),
            2: GroupDefinition("Backward", (2,)),
        },
        forward_group=2,  # analysis-forward = beam-Backward group (sees polarization)
        backward_group=1,  # analysis-backward = beam-Forward group
    )


def _gps_presets(
    up: tuple[int, ...],
    down: tuple[int, ...],
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> dict[str, PresetGrouping]:
    """Build the shared GPS preset set for a variant.

    *up/down/left/right* are the detector-id tuples for each transverse direction
    — one id each for the 6-detector BIN layout, the two ``_B``/``_F``
    sub-detector ids for the 11-detector ROOT layout.  Forward and Backward are
    always detectors 1 and 2.  Both GPS variants share this builder so the preset
    physics lives in one place and cannot drift between the two.
    """
    return {
        "Longitudinal": _gps_longitudinal_preset(),
        # Transverse (Vector): rotator-off transverse geometry. The spin is
        # antiparallel to the beam and precesses, so leg order only sets the
        # phase convention. We match musrfit's GPS ``WED(L)`` definition
        # (``instrument_def_psi.xml``): ``UD forward=3(U) backward=4(D)`` and
        # ``RL forward=5(R) backward=6(L)``. Group 1 = Up, 2 = Down give the
        # Up-Down pair forward=Up; group 3 = Right, 4 = Left give the Left-Right
        # pair forward=Right, both matching musrfit.
        "Transverse (Vector)": PresetGrouping(
            name="Transverse (Vector)",
            groups={
                1: GroupDefinition("Up-Down Up", up),
                2: GroupDefinition("Up-Down Down", down),
                3: GroupDefinition("Left-Right Right", right),
                4: GroupDefinition("Left-Right Left", left),
            },
            forward_group=1,
            backward_group=2,
            projections=(
                AsymmetryProjection(
                    "Up-Down", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]
                ),
                AsymmetryProjection(
                    "Left-Right", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Left-Right"]
                ),
            ),
        ),
        # Spin-rotated: with the spin rotator in transverse geometry the muon spin
        # is tipped up by ~50 deg (GPS User Guide, Section 13).
        #
        # Geometry. For surface muons the *initial* polarization is antiparallel
        # to the momentum, i.e. it points toward the Backward detector (−z). The
        # rotator then tips it up (+y). The rotated polarization therefore points
        # up-and-*backward* and lies in the plane between the **Backward** and
        # **Up** detectors — NOT Forward+Up. The group that sees the rotated
        # polarization (analysis-forward) is thus **B+U**, and the opposed group
        # (analysis-backward) is **F+D**. This is corroborated by musrfit's WEP
        # phases (B rel_phase=−45, U=+45, F=+135, D=+225): the polarization axis
        # bisects B and U. (The old F+U/B+D grouping put the up-and-*forward*
        # diagonal in the forward slot, which is roughly orthogonal to the true
        # polarization and loses amplitude — the bug this preset fix corrects.)
        # The name keeps the physical B/U/F/D detector letters.
        "Spin-rotated (B+U/F+D)": PresetGrouping(
            name="Spin-rotated (B+U/F+D)",
            groups={
                1: GroupDefinition("B+U", (2, *up)),
                2: GroupDefinition("F+D", (1, *down)),
            },
            forward_group=1,  # B+U sees the rotated polarization (analysis-forward)
            backward_group=2,  # F+D is the opposed group
        ),
        # WEP: follows musrfit's spin-rotated GPS setup -- F/B/U/D as four groups
        # exposed as the FB and UD asymmetry pairs, each reduced with its own
        # declared alpha (FB = 0.75, UD = 1.0). musrfit declares
        # ``FB forward=2(B) backward=1(F) alpha=0.75`` and
        # ``UD forward=3(U) backward=4(D) alpha=1.0``, so the FB projection's
        # forward leg is the Backward-named group and the UD forward leg is Up.
        # See _build_gps for the verbatim musrfit definition and provenance.
        "WEP (spin-rotated)": PresetGrouping(
            name="WEP (spin-rotated)",
            groups={
                1: GroupDefinition("F", (1,)),
                2: GroupDefinition("B", (2,)),
                3: GroupDefinition("U", up),
                4: GroupDefinition("D", down),
            },
            forward_group=2,  # analysis-forward = B (matches FB projection below)
            backward_group=1,  # analysis-backward = F
            projections=(
                # FB: A = (B − 0.75 F)/(B + 0.75 F). forward=B(group 2),
                # backward=F(group 1), alpha=0.75 -- musrfit's leg order and sign.
                AsymmetryProjection(
                    "FB", 2, 1, alpha=0.75, tint=TRANSVERSE_PROJECTION_TINTS["Fwd-Back"]
                ),
                # UD: forward=U(group 3), backward=D(group 4), alpha=1.0.
                AsymmetryProjection("UD", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]),
            ),
        ),
    }


def _build_gps() -> InstrumentLayout:
    """Build the PSI GPS detector layout (General Purpose Spectrometer, πM3.2).

    GPS has six positron detectors defined with respect to the beam direction —
    **Forward (F), Backward (B), Up (U), Down (D), Right (R), Left (L)** — around
    the sample, plus a muon counter (M) and veto detectors (a backward veto
    pyramid and a forward veto) that feed the coincidence logic but are *not*
    stored as histograms.  U/D/R/L are each two physical subdetectors read by
    SiPM arrays; in the PSI-BIN export each direction is one combined plate.
    Source: *GPS User Guide* (A. Amato & H. Luetkens), Section 4 "The detectors".

    PSI GPS ``deltat_tdc_gps_*.bin`` files store the six histograms in the fixed
    order ``Forw, Back, Up, Down, Righ, Left`` (verified across runs from 2016
    to 2025).  Because detector IDs in this module map positionally to histogram
    indices (detector *N* → histogram ``N − 1``), the IDs are F→1, B→2, U→3,
    D→4, R→5, L→6.

    The six detectors lie on three orthogonal axes, so the schematic uses **two
    plan panels** (following the FLAME beam convention, beam → +z = Forward):

    * **Top view** (looking down) — the horizontal plane: Forward/Backward on the
      beam axis, Left/Right in-plane (top/bottom); Up/Down point out of the page
      and are drawn end-on (⊙/⊗) for context.
    * **Side view** (from the left) — the vertical plane: Up/Down in-plane,
      Forward/Backward shown read-only; Left/Right point out of the page (⊙/⊗).

    Each detector is clickable (active) in its home panel and read-only in the
    other, which keeps the 3-D geometry legible.

    **PSI beam-vs-analysis convention.** PSI names detectors by **beam**
    direction: Forward (F) is downstream (+z), Backward (B) upstream (−z).  For
    surface muons the spin is *antiparallel* to the momentum, so the initial
    polarization points toward the **Backward** detector.  The PSI/musrfit
    analysis convention (GPS instrument paper, Amato *et al.*,
    Rev. Sci. Instrum. 88, 093301 (2017), Eq. 2) is therefore

        A = (B − α F) / (B + α F),

    i.e. the *Backward*-named group occupies the analysis-**forward** slot.  All
    GPS presets here declare that analysis convention directly in their
    ``forward_group``/``backward_group`` (and projection legs) while keeping the
    physical beam-referenced group *names* (F/B/U/D), so a preset is correct
    standalone in the headless core — no GUI swap is needed to compensate.

    **Spin rotator and the B+U / F+D preset.** GPS sits behind a spin rotator on
    πM3.2 — crossed electric and magnetic fields (a Wien filter), GPS User Guide
    Section 13.  In *longitudinal geometry* it acts as a velocity separator that
    cleans positron contamination from the beam (and tips the polarization up by
    ~7°).  In *transverse geometry* the field is raised so the muon spin is
    rotated **up by ~50°**, giving a large transverse component for TF work.  The
    initial polarization points toward **Backward** (−z, surface-muon geometry);
    rotating it up by ~45–50° in the vertical plane leaves it pointing
    up-and-*backward*, along the **Backward–Up diagonal**.  A plain
    Forward/Backward asymmetry then only sees its cosine projection and loses
    amplitude.  Summing **Backward+Up** (which sees the rotated spin) against
    **Forward+Down** realigns one asymmetry axis with the rotated spin and
    recovers the full amplitude — this is the ``Spin-rotated (B+U/F+D)`` preset.
    (The bisector of musrfit's WEP phases below, B at −45° and U at +45°,
    confirms the polarization lies along the B–U diagonal, not F–U.)

    **WEP mode — follows musrfit.** musrfit ships a GPS setup named
    ``WEP`` for exactly this spin-rotated configuration, and the
    ``WEP (spin-rotated)`` preset reproduces its grouping convention so analyses
    match what PSI users already expect.  Source:
    ``musredit_qt5/musrWiz/instrument_defs/instrument_def_psi.xml``,
    ``<instrument name="GPS"><tf name="WEP">``.  musrfit does **not** sum
    detectors; it keeps F, B, U, D as four separate logic detectors, each given a
    relative phase that encodes the ~45° rotation, and fits two asymmetry pairs::

        logic_detector       B  rel_phase=-45   forward=2  (histogram 1)
        logic_detector       U  rel_phase=+45   forward=3  (histogram 2)
        logic_detector       F  rel_phase=+135  forward=1  (histogram 0)
        logic_detector       D  rel_phase=+225  forward=4  (histogram 3)
        logic_asym_detector  FB forward=2 backward=1  alpha=0.75
        logic_asym_detector  UD forward=3 backward=4  alpha=1.0

    The preset here maps F/B/U/D to four groups exposed as the **FB** and **UD**
    projections, following musrfit's leg order: **FB** has forward = B (group 2),
    backward = F (group 1) so it reduces to (B − 0.75 F)/(B + 0.75 F), and **UD**
    has forward = U (group 3), backward = D (group 4).  The FB projection declares
    musrfit's default ``alpha = 0.75`` and the UD projection ``alpha = 1.0``; the
    reduction **applies each projection's own declared alpha**, so reducing or
    fitting the FB pair uses 0.75 and the UD pair uses 1.0 (see
    :meth:`asymmetry.gui.mainwindow.MainWindow._resolve_vector_alpha_values`).
    The per-detector phase offsets musrfit uses to encode the rotation are a
    fitting detail not stored in the layout — only the groupings.
    (``Spin-rotated (B+U/F+D)`` above is *our* combined-pair alternative for the
    same physics, not a musrfit construct.)
    """
    top = BankLayout(
        name="Top view",
        segments=(
            _plan_rectangle(1, "Forward", 2.70, 0.0, 0.75, 1.70),
            _plan_rectangle(2, "Backward", -2.70, 0.0, 0.75, 1.70),
            _plan_rectangle(6, "Left", 0.0, 1.95, 1.70, 0.75),
            _plan_rectangle(5, "Right", 0.0, -1.95, 1.70, 0.75),
            _gps_endon(3, "Up", 0.0, 0.70, into=False),
            _gps_endon(4, "Down", 0.0, -0.70, into=True),
        ),
    )
    side = BankLayout(
        name="Side view",
        segments=(
            _plan_rectangle(3, "Up", 0.0, 1.95, 1.70, 0.75),
            _plan_rectangle(4, "Down", 0.0, -1.95, 1.70, 0.75),
            _plan_rectangle(1, "Forward", 2.70, 0.0, 0.75, 1.70, read_only=True),
            _plan_rectangle(2, "Backward", -2.70, 0.0, 0.75, 1.70, read_only=True),
            _gps_endon(6, "Left", 0.0, 0.70, into=False),
            _gps_endon(5, "Right", 0.0, -0.70, into=True),
        ),
    )

    # Transverse directions map to single combined detectors in the BIN export:
    # U=3, D=4, L=6, R=5.
    presets = _gps_presets((3,), (4,), (6,), (5,))

    return InstrumentLayout(
        name="GPS",
        n_detectors=6,
        banks=(top, side),
        presets=presets,
        view="plan",
    )


def _build_gps_subdetectors() -> InstrumentLayout:
    """Build the PSI GPS **ROOT sub-detector** layout (11 histograms).

    GPS ``deltat_tdc_gps_*.root`` (MusrRoot) files expose the *raw* sub-detectors
    rather than the six combined detectors of the PSI-BIN export.  The
    transverse detectors are each physically split into a backward (upstream,
    ``_B``) and forward (downstream, ``_F``) part, and a Mobile detector is
    added.  The histograms appear in this fixed order (verified on 2025 data):

    ``Forw, Back, Up_B, Up_F, Down_B, Down_F, Right_B, Right_F, Left_B, Left_F,
    Mob-RL`` → detector IDs 1…11 (detector *N* → histogram ``N − 1``).

    Source: Amato *et al.*, *Rev. Sci. Instrum.* 88, 093301 (2017),
    `arXiv:1705.10687 <https://arxiv.org/abs/1705.10687>`_, Section IIA:
    ``R = R_back + R_forw (+ P_mob)``, ``L = L_back + L_forw (+ P_mob)``, and
    "the detectors U and D are also physically split between a forward and a
    backward part".

    The **Mobile** detector (``P_mob`` / ``Mob-RL``) is mounted on the two
    cryostat ports on the horizontal axis ⊥ to the beam; it is added to ``R``
    (first port) *or* ``L`` (second port) depending on which port is in use —
    information **not recorded in the data file**.  It is therefore left
    *ungrouped* by default (matching musrfit, whose GPS definition does not
    group it); the user assigns it to the side matching their setup.

    This is the 11-detector variant of :func:`_build_gps`; both share the
    display name "GPS" (see :class:`InstrumentLayout.display_name`) and the same
    two-panel (top + side) presentation.  Each transverse plate is split along
    the beam into a backward (``_B``, upstream/−z) and forward (``_F``,
    downstream/+z) half, so split detectors render as two boxes side by side.
    The **Mobile** detector rides on the horizontal Left–Right axis between the
    cryostat ports; following Amato *et al.* Fig. 2 it is drawn as a bar at the
    top of the chamber, just above the sample, and is left ungrouped (added to R
    or L per the cryo port, which the file does not record).
    """
    top = BankLayout(
        name="Top view",
        segments=(
            _plan_rectangle(1, "Forward", 2.95, 0.0, 0.78, 1.70),
            _plan_rectangle(2, "Backward", -2.95, 0.0, 0.78, 1.70),
            # In-plane Left (top) / Right (bottom), each split _B (−z) / _F (+z).
            _plan_rectangle(9, "Left_B", -0.50, 1.95, 0.85, 0.75),
            _plan_rectangle(10, "Left_F", 0.50, 1.95, 0.85, 0.75),
            _plan_rectangle(7, "Right_B", -0.50, -1.95, 0.85, 0.75),
            _plan_rectangle(8, "Right_F", 0.50, -1.95, 0.85, 0.75),
            # Mobile bar at the top of the chamber, above the sample (Fig. 2).
            _plan_rectangle(11, "Mob-RL", 0.0, 1.18, 0.95, 0.42),
            # Up/Down point out of the page here -> end-on context markers.
            _gps_endon(3, "Up_B", -0.48, 0.45, into=False),
            _gps_endon(4, "Up_F", 0.48, 0.45, into=False),
            _gps_endon(5, "Down_B", -0.48, -0.45, into=True),
            _gps_endon(6, "Down_F", 0.48, -0.45, into=True),
        ),
    )
    side = BankLayout(
        name="Side view",
        segments=(
            # In-plane Up (top) / Down (bottom), each split _B (−z) / _F (+z).
            _plan_rectangle(3, "Up_B", -0.50, 1.95, 0.85, 0.75),
            _plan_rectangle(4, "Up_F", 0.50, 1.95, 0.85, 0.75),
            _plan_rectangle(5, "Down_B", -0.50, -1.95, 0.85, 0.75),
            _plan_rectangle(6, "Down_F", 0.50, -1.95, 0.85, 0.75),
            _plan_rectangle(1, "Forward", 2.95, 0.0, 0.78, 1.70, read_only=True),
            _plan_rectangle(2, "Backward", -2.95, 0.0, 0.78, 1.70, read_only=True),
            # Left/Right point out of the page here; Mobile rides this same axis.
            _gps_endon(9, "Left_B", -0.48, 0.45, into=False),
            _gps_endon(10, "Left_F", 0.48, 0.45, into=False),
            _gps_endon(7, "Right_B", -0.48, -0.45, into=True),
            _gps_endon(8, "Right_F", 0.48, -0.45, into=True),
            _gps_endon(11, "Mob-RL", 0.0, 1.30, into=False),
        ),
    )

    # Each transverse direction combines its two _B/_F sub-detectors:
    # U=(3,4), D=(5,6), L=(9,10), R=(7,8). Mob-RL (11) is left ungrouped.
    presets = _gps_presets((3, 4), (5, 6), (9, 10), (7, 8))

    return InstrumentLayout(
        name="GPS-RD",
        n_detectors=11,
        banks=(top, side),
        presets=presets,
        view="plan",
        display_name="GPS",
    )


# ---------------------------------------------------------------------------
# HAL-9500 layout builder
# ---------------------------------------------------------------------------


def _build_hal() -> InstrumentLayout:
    """Build the HAL-9500 detector layout (PSI high-field spectrometer).

    HAL-9500 (πE3 beamline) has 16 positron detectors arranged as two
    octagonal rings of eight — a **forward** ring (F1–F8) and a **backward**
    ring (B1–B8) — plus a central muon-veto detector (MV).  Viewed along the
    beam axis the two rings project onto the same octagon.

    This is a distinct instrument from the ISIS *HiFi* spectrometer
    (:func:`_build_hifi`); the shared ``hifi`` token in PSI run names
    (``tdc_hifi_*``) is a legacy naming collision.

    PSI data files store the histograms in the fixed order
    ``MV, F1…F8, B1…B8``.  Because detector IDs in this module map positionally
    to histogram indices (detector *N* → histogram ``N − 1``), the IDs are:

    * MV → 1   (histogram 0, the muon veto)
    * F1–F8 → 2–9   (histograms 1–8, forward ring)
    * B1–B8 → 10–17 (histograms 9–16, backward ring)

    Source: `PSI HAL-9500 instrument page
    <https://www.psi.ch/en/smus/hal-9500>`_; musrfit ``instrument_def_psi.xml``
    (``<instrument name="HAL9500">``) and the shipped ``tdc_hifi_2014_00153``
    example data (see ``tests/test_psi_loader.py``).
    """
    pitch = 45.0  # degrees per sector (360/8)
    # Each detector is a rectangular bar lying along one edge of the octagon
    # (constant-width slabs forming the perimeter), matching the schematic.
    apothem = math.cos(math.radians(22.5))  # distance to an octagon edge midpoint
    thickness = 0.26  # radial depth of each bar
    bar_width = 0.54  # tangential length; small gaps at the octagon vertices
    center_r = apothem - thickness / 2.0  # bar centre; outer face sits on the octagon

    def _ring(start_id: int, prefix: str) -> list[DetectorSegment]:
        segs: list[DetectorSegment] = []
        for k in range(8):
            # k = 0 at the top (90°), numbering clockwise to match the schematic.
            angle = (90.0 - k * pitch) % 360.0
            rad = math.radians(angle)
            segs.append(
                DetectorSegment(
                    detector_id=start_id + k,
                    sector_index=k,
                    ring_index=0,
                    angle_center_deg=angle,
                    angle_half_width_deg=0.0,
                    r_inner=0.0,
                    r_outer=0.0,
                    shape="rectangle",  # edge-aligned bar forming one octagon side
                    label=f"{prefix}{k + 1}",
                    x_center=center_r * math.cos(rad),
                    y_center=center_r * math.sin(rad),
                    width=bar_width,
                    height=thickness,
                    rotation_deg=angle - 90.0,  # long edge tangential to the ring
                )
            )
        return segs

    fwd_segs = _ring(2, "F")  # F1–F8 -> detector IDs 2–9
    bwd_segs = _ring(10, "B")  # B1–B8 -> detector IDs 10–17

    # Central muon veto (MV) -> detector ID 1, drawn as a disc at the ring centre.
    # half_width 180° is rendered/hit-tested as a full circle by the widget.
    mv = DetectorSegment(
        detector_id=1,
        sector_index=0,
        ring_index=0,
        angle_center_deg=90.0,
        angle_half_width_deg=180.0,
        r_inner=0.0,
        r_outer=0.26,
        label="MV",
    )

    banks = (
        BankLayout(name="Forward", segments=(mv, *fwd_segs)),
        BankLayout(name="Backward", segments=tuple(bwd_segs)),
    )

    fwd_ring = tuple(range(2, 10))  # F1–F8
    bwd_ring = tuple(range(10, 18))  # B1–B8

    presets: dict[str, PresetGrouping] = {}

    # Longitudinal (default): forward ring vs backward ring.  HAL-9500 is a PSI
    # instrument, so the rings are named by *beam* direction; for surface muons
    # the initial polarization points toward the Backward ring.  Declare the
    # analysis-forward slot as the Backward ring (group 2) so the preset follows
    # the PSI convention A = (B − αF)/(B + αF) standalone — matching the GPS/FLAME
    # presets.  (The Transverse/Per-octant presets below use opposed detector
    # pairs, not a beam F/B split, so their pairing is left as musrfit's.)
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", fwd_ring),
            2: GroupDefinition("Backward", bwd_ring),
        },
        forward_group=2,  # analysis-forward = beam-Backward ring (sees polarization)
        backward_group=1,  # analysis-backward = beam-Forward ring
    )

    # Transverse (opposed pairs): musrfit's high-field scheme — each forward
    # detector is its own group so any diametrically-opposed pair (F_k vs
    # F_{k+4}, 180° apart) can form the asymmetry.  Defaults to F1 vs F5 (0°).
    presets["Transverse (opposed pairs)"] = PresetGrouping(
        name="Transverse (opposed pairs)",
        groups={k + 1: GroupDefinition(f"F{k + 1}", (2 + k,)) for k in range(8)},
        forward_group=1,  # F1
        backward_group=5,  # F5 (180° opposite F1)
    )

    # Per-octant: one group per azimuthal sector, combining the forward and
    # backward wedge at that angle.  Useful for angle-resolved high-field work.
    presets["Per-octant"] = PresetGrouping(
        name="Per-octant",
        groups={k + 1: GroupDefinition(f"Octant {k + 1}", (2 + k, 10 + k)) for k in range(8)},
        forward_group=1,  # octant 1 (top)
        backward_group=5,  # octant 5 (bottom, 180° opposite)
    )

    return InstrumentLayout(
        name="HAL",
        n_detectors=17,
        banks=banks,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LAYOUTS: dict[str, InstrumentLayout] | None = None


def _build_registry() -> dict[str, InstrumentLayout]:
    return {
        "HiFi": _build_hifi(),
        "MuSR": _build_musr(),
        "EMU": _build_emu(),
        "FLAME": _build_flame(),
        "HAL": _build_hal(),
        "GPS": _build_gps(),
        "GPS-RD": _build_gps_subdetectors(),
    }


def get_instrument_layout(name: str) -> InstrumentLayout:
    """Return the :class:`InstrumentLayout` for *name*.

    Parameters
    ----------
    name:
        One of the names in :data:`INSTRUMENT_NAMES` (case-sensitive).

    Raises
    ------
    KeyError
        If *name* is not a known instrument.
    """
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = _build_registry()
    if name in _LAYOUTS:
        return _LAYOUTS[name]
    resolved = _canonical_instrument_name(name)
    if resolved is not None:
        return _LAYOUTS[resolved]
    return _LAYOUTS[name]


def recommend_grouping_preset(layout: InstrumentLayout, field_direction: str | None) -> str | None:
    """Recommend a grouping preset for *layout* given the run's field geometry.

    A pure helper the Grouping / Detector-Layout dialog uses to nudge the user
    away from a grouping that washes out the signal — never to auto-apply one.

    Motivating case (B8a, live testing): PSI GPS transverse-field (TF) runs
    default to the ``Longitudinal`` (Forward/Backward) preset, but the spin
    rotator on πM3.2 tips the muon spin ~50° up (see :func:`_build_gps`), so a
    plain Forward/Backward asymmetry only sees the cosine projection and the
    time-domain fit collapses.  The remedy is the ``Spin-rotated (B+U/F+D)``
    preset, which realigns one asymmetry axis with the rotated spin.  Nothing in
    the GUI hinted at this, which caused a long dead-end in live testing.

    Parameters
    ----------
    layout:
        The instrument layout whose presets are the recommendation candidates.
    field_direction:
        The applied-field geometry from run metadata
        (``metadata["field_direction"]``): ``"Transverse"``, ``"Longitudinal"``,
        ``"Zero field"``, or an empty/unknown string.  These values are produced
        by
        :meth:`asymmetry.core.io.nexus.NeXusMuonLoader._field_direction_from_state`
        and the PSI free-text classifier
        :func:`asymmetry.core.io.base.field_direction_from_text`.

    Returns
    -------
    str or None
        The name of a preset in ``layout.presets`` to recommend, or ``None``
        when no nudge applies: non-transverse geometry (longitudinal / zero
        field), an unknown/absent field direction, or no suitable transverse
        preset on this instrument.  For transverse data a single forward/backward
        pair (a preset with no declared projections, e.g. GPS
        ``Spin-rotated (B+U/F+D)``) is preferred over a multi-projection
        vector/WEP preset, so the recommendation is directly usable for the basic
        time-domain fit the longitudinal default would collapse.

    Notes
    -----
    This is a *recommendation* only.  Whether to surface the nudge — i.e. whether
    the current preset already matches the recommendation — is the caller's
    decision; this helper does not know the current preset.
    """
    if (field_direction or "").strip().lower() != "transverse":
        # No nudge for longitudinal / zero-field / unknown geometries.
        return None

    # Candidate transverse presets: name carries a spin-rotated or transverse
    # marker. The longitudinal default never matches these tokens, so it is
    # excluded automatically.
    transverse = [
        name for name in layout.presets if "rotat" in name.lower() or "transverse" in name.lower()
    ]
    if not transverse:
        return None

    # Prefer a single forward/backward pair (no declared projections): it drops
    # straight into the ordinary time-domain asymmetry fit that the longitudinal
    # default washes out. Multi-projection presets (vector / WEP) are a second
    # choice. Declaration order breaks ties within each tier.
    single_pair = [name for name in transverse if not layout.presets[name].projections]
    return (single_pair or transverse)[0]


def instrument_display_name(name: str) -> str:
    """Return the user-facing dropdown name for a layout registry key.

    Variant keys (e.g. ``"GPS-RD"``) map to their shared display name
    (``"GPS"``); every other key maps to itself.
    """
    try:
        return get_instrument_layout(name).display
    except KeyError:
        return name


def _variant_families() -> dict[str, list[str]]:
    """Map each display name to its registry keys, in ``INSTRUMENT_NAMES`` order.

    Two layouts that share a ``display_name`` are variants of one physical
    instrument (e.g. the 6-detector ``"GPS"`` and the 11-detector ``"GPS-RD"``
    both display "GPS"); the first key listed is the family default.  The family
    relationship is derived from the layouts themselves, so adding a variant only
    requires giving it the shared ``display_name`` — no separate registry to keep
    in sync.
    """
    families: dict[str, list[str]] = {}
    for key in INSTRUMENT_NAMES:
        families.setdefault(instrument_display_name(key), []).append(key)
    return families


def instrument_choices_for(active_name: str | None = None) -> list[tuple[str, str]]:
    """Return ``[(display_name, registry_key)]`` for the instrument dropdown.

    Variant families (e.g. the 6-detector PSI-BIN ``"GPS"`` and the 11-detector
    ROOT ``"GPS-RD"``) collapse to a single entry under their shared display
    name, so the user only ever sees one "GPS".  The variant exposed for a family
    is the one whose key equals *active_name* (the currently-loaded layout);
    otherwise the family default (first registry key) is used.
    """
    choices: list[tuple[str, str]] = []
    for display, keys in _variant_families().items():
        if len(keys) == 1:
            choices.append((display, keys[0]))
        else:
            chosen = active_name if active_name in keys else keys[0]
            choices.append((display, chosen))
    return choices


def variant_for_histograms(name: str, n_histograms: int) -> str:
    """Return the registry key of *name*'s variant whose detector count fits the run.

    Layout variants that share a display name (e.g. GPS BIN with 6 detectors vs
    GPS ROOT with 11) are distinguished by detector count.  Given a layout *name*
    and a run's ``n_histograms``, return the sibling variant whose
    ``n_detectors`` equals that count, so the layout always matches the data.
    Returns *name* unchanged when it has no sibling, the count is unknown, or no
    sibling fits.
    """
    keys = _variant_families().get(instrument_display_name(name), [name])
    if len(keys) <= 1 or not n_histograms:
        return name
    for key in keys:
        try:
            if get_instrument_layout(key).n_detectors == int(n_histograms):
                return key
        except KeyError:
            continue
    return name


def _canonical_instrument_name(raw: object) -> str | None:
    """Normalize free-form instrument text to a canonical instrument name."""
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None

    compact = re.sub(r"[^a-z0-9]+", "", token.lower())
    if "flame" in compact:
        return "FLAME"
    if "gps" in compact:
        return "GPS"
    if "hifi" in compact:
        return "HiFi"
    if "musr" in compact:
        return "MuSR"
    if "emu" in compact:
        return "EMU"
    return None


def _metadata_labels(metadata: dict) -> list[str]:
    """Return detector-label hints from metadata-like dictionaries."""
    raw = metadata.get("histogram_labels") or metadata.get("detector_labels")
    labels: list[str] = []
    if isinstance(raw, (list, tuple)):
        labels.extend(str(label) for label in raw)
    group_names = metadata.get("group_names")
    if isinstance(group_names, dict):
        labels.extend(str(label) for label in group_names.values())
    return labels


def _labels_match_flame(labels: list[str], n_histograms: int) -> bool:
    """Return True when detector labels match the FLAME eight-counter layout."""
    if n_histograms != 8:
        return False
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    has_main = (
        bool({"forw", "forward"} & compact)
        and bool({"back", "backward"} & compact)
        and "left" in compact
        and bool({"right", "righ"} & compact)
    )
    has_corners = {"rf", "rb", "lf", "lb"}.issubset(compact)
    return has_main and has_corners


def _is_psi_hal(metadata: dict, source_file: str | None) -> bool:
    """Return True when PSI metadata/filename identifies a HAL-9500 run.

    HAL-9500 files report ``instrument = "HIFI"`` and run names of the form
    ``tdc_hifi_*``; both carry the ``hifi`` token. Caller must already have
    established that the data is from PSI.
    """
    tokens = [
        str(metadata.get(key, ""))
        for key in ("instrument", "instrument_name", "spectrometer", "name")
    ]
    if source_file:
        tokens.append(Path(source_file).stem)
    compact = re.sub(r"[^a-z0-9]+", "", " ".join(tokens).lower())
    # Match the legacy "hifi" run-name/instrument token and the explicit
    # "hal9500" name; avoid a bare "hal" substring that would catch unrelated
    # words (e.g. a sample name like "halite").
    return "hifi" in compact or "hal9500" in compact


def _labels_match_gps(labels: list[str], n_histograms: int) -> bool:
    """Return True when detector labels match the GPS six-counter layout.

    GPS ``deltat_tdc_gps_*.bin`` files store exactly six histograms labelled
    ``Forw, Back, Up, Down, Righ, Left`` (the muon and veto counters are not
    stored as histograms).
    """
    if n_histograms != 6:
        return False
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    return (
        bool({"forw", "forward"} & compact)
        and bool({"back", "backward"} & compact)
        and "up" in compact
        and "down" in compact
        and bool({"righ", "right"} & compact)
        and "left" in compact
    )


def _labels_match_gps_subdetectors(labels: list[str]) -> bool:
    """Return True when detector labels match the GPS ROOT sub-detector layout.

    GPS ``deltat_tdc_gps_*.root`` (MusrRoot) files store the raw sub-detectors:
    ``Forw, Back, Up_B, Up_F, Down_B, Down_F, Right_B, Right_F, Left_B, Left_F,
    Mob-RL``.  Matching is on the label set alone (not the histogram count) so it
    also serves as a fallback when the count is reported unexpectedly.
    """
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    needed = {"upb", "upf", "downb", "downf", "rightb", "rightf", "leftb", "leftf"}
    has_fb = bool({"forw", "forward"} & compact) and bool({"back", "backward"} & compact)
    has_mobile = any("mob" in token for token in compact)
    return has_fb and needed.issubset(compact) and has_mobile


def _is_psi_gps(metadata: dict, source_file: str | None, n_histograms: int) -> bool:
    """Return True when PSI metadata/filename identifies a GPS run.

    GPS files report a ``GPS`` instrument string (BIN: ``"GPS"``; ROOT:
    ``"LMU_BULKMUSR_GPS"``) and run names of the form ``deltat_tdc_gps_*``; both
    carry the ``gps`` token (distinct from the GPD decay-channel instrument's
    ``gpd`` token).  As a fallback, either the six-counter BIN label set or the
    eleven-counter ROOT sub-detector label set identifies GPS.  Caller must
    already have established that the data is from PSI.
    """
    tokens = [
        str(metadata.get(key, ""))
        for key in ("instrument", "instrument_name", "beamline", "spectrometer", "name")
    ]
    if source_file:
        tokens.append(Path(source_file).stem)
    compact = re.sub(r"[^a-z0-9]+", "", " ".join(tokens).lower())
    if "gps" in compact:
        return True
    labels = _metadata_labels(metadata)
    return _labels_match_gps(labels, n_histograms) or _labels_match_gps_subdetectors(labels)


def _gps_variant(metadata: dict, n_histograms: int) -> str:
    """Return the GPS layout registry key for a recognised GPS run.

    The ROOT sub-detector export has 11 histograms; the PSI-BIN export has the 6
    combined detectors.  Returns ``"GPS-RD"`` for the former and ``"GPS"`` for
    the latter.
    """
    if n_histograms == 11 or _labels_match_gps_subdetectors(_metadata_labels(metadata)):
        return "GPS-RD"
    return "GPS"


def detect_instrument(
    n_histograms: int,
    *,
    metadata: dict | None = None,
    source_file: str | None = None,
) -> str | None:
    """Guess the instrument name from the number of histograms.

    This is a heuristic for the automatic instrument selection in the
    detector layout editor.  The caller should always allow the user to
    override the returned value.

    Parameters
    ----------
    n_histograms:
        Total number of histograms / raw-detector channels in the run.
    metadata:
        Optional run metadata. If an instrument-like field is present, it takes
        priority over the histogram-count heuristic.
    source_file:
        Optional source filename/path used as a final lightweight hint.

    Returns
    -------
    str or None
        Canonical instrument name when detection succeeds, else ``None``.
    """
    psi_data = False
    if isinstance(metadata, dict):
        facility = str(metadata.get("facility", "")).strip().lower()
        psi_data = facility == "psi" or bool(metadata.get("psi_format"))
        # PSI HAL-9500 (high-field): its run files carry the legacy "hifi"
        # token / "HIFI" instrument string, but it is a distinct instrument
        # from the ISIS HiFi spectrometer. Route it to the HAL layout before
        # the generic ISIS-name resolution below (which would mis-map it).
        if psi_data and _is_psi_hal(metadata, source_file):
            return "HAL"
        # PSI GPS (General Purpose Spectrometer): route by its "gps" token or the
        # GPS label sets, before the generic PSI return-None path below would
        # discard it.  Pick the 6-detector BIN ("GPS") or 11-detector ROOT
        # sub-detector ("GPS-RD") variant by histogram count.
        if psi_data and _is_psi_gps(metadata, source_file, n_histograms):
            return _gps_variant(metadata, n_histograms)
        for key in (
            "instrument",
            "instrument_name",
            "beamline",
            "spectrometer",
            "facility_instrument",
            "name",
        ):
            resolved = _canonical_instrument_name(metadata.get(key))
            if resolved is not None:
                if psi_data and resolved not in {"FLAME", "GPS"}:
                    return None
                return resolved
        if _labels_match_flame(_metadata_labels(metadata), n_histograms):
            return "FLAME"

    if source_file:
        path_token = _canonical_instrument_name(Path(source_file).stem)
        if path_token is not None:
            if psi_data and path_token not in {"FLAME", "GPS"}:
                return None
            return path_token

    if psi_data:
        return None

    if n_histograms == 64:
        return "HiFi"
    if n_histograms == 96:
        return "EMU"
    return None
