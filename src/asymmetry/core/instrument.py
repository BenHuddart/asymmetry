"""Instrument layout definitions for muon spectrometers.

This module provides static geometric descriptions of the detector arrangements
for HiFi, EMU, MuSR, PSI FLAME, and PSI HAL-9500, along with standard grouping
presets.  The
data here is used by the interactive detector layout editor
(:class:`~asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog`)
but has no GUI dependencies and can be used independently.

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
    "PROJECTION_TINTS",
    "derive_projection_pairs",
    "get_instrument_layout",
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
        PSI FLAME.
    label:
        Optional short detector label displayed alongside the detector ID.
    x_center, y_center, width, height, rotation_deg:
        Rectangle geometry in an arbitrary plan-view coordinate system.
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
    """

    name: str
    n_detectors: int
    banks: tuple[BankLayout, ...]
    presets: dict[str, PresetGrouping]
    view: str = "radial"
    reference_arrows: tuple[ReferenceArrow, ...] = ()

    @property
    def all_segments(self) -> list[DetectorSegment]:
        """All segments from all banks, in bank order."""
        segs: list[DetectorSegment] = []
        for bank in self.banks:
            segs.extend(bank.segments)
        return segs

    @property
    def default_preset_name(self) -> str:
        """Name of the first (default) preset."""
        return next(iter(self.presets))


# ---------------------------------------------------------------------------
# Projection derivation
# ---------------------------------------------------------------------------

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

#: Canonical names of all supported instruments.
INSTRUMENT_NAMES: Final[tuple[str, ...]] = ("HiFi", "MuSR", "EMU", "FLAME", "HAL")


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

    # Transverse — Left/Right orientated along beam direction
    # Left  = detectors 5–13 (forward)  + 52–60 (backward)
    # Right = detectors 21–29 (forward) + 36–44 (backward)
    presets["Transverse (Left–Right)"] = PresetGrouping(
        name="Transverse (Left–Right)",
        groups={
            1: GroupDefinition("Left", tuple(range(5, 14)) + tuple(range(52, 61))),
            2: GroupDefinition("Right", tuple(range(21, 30)) + tuple(range(36, 45))),
        },
        forward_group=1,
        backward_group=2,
    )

    # Transverse — Top/Bottom
    # Top    = detectors 13–21 (forward) + 44–52 (backward)
    # Bottom = detectors 1–5, 29–36 (forward) + 60–64 (backward)
    presets["Transverse (Top–Bottom)"] = PresetGrouping(
        name="Transverse (Top–Bottom)",
        groups={
            1: GroupDefinition(
                "Top",
                tuple(range(13, 22)) + tuple(range(44, 53)),
            ),
            2: GroupDefinition(
                "Bottom",
                tuple(range(1, 6)) + tuple(range(29, 37)) + tuple(range(60, 65)),
            ),
        },
        forward_group=1,
        backward_group=2,
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

    # Transverse — Top/Bottom
    # Top    = 17–24 (backward ring) + 49–56 (forward ring)
    # Bottom = 1–8   (backward ring) + 33–40 (forward ring)
    presets["Transverse (Top–Bottom)"] = PresetGrouping(
        name="Transverse (Top–Bottom)",
        groups={
            1: GroupDefinition("Top", tuple(range(17, 25)) + tuple(range(49, 57))),
            2: GroupDefinition("Bottom", tuple(range(1, 9)) + tuple(range(33, 41))),
        },
        forward_group=1,
        backward_group=2,
    )

    # Transverse — Forward/Backward (beam-direction)
    # Forward  = 9–16  (backward ring) + 57–64 (forward ring)
    # Backward = 25–32 (backward ring) + 41–48 (forward ring)
    presets["Transverse (Forward–Backward)"] = PresetGrouping(
        name="Transverse (Forward–Backward)",
        groups={
            1: GroupDefinition("Forward", tuple(range(9, 17)) + tuple(range(57, 65))),
            2: GroupDefinition("Backward", tuple(range(25, 33)) + tuple(range(41, 49))),
        },
        forward_group=1,
        backward_group=2,
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
    radial rings (inner, middle, outer).
    Source: *EMU User Guide*, page 34.

    Numbering formula for azimuth sector *s* (0–15):

    * Forward bank: inner = ``1 + 3s``, middle = ``2 + 3s``, outer = ``3 + 3s``.
    * Backward bank: inner = ``49 + 3s``, middle = ``50 + 3s``, outer = ``51 + 3s``.

    Angular convention: looking into the instrument from downstream.
    Sector 0 is at 12 o'clock (90°); numbers increase clockwise.
    Sector *s* centre: ``(90 − 22.5 × s) mod 360°``.
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
    # The EMU vector mode follows the paper/manual octant model where each bank
    # is split into four 12-detector octants (4 sectors × 3 rings):
    #   sectors 0-3   = upper-right
    #   sectors 4-7   = lower-right
    #   sectors 8-11  = lower-left
    #   sectors 12-15 = upper-left
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


def _flame_rectangle(
    detector_id: int,
    name: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> DetectorSegment:
    """Return one FLAME detector plate in the top-view schematic."""
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
        _flame_rectangle(1, "Forward", 3.55, 0.0, 0.74, 1.72),
        _flame_rectangle(2, "Backward", -3.55, 0.0, 0.74, 1.72),
        _flame_rectangle(3, "Right", 0.0, -2.18, 2.18, 0.82),
        _flame_rectangle(4, "Left", 0.0, 2.18, 2.18, 0.82),
        _flame_rectangle(5, "R_F", 1.55, -2.18, 0.82, 0.82),
        _flame_rectangle(6, "R_B", -1.55, -2.18, 0.82, 0.82),
        _flame_rectangle(7, "L_F", 1.55, 2.18, 0.82, 0.82),
        _flame_rectangle(8, "L_B", -1.55, 2.18, 0.82, 0.82),
    )
    banks = (BankLayout(name="FLAME top view", segments=segments),)

    presets: dict[str, PresetGrouping] = {}
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", (1,)),
            2: GroupDefinition("Backward", (2,)),
        },
        forward_group=1,
        backward_group=2,
    )
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

    # Longitudinal (default): forward ring vs backward ring.
    presets["Longitudinal"] = PresetGrouping(
        name="Longitudinal",
        groups={
            1: GroupDefinition("Forward", fwd_ring),
            2: GroupDefinition("Backward", bwd_ring),
        },
        forward_group=1,
        backward_group=2,
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
                if psi_data and resolved != "FLAME":
                    return None
                return resolved
        if _labels_match_flame(_metadata_labels(metadata), n_histograms):
            return "FLAME"

    if source_file:
        path_token = _canonical_instrument_name(Path(source_file).stem)
        if path_token is not None:
            if psi_data and path_token != "FLAME":
                return None
            return path_token

    if psi_data:
        return None

    if n_histograms == 64:
        return "HiFi"
    if n_histograms == 96:
        return "EMU"
    return None
