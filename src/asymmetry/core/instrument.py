"""Instrument layout definitions for muon spectrometers.

This module provides static geometric descriptions of the detector arrangements
for HiFi, EMU, MuSR, and PSI FLAME, along with standard grouping presets.  The
data here is used by the interactive detector layout editor
(:class:`~asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog`)
but has no GUI dependencies and can be used independently.

Detector IDs are always **1-based** in this module, matching the instrument
manual conventions.  Conversion to 0-based indices for internal computation
is the responsibility of the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "DetectorSegment",
    "BankLayout",
    "GroupDefinition",
    "PresetGrouping",
    "ReferenceArrow",
    "InstrumentLayout",
    "INSTRUMENT_NAMES",
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
    """

    name: str
    groups: dict[int, GroupDefinition]
    forward_group: int
    backward_group: int


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
# Instrument registry
# ---------------------------------------------------------------------------

#: Canonical names of all supported instruments.
INSTRUMENT_NAMES: Final[tuple[str, ...]] = ("HiFi", "MuSR", "EMU", "FLAME")


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
# Registry
# ---------------------------------------------------------------------------

_LAYOUTS: dict[str, InstrumentLayout] | None = None


def _build_registry() -> dict[str, InstrumentLayout]:
    return {
        "HiFi": _build_hifi(),
        "MuSR": _build_musr(),
        "EMU": _build_emu(),
        "FLAME": _build_flame(),
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
