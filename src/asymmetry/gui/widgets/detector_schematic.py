"""Interactive detector schematic widget for the grouping editor.

This widget renders a 2D schematic of a muon spectrometer's detector
arrangement using matplotlib wedge or rectangle patches.  Users can click
individual detector segments to toggle their membership in the currently active
group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from asymmetry.core.instrument import DetectorSegment, InstrumentLayout

__all__ = ["DetectorSchematicWidget"]

# ---------------------------------------------------------------------------
# Colour palette — one colour per group slot (up to 8)
# ---------------------------------------------------------------------------

#: RGBA tuples for groups 1-8.  Index 0 = group 1.
_GROUP_COLOURS: tuple[tuple[float, float, float, float], ...] = (
    (0.20, 0.47, 0.80, 0.85),  # 1 blue
    (0.90, 0.33, 0.24, 0.85),  # 2 red
    (0.18, 0.65, 0.35, 0.85),  # 3 green
    (0.95, 0.62, 0.10, 0.85),  # 4 orange
    (0.58, 0.28, 0.72, 0.85),  # 5 purple
    (0.85, 0.13, 0.55, 0.85),  # 6 magenta
    (0.13, 0.72, 0.80, 0.85),  # 7 cyan
    (0.62, 0.45, 0.20, 0.85),  # 8 brown
)

_EMPTY_COLOUR = (0.96, 0.96, 0.96, 1.0)  # light grey — detector in no group
_EDGE_COLOUR = (0.35, 0.35, 0.35, 1.0)  # dark grey edge
_BEAM_HOLE_COLOUR = (0.80, 0.88, 0.95, 1.0)  # pale blue for beam hole
_HOVER_EDGE_COLOUR = (0.0, 0.0, 0.0, 1.0)

# Read-only / context styling for multi-panel plan layouts: a detector shown for
# spatial context in a panel where it is edited elsewhere (dashed grey box, or a
# grey end-on ⊙/⊗ marker).
_READONLY_FACE = (0.93, 0.93, 0.93, 1.0)
_READONLY_EDGE = (0.60, 0.60, 0.60, 1.0)
_ENDON_COLOUR = (0.55, 0.58, 0.62, 1.0)
_SAMPLE_FACE = (0.46, 0.38, 0.85, 1.0)


def _group_colour(group_id: int) -> tuple[float, float, float, float]:
    """Return RGBA face colour for *group_id* (1-based)."""
    idx = (group_id - 1) % len(_GROUP_COLOURS)
    return _GROUP_COLOURS[idx]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class DetectorSchematicWidget(QWidget):
    """Interactive matplotlib-backed detector schematic.

    Parameters
    ----------
    instrument:
        Instrument layout to render.
    parent:
        Parent Qt widget.

    Signals
    -------
    detector_toggled(int, bool)
        Emitted when the user clicks a detector segment.  Arguments are the
        1-based ``detector_id`` and ``True`` if the detector is now included
        in the active group, ``False`` if it has been removed.
    """

    detector_toggled: Signal = Signal(int, bool)
    #: Emitted in exclude mode: (1-based detector id, now_excluded).
    detector_exclusion_toggled: Signal = Signal(int, bool)

    def __init__(
        self,
        instrument: InstrumentLayout,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._instrument = instrument

        # State -------------------------------------------------------
        # groups[gid] = set of included detector_ids (1-based)
        self._groups: dict[int, set[int]] = {}
        self._active_group: int = 1

        # Patch map: detector_id → clickable (active) matplotlib patch
        self._patches: dict[int, Wedge | Rectangle] = {}
        # Read-only context rectangles (a detector shown in a panel where it is
        # edited elsewhere); kept so exclusion hatching stays consistent across
        # both panels. Entries are (detector_id, patch).
        self._readonly_patches: list[tuple[int, Rectangle]] = []
        # Axis map: bank index → polar axes
        self._axes: list = []

        self._excluded: set[int] = set()
        self._exclude_mode = False
        self._setup_ui()
        self._build_schematic()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Create the matplotlib figure canvas."""
        n_banks = len(self._instrument.banks)
        fig_width = max(4.0, n_banks * 3.2)
        fig_height = 4.6 if self._instrument.view == "plan" else 3.8

        self._fig = Figure(figsize=(fig_width, fig_height), facecolor="white")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumSize(QSize(300, 240))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Schematic construction
    # ------------------------------------------------------------------

    def _build_schematic(self) -> None:
        """Draw all detector banks onto the figure."""
        self._fig.clear()
        self._patches.clear()
        self._readonly_patches.clear()
        self._axes.clear()

        if self._instrument.view == "plan":
            self._build_plan_schematic()
            return

        n_banks = len(self._instrument.banks)
        for bi, bank in enumerate(self._instrument.banks):
            ax = self._fig.add_subplot(1, n_banks, bi + 1)
            ax.set_aspect("equal")
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
            ax.axis("off")
            ax.set_title(bank.name, fontsize=10, fontweight="bold", pad=4)
            self._axes.append(ax)

            # Draw all segments for this bank
            for seg in bank.segments:
                patch = self._make_radial_patch(seg)
                ax.add_patch(patch)
                self._patches[seg.detector_id] = patch
                # Label at centroid
                self._add_label(ax, seg)

            # Annular instruments (wedge banks) get a central beam-hole disc.
            # Banks built from edge-rectangles (HAL) define their own centre via
            # the MV disc, so the generic hole is skipped.
            annular = [s for s in bank.segments if s.shape == "wedge" and s.r_inner > 0]
            if annular and not any(s.shape == "rectangle" for s in bank.segments):
                hole = Circle(
                    (0, 0),
                    radius=min(s.r_inner for s in annular) - 0.01,
                    facecolor=_BEAM_HOLE_COLOUR,
                    edgecolor=_EDGE_COLOUR,
                    linewidth=0.8,
                    zorder=5,
                )
                ax.add_patch(hole)

        self._fig.tight_layout(pad=0.4)
        self._canvas.draw_idle()

    def _build_plan_schematic(self) -> None:
        """Draw one or more top-view (plan) detector panels side by side.

        Single-panel instruments (FLAME) render one top view with their
        reference arrows.  Multi-panel instruments (GPS: top + side view) render
        one panel per bank; a detector is clickable in its home panel and shown
        read-only (dashed box or end-on ⊙/⊗ marker) in the others.
        """
        banks = self._instrument.banks
        multi = len(banks) > 1
        for bi, bank in enumerate(banks):
            ax = self._fig.add_subplot(1, len(banks), bi + 1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(bank.name, fontsize=10, fontweight="bold", pad=4)
            self._axes.append(ax)
            self._draw_plan_panel(ax, bank, multi=multi)

        self._fig.tight_layout(pad=0.4)
        self._canvas.draw_idle()

    def _draw_plan_panel(self, ax, bank, *, multi: bool) -> None:
        """Draw the detectors, sample and axis cues for one plan panel."""
        bound_segs: list[DetectorSegment] = []
        for seg in bank.segments:
            if seg.shape == "rectangle":
                patch = self._make_rectangle(seg)
                ax.add_patch(patch)
                self._add_label(ax, seg)
                bound_segs.append(seg)
                if seg.read_only:
                    self._apply_read_only_style(patch)
                    self._readonly_patches.append((seg.detector_id, patch))
                else:
                    # Only active (clickable) segments take a group colour.
                    self._patches[seg.detector_id] = patch
            elif seg.shape in ("endon_out", "endon_in"):
                self._draw_endon(ax, seg)

        ax.add_patch(
            Circle(
                (0, 0),
                radius=0.16,
                facecolor=_SAMPLE_FACE,
                edgecolor=(0.1, 0.1, 0.1, 1.0),
                linewidth=0.8,
                zorder=6,
            )
        )

        min_x, max_x, min_y, max_y = self._plan_bounds(bound_segs or bank.segments)

        # Direction cues are driven by the layout's data, not the panel count:
        # an instrument that declares reference_arrows always gets them (so a
        # future multi-panel instrument is not silently stripped of its cues);
        # otherwise a beam (+z) arrow is drawn. The sample label and +z/+y axis
        # indicators stay on single-panel views to avoid crowding multi panels.
        if self._instrument.reference_arrows:
            self._draw_reference_arrows(ax)
        else:
            self._draw_beam_arrow(ax, min_x, max_x, min_y)
        if not multi:
            ax.text(0, -0.28, "sample", ha="center", va="top", fontsize=8, color="#333333")
            self._draw_plan_axes(ax, min_x, max_x, min_y, max_y)

        ax.set_xlim(min_x - 0.45, max_x + 0.8)
        ax.set_ylim(min_y - 0.55, max_y + 0.55)

    def _apply_read_only_style(self, patch: Rectangle) -> None:
        """Style a detector rectangle as read-only context (dashed grey box)."""
        patch.set_facecolor(_READONLY_FACE)
        patch.set_edgecolor(_READONLY_EDGE)
        patch.set_linestyle((0, (4, 3)))
        patch.set_linewidth(1.0)

    def _draw_endon(self, ax, seg: DetectorSegment) -> None:
        """Draw a detector seen end-on: ⊙ (toward viewer) or ⊗ (away)."""
        x, y = seg.x_center, seg.y_center
        r = 0.16
        ax.add_patch(
            Circle(
                (x, y),
                radius=r,
                facecolor="none",
                edgecolor=_ENDON_COLOUR,
                linewidth=1.4,
                zorder=4,
            )
        )
        if seg.shape == "endon_out":
            ax.add_patch(Circle((x, y), radius=0.035, facecolor=_ENDON_COLOUR, zorder=4))
        else:
            d = r * 0.6
            ax.plot(
                [x - d, x + d],
                [y - d, y + d],
                color=_ENDON_COLOUR,
                linewidth=1.4,
                zorder=4,
            )
            ax.plot(
                [x - d, x + d],
                [y + d, y - d],
                color=_ENDON_COLOUR,
                linewidth=1.4,
                zorder=4,
            )
        if seg.label:
            # For a split (_B/_F) pair sitting side by side, place each label
            # *outward* (left member to its left, right member to its right) so
            # neither label crowds the central sample or the other member. A lone
            # centred marker (e.g. Mobile, or the 6-detector Up/Down) labels below.
            if seg.x_center < -1e-6:
                ax.text(
                    x - r - 0.08, y, seg.label, ha="right", va="center",
                    fontsize=7, color=_ENDON_COLOUR, zorder=4,
                )
            elif seg.x_center > 1e-6:
                ax.text(
                    x + r + 0.08, y, seg.label, ha="left", va="center",
                    fontsize=7, color=_ENDON_COLOUR, zorder=4,
                )
            else:
                ax.text(
                    x, y - r - 0.06, seg.label, ha="center", va="top",
                    fontsize=7, color=_ENDON_COLOUR, zorder=4,
                )

    def _draw_beam_arrow(self, ax, min_x: float, max_x: float, min_y: float) -> None:
        """Draw the muon-beam (+z) arrow along the bottom of a plan panel."""
        y = min_y - 0.30
        x0 = min_x * 0.55
        x1 = max_x * 0.55
        ax.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#555555",
                zorder=5,
            )
        )
        ax.text(x0, y - 0.05, "beam  +z →", ha="left", va="top", fontsize=8, color="#555555")

    def _draw_reference_arrows(self, ax) -> None:
        """Draw the instrument-level reference arrows (single-panel plan)."""
        for arrow in self._instrument.reference_arrows:
            ax.add_patch(
                FancyArrowPatch(
                    arrow.start,
                    arrow.end,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.8,
                    color=arrow.color,
                    zorder=5,
                )
            )
            label_lower = arrow.label.lower()
            if "beam" in label_lower:
                label_x = (arrow.start[0] + arrow.end[0]) / 2.0
                label_y = arrow.start[1] - 0.16
                ha, va = "center", "top"
            elif "spin" in label_lower:
                label_x = (arrow.start[0] + arrow.end[0]) / 2.0
                label_y = arrow.start[1] + 0.16
                ha, va = "center", "bottom"
            else:
                label_x = arrow.end[0] + 0.08
                label_y = arrow.end[1] + 0.08
                ha, va = "left", "bottom"
            ax.text(label_x, label_y, arrow.label, ha=ha, va=va, fontsize=8, color=arrow.color)

    def _draw_plan_axes(
        self,
        ax,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> None:
        """Draw +z / +y coordinate indicator arrows for a single-panel plan view."""
        axis_colour = "#555555"
        z_y = min_y - 0.30
        z_x_end = max_x - 0.05
        z_x_start = z_x_end - 0.5
        ax.add_patch(
            FancyArrowPatch(
                (z_x_start, z_y),
                (z_x_end, z_y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=axis_colour,
                zorder=5,
            )
        )
        ax.text(z_x_start - 0.08, z_y, "+z", ha="right", va="center", fontsize=8, color=axis_colour)
        y_x = min_x - 0.30
        y_y_start = max_y - 0.65
        y_y_end = max_y - 0.05
        ax.add_patch(
            FancyArrowPatch(
                (y_x, y_y_start),
                (y_x, y_y_end),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=axis_colour,
                zorder=5,
            )
        )
        ax.text(y_x, y_y_start - 0.08, "+y", ha="center", va="top", fontsize=8, color=axis_colour)

    def _make_wedge(self, seg: DetectorSegment) -> Wedge:
        """Create a matplotlib :class:`~matplotlib.patches.Wedge` for *seg*."""
        # matplotlib Wedge: theta1 < theta2, both in degrees, measured CCW from +x
        # Our angle_center_deg is CCW from +x already.
        theta1 = seg.angle_start_deg
        theta2 = seg.angle_end_deg

        # Handle wrap-around
        if theta2 < theta1:
            theta1, theta2 = theta2, theta1

        r_outer = seg.r_outer
        width = seg.r_outer - seg.r_inner

        patch = Wedge(
            center=(0, 0),
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=width,
            facecolor=_EMPTY_COLOUR,
            edgecolor=_EDGE_COLOUR,
            linewidth=0.6,
            picker=True,
            zorder=2,
        )
        return patch

    def _make_radial_patch(self, seg: DetectorSegment):
        """Create the patch for a radial-bank segment, dispatching on shape.

        Banks mix curved wedges (HiFi/MuSR/EMU rings) with edge-aligned
        rectangles (the HAL-9500 octagon of detector bars).
        """
        if seg.shape == "rectangle":
            return self._make_rectangle(seg)
        return self._make_wedge(seg)

    def _make_rectangle(self, seg: DetectorSegment) -> Rectangle:
        """Create a matplotlib rectangle patch for a plan-view detector."""
        patch = Rectangle(
            (seg.x_center - seg.width / 2.0, seg.y_center - seg.height / 2.0),
            seg.width,
            seg.height,
            angle=seg.rotation_deg,
            rotation_point="center",
            facecolor=_EMPTY_COLOUR,
            edgecolor=_EDGE_COLOUR,
            linewidth=0.8,
            picker=True,
            zorder=2,
        )
        return patch

    def _add_label(self, ax, seg: DetectorSegment) -> None:
        """Draw the detector number at the segment centroid."""
        if seg.shape == "rectangle":
            # Plan-view banks (FLAME) show "id + name"; radial edge-bars (HAL)
            # show just the physical label to stay uncluttered.
            if self._instrument.view == "plan":
                label = f"{seg.detector_id}\n{seg.label}" if seg.label else str(seg.detector_id)
            else:
                label = seg.label if seg.label else str(seg.detector_id)
            # Read-only context plates use a muted label so they read as
            # non-clickable, matching their dashed-grey box.
            colour = _ENDON_COLOUR if seg.read_only else (0.15, 0.15, 0.15)
            ax.text(
                seg.x_center,
                seg.y_center,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
                zorder=3,
            )
            return

        angle_rad = math.radians(seg.angle_center_deg)
        r_mid = (seg.r_inner + seg.r_outer) / 2.0
        x = r_mid * math.cos(angle_rad)
        y = r_mid * math.sin(angle_rad)

        # Determine font size based on segment density
        n_sectors = max(len(set(s.sector_index for s in self._instrument.all_segments)), 1)
        fontsize = max(3.5, min(6.5, 130.0 / n_sectors))

        # Prefer the physical detector label (e.g. "F1", "MV") when present;
        # fall back to the bare detector ID for label-free banks (HiFi/MuSR/EMU).
        text = seg.label if seg.label else str(seg.detector_id)

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=(0.15, 0.15, 0.15),
            zorder=3,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_active_group(self, group_id: int) -> None:
        """Change which group is currently active (receives click toggles).

        Parameters
        ----------
        group_id:
            1-based group identifier.
        """
        self._active_group = group_id
        self._refresh_colours()

    def set_group_detectors(self, group_id: int, detector_ids: set[int]) -> None:
        """Set detector membership for one group and refresh the display.

        Parameters
        ----------
        group_id:
            1-based group identifier.
        detector_ids:
            Set of 1-based detector IDs belonging to the group.
        """
        self._groups[group_id] = set(detector_ids)
        self._refresh_colours()

    def set_all_groups(
        self,
        groups: dict[int, list[int]],
        active_group: int,
    ) -> None:
        """Replace all group assignments and set the active group.

        Parameters
        ----------
        groups:
            Mapping from group ID (1-based) to list of 1-based detector IDs.
        active_group:
            Group ID that should be considered active.
        """
        self._groups = {gid: set(ids) for gid, ids in groups.items()}
        self._active_group = active_group
        self._refresh_colours()

    def set_excluded_detectors(self, detector_ids: set[int]) -> None:
        """Mark detectors (1-based ids) as excluded and refresh the display."""
        self._excluded = {int(v) for v in detector_ids}
        self._refresh_colours()

    def set_exclude_mode(self, enabled: bool) -> None:
        """Toggle click-to-exclude mode (clicks edit the exclusion set)."""
        self._exclude_mode = bool(enabled)

    def get_excluded_detectors(self) -> set[int]:
        """Return the current exclusion set (1-based ids)."""
        return set(self._excluded)

    def get_filled_detectors(self) -> set[int]:
        """Return the set of detectors currently in the active group."""
        return set(self._groups.get(self._active_group, set()))

    # ------------------------------------------------------------------
    # Colour refresh
    # ------------------------------------------------------------------

    def _detector_groups(self, det_id: int) -> list[int]:
        """Return all group IDs that include *det_id*."""
        return [gid for gid, members in self._groups.items() if det_id in members]

    def _refresh_colours(self) -> None:
        """Recolour all patches to reflect the current group assignments."""
        for det_id, patch in self._patches.items():
            if det_id in self._excluded:
                patch.set_facecolor((0.55, 0.55, 0.55, 0.45))
                patch.set_hatch("xx")
                patch.set_linewidth(0.6)
                continue
            patch.set_hatch(None)
            gids = self._detector_groups(det_id)
            if gids:
                # If a detector belongs to multiple groups, prioritise the active
                # group colour so editing state remains obvious.
                gid = self._active_group if self._active_group in gids else min(gids)
                patch.set_facecolor(_group_colour(gid))
                patch.set_linewidth(1.5 if self._active_group in gids else 0.8)
            else:
                patch.set_facecolor(_EMPTY_COLOUR)
                patch.set_linewidth(0.6)

        # Read-only context plates (the same detector shown in another panel) keep
        # their dashed-grey style, but mirror the exclusion hatch so an excluded
        # detector reads as excluded in every panel it appears in.
        for det_id, patch in self._readonly_patches:
            if det_id in self._excluded:
                patch.set_hatch("xx")
            else:
                patch.set_hatch(None)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        """Handle a mouse click on the figure canvas."""
        if event.button != 1:  # left click only
            return
        if event.inaxes is None:
            return

        ax = event.inaxes
        if ax not in self._axes:
            return

        # Find which bank this axis belongs to
        bank_idx = self._axes.index(ax)
        bank = self._instrument.banks[bank_idx]

        # Hit-test: find the topmost patch at click position. Read-only context
        # segments (dashed boxes, end-on ⊙/⊗ markers) are not clickable here —
        # they are edited in the panel where the detector lies in-plane.
        hit_seg: DetectorSegment | None = None
        for seg in bank.segments:
            if seg.read_only:
                continue
            if self._point_in_segment(event.xdata, event.ydata, seg):
                # In case of stacked patches (e.g. EMU rings), pick the smallest
                if hit_seg is None or (
                    (seg.r_outer - seg.r_inner) < (hit_seg.r_outer - hit_seg.r_inner)
                ):
                    hit_seg = seg

        if hit_seg is None:
            return

        det_id = hit_seg.detector_id
        if self._exclude_mode:
            if det_id in self._excluded:
                self._excluded.discard(det_id)
                excluded_now = False
            else:
                self._excluded.add(det_id)
                excluded_now = True
            self._refresh_colours()
            self.detector_exclusion_toggled.emit(det_id, excluded_now)
            return

        active_members = self._groups.setdefault(self._active_group, set())

        if det_id in active_members:
            # Remove from active group
            active_members.discard(det_id)
            included = False
        else:
            active_members.add(det_id)
            included = True

        self._refresh_colours()
        self.detector_toggled.emit(det_id, included)

    @staticmethod
    def _point_in_segment(x: float, y: float, seg: DetectorSegment) -> bool:
        """Return ``True`` if the point (x, y) falls within *seg*."""
        if x is None or y is None:
            return False
        if seg.shape in ("endon_out", "endon_in"):
            # End-on markers are read-only context only.
            return False
        if seg.shape == "rectangle":
            return DetectorSchematicWidget._point_in_rectangle(x, y, seg)

        r = math.sqrt(x * x + y * y)
        if not (seg.r_inner <= r <= seg.r_outer):
            return False

        # A full-circle wedge (e.g. the MV disc) has no meaningful angular
        # bounds, so any point within the radius is inside.
        if seg.angle_half_width_deg >= 179.95:
            return True

        # Compute CCW angle from +x axis, normalised to [0, 360)
        angle = math.degrees(math.atan2(y, x)) % 360.0

        # Normalise segment bounds to [0, 360) and handle wrap-around
        a1 = seg.angle_start_deg % 360.0
        a2 = seg.angle_end_deg % 360.0

        if a1 <= a2:
            return a1 <= angle <= a2
        else:
            # Wraps across 0°
            return angle >= a1 or angle <= a2

    @staticmethod
    def _point_in_rectangle(x: float, y: float, seg: DetectorSegment) -> bool:
        """Return ``True`` if the point falls within a rectangular segment."""
        dx = x - seg.x_center
        dy = y - seg.y_center
        theta = math.radians(-seg.rotation_deg)
        local_x = dx * math.cos(theta) - dy * math.sin(theta)
        local_y = dx * math.sin(theta) + dy * math.cos(theta)
        return abs(local_x) <= seg.width / 2.0 and abs(local_y) <= seg.height / 2.0

    @staticmethod
    def _plan_bounds(segments: list[DetectorSegment]) -> tuple[float, float, float, float]:
        """Return approximate bounds for plan-view rectangle segments."""
        if not segments:
            return -1.0, 1.0, -1.0, 1.0
        min_x = min(seg.x_center - seg.width / 2.0 for seg in segments)
        max_x = max(seg.x_center + seg.width / 2.0 for seg in segments)
        min_y = min(seg.y_center - seg.height / 2.0 for seg in segments)
        max_y = max(seg.y_center + seg.height / 2.0 for seg in segments)
        return min_x, max_x, min_y, max_y

    # ------------------------------------------------------------------
    # Rebuild when instrument changes
    # ------------------------------------------------------------------

    def set_instrument(self, instrument: InstrumentLayout) -> None:
        """Replace the displayed instrument layout.

        Parameters
        ----------
        instrument:
            New instrument layout to render.
        """
        self._instrument = instrument
        self._groups = {}
        self._active_group = 1
        self._build_schematic()

    def sizeHint(self) -> QSize:
        n_banks = len(self._instrument.banks)
        return QSize(max(380, n_banks * 200), 280)
