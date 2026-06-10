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

        # Patch map: detector_id → matplotlib patch
        self._patches: dict[int, Wedge | Rectangle] = {}
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
        n_banks = 1 if self._instrument.view == "plan" else len(self._instrument.banks)
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
        """Draw a single top-view rectangular detector layout."""
        ax = self._fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(self._instrument.banks[0].name, fontsize=10, fontweight="bold", pad=4)
        self._axes.append(ax)

        segments = self._instrument.all_segments
        for seg in segments:
            patch = self._make_rectangle(seg)
            ax.add_patch(patch)
            self._patches[seg.detector_id] = patch
            self._add_label(ax, seg)

        sample = Circle(
            (0, 0),
            radius=0.16,
            facecolor=(0.46, 0.38, 0.85, 1.0),
            edgecolor=(0.1, 0.1, 0.1, 1.0),
            linewidth=0.8,
            zorder=6,
        )
        ax.add_patch(sample)
        ax.text(0, -0.28, "sample", ha="center", va="top", fontsize=8, color="#333333")

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
                ha = "center"
                va = "top"
            elif "spin" in label_lower:
                label_x = (arrow.start[0] + arrow.end[0]) / 2.0
                label_y = arrow.start[1] + 0.16
                ha = "center"
                va = "bottom"
            else:
                label_x = arrow.end[0] + 0.08
                label_y = arrow.end[1] + 0.08
                ha = "left"
                va = "bottom"
            ax.text(
                label_x,
                label_y,
                arrow.label,
                ha=ha,
                va=va,
                fontsize=8,
                color=arrow.color,
            )

        axis_colour = "#555555"
        ax.add_patch(
            FancyArrowPatch(
                (2.75, -1.22),
                (3.25, -1.22),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=axis_colour,
                zorder=5,
            )
        )
        ax.text(2.68, -1.22, "+z", ha="right", va="center", fontsize=8, color=axis_colour)
        ax.add_patch(
            FancyArrowPatch(
                (-4.25, 1.05),
                (-4.25, 1.72),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=axis_colour,
                zorder=5,
            )
        )
        ax.text(
            -4.25,
            0.96,
            "+y",
            ha="center",
            va="top",
            fontsize=8,
            color=axis_colour,
        )

        min_x, max_x, min_y, max_y = self._plan_bounds(segments)
        ax.set_xlim(min_x - 0.45, max_x + 0.8)
        ax.set_ylim(min_y - 0.55, max_y + 0.55)
        self._fig.tight_layout(pad=0.4)
        self._canvas.draw_idle()

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
            ax.text(
                seg.x_center,
                seg.y_center,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=(0.15, 0.15, 0.15),
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

        # Hit-test: find the topmost patch at click position
        hit_seg: DetectorSegment | None = None
        for seg in bank.segments:
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
        n_banks = 1 if self._instrument.view == "plan" else len(self._instrument.banks)
        return QSize(max(380, n_banks * 200), 280)
