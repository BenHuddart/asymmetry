"""Qt-free matplotlib decorations shared by the editor and profile diagrams."""

import math

from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.transforms import ScaledTranslation


def schematic_axes(figure, layout):
    """Keep unlocated auxiliary readouts in a separate strip, not a view position."""
    grid = figure.add_gridspec(
        2 if layout.auxiliary else 1,
        len(layout.banks),
        height_ratios=(5, 1) if layout.auxiliary else (1,),
    )
    axes = tuple(figure.add_subplot(grid[0, index]) for index in range(len(layout.banks)))
    auxiliary = figure.add_subplot(grid[1, :]) if layout.auxiliary else None
    if auxiliary is not None:
        auxiliary.set_aspect("equal")
        auxiliary.axis("off")
        auxiliary.set_title("Auxiliary — position unspecified", fontsize=8, pad=2)
    return axes, auxiliary


def draw_beam(axes, projection):
    """Return labelled artists and exact hover text; never guess an unknown axial sense."""
    if projection is None:
        return ()
    dx, dy = projection.screen
    length = math.hypot(dx, dy)
    center = (0.5, -0.04)
    # Inch offsets preserve the projected screen angle on nonsquare or resized axes.
    inch_offsets = axes.figure.dpi_scale_trans + ScaledTranslation(*center, axes.transAxes)
    artists = []
    tooltip = "Muon beam direction"
    if length > 1e-12:
        delta = (0.45 * dx / length, 0.45 * dy / length)
        arrow = FancyArrowPatch(
            (-delta[0], -delta[1]),
            delta,
            transform=inch_offsets,
            arrowstyle="-|>",
            mutation_scale=12,
            color="0.35",
            clip_on=False,
        )
        axes.add_patch(arrow)
        artists.append(arrow)
    elif projection.toward is not None and abs(projection.toward) > 1e-12:
        toward = projection.toward > 0
        tooltip = "Muon beam out of the screen" if toward else "Muon beam into the screen"
        circle = Circle(
            (0, 0), 0.06, transform=inch_offsets, fill=False, color="0.35", clip_on=False
        )
        axes.add_patch(circle)
        artists.extend(
            (
                circle,
                axes.text(
                    0,
                    0,
                    "·" if toward else "×",
                    transform=inch_offsets,
                    ha="center",
                    va="center",
                    color="0.35",
                ),
            )
        )
    if length <= 1e-12 and projection.toward is None:
        tooltip += "\nView along beam axis; viewing sense unspecified"
    artists.append(
        axes.text(
            0.5,
            -0.12,
            "Beam",
            transform=axes.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    )
    return tuple((artist, tooltip) for artist in artists)


def draw_readout_rows(axes, segments):
    """Label each declared readout row once, outside its selectable boxes."""
    rows = dict.fromkeys(segment.row_label for segment in segments if segment.row_label is not None)
    for row in rows:
        axes.text(*row.position, row.text, ha="left", va="center", fontsize=7, color="0.35")
