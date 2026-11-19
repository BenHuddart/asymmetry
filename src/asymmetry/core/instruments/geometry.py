"""Frozen two-dimensional arrangements, expanded without plotting dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Schematic coordinates and dimensions must be finite")


@dataclass(frozen=True)
class Wedge:
    """Polar-origin annular sector with angles in degrees."""

    inner: float
    outer: float
    half_angle: float

    def __post_init__(self):
        _finite(self.inner, self.outer, self.half_angle)
        if not 0 <= self.inner < self.outer or not 0 < self.half_angle <= 180:
            raise ValueError("Wedge requires 0 <= inner < outer and 0 < half_angle <= 180")


@dataclass(frozen=True)
class Rectangle:
    """Rectangle dimensions and rotation relative to the arrangement angle."""

    width: float
    height: float
    rotation: float = 0.0

    def __post_init__(self):
        _finite(self.width, self.height, self.rotation)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle dimensions must be positive")


@dataclass(frozen=True)
class EndOn:
    """Noninteractive toward/away marker, not a detector acceptance model."""

    toward: bool = True
    radius: float = 0.16

    def __post_init__(self):
        _finite(self.radius)
        if self.radius <= 0:
            raise ValueError("End-on marker radius must be positive")


Shape = Wedge | Rectangle | EndOn


@dataclass(frozen=True)
class Occurrence:
    """One channel occurrence or noninteractive hardware context occurrence."""

    channel: str | None = None
    hardware: str | None = None
    editable: bool = True
    label: str | None = None

    def __post_init__(self):
        if (self.channel is None) == (self.hardware is None):
            raise ValueError("Occurrence must reference exactly one channel or hardware element")
        if self.hardware is not None and self.editable:
            raise ValueError("Hardware context cannot be editable")


@dataclass(frozen=True)
class Placement:
    """Resolved shape at a Cartesian centre and a CCW angle from the right."""

    target: Occurrence
    shape: Shape
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    sector: int = 0
    ring: int = 0

    def __post_init__(self):
        _finite(self.x, self.y, self.angle)
        if not isinstance(self.shape, (Wedge, Rectangle, EndOn)):
            raise ValueError("Unknown drawing primitive")
        if isinstance(self.shape, Wedge) and (self.x != 0 or self.y != 0):
            raise ValueError("Wedges use the polar origin; translate the view, not a sector")
        if isinstance(self.shape, EndOn) and self.target.editable:
            raise ValueError("End-on markers are noninteractive context")


@dataclass(frozen=True)
class Explicit:
    """Finite placements for irregular geometry and numbering exceptions."""

    elements: tuple[Placement, ...]

    def __post_init__(self):
        if not self.elements:
            raise ValueError("Explicit arrangement needs at least one placement")


@dataclass(frozen=True)
class Circular:
    """Sector-major channel rows; columns select concentric shapes/radii.

    Wedges share the polar origin. Rectangles and markers are positioned at
    their declared radius, independently of their shape. No membership is
    inferred from the resulting drawing.
    """

    rows: tuple[tuple[Occurrence, ...], ...]
    shapes: tuple[Shape, ...]
    radii: tuple[float, ...]
    start_angle: float
    step_angle: float
    radial_rotation: bool = False

    def __post_init__(self):
        _finite(self.start_angle, self.step_angle, *self.radii)
        if not self.rows or not self.shapes or len(self.radii) != len(self.shapes):
            raise ValueError("Circular arrangement needs rows and one radius per shape")
        if self.step_angle == 0 or any(radius < 0 for radius in self.radii):
            raise ValueError("Circular step must be nonzero and radii nonnegative")
        if any(len(row) != len(self.shapes) for row in self.rows):
            raise ValueError("Circular channel rows must match the concentric shape count")
        if any(
            isinstance(shape, Wedge) and radius != 0
            for shape, radius in zip(self.shapes, self.radii, strict=True)
        ):
            raise ValueError("Wedge radius is encoded by its inner/outer bounds")


@dataclass(frozen=True)
class Grid:
    """Rectangular rows with explicit channel ordering and Cartesian spacing."""

    rows: tuple[tuple[Occurrence, ...], ...]
    shape: Rectangle | EndOn
    origin: tuple[float, float] = (0.0, 0.0)
    step: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        _finite(*self.origin, *self.step)
        if (
            not self.rows
            or not self.rows[0]
            or any(len(row) != len(self.rows[0]) for row in self.rows)
        ):
            raise ValueError("Grid requires nonempty equally sized rows")
        if (len(self.rows[0]) > 1 and self.step[0] == 0) or (
            len(self.rows) > 1 and self.step[1] == 0
        ):
            raise ValueError("Grid spacing cannot collapse distinct rows or columns")


@dataclass(frozen=True)
class View:
    """Named schematic or dimensional view, independent of physical banks."""

    key: str
    label: str
    mode: str
    convention: str
    arrangement: Circular | Grid | Explicit
    scale: str = "schematic"
    units: str = "dimensionless"

    def __post_init__(self):
        if not self.key or not self.label or not self.convention:
            raise ValueError("View needs a key, label and viewing convention")
        if self.mode not in {"radial", "plan"} or self.scale not in {"schematic", "physical"}:
            raise ValueError("Unknown view mode or coordinate scale")
        if not self.units or (self.scale == "physical" and self.units == "dimensionless"):
            raise ValueError("Physical view coordinates require declared units")
        for element in expand_view(self):
            if self.mode == "plan" and isinstance(element.shape, Wedge):
                raise ValueError("Plan views cannot contain wedges")
            if self.mode == "radial" and isinstance(element.shape, EndOn):
                raise ValueError("Radial views cannot contain end-on markers")


def expand_view(view: View) -> tuple[Placement, ...]:
    """Expand validated arrangements into the same finite drawing primitives."""
    arrangement = view.arrangement
    if isinstance(arrangement, Explicit):
        return arrangement.elements
    elements = []
    for row_index, row in enumerate(arrangement.rows):
        for column, target in enumerate(row):
            if isinstance(arrangement, Circular):
                angle = (arrangement.start_angle + row_index * arrangement.step_angle) % 360
                shape = arrangement.shapes[column]
                radius = arrangement.radii[column]
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))
                elements.append(Placement(target, shape, x, y, angle, row_index, column))
            else:
                elements.append(
                    Placement(
                        target,
                        arrangement.shape,
                        arrangement.origin[0] + column * arrangement.step[0],
                        arrangement.origin[1] + row_index * arrangement.step[1],
                        sector=row_index * len(row) + column,
                    )
                )
    return tuple(elements)
