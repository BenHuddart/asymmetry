"""Frozen two-dimensional arrangements, expanded without plotting dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Schematic coordinates and dimensions must be finite")


def _unit_direction(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalise a finite nonzero direction without overflow or subnormal underflow."""
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("A direction requires three finite coordinates")
    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise ValueError("A direction must be nonzero")
    scaled = tuple(value / scale for value in vector)
    length = math.hypot(*scaled)
    return tuple(value / length for value in scaled)


@dataclass(frozen=True)
class BeamDirection:
    """Unit beam vector in the profile's declared common arrangement frame.

    Coordinates specify direction only. Existing two-dimensional placements
    remain local to each drawing view; they are not three-dimensional positions.
    """

    vector: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "vector", _unit_direction(self.vector))


@dataclass(frozen=True)
class ViewOrientation:
    """Screen axes in the same arrangement frame as the beam.

    ``right`` and ``up`` define the drawing plane. An explicit ``toward`` axis
    points out of the screen; omission retains unknown axial viewing sense.
    Axes are normalised and must be perpendicular within 1e-12. Either handedness
    is accepted, including explicitly mirrored drawings, without inferring a
    physical observer or requiring a proper rotation.
    """

    right: tuple[float, float, float]
    up: tuple[float, float, float]
    toward: tuple[float, float, float] | None = None

    def __post_init__(self):
        axes = ["right", "up"] + (["toward"] if self.toward is not None else [])
        coordinates = (self.right, self.up) + ((self.toward,) if self.toward is not None else ())
        directions = [_unit_direction(vector) for vector in coordinates]
        for index, direction in enumerate(directions):
            if any(
                abs(sum(a * b for a, b in zip(direction, other, strict=True))) > 1e-12
                for other in directions[:index]
            ):
                raise ValueError("View direction axes must be mutually perpendicular")
        for axis, direction in zip(axes, directions, strict=True):
            object.__setattr__(self, axis, direction)


@dataclass(frozen=True)
class BeamProjection:
    """Unit beam's signed screen components and optional out-of-screen component.

    Positive ``toward`` means out of the screen, negative means into it, and
    ``None`` preserves unknown sense, including a beam with zero screen components.
    """

    screen: tuple[float, float]
    toward: float | None


def project_beam(
    beam: BeamDirection | None, orientation: ViewOrientation | None
) -> BeamProjection | None:
    """Project a declared beam onto explicit screen axes; missing declarations stay unknown."""
    if beam is None or orientation is None:
        return None
    return BeamProjection(
        tuple(
            sum(a * b for a, b in zip(beam.vector, axis, strict=True))
            for axis in (orientation.right, orientation.up)
        ),
        sum(a * b for a, b in zip(beam.vector, orientation.toward, strict=True))
        if orientation.toward is not None
        else None,
    )


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
class RowLabel:
    """A readout-row caption anchored by the arrangement, not inferred by renderers."""

    text: str
    position: tuple[float, float]

    def __post_init__(self):
        if len(self.position) != 2:
            raise ValueError("A row label needs two screen coordinates")
        _finite(*self.position)
        if not self.text:
            raise ValueError("A row label needs text")
        object.__setattr__(self, "position", tuple(self.position))


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
    rotate: bool = True
    row_label: RowLabel | None = None

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
class BankRow:
    """Left-to-right (or bottom-to-top) readouts, not physical detector sums."""

    targets: tuple[Occurrence, ...]
    widths: tuple[float, ...]
    thickness: float = 0.75
    label: str = ""

    def __post_init__(self):
        _finite(*self.widths, self.thickness)
        if not self.targets or len(self.targets) != len(self.widths):
            raise ValueError("A bank row needs one width per readout")
        if min(*self.widths, self.thickness) <= 0:
            raise ValueError("Bank row dimensions must be positive")


@dataclass(frozen=True)
class Directional:
    """Ordered rows outward from the sample in one cardinal screen direction.

    Within a horizontal bank readouts run left to right, and within a vertical
    bank bottom to top. Thus backward/forward order does not flip across a view.
    Dimensions and spacing are schematic, independent of acquired ordinals.
    """

    direction: str
    rows: tuple[BankRow, ...]
    distance: float = 2.4
    gap: float = 0.15

    def __post_init__(self):
        _finite(self.distance, self.gap)
        if self.direction not in {"right", "left", "up", "down"} or not self.rows:
            raise ValueError("Directional banks need a cardinal direction and rows")
        if self.distance <= 0 or self.gap < 0:
            raise ValueError("Bank distance must be positive and gap nonnegative")


@dataclass(frozen=True)
class Composite:
    """Flat collection of arrangements within one view, without scene hierarchy."""

    parts: tuple[Directional | Circular | Grid | Explicit, ...]

    def __post_init__(self):
        if not self.parts or any(
            not isinstance(part, (Directional, Circular, Grid, Explicit)) for part in self.parts
        ):
            raise ValueError("Composite requires nonempty flat drawing arrangements")


@dataclass(frozen=True)
class View:
    """Named schematic or dimensional view, independent of physical banks."""

    key: str
    label: str
    mode: str
    convention: str
    arrangement: Circular | Grid | Explicit | Directional | Composite
    scale: str = "schematic"
    units: str = "dimensionless"
    orientation: ViewOrientation | None = None

    def __post_init__(self):
        if self.orientation is not None and not isinstance(self.orientation, ViewOrientation):
            raise ValueError("View orientation requires a typed ViewOrientation")
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
    return expand_arrangement(view.arrangement)


def expand_arrangement(arrangement) -> tuple[Placement, ...]:
    """Expand profile drawing data without changing readout identities."""
    if isinstance(arrangement, Composite):
        return tuple(element for part in arrangement.parts for element in expand_arrangement(part))
    if isinstance(arrangement, Directional):
        dx, dy = {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1)}[
            arrangement.direction
        ]
        elements = []
        edge = arrangement.distance - arrangement.rows[0].thickness / 2
        for row_index, row in enumerate(arrangement.rows):
            distance = edge + row.thickness / 2
            half_span = (sum(row.widths) + arrangement.gap * (len(row.widths) - 1)) / 2
            cursor = -half_span
            caption = (
                RowLabel(
                    row.label,
                    (
                        dx * distance + abs(dy) * (half_span + 0.12),
                        dy * distance + abs(dx) * (half_span + 0.12),
                    ),
                )
                if row.label
                else None
            )
            for column, (target, width) in enumerate(zip(row.targets, row.widths, strict=True)):
                offset = cursor + width / 2
                elements.append(
                    Placement(
                        target,
                        Rectangle(width, row.thickness) if dy else Rectangle(row.thickness, width),
                        dx * distance + abs(dy) * offset,
                        dy * distance + abs(dx) * offset,
                        sector=row_index,
                        ring=column,
                        row_label=caption,
                    )
                )
                cursor += width + arrangement.gap
            edge += row.thickness + arrangement.gap
        return tuple(elements)
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
                elements.append(
                    Placement(
                        target, shape, x, y, angle, row_index, column, arrangement.radial_rotation
                    )
                )
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
