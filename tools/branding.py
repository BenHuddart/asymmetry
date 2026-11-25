"""Render the SVG master into application, installer, and documentation artwork."""

from __future__ import annotations

import argparse
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asymmetry.resources import ICON_SIZES  # noqa: E402

RESOURCES = ROOT / "src/asymmetry/resources"
MASTER = RESOURCES / "logo.svg"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
ICNS_RENDITIONS = {
    b"icp4": 16,
    b"ic11": 32,
    b"icp5": 32,
    b"ic12": 64,
    b"icp6": 64,
    b"ic07": 128,
    b"ic13": 256,
    b"ic08": 256,
    b"ic14": 512,
    b"ic09": 512,
    b"ic10": 1024,
}


def adjust_path_coordinates(path: ET.Element, offsets: dict[int, float]) -> None:
    """Apply the master's optical offsets to its ordered SVG path coordinates."""
    numbers = NUMBER.findall(path.attrib["d"])
    if not offsets or min(offsets) < 0 or max(offsets) >= len(numbers):
        raise ValueError("optical offsets must address coordinates in the master path")
    index = iter(range(len(numbers)))
    path.set(
        "d",
        NUMBER.sub(
            lambda match: f"{float(match.group()) + offsets.get(next(index), 0):g}",
            path.attrib["d"],
        ),
    )


def icon_svg(size: int, platform: str, *, refined: bool = True) -> bytes:
    """Lay out the mark without changing its aspect ratio or adding runtime masks."""
    if size <= 0 or platform not in ("windows", "macos"):
        raise ValueError("icon size must be positive and platform must be windows or macos")
    root = ET.parse(MASTER).getroot()
    group = root.find(f"{{{SVG_NS}}}g")
    if group is None:
        raise ValueError("SVG master must contain the asymmetry mark group")
    paths = {path.attrib["id"]: path for path in group}
    if paths.keys() != {"time-axis", "a-wave"}:
        raise ValueError("SVG master must contain the time-axis and a-wave paths")
    if refined and size <= 32:
        # SVG y increases downward: lift outer peaks and lower inner peaks to
        # strengthen the wave. Coordinate offsets belong to the master artwork.
        wave = paths["a-wave"]
        strength = 1 - (min(24, max(16, size)) - 16) * 0.35 / 8
        if size > 24:
            strength -= (size - 24) * 0.30 / 8
        offsets = {
            int(index): float(delta) * strength
            for pair in wave.attrib["data-small-offsets"].split()
            for index, delta in [pair.split(":")]
        }
        adjust_path_coordinates(wave, offsets)
        axis = paths["time-axis"]
        samples = [
            tuple(map(float, item.split(":"))) for item in axis.attrib["data-small-axis"].split()
        ]
        for (lower, width, head), (upper, next_width, next_head) in zip(samples, samples[1:]):
            if size <= upper:
                fraction = (max(size, lower) - lower) / (upper - lower)
                width += fraction * (next_width - width)
                head += fraction * (next_head - head)
                break
        numbers = [float(item) for item in NUMBER.findall(axis.attrib["d"])]
        if len(numbers) != 12:
            raise ValueError("the time axis must have 12 coordinates for optical adjustment")
        centre = numbers[6]
        offsets = {
            int(index): float(delta)
            for pair in axis.attrib["data-small-x-offsets"].split()
            for index, delta in [pair.split(":")]
        }
        for index, target in (
            (1, centre - width / 2),
            (4, centre - head),
            (8, centre + head),
            (10, centre + width / 2),
        ):
            offsets[index] = target - numbers[index]
        adjust_path_coordinates(axis, offsets)
    root.set("viewBox", "0 0 1440 1440")
    root.set("width", str(size))
    root.set("height", str(size))
    if platform == "windows":
        group.set("transform", "translate(-48 203.5)")
    else:
        group.set("transform", "translate(235.872 360.528) scale(.72474251497) translate(-100 -20)")
        tile = ET.Element(
            f"{{{SVG_NS}}}rect",
            {
                "x": "129.6",
                "y": "129.6",
                "width": "1180.8",
                "height": "1180.8",
                "rx": "263.3184",
                "fill": "#FAFAF9",
            },
        )
        root.insert(list(root).index(group), tile)
    return ET.tostring(root)


def render_svg(svg: bytes, width: int, height: int) -> Image.Image:
    """Render vector artwork directly at the requested pixel dimensions."""
    renderer = QSvgRenderer(QByteArray(svg))
    if not renderer.isValid():
        raise ValueError("branding SVG must be valid and renderable")
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(BytesIO(bytes(buffer.data()))).convert("RGBA")


def render_icon(size: int, platform: str, *, refined: bool = True) -> Image.Image:
    """Render a platform icon, applying optical adjustments through 32 pixels."""
    return render_svg(icon_svg(size, platform, refined=refined), size, size)


def write_ico(path: Path, platform: str) -> None:
    """Preserve independently rendered small icons in a multi-resolution ICO."""
    sizes = [size for size in ICON_SIZES if size <= 256]
    frames = [render_icon(size, platform) for size in sizes]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(
        path, format="ICO", sizes=[(size, size) for size in sizes], append_images=frames[:-1]
    )


def write_icns(path: Path) -> None:
    """Write PNG-backed ICNS entries, including native 16 px and Retina sizes."""
    entries = []
    for tag, size in ICNS_RENDITIONS.items():
        stream = BytesIO()
        render_icon(size, "macos").save(stream, format="PNG")
        data = stream.getvalue()
        entries.append(struct.pack(">4sI", tag, len(data) + 8) + data)
    payload = b"".join(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(">4sI", b"icns", len(payload) + 8) + payload)


def generated_assets() -> Iterator[tuple[Path, bytes]]:
    """Yield deterministic package and docs renditions of the authoritative SVG."""
    for platform in ("windows", "macos"):
        for size in ICON_SIZES:
            stream = BytesIO()
            render_icon(size, platform).save(stream, format="PNG")
            yield RESOURCES / f"icon_{platform}_{size}.png", stream.getvalue()
    stream = BytesIO()
    render_svg(MASTER.read_bytes(), 668, 496).save(stream, format="PNG")
    yield RESOURCES / "logo.png", stream.getvalue()
    yield ROOT / "docs/_static/logo.svg", MASTER.read_bytes()
    panel = ET.parse(MASTER).getroot()
    x, y, width, height = panel.attrib["viewBox"].split()
    panel.insert(
        0,
        ET.Element(
            f"{{{SVG_NS}}}rect",
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rx": "48",
                "fill": "#FFFFFF",
            },
        ),
    )
    panel.find(f"{{{SVG_NS}}}desc").text = "The Asymmetry mark on a light backing."
    yield ROOT / "docs/_static/logo-on-light.svg", ET.tostring(panel)
    # A light tile keeps the browser-tab icon legible in both browser themes.
    frames = [render_icon(size, "macos") for size in (16, 24, 32, 48)]
    stream = BytesIO()
    frames[-1].save(
        stream, format="ICO", sizes=[frame.size for frame in frames], append_images=frames[:-1]
    )
    yield ROOT / "docs/_static/favicon.ico", stream.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check generated assets without writing")
    mode.add_argument(
        "--platform", choices=("windows", "macos"), help="also build the installer icon"
    )
    args = parser.parse_args(argv)
    stale = []
    for path, content in generated_assets():
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if stale:
        parser.exit(
            1, "Regenerate branding with python tools/branding.py:\n" + "\n".join(stale) + "\n"
        )
    if args.platform == "windows":
        write_ico(ROOT / "build/icons/Asymmetry.ico", "windows")
    elif args.platform == "macos":
        write_icns(ROOT / "build/icons/Asymmetry.icns")
        for points in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                suffix = "@2x" if scale == 2 else ""
                path = ROOT / f"build/icons/Asymmetry.iconset/icon_{points}x{points}{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                render_icon(points * scale, "macos").save(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
