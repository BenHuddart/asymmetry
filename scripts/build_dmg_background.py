"""Generate an Asymmetry-themed drag-to-Applications DMG background image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 640
HEIGHT = 360


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_background(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGBA", (WIDTH, HEIGHT), "#f1f1f1")
    draw = ImageDraw.Draw(image)

    # Subtle vertical gradient for depth.
    for y in range(HEIGHT):
        ratio = y / max(1, HEIGHT - 1)
        shade = int(246 - (14 * ratio))
        draw.line([(0, y), (WIDTH, y)], fill=(shade, shade, shade, 255))

    # Red accent rails to match the Asymmetry logo palette.
    draw.rectangle((0, 0, WIDTH, 8), fill="#f32735")
    draw.rectangle((0, HEIGHT - 8, WIDTH, HEIGHT), fill="#f32735")

    # Icon drop targets.
    card_fill = "#fbfbfb"
    card_outline = "#d0d0d0"
    draw.rounded_rectangle((44, 86, 212, 266), radius=30, fill=card_fill, outline=card_outline, width=3)
    draw.rounded_rectangle((428, 86, 596, 266), radius=30, fill=card_fill, outline=card_outline, width=3)

    title_font = _load_font(30)
    subtitle_font = _load_font(18)
    draw.text((48, 26), "Install Asymmetry", fill="#2f2f2f", font=title_font)
    draw.text((48, 66), "Drag the app into Applications", fill="#6a6a6a", font=subtitle_font)

    arrow_points = [
        (246, 176),
        (366, 176),
        (366, 150),
        (422, 190),
        (366, 230),
        (366, 204),
        (246, 204),
    ]
    draw.polygon(arrow_points, fill="#f32735")

    image.save(output_path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Destination PNG path")
    args = parser.parse_args()
    build_background(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
