from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MACOS_ICON_PATH = ROOT / "packaging" / "macos_icon.py"


def _load_macos_icon():
    spec = importlib.util.spec_from_file_location("asymmetry_macos_icon", MACOS_ICON_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_icon(path: Path) -> None:
    Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(path)


def test_macos_icon_image_uses_rounded_square_mask(tmp_path: Path) -> None:
    macos_icon = _load_macos_icon()
    source = tmp_path / "source.png"
    _write_source_icon(source)

    icon = macos_icon.make_macos_icon_image(source, size=256)

    assert icon.size == (256, 256)
    assert icon.getpixel((0, 0))[3] == 0
    assert icon.getpixel((255, 0))[3] == 0
    assert icon.getpixel((0, 255))[3] == 0
    assert icon.getpixel((128, 0))[3] == 0
    assert icon.getpixel((128, 23))[3] == 255
    assert icon.getpixel((128, 128))[3] == 255


def test_write_iconset_outputs_all_macos_renditions(tmp_path: Path) -> None:
    macos_icon = _load_macos_icon()
    source = tmp_path / "source.png"
    iconset = tmp_path / "Asymmetry.iconset"
    _write_source_icon(source)

    generated = macos_icon.write_iconset(source, iconset)

    expected_names = {
        "icon_16x16.png",
        "icon_16x16@2x.png",
        "icon_32x32.png",
        "icon_32x32@2x.png",
        "icon_128x128.png",
        "icon_128x128@2x.png",
        "icon_256x256.png",
        "icon_256x256@2x.png",
        "icon_512x512.png",
        "icon_512x512@2x.png",
    }
    assert {path.name for path in generated} == expected_names

    largest = Image.open(iconset / "icon_512x512@2x.png").convert("RGBA")
    assert largest.size == (1024, 1024)
    assert largest.getpixel((0, 0))[3] == 0
    assert largest.getpixel((512, 0))[3] == 0
    assert largest.getpixel((512, 92))[3] == 255
    assert largest.getpixel((512, 512))[3] == 255


def test_write_icns_outputs_macos_icon_file(tmp_path: Path) -> None:
    macos_icon = _load_macos_icon()
    source = tmp_path / "source.png"
    icns = tmp_path / "Asymmetry.icns"
    _write_source_icon(source)

    macos_icon.write_icns(source, icns)

    assert icns.read_bytes().startswith(b"icns")
