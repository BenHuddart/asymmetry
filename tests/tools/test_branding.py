"""Branding renditions preserve the mark and the supplied small-size artwork."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6.QtSvg")

from tools import branding


def test_master_is_transparent_vector_artwork():
    root = ET.parse(branding.MASTER).getroot()
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    assert {path.attrib["fill"] for path in paths} == {"#E60007", "#1F4D8A"}
    assert {path.attrib["id"] for path in paths} == {"time-axis", "a-wave"}
    assert not root.findall(".//{http://www.w3.org/2000/svg}image")
    assert not root.findall(".//{http://www.w3.org/2000/svg}text")


@pytest.mark.parametrize("size", [16, 20, 24, 32])
def test_small_icons_strengthen_the_blue_axis_without_changing_the_a(size):
    original = branding.render_icon(size, "windows", refined=False)
    refined = branding.render_icon(size, "windows")
    assert original.size == refined.size == (size, size)
    # The A occupies the left half; only the wave and time axis need extra ink.
    assert (
        original.crop((0, 0, size // 2, size)).tobytes()
        == refined.crop((0, 0, size // 2, size)).tobytes()
    )

    def blue_coverage(image):
        pixels = np.asarray(image)
        blue = (pixels[:, :, 2] > pixels[:, :, 0]) & (pixels[:, :, 2] > pixels[:, :, 1])
        return pixels[:, :, 3][blue].sum()

    assert blue_coverage(refined) > blue_coverage(original)


def test_macos_tile_preserves_mark_proportions_and_clear_space():
    image = branding.render_icon(512, "macos")
    assert image.getpixel((0, 0))[3] == 0
    assert image.getpixel((256, 0))[3] == 0
    assert image.getpixel((256, 50)) == (250, 250, 249, 255)
    pixels = np.asarray(image)
    ink = Image.fromarray(((pixels[:, :, 3] > 0) & (pixels[:, :, 0] < 245)).astype("uint8") * 255)
    left, top, right, bottom = ink.getbbox()
    assert (bottom - top) / (right - left) == pytest.approx(891 / 1236, abs=0.015)
    assert left > 70 and top > 70 and right < 442 and bottom < 442


def test_windows_container_preserves_each_rendered_size(tmp_path):
    output = tmp_path / "Asymmetry.ico"
    branding.write_ico(output, "windows")
    icon = Image.open(output)
    assert icon.ico.sizes() == {(size, size) for size in branding.ICON_SIZES if size <= 256}
    for size in (16, 24, 32, 256):
        assert (
            icon.ico.getimage((size, size)).tobytes()
            == branding.render_icon(size, "windows").tobytes()
        )


def test_icns_contains_native_and_retina_images_without_resampling(tmp_path):
    output = tmp_path / "Asymmetry.icns"
    branding.write_icns(output)
    data = output.read_bytes()
    assert data[:4] == b"icns"
    assert struct.unpack(">I", data[4:8])[0] == len(data)
    entries = {}
    offset = 8
    while offset < len(data):
        tag, length = struct.unpack(">4sI", data[offset : offset + 8])
        entries[tag] = Image.open(BytesIO(data[offset + 8 : offset + length])).convert("RGBA")
        offset += length
    assert set(entries) == set(branding.ICNS_RENDITIONS)
    for tag, size in branding.ICNS_RENDITIONS.items():
        assert entries[tag].size == (size, size)
        assert entries[tag].tobytes() == branding.render_icon(size, "macos").tobytes()


def test_checked_in_renditions_match_the_master():
    for path, content in branding.generated_assets():
        if path.suffix != ".svg":
            actual = Image.open(path)
            expected = Image.open(BytesIO(content))
            if actual.format == "ICO":
                assert actual.ico.sizes() == expected.ico.sizes()
            actual = actual.convert("RGBA")
            expected = expected.convert("RGBA")
            assert actual.size == expected.size
            difference = np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
            assert difference.mean() < 0.5, f"Regenerate branding asset: {path.name}"
        else:
            assert path.read_bytes() == content, f"Regenerate branding asset: {path.name}"


@pytest.mark.parametrize("platform", ["windows", "macos"])
def test_check_mode_rejects_output_modes_without_writing(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(branding, "ROOT", tmp_path)
    monkeypatch.setattr(branding, "generated_assets", lambda: iter(()))
    with pytest.raises(SystemExit) as error:
        branding.main(["--check", "--platform", platform])
    assert error.value.code == 2
    assert not list(tmp_path.rglob("*"))
