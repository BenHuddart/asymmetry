"""Image layout reserves source dimensions before deferred assets arrive."""

from html.parser import HTMLParser

from PIL import Image

from docs._ext.lazy_images import _patch_html_file


class Images(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.tags = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.tags.append(dict(attrs))


def test_reserve_raster_and_svg_dimensions_without_losing_percentage_style(tmp_path):
    Image.new("RGB", (640, 320)).save(tmp_path / "figure.png")
    (tmp_path / "diagram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200" viewBox="0 0 800 200"></svg>'
    )
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="figure.png" class="logo"/>'
        '<img src="figure.png" style="width: 100%;" alt="Counts &amp; errors" />'
        '<img src="diagram.svg" />'
    )
    assert _patch_html_file(page)
    logo, figure, diagram = Images(page.read_text()).tags
    for tag in (logo, figure):
        assert (tag["width"], tag["height"]) == ("640", "320")
    assert logo["loading"] == "eager"
    assert figure["loading"] == "lazy"
    assert figure["style"] == "width: 100%;"
    assert figure["alt"] == "Counts & errors"
    assert (diagram["width"], diagram["height"]) == ("800", "200")
    assert not _patch_html_file(page)


def test_preserve_authored_dimensions_and_remote_images(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="https://example.invalid/plot.png" width="20" height="10" loading="eager" />'
    )
    _patch_html_file(page)
    (tag,) = Images(page.read_text()).tags
    assert (tag["width"], tag["height"], tag["loading"]) == ("20", "10", "eager")
