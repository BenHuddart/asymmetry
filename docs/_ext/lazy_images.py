"""Reserve image space and defer content images without delaying the site logo."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from sphinx.application import Sphinx
from sphinx.util.images import get_image_size

_IMG_TAG_RE = re.compile(r'<img\b[^>]*>', re.IGNORECASE)


class ImageAttributes(HTMLParser):
    """Decode generated attributes without changing the original HTML tag."""

    def __init__(self, tag: str):
        super().__init__()
        self.attributes: dict[str, str | None] = {}
        self.feed(tag)

    def handle_starttag(self, tag, attrs):
        self.attributes = dict(attrs)


def _patch_html_file(path: Path) -> bool:
    """Reserve each local image's aspect ratio and preserve authored sizing."""
    original = path.read_text(encoding='utf-8')

    def image_layout(match: re.Match[str]) -> str:
        tag = match.group()
        attrs = ImageAttributes(tag).attributes
        additions = []
        if 'loading' not in attrs:
            loading = 'eager' if 'logo' in (attrs.get('class') or '').split() else 'lazy'
            additions.append(f'loading="{loading}"')
        if 'decoding' not in attrs:
            additions.append('decoding="async"')
        source = urlsplit(attrs.get('src') or '')
        if not source.scheme and not source.netloc and source.path:
            size = get_image_size(path.parent / unquote(source.path))
            if size and not any(key in attrs for key in ('width', 'height')):
                width, height = size
                additions.extend((f'width="{width}"', f'height="{height}"'))
        return '<img ' + ' '.join(additions) + tag[4:] if additions else tag

    patched = _IMG_TAG_RE.sub(image_layout, original)
    if patched == original:
        return False
    path.write_text(patched, encoding='utf-8')
    return True


def _on_build_finished(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.name not in ('html', 'dirhtml'):
        return
    for html_path in Path(app.outdir).rglob('*.html'):
        _patch_html_file(html_path)


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect('build-finished', _on_build_finished)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
