"""Sphinx extension: add ``loading="lazy"`` / ``decoding="async"`` to img tags.

Screenshots in this doc set are DPR-2.0 PNGs (see
``docs/screenshots/scenarios/_base.py``) and the corpus is only growing, so a
content-heavy reference page can ship several megabytes of images. Deferring
the ones the visitor hasn't scrolled to yet keeps first paint fast without
touching the images themselves.

Implemented as a ``build-finished`` post-process over the emitted HTML rather
than a ``docutils`` translator/node-visitor override: the theme's own
``visit_image`` already renders the ``<img>`` tag (srcset, classes, etc.), and
duplicating that logic here would be one more place to keep in sync with
``sphinx_rtd_theme``. A regex over the built files is simpler and cannot
collide with the theme's rendering. Simplicity is chosen over precise
largest-contentful-paint tuning: every page's *first* image is also marked
lazy, which is a fine trade for a docs site.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

# Matches an opening `<img ...>` tag that does not already declare `loading=`,
# so re-running the build (or this extension twice) stays idempotent.
_IMG_TAG_RE = re.compile(rb"<img(?![^>]*\bloading=)([^>]*)>")


def _add_lazy_attrs(match: re.Match[bytes]) -> bytes:
    return b'<img loading="lazy" decoding="async"' + match.group(1) + b">"


def _patch_html_file(path: Path) -> bool:
    """Rewrite ``path`` in place; return whether it changed."""
    original = path.read_bytes()
    patched = _IMG_TAG_RE.sub(_add_lazy_attrs, original)
    if patched == original:
        return False
    path.write_bytes(patched)
    return True


def _on_build_finished(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None:
        return
    if app.builder.name not in ("html", "dirhtml"):
        return

    outdir = Path(app.outdir)
    for html_path in outdir.rglob("*.html"):
        _patch_html_file(html_path)


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("build-finished", _on_build_finished)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
