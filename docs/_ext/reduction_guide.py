"""Load the reduction teaching controls only on their owning page."""

from sphinx.application import Sphinx


def add_guide_assets(app: Sphinx, pagename: str, templatename, context, doctree) -> None:
    if pagename == "explanation/reduction_pipeline":
        app.add_css_file("reduction-guide.css")
        app.add_js_file("reduction-guide.mjs", type="module")


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("html-page-context", add_guide_assets)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
