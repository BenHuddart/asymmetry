"""Sphinx policy for source/offline and publishing builds."""

import os
from pathlib import Path

from docutils import nodes
from sphinx import version_info
from sphinx.errors import ConfigError
from sphinx.util import logging

from docs.screenshots.capture import (
    check_screenshot_references,
    missing_screenshots,
    scenario_names_from_source,
)
from docs.screenshots.capture_corpus import _static_scenarios_by_module

logger = logging.getLogger(__name__)


# No inventories are acquired during either mode. These authored external
# objects retain their readable names without links; local targets stay strict.
OFFLINE_EXTERNAL_REFERENCES = (
    ("py:class", "numpy.random.Generator"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:func", "scipy.optimize.least_squares"),
    ("py:func", "numpy.fft.rfft"),
    ("py:func", "numpy.linalg.eigvalsh"),
    ("py:class", "enum.Enum"),
    ("py:class", "abc.ABC"),
    ("py:class", "pathlib.Path"),
    ("py:class", "PySide6.QtWidgets.QWidget"),
    ("py:class", "PySide6.QtWidgets.QDialog"),
    ("py:class", "PySide6.QtWidgets.QMainWindow"),
    ("py:class", "PySide6.QtCore.QSettings"),
    ("py:mod", "iminuit"),
    ("py:mod", "warnings"),
)


def configure(app, config):
    if config.docs_build_mode not in ("source", "publish"):
        raise ConfigError(f"Unknown docs_build_mode: {config.docs_build_mode!r}")
    config.intersphinx_mapping = {}
    if version_info >= (7, 3):
        config.show_warning_types = True
    config.nitpicky = True
    config.nitpick_ignore = list(OFFLINE_EXTERNAL_REFERENCES)
    config.autodoc_typehints = "none"
    config.napoleon_use_param = False
    config.napoleon_use_rtype = False
    docs = Path(app.srcdir)
    scenarios = docs / "screenshots/scenarios"
    problems = check_screenshot_references(docs, scenarios)
    if config.docs_build_mode == "publish":
        problems.extend(
            f"missing screenshot: {n}"
            for n in missing_screenshots(docs / "_generated/screenshots", scenarios)
        )
    if problems:
        raise ConfigError("Screenshot contract failed:\n" + "\n".join(problems))
    logger.info(
        "[docs] mode=%s; external inventories disabled (offline limitation)", config.docs_build_mode
    )
    logger.info(
        "[docs] declared external references without inventory links: %s",
        ", ".join(f"{kind}:{target}" for kind, target in OFFLINE_EXTERNAL_REFERENCES),
    )


def generated_images(app, doctree):
    images = list(doctree.findall(nodes.image))
    if not images:
        return
    docs = Path(app.srcdir)
    scenarios = docs / "screenshots/scenarios"
    ordinary_dir = docs / "_generated/screenshots"
    corpus_dir = docs / "_generated/corpus_screenshots"
    registered = {ordinary_dir / f"{name}.png" for name in scenario_names_from_source(scenarios)}
    registered.update(
        corpus_dir / f"{name}.png"
        for names in _static_scenarios_by_module(scenarios / "corpus").values()
        for name in names
    )
    omitted = []
    for node in images:
        uri = node["uri"]
        if "://" in uri or uri.startswith("data:"):
            continue
        # Registry identity follows the declared path, before symlinks resolve;
        # an alias cannot turn an unknown generated name into a registered asset.
        parent = (
            docs if uri.startswith(("/", "\\")) else Path(app.env.doc2path(app.env.docname)).parent
        )
        path = Path(os.path.abspath(parent / uri.lstrip("/\\")))
        if (
            any(path.is_relative_to(directory) for directory in (ordinary_dir, corpus_dir))
            and path not in registered
        ):
            logger.warning(
                "Generated image has no registered scenario: %s",
                path.relative_to(docs).as_posix(),
                location=node,
                type="docs",
                subtype="asset",
            )
        if app.config.docs_build_mode == "source" and path in registered and not path.is_file():
            notice = nodes.admonition(classes=["note"])
            notice += nodes.title(text="Optional generated screenshot")
            notice += nodes.paragraph(
                text=f"Source build: {path.name} is not captured. Publishing requires ordinary GUI captures; corpus figures may use labelled placeholders."
            )
            node.replace_self(notice)
            omitted.append(path.name)
    if omitted:
        logger.info(
            "[docs] %s: optional registered images omitted (%d): %s",
            app.env.docname,
            len(omitted),
            ", ".join(omitted),
        )


def rendered_markup(app, doctree):
    """Reject valid RST that accidentally exposes role syntax or initial lists."""
    for node in doctree.findall(nodes.Text):
        if ":math:`" in str(node) and not isinstance(
            node.parent, (nodes.literal, nodes.literal_block)
        ):
            logger.warning(
                "Literal math role in prose; separate mathematics from emphasis",
                location=node.parent,
                type="docs",
                subtype="markup",
            )
    for reference in doctree.findall(
        lambda node: isinstance(node, (nodes.footnote, nodes.citation))
    ):
        for node in reference.findall(nodes.enumerated_list):
            if node["enumtype"] in ("upperalpha", "loweralpha"):
                logger.warning(
                    "Alphabetic list in a reference; escape the first author initial",
                    location=node,
                    type="docs",
                    subtype="markup",
                )


def setup(app):
    app.add_config_value("docs_build_mode", "source", "env", types=[str])
    app.connect("config-inited", configure, priority=100)
    app.connect("doctree-read", generated_images, priority=100)
    app.connect("doctree-read", rendered_markup)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
