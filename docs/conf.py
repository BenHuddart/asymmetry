# Configuration file for the Sphinx documentation builder.

import importlib
import inspect
import os
import sys
from pathlib import Path

_DOCS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_DOCS_ROOT.parent / 'src'))
sys.path.insert(0, str(_DOCS_ROOT / '_ext'))
sys.path.insert(0, str(_DOCS_ROOT.parent))

# Imported as a module (not `from asymmetry import __version__`) because
# `linkcode_resolve` below also needs `asymmetry.__file__` to anchor source paths.
import asymmetry
from tools.docs_source import source_identity

# -- Project information -----------------------------------------------------
project = 'Asymmetry'
copyright = '2026, Asymmetry Contributors'
author = 'Asymmetry Contributors'
_source = source_identity(asymmetry.__file__, asymmetry.__version__)
release = _source.label
html_context = {'docs_source_label': _source.label, 'docs_source_revision': _source.revision}

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'lazy_images',
    'reduction_guide',
    'tools.docs_policy',
    'tools.instrument_docs',
]

instrument_review_directory = 'reference/instruments'

templates_path = ['_templates']
exclude_patterns = ['_build', '_generated', 'Thumbs.db', '.DS_Store']
html_css_files = ['custom.css']
# Search uses titles and headings without fetching every matching page for a snippet.
html_show_search_summary = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.ico'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': True,
}

# mathjax: pin an exact version (rather than the floating "@4" major-version
# tag) so the CDN URL is immutable and can be cached long-term, and use the
# tex-chtml bundle (rather than tex-mml-chtml) to drop the unused MathML
# input processor -- this repo only authors math with the tex extension.
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-chtml.js'

# -- Extension configuration -------------------------------------------------

# autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# autosummary
autosummary_generate = True

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Source line numbers and links refer to the same imported, clean checkout.
_GITHUB_BLOB = 'https://github.com/BenHuddart/asymmetry/blob/{ref}/{path}#L{start}-L{end}'
_SOURCE_REF = _source.revision

# Root of the importable source tree (``.../src``), derived from the package
# actually imported so the links are correct from a git worktree too.
_SRC_ROOT = os.path.realpath(os.path.join(os.path.dirname(asymmetry.__file__), os.pardir))


def _unwrap_object(obj):
    """Peel decorators/descriptors off ``obj`` until source lookup can work."""
    if isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    if obj is None:
        return None
    return inspect.unwrap(obj)


def linkcode_resolve(domain, info):
    """Return a GitHub URL for the Python object described by ``info``."""
    if domain != 'py' or _SOURCE_REF is None:
        return None
    module_name = info.get('module')
    fullname = info.get('fullname')
    if not module_name or not fullname:
        return None

    try:
        obj = importlib.import_module(module_name)
        for part in fullname.split('.'):
            obj = getattr(obj, part)
        obj = _unwrap_object(obj)
        if obj is None:
            return None
        source_file = inspect.getsourcefile(obj)
        if not source_file:
            return None
        lines, start = inspect.getsourcelines(obj)
    except Exception:
        # Attributes without source, C extensions, lazily-created objects,
        # re-exported names -- all simply get no [source] link.
        return None

    rel_path = os.path.relpath(os.path.realpath(source_file), _SRC_ROOT)
    if rel_path.startswith(os.pardir) or os.path.isabs(rel_path):
        return None

    return _GITHUB_BLOB.format(
        ref=_SOURCE_REF,
        path='src/' + rel_path.replace(os.sep, '/'),
        start=start,
        end=start + len(lines) - 1,
    )


# Inventory lookups are disabled by tools.docs_policy in both gated modes.
intersphinx_mapping = {}
