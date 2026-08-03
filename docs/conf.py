# Configuration file for the Sphinx documentation builder.

import importlib
import inspect
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('_ext'))

# Imported as a module (not `from asymmetry import __version__`) because
# `linkcode_resolve` below also needs `asymmetry.__file__` to anchor source paths.
import asymmetry

# -- Project information -----------------------------------------------------
project = 'Asymmetry'
copyright = '2026, Asymmetry Contributors'
author = 'Asymmetry Contributors'
release = asymmetry.__version__

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
]

templates_path = ['_templates']
exclude_patterns = ['_build', '_generated', 'Thumbs.db', '.DS_Store']
html_css_files = ['custom.css']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'logo.png'
html_favicon = 'logo.png'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': True,
}

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

# linkcode
#
# ``sphinx.ext.viewcode`` used to self-host a highlighted copy of every module
# under ``_modules/``. For this codebase that meant ~15 MB of generated HTML,
# with single pages (``_modules/asymmetry/gui/mainwindow.html``) reaching 2.7 MB
# and ~80k ``<span>`` elements -- slow enough to stall the published site in the
# browser. ``linkcode`` instead points each ``[source]`` link at GitHub.
#
# Links are pinned to the ``v{release}`` tag rather than ``main``: the tag always
# exists by the time a build is published (the version in ``pyproject.toml`` is
# the last released one), and the line anchors stay valid instead of drifting as
# the branch moves. A deploy from ``main`` between releases therefore points at
# the previous release's source, which is the intended trade for stable anchors.
_GITHUB_BLOB = 'https://github.com/BenHuddart/asymmetry/blob/{ref}/{path}#L{start}-L{end}'

# Released docs pin to the release tag. A local build of an uninstalled checkout
# has no distribution metadata (``__version__`` falls back to ``'unknown'``), so
# point those at ``main`` rather than emitting a dead ``vunknown`` ref.
_SOURCE_REF = f'v{release}' if re.match(r'^\d+\.\d+', release) else 'main'

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
    if domain != 'py':
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


# intersphinx
intersphinx_timeout = 10  # seconds; prevents a stalled CDN from blocking sphinx-build forever
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'lmfit': ('https://lmfit.github.io/lmfit-py/', None),
}
