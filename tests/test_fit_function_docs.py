"""Documentation-placement policy for fit components.

Every fit component must be documented in the user-guide page that
corresponds to its category in the component picker
(``docs/user_guide/fit_functions/``), so the docs mirror the builder's
submenus. See the "Documentation policy" section of
``docs/user_guide/fit_functions/index.rst``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.fitting.composite import COMPONENTS

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "fit_functions"

#: Picker category -> documentation page. A new category must add a page here
#: (and to the fit_functions toctree) before components can use it.
CATEGORY_PAGES: dict[str, str] = {
    "Relaxation": "relaxation.rst",
    "Oscillation": "oscillation.rst",
    "Kubo-Toyabe": "kubo_toyabe.rst",
    "Muonium": "muonium.rst",
    "Nuclear dipolar": "nuclear_dipolar.rst",
    "Background": "background.rst",
    "Frequency Domain": "frequency_domain.rst",
}


def test_every_component_category_has_a_documentation_page() -> None:
    categories = {definition.category for definition in COMPONENTS.values()}
    unmapped = categories - set(CATEGORY_PAGES)
    assert not unmapped, (
        f"Component categories without a documentation page: {sorted(unmapped)}. "
        "Add a page under docs/user_guide/fit_functions/ and register it in "
        "tests/test_fit_function_docs.py::CATEGORY_PAGES."
    )
    for page in CATEGORY_PAGES.values():
        assert (DOCS_DIR / page).is_file(), f"Missing documentation page {page}"


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_documented_in_its_category_page(name: str) -> None:
    definition = COMPONENTS[name]
    page = DOCS_DIR / CATEGORY_PAGES[definition.category]
    text = page.read_text(encoding="utf-8")
    assert name in text, (
        f"Component '{name}' (category '{definition.category}') is not documented in "
        f"docs/user_guide/fit_functions/{page.name}. Document new fit functions in the "
        "page matching their picker submenu."
    )


def test_every_component_has_explicit_applicability_text() -> None:
    from asymmetry.core.fitting.component_docs import FIT_COMPONENT_APPLICABILITY

    missing = [name for name in COMPONENTS if name not in FIT_COMPONENT_APPLICABILITY]
    assert not missing, (
        f"Components without explicit applicability text (falling back to the generic "
        f"placeholder): {missing}. Add a physically motivated entry to "
        "FIT_COMPONENT_APPLICABILITY in core/fitting/component_docs.py."
    )


def test_category_pages_are_in_the_toctree() -> None:
    index_text = (DOCS_DIR / "index.rst").read_text(encoding="utf-8")
    for page in CATEGORY_PAGES.values():
        assert page.removesuffix(".rst") in index_text, (
            f"{page} is missing from the fit_functions toctree"
        )
