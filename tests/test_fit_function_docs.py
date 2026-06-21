"""Documentation-placement policy for fit components.

Every fit component must be documented in the user-guide page that
corresponds to its category in the component picker
(``docs/reference/fit_functions/``), so the docs mirror the builder's
submenus. See the "Documentation policy" section of
``docs/reference/fit_functions/index.rst``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.fitting.composite import CATEGORY_REGISTRY, COMPONENTS

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "reference" / "fit_functions"

#: Picker category -> documentation page, from the canonical registry in core.
#: A new category must be added to CATEGORY_REGISTRY (with a docs page stem)
#: and get a page in the fit_functions toctree before components can use it.
CATEGORY_PAGES: dict[str, str] = {
    category: f"{stem}.rst" for category, stem in CATEGORY_REGISTRY.items() if stem
}


def _builtin_components() -> dict[str, object]:
    """The components the documentation policy applies to.

    User components (registered through the user-function facade) are exempt
    **by their ``user`` flag** — never by name list — since plugin authors
    cannot add pages to the shipped documentation.
    """
    return {name: definition for name, definition in COMPONENTS.items() if not definition.user}


def test_every_component_category_has_a_documentation_page() -> None:
    categories = {definition.category for definition in _builtin_components().values()}
    unmapped = categories - set(CATEGORY_PAGES)
    assert not unmapped, (
        f"Component categories without a documentation page: {sorted(unmapped)}. "
        "Add a page under docs/reference/fit_functions/ and register it in "
        "tests/test_fit_function_docs.py::CATEGORY_PAGES."
    )
    for page in CATEGORY_PAGES.values():
        assert (DOCS_DIR / page).is_file(), f"Missing documentation page {page}"


@pytest.mark.parametrize("name", sorted(_builtin_components()))
def test_component_documented_in_its_category_page(name: str) -> None:
    definition = COMPONENTS[name]
    page = DOCS_DIR / CATEGORY_PAGES[definition.category]
    text = page.read_text(encoding="utf-8")
    assert name in text, (
        f"Component '{name}' (category '{definition.category}') is not documented in "
        f"docs/reference/fit_functions/{page.name}. Document new fit functions in the "
        "page matching their picker submenu."
    )


def test_every_component_has_explicit_applicability_text() -> None:
    from asymmetry.core.fitting.component_docs import FIT_COMPONENT_APPLICABILITY

    missing = [name for name in _builtin_components() if name not in FIT_COMPONENT_APPLICABILITY]
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


def test_name_collisions_resolve_by_registry_kind() -> None:
    """'Constant' exists as both a fit component and a parameter-trend model;
    the kind-aware lookup must return each registry's own text."""
    from asymmetry.core.fitting.component_docs import get_component_applicability

    fit_text = get_component_applicability("Constant", kind="fit")
    trend_text = get_component_applicability("Constant", kind="parameter_model")
    assert "background" in fit_text.lower()
    assert "independent of x" in trend_text
    assert fit_text != trend_text


def test_applicability_text_cites_via_reference_lists() -> None:
    """Documentation policy: component-info applicability text must not cite
    textbook equation numbers or inline journal references — literature lives
    in the APS-style reference lists rendered below the applicability."""
    from asymmetry.core.fitting.component_docs import (
        FIT_COMPONENT_APPLICABILITY,
        PARAMETER_MODEL_APPLICABILITY,
    )

    for name, text in {
        **FIT_COMPONENT_APPLICABILITY,
        **PARAMETER_MODEL_APPLICABILITY,
    }.items():
        lowered = text.lower()
        assert "eqn" not in lowered, name
        assert "eq." not in lowered, name
        assert "ms-intro" not in lowered, name
        assert "phys. rev." not in text, name
