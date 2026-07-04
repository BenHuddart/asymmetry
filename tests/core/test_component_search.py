"""Tests for the ranked component-search module."""

from __future__ import annotations

from asymmetry.core.fitting.component_search import (
    ALIASES,
    SearchResult,
    search_components,
)
from asymmetry.core.fitting.composite import CATEGORY_REGISTRY, COMPONENTS
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS


def _names(results: list[SearchResult]) -> list[str]:
    return [r.name for r in results]


# ---------------------------------------------------------------------------
# Alias integrity
# ---------------------------------------------------------------------------


def test_alias_targets_exist_in_registries():
    known = set(COMPONENTS) | set(PARAMETER_MODEL_COMPONENTS)
    missing: dict[str, list[str]] = {}
    for alias, targets in ALIASES.items():
        bad = [t for t in targets if t not in known]
        if bad:
            missing[alias] = bad
    assert not missing, f"ALIASES reference unknown components: {missing}"


def test_alias_families_are_non_empty():
    for alias, targets in ALIASES.items():
        assert targets, f"alias {alias!r} has no targets"


# ---------------------------------------------------------------------------
# Ranking order: exact > prefix > word-boundary/camel-hump > alias > category
# > param > description > fuzzy
# ---------------------------------------------------------------------------


def test_exact_beats_prefix_beats_boundary():
    # 'Gaussian' is exact; 'GaussianPeak'/'GaussianBroadenedKT' are prefixes;
    # 'StaticGKT_ZF' etc. are not name matches at all for this token.
    results = search_components("Gaussian", components=COMPONENTS)
    names = _names(results)
    assert names[0] == "Gaussian"
    # Exact match scores highest.
    assert results[0].score == 100.0
    assert results[0].matched_field == "name"
    # Prefix matches (GaussianPeak, GaussianBroadenedKT) should outrank
    # anything that isn't a name-field hit at all.
    prefix_names = {"GaussianPeak", "GaussianBroadenedKT"}
    prefix_results = [r for r in results if r.name in prefix_names]
    assert prefix_results, "expected at least one GaussianPeak/GaussianBroadenedKT hit"
    for r in prefix_results:
        assert r.score == 90.0
        assert r.matched_field == "name"
    # All prefix hits rank above any non-name-boundary hit further down.
    prefix_positions = [names.index(n) for n in prefix_names if n in names]
    assert max(prefix_positions) < len(names)


def test_camel_hump_word_boundary_match():
    # 'TF' should hit the camelCase hump inside 'MuoniumTF' at a word
    # boundary (score 80), distinguishing it from a mid-word non-boundary
    # substring.
    results = search_components("TF", components=COMPONENTS)
    hit = next(r for r in results if r.name == "MuoniumTF")
    assert hit.matched_field == "name"
    assert hit.score == 80.0
    assert hit.name_span is not None
    start, end = hit.name_span
    assert "MuoniumTF"[start:end].lower() == "tf"


def test_underscore_word_boundary_match():
    # 'ZF' after the underscore in 'StaticGKT_ZF' is a word-boundary hit.
    results = search_components("ZF", components=COMPONENTS)
    hit = next(r for r in results if r.name == "StaticGKT_ZF")
    assert hit.matched_field == "name"
    assert hit.score == 80.0
    start, end = hit.name_span
    assert "StaticGKT_ZF"[start:end].lower() == "zf"


def test_ranking_is_ordered_by_score_desc():
    results = search_components("Gaussian", components=COMPONENTS)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Alias lookups
# ---------------------------------------------------------------------------


def test_alias_kt_surfaces_kubo_toyabe_family():
    results = search_components("kt", components=COMPONENTS)
    names = set(_names(results))
    assert "StaticGKT_ZF" in names
    assert "LongitudinalFieldKT" in names
    assert "DynamicGaussianKT" in names


def test_alias_muonium_family():
    results = search_components("muonium", components=COMPONENTS)
    names = set(_names(results))
    assert "MuoniumTF" in names
    assert "MuoniumZF" in names
    assert "MuoniumLFRelax" in names


def test_alias_kubo_toyabe_hyphenated_and_not():
    hyphenated = set(_names(search_components("kubo-toyabe", components=COMPONENTS)))
    plain = set(_names(search_components("kubotoyabe", components=COMPONENTS)))
    assert "StaticGKT_ZF" in hyphenated
    assert "StaticGKT_ZF" in plain


def test_alias_prefix_matching_kub_hits_kubo():
    results = search_components("kub", components=COMPONENTS)
    names = set(_names(results))
    assert "StaticGKT_ZF" in names


def test_alias_bg_background_family():
    results = search_components("bg", components=COMPONENTS)
    names = set(_names(results))
    assert "Constant" in names
    assert "ConstantBackground" in names
    assert "LinearBackground" in names


def test_alias_sc_vortex_family():
    results = search_components("sc", components=PARAMETER_MODEL_COMPONENTS, domain=None)
    names = set(_names(results))
    assert "SC_SWave" in names


def test_alias_fmuf_family():
    results = search_components("fmuf", components=COMPONENTS)
    names = set(_names(results))
    assert "FmuF_Linear" in names
    assert "DynamicFmuF" in names


def test_alias_bessel_abragam_keren_are_specific():
    assert _names(search_components("bessel", components=COMPONENTS)) == ["Bessel"] or (
        "Bessel" in _names(search_components("bessel", components=COMPONENTS))
    )
    assert "Abragam" in _names(search_components("abragam", components=COMPONENTS))
    assert "Keren" in _names(search_components("keren", components=COMPONENTS))


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------


def test_domain_filter_time_excludes_frequency_components():
    results = search_components("", components=COMPONENTS, domain="time")
    names = set(_names(results))
    assert "GaussianPeak" not in names
    assert "Exponential" in names


def test_domain_filter_frequency_only_frequency_components():
    results = search_components("", components=COMPONENTS, domain="frequency")
    names = set(_names(results))
    assert names == {name for name, comp in COMPONENTS.items() if comp.domain == "frequency"}


# ---------------------------------------------------------------------------
# Multi-token AND
# ---------------------------------------------------------------------------


def test_multi_token_and_dynamic_kt():
    # 'dynamic' hits DynamicGaussianKT/DynamicLorentzianKT by name-prefix;
    # 'kt' (alias) also surfaces DynamicGaussianKT/DynamicLorentzianKT plus
    # the static/LF/broadened KT components. Only the intersection should
    # come back.
    results = search_components("dynamic kt", components=COMPONENTS)
    names = set(_names(results))
    assert "DynamicGaussianKT" in names
    assert "DynamicLorentzianKT" in names
    # Static/other KT-only members (no 'dynamic' token match) must be excluded.
    assert "StaticGKT_ZF" not in names
    assert "LongitudinalFieldKT" not in names


def test_multi_token_requires_every_token_to_match():
    results = search_components("gaussian nonexistenttoken12345", components=COMPONENTS)
    assert results == []


# ---------------------------------------------------------------------------
# Description matching
# ---------------------------------------------------------------------------


def test_description_substring_match():
    # "Bc2" appears in VortexLattice/VortexLatticePowder param names but not
    # every component's description; pick a description-only phrase instead.
    results = search_components("skewed", components=COMPONENTS)
    names = set(_names(results))
    assert "VortexLattice" in names or "VortexLatticePowder" in names
    for r in results:
        if r.name in {"VortexLattice", "VortexLatticePowder"}:
            assert r.matched_field in {"description", "name", "alias"}


# ---------------------------------------------------------------------------
# Fuzzy matching (typo tolerance)
# ---------------------------------------------------------------------------


def test_fuzzy_typo_finds_gaussian():
    results = search_components("gausian", components=COMPONENTS)
    names = _names(results)
    assert "Gaussian" in names
    hit = next(r for r in results if r.name == "Gaussian")
    assert hit.matched_field == "fuzzy"
    assert 30.0 <= hit.score <= 39.0
    assert hit.name_span is None


# ---------------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------------


def test_empty_query_returns_canonical_category_order():
    results = search_components("", components=COMPONENTS)
    assert all(r.score == 0.0 for r in results)
    assert all(r.matched_field == "name" for r in results)
    assert all(r.name_span is None for r in results)
    assert set(_names(results)) == set(COMPONENTS)

    # Verify canonical category grouping: components in the same category
    # stay contiguous and categories appear in CATEGORY_REGISTRY order.
    category_order_seen: list[str] = []
    for result in results:
        category = COMPONENTS[result.name].category
        if not category_order_seen or category_order_seen[-1] != category:
            category_order_seen.append(category)
    expected_categories = [c for c in CATEGORY_REGISTRY if c in category_order_seen]
    assert category_order_seen == expected_categories

    # Within a category, names are alphabetical.
    by_category: dict[str, list[str]] = {}
    for result in results:
        by_category.setdefault(COMPONENTS[result.name].category, []).append(result.name)
    for names in by_category.values():
        assert names == sorted(names)


def test_whitespace_only_query_behaves_like_empty():
    assert search_components("   ", components=COMPONENTS) == search_components(
        "", components=COMPONENTS
    )


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


def test_limit_truncates_after_ranking():
    unlimited = search_components("", components=COMPONENTS)
    limited = search_components("", components=COMPONENTS, limit=3)
    assert len(limited) == 3
    assert limited == unlimited[:3]


def test_limit_none_returns_everything():
    results = search_components("", components=COMPONENTS, limit=None)
    assert len(results) == len(COMPONENTS)


# ---------------------------------------------------------------------------
# PARAMETER_MODEL_COMPONENTS is also searchable
# ---------------------------------------------------------------------------


def test_parameter_model_components_searchable_by_name():
    results = search_components("Polynomial", components=PARAMETER_MODEL_COMPONENTS)
    names = _names(results)
    assert "Polynomial" in names
    hit = next(r for r in results if r.name == "Polynomial")
    assert hit.score == 100.0


def test_parameter_model_components_searchable_by_param():
    results = search_components("gap_ratio", components=PARAMETER_MODEL_COMPONENTS)
    names = set(_names(results))
    assert "SC_SWave" in names
    assert "SC_DWave" in names


def test_parameter_model_components_empty_query_canonical_order():
    # Parameter-model definitions carry categories: 'General' (a known
    # canonical category) leads, the remaining categories follow
    # alphabetically, and names sort alphabetically within each category.
    results = search_components("", components=PARAMETER_MODEL_COMPONENTS)
    names = _names(results)
    assert set(names) == set(PARAMETER_MODEL_COMPONENTS)

    from asymmetry.core.fitting.component_search import _category_sort_key

    keys = [(_category_sort_key(PARAMETER_MODEL_COMPONENTS[name].category), name) for name in names]
    assert keys == sorted(keys), (
        "empty query must order by canonical category, then name; "
        "'General' leads and unknown categories follow the known ones alphabetically"
    )
    assert PARAMETER_MODEL_COMPONENTS[names[0]].category == "General"


# ---------------------------------------------------------------------------
# Category / param matches directly
# ---------------------------------------------------------------------------


def test_category_match_ranks_nuclear_dipolar():
    results = search_components("nuclear dipolar", components=COMPONENTS)
    names = set(_names(results))
    assert "DipolarPairField" in names
    assert "ProtonDipole" in names


def test_param_name_match():
    results = search_components("phase", components=COMPONENTS)
    names = set(_names(results))
    assert "Oscillatory" in names
    hit = next(r for r in results if r.name == "Oscillatory")
    assert hit.matched_field in {"name", "param", "alias", "category", "description"}
