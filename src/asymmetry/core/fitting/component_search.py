"""Ranked, deep search over a fit-component registry.

Powers the searchable component-library panel in the GUI. This module is
pure Python (no Qt) and works over *any* mapping of component name to a
descriptor exposing ``name``, ``description``, ``param_names``, and
``category`` — in practice
:data:`asymmetry.core.fitting.composite.COMPONENTS` (time/frequency
components) and
:data:`asymmetry.core.fitting.parameter_models.PARAMETER_MODEL_COMPONENTS`
(the trending/global-fit basis functions), so the caller passes the
registry it wants searched rather than this module importing one.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from asymmetry.core.fitting.composite import CATEGORY_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from asymmetry.core.fitting.composite import ComponentDefinition

#: Canonical category display order. Both ``COMPONENTS``-shaped and
#: ``PARAMETER_MODEL_COMPONENTS``-shaped registries must rank consistently, and
#: the latter has no category field at all (see ``_category_of``), so this is
#: imported directly from the single source of truth rather than duplicated.
_CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_REGISTRY)


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit from :func:`search_components`."""

    name: str
    score: float
    matched_field: str  # 'name' | 'alias' | 'category' | 'param' | 'description' | 'fuzzy'
    #: Character span of the match within the component NAME, for
    #: highlighting. ``None`` when the match is not in the name (e.g. an
    #: alias, category, parameter, or description hit with no literal
    #: occurrence in the name).
    name_span: tuple[int, int] | None


#: Alias -> component names the alias should surface, in addition to whatever
#: direct name/category/param/description matches already find. Every target
#: must be a real key in ``COMPONENTS`` or ``PARAMETER_MODEL_COMPONENTS`` —
#: enforced by ``test_alias_targets_exist_in_registries`` so a renamed or
#: removed component fails the test suite instead of silently going stale.
ALIASES: dict[str, tuple[str, ...]] = {
    # kt / kubo-toyabe / kubo -- Kubo-Toyabe family
    "kt": (
        "StaticGKT_ZF",
        "LongitudinalFieldKT",
        "DynamicGaussianKT",
        "DynamicLorentzianKT",
        "GaussianBroadenedKT",
    ),
    "kubo": (
        "StaticGKT_ZF",
        "LongitudinalFieldKT",
        "DynamicGaussianKT",
        "DynamicLorentzianKT",
        "GaussianBroadenedKT",
    ),
    "kubo-toyabe": (
        "StaticGKT_ZF",
        "LongitudinalFieldKT",
        "DynamicGaussianKT",
        "DynamicLorentzianKT",
        "GaussianBroadenedKT",
    ),
    "kubotoyabe": (
        "StaticGKT_ZF",
        "LongitudinalFieldKT",
        "DynamicGaussianKT",
        "DynamicLorentzianKT",
        "GaussianBroadenedKT",
    ),
    # zf / zero-field
    "zf": ("StaticGKT_ZF", "MuoniumZF"),
    "zero-field": ("StaticGKT_ZF", "MuoniumZF"),
    "zerofield": ("StaticGKT_ZF", "MuoniumZF"),
    # tf / transverse (field)
    "tf": (
        "Oscillatory",
        "OscillatoryField",
        "VortexLattice",
        "VortexLatticePowder",
        "MuoniumTF",
        "MuoniumLowTF",
        "MuoniumHighTF",
        "MuoniumHighTFAniso",
    ),
    "transverse": (
        "Oscillatory",
        "OscillatoryField",
        "VortexLattice",
        "VortexLatticePowder",
        "MuoniumTF",
        "MuoniumLowTF",
        "MuoniumHighTF",
        "MuoniumHighTFAniso",
    ),
    # lf / longitudinal (field)
    "lf": (
        "LongitudinalFieldKT",
        "MuoniumLFRelax",
        "DiffusionLF_1D",
        "DiffusionLF_2D",
        "DiffusionLF_3D",
        "BallisticLF_1D",
        "BallisticLF_2D",
        "BallisticLF_3D",
    ),
    "longitudinal": (
        "LongitudinalFieldKT",
        "MuoniumLFRelax",
        "DiffusionLF_1D",
        "DiffusionLF_2D",
        "DiffusionLF_3D",
        "BallisticLF_1D",
        "BallisticLF_2D",
        "BallisticLF_3D",
    ),
    # mu / muonium
    "mu": (
        "MuoniumTF",
        "MuoniumLowTF",
        "MuoniumZF",
        "MuoniumHighTF",
        "MuoniumHighTFAniso",
        "MuoniumLFRelax",
    ),
    "muonium": (
        "MuoniumTF",
        "MuoniumLowTF",
        "MuoniumZF",
        "MuoniumHighTF",
        "MuoniumHighTFAniso",
        "MuoniumLFRelax",
        "MuRepolarisation",
    ),
    # sc / superconductor / vortex / lambda -- vortex-lattice lineshape family
    "sc": (
        "VortexLattice",
        "VortexLatticePowder",
        "SC_SWave",
        "SC_DWave",
        "SC_AnisotropicS_Cos4",
        "SC_NonmonotonicD",
        "SC_PWaveAxial",
        "SC_ExtendedS",
        "SC_SPlusG",
        "SC_AlphaModel",
        "SC_TwoGap_SS",
        "SC_TwoGap_SD",
        "SC_SWave_Q",
        "SC_DWave_Q",
        "SC_SPlusG_Q",
        "SC_Brandt_VortexLattice",
        "SC_Brandt_VortexLattice_Powder",
    ),
    "superconductor": (
        "SC_SWave",
        "SC_DWave",
        "SC_AnisotropicS_Cos4",
        "SC_NonmonotonicD",
        "SC_PWaveAxial",
        "SC_ExtendedS",
        "SC_SPlusG",
        "SC_AlphaModel",
        "SC_TwoGap_SS",
        "SC_TwoGap_SD",
        "SC_SWave_Q",
        "SC_DWave_Q",
        "SC_SPlusG_Q",
        "SC_Brandt_VortexLattice",
        "SC_Brandt_VortexLattice_Powder",
    ),
    "vortex": (
        "VortexLattice",
        "VortexLatticePowder",
        "SC_Brandt_VortexLattice",
        "SC_Brandt_VortexLattice_Powder",
    ),
    "lambda": ("VortexLattice", "VortexLatticePowder"),
    # bg / background / baseline / flat
    "bg": ("Constant", "ConstantBackground", "LinearBackground", "Lambda_bg"),
    "background": ("Constant", "ConstantBackground", "LinearBackground", "Lambda_bg"),
    "baseline": ("Constant", "ConstantBackground", "LinearBackground", "Lambda_bg"),
    "flat": ("Constant", "ConstantBackground"),
    # stretched / beta
    "stretched": ("StretchedExponential",),
    "beta": ("StretchedExponential",),
    # fmuf / f-mu-f / fluorine
    "fmuf": ("MuF", "FmuF_Linear", "DynamicFmuF", "FmuF_General", "FmuF_Triangle"),
    "f-mu-f": ("MuF", "FmuF_Linear", "DynamicFmuF", "FmuF_General", "FmuF_Triangle"),
    "fluorine": ("MuF", "FmuF_Linear", "DynamicFmuF", "FmuF_General", "FmuF_Triangle"),
    # dipolar / dipole
    "dipolar": ("DipolarPairField", "ProtonDipole", "ElectronDipole", "DipolarSpinJ"),
    "dipole": ("DipolarPairField", "ProtonDipole", "ElectronDipole", "DipolarSpinJ"),
    # relax / relaxation / decay -- the simple relaxation components
    "relax": (
        "Exponential",
        "Gaussian",
        "StretchedExponential",
        "RischKehr",
        "Keren",
        "Abragam",
        "ExponentialDecay",
    ),
    "relaxation": (
        "Exponential",
        "Gaussian",
        "StretchedExponential",
        "RischKehr",
        "Keren",
        "Abragam",
        "ExponentialDecay",
    ),
    "decay": (
        "Exponential",
        "Gaussian",
        "StretchedExponential",
        "ExponentialDecay",
    ),
    # osc / precession / cosine / frequency -- oscillatory components
    "osc": ("Oscillatory", "OscillatoryField", "Bessel", "VortexLattice", "VortexLatticePowder"),
    "precession": (
        "Oscillatory",
        "OscillatoryField",
        "Bessel",
        "VortexLattice",
        "VortexLatticePowder",
    ),
    "cosine": ("Oscillatory", "OscillatoryField"),
    "frequency": ("Oscillatory", "OscillatoryField", "Bessel"),
    # bessel
    "bessel": ("Bessel",),
    # abragam
    "abragam": ("Abragam",),
    # keren
    "keren": ("Keren",),
}


def _alias_targets_for_token(token: str) -> tuple[str, ...] | None:
    """Return merged alias targets for *token*, or ``None`` if no alias matches.

    A token matches an alias key exactly, or matches a prefix of it (so
    ``'kub'`` hits ``'kubo'``). Hyphenated alias keys (e.g. ``'kubo-toyabe'``)
    also match the token with the hyphen stripped, so a query typed either
    with or without the hyphen surfaces the same family.

    A short prefix can match more than one alias family (e.g. ``'b'`` hits
    ``bg``/``background``/``baseline``/``beta``/``bessel``): every matching
    key's targets are merged into one deduped tuple, in first-seen order
    across ``ALIASES`` (dict insertion order), rather than stopping at the
    first match.
    """
    token_nohyphen = token.replace("-", "")
    merged: list[str] = []
    seen: set[str] = set()
    matched = False
    for alias_key, targets in ALIASES.items():
        key_nohyphen = alias_key.replace("-", "")
        if not (alias_key.startswith(token) or key_nohyphen.startswith(token_nohyphen)):
            continue
        matched = True
        for target in targets:
            if target not in seen:
                seen.add(target)
                merged.append(target)
    return tuple(merged) if matched else None


def _category_of(component: ComponentDefinition) -> str:
    """Return the category of *component*, defaulting to 'General'.

    ``ParameterModelComponentDefinition`` (the ``PARAMETER_MODEL_COMPONENTS``
    descriptor) has no ``category`` attribute at all, so this falls back to
    the shared default bucket rather than requiring a category field on every
    registry.
    """
    return getattr(component, "category", "General") or "General"


def _category_sort_key(category: str) -> tuple[int, str]:
    """Sort key implementing canonical category order, then alphabetical."""
    try:
        index = _CATEGORY_ORDER.index(category)
    except ValueError:
        index = len(_CATEGORY_ORDER)
    return (index, category)


def _word_boundary_span(text_lower: str, token: str) -> tuple[int, int] | None:
    """Return the span of the first word-boundary occurrence of *token*.

    A "word boundary" start is the beginning of the string, a position right
    after an underscore, or a camelCase/PascalCase hump (an uppercase letter
    in the *original*-case name that follows a lowercase letter or digit).
    Matching itself is case-insensitive; ``text_lower`` is the lower-cased
    name and ``token`` is already lower-cased by the caller.
    """
    if not token:
        return None
    start = 0
    while True:
        idx = text_lower.find(token, start)
        if idx == -1:
            return None
        if idx == 0 or text_lower[idx - 1] == "_":
            return (idx, idx + len(token))
        start = idx + 1


def _camel_hump_span(name: str, token: str) -> tuple[int, int] | None:
    """Return the span of *token* starting at a camelCase hump in *name*.

    A hump is an uppercase letter preceded by a lowercase letter or digit
    (e.g. the 'T' in 'MuoniumTF', or the 'K' in a hypothetical 'FooKT').
    Matching is case-insensitive against ``token``.
    """
    token_lower = token.lower()
    name_lower = name.lower()
    for i in range(1, len(name)):
        prev = name[i - 1]
        cur = name[i]
        if cur.isupper() and (prev.islower() or prev.isdigit()):
            if name_lower.startswith(token_lower, i):
                return (i, i + len(token))
    return None


def _underscore_span(name_lower: str, token: str) -> tuple[int, int] | None:
    """Return the span of *token* immediately following an underscore."""
    marker = "_" + token
    idx = name_lower.find(marker)
    if idx == -1:
        return None
    start = idx + 1
    return (start, start + len(token))


def _score_token_against_name(name: str, token: str) -> tuple[float, tuple[int, int] | None] | None:
    """Score *token* against a component *name* alone (exact/prefix/boundary/fuzzy).

    Returns ``(score, name_span)`` for the best of: exact match (100), name
    prefix (90), or a word-boundary substring match -- start-of-string,
    after an underscore, or at a camelCase hump (80). A substring match that
    is not at a recognised word boundary falls through to fuzzy matching
    (30-39, scaled by :class:`difflib.SequenceMatcher` ratio), tried last as
    a typo-tolerant fallback. Returns ``None`` if nothing matches.
    """
    name_lower = name.lower()
    token_lower = token.lower()

    if name_lower == token_lower:
        return (100.0, (0, len(name)))

    if name_lower.startswith(token_lower):
        return (90.0, (0, len(token)))

    boundary_span = _word_boundary_span(name_lower, token_lower)
    if boundary_span is not None:
        return (80.0, boundary_span)

    hump_span = _camel_hump_span(name, token)
    if hump_span is not None:
        return (80.0, hump_span)

    underscore_span = _underscore_span(name_lower, token_lower)
    if underscore_span is not None:
        return (80.0, underscore_span)

    ratio = difflib.SequenceMatcher(a=name_lower, b=token_lower).ratio()
    if ratio >= 0.75:
        return (30.0 + 9.0 * (ratio - 0.75) / 0.25, None)

    return None


def _score_token(
    token: str,
    name: str,
    component: ComponentDefinition,
) -> tuple[float, str, tuple[int, int] | None] | None:
    """Return ``(score, matched_field, name_span)`` for one token, or ``None``."""
    best: tuple[float, str, tuple[int, int] | None] | None = None

    def consider(score: float, field: str, span: tuple[int, int] | None) -> None:
        nonlocal best
        if best is None or score > best[0]:
            best = (score, field, span)

    name_hit = _score_token_against_name(name, token)
    if name_hit is not None:
        score, span = name_hit
        field = "fuzzy" if span is None else "name"
        consider(score, field, span)

    alias_targets = _alias_targets_for_token(token.lower())
    if alias_targets is not None and name in alias_targets:
        consider(70.0, "alias", None)

    category = _category_of(component).lower()
    token_lower = token.lower()
    if category == token_lower or category.startswith(token_lower) or token_lower in category:
        consider(60.0, "category", None)

    for param_name in component.param_names:
        if token_lower == param_name.lower() or token_lower in param_name.lower():
            consider(50.0, "param", None)
            break

    description = getattr(component, "description", "") or ""
    if token_lower in description.lower():
        consider(40.0, "description", None)

    return best


def _canonical_order_key(name: str, component: ComponentDefinition) -> tuple[tuple[int, str], str]:
    return (_category_sort_key(_category_of(component)), name)


def search_components(
    query: str,
    *,
    components: Mapping[str, ComponentDefinition],
    domain: str | None = None,
    limit: int | None = None,
) -> list[SearchResult]:
    """Return ranked :class:`SearchResult` hits for *query* over *components*.

    ``components`` is any name -> descriptor mapping exposing ``name``,
    ``description``, ``param_names``, and (optionally) ``category`` and
    ``domain`` attributes -- both
    :data:`asymmetry.core.fitting.composite.COMPONENTS` and
    :data:`asymmetry.core.fitting.parameter_models.PARAMETER_MODEL_COMPONENTS`
    satisfy this shape.

    An empty/whitespace query returns every component (after the ``domain``
    filter) in canonical category order, each scored 0.0. Multi-token queries
    require every token to match somewhere (AND); the reported score/field
    come from the first token's best match, and ``name_span`` from the first
    token's name-field match (``None`` if the first token didn't match the
    name).
    """
    if domain is not None:
        candidates = {
            name: comp
            for name, comp in components.items()
            if getattr(comp, "domain", None) == domain
        }
    else:
        candidates = dict(components)

    stripped = query.strip()
    if not stripped:
        ordered = sorted(candidates.items(), key=lambda item: _canonical_order_key(*item))
        results = [
            SearchResult(name=name, score=0.0, matched_field="name", name_span=None)
            for name, _component in ordered
        ]
        return results[:limit] if limit is not None else results

    tokens = stripped.split()

    scored: list[tuple[SearchResult, tuple[int, str]]] = []
    for name, component in candidates.items():
        per_token_hits: list[tuple[float, str, tuple[int, int] | None]] = []
        matched_all = True
        for token in tokens:
            hit = _score_token(token, name, component)
            if hit is None:
                matched_all = False
                break
            per_token_hits.append(hit)
        if not matched_all:
            continue

        first_score, first_field, first_span = per_token_hits[0]
        result = SearchResult(
            name=name,
            score=first_score,
            matched_field=first_field,
            name_span=first_span if first_field == "name" else None,
        )
        scored.append((result, _canonical_order_key(name, component)))

    scored.sort(key=lambda pair: (-pair[0].score, pair[1]))
    results = [result for result, _order_key in scored]
    return results[:limit] if limit is not None else results


__all__ = ["ALIASES", "SearchResult", "search_components"]
