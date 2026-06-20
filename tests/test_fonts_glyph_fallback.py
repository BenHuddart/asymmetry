"""Regression tests for scientific-glyph coverage of the value font.

IBM Plex Mono's own cmap lacks Greek (U+0370–03FF), the MICRO SIGN (U+00B5),
and the Unicode super/subscript block (U+207B–209F). The live Windows/macOS app
renders these via Qt's automatic system-font substitution, but a bare-Linux/CI
host (and the offscreen QPA used here) performs no such substitution. These
tests pin the explicit fallback wiring and the U+00B5 micro-prefix normalization
so coverage is guaranteed on every platform.
"""

from __future__ import annotations

import pytest

# Representative glyphs the bundled value font cannot supply on its own.
_REPRESENTATIVE_GLYPHS = ["χ", "ν", "φ", "λ", "σ", "Δ", "Γ", "µ", "⁻", "¹"]

_MICRO_SIGN = "µ"  # µ — the prefix that must be used in unit strings
_GREEK_MU = "μ"  # μ — reserved for genuine Greek (muon symbols)


@pytest.fixture
def clean_substitutes(qapp: object):
    """Snapshot, clear, and restore the process-global IBM Plex Mono substitutes.

    ``QFont``'s substitution table is process-wide and outlives the
    session-scoped ``QApplication``, so without this the registration tests
    would pass on residual state left by an earlier test (or app startup) even
    if the wiring under test regressed. Clearing first makes each assertion
    reflect only the call it exercises; restoring afterwards keeps the live app
    behaviour intact for other tests.
    """
    if qapp is None:
        yield
        return
    from PySide6.QtGui import QFont

    from asymmetry.gui.styles import fonts

    saved = QFont.substitutes(fonts._FAMILY)
    QFont.removeSubstitutions(fonts._FAMILY)
    try:
        yield
    finally:
        QFont.removeSubstitutions(fonts._FAMILY)
        if saved:
            QFont.insertSubstitutions(fonts._FAMILY, saved)


def test_glyph_fallbacks_registered(qapp: object, clean_substitutes: None) -> None:
    """``register_bundled_fonts`` registers the Greek/µ/superscript fallbacks.

    Runs against a freshly cleared substitution table, so this fails if the
    registration call is gutted — it cannot pass on residual global state.
    """
    if qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtGui import QFont

    from asymmetry.gui.styles import fonts

    fonts.register_bundled_fonts()

    substitutes = {name.lower() for name in QFont.substitutes(fonts._FAMILY)}
    missing = [f for f in fonts._GLYPH_FALLBACKS if f.lower() not in substitutes]
    assert not missing, (
        f"glyph-coverage fallbacks not registered for {fonts._FAMILY!r}: {missing}; "
        f"registered substitutes = {sorted(substitutes)}"
    )


def test_register_glyph_fallbacks_is_idempotent(qapp: object, clean_substitutes: None) -> None:
    """Re-registering must not duplicate the substitution entries."""
    if qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtGui import QFont

    from asymmetry.gui.styles import fonts

    fonts.register_glyph_fallbacks()
    fonts.register_glyph_fallbacks()

    subs = [name.lower() for name in QFont.substitutes(fonts._FAMILY)]
    for family in fonts._GLYPH_FALLBACKS:
        assert subs.count(family.lower()) == 1, f"{family!r} duplicated: {subs}"


def test_mono_font_carries_fallback_families(qapp: object) -> None:
    """``mono_font`` declares the fallback families for per-glyph substitution.

    Qt6 uses the families list on a ``QFont`` as the character-level fallback
    chain, so the glyph-coverage faces must appear after the primary family.
    """
    if qapp is None:
        pytest.skip("PySide6 unavailable")
    from asymmetry.gui.styles import fonts

    families = fonts.mono_font().families()
    assert families[0] == fonts._FAMILY
    for family in fonts._GLYPH_FALLBACKS:
        assert family in families, f"{family!r} missing from mono_font fallbacks: {families}"


def test_value_font_renders_representative_glyphs(qapp: object) -> None:
    """On a real display the value font (incl. fallbacks) renders every glyph.

    The offscreen QPA ships no real font database and performs no automatic
    substitution, so ``inFont`` is meaningless there — this test is skipped
    under it (the registration tests above guard the headless/CI path). On a
    genuine platform font engine it asserts, so it would catch a regression that
    dropped the fallback families from :func:`mono_font`.
    """
    if qapp is None:
        pytest.skip("PySide6 unavailable")
    if qapp.platformName() == "offscreen":
        pytest.skip("offscreen QPA performs no font fallback; covered by registration tests")
    from PySide6.QtGui import QFontMetricsF

    from asymmetry.gui.styles import fonts

    fonts.register_bundled_fonts()
    fm = QFontMetricsF(fonts.mono_font())
    unrenderable = [ch for ch in _REPRESENTATIVE_GLYPHS if not fm.inFont(ch)]
    assert not unrenderable, f"value font cannot render representative glyphs: {unrenderable!r}"


def test_parameter_units_use_micro_sign_not_greek_mu() -> None:
    """Unit strings must use MICRO SIGN (U+00B5), never GREEK MU (U+03BC).

    The micro prefix (µs, µs⁻¹) is a unit, not a Greek letter; normalizing it to
    U+00B5 keeps the codepoint consistent with the registered glyph fallback and
    distinct from genuine muon symbols.
    """
    from asymmetry.core.fitting.parameters import PARAM_INFO_REGISTRY

    offenders = [
        (key, info.unit)
        for key, info in PARAM_INFO_REGISTRY.items()
        if _GREEK_MU in (info.unit or "")
    ]
    assert not offenders, (
        "parameter units contain GREEK MU (U+03BC) where MICRO SIGN (U+00B5) is "
        f"expected: {offenders}"
    )

    # And the canonical relaxation-rate unit is spelled with the micro sign.
    assert PARAM_INFO_REGISTRY["Lambda"].unit == f"{_MICRO_SIGN}s⁻¹"
