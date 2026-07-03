"""B8a: TF data on a longitudinal-default instrument should recommend a
transverse / spin-rotated grouping preset.

Live-testing finding (Round-3): PSI GPS transverse-field (TF) runs default to
the ``Longitudinal`` (Forward/Backward) preset, which washes out the precession
so the time-domain fit collapses. Nothing in the GUI hints that the user should
switch to ``Spin-rotated (B+U/F+D)``. The remedy is a small, pure recommendation
helper the Grouping / Detector-Layout dialog can surface as a nudge.

This test pins the core recommendation logic; the GUI wiring (showing the nudge)
is the implementer's choice.
"""

from __future__ import annotations

import pytest

from asymmetry.core.instrument import get_instrument_layout


def test_transverse_field_recommends_spin_rotated_preset_on_gps() -> None:
    try:
        from asymmetry.core.instrument import recommend_grouping_preset
    except ImportError:
        pytest.fail(
            "recommend_grouping_preset() helper does not exist yet — B8a fix should "
            "add a pure (instrument, field_direction) -> preset-name recommender."
        )

    gps = get_instrument_layout("GPS")

    # TF data: the Longitudinal default is wrong; recommend a spin-rotated preset.
    rec_tf = recommend_grouping_preset(gps, "Transverse")
    assert rec_tf is not None, "expected a transverse recommendation for TF data"
    assert rec_tf in gps.presets, f"recommended preset {rec_tf!r} not in GPS presets"
    assert rec_tf != "Longitudinal", "TF data must not be steered to the Longitudinal preset"
    assert "rotat" in rec_tf.lower() or "transverse" in rec_tf.lower(), (
        f"expected a spin-rotated/transverse preset for TF data, got {rec_tf!r}"
    )


def test_longitudinal_field_does_not_nudge_off_longitudinal() -> None:
    from asymmetry.core.instrument import recommend_grouping_preset

    gps = get_instrument_layout("GPS")
    rec_lf = recommend_grouping_preset(gps, "Longitudinal")
    # No nudge needed for LF/ZF: either no recommendation, or the Longitudinal preset.
    assert rec_lf in (None, "Longitudinal")
