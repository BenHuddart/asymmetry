"""Synthetic μSR datasets used by the screenshot scenarios.

See :mod:`docs.screenshots.data.archetypes` for the material catalogue and
the per-page rationale linking each generator to a textbook archetype.
"""

from .archetypes import (  # noqa: F401
    make_ag_lf_decoupling,
    make_ag_zf_gkt,
    make_alc_field_scan,
    make_emu_vector,
    make_euo_composite,
    make_euo_tf_tscan,
    make_generic_tf_for_processing,
    make_mgb2_sigma_t,
    make_pbf2_fmuf,
    make_ybco_knight_grouped,
    make_ybco_vortex_lattice,
)
