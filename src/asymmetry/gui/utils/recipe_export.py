"""Write the shared disclosed recipe header for plotted data and GLE sidecars."""

import json

from asymmetry.core.data.recipe_export import exported_recipe


def write_recipe_header(stream, payload, *, x_range=None, bake_waterfall_offset=True):
    """Keep the header bound to the actual writer's row selection and offset."""
    recipe = exported_recipe(payload, x_range=x_range, bake_waterfall_offset=bake_waterfall_offset)
    stream.write(
        "! effective_recipe: " + json.dumps(recipe, sort_keys=True, allow_nan=False) + "\n"
    )
    display = {
        key: value
        for key, value in payload.get("run_metadata", {}).items()
        if key
        in {"rrf_frame", "rrf_frequency_mhz", "rrf_phase_deg", "rrf_bandwidth_mhz", "rrf_trim"}
    }
    if display:
        stream.write(
            "! display_transform: " + json.dumps(display, sort_keys=True, allow_nan=False) + "\n"
        )
