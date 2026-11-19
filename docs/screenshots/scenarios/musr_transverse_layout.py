"""MuSR's manual transverse memberships in the actual Detector Layout editor.

Companion to :doc:`/reference/detector_grouping` and the sourced
:doc:`/reference/instruments/isis_musr` definition. The two displayed projections
retain the manual memberships and experiment-specific balance coefficients.
"""

from asymmetry.core.instrument import get_instrument_layout

from ._base import Scenario, register


class MuSRTransverseLayoutScenario(Scenario):
    name = "musr_transverse_layout"
    description = (
        "MuSR transverse Top-Bottom and Fwd-Back projections in the Detector Layout editor."
    )
    size = (1100, 640)

    def build(self):
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        layout = get_instrument_layout("MuSR")
        preset = layout.presets["Transverse (Vector)"]
        dialog = DetectorLayoutDialog(
            layout,
            {key: list(group.detector_ids) for key, group in preset.groups.items()},
            group_names={key: group.name for key, group in preset.groups.items()},
            initial_preset_name=preset.name,
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
            projections=[projection.to_payload() for projection in preset.projections],
        )
        dialog._preset_combo.setCurrentText(preset.name)
        return dialog


register(MuSRTransverseLayoutScenario())
