"""PSI profile direction cues and repeated GPS readout contexts in the real editor."""

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

from ._base import Scenario, register


class PsiFlameLayoutScenario(Scenario):
    name = "psi_flame_layout"
    description = "FLAME's declared beam direction with its longitudinal preset."
    size = (1100, 680)

    def build(self):
        layout = get_instrument_layout("FLAME")
        preset = layout.presets["Longitudinal"]
        return DetectorLayoutDialog(
            layout,
            {gid: list(group.detector_ids) for gid, group in preset.groups.items()},
            group_names={gid: group.name for gid, group in preset.groups.items()},
            initial_preset_name=preset.name,
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
        )


class PsiGpsReadoutLayoutScenario(Scenario):
    name = "psi_gps_readout_layout"
    description = (
        "GPS combined/split readout rows, one beam cue per view, and a separate Mobile readout."
    )
    size = (1280, 780)

    def build(self):
        layout = get_instrument_layout("GPS-RD15")
        preset = layout.presets["WEP (spin-rotated)"]
        dialog = DetectorLayoutDialog(
            layout,
            {gid: list(group.detector_ids) for gid, group in preset.groups.items()},
            group_names={gid: group.name for gid, group in preset.groups.items()},
            initial_preset_name=preset.name,
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
            projections=[projection.to_payload() for projection in preset.projections],
        )
        dialog._preset_combo.setCurrentText(preset.name)
        return dialog


register(PsiFlameLayoutScenario())
register(PsiGpsReadoutLayoutScenario())


class PsiHalLayoutScenario(Scenario):
    name = "psi_hal_layout"
    description = (
        "HAL tangent readouts and selectable unlocated MV; axial viewing sense unspecified."
    )
    size = (1180, 760)

    def build(self):
        layout = get_instrument_layout("HAL")
        preset = layout.presets["Per-octant"]
        return DetectorLayoutDialog(
            layout,
            {gid: list(group.detector_ids) for gid, group in preset.groups.items()},
            group_names={gid: group.name for gid, group in preset.groups.items()},
            initial_preset_name=preset.name,
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
        )


register(PsiHalLayoutScenario())
