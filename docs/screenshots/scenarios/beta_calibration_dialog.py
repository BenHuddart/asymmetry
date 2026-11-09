"""Joint beta calibration uses the shared TF calibration controls."""

from ._base import register
from .alpha_calibration_dialog import FreeBetaScenario


class BetaCalibrationDialogScenario(FreeBetaScenario):
    name = "beta_calibration_dialog"
    description = "Joint TF alpha/beta calibration with a selected model and conditional errors."


register(BetaCalibrationDialogScenario())
