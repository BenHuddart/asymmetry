"""Startup branding at the same dimensions and styling as the application."""

from PySide6.QtWidgets import QApplication, QWidget

from ._base import Scenario, register


class StartupScenario(Scenario):
    name = "startup"
    description = "The startup splash while Asymmetry prepares the project window."
    size = (420, 300)

    def build(self) -> QWidget:
        from asymmetry.gui.app import _create_splash_screen, _load_resource_pixmap

        app = QApplication.instance()
        return _create_splash_screen(app, _load_resource_pixmap("logo.png"))


register(StartupScenario())
