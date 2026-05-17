"""Central tabbed workspace for time- and frequency-domain plots."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget


class PlotWorkspacePanel(QWidget):
    """Own the main-window plot tabs and active-domain state."""

    active_domain_changed = Signal(str)

    def __init__(
        self,
        *,
        time_panel: QWidget,
        frequency_panel: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._time_panel = time_panel
        self._frequency_panel = frequency_panel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._time_panel, "Time Domain")
        self._tabs.addTab(self._frequency_panel, "Frequency Domain")
        self._tabs.currentChanged.connect(self._on_current_tab_changed)
        layout.addWidget(self._tabs)

    def time_panel(self) -> QWidget:
        """Return the time-domain plot panel."""
        return self._time_panel

    def frequency_panel(self) -> QWidget:
        """Return the frequency-domain plot panel."""
        return self._frequency_panel

    def active_domain(self) -> str:
        """Return the currently selected plot domain."""
        return "frequency" if self._tabs.currentIndex() == 1 else "time"

    def set_active_domain(self, domain: str) -> None:
        """Select the visible plot domain tab."""
        token = str(domain).strip().lower()
        self._tabs.setCurrentIndex(1 if token == "frequency" else 0)

    def get_state(self) -> dict:
        """Return the serializable workspace state."""
        return {"active_domain": self.active_domain()}

    def current_panel(self) -> QWidget:
        """Return the currently visible plot panel."""
        return self._frequency_panel if self.active_domain() == "frequency" else self._time_panel

    def clear(self) -> None:
        """Clear both tabs and return to the time-domain view."""
        for panel in (self._time_panel, self._frequency_panel):
            if hasattr(panel, "clear"):
                panel.clear()
        self.set_active_domain("time")

    def export_current_plot(self) -> None:
        """Export the currently visible plot panel."""
        panel = self.current_panel()
        if hasattr(panel, "export_current_plot"):
            panel.export_current_plot()

    def restore_state(self, state: dict | None) -> None:
        """Restore tab selection from saved state."""
        if not isinstance(state, dict):
            self.set_active_domain("time")
            return
        self.set_active_domain(str(state.get("active_domain", "time")))

    def _on_current_tab_changed(self, _index: int) -> None:
        """Broadcast the newly active plot domain."""
        self.active_domain_changed.emit(self.active_domain())
