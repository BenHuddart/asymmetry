"""Central tabbed workspace for FB asymmetry, grouped, and frequency plots."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStackedWidget, QTabBar, QVBoxLayout, QWidget


class PlotWorkspacePanel(QWidget):
    """Own the main-window plot tabs and active-domain state."""

    active_domain_changed = Signal(str)
    active_view_changed = Signal(str)

    _VIEW_TOKENS = ("fb_asymmetry", "groups", "frequency")
    _VIEW_LABELS = {
        "fb_asymmetry": "FB Asymmetry",
        "groups": "Individual Groups",
        "frequency": "Frequency Domain",
    }

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
        self._last_time_view = "fb_asymmetry"
        self._enabled_views = {"fb_asymmetry", "frequency"}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setExpanding(False)
        for token in self._VIEW_TOKENS:
            self._tab_bar.addTab(self._VIEW_LABELS[token])
        self._tab_bar.currentChanged.connect(self._on_current_tab_changed)
        layout.addWidget(self._tab_bar)

        self._panel_stack = QStackedWidget()
        self._panel_stack.addWidget(self._time_panel)
        self._panel_stack.addWidget(self._frequency_panel)
        layout.addWidget(self._panel_stack)

        self.set_available_views(["fb_asymmetry"])
        self.set_active_view("fb_asymmetry")

    def time_panel(self) -> QWidget:
        """Return the time-domain plot panel."""
        return self._time_panel

    def frequency_panel(self) -> QWidget:
        """Return the frequency-domain plot panel."""
        return self._frequency_panel

    def active_view(self) -> str:
        """Return the currently selected top-level plot view token."""
        index = self._tab_bar.currentIndex()
        if index < 0 or index >= len(self._VIEW_TOKENS):
            return "fb_asymmetry"
        return self._VIEW_TOKENS[index]

    def active_domain(self) -> str:
        """Return the currently selected plot domain."""
        return "frequency" if self.active_view() == "frequency" else "time"

    def set_available_views(self, views: list[str]) -> None:
        """Enable the supported top-level plot views for the current selection."""
        enabled = {"fb_asymmetry", "frequency"}
        for token in views:
            normalized = str(token).strip().lower()
            if normalized in self._VIEW_TOKENS:
                enabled.add(normalized)
        self._enabled_views = enabled
        for index, token in enumerate(self._VIEW_TOKENS):
            self._tab_bar.setTabEnabled(index, token in self._enabled_views)

        if self.active_view() not in self._enabled_views:
            fallback = self._last_time_view if self._last_time_view in self._enabled_views else "fb_asymmetry"
            self.set_active_view("frequency" if self.active_domain() == "frequency" else fallback)

    def set_active_view(self, view: str) -> None:
        """Select the visible top-level plot view."""
        token = str(view).strip().lower()
        if token == "time":
            token = self._last_time_view
        if token not in self._VIEW_TOKENS:
            token = "fb_asymmetry"
        if token not in self._enabled_views:
            token = "fb_asymmetry" if token != "frequency" else "frequency"
        target_index = self._VIEW_TOKENS.index(token)
        if self._tab_bar.currentIndex() == target_index:
            self._panel_stack.setCurrentWidget(
                self._frequency_panel if token == "frequency" else self._time_panel
            )
            return
        self._tab_bar.setCurrentIndex(target_index)

    def set_active_domain(self, domain: str) -> None:
        """Select the visible plot domain tab."""
        token = str(domain).strip().lower()
        self.set_active_view("frequency" if token == "frequency" else self._last_time_view)

    def get_state(self) -> dict:
        """Return the serializable workspace state."""
        return {"active_domain": self.active_domain(), "active_view": self.active_view()}

    def current_panel(self) -> QWidget:
        """Return the currently visible plot panel."""
        return self._frequency_panel if self.active_domain() == "frequency" else self._time_panel

    def clear(self) -> None:
        """Clear both tabs and return to the time-domain view."""
        for panel in (self._time_panel, self._frequency_panel):
            if hasattr(panel, "clear"):
                panel.clear()
        self._last_time_view = "fb_asymmetry"
        self.set_available_views(["fb_asymmetry"])
        self.set_active_view("fb_asymmetry")

    def export_current_plot(self) -> None:
        """Export the currently visible plot panel."""
        panel = self.current_panel()
        if hasattr(panel, "export_current_plot"):
            panel.export_current_plot()

    def restore_state(self, state: dict | None) -> None:
        """Restore tab selection from saved state."""
        if not isinstance(state, dict):
            self.set_active_view("fb_asymmetry")
            return
        active_view = state.get("active_view")
        if isinstance(active_view, str):
            self.set_active_view(active_view)
            return
        self.set_active_domain(str(state.get("active_domain", "time")))

    def _on_current_tab_changed(self, _index: int) -> None:
        """Broadcast the newly active plot domain."""
        active_view = self.active_view()
        if active_view != "frequency":
            self._last_time_view = active_view
        self._panel_stack.setCurrentWidget(
            self._frequency_panel if active_view == "frequency" else self._time_panel
        )
        self.active_view_changed.emit(active_view)
        self.active_domain_changed.emit(self.active_domain())
