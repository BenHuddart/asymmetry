"""Central workspace for FB asymmetry, grouped, and frequency plots.

The toolbar's Domain segmented control is the sole source of truth for which
view is active.  This panel owns the QStackedWidget that shows the appropriate
PlotPanel; it holds no tab bar of its own.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget


class PlotWorkspacePanel(QWidget):
    """Own the main-window plot stack and active-domain state."""

    active_domain_changed = Signal(str)
    active_view_changed = Signal(str)

    _VIEW_TOKENS = ("fb_asymmetry", "groups", "reconstruction", "frequency", "maxent")
    #: View tokens that resolve to the frequency-domain plot panel.
    _FREQUENCY_VIEWS = frozenset({"frequency", "maxent"})

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
        self._enabled_views: set[str] = {"fb_asymmetry", "frequency"}
        self._active_view = "fb_asymmetry"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._panel_stack = QStackedWidget()
        self._panel_stack.addWidget(self._time_panel)
        self._panel_stack.addWidget(self._frequency_panel)
        layout.addWidget(self._panel_stack)

        self.set_available_views(["fb_asymmetry"])
        self.set_active_view("fb_asymmetry")

    # ── read-only accessors ───────────────────────────────────────────────────

    def time_panel(self) -> QWidget:
        """Return the time-domain plot panel."""
        return self._time_panel

    def frequency_panel(self) -> QWidget:
        """Return the frequency-domain plot panel."""
        return self._frequency_panel

    def active_view(self) -> str:
        """Return the currently selected top-level plot view token."""
        return self._active_view

    def active_domain(self) -> str:
        """Return the currently selected plot domain."""
        return "frequency" if self._active_view in self._FREQUENCY_VIEWS else "time"

    def is_view_enabled(self, view: str) -> bool:
        """Return whether *view* is currently available for selection."""
        return str(view).strip().lower() in self._enabled_views

    def enabled_views(self) -> frozenset[str]:
        """Return the set of currently available view tokens."""
        return frozenset(self._enabled_views)

    def current_panel(self) -> QWidget:
        """Return the currently visible plot panel."""
        return (
            self._frequency_panel
            if self._active_view in self._FREQUENCY_VIEWS
            else self._time_panel
        )

    # ── mutators ─────────────────────────────────────────────────────────────

    def set_available_views(self, views: list[str]) -> None:
        """Update the set of enabled views.

        ``fb_asymmetry`` and ``frequency`` are always available.  Callers add
        ``groups`` when grouped-time-domain data is ready to plot.  If the
        currently active view becomes unavailable the workspace falls back to
        the last known time view (or ``fb_asymmetry``).
        """
        enabled: set[str] = {"fb_asymmetry", "frequency"}
        for token in views:
            normalized = str(token).strip().lower()
            if normalized in self._VIEW_TOKENS:
                enabled.add(normalized)
        self._enabled_views = enabled

        if self._active_view not in self._enabled_views:
            fallback = (
                self._last_time_view
                if self._last_time_view in self._enabled_views
                else "fb_asymmetry"
            )
            self.set_active_view("frequency" if self.active_domain() == "frequency" else fallback)

    def set_active_view(self, view: str) -> None:
        """Select the visible top-level plot view and emit change signals.

        A call with the already-active view token is a no-op — no signals are
        emitted, preventing spurious re-renders of FFT data or fit panels.
        """
        token = str(view).strip().lower()
        if token == "time":
            token = self._last_time_view
        if token not in self._VIEW_TOKENS:
            token = "fb_asymmetry"
        if token not in self._enabled_views:
            token = "fb_asymmetry" if token != "frequency" else "frequency"

        if token == self._active_view:
            # Ensure the stack is in sync even if no signal is needed.
            self._panel_stack.setCurrentWidget(
                self._frequency_panel if token in self._FREQUENCY_VIEWS else self._time_panel
            )
            return

        prev_domain = self.active_domain()
        self._active_view = token
        if token not in self._FREQUENCY_VIEWS:
            self._last_time_view = token
        self._panel_stack.setCurrentWidget(
            self._frequency_panel if token in self._FREQUENCY_VIEWS else self._time_panel
        )
        self.active_view_changed.emit(token)
        if self.active_domain() != prev_domain:
            self.active_domain_changed.emit(self.active_domain())

    def set_active_domain(self, domain: str) -> None:
        """Select the visible plot domain."""
        token = str(domain).strip().lower()
        self.set_active_view("frequency" if token == "frequency" else self._last_time_view)

    # ── persistence ───────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return the serialisable workspace state."""
        return {"active_domain": self.active_domain(), "active_view": self.active_view()}

    def restore_state(self, state: dict | None) -> None:
        """Restore view selection from saved state."""
        if not isinstance(state, dict):
            self.set_active_view("fb_asymmetry")
            return
        active_view = state.get("active_view")
        if isinstance(active_view, str):
            self.set_active_view(active_view)
            return
        self.set_active_domain(str(state.get("active_domain", "time")))

    # ── actions ───────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear both panels and return to the default time-domain view."""
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
