"""Verify toolbar button styling produced by the centralised helpers.

Phases 11-12 replaced inline hex-colour strings on toolbar buttons with calls
to `build_segmented_button_qss()` and `build_nav_button_qss()`.  These tests
confirm that the generated QSS strings satisfy the key constraints that
prevented text clipping and matched the BENCH design spec.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import build_nav_button_qss, build_segmented_button_qss


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestBuildSegmentedButtonQss:
    def test_base_rule_has_font_weight_600(self) -> None:
        """font-weight: 600 must be in the *base* rule, not only :checked.

        If font-weight is set only in :checked, the button text widens when
        activated, causing clipping inside a fixed-padding container.
        """
        qss = build_segmented_button_qss()
        # Split at :checked to isolate the base QPushButton rule
        base_rule = qss.split("QPushButton:checked")[0]
        assert "font-weight: 600" in base_rule

    def test_checked_rule_also_has_font_weight_600(self) -> None:
        """Both states must declare font-weight so Qt doesn't revert."""
        qss = build_segmented_button_qss()
        checked_rule = qss.split("QPushButton:checked")[1]
        assert "font-weight: 600" in checked_rule

    def test_base_rule_has_explicit_padding(self) -> None:
        """Padding must be in the base rule so Qt pseudo-state cascade works."""
        qss = build_segmented_button_qss()
        base_rule = qss.split("QPushButton:checked")[0]
        assert "padding:" in base_rule

    def test_checked_rule_has_matching_padding(self) -> None:
        """Padding must be repeated in :checked — Qt reverts unset properties."""
        qss = build_segmented_button_qss()
        checked_rule = qss.split("QPushButton:checked")[1]
        assert "padding:" in checked_rule

    def test_default_padding_is_10px_horizontal(self) -> None:
        qss = build_segmented_button_qss()
        assert "padding: 2px 10px" in qss

    def test_min_width_applied_when_specified(self) -> None:
        qss = build_segmented_button_qss(min_width=28)
        assert "min-width: 28px" in qss

    def test_compact_padding_when_padding_h_6(self) -> None:
        qss = build_segmented_button_qss(min_width=28, padding_h=6)
        assert "padding: 2px 6px" in qss

    def test_no_raw_hex_in_output(self) -> None:
        """QSS should reference token values, not introduce new hex literals."""
        qss = build_segmented_button_qss()
        # The values in the QSS must come from tokens — spot-check a couple
        assert tokens.ACCENT_SOFT in qss  # checked background
        assert tokens.ACCENT in qss  # checked border/text
        assert tokens.SURFACE in qss  # unchecked background
        assert tokens.BORDER in qss  # unchecked border

    def test_no_default_font_weight_in_base_only_checked(self) -> None:
        """Guard against accidentally moving font-weight back to :checked only."""
        qss = build_segmented_button_qss()
        # Count occurrences — both base and checked must declare it
        assert qss.count("font-weight: 600") >= 2


class TestBuildNavButtonQss:
    def test_base_rule_has_min_width_60(self) -> None:
        qss = build_nav_button_qss()
        base_rule = qss.split("QPushButton:checked")[0]
        assert "min-width: 60px" in base_rule

    def test_checked_rule_repeats_min_width(self) -> None:
        """min-width must be repeated so Qt doesn't revert to default."""
        qss = build_nav_button_qss()
        checked_rule = qss.split("QPushButton:checked")[1]
        assert "min-width: 60px" in checked_rule

    def test_base_rule_has_explicit_padding(self) -> None:
        qss = build_nav_button_qss()
        base_rule = qss.split("QPushButton:checked")[0]
        assert "padding:" in base_rule

    def test_checked_rule_has_matching_padding(self) -> None:
        qss = build_nav_button_qss()
        checked_rule = qss.split("QPushButton:checked")[1]
        assert "padding:" in checked_rule

    def test_base_rule_has_font_weight_600(self) -> None:
        qss = build_nav_button_qss()
        base_rule = qss.split("QPushButton:checked")[0]
        assert "font-weight: 600" in base_rule

    def test_no_raw_hex_in_output(self) -> None:
        qss = build_nav_button_qss()
        assert tokens.ACCENT_SOFT in qss
        assert tokens.ACCENT in qss
        assert tokens.BORDER_STRONG in qss


class TestDomainButtonsOnMainWindow:
    """Integration: the actual toolbar buttons carry the expected QSS."""

    @pytest.fixture
    def mainwindow(self, qapp):
        from PySide6.QtCore import QSettings

        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

        s = QSettings()
        s.setValue(UI_SCALE_SETTINGS_KEY, 1.0)
        return MainWindow()

    def test_domain_buttons_use_centralised_qss(self, mainwindow) -> None:
        """Domain buttons must have a non-empty per-widget stylesheet."""
        buttons = mainwindow._domain_buttons
        assert buttons, "No domain buttons found"
        for btn in buttons:
            ss = btn.styleSheet()
            assert ss, f"Domain button '{btn.text()}' has empty styleSheet"
            assert "font-weight: 600" in ss
            assert "padding:" in ss

    def test_view_mode_buttons_have_compact_padding(self, mainwindow) -> None:
        """View-mode buttons must use 6px horizontal padding (not 10px)."""
        buttons = mainwindow._view_mode_buttons
        assert buttons, "No view-mode buttons found"
        for btn in buttons:
            ss = btn.styleSheet()
            assert "padding: 2px 6px" in ss, (
                f"View-mode button '{btn.text()}' missing compact padding"
            )

    def test_view_mode_buttons_have_min_width(self, mainwindow) -> None:
        for btn in mainwindow._view_mode_buttons:
            assert "min-width: 28px" in btn.styleSheet()

    def test_domain_buttons_checked_state_uses_accent(self, mainwindow) -> None:
        for btn in mainwindow._domain_buttons:
            ss = btn.styleSheet()
            assert tokens.ACCENT in ss
            assert tokens.ACCENT_SOFT in ss

    def test_checkbox_indicators_are_explicitly_styled(self, mainwindow) -> None:
        """Checkbox/item-view indicators must be explicitly styled.

        A global QSS suppresses the native indicator on Windows, leaving an
        unstyled box. Both QCheckBox widgets and item-view check states must be
        covered, with an accent fill + checkmark image when checked.
        """
        qss = mainwindow._ui_manager.build_stylesheet(1.0)
        assert "QCheckBox::indicator" in qss
        assert "QAbstractItemView::indicator" in qss
        # The checked indicator rule body uses the accent fill + checkmark image.
        checked_body = qss.split("indicator:checked {")[1].split("}")[0]
        assert tokens.ACCENT in checked_body
        assert "checkmark.svg" in checked_body


class TestSegmentedButtonDisabledState:
    def test_disabled_rule_visibly_dims_the_button(self) -> None:
        """Data-gated segments must read as greyed out, not clickable.

        Per-widget stylesheets replace the global sheet entirely, so the
        disabled state has to be declared here — without it a disabled
        'Individual groups' / 'MaxEnt' button looked identical to an enabled
        one."""
        from asymmetry.gui.styles import tokens

        qss = build_segmented_button_qss()
        assert "QPushButton:disabled" in qss
        disabled_rule = qss.split("QPushButton:disabled")[1]
        assert tokens.TEXT_DIM in disabled_rule
        assert tokens.SURFACE_ALT in disabled_rule
        # Padding and font-weight repeated so Qt's pseudo-state cascade
        # doesn't reflow the segment when it is enabled/disabled.
        assert "padding:" in disabled_rule
        assert "font-weight: 600" in disabled_rule
