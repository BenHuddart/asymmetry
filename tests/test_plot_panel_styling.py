"""Verify that the BENCH plot-styling helpers are wired into PlotPanel.

Phase 13 introduced `styles/plots.py` with `style_axes`, `style_figure`,
`style_legend`, and `draw_fit_range_span`, and applied them at every
axes-creation / clear site in `plot_panel.py`.  These tests check that the
styling is actually applied and that the new header / footer widgets exist.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

from asymmetry.gui.styles import tokens


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    from asymmetry.gui.panels.plot_panel import PlotPanel

    return PlotPanel()


@pytest.fixture
def dataset():
    from asymmetry.core.data.dataset import MuonDataset

    t = np.linspace(0.0, 10.0, 200)
    a = 0.2 * np.exp(-0.4 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 99})


class TestAxesStyling:
    """style_axes() must have been called after every plot_dataset / clear."""

    def test_axes_facecolor_is_bench_surface_after_plot(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        color = panel._ax.get_facecolor()
        # Surface is "#ffffff" → (1.0, 1.0, 1.0, 1.0) in RGBA
        assert color[0] == pytest.approx(1.0)
        assert color[1] == pytest.approx(1.0)
        assert color[2] == pytest.approx(1.0)

    def test_figure_facecolor_is_bench_surface(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        color = panel._figure.get_facecolor()
        assert color[0] == pytest.approx(1.0)
        assert color[1] == pytest.approx(1.0)
        assert color[2] == pytest.approx(1.0)

    def test_spines_use_open_frame_grammar(self, panel, dataset) -> None:
        """Design handoff: left/bottom spines only, in BENCH PLOT_AXIS."""
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        import matplotlib.colors as mcolors

        expected = mcolors.to_rgba(tokens.PLOT_AXIS)
        for side in ("left", "bottom"):
            spine = panel._ax.spines[side]
            assert spine.get_visible()
            assert spine.get_edgecolor() == pytest.approx(expected, abs=0.01)
        for side in ("top", "right"):
            assert not panel._ax.spines[side].get_visible()

    def test_axes_styled_after_clear(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        panel.clear()
        color = panel._ax.get_facecolor()
        assert color[0] == pytest.approx(1.0)

    def test_grid_is_enabled(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        # At least one grid line should exist after style_axes()
        has_grid = any(
            line.get_visible() for line in panel._ax.get_xgridlines() + panel._ax.get_ygridlines()
        )
        assert has_grid, "Grid should be enabled after style_axes()"


class TestFitRangeSpan:
    """draw_fit_range_span() must use BENCH accent colours, not gold/darkorange."""

    def test_fit_range_span_uses_accent_color(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        panel.set_fit_range(1.0, 5.0)

        spans = panel._fit_span_artists
        assert spans, "Expected at least one fit-range span artist"

        import matplotlib.colors as mcolors

        expected_rgb = mcolors.to_rgb(tokens.PLOT_FIT_RANGE_FACE)
        # get_facecolor() returns RGBA array or tuple depending on mpl version
        raw = spans[0].get_facecolor()
        actual_rgba = mcolors.to_rgba(
            raw[0]
            if hasattr(raw, "__len__") and len(raw) > 1 and hasattr(raw[0], "__len__")
            else raw
        )
        assert actual_rgba[:3] == pytest.approx(expected_rgb, abs=0.02), (
            f"Fit span colour {actual_rgba[:3]} does not match BENCH accent {expected_rgb}"
        )

    def test_fit_range_handles_use_accent_color(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        panel.set_fit_range(1.0, 5.0)

        handles = panel._fit_min_handles
        assert handles, "Expected at least one fit-range handle artist"
        import matplotlib.colors as mcolors

        expected = mcolors.to_rgb(tokens.PLOT_FIT_RANGE_EDGE)
        actual_rgba = mcolors.to_rgba(handles[0].get_color())
        assert actual_rgba[:3] == pytest.approx(expected, abs=0.02)


class TestPlotHeader:
    """The title-strip header widget must exist and update correctly."""

    def test_header_widget_exists(self, panel) -> None:
        assert hasattr(panel, "_plot_header"), "_plot_header widget missing"

    def test_header_title_label_exists(self, panel) -> None:
        assert hasattr(panel, "_header_title_label"), "_header_title_label missing"

    def test_header_meta_label_exists(self, panel) -> None:
        assert hasattr(panel, "_header_meta_label"), "_header_meta_label missing"

    def test_header_title_shows_domain_name_when_empty(self, panel) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        text = panel._header_title_label.text()
        assert "asymmetry" in text.lower() or "spectrum" in text.lower() or text == "", (
            f"Unexpected header title when no dataset: '{text}'"
        )

    def test_header_title_shows_run_label_after_plot(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        text = panel._header_title_label.text()
        assert "asymmetry" in text.lower(), (
            f"Header title should mention domain after plot_dataset: '{text}'"
        )

    def test_header_title_clears_on_panel_clear(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        panel.clear()
        text = panel._header_title_label.text()
        # After clear(), title should revert to just the domain name (no run #)
        assert "—" not in text, f"Header should show no run after clear(), got: '{text}'"

    def test_header_meta_label_shows_alpha_after_plot(self, panel) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        from asymmetry.core.data.dataset import MuonDataset, Run

        t = np.linspace(0, 10, 100)
        a = 0.2 * np.exp(-0.4 * t)
        e = np.full_like(t, 0.01)
        ds = MuonDataset(
            time=t,
            asymmetry=a,
            error=e,
            metadata={"run_number": 42},
            run=Run(run_number=42, grouping={"alpha": 1.5}),
        )
        panel.plot_dataset(ds)
        assert "1.5" in panel._header_meta_label.text()

    def test_header_meta_label_empty_for_multi_dataset(self, panel) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        from asymmetry.core.data.dataset import MuonDataset

        t = np.linspace(0, 10, 50)
        a = 0.1 * np.exp(-0.3 * t)
        e = np.full_like(t, 0.01)
        ds1 = MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 1})
        ds2 = MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 2})
        panel.plot_datasets([ds1, ds2])
        # Multi-dataset view: no per-run alpha in header
        assert panel._header_meta_label.text() == ""

    def test_header_title_multi_dataset_shows_count(self, panel) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        from asymmetry.core.data.dataset import MuonDataset

        t = np.linspace(0, 10, 50)
        a = 0.1 * np.exp(-0.3 * t)
        e = np.full_like(t, 0.01)
        datasets = [
            MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": i}) for i in range(3)
        ]
        panel.plot_datasets(datasets)
        assert "3" in panel._header_title_label.text()


class TestPlotFooter:
    """The footer bar must contain the annotation and export controls."""

    def test_footer_widget_exists(self, panel) -> None:
        assert hasattr(panel, "_plot_footer"), "_plot_footer widget missing"

    def test_add_label_btn_exists_in_footer(self, panel) -> None:
        assert hasattr(panel, "_add_label_btn"), "_add_label_btn missing"
        # The button must be a child of the footer widget, not the limit toolbar
        footer = panel._plot_footer
        from PySide6.QtWidgets import QPushButton

        btns = footer.findChildren(QPushButton)
        btn_texts = [b.text() for b in btns]
        assert "Add Annotation" in btn_texts, (
            f"'Add Annotation' not found in footer; footer buttons: {btn_texts}"
        )

    def test_export_btn_exists_in_footer(self, panel) -> None:
        assert hasattr(panel, "_export_gle_btn"), "_export_gle_btn missing"
        footer = panel._plot_footer
        from PySide6.QtWidgets import QPushButton

        btns = footer.findChildren(QPushButton)
        btn_texts = [b.text() for b in btns]
        assert "Export Plot(s) to GLE" in btn_texts

    def test_format_combo_exists_in_footer(self, panel) -> None:
        assert hasattr(panel, "_gle_format_combo"), "_gle_format_combo missing"
        footer = panel._plot_footer
        from PySide6.QtWidgets import QComboBox

        combos = footer.findChildren(QComboBox)
        assert combos, "No QComboBox found in footer"
        texts = [combos[0].itemText(i) for i in range(combos[0].count())]
        assert "PDF" in texts
        assert "EPS" in texts

    def test_add_annotation_btn_is_checkable(self, panel) -> None:
        assert panel._add_label_btn.isCheckable()

    def test_add_label_not_in_limit_toolbar_row2(self, panel) -> None:
        """Ensure the button is NOT in the old limit-toolbar location."""
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        # The limit_toolbar QVBoxLayout must not contain a button with this text
        from PySide6.QtWidgets import QPushButton

        # Walk the limit_toolbar layout tree
        def _collect_buttons(layout) -> list[str]:
            found = []
            if layout is None:
                return found
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QPushButton):
                    found.append(item.widget().text())
                elif item.layout():
                    found.extend(_collect_buttons(item.layout()))
            return found

        limit_btns = _collect_buttons(panel._limit_toolbar)
        assert "Add Annotation" not in limit_btns, (
            "'Add Annotation' is still in the limit toolbar — it should be in the footer"
        )


class TestHandoffPlotGrammar:
    """Axis labels, legend text, and the zero reference line (design handoff)."""

    def test_axis_labels_use_muted_label_grey(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        import matplotlib.colors as mcolors

        expected = mcolors.to_rgba(tokens.PLOT_TICK_LABEL)
        for label in (panel._ax.xaxis.label, panel._ax.yaxis.label):
            assert mcolors.to_rgba(label.get_color()) == pytest.approx(expected, abs=0.01)
            assert label.get_fontsize() == pytest.approx(10.0)

    def test_zero_reference_line_drawn_under_data(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        import matplotlib.colors as mcolors

        expected = mcolors.to_rgba(tokens.PLOT_ZERO_LINE)
        zero_lines = [
            line
            for line in panel._ax.get_lines()
            if np.allclose(mcolors.to_rgba(line.get_color()), expected)
            and np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)
        ]
        assert zero_lines, "No y = 0 reference line found on the time plot"

    def test_legend_entries_are_monospaced(self, panel, dataset) -> None:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(dataset)
        legend = panel._ax.get_legend()
        assert legend is not None
        texts = legend.get_texts()
        assert texts
        for text in texts:
            assert "IBM Plex Mono" in str(text.get_fontfamily())
            assert text.get_fontsize() == pytest.approx(9.0)


def _offset_dataset():
    from asymmetry.core.data.dataset import MuonDataset

    t = np.linspace(0.0, 10.0, 100)
    counts = 1000.0 + 200.0 * np.sin(t)
    return MuonDataset(
        time=t,
        asymmetry=counts,
        error=np.full_like(t, 5.0),
        metadata={"run_number": 7},
    )


class TestZeroLineAutoscale:
    def test_zero_line_does_not_anchor_positive_data_to_zero(self, panel) -> None:
        """Regression: axhline registered y=0 in the data limits, so grouped
        counts (~N0, far from zero) autoscaled to include 0 and squashed the
        signal. The reference line must stay out of autoscale."""
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        panel.plot_dataset(_offset_dataset())
        y_min, _y_max = panel._ax.get_ylim()
        assert y_min > 400.0, f"y-axis anchored toward zero: ylim starts at {y_min}"
