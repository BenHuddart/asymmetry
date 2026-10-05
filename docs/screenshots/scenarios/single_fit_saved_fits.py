"""Three saved single fits of one ZF Ag run: the Saved fits row and the Compare window.

The ZF Ag polycrystal (static Gaussian Kubo–Toyabe, Δ ≈ 0.39 μs⁻¹; Kubo &
Toyabe 1966, Blundell et al. Ch 5.2) is fitted three times on the Single tab
with the real engine: ``StaticGKT_ZF + Constant`` and ``Gaussian + Constant``
over 0.05–8 μs, then the Kubo–Toyabe again over 0.5–6 μs. The two full-window
fits saw the same points, so the Compare window ranks them; the narrow one saw
different data and is listed without a Δ (docs/plans/single-fit-compare.md
D7). A Gaussian cannot recover to 1/3 after the Kubo–Toyabe dip near
t = √3/Δ ≈ 4.4 μs, so its residuals against the data carry that structure.

The Saved fits row and the Compare window are two windows, so the figure is
composed: a grab of the Single tab on the left and the Compare window,
reparented as a plain child, on the right — the same window the user sees,
drawn in place.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, _process_events_for, register


class SingleFitSavedFitsScenario(Scenario):
    name = "single_fit_saved_fits"
    description = "Three saved single fits of one Ag run, ranked in the Compare saved fits window."
    size = (1500, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resize(1500, 920)
        window._on_fit()
        dataset = make_ag_zf_gkt()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=80)

        single_tab = window._fit_panel._single_tab
        kubo_toyabe = CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"])
        gaussian = CompositeModel(["Gaussian", "Constant"], operators=["+"])
        for model, (t_min, t_max) in (
            (gaussian, (0.05, 8.0)),
            (kubo_toyabe, (0.5, 6.0)),
            (kubo_toyabe, (0.05, 8.0)),
        ):
            # A programmatic setValue commits the range to the plot, which owns it.
            single_tab._fit_range_min_spin.setValue(t_min)
            single_tab._fit_range_max_spin.setValue(t_max)
            single_tab._set_composite_model(model)
            _process_events_for(milliseconds=40)
            single_tab._run_fit()
            single_tab.wait_for_fit()
            _process_events_for(milliseconds=40)

        representation = window._project_model.representation(
            dataset.run_number, window._active_representation_type()
        )
        fits = representation.fit_set(None).fits
        assert len(fits) == 3, f"expected three saved fits, found {len(fits)}"
        gaussian_fit = fits[0]

        single_tab._compare_fits_btn.click()
        compare = window._saved_fit_compare_window
        compare.panel.set_b(gaussian_fit.fit_id)
        for _ in range(100):
            _process_events_for(milliseconds=30)
            if len(compare._curves) >= 2:
                break

        container = QWidget()
        container.setStyleSheet(window.styleSheet())
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        tab_shot = QLabel()
        tab_shot.setPixmap(single_tab.grab())
        tab_shot.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(tab_shot, 0)
        compare.setParent(container, Qt.WindowType.Widget)
        layout.addWidget(compare, 1)
        self._main_window = window
        return container

    def teardown(self, widget: QWidget) -> None:
        self._main_window._dirty = False
        self._main_window.close()
        self._main_window.deleteLater()
        super().teardown(widget)


register(SingleFitSavedFitsScenario())
