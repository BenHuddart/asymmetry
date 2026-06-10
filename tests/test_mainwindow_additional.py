"""Additional tests for mainwindow functionality."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings, Qt  # type: ignore
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar, QWidget  # type: ignore

import asymmetry.core.fourier.spectrum as spectrum_module
import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting import CompositeModel
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.project import load_project, save_project
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.styles import tokens
from tests._qt_helpers import wait_for


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    """Create a mainwindow for testing."""
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _make_dataset(run_number: int, *, with_grouping: bool) -> MuonDataset:
    counts = np.array([100.0, 95.0, 90.0, 85.0], dtype=float)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts, bin_width=0.01),
            Histogram(counts=counts * 0.8, bin_width=0.01),
        ],
        metadata={"run_number": run_number, "field": 100.0},
        grouping=(
            {
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 3,
                "bunching_factor": 1,
                "deadtime_correction": False,
            }
            if with_grouping
            else {}
        ),
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "field": 100.0},
        run=run,
    )


def _make_fourier_ready_dataset(run_number: int, *, with_grouping: bool) -> MuonDataset:
    time = np.arange(256, dtype=float) * 0.05
    frequency = 12.0 / (time.size * 0.05)
    phase_a = np.deg2rad(32.0)
    phase_b = np.deg2rad(-18.0)
    counts_a = 1000.0 + 150.0 * np.cos(2.0 * np.pi * frequency * time + phase_a)
    counts_b = 900.0 + 120.0 * np.cos(2.0 * np.pi * frequency * time + phase_b)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_a, bin_width=0.05),
            Histogram(counts=counts_b, bin_width=0.05),
        ],
        metadata={"run_number": run_number, "field": 100.0},
        grouping=(
            {
                "groups": {1: [1], 2: [2]},
                "group_names": {1: "Left", 2: "Right"},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": 255,
                "bunching_factor": 1,
                "deadtime_correction": False,
            }
            if with_grouping
            else {}
        ),
    )
    asymmetry = 0.2 * np.cos(2.0 * np.pi * frequency * time + phase_a)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number, "field": 100.0},
        run=run,
    )


def _make_two_period_vector_dataset(run_number: int) -> MuonDataset:
    time = np.arange(4, dtype=float) * 0.01

    def _hist(values: list[float]) -> Histogram:
        return Histogram(counts=np.asarray(values, dtype=float), bin_width=0.01, t0_bin=0)

    red_histograms = [
        _hist([100.0, 100.0, 100.0, 100.0]),
        _hist([50.0, 50.0, 50.0, 50.0]),
        _hist([80.0, 80.0, 80.0, 80.0]),
        _hist([20.0, 20.0, 20.0, 20.0]),
        _hist([70.0, 70.0, 70.0, 70.0]),
        _hist([30.0, 30.0, 30.0, 30.0]),
    ]
    green_histograms = [
        _hist([60.0, 60.0, 60.0, 60.0]),
        _hist([90.0, 90.0, 90.0, 90.0]),
        _hist([40.0, 40.0, 40.0, 40.0]),
        _hist([80.0, 80.0, 80.0, 80.0]),
        _hist([20.0, 20.0, 20.0, 20.0]),
        _hist([100.0, 100.0, 100.0, 100.0]),
    ]

    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(
                counts=np.asarray(hist.counts, dtype=float).copy(),
                bin_width=float(hist.bin_width),
                t0_bin=int(hist.t0_bin),
                good_bin_start=int(hist.good_bin_start),
                good_bin_end=int(hist.good_bin_end),
            )
            for hist in red_histograms
        ],
        metadata={"run_number": run_number, "field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]},
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "vector_axis": "P_z",
            "alpha": 1.0,
            "alpha_x": 1.0,
            "alpha_y": 1.0,
            "alpha_z": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "period_histograms": [red_histograms, green_histograms],
            "period_mode": str(mw_module.PeriodMode.RED),
            "period_good_frames": [1.0, 1.0],
            "period_dead_time_us": [[], []],
        },
    )
    return MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.ones_like(time),
        metadata={"run_number": run_number, "field": 100.0},
        run=run,
    )


def _make_deadtime_dataset(run_number: int, *, good_frames: float) -> MuonDataset:
    """Two-detector dataset wired for File-mode deadtime correction.

    Counts and deadtime are chosen so the correction stays a tiny (<1%) bump
    when normalised by the *correct* ``good_frames`` but makes the
    ``1 - N*tau/(dt*good_frames)`` denominator clip — and the asymmetry saturate
    at +-100% — if ``good_frames`` is replaced by a much smaller value or lost.
    """
    counts_f = np.full(8, 20000.0)
    counts_b = np.full(8, 5000.0)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_f, bin_width=0.01, t0_bin=0),
            Histogram(counts=counts_b, bin_width=0.01, t0_bin=0),
        ],
        metadata={"run_number": run_number, "field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 7,
            "bunching_factor": 1,
            "deadtime_correction": True,
            "deadtime_method": "file",
            "deadtime_mode": "file",
            "dead_time_us": [0.01, 0.01],
            "good_frames": float(good_frames),
        },
    )
    t = np.arange(8, dtype=float) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "field": 100.0},
        run=run,
    )


class TestMainWindowFourier:
    def test_fill_fourier_phases_populates_group_phase_table(self, mainwindow: MainWindow) -> None:
        dataset = _make_fourier_ready_dataset(8801, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8801)
        mainwindow._fourier_panel._phase_mode_radio.setChecked(True)

        mainwindow._on_fill_fourier_phases()

        phases = mainwindow._fourier_panel.group_phase_table()
        assert mainwindow._fourier_panel._use_phase_table_check.isChecked() is True
        assert set(phases) == {1, 2}
        assert (
            mainwindow._fourier_panel._phase_table.item(0, 2).foreground().color().name()
            == tokens.OK
        )

    def test_maxent_domain_button_enabled_after_switch_from_fft(
        self, mainwindow: MainWindow
    ) -> None:
        # FFT and MaxEnt share the frequency domain, so switching between them
        # fires active_view_changed but NOT active_domain_changed -- and
        # _refresh_time_view_selector (which used to be the only place that set
        # the button's enabled state) runs on selection/domain change only. A
        # pure FFT -> MaxEnt view change must still re-evaluate the button so it
        # is not left stale-disabled for a maxent-capable dataset.
        dataset = _make_fourier_ready_dataset(8809, with_grouping=True)
        assert mainwindow._dataset_supports_maxent(dataset) is True
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8809)
        mainwindow._refresh_time_view_selector()
        # Move into the FFT view and simulate the stale-disabled MaxEnt button.
        mainwindow._plot_workspace.set_active_view("frequency")
        mainwindow._domain_buttons[3].setEnabled(False)
        # The exact handler wired to active_view_changed for an FFT -> MaxEnt
        # toolbar switch -- no reselection in the F-B view.
        mainwindow._on_plot_workspace_view_changed("maxent")
        assert mainwindow._domain_buttons[3].isEnabled() is True

    def test_maxent_domain_button_disabled_for_unsupported_after_view_change(
        self, mainwindow: MainWindow
    ) -> None:
        # The same re-evaluation must turn the button OFF when the active dataset
        # cannot produce MaxEnt spectra (no raw grouped histograms).
        dataset = _make_dataset(8811, with_grouping=False)
        assert mainwindow._dataset_supports_maxent(dataset) is False
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8811)
        mainwindow._plot_workspace.set_active_view("frequency")
        mainwindow._domain_buttons[3].setEnabled(True)  # stale-enabled
        mainwindow._on_plot_workspace_view_changed("frequency")
        assert mainwindow._domain_buttons[3].isEnabled() is False

    def test_maxent_divergence_surfaces_visible_warning(self, mainwindow: MainWindow) -> None:
        """A diverged MaxEnt result must raise a visible, warning-coloured status
        (not just the small diagnostics line); the engine keeps the last stable
        spectrum so the plot is not diverged junk."""
        import dataclasses

        from asymmetry.core.maxent import MaxEntConfig, maxent

        dataset = _make_fourier_ready_dataset(8821, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8821)
        config = MaxEntConfig(
            n_spectrum_points=128,
            f_min_mhz=0.2,
            f_max_mhz=3.0,
            auto_window=False,
            inner_iterations=4,
            fit_phases=False,
        )
        # Flag the (real, valid) result as diverged to drive the GUI branch.
        result = maxent(dataset.run, config, cycles=12)
        diverged = dataclasses.replace(
            result, diverged=True, converged=False, stop_reason="diverged"
        )
        mainwindow._maxent_active_run_number = 8821
        mainwindow._maxent_active_run = dataset.run
        mainwindow._maxent_active_config = config
        mainwindow._maxent_active_cycles = 12

        mainwindow._on_maxent_worker_finished(diverged)

        status = mainwindow._maxent_panel._status_label.text()
        assert "diverged" in status.lower()
        assert tokens.WARN in status  # rendered in the warning colour

    def test_compute_fourier_plots_frequency_domain_dataset(self, mainwindow: MainWindow) -> None:
        dataset = _make_fourier_ready_dataset(8802, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8802)
        mainwindow._fourier_panel._power_sqrt_radio.setChecked(True)

        mainwindow._on_compute_fourier()

        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert mainwindow._plot_workspace.active_domain() == "frequency"
        assert plotted.metadata["plot_domain"] == "frequency"
        assert plotted.metadata["x_label"] == "Frequency (MHz)"
        assert plotted.metadata["y_label"] == "FFT (Power)^1/2 (a.u.)"
        assert plotted.metadata["fourier_group_output"] == "average"
        assert plotted.metadata["group_ids"] == [1, 2]

    def test_group_phase_estimation_uses_field_centered_window(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dt = 0.0000244140625
        time = np.arange(8192, dtype=float) * dt
        field_frequency = 25000.0 * 135.538817 * 1.0e-4
        shared_low_frequency = 0.08
        phase_a = 25.0
        phase_b = 110.0

        def _counts(phase_degrees: float) -> np.ndarray:
            envelope = np.exp(-time / 2.1969811)
            low = 600.0 * np.cos(2.0 * np.pi * shared_low_frequency * time - 0.45)
            high = 120.0 * np.cos(2.0 * np.pi * field_frequency * time + np.deg2rad(phase_degrees))
            return 1500.0 + envelope * (low + high)

        run = Run(
            run_number=8812,
            histograms=[
                Histogram(counts=_counts(phase_a), bin_width=dt, t0_bin=0),
                Histogram(counts=_counts(phase_b), bin_width=dt, t0_bin=0),
            ],
            metadata={"run_number": 8812, "field": 25000.0},
            grouping={
                "groups": {1: [1], 2: [2]},
                "group_names": {1: "A", 2: "B"},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": 0,
                "last_good_bin": int(time.size - 1),
                "bunching_factor": 1,
                "deadtime_correction": False,
            },
        )
        dataset = MuonDataset(
            time=time,
            asymmetry=np.zeros_like(time),
            error=np.full_like(time, 0.01),
            metadata={"run_number": 8812, "field": 25000.0},
            run=run,
        )

        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8812)

        phases = mainwindow._estimate_group_fourier_phases(
            dataset,
            {
                "window": "none",
                "padding": 8,
                "t0_offset_us": 0.0,
                "auto_phase_method": "Peak",
            },
        )

        def _angle_distance(actual: float, expected: float) -> float:
            return float(abs(np.angle(np.exp(1j * np.deg2rad(actual - expected)), deg=True)))

        assert _angle_distance(phases[1], phase_a) < 15.0
        assert _angle_distance(phases[2], phase_b) < 15.0
        assert _angle_distance(phases[1], phases[2]) > 45.0

    def test_compute_group_fourier_always_plots_average_selected_groups(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8804, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8804)
        mainwindow._fourier_panel._use_phase_table_check.setChecked(True)
        mainwindow._fourier_panel.set_group_definitions(
            {1: "Left", 2: "Right"}, {1: 32.0, 2: -18.0}
        )

        mainwindow._on_compute_fourier()

        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert mainwindow._plot_workspace.active_domain() == "frequency"
        assert plotted.metadata["fourier_group_output"] == "average"
        assert plotted.metadata["run_label"] == "8804 Average"
        assert plotted.metadata["group_ids"] == [1, 2]

    def test_compute_group_fourier_honours_enabled_groups(self, mainwindow: MainWindow) -> None:
        dataset = _make_fourier_ready_dataset(8805, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8805)
        mainwindow._fourier_panel.set_group_definitions(
            {1: "Left", 2: "Right"},
            {1: 32.0, 2: -18.0},
            {1: True, 2: False},
        )

        mainwindow._on_compute_fourier()

        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert plotted.metadata["run_label"] == "8805 Average (Left)"
        assert plotted.metadata["group_ids"] == [1]

    def test_fft_recipe_round_trip_recomputes_identical_spectrum(
        self, mainwindow: MainWindow
    ) -> None:
        """Recipe-only persistence: a saved FFT recomputes to the same spectrum."""
        dataset = _make_fourier_ready_dataset(8820, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8820)
        mainwindow._on_compute_fourier()
        generated = mainwindow._frequency_spectra_by_run[8820][0]

        state = mainwindow.collect_project_state()
        entry = next(d for d in state["datasets"] if d["run_number"] == 8820)
        assert "freq_fft" in entry["representations"]
        assert "fourier_config" in entry["representations"]["freq_fft"]["recipe"]

        # Simulate reload: drop the in-memory cache and recompute from recipe.
        mainwindow._frequency_spectra_by_run = {}
        mainwindow._restore_frequency_representations(state)
        recomputed = mainwindow._frequency_spectra_by_run[8820][0]

        np.testing.assert_allclose(recomputed.time, generated.time)
        np.testing.assert_allclose(recomputed.asymmetry, generated.asymmetry)
        np.testing.assert_allclose(recomputed.error, generated.error)

    def test_apply_fourier_settings_to_selected_runs(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'Apply to selection' copies the FFT recipe to other selected runs."""
        ds1 = _make_fourier_ready_dataset(8830, with_grouping=True)
        ds2 = _make_fourier_ready_dataset(8831, with_grouping=True)
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        mainwindow._on_dataset_selected(8830)
        mainwindow._on_compute_fourier()

        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [ds1, ds2])
        mainwindow._on_apply_fourier_to_selection()

        # The second run now has a generated spectrum and the same recipe.
        assert 8831 in mainwindow._frequency_spectra_by_run
        source_rep = mainwindow._project_model.representation(8830, RepresentationType.FREQ_FFT)
        target_rep = mainwindow._project_model.representation(8831, RepresentationType.FREQ_FFT)
        assert target_rep is not None
        assert target_rep.recipe["fourier_config"] == source_rep.recipe["fourier_config"]

    def test_apply_fourier_to_selection_requires_prior_compute(
        self, mainwindow: MainWindow
    ) -> None:
        dataset = _make_fourier_ready_dataset(8832, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8832)
        # No FFT computed yet -> nothing to apply.
        mainwindow._on_apply_fourier_to_selection()
        assert 8832 not in mainwindow._frequency_spectra_by_run

    def test_fourier_inclusion_seeded_from_grouping_default(self, mainwindow: MainWindow) -> None:
        # A group flagged excluded in the grouping (e.g. a HAL-9500 MV veto)
        # should start unchecked in the FFT panel and be left out of the average.
        dataset = _make_fourier_ready_dataset(8807, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping["included_groups"] = {1: True, 2: False}
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8807)

        enabled = mainwindow._fourier_panel.group_enabled_table()
        assert enabled.get(1) is True
        assert enabled.get(2) is False

        mainwindow._on_compute_fourier()
        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert plotted.metadata["group_ids"] == [1]

    def test_compute_group_fourier_can_average_selected_groups(
        self, mainwindow: MainWindow
    ) -> None:
        dataset = _make_fourier_ready_dataset(8806, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8806)
        mainwindow._fourier_panel._estimate_average_error_check.setChecked(True)
        mainwindow._fourier_panel._use_phase_table_check.setChecked(True)
        mainwindow._fourier_panel.set_group_definitions(
            {1: "Left", 2: "Right"},
            {1: 32.0, 2: -18.0},
            {1: True, 2: False},
        )

        mainwindow._on_compute_fourier()

        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert plotted.metadata["fourier_group_output"] == "average"
        assert plotted.metadata["group_ids"] == [1]
        assert plotted.metadata["run_label"] == "8806 Average (Left)"
        assert np.allclose(plotted.error, np.zeros_like(plotted.error))

    def test_compute_group_fourier_uses_active_fit_range(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8815, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8815)
        mainwindow._plot_panel._set_fit_range(1.0, 4.0, emit_signal=False, redraw=False)

        calls: list[tuple[float | None, float | None]] = []

        def _fake_fft_complex_asymmetry(
            _dataset: MuonDataset, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            calls.append((kwargs.get("t_min"), kwargs.get("t_max")))
            return np.array([0.0, 1.0]), np.array([0.0 + 0.0j, 1.0 + 0.0j])

        # The averaged-FFT maths now lives in the shared spectrum core; patch
        # the symbol where it is actually called.
        monkeypatch.setattr(spectrum_module, "fft_complex_asymmetry", _fake_fft_complex_asymmetry)

        mainwindow._on_compute_fourier()

        assert calls == [(1.0, 4.0), (1.0, 4.0)]

    def test_compute_group_fourier_reuses_precomputed_group_inputs(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8817, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8817)

        build_calls: list[dict[str, object]] = []

        def _fake_build_group_signal_dataset(
            run: Run,
            group_id: int,
            **kwargs: object,
        ) -> MuonDataset:
            build_calls.append(dict(kwargs))
            return MuonDataset(
                time=np.array([0.0, 1.0]),
                asymmetry=np.array([1.0, 0.5]),
                error=np.array([0.1, 0.1]),
                metadata={
                    "run_number": run.run_number,
                    "run_label": f"{run.run_number} G{group_id}",
                },
                run=run,
            )

        def _fake_fft_complex_asymmetry(
            _dataset: MuonDataset,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.array([0.0, 1.0]), np.array([1.0 + 0.0j, 0.5 + 0.0j])

        monkeypatch.setattr(
            spectrum_module, "build_group_signal_dataset", _fake_build_group_signal_dataset
        )
        monkeypatch.setattr(spectrum_module, "fft_complex_asymmetry", _fake_fft_complex_asymmetry)

        mainwindow._on_compute_fourier()

        assert len(build_calls) == 2
        first_prepared = build_calls[0].get("prepared_histograms")
        assert first_prepared is not None
        assert build_calls[1].get("prepared_histograms") is first_prepared
        first_t0 = build_calls[0].get("reference_t0_bin")
        assert first_t0 is not None
        assert build_calls[1].get("reference_t0_bin") == first_t0

    def test_compute_group_fourier_average_can_estimate_errors(
        self, mainwindow: MainWindow
    ) -> None:
        dataset = _make_fourier_ready_dataset(8807, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8807)
        mainwindow._fourier_panel._estimate_average_error_check.setChecked(True)

        mainwindow._on_compute_fourier()

        plotted = mainwindow._frequency_plot_panel._current_dataset
        assert plotted is not None
        assert np.any(plotted.error > 0.0)
        assert "Mean error" in mainwindow._fourier_panel._average_summary_label.text()

    def test_frequency_plot_is_cached_per_dataset(self, mainwindow: MainWindow) -> None:
        dataset_a = _make_fourier_ready_dataset(8809, with_grouping=True)
        dataset_b = _make_fourier_ready_dataset(8810, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset_a)
        mainwindow._data_browser.add_dataset(dataset_b)

        mainwindow._on_dataset_selected(8809)
        mainwindow._on_compute_fourier()
        plotted_a = mainwindow._frequency_plot_panel._current_dataset

        mainwindow._on_dataset_selected(8810)
        assert mainwindow._frequency_plot_panel._current_dataset is None

        mainwindow._on_dataset_selected(8809)
        restored = mainwindow._frequency_plot_panel._current_dataset

        assert plotted_a is not None
        assert restored is not None
        assert restored.metadata["run_number"] == 8809
        assert np.allclose(restored.asymmetry, plotted_a.asymmetry)

    def test_frequency_switch_to_uncached_dataset_preserves_x_limits(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset_a = _make_fourier_ready_dataset(8812, with_grouping=True)
        dataset_b = _make_fourier_ready_dataset(8815, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset_a)
        mainwindow._data_browser.add_dataset(dataset_b)

        mainwindow._on_dataset_selected(8812)
        mainwindow._on_compute_fourier()
        _x_min, _x_max, y_min, y_max = mainwindow._frequency_plot_panel.get_view_limits()
        mainwindow._frequency_plot_panel.set_view_limits(1.25, 3.75, y_min, y_max)

        mainwindow._on_dataset_selected(8815)

        x_min, x_max, _y_min, _y_max = mainwindow._frequency_plot_panel.get_view_limits()
        assert mainwindow._frequency_plot_panel._current_dataset is None
        assert x_min == pytest.approx(1.25)
        assert x_max == pytest.approx(3.75)

        mainwindow._on_compute_fourier()

        computed_x_min, computed_x_max, _computed_y_min, _computed_y_max = (
            mainwindow._frequency_plot_panel.get_view_limits()
        )
        assert computed_x_min == pytest.approx(1.25)
        assert computed_x_max == pytest.approx(3.75)

    def test_fourier_phase_tables_persist_per_dataset(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset_a = _make_fourier_ready_dataset(8820, with_grouping=True)
        dataset_b = _make_fourier_ready_dataset(8821, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset_a)
        mainwindow._data_browser.add_dataset(dataset_b)

        mainwindow._on_dataset_selected(8820)
        mainwindow._fourier_panel._phase_mode_radio.setChecked(True)
        mainwindow._fourier_panel._use_phase_table_check.setChecked(True)
        mainwindow._fourier_panel.set_group_phases({1: 11.0, 2: -7.0}, auto_filled=True)

        mainwindow._on_dataset_selected(8821)
        mainwindow._fourier_panel._phase_mode_radio.setChecked(True)
        mainwindow._fourier_panel._use_phase_table_check.setChecked(True)
        mainwindow._fourier_panel.set_group_phases({1: 3.0, 2: 4.0}, auto_filled=False)

        mainwindow._on_dataset_selected(8820)
        assert mainwindow._fourier_panel.group_phase_table() == pytest.approx({1: 11.0, 2: -7.0})
        assert (
            mainwindow._fourier_panel._phase_table.item(0, 2).foreground().color().name()
            == tokens.OK
        )

        mainwindow._on_dataset_selected(8821)
        assert mainwindow._fourier_panel.group_phase_table() == pytest.approx({1: 3.0, 2: 4.0})
        assert (
            mainwindow._fourier_panel._phase_table.item(0, 2).foreground().color().name()
            == tokens.ACCENT
        )

    def test_maxent_panel_settings_persist_per_dataset_while_browsing(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset_a = _make_fourier_ready_dataset(8850, with_grouping=True)
        dataset_b = _make_fourier_ready_dataset(8851, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset_a)
        mainwindow._data_browser.add_dataset(dataset_b)

        mainwindow._on_dataset_selected(8850)
        mainwindow._maxent_panel._points_spin.setValue(128)
        mainwindow._maxent_panel._t_min_edit.setText("0.25")
        mainwindow._maxent_panel._t_max_edit.setText("4.5")
        mainwindow._maxent_panel._time_binning_spin.setValue(4)
        table_a = mainwindow._maxent_panel._group_table
        table_a.item(0, 2).setText("21.0")
        table_a.item(1, 0).setCheckState(Qt.CheckState.Unchecked)

        mainwindow._on_dataset_selected(8851)
        assert mainwindow._maxent_panel._points_spin.value() == 128
        assert mainwindow._maxent_panel._time_binning_spin.value() == 4
        assert mainwindow._maxent_panel._t_min_edit.text() == "0.25"
        assert mainwindow._maxent_panel.selected_group_ids() == [1, 2]
        mainwindow._maxent_panel._points_spin.setValue(64)
        mainwindow._maxent_panel._time_binning_spin.setValue(2)

        mainwindow._on_dataset_selected(8850)

        assert mainwindow._maxent_panel._points_spin.value() == 128
        assert mainwindow._maxent_panel._time_binning_spin.value() == 4
        assert mainwindow._maxent_panel.group_phase_table()[1] == pytest.approx(21.0)
        assert mainwindow._maxent_panel.selected_group_ids() == [1]

        mainwindow._on_dataset_selected(8851)
        assert mainwindow._maxent_panel._points_spin.value() == 64
        assert mainwindow._maxent_panel._time_binning_spin.value() == 2

    def test_maxent_panel_loads_persisted_recipe_when_selecting_dataset(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8852, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        representation = mainwindow._project_model.ensure_dataset(8852).ensure(
            RepresentationType.FREQ_MAXENT
        )
        representation.recipe = {
            "maxent_config": {
                "n_spectrum_points": 128,
                "f_min_mhz": 0.5,
                "f_max_mhz": 3.5,
                "auto_window": False,
                "t_min_us": 0.4,
                "t_max_us": 5.2,
                "time_binning_factor": 8,
                # Panel-authored recipes carry a phase entry for every group
                # present at compute time; group 1 being known-but-unselected
                # is what keeps it excluded on restore.
                "selected_group_ids": [2],
                "group_phase_degrees": {1: 0.0, 2: 44.0},
            }
        }

        mainwindow._on_dataset_selected(8852)

        assert mainwindow._maxent_panel._points_spin.value() == 128
        assert mainwindow._maxent_panel._auto_window_check.isChecked() is False
        assert mainwindow._maxent_panel._f_min_edit.text() == "0.5"
        assert mainwindow._maxent_panel._t_max_edit.text() == "5.2"
        assert mainwindow._maxent_panel._time_binning_spin.value() == 8
        assert mainwindow._maxent_panel.selected_group_ids() == [2]
        assert mainwindow._maxent_panel.group_phase_table()[2] == pytest.approx(44.0)

    def test_maxent_group_enabled_derivation_defaults_unknown_groups_on(self) -> None:
        """Groups a stored selection never knew about (re-grouped run, recipe
        copied across different groupings) default to enabled; groups that were
        known but unselected stay disabled; no selection info means all on."""
        derive = MainWindow._derive_group_enabled_table
        names = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

        # Panel-authored recipe: every group known, only 1 selected.
        state = {"selected_group_ids": [1], "group_phase_degrees": {1: 0.0, 2: 0.0}}
        assert derive(state, names) == {1: True, 2: False, 3: True, 4: True}
        # Disjoint recipe (copied from a run with other groups): all enabled.
        state = {"selected_group_ids": [7, 8], "group_phase_degrees": {7: 0.0, 8: 0.0}}
        assert derive(state, names) == {1: True, 2: True, 3: True, 4: True}
        # No selection constraint at all: all enabled.
        assert derive({}, names) == {1: True, 2: True, 3: True, 4: True}

    def test_apply_maxent_to_selection_resets_target_state(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Applying settings discards the target's old result outright: stale
        diagnostics and stored panel drafts must go with it."""
        source = _make_fourier_ready_dataset(8860, with_grouping=True)
        target = _make_fourier_ready_dataset(8861, with_grouping=True)
        mainwindow._data_browser.add_dataset(source)
        mainwindow._data_browser.add_dataset(target)
        source_rep = mainwindow._project_model.ensure_dataset(8860).ensure(
            RepresentationType.FREQ_MAXENT
        )
        source_rep.recipe = {"maxent_config": {"n_spectrum_points": 64, "auto_window": False}}
        target_rep = mainwindow._project_model.ensure_dataset(8861).ensure(
            RepresentationType.FREQ_MAXENT
        )
        target_rep.result_metadata = {"cycles": 25, "diagnostics": {"chi2": [1.0]}}
        mainwindow._maxent_panel_state_by_run[8861] = {"n_spectrum_points": 9999}

        mainwindow._on_dataset_selected(8860)
        mainwindow._data_browser.get_selected_datasets = lambda: [source, target]
        mainwindow._on_apply_maxent_to_selection()

        assert target_rep.recipe["maxent_config"]["n_spectrum_points"] == 64
        assert target_rep.result_metadata == {}
        assert 8861 not in mainwindow._maxent_panel_state_by_run

    def test_record_maxent_recipe_does_not_clobber_panel_state_store(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Regression: the worker-finish recipe write must not overwrite panel
        drafts the user stored (via a dataset switch) while the compute ran."""
        from asymmetry.core.maxent import MaxEntConfig

        draft = {"n_spectrum_points": 1234, "marker": True}
        mainwindow._maxent_panel_state_by_run[8862] = dict(draft)
        spectrum = MuonDataset(
            time=np.linspace(0.0, 5.0, 16),
            asymmetry=np.zeros(16),
            error=np.zeros(16),
            metadata={"run_number": 8862},
            run=None,
        )

        mainwindow._record_frequency_maxent_recipe(
            8862, MaxEntConfig(), spectrum, {"cycles": [3]}, total_cycles=3
        )

        assert mainwindow._maxent_panel_state_by_run[8862] == draft

    @pytest.mark.timeout(300)
    def test_maxent_worker_finish_leaves_view_when_user_moved_on(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Finishing a compute must not yank the workspace to the maxent view
        when the user selected a different run / view mid-compute."""
        computed = _make_fourier_ready_dataset(8863, with_grouping=True)
        other = _make_fourier_ready_dataset(8864, with_grouping=True)
        mainwindow._data_browser.add_dataset(computed)
        mainwindow._data_browser.add_dataset(other)
        mainwindow._on_dataset_selected(8863)
        mainwindow._maxent_panel._points_spin.setValue(64)
        mainwindow._maxent_panel._inner_spin.setValue(1)
        mainwindow._maxent_panel._auto_window_check.setChecked(False)
        mainwindow._maxent_panel._f_min_edit.setText("0.1")
        mainwindow._maxent_panel._f_max_edit.setText("4.0")

        mainwindow._on_compute_maxent(1)
        # The worker's finished signal is queued, so these run first even if
        # the computation is already done.
        mainwindow._on_dataset_selected(8864)
        mainwindow._plot_workspace.set_active_view("fb_asymmetry")
        wait_for(lambda: mainwindow._maxent_thread is None, QApplication.instance(), timeout_s=5.0)

        assert mainwindow._plot_workspace.active_view() == "fb_asymmetry"
        assert 8863 in mainwindow._frequency_cache(RepresentationType.FREQ_MAXENT)

    def test_corrupted_maxent_recipe_does_not_break_dataset_selection(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Malformed recipe entries from a project file must degrade, not raise
        out of the dataset-selection slot."""
        dataset = _make_fourier_ready_dataset(8865, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        representation = mainwindow._project_model.ensure_dataset(8865).ensure(
            RepresentationType.FREQ_MAXENT
        )
        representation.recipe = {
            "maxent_config": {
                "n_spectrum_points": 128,
                "selected_group_ids": [1, "abc"],
                "group_phase_degrees": {"bogus-key": 1.0, "2": "not-a-number", "1": 15.0},
                "outer_cycles": "bad",
            }
        }

        mainwindow._on_dataset_selected(8865)

        assert mainwindow._maxent_panel._points_spin.value() == 128
        assert mainwindow._maxent_panel.group_phase_table()[1] == pytest.approx(15.0)

    def test_maxent_draft_settings_round_trip_with_project(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8853, with_grouping=True)
        assert dataset.run is not None
        source_file = tmp_path / "run_8853.mdu"
        source_file.write_text("placeholder", encoding="utf-8")
        dataset.run.source_file = str(source_file)
        dataset.metadata["source_file"] = str(source_file)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8853)
        mainwindow._maxent_panel._points_spin.setValue(256)
        mainwindow._maxent_panel._time_binning_spin.setValue(16)
        mainwindow._maxent_panel._t_min_edit.setText("0.75")
        mainwindow._maxent_panel._t_max_edit.setText("6.25")
        table = mainwindow._maxent_panel._group_table
        table.item(0, 2).setText("13.5")
        table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)

        state = mainwindow.collect_project_state()
        assert state["maxent_state_by_run"]["8853"]["time_binning_factor"] == 16
        project_path = tmp_path / "maxent_draft.asymp"
        save_project(state, project_path)
        loaded_state = load_project(project_path)
        restored_window = MainWindow()

        def _fake_load_file(_path: str) -> MuonDataset:
            loaded = _make_fourier_ready_dataset(8853, with_grouping=True)
            assert loaded.run is not None
            loaded.run.source_file = str(source_file)
            loaded.metadata["source_file"] = str(source_file)
            return loaded

        monkeypatch.setattr(restored_window, "_load_file", _fake_load_file)
        restored_window.restore_project_state(loaded_state, str(project_path))

        assert restored_window._maxent_panel._points_spin.value() == 256
        assert restored_window._maxent_panel._time_binning_spin.value() == 16
        assert restored_window._maxent_panel._t_min_edit.text() == "0.75"
        assert restored_window._maxent_panel.group_phase_table()[1] == pytest.approx(13.5)
        assert restored_window._maxent_panel.selected_group_ids() == [1]

    def test_frequency_axis_toggle_can_show_relative_values(self, mainwindow: MainWindow) -> None:
        dataset = _make_fourier_ready_dataset(8811, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8811)
        mainwindow._on_compute_fourier()
        abs_x_min, abs_x_max, _abs_y_min, _abs_y_max = (
            mainwindow._frequency_plot_panel.get_view_limits()
        )
        abs_axis_min, abs_axis_max = mainwindow._frequency_plot_panel._ax.get_xlim()

        mainwindow._frequency_axis_relative_check.setChecked(True)

        x_min, x_max, _y_min, _y_max = mainwindow._frequency_plot_panel.get_view_limits()
        center = 100.0 * 135.538817 * 1.0e-4
        plotted = mainwindow._frequency_plot_panel._current_dataset

        assert plotted is not None
        assert mainwindow._frequency_plot_panel._ax.get_xlabel() == "Frequency (MHz)"
        assert x_min == pytest.approx(abs_x_min - center, abs=1e-3)
        assert x_max == pytest.approx(abs_x_max - center, abs=1e-3)
        assert mainwindow._frequency_plot_panel._ax.get_xlim()[0] == pytest.approx(
            abs_axis_min, abs=1e-3
        )
        assert mainwindow._frequency_plot_panel._ax.get_xlim()[1] == pytest.approx(
            abs_axis_max, abs=1e-3
        )

    def test_frequency_phase_window_narrows_wide_relative_view(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8813, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8813)
        mainwindow._on_compute_fourier()

        mainwindow._frequency_axis_relative_check.setChecked(True)
        _x_min, _x_max, y_min, y_max = mainwindow._frequency_plot_panel.get_view_limits()
        mainwindow._frequency_plot_panel.set_view_limits(-40.0, 40.0, y_min, y_max)

        freqs = np.linspace(0.0, 80.0, 801)
        lo, hi = mainwindow._resolve_fourier_phase_window_mhz(dataset, freqs)
        center = 100.0 * 135.538817 * 1.0e-4

        assert lo == pytest.approx(max(0.0, center - 10.0), abs=1e-3)
        assert hi == pytest.approx(center + 10.0, abs=1e-3)

    def test_frequency_recompute_preserves_x_limits(self, mainwindow: MainWindow) -> None:
        dataset = _make_fourier_ready_dataset(8814, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8814)
        mainwindow._on_compute_fourier()

        mainwindow._frequency_axis_relative_check.setChecked(True)
        _x_min, _x_max, y_min, y_max = mainwindow._frequency_plot_panel.get_view_limits()
        mainwindow._frequency_plot_panel.set_view_limits(-0.75, 0.25, y_min, y_max)

        mainwindow._on_compute_fourier()

        x_min, x_max, _y_min, _y_max = mainwindow._frequency_plot_panel.get_view_limits()
        center = 100.0 * 135.538817 * 1.0e-4

        assert x_min == pytest.approx(-0.75, abs=1e-6)
        assert x_max == pytest.approx(0.25, abs=1e-6)
        assert mainwindow._frequency_plot_panel._ax.get_xlim()[0] == pytest.approx(
            center - 0.75, abs=1e-3
        )
        assert mainwindow._frequency_plot_panel._ax.get_xlim()[1] == pytest.approx(
            center + 0.25, abs=1e-3
        )

    @pytest.mark.timeout(300)
    def test_compute_maxent_uses_maxent_view_and_separate_cache(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8840, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8840)
        mainwindow._maxent_panel._points_spin.setValue(64)
        mainwindow._maxent_panel._inner_spin.setValue(1)
        mainwindow._maxent_panel._auto_window_check.setChecked(False)
        mainwindow._maxent_panel._f_min_edit.setText("0.1")
        mainwindow._maxent_panel._f_max_edit.setText("4.0")
        mainwindow._maxent_panel._time_binning_spin.setValue(2)

        mainwindow._on_compute_maxent(1)
        wait_for(lambda: mainwindow._maxent_thread is None, QApplication.instance(), timeout_s=10.0)

        assert mainwindow._plot_workspace.active_view() == "maxent"
        assert mainwindow._spectrum_stack.currentWidget() is mainwindow._maxent_panel
        assert 8840 not in mainwindow._frequency_spectra_by_run
        maxent_cache = mainwindow._frequency_cache(RepresentationType.FREQ_MAXENT)
        assert 8840 in maxent_cache
        assert maxent_cache[8840][0].metadata["frequency_representation"] == "maxent"
        representation = mainwindow._project_model.representation(
            8840, RepresentationType.FREQ_MAXENT
        )
        assert representation is not None
        assert representation.result_metadata["cycles"] == 1
        assert representation.recipe["maxent_config"]["time_binning_factor"] == 2

    @pytest.mark.timeout(300)
    def test_compute_maxent_preserves_unsaved_group_table_edits(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Regression: the pre-compute panel sync wiped in-table edits made
        after run selection (phases reset to 0, all groups re-included)."""
        dataset = _make_fourier_ready_dataset(8841, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8841)
        mainwindow._maxent_panel._points_spin.setValue(64)
        mainwindow._maxent_panel._inner_spin.setValue(1)
        mainwindow._maxent_panel._auto_window_check.setChecked(False)
        mainwindow._maxent_panel._f_min_edit.setText("0.1")
        mainwindow._maxent_panel._f_max_edit.setText("4.0")
        # Edit the group table in place without re-selecting the dataset.
        table = mainwindow._maxent_panel._group_table
        table.item(0, 2).setText("33.0")
        table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)

        mainwindow._on_compute_maxent(1)
        wait_for(lambda: mainwindow._maxent_thread is None, QApplication.instance(), timeout_s=10.0)

        representation = mainwindow._project_model.representation(
            8841, RepresentationType.FREQ_MAXENT
        )
        assert representation is not None
        config = representation.recipe["maxent_config"]
        assert config["group_phase_degrees"][1] == pytest.approx(33.0)
        assert config["selected_group_ids"] == [1]
        # The visible table still shows the edits after the compute.
        assert table.item(0, 2).text() == "33.000"
        assert table.item(1, 0).checkState() == Qt.CheckState.Unchecked

    def test_project_restore_persists_cached_fourier_spectra(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8816, with_grouping=True)
        assert dataset.run is not None
        source_file = tmp_path / "run_8816.mdu"
        source_file.write_text("placeholder", encoding="utf-8")
        dataset.run.source_file = str(source_file)
        dataset.metadata["source_file"] = str(source_file)

        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8816)
        mainwindow._on_compute_fourier()
        mainwindow._fourier_panel._phase_mode_radio.setChecked(True)
        mainwindow._fourier_panel._use_phase_table_check.setChecked(True)
        mainwindow._fourier_panel.set_group_phases({1: 14.0, 2: -9.0}, auto_filled=True)
        mainwindow._frequency_axis_relative_check.setChecked(True)
        _x_min, _x_max, y_min, y_max = mainwindow._frequency_plot_panel.get_view_limits()
        mainwindow._frequency_plot_panel.set_view_limits(-0.8, 0.4, y_min, y_max)

        state = mainwindow.collect_project_state()
        assert str(8816) in state.get("fourier_spectra_state", {})

        restored_window = MainWindow()

        def _fake_load_file(_path: str) -> MuonDataset:
            loaded = _make_fourier_ready_dataset(8816, with_grouping=True)
            assert loaded.run is not None
            loaded.run.source_file = str(source_file)
            loaded.metadata["source_file"] = str(source_file)
            return loaded

        monkeypatch.setattr(restored_window, "_load_file", _fake_load_file)

        restored_window.restore_project_state(state, str(tmp_path / "restored.asymp"))

        assert restored_window._plot_workspace.active_domain() == "frequency"
        restored_dataset = restored_window._frequency_plot_panel._current_dataset
        assert restored_dataset is not None
        assert restored_dataset.metadata.get("run_number") == 8816
        assert restored_window._frequency_plot_panel.is_frequency_axis_relative_to_reference()
        assert "group_phase_state_by_run" in state.get("fourier_state", {})
        assert restored_window._fourier_panel.group_phase_table() == pytest.approx(
            {1: 14.0, 2: -9.0}
        )
        assert (
            restored_window._fourier_panel._phase_table.item(0, 2).foreground().color().name()
            == tokens.OK
        )

    def test_project_round_trip_restores_frequency_fit_state(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8817, with_grouping=True)
        assert dataset.run is not None
        source_file = tmp_path / "run_8817.mdu"
        source_file.write_text("placeholder", encoding="utf-8")
        dataset.run.source_file = str(source_file)
        dataset.metadata["source_file"] = str(source_file)

        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(8817)
        mainwindow._on_compute_fourier()

        mainwindow._fit_panel.set_domain("frequency")
        mainwindow._fit_panel.set_dataset(mainwindow._active_frequency_fit_dataset())
        mainwindow._fit_panel.set_datasets(mainwindow._frequency_fit_datasets_for_selected_runs())

        single_model = CompositeModel(["LorentzianPeak", "LinearBackground"], operators=["+"])
        global_model = CompositeModel(["GaussianPeak", "LinearBackground"], operators=["+"])

        def _find_row(table, param_name: str) -> int:
            for row in range(table.rowCount()):
                name_item = table.item(row, 0)
                if name_item is None:
                    continue
                if name_item.data(mw_module.Qt.ItemDataRole.UserRole) == param_name:
                    return row
            raise AssertionError(f"Missing parameter row: {param_name}")

        mainwindow._fit_panel._single_tab._set_composite_model(single_model)
        single_table = mainwindow._fit_panel._single_tab._param_table
        single_table.item(_find_row(single_table, "height"), 1).setText("4.2")
        single_table.item(_find_row(single_table, "nu0"), 1).setText("2.75")
        single_table.item(_find_row(single_table, "fwhm"), 1).setText("0.33")
        single_table.item(_find_row(single_table, "bg"), 1).setText("0.18")
        single_table.item(_find_row(single_table, "slope"), 1).setText("0.12")
        mainwindow._fit_panel._single_tab._result_label.setText("Frequency single result marker")

        mainwindow._fit_panel._global_tab._set_composite_model(global_model)
        global_table = mainwindow._fit_panel._global_tab._param_table
        global_table.item(_find_row(global_table, "height"), 1).setText("5.1")
        global_table.item(_find_row(global_table, "nu0"), 1).setText("2.5")
        global_table.item(_find_row(global_table, "fwhm"), 1).setText("0.28")
        global_table.item(_find_row(global_table, "bg"), 1).setText("0.05")
        global_table.item(_find_row(global_table, "slope"), 1).setText("0.01")
        nu0_type_combo = global_table.cellWidget(_find_row(global_table, "nu0"), 2)
        assert nu0_type_combo is not None
        nu0_type_combo.setCurrentText("Local")
        slope_type_combo = global_table.cellWidget(_find_row(global_table, "slope"), 2)
        assert slope_type_combo is not None
        slope_type_combo.setCurrentText("Global")
        mainwindow._fit_panel._tabs.setCurrentIndex(1)

        state = mainwindow.collect_project_state()
        project_path = tmp_path / "frequency_fit_roundtrip.asymp"
        save_project(state, project_path)
        loaded_state = load_project(project_path)

        restored_window = MainWindow()

        def _fake_load_file(_path: str) -> MuonDataset:
            loaded = _make_fourier_ready_dataset(8817, with_grouping=True)
            assert loaded.run is not None
            loaded.run.source_file = str(source_file)
            loaded.metadata["source_file"] = str(source_file)
            return loaded

        monkeypatch.setattr(restored_window, "_load_file", _fake_load_file)

        restored_window.restore_project_state(loaded_state, str(project_path))

        restored_global_table = restored_window._fit_panel._global_tab._param_table
        restored_single_table = restored_window._fit_panel._single_tab._param_table

        assert restored_window._plot_workspace.active_domain() == "frequency"
        assert restored_window._fit_panel.domain() == "frequency"
        assert (
            restored_window._fit_panel.single_fit_formula_string() == single_model.formula_string()
        )
        assert (
            restored_window._fit_panel.global_fit_formula_string() == global_model.formula_string()
        )
        assert restored_window._fit_panel._tabs.currentIndex() == 1
        assert (
            restored_window._fit_panel._single_tab._result_label.text()
            == "Frequency single result marker"
        )
        assert (
            restored_single_table.item(_find_row(restored_single_table, "slope"), 1).text()
            == "0.12"
        )
        assert (
            restored_global_table.item(_find_row(restored_global_table, "nu0"), 1).text() == "2.5"
        )
        restored_nu0_type_combo = restored_global_table.cellWidget(
            _find_row(restored_global_table, "nu0"), 2
        )
        assert restored_nu0_type_combo is not None
        assert restored_nu0_type_combo.currentText() == "Local"


class TestMainWindowBasic:
    def test_initialization(self, mainwindow: MainWindow) -> None:
        """Test mainwindow initializes correctly."""
        assert mainwindow is not None
        assert mainwindow.windowTitle() != ""
        assert mainwindow._dock_fourier.minimumWidth() == 280

    def test_perf_logging_reports_selection_and_fourier_when_enabled(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_fourier_ready_dataset(8830, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        monkeypatch.setenv("ASYMMETRY_PERF_LOGGING", "1")

        mainwindow._on_dataset_selected(8830)
        mainwindow._on_compute_fourier()

        log_text = mainwindow._log_panel.to_plain_text()
        assert "PERF selection_plot:" in log_text
        assert "PERF dataset_selected:" in log_text
        assert "PERF compute_fourier:" in log_text

    def test_perf_logging_reports_load_file_batches_when_enabled(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_dataset(8831, with_grouping=True)
        monkeypatch.setenv("ASYMMETRY_PERF_LOGGING", "1")
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: dataset)

        mainwindow._load_files(["/tmp/run8831.nxs"])

        log_text = mainwindow._log_panel.to_plain_text()
        assert "PERF load_files_batch:" in log_text
        assert "files=1" in log_text

    def test_has_menu_bar(self, mainwindow: MainWindow) -> None:
        """Test menubar exists."""
        assert mainwindow.menuBar() is not None

    def test_options_menu_has_temperature_log_toggle(self, mainwindow: MainWindow) -> None:
        """Options menu should expose the sample-log temperature toggle."""
        options_menu = None
        for action in mainwindow.menuBar().actions():
            if action.text().replace("&", "") == "Options":
                options_menu = action.menu()
                break

        assert options_menu is not None
        assert any(action.text() == "Use temperature from log" for action in options_menu.actions())
        assert any(
            action.text() == "Enable performance logging" for action in options_menu.actions()
        )
        assert any(action.text() == "Enable plot decimation" for action in options_menu.actions())

    def test_perf_logging_option_persists_and_updates_action(
        self,
        mainwindow: MainWindow,
    ) -> None:
        action = mainwindow._perf_logging_action

        # Reset to known-off state regardless of any persisted QSettings contamination,
        # then toggle on so the toggled signal always fires.
        action.setChecked(False)
        action.setChecked(True)

        assert mainwindow._settings.value("debug/perf_logging", False, bool) is True
        assert "Performance logging enabled." in mainwindow._log_panel.to_plain_text()
        assert action.isChecked() is True

    def test_plot_decimation_option_defaults_on_and_updates_both_plot_panels(
        self,
        mainwindow: MainWindow,
    ) -> None:
        # Establish known-on state: any persisted QSettings contamination from a prior
        # run that toggled decimation off will leave the action unchecked on construction.
        mainwindow._plot_decimation_action.setChecked(True)

        assert mainwindow._plot_decimation_action.isChecked() is True
        assert mainwindow._plot_panel.decimation_enabled() is True
        assert mainwindow._frequency_plot_panel.decimation_enabled() is True

        mainwindow._plot_decimation_action.setChecked(False)

        assert mainwindow._settings.value("plot/enable_decimation", True, bool) is False
        assert mainwindow._plot_panel.decimation_enabled() is False
        assert mainwindow._frequency_plot_panel.decimation_enabled() is False
        assert "Plot decimation disabled." in mainwindow._log_panel.to_plain_text()

    def test_has_central_widget(self, mainwindow: MainWindow) -> None:
        """Test central widget exists."""
        assert mainwindow.centralWidget() is not None

    def test_view_menu_uses_fixed_ui_scale_actions_only(self, mainwindow: MainWindow) -> None:
        """View menu should only expose fixed UI scale actions."""
        toolbar = mainwindow.findChild(QToolBar)

        assert not hasattr(mainwindow, "_ui_scale_slider")
        assert not hasattr(mainwindow, "_ui_scale_value_label")
        assert toolbar is not None

    def test_view_menu_has_ui_scale_submenu(self, mainwindow: MainWindow) -> None:
        """View menu should expose the configured UI scale choices."""
        view_menu = None
        for action in mainwindow.menuBar().actions():
            if action.text().replace("&", "") == "View":
                view_menu = action.menu()
                break

        assert view_menu is not None
        scale_menu_action = next(
            (
                action
                for action in view_menu.actions()
                if action.menu() is not None and action.text() == "UI Scale"
            ),
            None,
        )
        assert scale_menu_action is not None

        texts = [action.text() for action in scale_menu_action.menu().actions() if action.text()]
        assert texts == ["80%", "90%", "100%", "110%", "120%"]

    def test_window_size(self, mainwindow: MainWindow) -> None:
        """Test window has reasonable size."""
        size = mainwindow.size()
        assert size.width() > 0
        assert size.height() > 0

    def test_plot_workspace_uses_fb_groups_and_frequency_tabs(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """The toolbar Domain buttons should expose the 2+2 Time/Frequency views."""
        labels = [btn.text() for btn in mainwindow._domain_buttons]
        assert labels == ["F-B asymmetry", "Individual groups", "FFT", "MaxEnt"]

    def test_on_fit_shows_fit_dock(self, mainwindow: MainWindow) -> None:
        """Fit action should unhide the fit dock if it starts hidden."""
        assert mainwindow._dock_fit.isHidden()
        mainwindow._on_fit()
        assert not mainwindow._dock_fit.isHidden()

    def test_on_fit_switches_dock_to_multi_group_fit_panel_from_group_view(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fit action should swap the fit dock to grouped-fit content from group view."""

        class _StubMultiGroupFitWindow(QWidget):
            def __init__(self, *_args, **_kwargs) -> None:
                super().__init__()
                self.grouped_fit_completed = SimpleNamespace(connect=lambda _callback: None)
                self.grouped_preview_requested = SimpleNamespace(connect=lambda _callback: None)
                self.last_dataset = None
                self.last_block_state = None
                self._title = "Multi-Group Fit"

            def set_dataset(self, dataset) -> None:
                self.last_dataset = dataset
                if dataset is None:
                    self._title = "Multi-Group Fit"
                    return
                self._title = (
                    f"Multi-Group Fit — {getattr(dataset, 'run_label', dataset.run_number)}"
                )

            def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
                self.last_block_state = (blocked, reason)

            def dock_title(self) -> str:
                return self._title

            def grouped_fit_formula_string(self) -> str:
                return "A(t)"

        dataset = _make_dataset(4201, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._current_dataset = dataset
        monkeypatch.setattr(mw_module, "MultiGroupFitWindow", _StubMultiGroupFitWindow)
        mainwindow._multi_group_fit_window = _StubMultiGroupFitWindow()
        mainwindow._fit_stack.addWidget(mainwindow._multi_group_fit_window)
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda dataset=None: [dataset or mainwindow._current_dataset],
        )
        mainwindow._refresh_time_view_selector()
        mainwindow._plot_workspace.set_active_view("groups")

        # Phase 4: switching to groups domain now shows the fit dock automatically.
        assert not mainwindow._dock_fit.isHidden()
        mainwindow._on_fit()

        assert isinstance(mainwindow._multi_group_fit_window, _StubMultiGroupFitWindow)
        assert mainwindow._fit_stack.currentWidget() is mainwindow._multi_group_fit_window
        assert not mainwindow._dock_fit.isHidden()

    def test_grouped_preview_request_updates_group_plots_without_fit_result(
        self,
        mainwindow: MainWindow,
    ) -> None:
        grouped_datasets = [
            _make_dataset(4204, with_grouping=True),
            _make_dataset(4205, with_grouping=True),
        ]
        captured: dict[str, object] = {}

        mainwindow._plot_panel.set_global_fits = lambda payload: captured.update(
            {"payload": payload}
        )
        mainwindow._plot_panel.plot_grouped_time_domain_subplots = lambda datasets: captured.update(
            {"datasets": list(datasets)}
        )

        preview_curves = {
            4204: (object(), (np.array([0.0, 1.0]), np.array([1.0, 0.5])), tuple()),
            4205: (object(), (np.array([0.0, 1.0]), np.array([0.8, 0.4])), tuple()),
        }

        mainwindow._on_grouped_preview_requested(
            grouped_datasets,
            preview_curves,
            fit_function="A(t)",
        )

        assert captured["datasets"] == grouped_datasets
        payload = captured["payload"]
        assert payload[4204][2] == "Grouped Preview"
        assert payload[4204][4] is None
        assert payload[4204][5] == "A(t)"

    def test_time_view_switch_replaces_visible_fit_dock_with_grouped_panel(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Switching between FB asymmetry and grouped view should swap the visible fit dock."""
        dataset = _make_dataset(4202, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._current_dataset = dataset
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda dataset=None: [dataset or mainwindow._current_dataset],
        )
        mainwindow._refresh_time_view_selector()

        mainwindow._on_fit()
        assert mainwindow._fit_stack.currentWidget() is mainwindow._fit_panel

        mainwindow._plot_workspace.set_active_view("groups")
        assert mainwindow._fit_stack.currentWidget() is mainwindow._multi_group_fit_window

        mainwindow._plot_workspace.set_active_view("fb_asymmetry")
        assert mainwindow._fit_stack.currentWidget() is mainwindow._fit_panel

    def test_frequency_view_can_switch_directly_to_grouped_time_view(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_dataset(4203, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._current_dataset = dataset
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda dataset=None: [dataset or mainwindow._current_dataset],
        )

        mainwindow._plot_workspace.set_active_view("frequency")
        mainwindow._refresh_time_view_selector()

        assert mainwindow._plot_workspace.active_view() == "frequency"
        assert mainwindow._plot_workspace.is_view_enabled("groups")

        mainwindow._plot_workspace.set_active_view("groups")

        assert mainwindow._plot_workspace.active_view() == "groups"
        assert mainwindow._plot_panel.current_time_view_mode() == "groups"

    def test_frequency_round_trip_restores_vector_polarization_selector(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(4206, with_grouping=False)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "groups": {
                    1: [1],
                    2: [2],
                    3: [1],
                    4: [2],
                    5: [1],
                    6: [2],
                },
                "group_names": {
                    1: "Pz Forward",
                    2: "Pz Backward",
                    3: "Py Top",
                    4: "Py Bottom",
                    5: "Px Left",
                    6: "Px Right",
                },
                "forward_group": 1,
                "backward_group": 2,
                "vector_axis": "P_z",
            }
        )

        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._current_dataset = dataset
        mainwindow._refresh_vector_axis_selector()

        assert mainwindow._plot_panel._polarization_combo.count() == 4
        assert not mainwindow._plot_panel._polarization_combo.isHidden()

        mainwindow._plot_workspace.set_active_view("frequency")

        assert mainwindow._plot_panel._polarization_combo.isHidden()

        mainwindow._plot_workspace.set_active_view("fb_asymmetry")

        assert mainwindow._plot_panel._polarization_combo.count() == 4
        assert not mainwindow._plot_panel._polarization_combo.isHidden()

    def test_grouped_view_hides_vector_selector_until_fb_returns(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_dataset(4207, with_grouping=False)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "groups": {
                    1: [1],
                    2: [2],
                    3: [1],
                    4: [2],
                    5: [1],
                    6: [2],
                },
                "group_names": {
                    1: "Pz Forward",
                    2: "Pz Backward",
                    3: "Py Top",
                    4: "Py Bottom",
                    5: "Px Left",
                    6: "Px Right",
                },
                "forward_group": 1,
                "backward_group": 2,
                "vector_axis": "P_z",
            }
        )

        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._current_dataset = dataset
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda target=None: [target or mainwindow._current_dataset],
        )
        mainwindow._refresh_time_view_selector()
        mainwindow._refresh_vector_axis_selector()

        assert mainwindow._plot_panel._polarization_combo.count() == 4
        assert not mainwindow._plot_panel._polarization_combo.isHidden()

        mainwindow._plot_workspace.set_active_view("groups")

        assert mainwindow._plot_panel._polarization_combo.isHidden()

        mainwindow._plot_workspace.set_active_view("fb_asymmetry")

        assert mainwindow._plot_panel._polarization_combo.count() == 4
        assert not mainwindow._plot_panel._polarization_combo.isHidden()

    def test_set_compact_mode_is_legacy_no_op(self, mainwindow: MainWindow) -> None:
        """Legacy compact-mode API should leave the standard shell intact."""
        mainwindow._on_fit()
        mainwindow.set_compact_mode(True)

        assert not mainwindow.compact_mode
        assert not mainwindow._dock_fit.isHidden()
        assert mainwindow._fit_stack.currentWidget() is mainwindow._fit_panel

    def test_ui_scale_action_updates_manager_and_settings(self, mainwindow: MainWindow) -> None:
        """Scale actions should delegate through UIManager and persist the choice."""
        mainwindow._ui_scale_actions[1.1].trigger()

        assert mainwindow._ui_manager.ui_scale == pytest.approx(1.1)
        assert mainwindow._ui_scale_actions[1.1].isChecked()

        settings = QSettings()
        assert float(settings.value(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)) == pytest.approx(1.1)

    def test_global_stylesheet_enlarges_spinbox_arrow_controls(
        self, mainwindow: MainWindow
    ) -> None:
        """Global stylesheet should reserve larger, legible spinbox arrow controls."""
        stylesheet = mainwindow._ui_manager.build_stylesheet(1.0)

        assert "QAbstractSpinBox::up-button" in stylesheet
        assert "QAbstractSpinBox::down-button" in stylesheet
        assert "QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow" in stylesheet
        assert "spin_up_arrow.svg" in stylesheet
        assert "spin_down_arrow.svg" in stylesheet
        assert "width: 18px;" in stylesheet
        assert "height: 10px;" in stylesheet

    def test_on_fourier_shows_fourier_dock(self, mainwindow: MainWindow) -> None:
        """Fourier action should unhide the Fourier dock if it starts hidden."""
        assert mainwindow._dock_fourier.isHidden()
        mainwindow._on_fourier()
        assert not mainwindow._dock_fourier.isHidden()

    def test_on_fit_parameters_shows_params_dock(self, mainwindow: MainWindow) -> None:
        """Fit Parameters action should unhide the dock if it starts hidden."""
        assert mainwindow._dock_fit_parameters.isHidden()
        mainwindow._on_fit_parameters()
        assert not mainwindow._dock_fit_parameters.isHidden()

    def test_on_export_current_plot_delegates_to_plot_panel(self, mainwindow: MainWindow) -> None:
        """Export menu/toolbar handler should delegate to PlotPanel exporter."""
        called = {"count": 0}

        def _mark_called() -> None:
            called["count"] += 1

        mainwindow._plot_panel.export_current_plot = _mark_called
        mainwindow._on_export_current_plot()
        assert called["count"] == 1

    def test_on_export_current_plot_uses_active_frequency_tab(self, mainwindow: MainWindow) -> None:
        """Export should follow the active time/frequency workspace tab."""
        called = {"count": 0}

        def _mark_called() -> None:
            called["count"] += 1

        mainwindow._frequency_plot_panel.export_current_plot = _mark_called
        mainwindow._plot_workspace.set_active_domain("frequency")
        mainwindow._on_export_current_plot()

        assert called["count"] == 1

    def test_on_export_logbook_delegates_to_data_browser(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """Export logbook action should delegate to the data browser exporter."""
        destination = tmp_path / "logbook.tsv"
        monkeypatch.setattr(
            mw_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(destination), "Tab-separated values (*.tsv)"),
        )

        remembered: dict[str, str] = {}
        monkeypatch.setattr(
            mw_module, "remember_export_path", lambda p: remembered.setdefault("path", p)
        )

        called: dict[str, str] = {}

        def _export(path: str) -> int:
            called["path"] = path
            return 3

        mainwindow._data_browser.get_all_datasets = lambda: [object()]
        mainwindow._data_browser.export_logbook_tsv = _export
        mainwindow._on_export_logbook()

        assert called["path"] == str(destination)
        assert remembered["path"] == str(destination)

    def test_on_export_logbook_uses_rtf_export_when_rtf_selected(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """Choosing RTF in the save dialog should route to RTF exporter."""
        destination = tmp_path / "logbook"
        monkeypatch.setattr(
            mw_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(destination), "Rich Text Format (*.rtf)"),
        )

        called: dict[str, str] = {}

        def _export(path: str) -> int:
            called["path"] = path
            return 2

        mainwindow._data_browser.get_all_datasets = lambda: [object()]
        mainwindow._data_browser.export_logbook_rtf = _export
        mainwindow._on_export_logbook()

        assert called["path"].endswith(".rtf")

    def test_on_export_logbook_uses_project_name_in_default_filename(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default export name should include project stem when project is named."""
        mainwindow._current_project_path = "/tmp/My Project.asymp"
        mainwindow._data_browser.get_all_datasets = lambda: [object()]

        captured: dict[str, str] = {}

        def _fake_get_save_file_name(_parent, _title, default_path, _filters):
            captured["default_path"] = str(default_path)
            return "", ""

        monkeypatch.setattr(mw_module.QFileDialog, "getSaveFileName", _fake_get_save_file_name)

        mainwindow._on_export_logbook()

        assert captured["default_path"].endswith("My_Project_logbook.tsv")

    def test_toolbar_places_export_logbook_after_open(self, mainwindow: MainWindow) -> None:
        """Toolbar should place Export logbook immediately after Open."""
        toolbar = mainwindow.findChild(QToolBar)
        assert toolbar is not None
        texts = [action.text() for action in toolbar.actions()]
        assert "Open" in texts
        assert "Export logbook" in texts
        assert texts.index("Export logbook") == texts.index("Open") + 1

    def test_toolbar_exposes_three_view_modes_and_main_bunch_control(
        self, mainwindow: MainWindow
    ) -> None:
        """Main toolbar should expose three saved view buttons and a bunch control."""
        assert [button.text() for button in mainwindow._view_mode_buttons] == ["1", "2", "3"]
        assert mainwindow._view_mode_buttons[0].isChecked()
        assert mainwindow._view_bunch_spin.value() == 1

    def test_view_modes_restore_per_mode_limits_and_bunch(self, mainwindow: MainWindow) -> None:
        """Switching between view modes should restore the saved limits and bunch factor."""
        mainwindow._plot_panel.set_view_limits(0.5, 8.0, -5.0, 12.0)
        mainwindow._view_bunch_spin.setValue(3)

        mainwindow._view_mode_buttons[1].click()
        mainwindow._plot_panel.set_view_limits(1.0, 4.0, -2.0, 6.0)
        mainwindow._view_bunch_spin.setValue(5)

        mainwindow._view_mode_buttons[0].click()

        assert mainwindow._view_bunch_spin.value() == 3
        assert mainwindow._plot_panel.get_view_limits() == pytest.approx((0.5, 8.0, -5.0, 12.0))

    def test_main_window_bunch_control_updates_grouping_bunch_factor(
        self, mainwindow: MainWindow
    ) -> None:
        """Toolbar bunch changes should stay synchronized with grouping metadata."""
        dataset = _make_dataset(4101, with_grouping=True)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._on_dataset_selected(int(dataset.run_number))

        mainwindow._view_bunch_spin.setValue(4)

        assert dataset.run is not None
        assert int(dataset.run.grouping.get("bunching_factor", 1)) == 4

    def test_collect_project_state_includes_view_modes(self, mainwindow: MainWindow) -> None:
        """Project state should persist saved view modes and the active mode index."""
        mainwindow._plot_panel.set_view_limits(0.25, 7.5, -4.0, 9.0)
        mainwindow._view_bunch_spin.setValue(2)

        state = mainwindow.collect_project_state()

        assert state["view_modes_state"]["active_index"] == 0
        assert state["view_modes_state"]["modes"][0]["bunch_factor"] == 2
        assert state["view_modes_state"]["modes"][0]["x_min"] == pytest.approx(0.25)
        assert state["view_modes_state"]["modes"][0]["x_max"] == pytest.approx(7.5)

    def test_toolbar_grouping_before_fit(self, mainwindow: MainWindow) -> None:
        """Toolbar should expose Grouping action before Fit for discoverability."""
        toolbar = mainwindow.findChild(QToolBar)
        assert toolbar is not None
        texts = [action.text() for action in toolbar.actions()]
        assert "Grouping" in texts
        assert "Fit" in texts
        assert texts.index("Grouping") < texts.index("Fit")

    def test_toolbar_has_domain_segmented_control(self, mainwindow: MainWindow) -> None:
        """Toolbar should expose four Domain buttons (Time 2 + Frequency 2)."""
        assert hasattr(mainwindow, "_domain_buttons")
        assert [btn.text() for btn in mainwindow._domain_buttons] == [
            "F-B asymmetry",
            "Individual groups",
            "FFT",
            "MaxEnt",
        ]
        assert mainwindow._domain_buttons[0].isChecked()
        assert not mainwindow._domain_buttons[1].isChecked()
        assert not mainwindow._domain_buttons[2].isChecked()
        # MaxEnt is reserved but not yet implemented.
        assert not mainwindow._domain_buttons[3].isEnabled()

    def test_domain_button_click_changes_workspace_view(self, mainwindow: MainWindow) -> None:
        """Clicking the Frequency domain button should switch the workspace view."""
        mainwindow._domain_buttons[2].click()
        assert mainwindow._plot_workspace.active_view() == "frequency"

    def test_workspace_view_change_syncs_domain_buttons(self, mainwindow: MainWindow) -> None:
        """Programmatic workspace view change should update Domain button checked state."""
        mainwindow._plot_workspace.set_active_view("frequency")
        assert mainwindow._domain_buttons[2].isChecked()
        assert not mainwindow._domain_buttons[0].isChecked()

        mainwindow._plot_workspace.set_active_view("fb_asymmetry")
        assert mainwindow._domain_buttons[0].isChecked()
        assert not mainwindow._domain_buttons[2].isChecked()

    def test_deadtime_correction_uses_good_frames(self, mainwindow: MainWindow) -> None:
        """Deadtime correction should include good-frame normalization (Mantid-style)."""
        counts = np.array([100.0], dtype=float)
        corrected = mainwindow._apply_deadtime_correction(
            counts,
            tau_us=0.01,
            bin_width_us=0.02,
            num_good_frames=1000.0,
        )
        expected = 100.0 / (1.0 - (100.0 * 0.01 / (0.02 * 1000.0)))
        assert corrected[0] == pytest.approx(expected)

    def test_background_correction_subtracts_grouped_mean_before_asymmetry(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Background correction follows musrfit's grouped-histogram ordering."""
        dataset = _make_dataset(7398, with_grouping=False)
        dataset.metadata["facility"] = "PSI"
        dataset.run.metadata["facility"] = "PSI"
        dataset.run.histograms = [
            Histogram(np.array([10.0, 10.0, 100.0, 100.0]), bin_width=0.01, t0_bin=2),
            Histogram(np.array([20.0, 20.0, 80.0, 80.0]), bin_width=0.01, t0_bin=2),
        ]
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 2,
            "first_good_bin": 2,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "background_correction": True,
            "background_range": [0, 1],
        }

        applied, dt_applied = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dt_applied is False
        assert dataset.run.grouping["background_method"] == "estimated"
        assert dataset.run.grouping["background_values"] == pytest.approx([10.0, 20.0])
        assert dataset.run.grouping["background_ranges"] == [[0, 1], [0, 1]]
        np.testing.assert_allclose(dataset.asymmetry, [20.0, 20.0])
        expected_error = (
            2.0
            * np.sqrt((60.0 * np.sqrt(105.0)) ** 2 + (90.0 * np.sqrt(90.0)) ** 2)
            / (150.0**2)
            * 100.0
        )
        np.testing.assert_allclose(dataset.error, [expected_error, expected_error])

    def test_background_requested_for_non_psi_data_is_not_applied(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Background correction is the PSI path, not an ISIS grouping fallback."""
        dataset = _make_dataset(7397, with_grouping=False)
        dataset.run.metadata["facility"] = "ISIS"
        dataset.metadata["facility"] = "ISIS"
        dataset.run.histograms = [
            Histogram(np.array([10.0, 10.0, 100.0, 100.0]), bin_width=0.01, t0_bin=2),
            Histogram(np.array([20.0, 20.0, 80.0, 80.0]), bin_width=0.01, t0_bin=2),
        ]
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 2,
            "first_good_bin": 2,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "background_correction": True,
            "background_range": [0, 1],
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run.grouping["background_correction"] is False
        assert "background_method" not in dataset.run.grouping
        np.testing.assert_allclose(dataset.asymmetry, [100.0 / 9.0, 100.0 / 9.0])

    def test_background_off_preserves_raw_isis_grouping_behavior(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Leaving background off keeps the previous raw grouping calculation."""
        dataset = _make_dataset(7399, with_grouping=False)
        dataset.run.metadata["instrument"] = "HIFI"
        dataset.metadata["instrument"] = "HIFI"
        dataset.run.histograms = [
            Histogram(np.array([10.0, 10.0, 100.0, 100.0]), bin_width=0.01, t0_bin=2),
            Histogram(np.array([20.0, 20.0, 80.0, 80.0]), bin_width=0.01, t0_bin=2),
        ]
        dataset.run.grouping["background_range"] = [0, 1]
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 2,
            "first_good_bin": 2,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "background_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run.grouping["background_correction"] is False
        assert "background_method" not in dataset.run.grouping
        np.testing.assert_allclose(dataset.asymmetry, [100.0 / 9.0, 100.0 / 9.0])

    def test_apply_grouping_does_not_auto_estimate_alpha_from_bunching_change(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Grouping apply should not recalculate alpha when bunching factor is changed."""
        dataset = _make_dataset(7401, with_grouping=False)
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 0.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 3,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run is not None
        assert dataset.run.grouping["alpha"] == pytest.approx(1.0)

    def test_apply_grouping_updates_t0_and_t_good_offset(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Applying grouping should persist and apply edited t0/offset controls."""
        dataset = _make_dataset(7406, with_grouping=False)
        assert dataset.run is not None
        dataset.run.grouping["bin_index_base"] = 1

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 1,
            "t_good_offset": 2,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run.grouping["t0_bin"] == 1
        assert dataset.run.grouping["t_good_offset"] == 2
        assert dataset.run.grouping["first_good_bin"] == 3
        assert dataset.run.grouping["bin_index_base"] == 1
        assert all(hist.t0_bin == 1 for hist in dataset.run.histograms)

    def test_apply_grouping_without_histograms_updates_bunching(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Datasets without raw histograms should still allow bunch-factor changes."""
        dataset = _make_dataset(7405, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7405.nxs"
        dataset.run.histograms = []
        original_points = len(dataset.time)

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 4,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert len(dataset.time) < original_points
        assert dataset.run.grouping["bunching_factor"] == 4

    def test_apply_grouping_without_histograms_restores_source_when_bunching_reduced(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Reducing bunching should rebuild from the original source arrays."""
        dataset = _make_dataset(7406, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7406.nxs"
        dataset.run.histograms = []
        original_time = dataset.time.copy()
        original_asymmetry = dataset.asymmetry.copy()
        original_error = dataset.error.copy()

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(
            dataset,
            {**payload, "bunching_factor": 4},
        )

        assert applied is True
        assert len(dataset.time) < len(original_time)

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(
            dataset,
            {**payload, "bunching_factor": 1},
        )

        assert applied is True
        np.testing.assert_array_equal(dataset.time, original_time)
        np.testing.assert_array_equal(dataset.asymmetry, original_asymmetry)
        np.testing.assert_array_equal(dataset.error, original_error)
        assert dataset.run.grouping["bunching_factor"] == 1

    def test_apply_grouping_without_histograms_applies_first_good_bin(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Datasets without raw histograms should apply the requested good-bin window."""
        dataset = _make_dataset(7407, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7407.nxs"
        dataset.run.histograms = []
        original_time = dataset.time.copy()
        original_asymmetry = dataset.asymmetry.copy()
        original_error = dataset.error.copy()

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 2,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        np.testing.assert_array_equal(dataset.time, original_time[2:4])
        np.testing.assert_array_equal(dataset.asymmetry, original_asymmetry[2:4])
        np.testing.assert_array_equal(dataset.error, original_error[2:4])
        assert dataset.run.grouping["first_good_bin"] == 2

    def test_vector_axis_pairs_detected_from_group_names(self, mainwindow: MainWindow) -> None:
        groups = {
            1: [1],
            2: [2],
            3: [1],
            4: [2],
            5: [1],
            6: [2],
        }
        names = {
            1: "Pz Forward",
            2: "Pz Backward",
            3: "Py Top",
            4: "Py Bottom",
            5: "Px Left",
            6: "Px Right",
        }

        pairs = mainwindow._vector_axis_pairs_for_grouping(groups, names)

        assert pairs["P_z"] == (1, 2)
        assert pairs["P_y"] == (3, 4)
        assert pairs["P_x"] == (5, 6)

    def test_apply_grouping_vector_axis_overrides_forward_backward(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7450, with_grouping=False)
        payload = {
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "vector_axis": "P_x",
            "instrument": "EMU",
            "alpha": 1.0,
            "alpha_x": 1.7,
            "alpha_y": 1.3,
            "alpha_z": 1.1,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run is not None
        assert dataset.run.grouping["forward_group"] == 5
        assert dataset.run.grouping["backward_group"] == 6
        assert dataset.run.grouping["vector_axis"] == "P_x"
        assert dataset.run.grouping["instrument"] == "EMU"
        assert dataset.run.grouping["alpha"] == pytest.approx(1.7)
        assert dataset.run.grouping["alpha_x"] == pytest.approx(1.7)
        assert dataset.run.grouping["alpha_y"] == pytest.approx(1.3)
        assert dataset.run.grouping["alpha_z"] == pytest.approx(1.1)

    def test_extract_grouping_overrides_includes_vector_axis(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7451, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping["vector_axis"] = "P_y"
        dataset.run.grouping["instrument"] = "MuSR"
        dataset.run.grouping["alpha_x"] = 1.05
        dataset.run.grouping["alpha_y"] = 1.15
        dataset.run.grouping["alpha_z"] = 1.25

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["vector_axis"] == "P_y"
        assert payload["instrument"] == "MuSR"
        assert payload["alpha_x"] == pytest.approx(1.05)
        assert payload["alpha_y"] == pytest.approx(1.15)
        assert payload["alpha_z"] == pytest.approx(1.25)

    def test_extract_grouping_overrides_includes_period_mode(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_two_period_vector_dataset(7452)
        assert dataset.run is not None
        dataset.run.grouping["period_mode"] = str(mw_module.PeriodMode.GREEN)

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["period_mode"] == str(mw_module.PeriodMode.GREEN)

    def test_extract_grouping_overrides_includes_t0_and_t_good_offset(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7453, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping["t0_bin"] = 2
        dataset.run.grouping["t_good_offset"] = 3
        dataset.run.grouping["first_good_bin"] = 5

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["t0_bin"] == 2
        assert payload["t_good_offset"] == 3
        assert payload["first_good_bin"] == 5

    def test_apply_grouping_period_mode_uses_selected_period_histograms(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_two_period_vector_dataset(7454)
        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        payload["period_mode"] = str(mw_module.PeriodMode.GREEN)

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        np.testing.assert_allclose(dataset.asymmetry, np.full(4, -20.0))
        assert dataset.run is not None
        assert dataset.run.grouping["period_mode"] == str(mw_module.PeriodMode.GREEN)

    def test_apply_grouping_green_minus_red_combines_asymmetry_space(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_two_period_vector_dataset(7455)
        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        payload["period_mode"] = str(mw_module.PeriodMode.GREEN_MINUS_RED)

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        red_asym, red_err = mw_module.compute_asymmetry(
            np.full(4, 100.0),
            np.full(4, 50.0),
            alpha=1.0,
        )
        green_asym, green_err = mw_module.compute_asymmetry(
            np.full(4, 60.0),
            np.full(4, 90.0),
            alpha=1.0,
        )
        np.testing.assert_allclose(dataset.asymmetry, (green_asym - red_asym) * 100.0)
        np.testing.assert_allclose(
            dataset.error,
            np.sqrt(np.square(green_err * 100.0) + np.square(red_err * 100.0)),
        )

    def test_build_vector_axis_datasets_preserves_selected_period_mode(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_two_period_vector_dataset(7456)
        assert dataset.run is not None
        dataset.run.grouping["period_mode"] = str(mw_module.PeriodMode.GREEN)

        axis_map = mainwindow._build_vector_axis_datasets([dataset])

        assert {axis for axis, values in axis_map.items() if values} == {"P_x", "P_y", "P_z"}
        np.testing.assert_allclose(axis_map["P_z"][0].asymmetry, np.full(4, -20.0))
        np.testing.assert_allclose(
            axis_map["P_y"][0].asymmetry,
            np.full(4, ((40.0 - 80.0) / (40.0 + 80.0)) * 100.0),
        )
        np.testing.assert_allclose(
            axis_map["P_x"][0].asymmetry,
            np.full(4, ((20.0 - 100.0) / (20.0 + 100.0)) * 100.0),
        )
        for axis in ("P_x", "P_y", "P_z"):
            clone = axis_map[axis][0]
            assert clone.run is not None
            assert clone.run.grouping["period_mode"] == str(mw_module.PeriodMode.GREEN)
            assert clone.run.grouping["vector_axis"] == axis

    def test_extract_grouping_overrides_includes_group_include_flags(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7455, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping["included_groups"] = {1: True, 2: False}

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["included_groups"] == {1: True, 2: False}

    def test_extract_grouping_overrides_preserves_hist_t0_pairs(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7454, with_grouping=True)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7454.nxs"
        dataset.run.histograms = []
        dataset.run.grouping["groups"] = {
            1: [(1, 100), (2, 100)],
            2: [(3, 100), (4, 100)],
        }
        dataset.run.grouping["bunching_factor"] = 4

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["groups"] == {
            1: [(1, 100), (2, 100)],
            2: [(3, 100), (4, 100)],
        }
        assert payload["bunching_factor"] == 4

    def test_extract_grouping_overrides_omits_file_deadtime_values(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7455, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping["deadtime_correction"] = True
        dataset.run.grouping["deadtime_mode"] = "file"
        dataset.run.grouping["deadtime_method"] = "file"
        dataset.run.grouping["dead_time_us"] = [0.01, 0.02]

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["deadtime_mode"] == "file"
        assert payload["deadtime_method"] == "file"
        assert "dead_time_us" not in payload

    def test_extract_grouping_overrides_preserves_manual_deadtime_values(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7456, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "deadtime_correction": True,
                "deadtime_mode": "manual",
                "deadtime_method": "manual",
                "deadtime_manual_us": 0.025,
                "dead_time_us": [0.025, 0.025],
            }
        )

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["deadtime_mode"] == "manual"
        assert payload["deadtime_method"] == "manual"
        assert payload["deadtime_manual_us"] == pytest.approx(0.025)
        assert payload["dead_time_us"] == pytest.approx([0.025, 0.025])

    def test_extract_grouping_overrides_preserves_calibrated_deadtime_values(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7458, with_grouping=True)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "deadtime_correction": True,
                "deadtime_mode": "manual",
                "deadtime_method": "calibrate",
                "dead_time_us": [0.011, 0.022],
                "deadtime_reference_run": 7458,
            }
        )

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["deadtime_mode"] == "manual"
        assert payload["deadtime_method"] == "calibrate"
        assert payload["dead_time_us"] == pytest.approx([0.011, 0.022])
        assert payload["deadtime_reference_run"] == 7458

    def test_apply_grouping_settings_uses_manual_deadtime_payload(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7457, with_grouping=False)
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "deadtime_correction": True,
            "deadtime_mode": "manual",
            "deadtime_method": "manual",
            "deadtime_manual_us": 0.01,
            "dead_time_us": [0.01, 0.01],
            "good_frames": 1000.0,
        }

        applied, deadtime_applied = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert deadtime_applied is True
        assert dataset.run is not None
        assert dataset.run.grouping["deadtime_correction"] is True
        assert dataset.run.grouping["deadtime_mode"] == "manual"
        assert dataset.run.grouping["deadtime_method"] == "manual"
        assert dataset.run.grouping["dead_time_us"] == pytest.approx([0.01, 0.01])

    def test_apply_grouping_vector_axis_falls_back_to_scalar_alpha_when_axis_keys_missing(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7452, with_grouping=False)
        payload = {
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "vector_axis": "P_y",
            "alpha": 1.4,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run is not None
        assert dataset.run.grouping["alpha"] == pytest.approx(1.4)
        assert dataset.run.grouping["alpha_x"] == pytest.approx(1.4)
        assert dataset.run.grouping["alpha_y"] == pytest.approx(1.4)
        assert dataset.run.grouping["alpha_z"] == pytest.approx(1.4)

    def test_normalize_vector_axis_supports_all(self, mainwindow: MainWindow) -> None:
        assert mainwindow._normalize_vector_axis("All") == "ALL"

    def test_all_axis_selection_does_not_reapply_grouping(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = {"render": 0, "apply": 0}
        monkeypatch.setattr(
            mainwindow,
            "_render_current_selection_plot",
            lambda: calls.__setitem__("render", calls["render"] + 1),
        )

        def _unexpected_apply(*_args, **_kwargs):
            calls["apply"] += 1
            return True, False

        monkeypatch.setattr(mainwindow, "_apply_grouping_settings_to_dataset", _unexpected_apply)

        mainwindow._on_plot_polarization_axis_changed("All")

        assert calls["render"] == 1
        assert calls["apply"] == 0

    def test_update_selected_datasets_syncs_selected_runs_to_active_vector_axis(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(7601, with_grouping=False)
        ds2 = _make_dataset(7602, with_grouping=False)

        for dataset in (ds1, ds2):
            assert dataset.run is not None
            dataset.run.grouping.update(
                {
                    "groups": {
                        1: [1],
                        2: [2],
                        3: [1],
                        4: [2],
                        5: [1],
                        6: [2],
                    },
                    "group_names": {
                        1: "Pz Forward",
                        2: "Pz Backward",
                        3: "Py Top",
                        4: "Py Bottom",
                        5: "Px Left",
                        6: "Px Right",
                    },
                    "forward_group": 1,
                    "backward_group": 2,
                    "vector_axis": "P_z",
                }
            )

        mainwindow._data_browser.get_selected_datasets = lambda: [ds1, ds2]
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: "P_x"

        monkeypatch.setattr(mainwindow._plot_panel, "plot_datasets", lambda _datasets: None)
        if hasattr(mainwindow._plot_panel, "set_active_label_group"):
            mainwindow._plot_panel.set_active_label_group = lambda _gid: None
        if hasattr(mainwindow._data_browser, "get_selected_group_ids"):
            mainwindow._data_browser.get_selected_group_ids = lambda: []
        mainwindow._fit_panel.set_datasets = lambda _datasets: None

        mainwindow._update_selected_datasets()

        assert ds1.run is not None and ds2.run is not None
        assert ds1.run.grouping.get("vector_axis") == "P_x"
        assert ds2.run.grouping.get("vector_axis") == "P_x"

    def test_render_current_selection_uses_most_recent_dataset_when_overlay_disabled(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(7651, with_grouping=False)
        ds2 = _make_dataset(7652, with_grouping=False)

        mainwindow._data_browser.get_selected_datasets = lambda: [ds1, ds2]
        if hasattr(mainwindow._data_browser, "get_selected_group_ids"):
            mainwindow._data_browser.get_selected_group_ids = lambda: []
        if hasattr(mainwindow._data_browser, "is_single_group_selected"):
            mainwindow._data_browser.is_single_group_selected = lambda: False
        if hasattr(mainwindow._data_browser, "get_current_dataset"):
            mainwindow._data_browser.get_current_dataset = lambda: ds2
        if hasattr(mainwindow._plot_panel, "is_overlay_enabled"):
            mainwindow._plot_panel.is_overlay_enabled = lambda: False
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: None

        plotted: list[int] = []
        monkeypatch.setattr(
            mainwindow._plot_panel, "plot_datasets", lambda _datasets: plotted.append(-1)
        )
        monkeypatch.setattr(
            mainwindow._plot_panel,
            "plot_dataset",
            lambda dataset: plotted.append(int(dataset.run_number)),
        )

        mainwindow._render_current_selection_plot()

        assert plotted == [7652]
        assert mainwindow._current_dataset is ds2

    def test_render_current_selection_group_fallback_when_overlay_disabled(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(7661, with_grouping=False)
        ds2 = _make_dataset(7662, with_grouping=False)
        ds3 = _make_dataset(7663, with_grouping=False)

        mainwindow._data_browser.get_selected_datasets = lambda: [ds1, ds2]
        if hasattr(mainwindow._data_browser, "get_selected_group_ids"):
            mainwindow._data_browser.get_selected_group_ids = lambda: ["g1"]
        if hasattr(mainwindow._data_browser, "is_single_group_selected"):
            mainwindow._data_browser.is_single_group_selected = lambda: True
        if hasattr(mainwindow._data_browser, "get_group_member_run_numbers"):
            mainwindow._data_browser.get_group_member_run_numbers = lambda _gid: [7661, 7662]
        if hasattr(mainwindow._data_browser, "get_current_dataset"):
            mainwindow._data_browser.get_current_dataset = lambda: None
        if hasattr(mainwindow._plot_panel, "is_overlay_enabled"):
            mainwindow._plot_panel.is_overlay_enabled = lambda: False
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: None

        plotted: list[int] = []
        monkeypatch.setattr(
            mainwindow._plot_panel, "plot_datasets", lambda _datasets: plotted.append(-1)
        )
        monkeypatch.setattr(
            mainwindow._plot_panel,
            "plot_dataset",
            lambda dataset: plotted.append(int(dataset.run_number)),
        )

        mainwindow._current_dataset = ds2
        mainwindow._render_current_selection_plot()

        mainwindow._current_dataset = ds3
        mainwindow._render_current_selection_plot()

        assert plotted == [7662, 7661]

    def test_dataset_selection_preserves_active_vector_axis(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_dataset(7701, with_grouping=False)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "groups": {
                    1: [1],
                    2: [2],
                    3: [1],
                    4: [2],
                    5: [1],
                    6: [2],
                },
                "group_names": {
                    1: "Pz Forward",
                    2: "Pz Backward",
                    3: "Py Top",
                    4: "Py Bottom",
                    5: "Px Left",
                    6: "Px Right",
                },
                "forward_group": 1,
                "backward_group": 2,
                "vector_axis": "P_z",
            }
        )

        mainwindow._data_browser.get_dataset = lambda _run_number: dataset
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: "P_x"

        calls = {"render": 0}
        monkeypatch.setattr(
            mainwindow,
            "_render_current_selection_plot",
            lambda: calls.__setitem__("render", calls["render"] + 1),
        )
        mainwindow._fit_panel.set_dataset = lambda _dataset: None

        mainwindow._on_dataset_selected(7701)

        assert dataset.run.grouping.get("vector_axis") == "P_x"
        assert calls["render"] == 1

    def test_dataset_selection_preserves_all_mode_rendering(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _make_dataset(7702, with_grouping=False)
        assert dataset.run is not None
        dataset.run.grouping.update(
            {
                "groups": {
                    1: [1],
                    2: [2],
                    3: [1],
                    4: [2],
                    5: [1],
                    6: [2],
                },
                "group_names": {
                    1: "Pz Forward",
                    2: "Pz Backward",
                    3: "Py Top",
                    4: "Py Bottom",
                    5: "Px Left",
                    6: "Px Right",
                },
                "forward_group": 1,
                "backward_group": 2,
                "vector_axis": "P_z",
            }
        )

        mainwindow._data_browser.get_dataset = lambda _run_number: dataset
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: "ALL"

        calls = {"render": 0}
        monkeypatch.setattr(
            mainwindow,
            "_render_current_selection_plot",
            lambda: calls.__setitem__("render", calls["render"] + 1),
        )
        mainwindow._fit_panel.set_dataset = lambda _dataset: None

        mainwindow._on_dataset_selected(7702)

        assert dataset.run.grouping.get("vector_axis") == "P_z"
        assert calls["render"] == 1

    def test_update_selected_datasets_blocks_fit_actions_in_all_mode(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(7711, with_grouping=False)
        ds2 = _make_dataset(7712, with_grouping=False)
        for dataset in (ds1, ds2):
            assert dataset.run is not None
            dataset.run.grouping.update(
                {
                    "groups": {
                        1: [1],
                        2: [2],
                        3: [1],
                        4: [2],
                        5: [1],
                        6: [2],
                    },
                    "group_names": {
                        1: "Pz Forward",
                        2: "Pz Backward",
                        3: "Py Top",
                        4: "Py Bottom",
                        5: "Px Left",
                        6: "Px Right",
                    },
                    "forward_group": 1,
                    "backward_group": 2,
                    "vector_axis": "P_z",
                }
            )

        mainwindow._current_dataset = ds1
        mainwindow._data_browser.get_selected_datasets = lambda: [ds1, ds2]
        mainwindow._data_browser.get_dataset = lambda run_number: (
            ds1 if int(run_number) == 7711 else ds2
        )
        if hasattr(mainwindow._data_browser, "get_selected_group_ids"):
            mainwindow._data_browser.get_selected_group_ids = lambda: []
        if hasattr(mainwindow._plot_panel, "set_active_label_group"):
            mainwindow._plot_panel.set_active_label_group = lambda _gid: None

        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: None)
        monkeypatch.setattr(mainwindow, "_refresh_vector_axis_selector", lambda: None)

        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: "ALL"
        mainwindow._update_selected_datasets()

        assert not mainwindow._fit_panel._single_tab._fit_btn.isEnabled()
        assert not mainwindow._fit_panel._single_tab._preview_btn.isEnabled()
        assert not mainwindow._fit_panel._global_tab._fit_btn.isEnabled()
        assert "Vector All mode" in mainwindow._fit_panel._single_tab._fit_btn.toolTip()
        assert "x, y, or z" in mainwindow._fit_panel._single_tab._fit_btn.toolTip()

        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: "P_x"
        monkeypatch.setattr(mainwindow, "_synchronize_targets_to_axis", lambda *_args, **_kwargs: 0)
        mainwindow._update_selected_datasets()

        assert mainwindow._fit_panel._single_tab._fit_btn.isEnabled()
        assert mainwindow._fit_panel._single_tab._preview_btn.isEnabled()
        assert mainwindow._fit_panel._global_tab._fit_btn.isEnabled()

    def test_run_info_inclusion_handler_updates_data_browser(self, mainwindow: MainWindow) -> None:
        """Run Info include/exclude signal should add/remove data-browser columns."""
        calls: list[tuple[str, str]] = []

        mainwindow._data_browser.add_extra_column = lambda key: calls.append(("add", key))
        mainwindow._data_browser.remove_extra_column = lambda key: calls.append(("remove", key))

        mainwindow._on_run_info_field_inclusion_changed("run_info.points", True)
        mainwindow._on_run_info_field_inclusion_changed("run_info.points", False)

        assert calls == [("add", "run_info.points"), ("remove", "run_info.points")]

    def test_options_temperature_log_toggle_updates_data_browser(
        self, mainwindow: MainWindow
    ) -> None:
        """Options toggle should switch fixed temperature column to log mean and back."""
        dataset = _make_dataset(6101, with_grouping=True)
        dataset.metadata.update(
            {
                "title": "temperature scan",
                "temperature": 50.0,
                "field": 100.0,
                "comment": "",
                "nexus_time_series": {
                    "musrroot_slow_control/Sample Temperature": {
                        "units": "K",
                        "time": [0.0, 10.0],
                        "values": [4.8, 5.2],
                        "mean": 5.0,
                        "min": 4.8,
                        "max": 5.2,
                    }
                },
            }
        )
        if dataset.run is not None:
            dataset.run.metadata.update(dataset.metadata)
        mainwindow._data_browser.add_dataset(dataset)

        assert mainwindow._data_browser._table.item(0, 2).text() == "50.00"

        mainwindow._use_temperature_from_log_action.setChecked(True)

        assert mainwindow._data_browser._table.item(0, 2).text() == "5.00"
        assert "temperature" in mainwindow._data_browser.get_extra_columns()

        mainwindow._use_temperature_from_log_action.setChecked(False)

        assert mainwindow._data_browser._table.item(0, 2).text() == "50.00"
        assert "temperature" not in mainwindow._data_browser.get_extra_columns()

    def test_run_info_temperature_inclusion_syncs_options_action(
        self, mainwindow: MainWindow
    ) -> None:
        """Get Info temperature checkbox should stay in sync with Options."""
        mainwindow._on_run_info_field_inclusion_changed("temperature", True)
        assert mainwindow._use_temperature_from_log_action.isChecked()

        mainwindow._on_run_info_field_inclusion_changed("temperature", False)
        assert not mainwindow._use_temperature_from_log_action.isChecked()

    def test_run_info_temperature_inclusion_overrides_single_dataset(
        self, mainwindow: MainWindow
    ) -> None:
        """A run-specific Get Info change should not alter the global option."""
        ds1 = _make_dataset(6111, with_grouping=True)
        ds1.metadata.update(
            {
                "title": "run one",
                "temperature": 50.0,
                "field": 100.0,
                "comment": "",
                "nexus_time_series": {
                    "musrroot_slow_control/Sample Temperature": {
                        "units": "K",
                        "time": [0.0, 10.0],
                        "values": [4.8, 5.2],
                        "mean": 5.0,
                        "min": 4.8,
                        "max": 5.2,
                    }
                },
            }
        )
        ds2 = _make_dataset(6112, with_grouping=True)
        ds2.metadata.update(
            {
                "title": "run two",
                "temperature": 60.0,
                "field": 100.0,
                "comment": "",
                "nexus_time_series": {
                    "musrroot_slow_control/Sample Temperature": {
                        "units": "K",
                        "time": [0.0, 10.0],
                        "values": [6.8, 7.2],
                        "mean": 7.0,
                        "min": 6.8,
                        "max": 7.2,
                    }
                },
            }
        )
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)

        mainwindow._use_temperature_from_log_action.setChecked(True)

        assert mainwindow._data_browser._table.item(0, 2).text() == "5.00"
        assert mainwindow._data_browser._table.item(1, 2).text() == "7.00"
        assert mainwindow._run_info_included_fields_for_dataset(6111) >= {"temperature"}
        assert mainwindow._run_info_included_fields_for_dataset(6112) >= {"temperature"}

        mainwindow._on_run_info_field_inclusion_changed(
            "temperature",
            False,
            run_number=6111,
        )

        assert mainwindow._use_temperature_from_log_action.isChecked()
        assert mainwindow._data_browser._table.item(0, 2).text() == "50.00"
        assert mainwindow._data_browser._table.item(1, 2).text() == "7.00"
        assert "temperature" not in mainwindow._run_info_included_fields_for_dataset(6111)
        assert "temperature" in mainwindow._run_info_included_fields_for_dataset(6112)

        mainwindow._use_temperature_from_log_action.setChecked(False)

        assert mainwindow._data_browser._table.item(0, 2).text() == "50.00"
        assert mainwindow._data_browser._table.item(1, 2).text() == "60.00"
        assert "temperature" not in mainwindow._run_info_included_fields_for_dataset(6111)
        assert "temperature" not in mainwindow._run_info_included_fields_for_dataset(6112)

        mainwindow._on_run_info_field_inclusion_changed(
            "temperature",
            True,
            run_number=6112,
        )

        assert not mainwindow._use_temperature_from_log_action.isChecked()
        assert mainwindow._data_browser._table.item(0, 2).text() == "50.00"
        assert mainwindow._data_browser._table.item(1, 2).text() == "7.00"
        assert "temperature" not in mainwindow._run_info_included_fields_for_dataset(6111)
        assert "temperature" in mainwindow._run_info_included_fields_for_dataset(6112)

    def test_cross_group_completion_shows_global_parameter_window(
        self, mainwindow: MainWindow
    ) -> None:
        """Accepted cross-group fit should open and focus the global-fit result window."""
        model = ParameterCompositeModel(["Linear"])
        fit_result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            message="Fit successful",
        )
        groups = [
            ParameterGroupData(
                group_id="g0",
                group_name="G0",
                x=np.array([1.0, 2.0], dtype=float),
                y=np.array([0.1, 0.2], dtype=float),
                yerr=np.array([0.01, 0.01], dtype=float),
                group_variable_value=1.0,
            )
        ]
        output = SimpleNamespace(model=model, fit_result=fit_result)

        mainwindow._on_cross_group_fit_completed("Lambda", groups, output)

        assert mainwindow._global_parameter_fit_window is not None
        assert mainwindow._global_parameter_fit_window.isVisible()

    def test_fit_parameters_delete_group_handler_clears_matching_run_fits(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Deleting a fit-parameter group should clear fit data for its runs."""
        captured: dict[str, list[int]] = {"fit": [], "plot": []}

        mainwindow._fit_panel.clear_fits_for_runs = lambda runs: (
            captured["fit"].extend(runs) or len(runs)
        )
        mainwindow._plot_panel.clear_fits_for_runs = lambda runs: (
            captured["plot"].extend(runs) or len(runs)
        )

        mainwindow._on_fit_parameters_group_fits_deleted("g1", [101, "102", 101, "bad"])

        assert captured["fit"] == [101, 102]
        assert captured["plot"] == [101, 102]

    def test_load_files_auto_applies_existing_grouping(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Newly loaded datasets should inherit existing project grouping settings."""
        existing = _make_dataset(7001, with_grouping=True)
        incoming = _make_dataset(7002, with_grouping=False)
        mainwindow._data_browser.add_dataset(existing)

        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")

        applied_payloads: list[dict] = []

        def _stub_apply(dataset, payload):
            assert int(dataset.run_number) == 7002
            applied_payloads.append(payload)
            return True, False

        monkeypatch.setattr(mainwindow, "_apply_grouping_settings_to_dataset", _stub_apply)

        mainwindow._load_files(["/tmp/new_run.nxs"])

        assert len(applied_payloads) == 1
        assert applied_payloads[0]["forward_group"] == 1
        assert applied_payloads[0]["backward_group"] == 2

    def test_load_files_auto_applies_existing_manual_deadtime_grouping(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing = _make_dataset(7003, with_grouping=True)
        incoming = _make_dataset(7004, with_grouping=False)
        assert existing.run is not None
        existing.run.grouping.update(
            {
                "deadtime_correction": True,
                "deadtime_mode": "manual",
                "deadtime_method": "manual",
                "deadtime_manual_us": 0.02,
                "dead_time_us": [0.02, 0.02],
            }
        )
        mainwindow._data_browser.add_dataset(existing)

        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")

        applied_payloads: list[dict] = []

        def _stub_apply(dataset, payload):
            assert int(dataset.run_number) == 7004
            applied_payloads.append(payload)
            return True, False

        monkeypatch.setattr(mainwindow, "_apply_grouping_settings_to_dataset", _stub_apply)

        mainwindow._load_files(["/tmp/new_run_with_manual_deadtime.nxs"])

        assert len(applied_payloads) == 1
        assert applied_payloads[0]["deadtime_mode"] == "manual"
        assert applied_payloads[0]["deadtime_method"] == "manual"
        assert applied_payloads[0]["dead_time_us"] == pytest.approx([0.02, 0.02])

    def test_load_files_auto_applies_grouping_from_highest_run_dataset(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auto-grouping should use the highest run number already in the browser."""
        low_run = _make_dataset(7001, with_grouping=True)
        high_run = _make_dataset(7010, with_grouping=True)
        incoming = _make_dataset(7020, with_grouping=False)

        assert low_run.run is not None
        low_run.run.grouping.update(
            {
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
            }
        )
        assert high_run.run is not None
        high_run.run.grouping.update(
            {
                "groups": {5: [1], 6: [2]},
                "forward_group": 5,
                "backward_group": 6,
                "alpha": 2.5,
            }
        )

        mainwindow._data_browser.add_dataset(low_run)
        mainwindow._data_browser.add_dataset(high_run)
        mainwindow._current_dataset = low_run

        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")

        applied_payloads: list[dict] = []

        def _stub_apply(dataset, payload):
            applied_payloads.append(payload)
            return True, False

        monkeypatch.setattr(mainwindow, "_apply_grouping_settings_to_dataset", _stub_apply)

        mainwindow._load_files(["/tmp/new_run_uses_highest_grouping.nxs"])

        assert len(applied_payloads) == 1
        assert applied_payloads[0]["forward_group"] == 5
        assert applied_payloads[0]["backward_group"] == 6
        assert applied_payloads[0]["alpha"] == pytest.approx(2.5)

    def test_load_files_does_not_auto_apply_grouping_without_template(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No grouping should be auto-applied when project has no grouping template."""
        incoming = _make_dataset(7101, with_grouping=False)

        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")

        call_count = {"n": 0}

        def _stub_apply(_dataset, _payload):
            call_count["n"] += 1
            return True, False

        monkeypatch.setattr(mainwindow, "_apply_grouping_settings_to_dataset", _stub_apply)

        mainwindow._load_files(["/tmp/new_run_no_template.nxs"])

        assert call_count["n"] == 0

    def test_load_files_preserves_existing_fit_curves(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adding a new run should not clear fits stored for existing runs."""
        existing = _make_dataset(7201, with_grouping=False)
        incoming = _make_dataset(7202, with_grouping=False)

        mainwindow._data_browser.add_dataset(existing)
        mainwindow._plot_panel.plot_dataset(existing)

        t_fit = np.array([0.0, 0.01, 0.02, 0.03], dtype=float)
        y_fit = np.array([0.1, 0.09, 0.08, 0.07], dtype=float)
        mainwindow._plot_panel.plot_fit(t_fit, y_fit, label="Fit")

        assert 7201 in mainwindow._plot_panel._fit_curves

        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")

        mainwindow._load_files(["/tmp/new_run_keeps_fits.nxs"])

        assert 7201 in mainwindow._plot_panel._fit_curves
        stored_t, stored_y, stored_label = mainwindow._plot_panel._fit_curves[7201]
        np.testing.assert_allclose(stored_t, t_fit)
        np.testing.assert_allclose(stored_y, y_fit)
        assert stored_label == "Fit"

    def test_load_files_duplicate_path_no_keeps_existing_dataset(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Choosing No on duplicate-path prompt should skip reloading that file."""
        existing = _make_dataset(7301, with_grouping=False)
        existing.run.source_file = "/tmp/duplicate_no.nxs"
        incoming = _make_dataset(7301, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_no.nxs"
        incoming.metadata["comment"] = "NEW"

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_a, **_k: QMessageBox.StandardButton.No,
        )

        mainwindow._load_files(["/tmp/duplicate_no.nxs"])

        result = mainwindow._data_browser.get_dataset(7301)
        assert result is not None
        assert result.metadata.get("comment") != "NEW"

    def test_load_files_duplicate_path_yes_replaces_existing_dataset(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Choosing Yes on duplicate-path prompt should replace matching datasets."""
        existing = _make_dataset(7302, with_grouping=False)
        existing.run.source_file = "/tmp/duplicate_yes.nxs"
        incoming = _make_dataset(7302, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_yes.nxs"
        incoming.metadata["comment"] = "NEW"

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_a, **_k: QMessageBox.StandardButton.Yes,
        )

        mainwindow._load_files(["/tmp/duplicate_yes.nxs"])

        result = mainwindow._data_browser.get_dataset(7302)
        assert result is not None
        assert result.metadata.get("comment") == "NEW"

    def test_load_files_duplicate_path_yes_to_all_applies_to_rest(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Yes to All should prompt once and overwrite all duplicate files in batch."""
        first_existing = _make_dataset(7303, with_grouping=False)
        first_existing.run.source_file = "/tmp/duplicate_all_1.nxs"
        second_existing = _make_dataset(7304, with_grouping=False)
        second_existing.run.source_file = "/tmp/duplicate_all_2.nxs"
        mainwindow._data_browser.add_dataset(first_existing)
        mainwindow._data_browser.add_dataset(second_existing)

        first_incoming = _make_dataset(7303, with_grouping=False)
        first_incoming.run.source_file = "/tmp/duplicate_all_1.nxs"
        first_incoming.metadata["comment"] = "NEW1"
        second_incoming = _make_dataset(7304, with_grouping=False)
        second_incoming.run.source_file = "/tmp/duplicate_all_2.nxs"
        second_incoming.metadata["comment"] = "NEW2"

        def _stub_load_file(path: str):
            if path.endswith("duplicate_all_1.nxs"):
                return first_incoming
            if path.endswith("duplicate_all_2.nxs"):
                return second_incoming
            raise AssertionError(f"Unexpected path: {path}")

        prompt_calls = {"n": 0}

        def _stub_question(*_a, **_k):
            prompt_calls["n"] += 1
            return QMessageBox.StandardButton.YesToAll

        monkeypatch.setattr(mainwindow, "_load_file", _stub_load_file)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(QMessageBox, "question", _stub_question)

        mainwindow._load_files(
            [
                "/tmp/duplicate_all_1.nxs",
                "/tmp/duplicate_all_2.nxs",
            ]
        )

        assert prompt_calls["n"] == 1
        first_result = mainwindow._data_browser.get_dataset(7303)
        second_result = mainwindow._data_browser.get_dataset(7304)
        assert first_result is not None
        assert second_result is not None
        assert first_result.metadata.get("comment") == "NEW1"
        assert second_result.metadata.get("comment") == "NEW2"

    def test_load_files_duplicate_prompt_includes_run_number(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Duplicate-file prompt should show the already-loaded run number."""
        existing = _make_dataset(7305, with_grouping=False)
        existing.run.source_file = "/tmp/duplicate_message.nxs"
        incoming = _make_dataset(7305, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_message.nxs"

        captured = {"text": ""}

        def _stub_question(_parent, _title, text, *_args, **_kwargs):
            captured["text"] = text
            return QMessageBox.StandardButton.No

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(QMessageBox, "question", _stub_question)

        mainwindow._load_files(["/tmp/duplicate_message.nxs"])

        assert "Run number(s): 7305" in captured["text"]

    def test_grouping_apply_preserves_multi_plot_selection(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Applying grouping with multiple selected datasets should keep multi-plot view."""
        ds1 = _make_dataset(8101, with_grouping=False)
        ds2 = _make_dataset(8102, with_grouping=False)
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        mainwindow._current_dataset = ds1
        if hasattr(mainwindow._plot_panel, "is_overlay_enabled"):
            mainwindow._plot_panel.is_overlay_enabled = lambda: True
        if hasattr(mainwindow._plot_panel, "get_current_polarization_axis"):
            mainwindow._plot_panel.get_current_polarization_axis = lambda: None

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                pass

            def exec(self):
                return self.DialogCode.Accepted

            def get_grouping_result(self):
                return {
                    "run_numbers": [8101, 8102],
                    "groups": {1: [1], 2: [2]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 1.0,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 2,
                    "deadtime_correction": False,
                }

        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)
        monkeypatch.setattr(
            mainwindow,
            "_apply_grouping_settings_to_dataset",
            lambda _dataset, _payload: (True, False),
        )
        monkeypatch.setattr(mainwindow._data_browser, "_rebuild_table", lambda: None)
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [ds1, ds2])

        calls = {"multi": 0, "single": 0}
        monkeypatch.setattr(
            mainwindow._plot_panel,
            "plot_datasets",
            lambda _datasets: calls.__setitem__("multi", calls["multi"] + 1),
        )
        monkeypatch.setattr(
            mainwindow._plot_panel,
            "plot_dataset",
            lambda _dataset: calls.__setitem__("single", calls["single"] + 1),
        )

        mainwindow._open_shared_grouping_dialog(selected_run_number=8101)

        assert calls["multi"] == 1
        assert calls["single"] == 0

    def test_grouping_apply_vector_alphas_to_multiple_selected_runs(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Vector alpha_x/y/z values should be applied to all selected target runs."""
        ds1 = _make_dataset(8111, with_grouping=False)
        ds2 = _make_dataset(8112, with_grouping=False)
        ds3 = _make_dataset(8113, with_grouping=False)
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        mainwindow._data_browser.add_dataset(ds3)
        mainwindow._current_dataset = ds1

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                pass

            def exec(self):
                return self.DialogCode.Accepted

            def get_grouping_result(self):
                return {
                    "run_numbers": [8111, 8112],
                    "groups": {
                        1: [1],
                        2: [2],
                        3: [1],
                        4: [2],
                        5: [1],
                        6: [2],
                    },
                    "group_names": {
                        1: "Pz Forward",
                        2: "Pz Backward",
                        3: "Py Top",
                        4: "Py Bottom",
                        5: "Px Left",
                        6: "Px Right",
                    },
                    "forward_group": 1,
                    "backward_group": 2,
                    "vector_axis": "P_y",
                    "alpha": 1.0,
                    "alpha_x": 1.11,
                    "alpha_y": 1.22,
                    "alpha_z": 1.33,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                }

        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)
        monkeypatch.setattr(mainwindow._data_browser, "_rebuild_table", lambda: None)
        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: None)
        monkeypatch.setattr(mainwindow, "_refresh_vector_axis_selector", lambda: None)

        mainwindow._open_shared_grouping_dialog(selected_run_number=8111)

        assert ds1.run is not None and ds2.run is not None and ds3.run is not None
        assert ds1.run.grouping.get("alpha_x") == pytest.approx(1.11)
        assert ds1.run.grouping.get("alpha_y") == pytest.approx(1.22)
        assert ds1.run.grouping.get("alpha_z") == pytest.approx(1.33)
        assert ds2.run.grouping.get("alpha_x") == pytest.approx(1.11)
        assert ds2.run.grouping.get("alpha_y") == pytest.approx(1.22)
        assert ds2.run.grouping.get("alpha_z") == pytest.approx(1.33)

        # Unselected run should remain unchanged.
        assert "alpha_x" not in ds3.run.grouping
        assert "alpha_y" not in ds3.run.grouping
        assert "alpha_z" not in ds3.run.grouping

    def test_on_grouping_current_passes_selected_runs_to_dialog(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Grouping launch should preselect currently highlighted datasets."""
        ds1 = _make_dataset(8201, with_grouping=False)
        ds2 = _make_dataset(8202, with_grouping=False)

        mainwindow._current_dataset = ds1
        monkeypatch.setattr(mainwindow._data_browser, "get_all_datasets", lambda: [ds1, ds2])
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [ds1, ds2])

        captured = {"selected_run_numbers": None}

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **kwargs):
                captured["selected_run_numbers"] = kwargs.get("selected_run_numbers")

            def exec(self):
                return 0

        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)

        mainwindow._on_grouping_current()

        assert captured["selected_run_numbers"] == [8201, 8202]

    def test_grouping_current_for_combined_dataset_uses_hidden_sources(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(8401, with_grouping=True)
        ds2 = _make_dataset(8402, with_grouping=True)
        assert ds1.run is not None and ds2.run is not None
        ds1.run.source_file = "/tmp/run_8401.nxs"
        ds2.run.source_file = "/tmp/run_8402.nxs"
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        combined_rn = mainwindow._data_browser.add_combined_dataset([8401, 8402])
        assert combined_rn is not None
        combined_ds = mainwindow._data_browser.get_dataset(combined_rn)
        assert combined_ds is not None

        mainwindow._current_dataset = combined_ds
        monkeypatch.setattr(
            mainwindow._data_browser, "get_selected_datasets", lambda: [combined_ds]
        )

        captured = {
            "run_numbers": None,
            "selected_run_number": None,
            "selected_run_numbers": None,
        }

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, datasets, **kwargs):
                captured["run_numbers"] = [int(ds.run_number) for ds in datasets]
                captured["selected_run_number"] = kwargs.get("selected_run_number")
                captured["selected_run_numbers"] = kwargs.get("selected_run_numbers")

            def exec(self):
                return 0

        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)

        mainwindow._on_grouping_current()

        assert captured["run_numbers"] == [8401, 8402]
        assert captured["selected_run_number"] == 8401
        assert captured["selected_run_numbers"] == [8401, 8402]

    def test_grouping_apply_from_combined_dataset_updates_sources_and_rebuilds(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(8411, with_grouping=True)
        ds2 = _make_dataset(8412, with_grouping=True)
        assert ds1.run is not None and ds2.run is not None
        ds1.run.source_file = "/tmp/run_8411.nxs"
        ds2.run.source_file = "/tmp/run_8412.nxs"
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        combined_rn = mainwindow._data_browser.add_combined_dataset([8411, 8412])
        assert combined_rn is not None
        combined_ds = mainwindow._data_browser.get_dataset(combined_rn)
        assert combined_ds is not None

        mainwindow._current_dataset = combined_ds
        monkeypatch.setattr(
            mainwindow._data_browser, "get_selected_datasets", lambda: [combined_ds]
        )
        monkeypatch.setattr(mainwindow._data_browser, "_rebuild_table", lambda: None)
        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: None)
        monkeypatch.setattr(mainwindow, "_refresh_vector_axis_selector", lambda: None)

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                pass

            def exec(self):
                return self.DialogCode.Accepted

            def get_grouping_result(self):
                return {
                    "run_numbers": [8411, 8412],
                    "groups": {1: [1], 2: [2]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 2.5,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                }

        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)

        mainwindow._on_grouping_current()

        assert ds1.run.grouping["alpha"] == pytest.approx(2.5)
        assert ds2.run.grouping["alpha"] == pytest.approx(2.5)
        assert combined_ds.run is not None
        assert combined_ds.run.grouping["alpha"] == pytest.approx(2.5)
        assert mainwindow._current_dataset is combined_ds

        monkeypatch.setattr(
            mainwindow._data_browser, "_get_selected_run_numbers", lambda: [combined_rn]
        )
        mainwindow._data_browser._separate_combined()

        restored_1 = mainwindow._data_browser.get_dataset(8411)
        restored_2 = mainwindow._data_browser.get_dataset(8412)
        assert restored_1 is ds1
        assert restored_2 is ds2
        assert restored_1.run.grouping["alpha"] == pytest.approx(2.5)
        assert restored_2.run.grouping["alpha"] == pytest.approx(2.5)


class TestPerRunGoodFramesNormaliser:
    """``good_frames`` is the per-run dead-time normaliser and must never be
    dropped on regroup nor inherited from another run's grouping template.

    Regression coverage for the HIGH-severity bug where enabling dead-time
    correction made the time-domain asymmetry diverge and saturate at +-100%
    because a run lost (or inherited the wrong) ``good_frames``.
    """

    def test_file_deadtime_stays_bounded_when_good_frames_preserved(
        self, mainwindow: MainWindow
    ) -> None:
        # Run B carries its own large good_frames; a template from another run
        # supplies a much smaller value that would clip the correction.
        dataset = _make_deadtime_dataset(34998, good_frames=1_395_561.0)
        template_from_other_run = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 7,
            "bunching_factor": 1,
            "deadtime_correction": True,
            "deadtime_mode": "file",
            "good_frames": 11_299.0,  # a different run's normaliser
        }

        applied, deadtime_applied = mainwindow._apply_grouping_settings_to_dataset(
            dataset, template_from_other_run
        )

        assert applied is True
        assert deadtime_applied is True
        assert dataset.run is not None
        # The run keeps its own normaliser, not the template's.
        assert dataset.run.grouping["good_frames"] == pytest.approx(1_395_561.0)
        # Asymmetry is the physical 60% (20000 vs 5000), not a +-100% saturation.
        max_abs = float(np.max(np.abs(dataset.asymmetry)))
        assert max_abs < 90.0
        assert max_abs == pytest.approx(60.0, abs=1.0)

    def test_apply_grouping_does_not_overwrite_own_good_frames(
        self, mainwindow: MainWindow
    ) -> None:
        # Loading run B after run A: A's template must not clobber B's value.
        run_a = _make_deadtime_dataset(34997, good_frames=11_299.0)
        run_b = _make_deadtime_dataset(34998, good_frames=1_395_561.0)

        template = mainwindow._extract_grouping_overrides(run_a)
        assert template["good_frames"] == pytest.approx(11_299.0)

        mainwindow._apply_grouping_settings_to_dataset(run_b, template)

        assert run_b.run is not None
        assert run_b.run.grouping["good_frames"] == pytest.approx(1_395_561.0)

    def test_auto_grouping_template_strips_per_run_normalisers(
        self, mainwindow: MainWindow
    ) -> None:
        # The cross-run auto-grouping payload must not carry per-run scalars.
        template = mainwindow._extract_grouping_overrides(
            _make_deadtime_dataset(34997, good_frames=11_299.0)
        )
        payload = dict(template)
        for key in mainwindow._PER_RUN_NORMALISER_KEYS:
            payload.pop(key, None)

        assert "good_frames" not in payload
        # Group definitions and deadtime mode remain shareable template fields.
        assert payload["groups"] == {1: [1], 2: [2]}
        assert payload["deadtime_mode"] == "file"

    def test_period_run_keeps_per_period_good_frames_through_regroup(
        self, mainwindow: MainWindow
    ) -> None:
        dataset = _make_two_period_vector_dataset(103277)
        dataset.run.grouping["period_good_frames"] = [28108.0, 27950.0]
        dataset.run.grouping["good_frames"] = 28108.0

        regroup_payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "period_mode": str(mw_module.PeriodMode.RED),
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, regroup_payload)

        assert applied is True
        assert dataset.run is not None
        # Per-period normalisers survive the grouping round-trip.
        assert dataset.run.grouping["period_good_frames"] == pytest.approx([28108.0, 27950.0])
        assert dataset.run.grouping["good_frames"] == pytest.approx(28108.0)


class TestPlotWorkspaceDomainPhase7:
    """Phase 7 — toolbar-driven domain switching."""

    def test_groups_button_disabled_when_no_grouped_data(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Individual-groups toolbar button is disabled when no grouped data exists."""
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda _dataset=None: [],
        )
        mainwindow._refresh_time_view_selector()
        assert not mainwindow._domain_buttons[1].isEnabled()

    def test_groups_button_enabled_when_grouped_data_present(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Individual-groups toolbar button is enabled when grouped data is available."""
        monkeypatch.setattr(
            mainwindow,
            "_grouped_time_domain_display_datasets",
            lambda dataset=None: [dataset or mainwindow._current_dataset],
        )
        time_arr = np.linspace(0, 8, 100)
        asym = np.zeros(100)
        err = np.ones(100) * 0.01
        run = Run(run_number=42, grouping={}, metadata={})
        ds = MuonDataset(time=time_arr, asymmetry=asym, error=err, run=run, metadata={})
        mainwindow._current_dataset = ds
        mainwindow._refresh_time_view_selector()
        assert mainwindow._domain_buttons[1].isEnabled()

    def test_no_signal_on_same_view_click(self, mainwindow: MainWindow, qapp: QApplication) -> None:
        """Selecting the already-active view emits no active_view_changed signal."""
        emitted: list[str] = []
        mainwindow._plot_workspace.active_view_changed.connect(emitted.append)
        current = mainwindow._plot_workspace.active_view()
        mainwindow._plot_workspace.set_active_view(current)
        mainwindow._plot_workspace.active_view_changed.disconnect(emitted.append)
        assert emitted == []

    def test_view_changed_emits_both_signals_on_domain_hop(
        self, mainwindow: MainWindow, qapp: QApplication
    ) -> None:
        """fb_asymmetry → frequency fires both active_view_changed and active_domain_changed."""
        mainwindow._plot_workspace.set_active_view("fb_asymmetry")
        view_emissions: list[str] = []
        domain_emissions: list[str] = []
        mainwindow._plot_workspace.active_view_changed.connect(view_emissions.append)
        mainwindow._plot_workspace.active_domain_changed.connect(domain_emissions.append)

        mainwindow._plot_workspace.set_active_view("frequency")

        mainwindow._plot_workspace.active_view_changed.disconnect(view_emissions.append)
        mainwindow._plot_workspace.active_domain_changed.disconnect(domain_emissions.append)
        assert view_emissions == ["frequency"]
        assert domain_emissions == ["frequency"]

    def test_same_domain_hop_no_domain_signal(
        self, mainwindow: MainWindow, qapp: QApplication
    ) -> None:
        """fb_asymmetry → groups (both time domain) does not emit active_domain_changed."""
        mainwindow._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mainwindow._plot_workspace.set_active_view("fb_asymmetry")
        view_emissions: list[str] = []
        domain_emissions: list[str] = []
        mainwindow._plot_workspace.active_view_changed.connect(view_emissions.append)
        mainwindow._plot_workspace.active_domain_changed.connect(domain_emissions.append)

        mainwindow._plot_workspace.set_active_view("groups")

        mainwindow._plot_workspace.active_view_changed.disconnect(view_emissions.append)
        mainwindow._plot_workspace.active_domain_changed.disconnect(domain_emissions.append)
        assert view_emissions == ["groups"]
        assert domain_emissions == []

    def test_state_round_trip(self, mainwindow: MainWindow) -> None:
        """Workspace state serialises and restores correctly for each view."""
        from asymmetry.gui.panels.plot_workspace_panel import PlotWorkspacePanel

        for view in ("fb_asymmetry", "frequency"):
            mainwindow._plot_workspace.set_active_view(view)
            state = mainwindow._plot_workspace.get_state()
            assert state["active_view"] == view

            dummy_workspace = PlotWorkspacePanel(
                time_panel=mainwindow._plot_panel,
                frequency_panel=mainwindow._frequency_plot_panel,
            )
            dummy_workspace.restore_state(state)
            assert dummy_workspace.active_view() == view


class TestBackgroundModesReduction:
    """data-reduction-parity Phase 2: tail-fit and reference-run modes."""

    @staticmethod
    def _long_run_dataset(run_number: int, *, scale: float = 1.0) -> MuonDataset:
        rng = np.random.default_rng(run_number)
        n, t0, width = 600, 5, 0.016
        t = (np.arange(n) - t0) * width
        intensity = 4000.0 * np.exp(-np.clip(t, 0.0, None) / 2.1969811) * (t >= 0) + 2.0
        counts_f = rng.poisson(np.clip(intensity * width * scale, 0.0, None)).astype(float)
        counts_b = rng.poisson(np.clip(intensity * width * scale * 0.8, 0.0, None)).astype(float)
        run = Run(
            run_number=run_number,
            histograms=[
                Histogram(counts=counts_f, bin_width=width, t0_bin=t0),
                Histogram(counts=counts_b, bin_width=width, t0_bin=t0),
            ],
            metadata={"run_number": run_number, "facility": "ISIS"},
            grouping={"good_frames": 1000.0 * scale},
        )
        time = (np.arange(n) - t0) * width
        return MuonDataset(
            time=time,
            asymmetry=np.zeros(n),
            error=np.full(n, 0.01),
            metadata={"run_number": run_number, "facility": "ISIS"},
            run=run,
        )

    def _payload(self, **extra) -> dict:
        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 5,
            "first_good_bin": 8,
            "last_good_bin": 599,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "background_correction": True,
        }
        payload.update(extra)
        return payload

    def test_tail_fit_mode_applies_on_pulsed_data(self, mainwindow: MainWindow) -> None:
        dataset = self._long_run_dataset(8101)
        payload = self._payload(background_mode="tail_fit")

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert dataset.run.grouping["background_correction"] is True
        assert dataset.run.grouping["background_mode"] == "tail_fit"
        assert dataset.run.grouping["background_method"] == "tail_fit"
        details = dataset.run.grouping["background_details"]
        # True flat rate is 2.0 counts/us in each group's source intensity.
        assert details["forward_rate_per_us"] == pytest.approx(
            2.0, abs=4 * (details["forward_rate_error_per_us"] or 1.0)
        )
        assert np.all(np.isfinite(dataset.asymmetry))

    def test_reference_run_mode_subtracts_loaded_dataset(self, mainwindow: MainWindow) -> None:
        sample = self._long_run_dataset(8102)
        reference = MuonDataset(
            time=sample.time.copy(),
            asymmetry=np.zeros_like(sample.time),
            error=np.full_like(sample.time, 0.01),
            metadata={"run_number": 8103, "facility": "ISIS"},
            run=Run(
                run_number=8103,
                histograms=[
                    Histogram(counts=h.counts.copy(), bin_width=h.bin_width, t0_bin=h.t0_bin)
                    for h in sample.run.histograms
                ],
                metadata={"run_number": 8103, "facility": "ISIS"},
                grouping={"good_frames": 1000.0},
            ),
        )
        mainwindow._data_browser.add_dataset(reference)
        payload = self._payload(
            background_mode="reference_run",
            background_run={"run_number": 8103, "source_file": ""},
        )

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(sample, payload)

        assert applied is True
        assert sample.run.grouping["background_method"] == "reference_run"
        assert sample.run.grouping["background_details"]["scale"] == pytest.approx(1.0)
        assert sample.run.grouping["background_run"]["run_number"] == 8103
        # Identical reference at scale 1 means full self-subtraction: the
        # corrected counts are zero, so the asymmetry collapses to zero.
        good = slice(8, 600)
        np.testing.assert_allclose(sample.asymmetry[good], 0.0)

    def test_reference_run_mode_without_resolvable_reference_skips(
        self, mainwindow: MainWindow
    ) -> None:
        dataset = self._long_run_dataset(8104)
        payload = self._payload(
            background_mode="reference_run",
            background_run={"run_number": 999999, "source_file": ""},
        )

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        # Reduction proceeds without subtraction; method records the miss.
        assert dataset.run.grouping.get("background_method") == "missing_reference"


class TestReductionSettingsPersistence:
    """Review fix: the new grouping keys survive non-dialog apply flows and
    round-trip through the project extractor."""

    NEW_KEYS = {
        "background_mode": "tail_fit",
        "excluded_detectors": [2],
        "binning_mode": "variable",
        "bin0_us": 0.1,
        "bin10_us": 0.5,
        "alpha_method": "diamagnetic",
        "alpha_error": 0.01,
        "alpha_reference_run": 9001,
    }

    def _configured_dataset(self) -> MuonDataset:
        dataset = _make_dataset(9001, with_grouping=True)
        counts = dataset.run.histograms[0].counts
        dataset.run.histograms = [Histogram(counts=counts.copy(), bin_width=0.01) for _ in range(4)]
        dataset.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
        dataset.run.grouping.update(self.NEW_KEYS)
        dataset.run.grouping["background_correction"] = True
        dataset.run.grouping["background_run"] = {"run_number": 9002, "source_file": "/x.nxs"}
        dataset.run.grouping["period_mapping"] = {"1": "red", "2": "green"}
        return dataset

    def test_extractor_emits_the_new_keys(self, mainwindow: MainWindow) -> None:
        dataset = self._configured_dataset()
        payload = mainwindow._extract_grouping_overrides(dataset)
        for key, value in self.NEW_KEYS.items():
            assert payload[key] == value, key
        assert payload["background_run"]["run_number"] == 9002
        assert payload["period_mapping"] == {"1": "red", "2": "green"}

    def test_extracted_payload_reapplies_without_erasing_settings(
        self, mainwindow: MainWindow
    ) -> None:
        """The toolbar/vector/project flows apply extracted payloads; the
        settings must survive the round trip."""
        dataset = self._configured_dataset()
        payload = mainwindow._extract_grouping_overrides(dataset)
        payload["bunching_factor"] = 7  # what the toolbar flow changes

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        grouping = dataset.run.grouping
        assert grouping["excluded_detectors"] == [2]
        assert grouping["binning_mode"] == "variable"
        assert grouping["bin0_us"] == pytest.approx(0.1)
        assert grouping["background_mode"] == "tail_fit"
        assert grouping["background_run"]["run_number"] == 9002
        assert grouping["alpha_method"] == "diamagnetic"
        assert grouping["bunching_factor"] == 7

    def test_dialog_payload_without_keys_still_clears_them(self, mainwindow: MainWindow) -> None:
        """The pop-when-absent contract stays for dialog payloads, which
        always carry the keys they own — absence means 'cleared'."""
        dataset = self._configured_dataset()
        payload = mainwindow._extract_grouping_overrides(dataset)
        for key in ("binning_mode", "bin0_us", "bin10_us"):
            payload.pop(key)

        mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert "binning_mode" not in dataset.run.grouping
