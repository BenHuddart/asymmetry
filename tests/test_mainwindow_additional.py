"""Additional tests for mainwindow functionality."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import numpy as np

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar  # type: ignore

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.gui.mainwindow import MainWindow
import asymmetry.gui.mainwindow as mw_module


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


class TestMainWindowBasic:
    def test_initialization(self, mainwindow: MainWindow) -> None:
        """Test mainwindow initializes correctly."""
        assert mainwindow is not None
        assert mainwindow.windowTitle() != ""

    def test_has_menu_bar(self, mainwindow: MainWindow) -> None:
        """Test menubar exists."""
        assert mainwindow.menuBar() is not None

    def test_has_central_widget(self, mainwindow: MainWindow) -> None:
        """Test central widget exists."""
        assert mainwindow.centralWidget() is not None

    def test_window_size(self, mainwindow: MainWindow) -> None:
        """Test window has reasonable size."""
        size = mainwindow.size()
        assert size.width() > 0
        assert size.height() > 0

    def test_on_fit_shows_fit_dock(self, mainwindow: MainWindow) -> None:
        """Fit action should unhide the fit dock if it starts hidden."""
        assert mainwindow._dock_fit.isHidden()
        mainwindow._on_fit()
        assert not mainwindow._dock_fit.isHidden()

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
        monkeypatch.setattr(mw_module, "remember_export_path", lambda p: remembered.setdefault("path", p))

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

    def test_toolbar_grouping_before_fit(self, mainwindow: MainWindow) -> None:
        """Toolbar should expose Grouping action before Fit for discoverability."""
        toolbar = mainwindow.findChild(QToolBar)
        assert toolbar is not None
        texts = [action.text() for action in toolbar.actions()]
        assert "Grouping" in texts
        assert "Fit" in texts
        assert texts.index("Grouping") < texts.index("Fit")

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

    def test_apply_grouping_with_source_bunching_avoids_extra_rebinning_at_source_value(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Requested bunch factor equal to source factor should not rebunch again."""
        dataset = _make_dataset(7402, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7402.wim"
        dataset.run.grouping["source_bunching_factor"] = 2
        original_points = len(dataset.time)

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 2,
            "source_bunching_factor": 2,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert len(dataset.time) == original_points

    def test_apply_grouping_rejects_bunching_not_multiple_of_source(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Grouping payloads with non-multiple bunching factors should be rejected."""
        dataset = _make_dataset(7403, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7403.wim"
        dataset.run.grouping["source_bunching_factor"] = 10

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 15,
            "source_bunching_factor": 10,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is False

    def test_apply_grouping_nexus_keeps_non_multiple_bunching_behavior(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Non-WIM datasets should keep legacy bunching behavior (no multiple constraint)."""
        dataset = _make_dataset(7404, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7404.nxs"
        dataset.run.grouping["source_bunching_factor"] = 10

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 15,
            "source_bunching_factor": 10,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True

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

    def test_apply_grouping_wim_without_histograms_updates_bunching(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """WIM datasets without raw histograms should still allow bunch-factor changes."""
        dataset = _make_dataset(7405, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7405.wim"
        dataset.run.histograms = []
        dataset.run.grouping["source_bunching_factor"] = 2
        original_points = len(dataset.time)

        payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 4,
            "source_bunching_factor": 2,
            "enforce_source_bunching": True,
            "deadtime_correction": False,
        }

        applied, _ = mainwindow._apply_grouping_settings_to_dataset(dataset, payload)

        assert applied is True
        assert len(dataset.time) < original_points
        assert dataset.run.grouping["bunching_factor"] == 4

    def test_apply_grouping_wim_without_histograms_restores_source_when_bunching_reduced(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """Reducing WIM bunching should rebuild from the original source arrays."""
        dataset = _make_dataset(7406, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7406.wim"
        dataset.run.histograms = []
        dataset.run.grouping["source_bunching_factor"] = 2
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
            "source_bunching_factor": 2,
            "enforce_source_bunching": True,
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
            {**payload, "bunching_factor": 2},
        )

        assert applied is True
        np.testing.assert_array_equal(dataset.time, original_time)
        np.testing.assert_array_equal(dataset.asymmetry, original_asymmetry)
        np.testing.assert_array_equal(dataset.error, original_error)
        assert dataset.run.grouping["bunching_factor"] == 2

    def test_apply_grouping_wim_without_histograms_keeps_early_points_despite_first_good_bin(
        self,
        mainwindow: MainWindow,
    ) -> None:
        """WIM first-good-bin metadata should not trim already-processed source arrays."""
        dataset = _make_dataset(7407, with_grouping=False)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7407.wim"
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
        np.testing.assert_array_equal(dataset.time, original_time)
        np.testing.assert_array_equal(dataset.asymmetry, original_asymmetry)
        np.testing.assert_array_equal(dataset.error, original_error)
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

    def test_extract_grouping_overrides_preserves_wim_hist_t0_pairs(
        self,
        mainwindow: MainWindow,
    ) -> None:
        dataset = _make_dataset(7454, with_grouping=True)
        assert dataset.run is not None
        dataset.run.source_file = "/tmp/run_7454.wim"
        dataset.run.histograms = []
        dataset.run.grouping["groups"] = {
            1: [(1, 100), (2, 100)],
            2: [(3, 100), (4, 100)],
        }
        dataset.run.grouping["source_bunching_factor"] = 2
        dataset.run.grouping["bunching_factor"] = 4

        payload = mainwindow._extract_grouping_overrides(dataset)

        assert payload is not None
        assert payload["groups"] == {
            1: [(1, 100), (2, 100)],
            2: [(3, 100), (4, 100)],
        }
        assert payload["source_bunching_factor"] == 2
        assert payload["bunching_factor"] == 4

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
        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: calls.__setitem__("render", calls["render"] + 1))

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
        monkeypatch.setattr(mainwindow._plot_panel, "plot_datasets", lambda _datasets: plotted.append(-1))
        monkeypatch.setattr(mainwindow._plot_panel, "plot_dataset", lambda dataset: plotted.append(int(dataset.run_number)))

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
        monkeypatch.setattr(mainwindow._plot_panel, "plot_datasets", lambda _datasets: plotted.append(-1))
        monkeypatch.setattr(mainwindow._plot_panel, "plot_dataset", lambda dataset: plotted.append(int(dataset.run_number)))

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
        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: calls.__setitem__("render", calls["render"] + 1))
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
        monkeypatch.setattr(mainwindow, "_render_current_selection_plot", lambda: calls.__setitem__("render", calls["render"] + 1))
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
        mainwindow._data_browser.get_dataset = (
            lambda run_number: ds1 if int(run_number) == 7711 else ds2
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

    def test_cross_group_completion_shows_global_parameter_window(self, mainwindow: MainWindow) -> None:
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

        mainwindow._fit_panel.clear_fits_for_runs = lambda runs: captured["fit"].extend(runs) or len(runs)
        mainwindow._plot_panel.clear_fits_for_runs = lambda runs: captured["plot"].extend(runs) or len(runs)

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

        mainwindow._load_files(["/tmp/new_run.wim"])

        assert len(applied_payloads) == 1
        assert applied_payloads[0]["forward_group"] == 1
        assert applied_payloads[0]["backward_group"] == 2

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
        low_run.run.grouping.update({
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
        })
        assert high_run.run is not None
        high_run.run.grouping.update({
            "groups": {5: [1], 6: [2]},
            "forward_group": 5,
            "backward_group": 6,
            "alpha": 2.5,
        })

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

        mainwindow._load_files(["/tmp/new_run_uses_highest_grouping.wim"])

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

        mainwindow._load_files(["/tmp/new_run_no_template.wim"])

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

        mainwindow._load_files(["/tmp/new_run_keeps_fits.wim"])

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
        existing.run.source_file = "/tmp/duplicate_no.wim"
        incoming = _make_dataset(7301, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_no.wim"
        incoming.metadata["comment"] = "NEW"

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_a, **_k: QMessageBox.StandardButton.No,
        )

        mainwindow._load_files(["/tmp/duplicate_no.wim"])

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
        existing.run.source_file = "/tmp/duplicate_yes.wim"
        incoming = _make_dataset(7302, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_yes.wim"
        incoming.metadata["comment"] = "NEW"

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_a, **_k: QMessageBox.StandardButton.Yes,
        )

        mainwindow._load_files(["/tmp/duplicate_yes.wim"])

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
        first_existing.run.source_file = "/tmp/duplicate_all_1.wim"
        second_existing = _make_dataset(7304, with_grouping=False)
        second_existing.run.source_file = "/tmp/duplicate_all_2.wim"
        mainwindow._data_browser.add_dataset(first_existing)
        mainwindow._data_browser.add_dataset(second_existing)

        first_incoming = _make_dataset(7303, with_grouping=False)
        first_incoming.run.source_file = "/tmp/duplicate_all_1.wim"
        first_incoming.metadata["comment"] = "NEW1"
        second_incoming = _make_dataset(7304, with_grouping=False)
        second_incoming.run.source_file = "/tmp/duplicate_all_2.wim"
        second_incoming.metadata["comment"] = "NEW2"

        def _stub_load_file(path: str):
            if path.endswith("duplicate_all_1.wim"):
                return first_incoming
            if path.endswith("duplicate_all_2.wim"):
                return second_incoming
            raise AssertionError(f"Unexpected path: {path}")

        prompt_calls = {"n": 0}

        def _stub_question(*_a, **_k):
            prompt_calls["n"] += 1
            return QMessageBox.StandardButton.YesToAll

        monkeypatch.setattr(mainwindow, "_load_file", _stub_load_file)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(QMessageBox, "question", _stub_question)

        mainwindow._load_files([
            "/tmp/duplicate_all_1.wim",
            "/tmp/duplicate_all_2.wim",
        ])

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
        existing.run.source_file = "/tmp/duplicate_message.wim"
        incoming = _make_dataset(7305, with_grouping=False)
        incoming.run.source_file = "/tmp/duplicate_message.wim"

        captured = {"text": ""}

        def _stub_question(_parent, _title, text, *_args, **_kwargs):
            captured["text"] = text
            return QMessageBox.StandardButton.No

        mainwindow._data_browser.add_dataset(existing)
        monkeypatch.setattr(mainwindow, "_load_file", lambda _path: incoming)
        monkeypatch.setattr(mainwindow, "_maybe_apply_comment_field", lambda *a, **k: "none")
        monkeypatch.setattr(QMessageBox, "question", _stub_question)

        mainwindow._load_files(["/tmp/duplicate_message.wim"])

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

    def test_grouping_request_uses_wim_dialog_for_wim_reference(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WIM runs should open the dedicated WIM grouping dialog."""
        ds = _make_dataset(8301, with_grouping=False)
        assert ds.run is not None
        ds.run.source_file = "/tmp/run_8301.wim"
        mainwindow._current_dataset = ds

        monkeypatch.setattr(mainwindow._data_browser, "get_all_datasets", lambda: [ds])
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [ds])

        calls = {"wim": 0, "normal": 0}

        class _StubWimGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                calls["wim"] += 1

            def exec(self):
                return 0

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                calls["normal"] += 1

            def exec(self):
                return 0

        monkeypatch.setattr(mw_module, "WimGroupingDialog", _StubWimGroupingDialog)
        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)

        mainwindow._on_grouping_current()

        assert calls["wim"] == 1
        assert calls["normal"] == 0

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
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [combined_ds])

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
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [combined_ds])
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

        monkeypatch.setattr(mainwindow._data_browser, "_get_selected_run_numbers", lambda: [combined_rn])
        mainwindow._data_browser._separate_combined()

        restored_1 = mainwindow._data_browser.get_dataset(8411)
        restored_2 = mainwindow._data_browser.get_dataset(8412)
        assert restored_1 is ds1
        assert restored_2 is ds2
        assert restored_1.run.grouping["alpha"] == pytest.approx(2.5)
        assert restored_2.run.grouping["alpha"] == pytest.approx(2.5)

    def test_grouping_request_uses_wim_dialog_for_combined_wim_sources(
        self,
        mainwindow: MainWindow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds1 = _make_dataset(8421, with_grouping=True)
        ds2 = _make_dataset(8422, with_grouping=True)
        assert ds1.run is not None and ds2.run is not None
        ds1.run.source_file = "/tmp/run_8421.wim"
        ds2.run.source_file = "/tmp/run_8422.wim"
        mainwindow._data_browser.add_dataset(ds1)
        mainwindow._data_browser.add_dataset(ds2)
        combined_rn = mainwindow._data_browser.add_combined_dataset([8421, 8422])
        assert combined_rn is not None
        combined_ds = mainwindow._data_browser.get_dataset(combined_rn)
        assert combined_ds is not None

        mainwindow._current_dataset = combined_ds
        monkeypatch.setattr(mainwindow._data_browser, "get_selected_datasets", lambda: [combined_ds])

        calls = {"wim": 0, "normal": 0}

        class _StubWimGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                calls["wim"] += 1

            def exec(self):
                return 0

        class _StubGroupingDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *_args, **_kwargs):
                calls["normal"] += 1

            def exec(self):
                return 0

        monkeypatch.setattr(mw_module, "WimGroupingDialog", _StubWimGroupingDialog)
        monkeypatch.setattr(mw_module, "GroupingDialog", _StubGroupingDialog)

        mainwindow._on_grouping_current()

        assert calls["wim"] == 1
        assert calls["normal"] == 0
