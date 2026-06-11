"""Tests for project file schema, migration, IO round-trips, and GUI restore."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

# ── schema / migration unit tests ─────────────────────────────────────────────
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    load_project,
    save_project,
)
from asymmetry.core.project.schema import migrate_to_current, validate


def _minimal_state() -> dict:
    """Return the smallest valid project state dict."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_with_app_version": "0.1.0",
        "datasets": [],
        "combined_datasets": [],
        "browser_state": {
            "sort_column": -1,
            "sort_order": "ascending",
            "filters": {},
            "selected_run_numbers": [],
            "selected_group_ids": [],
            "data_groups": [],
            "extra_columns": [],
        },
        "plot_state": {
            "current_run_number": None,
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "fit_curve": None,
            "fit_curves": {},
        },
        "single_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [],
        },
        "global_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [],
        },
        "fit_ui_state": {
            "active_tab_index": 0,
        },
        "fit_parameters_state": {
            "rows": [],
        },
        "fourier_state": {
            "window": "none",
            "padding": 1,
            "phase_degrees": 0.0,
            "t0_offset_us": 0.0,
            "display": "(Power)^1/2",
            "auto_phase": False,
            "auto_phase_method": "Peak",
            "use_phase_table": False,
            "group_phase_table": {},
            "group_auto_filled_ids": [],
        },
        "frequency_fit_state": {
            "domain": "frequency",
            "single_fit_state": {},
            "global_fit_state": {},
            "fit_ui_state": {},
        },
    }


class TestSchemaMigration:
    def test_current_version_passes_through(self):
        state = {"schema_version": 7, "datasets": []}
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert result["datasets"] == []

    def test_v1_migrates_to_v2(self):
        state = {"schema_version": 1, "datasets": []}
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert "browser_state" in result

    def test_v2_migrates_to_v3_with_extra_columns(self):
        state = {"schema_version": 2, "datasets": [], "browser_state": {}}
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert result["browser_state"]["extra_columns"] == []

    def test_v3_vector_grouping_adds_per_axis_alpha_fields(self):
        state = {
            "schema_version": 3,
            "datasets": [
                {
                    "run_number": 5001,
                    "source_file": "/tmp/run_5001.nxs",
                    "metadata_overrides": {"field": 100.0},
                    "grouping_overrides": {
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
                        "alpha": 1.7,
                    },
                }
            ],
        }

        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        grouping = result["datasets"][0]["grouping_overrides"]
        assert grouping["alpha_x"] == pytest.approx(1.7)
        assert grouping["alpha_y"] == pytest.approx(1.7)
        assert grouping["alpha_z"] == pytest.approx(1.7)

    def test_v4_migrates_to_v5_with_frequency_fit_state(self):
        state = {"schema_version": 4, "datasets": []}
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert result["frequency_fit_state"]["domain"] == "frequency"

    def test_unsupported_future_version_raises(self):
        state = {"schema_version": 999, "datasets": []}
        with pytest.raises(UnsupportedSchemaVersion):
            migrate_to_current(state)

    def test_missing_version_without_project_shape_raises(self):
        state = {"not_a_project": True}
        with pytest.raises(UnsupportedSchemaVersion):
            migrate_to_current(state)

    def test_validate_passes_valid_state(self):
        validate({"schema_version": 5, "datasets": []})

    def test_validate_raises_on_missing_datasets_key(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate({"schema_version": 5})

    def test_validate_raises_on_missing_schema_version(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate({"datasets": []})

    def test_unknown_top_level_fields_are_allowed(self):
        """Future-proofing: unknown fields must not break validation."""
        state = {"schema_version": 5, "datasets": [], "future_field": "keep me"}
        validate(state)  # must not raise

    def test_current_schema_version_constant(self):
        assert CURRENT_SCHEMA_VERSION == 8


def _composite_model_dict(component: str = "Exponential") -> dict:
    return {
        "component_names": [component],
        "operators": [],
        "open_parentheses": [0],
        "close_parentheses": [0],
        "fraction_groups": [],
    }


class TestSchemaMigrationV5toV6:
    """v5 → v6 introduces recipe-only representations and batches."""

    def _v5_state(self) -> dict:
        return {
            "schema_version": 5,
            "datasets": [
                {"run_number": 100, "source_file": "/tmp/a.nxs"},
                {"run_number": 200, "source_file": "/tmp/b.nxs"},
                {"run_number": 300, "source_file": "/tmp/c.nxs"},
            ],
            "single_fit_state": {
                "model_name": "Composite",
                "composite_model": _composite_model_dict("Gaussian"),
                "parameters": [{"name": "A", "value": 0.3}],
                "result_html": "<b>chi2 = 1.0</b>",
                "active_run_number": 200,
                "states_by_run": {
                    "100": {
                        "model_name": "Composite",
                        "composite_model": _composite_model_dict("Exponential"),
                        "parameters": [{"name": "A", "value": 0.2}],
                        "result_html": "<b>chi2 = 2.0</b>",
                    },
                },
            },
            "frequency_fit_state": {
                "domain": "frequency",
                "single_fit_state": {
                    "states_by_run": {
                        "100": {
                            "model_name": "Composite",
                            "composite_model": _composite_model_dict("GaussianPeak"),
                            "parameters": [{"name": "height", "value": 1.0}],
                            "result_html": "<b>freq fit</b>",
                        },
                    },
                },
            },
            "fourier_state": {
                "window": "gaussian",
                "padding": 2,
                "display": "Cos",
                "auto_phase": True,
            },
            # Only runs 100 and 300 had a generated spectrum.
            "fourier_spectra_state": {
                "100": [{"time": [0.0], "asymmetry": [1.0], "error": [0.0], "metadata": {}}],
                "300": [{"time": [0.0], "asymmetry": [1.0], "error": [0.0], "metadata": {}}],
            },
        }

    def test_sets_version_and_empty_batches(self):
        result = migrate_to_current(self._v5_state())
        assert result["schema_version"] == 8
        assert result["batches"] == []

    def test_per_run_single_fit_lands_on_right_dataset(self):
        result = migrate_to_current(self._v5_state())
        by_run = {ds["run_number"]: ds for ds in result["datasets"]}

        fit_100 = by_run[100]["representations"]["time_fb_asymmetry"]["fit"]
        assert fit_100["model"]["component_names"] == ["Exponential"]
        assert fit_100["provenance"] == "single"
        assert fit_100["result"] == {"result_html": "<b>chi2 = 2.0</b>"}

    def test_bare_active_single_state_maps_to_active_run_only(self):
        result = migrate_to_current(self._v5_state())
        by_run = {ds["run_number"]: ds for ds in result["datasets"]}

        # Run 200 is the active run -> inherits the bare composite model.
        fit_200 = by_run[200]["representations"]["time_fb_asymmetry"]["fit"]
        assert fit_200["model"]["component_names"] == ["Gaussian"]
        # Run 300 had no single fit -> no time_fb_asymmetry representation.
        assert "time_fb_asymmetry" not in by_run[300].get("representations", {})

    def test_frequency_single_fit_and_fourier_recipe(self):
        result = migrate_to_current(self._v5_state())
        by_run = {ds["run_number"]: ds for ds in result["datasets"]}

        freq_100 = by_run[100]["representations"]["freq_fft"]
        assert freq_100["fit"]["model"]["component_names"] == ["GaussianPeak"]
        config = freq_100["recipe"]["fourier_config"]
        assert config["window"] == "gaussian"
        assert config["padding"] == 2
        assert config["display"] == "Cos"
        # Non-recipe keys (e.g. auto_phase) are not carried into the recipe.
        assert "auto_phase" not in config

    def test_fourier_recipe_applied_even_without_freq_fit(self):
        result = migrate_to_current(self._v5_state())
        by_run = {ds["run_number"]: ds for ds in result["datasets"]}
        # Run 300 has no frequency fit but still gets the FFT recipe + empty slot.
        freq_300 = by_run[300]["representations"]["freq_fft"]
        assert freq_300["recipe"]["fourier_config"]["window"] == "gaussian"
        assert freq_300["fit"]["provenance"] == "none"

    def test_old_blobs_preserved(self):
        result = migrate_to_current(self._v5_state())
        assert "single_fit_state" in result
        assert "fourier_state" in result
        assert result["single_fit_state"]["states_by_run"]["100"]["result_html"] == (
            "<b>chi2 = 2.0</b>"
        )

    def test_migration_without_fits_adds_only_batches(self):
        state = {"schema_version": 5, "datasets": [{"run_number": 1}]}
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert result["batches"] == []
        assert "representations" not in result["datasets"][0]


class TestSchemaMigrationV6toV7:
    """v6 → v7 generalises batches into fit series and structures trend state."""

    def test_sets_version(self):
        result = migrate_to_current({"schema_version": 6, "datasets": []})
        assert result["schema_version"] == 8

    def test_series_gain_member_kind_and_defaults(self):
        state = {
            "schema_version": 6,
            "datasets": [],
            "batches": [
                {
                    "batch_id": "b1",
                    "rep_type": "time_fb_asymmetry",
                    "member_run_numbers": [10, 11],
                }
            ],
        }
        result = migrate_to_current(state)
        series = result["batches"][0]
        assert series["member_kind"] == "runs"
        assert series["nuisance_params"] == []
        assert series["member_source_run"] == {}
        # Pre-existing fields are preserved (additive migration).
        assert series["member_run_numbers"] == [10, 11]


class TestSchemaMigrationV7toV8:
    """v7 → v8 adds a freeform ``extra`` dict to each series."""

    def test_series_gain_empty_extra(self):
        state = {
            "schema_version": 7,
            "datasets": [],
            "batches": [
                {
                    "batch_id": "b1",
                    "rep_type": "time_fb_asymmetry",
                    "member_run_numbers": [10],
                    "member_kind": "runs",
                    "nuisance_params": [],
                    "member_source_run": {},
                }
            ],
        }
        result = migrate_to_current(state)
        assert result["schema_version"] == 8
        assert result["batches"][0]["extra"] == {}

    def test_existing_extra_preserved(self):
        state = {
            "schema_version": 7,
            "datasets": [],
            "batches": [
                {"batch_id": "b1", "rep_type": "time_fb_asymmetry", "extra": {"kind": "x"}}
            ],
        }
        result = migrate_to_current(state)
        assert result["batches"][0]["extra"] == {"kind": "x"}

    def test_current_version_is_idempotent(self):
        # A file already at the current version passes through untouched (no
        # migration block re-fires and mutates already-current state).
        state = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "batches": [
                {"batch_id": "b1", "rep_type": "time_fb_asymmetry", "extra": {"kind": "x"}}
            ],
        }
        result = migrate_to_current(state)
        assert result["schema_version"] == CURRENT_SCHEMA_VERSION
        assert result["batches"][0]["extra"] == {"kind": "x"}

    def test_handles_missing_or_malformed_batches(self):
        # batches absent, None, or carrying a non-dict element must not crash.
        assert migrate_to_current({"schema_version": 7, "datasets": []})["schema_version"] == 8
        none_batches = {"schema_version": 7, "datasets": [], "batches": None}
        assert migrate_to_current(none_batches)["schema_version"] == 8
        out = migrate_to_current(
            {"schema_version": 7, "datasets": [], "batches": ["stray", {"batch_id": "b"}]}
        )
        assert out["batches"][0] == "stray"
        assert out["batches"][1]["extra"] == {}

    def test_trend_state_unknown_keys_preserved_under_legacy(self):
        state = {
            "schema_version": 6,
            "datasets": [
                {
                    "run_number": 7,
                    "representations": {
                        "time_fb_asymmetry": {
                            "recipe": {},
                            "fit": {"provenance": "none"},
                            "trend_state": {"x_key": "field", "mystery": 1},
                        }
                    },
                }
            ],
        }
        result = migrate_to_current(state)
        ts = result["datasets"][0]["representations"]["time_fb_asymmetry"]["trend_state"]
        assert ts["x_key"] == "field"
        assert ts["legacy"] == {"mystery": 1}

    def test_empty_trend_state_stays_empty(self):
        state = {
            "schema_version": 6,
            "datasets": [
                {
                    "run_number": 7,
                    "representations": {"freq_fft": {"recipe": {}, "fit": {}, "trend_state": {}}},
                }
            ],
        }
        result = migrate_to_current(state)
        ts = result["datasets"][0]["representations"]["freq_fft"]["trend_state"]
        assert ts == {}


class TestProjectIO:
    def test_save_and_load_round_trip(self, tmp_path):
        state = _minimal_state()
        path = tmp_path / "test.asymp"
        save_project(state, path)

        loaded = load_project(path)
        assert loaded["schema_version"] == 8
        assert loaded["datasets"] == []

    def test_vector_alpha_xyz_persist_in_project_round_trip(self, tmp_path):
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 7001,
                "source_file": "/tmp/run7001.nxs",
                "metadata_overrides": {"field": 100.0},
                "grouping_overrides": {
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
                    "alpha": 1.0,
                    "alpha_x": 1.11,
                    "alpha_y": 1.22,
                    "alpha_z": 1.33,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                },
            }
        ]
        path = tmp_path / "vector_alpha_roundtrip.asymp"

        save_project(state, path)
        loaded = load_project(path)

        grouping = loaded["datasets"][0]["grouping_overrides"]
        assert grouping["alpha_x"] == pytest.approx(1.11)
        assert grouping["alpha_y"] == pytest.approx(1.22)
        assert grouping["alpha_z"] == pytest.approx(1.33)

    def test_file_is_valid_json(self, tmp_path):
        state = _minimal_state()
        path = tmp_path / "test.asymp"
        save_project(state, path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 8

    def test_optional_wizard_cache_state_round_trips(self, tmp_path):
        state = _minimal_state()
        state["single_fit_state"]["wizard_state"] = {
            "signature": {"run_number": 1001, "model": None},
            "recommendation": {"summary": "single cached"},
            "log_text": "single log",
        }
        state["global_fit_state"]["wizard_state"] = {
            "signature": {
                "run_numbers": [1001, 1002],
                "search_strategy": "staged_v2",
            },
            "recommendation": {"summary": "global cached"},
            "log_text": "global log",
        }
        state["global_fit_state"]["wizard_state_by_run_set"] = [
            {
                "run_numbers": [1001, 1002],
                "signature": {"run_numbers": [1001, 1002]},
                "recommendation": {"summary": "group 1"},
                "log_text": "group 1 log",
            },
            {
                "run_numbers": [1001, 1003],
                "signature": {"run_numbers": [1001, 1003]},
                "recommendation": {"summary": "group 2"},
                "log_text": "group 2 log",
            },
        ]

        path = tmp_path / "wizard_state_roundtrip.asymp"
        save_project(state, path)
        loaded = load_project(path)

        assert loaded["single_fit_state"]["wizard_state"]["log_text"] == "single log"
        assert loaded["single_fit_state"]["wizard_state"]["signature"]["run_number"] == 1001
        assert loaded["global_fit_state"]["wizard_state"]["log_text"] == "global log"
        assert (
            loaded["global_fit_state"]["wizard_state"]["signature"]["search_strategy"]
            == "staged_v2"
        )
        assert len(loaded["global_fit_state"]["wizard_state_by_run_set"]) == 2
        assert loaded["global_fit_state"]["wizard_state_by_run_set"][1]["log_text"] == "group 2 log"

    def test_numpy_arrays_serialised_as_lists(self, tmp_path):
        state = _minimal_state()
        state["plot_state"]["fit_curve"] = {
            "t": list(np.linspace(0, 10, 5)),
            "y": list(np.ones(5) * 0.2),
            "label": "Fit",
        }
        path = tmp_path / "test.asymp"
        save_project(state, path)
        loaded = load_project(path)
        fit_curve = loaded["plot_state"]["fit_curve"]
        assert isinstance(fit_curve["t"], list)
        assert len(fit_curve["t"]) == 5

    def test_link_groups_persist_in_project_round_trip(self, tmp_path):
        """Equality link groups on single-fit parameters survive save/load."""
        state = _minimal_state()
        state["single_fit_state"] = {
            "model_name": "Composite",
            "composite_model": _composite_model_dict("Oscillatory"),
            "parameters": [
                {"name": "Lambda_2", "value": 0.30, "link_group": 1},
                {"name": "Lambda_4", "value": 0.30, "link_group": 1},
                {"name": "frequency_1", "value": 1.389, "link_group": None},
            ],
            "result_html": "",
        }
        path = tmp_path / "links.asymp"
        save_project(state, path)
        loaded = load_project(path)

        by_name = {p["name"]: p for p in loaded["single_fit_state"]["parameters"]}
        assert by_name["Lambda_2"]["link_group"] == 1
        assert by_name["Lambda_4"]["link_group"] == 1
        assert by_name["frequency_1"]["link_group"] is None

    def test_load_unsupported_version_raises(self, tmp_path):
        bad_state = {"schema_version": 999, "datasets": []}
        path = tmp_path / "future.asymp"
        path.write_text(json.dumps(bad_state), encoding="utf-8")
        with pytest.raises(UnsupportedSchemaVersion):
            load_project(path)

    def test_load_missing_key_raises(self, tmp_path):
        bad_state = {"schema_version": 5}  # datasets key missing
        path = tmp_path / "bad.asymp"
        path.write_text(json.dumps(bad_state), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required keys"):
            load_project(path)

    def test_load_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_project(tmp_path / "nonexistent.asymp")

    def test_numpy_types_in_state_serialised(self, tmp_path):
        """numpy int/float types in collected state must not cause TypeError."""
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": np.int64(42),
                "source_file": "/data/run42.nxs",
                "metadata_overrides": {"field": np.float64(150.0)},
            }
        ]
        path = tmp_path / "numpy_types.asymp"
        save_project(state, path)
        loaded = load_project(path)
        assert loaded["datasets"][0]["run_number"] == 42
        assert loaded["datasets"][0]["metadata_overrides"]["field"] == pytest.approx(150.0)


# ── widget state helpers ───────────────────────────────────────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

from asymmetry.core.data.dataset import Histogram, MuonDataset


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _set_default_ui_scale():
    settings = QSettings()
    settings.setValue("ui/scale", 1.0)


def _make_dataset(run_number: int = 42) -> MuonDataset:
    from asymmetry.core.data.dataset import Run

    t = np.linspace(0, 10, 100)
    run = Run(run_number=run_number, source_file=f"/data/run{run_number}.nxs")
    run.metadata["field"] = 100.0
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-t),
        error=np.full_like(t, 0.01),
        metadata={"title": f"Run {run_number}", "temperature": 5.0, "field": 100.0, "comment": ""},
        run=run,
    )


class TestDataBrowserPanelState:
    def test_get_state_returns_required_keys(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        state = panel.get_state()
        assert "sort_column" in state
        assert "sort_order" in state
        assert "filters" in state
        assert "selected_run_numbers" in state

    def test_clear_empties_browser(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        panel.add_dataset(_make_dataset(1))
        panel.add_dataset(_make_dataset(2))
        assert panel._table.rowCount() == 2
        panel.clear()
        assert panel._table.rowCount() == 0
        assert not panel._datasets

    def test_restore_state_applies_sort(self, qapp):
        from PySide6.QtCore import Qt

        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        panel.add_dataset(_make_dataset(1))
        panel.add_dataset(_make_dataset(2))

        state = {
            "sort_column": 0,
            "sort_order": "descending",
            "filters": {},
            "selected_run_numbers": [],
        }
        panel.restore_state(state)
        assert panel._current_sort_column == 0
        assert panel._current_sort_order == Qt.SortOrder.DescendingOrder

    def test_add_combined_dataset_creates_entry(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        ds1 = _make_dataset(10)
        ds2 = _make_dataset(11)
        ds1.run.grouping = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 99,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }
        ds2.run.grouping = dict(ds1.run.grouping)
        panel.add_dataset(ds1)
        panel.add_dataset(ds2)

        combined_rn = panel.add_combined_dataset([10, 11])
        assert combined_rn is not None
        assert combined_rn < 0  # Combined IDs are always negative
        assert combined_rn in panel._combined_datasets
        assert panel._combined_datasets[combined_rn] == [10, 11]
        # Source runs are no longer individually accessible.
        assert 10 not in panel._datasets
        assert 11 not in panel._datasets

    def test_add_combined_dataset_missing_source_returns_none(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        ds1 = _make_dataset(10)
        ds1.run.grouping = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 99,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }
        panel.add_dataset(ds1)
        # Run 99 doesn't exist
        result = panel.add_combined_dataset([10, 99])
        assert result is None

    def test_extra_column_header_uses_run_info_label_for_known_field(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        panel.add_extra_column("run_info.points")

        header = panel._table.horizontalHeaderItem(len(panel._COLUMNS))
        assert header is not None
        assert header.text() == "Points"

    def test_extra_column_header_keeps_key_for_unknown_field(self, qapp):
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        panel = DataBrowserPanel()
        panel.add_extra_column("nexus_fields.sample.custom_value")

        header = panel._table.horizontalHeaderItem(len(panel._COLUMNS))
        assert header is not None
        assert header.text() == "nexus_fields.sample.custom_value"


class TestPlotPanelState:
    def test_get_state_returns_required_keys(self, qapp):
        from asymmetry.gui.panels.plot_panel import PlotPanel

        panel = PlotPanel()
        state = panel.get_state()
        for key in (
            "current_run_number",
            "bunch_factor",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "fit_curve",
            "fit_curves",
        ):
            assert key in state
        assert state["bunch_factor"] == 1

    def test_restore_state_accepts_legacy_bunch_factor_key(self, qapp):
        from asymmetry.gui.panels.plot_panel import PlotPanel

        panel = PlotPanel()
        if not panel._has_mpl:
            pytest.skip("matplotlib not available")

        panel.restore_state(
            {
                "bunch_factor": 4,
                "x_min": 0.0,
                "x_max": 5.0,
                "y_min": -20.0,
                "y_max": 20.0,
                "fit_curve": None,
                "fit_curves": {},
            }
        )
        state = panel.get_state()
        assert state["bunch_factor"] == 1

    def test_restore_state_restores_fit_curve(self, qapp):
        from asymmetry.gui.panels.plot_panel import PlotPanel

        panel = PlotPanel()
        if not panel._has_mpl:
            pytest.skip("matplotlib not available")

        fit_state = {
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "fit_curve": {"t": [0.0, 1.0, 2.0], "y": [0.2, 0.1, 0.05], "label": "Fit"},
            "fit_curves": {},
        }
        panel.restore_state(fit_state)
        assert panel._fit_curve is not None
        np.testing.assert_allclose(panel._fit_curve[0], [0.0, 1.0, 2.0])
        assert panel._fit_curve[2] == "Fit"

    def test_get_then_restore_is_idempotent(self, qapp):
        from asymmetry.gui.panels.plot_panel import PlotPanel

        panel = PlotPanel()
        if not panel._has_mpl:
            pytest.skip("matplotlib not available")

        ds = _make_dataset(42)
        panel.plot_dataset(ds)

        state1 = panel.get_state()

        panel2 = PlotPanel()
        panel2.restore_state(state1)
        state2 = panel2.get_state()

        assert state1["bunch_factor"] == state2["bunch_factor"]
        assert state1["x_min"] == pytest.approx(state2["x_min"])

    def test_restore_global_fit_curves(self, qapp):
        from asymmetry.gui.panels.plot_panel import PlotPanel

        panel = PlotPanel()
        if not panel._has_mpl:
            pytest.skip("matplotlib not available")

        fit_state = {
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "fit_curve": None,
            "fit_curves": {
                "42": {"t": [0.0, 1.0], "y": [0.2, 0.1], "label": "Global Fit"},
                "99": {"t": [0.0, 1.0], "y": [0.1, 0.05], "label": "Global Fit"},
            },
        }
        panel.restore_state(fit_state)
        assert 42 in panel._fit_curves
        assert 99 in panel._fit_curves
        assert panel._fit_curve is None


class TestFourierPanelState:
    def test_get_and_restore_round_trip(self, qapp):
        from asymmetry.gui.panels.fourier_panel import FourierPanel

        panel = FourierPanel()
        panel._filter_gaussian_radio.setChecked(True)
        panel._filter_start_edit.setText("0.75")
        panel._filter_time_constant_edit.setText("2.25")
        panel._padding_spin.setValue(4)
        panel._phase_spin.setText("32.5")
        panel._t0_offset_spin.setText("0.125")
        panel._phase_mode_radio.setChecked(True)
        panel._auto_method_combo.setCurrentText("Average")
        panel._use_phase_table_check.setChecked(True)
        panel._estimate_average_error_check.setChecked(True)
        panel.set_group_definitions({1: "Forward", 2: "Backward"}, {1: 12.5, 2: -8.0})

        state = panel.get_state()
        assert state == {
            "window": "gaussian",
            "filter_start_us": 0.75,
            "filter_time_constant_us": 2.25,
            "padding": 4,
            "phase_degrees": 32.5,
            "t0_offset_us": 0.125,
            "display": "Phase",
            "subtract_average_signal": True,
            "auto_phase_method": "Average",
            "use_phase_table": True,
            "estimate_average_error": True,
            "group_enabled_table": {1: True, 2: True},
            "group_phase_table": {1: 12.5, 2: -8.0},
            "group_auto_filled_ids": [],
            "pulse_compensation": False,
            "pulse_half_width_us": 0.0,
            "pulse_max_gain": 25.0,
            "baseline_mode": "none",
            "baseline_kappa": 2.0,
            "exclude_enabled": False,
            "diamag_exclusion": False,
            "diamag_half_width_mhz": 0.3,
            "exclusion_ranges": [],
            "remove_diamag": False,
            "burg_order_min": 2,
            "burg_order_max": 40,
        }

        panel2 = FourierPanel()
        panel2.restore_state(state)
        assert panel2._filter_gaussian_radio.isChecked() is True
        assert panel2._filter_start_edit.text() == "0.75"
        assert panel2._filter_time_constant_edit.text() == "2.25"
        assert panel2._padding_spin.value() == 4
        assert float(panel2._phase_spin.text()) == pytest.approx(32.5)
        assert float(panel2._t0_offset_spin.text()) == pytest.approx(0.125)
        assert panel2._current_display_mode() == "Phase"
        assert panel2._auto_method_combo.currentText() == "Average"
        assert panel2._use_phase_table_check.isChecked() is True
        assert panel2._estimate_average_error_check.isChecked() is True
        assert panel2.group_enabled_table() == {1: True, 2: True}
        assert panel2.group_phase_table() == pytest.approx({1: 12.5, 2: -8.0})


class TestFitPanelState:
    def test_single_get_state_returns_model_and_params(self, qapp):
        from asymmetry.gui.panels.fit_panel import SingleFitTab

        tab = SingleFitTab()
        state = tab.get_state()
        assert "model_name" in state
        assert "parameters" in state
        assert isinstance(state["parameters"], list)

    def test_single_restore_state_sets_model(self, qapp):
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.panels.fit_panel import SingleFitTab

        tab = SingleFitTab()
        state = {
            "model_name": "Composite",
            "composite_model": CompositeModel(["Gaussian", "Constant"], operators=["+"]).to_dict(),
            "parameters": [],
        }
        tab.restore_state(state)
        assert tab._composite_model.component_names == ["Gaussian", "Constant"]

    def test_fit_panel_get_single_state_delegates(self, qapp):
        from asymmetry.gui.panels.fit_panel import FitPanel

        panel = FitPanel()
        state = panel.get_single_state()
        assert "model_name" in state

    def test_fit_panel_get_global_state_delegates(self, qapp):
        from asymmetry.gui.panels.fit_panel import FitPanel

        panel = FitPanel()
        state = panel.get_global_state()
        assert "model_name" in state
        assert "parameters" in state

    def test_fit_panel_legacy_grouped_mode_state_falls_back_to_datasets_mode(self, qapp):
        from asymmetry.gui.panels.fit_panel import FitPanel

        panel = FitPanel()
        state = panel.get_global_state()
        state["mode"] = "grouped"
        state["group_parameters"] = {"N0": {"value": 1234.0, "type": "Local", "bounds": "0, inf"}}

        panel2 = FitPanel()
        panel2.restore_global_state(state)

        # The legacy "mode" key is ignored; member kind is fixed by construction.
        assert panel2._global_tab.is_grouped_time_domain_mode() is False

    def test_fit_panel_round_trips_result_text_and_ui_tab(self, qapp):
        from asymmetry.gui.panels.fit_panel import FitPanel

        panel = FitPanel()
        panel._single_tab._result_label.setText("<b>Saved Single Fit</b>")
        panel._global_tab._result_text.setHtml("<b>Saved Global Fit</b>")
        panel._tabs.setCurrentIndex(1)

        single_state = panel.get_single_state()
        global_state = panel.get_global_state()
        ui_state = panel.get_ui_state()

        panel2 = FitPanel()
        panel2.restore_single_state(single_state)
        panel2.restore_global_state(global_state)
        panel2.restore_ui_state(ui_state)

        assert "Saved Single Fit" in panel2._single_tab._result_label.text()
        assert "Saved Global Fit" in panel2._global_tab._result_text.toPlainText()
        assert panel2._tabs.currentIndex() == 1


# ── MainWindow project orchestration (headless) ────────────────────────────────


import asymmetry.gui.mainwindow as mw_module
from tests.test_gui_shell import (
    _StubDataBrowser,
    _StubFitPanel,
    _StubFitParams,
    _StubFourier,
    _StubLogPanel,
    _StubPlotPanel,
)


class _StubDataBrowserWithState(_StubDataBrowser):
    """Extended stub that adds minimal state helper support."""

    def __init__(self):
        super().__init__()
        self._combined_datasets = {}
        self._was_cleared = False
        self._use_temperature_from_log = False

    def clear(self):
        self._was_cleared = True
        self._datasets.clear()

    def get_state(self):
        return {
            "sort_column": -1,
            "sort_order": "ascending",
            "filters": {},
            "selected_run_numbers": [],
        }

    def restore_state(self, state):
        pass

    def add_combined_dataset(self, source_run_numbers):
        return None

    def use_temperature_from_log(self):
        return bool(self._use_temperature_from_log)

    def set_use_temperature_from_log(self, enabled):
        self._use_temperature_from_log = bool(enabled)


class _StubFitParamsClear(_StubFitParams):
    def __init__(self):
        super().__init__()
        self._was_cleared = False
        self._restored_state = None

    def clear(self):
        self._was_cleared = True

    def get_state(self):
        return {"rows": []}

    def restore_state(self, state):
        self._restored_state = state


class _StubFitParamsWithCrossGroup(_StubFitParamsClear):
    def __init__(self):
        super().__init__()
        self._last_cross_group_fit = {
            "fit_result": object(),
            "model": object(),
            "groups": [],
            "parameter_name": "Lambda",
            "x_key": "run",
            "fit_x_min": float("nan"),
            "fit_x_max": float("nan"),
        }

    @property
    def last_cross_group_fit(self):
        return self._last_cross_group_fit


class _StubPlotPanelWithState(_StubPlotPanel):
    def get_state(self):
        return {
            "current_run_number": None,
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "fit_curve": None,
            "fit_curves": {},
        }

    def restore_state(self, state, dataset=None):
        self._restored_state = state


class _StubFitPanelWithState(_StubFitPanel):
    def get_single_state(self):
        return {"model_name": "ExponentialRelaxation", "parameters": []}

    def restore_single_state(self, state):
        self._single_state = state

    def get_global_state(self):
        return {"model_name": "ExponentialRelaxation", "parameters": []}

    def restore_global_state(self, state):
        self._global_state = state

    def get_ui_state(self):
        return {"active_tab_index": 0}

    def restore_ui_state(self, state):
        self._ui_state = state


class _StubMultiGroupFitWindowWithState(QWidget):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.grouped_fit_completed = SimpleNamespace(connect=lambda _callback: None)
        self.grouped_preview_requested = SimpleNamespace(connect=lambda _callback: None)
        self.count_fit_completed = SimpleNamespace(connect=lambda _callback: None)
        self.count_grouping_promoted = SimpleNamespace(connect=lambda _callback: None)
        self._state = {"model_name": "Composite", "parameters": [], "result_html": ""}
        self.restored_state = None

    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_fit_blocked(self, blocked, reason=""):
        self._blocked = (blocked, reason)

    def dock_title(self):
        return "Multi-Group Fit"

    def grouped_fit_formula_string(self):
        return "A(t)"

    def get_state(self):
        return dict(self._state)

    def restore_state(self, state):
        self.restored_state = state


class TestFitParametersPanelState:
    def test_restore_then_get_state_round_trip(self, qapp):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        panel = FitParametersPanel()
        state = {
            "rows": [
                {
                    "run_number": 101,
                    "field": 100.0,
                    "temperature": 5.0,
                    "values": {"A0": 20.0, "Lambda": 0.4},
                    "errors": {"A0": 0.5, "Lambda": 0.02},
                },
                {
                    "run_number": 102,
                    "field": 200.0,
                    "temperature": 5.1,
                    "values": {"A0": 19.5, "Lambda": 0.5},
                    "errors": {"A0": 0.4, "Lambda": 0.03},
                },
            ],
            "varying_params": ["A0", "Lambda"],
            "inferred_x_key": "field",
            "x_axis": "Auto",
            "selected_y_params": ["Lambda"],
            "log_x": False,
            "log_y_params": ["Lambda"],
            "plot_mode": "Subplots",
        }

        panel.restore_state(state)
        out = panel.get_state()

        assert len(out["rows"]) == 2
        assert out["plot_mode"] == "Subplots"
        assert "Lambda" in out["log_y_params"]
        assert "Lambda" in out["selected_y_params"]

    def test_selected_y_persists_when_table_selection_transiently_empty(self, qapp):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        panel = FitParametersPanel()
        state = {
            "rows": [
                {
                    "run_number": 101,
                    "field": 100.0,
                    "temperature": 5.0,
                    "values": {"A0": 20.0, "Lambda": 0.4},
                    "errors": {"A0": 0.5, "Lambda": 0.02},
                },
                {
                    "run_number": 102,
                    "field": 200.0,
                    "temperature": 5.1,
                    "values": {"A0": 19.5, "Lambda": 0.5},
                    "errors": {"A0": 0.4, "Lambda": 0.03},
                },
            ],
            "varying_params": ["A0", "Lambda"],
            "inferred_x_key": "field",
            "x_axis": "Auto",
            "selected_y_params": ["Lambda"],
            "log_x": False,
            "log_y_params": [],
            "plot_mode": "Single Axes",
        }

        panel.restore_state(state)

        # Simulate a transient UI state where table selection appears empty.
        panel._y_selector_table.blockSignals(True)
        panel._y_selector_table.clearSelection()
        panel._y_selector_table.blockSignals(False)

        out = panel.get_state()
        assert "Lambda" in out["selected_y_params"]

    def test_composite_parameters_round_trip(self, qapp):
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        panel = FitParametersPanel()
        state = {
            "rows": [
                {
                    "run_number": 101,
                    "field": 100.0,
                    "temperature": 5.0,
                    "values": {
                        "A0": 20.0,
                        "Lambda": 0.4,
                        "Lambda_eff": 20.004,
                    },
                    "errors": {
                        "A0": 0.5,
                        "Lambda": 0.02,
                        "Lambda_eff": 0.5,
                    },
                    "covariance": {
                        "A0": {"A0": 0.25, "Lambda": 0.0},
                        "Lambda": {"A0": 0.0, "Lambda": 0.0004},
                    },
                },
                {
                    "run_number": 102,
                    "field": 200.0,
                    "temperature": 5.1,
                    "values": {
                        "A0": 19.5,
                        "Lambda": 0.5,
                        "Lambda_eff": 19.506,
                    },
                    "errors": {
                        "A0": 0.4,
                        "Lambda": 0.03,
                        "Lambda_eff": 0.4,
                    },
                },
            ],
            "varying_params": ["A0", "Lambda"],
            "composite_parameters": [
                {
                    "name": "Lambda_eff",
                    "expression": "sqrt(A0^2 + Lambda^2)",
                }
            ],
            "inferred_x_key": "field",
            "x_axis": "Auto",
            "selected_y_params": ["Lambda_eff"],
            "log_x": False,
            "log_y_params": [],
            "plot_mode": "Single Axes",
        }

        panel.restore_state(state)
        out = panel.get_state()

        assert out["composite_parameters"][0]["name"] == "Lambda_eff"
        assert out["selected_y_params"] == ["Lambda_eff"]
        assert "covariance" in out["rows"][0]


class _StubFourierWithState(_StubFourier):
    def get_state(self):
        return {
            "window": "none",
            "filter_start_us": 0.0,
            "filter_time_constant_us": 1.5,
            "padding": 1,
            "phase_degrees": 0.0,
            "t0_offset_us": 0.0,
            "display": "(Power)^1/2",
            "auto_phase": False,
            "auto_phase_method": "Peak",
            "use_phase_table": False,
            "estimate_average_error": False,
            "group_enabled_table": {},
            "group_phase_table": {},
            "group_auto_filled_ids": [],
        }

    def restore_state(self, state):
        pass


class TestMainWindowProjectState:
    def test_project_round_trip_restores_grouping_and_plot_limits_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        source_file = tmp_path / "run6001.nxs"
        source_file.write_bytes(b"\x00")

        def _make_groupable_dataset() -> MuonDataset:
            from asymmetry.core.data.dataset import Run

            h0 = Histogram(counts=np.array([10.0, 20.0, 30.0, 40.0]), bin_width=1.0)
            h1 = Histogram(counts=np.array([5.0, 10.0, 15.0, 20.0]), bin_width=1.0)
            run = Run(
                run_number=6001,
                histograms=[h0, h1],
                source_file=str(source_file),
                grouping={
                    "groups": {1: [1], 2: [2]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 1.0,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                },
            )
            run.metadata["field"] = 100.0
            t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            return MuonDataset(
                time=t,
                asymmetry=np.zeros_like(t),
                error=np.ones_like(t),
                metadata={"title": "Run 6001", "temperature": 5.0, "field": 100.0, "comment": ""},
                run=run,
            )

        # Save-side window: apply grouping and custom axis limits.
        window1 = mw_module.MainWindow()
        if not getattr(window1._plot_panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")

        ds = _make_groupable_dataset()
        window1._data_browser.add_dataset(ds)
        window1._current_dataset = ds

        grouping_payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }
        applied, _ = window1._apply_grouping_settings_to_dataset(ds, grouping_payload)
        assert applied

        window1._plot_panel.plot_dataset(ds)
        window1._plot_panel._x_min.setValue(0.25)
        window1._plot_panel._x_max.setValue(2.75)
        window1._plot_panel._y_min.setValue(20.0)
        window1._plot_panel._y_max.setValue(45.0)
        window1._plot_panel._apply_limits()

        state = window1.collect_project_state()
        path = tmp_path / "roundtrip.asymp"
        save_project(state, path)
        loaded_state = load_project(path)

        # Restore-side window: load fresh dataset from file and restore state.
        def _stub_load_file(self_inner, path_str: str):
            assert path_str == str(source_file)
            return _make_groupable_dataset()

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)
        window2 = mw_module.MainWindow()
        if not getattr(window2._plot_panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")

        window2.restore_project_state(loaded_state, str(path))

        restored = window2._data_browser.get_dataset(6001)
        assert restored is not None

        # Grouping asymmetry: (F-B)/(F+B) = 1/3, then scaled to percent.
        assert np.allclose(
            restored.asymmetry, np.full_like(restored.asymmetry, 33.3333333333), atol=1e-6
        )
        assert restored.run is not None
        assert restored.run.grouping.get("forward_group") == 1
        assert restored.run.grouping.get("backward_group") == 2

        # The saved fit range still spans the full dataset, so restoring it
        # widens the visible x-limits back out to include that range.
        assert window2._plot_panel._x_min.value() == pytest.approx(0.0)
        assert window2._plot_panel._x_max.value() == pytest.approx(3.0)
        assert window2._plot_panel.get_fit_range() == pytest.approx((0.0, 3.0))
        assert window2._plot_panel._y_min.value() == pytest.approx(20.0)
        assert window2._plot_panel._y_max.value() == pytest.approx(45.0)

    def test_period_mapped_dataset_survives_reload(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        """A 3+-period mapped dataset reloads even though the file's per-period
        siblings are numbered by the loader's scheme and never match the saved
        combined run number — the persisted period_mapping rebuilds it."""
        from asymmetry.core.data.dataset import Run
        from asymmetry.core.io.periods import combine_mapped_periods

        source_file = tmp_path / "multiperiod.nxs"
        source_file.write_bytes(b"\x00")

        def _period_dataset(run_number: int, level: float) -> MuonDataset:
            h0 = Histogram(counts=np.full(4, level), bin_width=1.0)
            h1 = Histogram(counts=np.full(4, level + 1.0), bin_width=1.0)
            run = Run(
                run_number=run_number,
                histograms=[h0, h1],
                source_file=str(source_file),
                grouping={
                    "groups": {1: [1], 2: [2]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 1.0,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                    "good_frames": 1000.0,
                },
                metadata={"source_run_number": 900},
            )
            t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            return MuonDataset(
                time=t,
                asymmetry=np.zeros_like(t),
                error=np.ones_like(t),
                metadata={"source_run_number": 900},
                run=run,
            )

        def _periods() -> list[MuonDataset]:
            return [
                _period_dataset(9001, 10.0),
                _period_dataset(9002, 20.0),
                _period_dataset(9003, 30.0),
                _period_dataset(9004, 40.0),
            ]

        mapping = {1: "red", 2: "red", 3: "green", 4: "ignore"}
        mapped = combine_mapped_periods(
            _periods(), mapping, source_run_number=900, source_file=str(source_file)
        )

        window1 = mw_module.MainWindow()
        if not getattr(window1._plot_panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        window1._data_browser.add_dataset(mapped)
        state = window1.collect_project_state()
        path = tmp_path / "mapped.asymp"
        save_project(state, path)
        loaded_state = load_project(path)

        # The reloaded file yields the per-period siblings, not the combined run.
        def _stub_load_file(self_inner, path_str: str):
            return _periods()

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)
        window2 = mw_module.MainWindow()
        if not getattr(window2._plot_panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        window2.restore_project_state(loaded_state, str(path))

        restored = window2._data_browser.get_dataset(900)
        assert restored is not None, "mapped dataset was dropped on reload"
        assert restored.run is not None
        # period_mapping persisted and the combined structure was rebuilt.
        assert restored.run.grouping.get("period_mapping") == {
            "1": "red",
            "2": "red",
            "3": "green",
            "4": "ignore",
        }
        # Red set sums periods 1+2 detector-wise: group 1 forward = 10 + 20 = 30.
        from asymmetry.core.transform import group_forward_backward

        fb = group_forward_backward(restored.run.histograms, restored.run.grouping)
        np.testing.assert_array_equal(fb.forward, np.full(4, 30.0))

    def test_project_round_trip_restores_nexus_grouping_bunching(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        source_file = tmp_path / "run6002.nxs"
        source_file.write_bytes(b"\x00")

        def _make_nexus_dataset() -> MuonDataset:
            from asymmetry.core.data.dataset import Run

            run = Run(
                run_number=6002,
                histograms=[],
                source_file=str(source_file),
                grouping={
                    "groups": {
                        1: [(1, 100), (2, 100)],
                        2: [(3, 100), (4, 100)],
                    },
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 1.0,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                },
            )
            run.metadata["field"] = 50.0
            t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            a = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
            e = np.full_like(t, 1.0)
            return MuonDataset(
                time=t,
                asymmetry=a,
                error=e,
                metadata={"title": "Run 6002", "temperature": 5.0, "field": 50.0, "comment": ""},
                run=run,
            )

        window1 = mw_module.MainWindow()
        ds = _make_nexus_dataset()
        window1._data_browser.add_dataset(ds)
        window1._current_dataset = ds

        grouping_payload = {
            "groups": {
                1: [(1, 100), (2, 100)],
                2: [(3, 100), (4, 100)],
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 2,
            "deadtime_correction": False,
        }
        applied, _ = window1._apply_grouping_settings_to_dataset(ds, grouping_payload)
        assert applied is True
        assert len(ds.time) == 2

        state = window1.collect_project_state()
        path = tmp_path / "roundtrip_nexus.asymp"
        save_project(state, path)
        loaded_state = load_project(path)

        def _stub_load_file(self_inner, path_str: str):
            assert path_str == str(source_file)
            return _make_nexus_dataset()

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)
        window2 = mw_module.MainWindow()
        window2.restore_project_state(loaded_state, str(path))

        restored = window2._data_browser.get_dataset(6002)
        assert restored is not None
        assert len(restored.time) == 2
        assert restored.run is not None
        assert restored.run.grouping["bunching_factor"] == 2
        assert restored.run.grouping["groups"] == {
            1: [(1, 100), (2, 100)],
            2: [(3, 100), (4, 100)],
        }

    def test_collect_project_state_structure(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        state = window.collect_project_state()

        assert state["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "datasets" in state
        assert "combined_datasets" in state
        assert "browser_state" in state
        assert "plot_state" in state
        assert "single_fit_state" in state
        assert "global_fit_state" in state
        assert "fit_ui_state" in state
        assert "fit_parameters_state" in state
        assert "global_parameter_fit_window_state" in state
        assert "fourier_state" in state
        assert "workspace_state" in state["plot_state"]
        assert "frequency_plot_state" in state["plot_state"]

    def test_restore_project_state_all_mode_renders_subplots_immediately(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        source_file = tmp_path / "run6101.nxs"
        source_file.write_bytes(b"\x00")

        def _make_vector_dataset() -> MuonDataset:
            from asymmetry.core.data.dataset import Run

            h0 = Histogram(counts=np.array([12.0, 11.0, 10.0, 9.0]), bin_width=1.0)
            h1 = Histogram(counts=np.array([8.0, 7.0, 6.0, 5.0]), bin_width=1.0)
            run = Run(
                run_number=6101,
                histograms=[h0, h1],
                source_file=str(source_file),
                grouping={
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
                    "alpha": 1.0,
                    "first_good_bin": 0,
                    "last_good_bin": 3,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                },
            )
            t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            return MuonDataset(
                time=t,
                asymmetry=np.zeros_like(t),
                error=np.ones_like(t),
                metadata={"run_number": 6101, "field": 100.0},
                run=run,
            )

        def _stub_load_file(self_inner, path_str: str):
            assert path_str == str(source_file)
            return _make_vector_dataset()

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)

        window = mw_module.MainWindow()
        if not getattr(window._plot_panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")

        grouping_payload = {
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
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 6101,
                "source_file": str(source_file),
                "metadata_overrides": {"field": 100.0},
                "grouping_overrides": grouping_payload,
            }
        ]
        state["plot_state"]["current_run_number"] = 6101
        state["plot_state"]["polarization_axis"] = "ALL"

        window.restore_project_state(state, str(tmp_path / "project.asymp"))

        assert window._plot_panel.get_current_polarization_axis() == "ALL"
        assert len(window._plot_panel._subplot_axes_by_polarization) == 3

    def test_collect_project_state_includes_global_parameter_fit_window_state(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()

        class _StateWindow:
            def get_state(self):
                return {"fit_share_x": True}

        window._global_parameter_fit_window = _StateWindow()
        state = window.collect_project_state()

        assert state["global_parameter_fit_window_state"] == {"fit_share_x": True}

    def test_collect_project_state_round_trips_to_file(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        state = window.collect_project_state()
        path = tmp_path / "test.asymp"
        save_project(state, path)
        loaded = load_project(path)
        assert loaded["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_collect_project_state_prunes_regenerated_fit_and_wizard_cache_payloads(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        class _StubPlotPanelBulkyState(_StubPlotPanelWithState):
            def get_state(self):
                state = dict(super().get_state())
                state.update(
                    {
                        "fit_curve": {"t": [0.0, 1.0], "y": [0.2, 0.1], "label": "Fit"},
                        "fit_curve_run_number": 1001,
                        "fit_curves": {"1001": {"t": [0.0, 1.0], "y": [0.2, 0.1], "label": "Fit"}},
                        "fit_curves_by_key": {
                            "1001::P_x": {"t": [0.0, 1.0], "y": [0.2, 0.1], "label": "Fit"}
                        },
                        "fit_components": [{"name": "Signal", "y": [0.2, 0.1]}],
                        "fit_components_by_run": {"1001": [{"name": "Signal", "y": [0.2, 0.1]}]},
                        "fit_components_by_key": {
                            "1001::P_x": [{"name": "Signal", "y": [0.2, 0.1]}]
                        },
                    }
                )
                return state

        class _StubFitPanelWizardState(_StubFitPanelWithState):
            def get_single_state(self):
                return {
                    "model_name": "ExponentialRelaxation",
                    "parameters": [],
                    "wizard_state": {"log_text": "cached single wizard"},
                    "states_by_run": {
                        "1001": {
                            "model_name": "ExponentialRelaxation",
                            "parameters": [],
                            "wizard_state": {"log_text": "cached nested wizard"},
                        }
                    },
                }

            def get_global_state(self):
                return {
                    "model_name": "ExponentialRelaxation",
                    "parameters": [],
                    "wizard_state": {"log_text": "cached global wizard"},
                    "wizard_state_by_run_set": [{"log_text": "cached global cache"}],
                }

        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWizardState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelBulkyState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        state = window.collect_project_state()

        assert state["plot_state"]["fit_curve"] is None
        assert state["plot_state"]["fit_curves"] == {}
        assert state["plot_state"]["fit_curves_by_key"] == {}
        assert state["plot_state"]["fit_components"] is None
        assert state["plot_state"]["fit_components_by_run"] == {}
        assert state["plot_state"]["fit_components_by_key"] == {}
        assert "wizard_state" not in state["single_fit_state"]
        assert "wizard_state" not in state["single_fit_state"]["states_by_run"]["1001"]
        assert "wizard_state" not in state["global_fit_state"]
        assert "wizard_state_by_run_set" not in state["global_fit_state"]

    def test_collect_project_state_includes_sources_for_combined_datasets(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        class _StubDataBrowserCombined(_StubDataBrowserWithState):
            def __init__(self):
                super().__init__()
                # Combined run entry currently visible in browser.
                self._datasets = {-1: _make_dataset(-1)}
                self._combined_datasets = {-1: [3039, 3040]}

                # Original source runs retained only in combined-source cache.
                source_a = _make_dataset(3039)
                source_b = _make_dataset(3040)
                self._combined_source_datasets = {-1: [source_a, source_b]}

        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserCombined)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        state = window.collect_project_state()

        saved_runs = {entry["run_number"] for entry in state["datasets"]}
        assert 3039 in saved_runs
        assert 3040 in saved_runs
        # Combined dataset definitions are still persisted separately.
        assert state["combined_datasets"] == [
            {"combined_run_number": -1, "source_run_numbers": [3039, 3040]}
        ]

    def test_collect_project_state_includes_grouping_overrides(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        class _StubDataBrowserGrouping(_StubDataBrowserWithState):
            def __init__(self):
                super().__init__()
                ds = _make_dataset(4042)
                ds.run.grouping = {
                    "groups": {1: [1, 2], 2: [3, 4]},
                    "included_groups": {1: True, 2: False},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": 1.23,
                    "first_good_bin": 5,
                    "last_good_bin": 55,
                    "bunching_factor": 4,
                    "deadtime_correction": True,
                    "instrument": "MuSR",
                }
                self._datasets = {4042: ds}

        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserGrouping)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        state = window.collect_project_state()
        entry = next(e for e in state["datasets"] if e["run_number"] == 4042)

        grouping = entry.get("grouping_overrides")
        assert grouping is not None
        assert grouping["forward_group"] == 1
        assert grouping["backward_group"] == 2
        assert grouping["included_groups"] == {1: True, 2: False}
        assert grouping["bunching_factor"] == 4
        assert grouping["instrument"] == "MuSR"

    def test_collect_project_state_includes_multi_group_fit_state(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)
        monkeypatch.setattr(
            mw_module,
            "MultiGroupFitWindow",
            _StubMultiGroupFitWindowWithState,
        )

        window = mw_module.MainWindow()
        assert isinstance(window._multi_group_fit_window, _StubMultiGroupFitWindowWithState)
        window._multi_group_fit_window._state = {
            "model_name": "Composite",
            "parameters": [{"name": "A_1", "value": 0.25, "type": "Free", "bounds": "0, 1"}],
            "result_html": "<b>Saved grouped fit</b>",
            "wizard_state": {"should": "be pruned"},
        }

        state = window.collect_project_state()

        assert "multi_group_fit_state" in state
        assert state["multi_group_fit_state"]["result_html"] == "<b>Saved grouped fit</b>"
        assert "wizard_state" not in state["multi_group_fit_state"]

    def test_restore_project_state_selects_matching_dataset_from_multiperiod_load(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        file_path = tmp_path / "multi.nxs"
        file_path.write_bytes(b"\x00")

        ds1 = _make_dataset(51001)
        ds2 = _make_dataset(51002)

        def _stub_load_file(self_inner, path):
            assert path == str(file_path)
            return [ds1, ds2]

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)

        window = mw_module.MainWindow()
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 51002,
                "source_file": str(file_path),
                "metadata_overrides": {"field": 123.0},
            }
        ]
        window.restore_project_state(state, str(tmp_path / "project.asymp"))

        assert 51002 in window._data_browser._datasets
        assert 51001 not in window._data_browser._datasets

    def test_restore_project_state_applies_grouping_overrides(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        file_path = tmp_path / "run42.nxs"
        file_path.write_bytes(b"\x00")

        grouping_payload = {
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "instrument": "HiFi",
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 10,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        def _stub_load_file(self_inner, path):
            assert path == str(file_path)
            return _make_dataset(42)

        applied_payloads: list[dict] = []

        def _stub_apply_grouping(self_inner, dataset, payload):
            applied_payloads.append(payload)
            return True, False

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)
        monkeypatch.setattr(
            mw_module.MainWindow,
            "_apply_grouping_settings_to_dataset",
            _stub_apply_grouping,
        )

        window = mw_module.MainWindow()
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 42,
                "source_file": str(file_path),
                "metadata_overrides": {"field": 100.0},
                "grouping_overrides": grouping_payload,
            }
        ]
        window.restore_project_state(state, str(tmp_path / "project.asymp"))

        assert applied_payloads == [grouping_payload]

    def test_restore_project_state_restores_multi_group_fit_state(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)
        monkeypatch.setattr(
            mw_module,
            "MultiGroupFitWindow",
            _StubMultiGroupFitWindowWithState,
        )

        file_path = tmp_path / "run43.nxs"
        file_path.write_bytes(b"\x00")

        def _stub_load_file(self_inner, path):
            assert path == str(file_path)
            return _make_dataset(43)

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)

        window = mw_module.MainWindow()
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 43,
                "source_file": str(file_path),
                "metadata_overrides": {"field": 100.0},
            }
        ]
        state["multi_group_fit_state"] = {
            "model_name": "Composite",
            "parameters": [{"name": "A_1", "value": 0.33, "type": "Free", "bounds": "0, 1"}],
            "result_html": "<b>Saved grouped fit</b>",
        }

        window.restore_project_state(state, str(tmp_path / "project.asymp"))

        assert isinstance(window._multi_group_fit_window, _StubMultiGroupFitWindowWithState)
        assert window._multi_group_fit_window.restored_state == state["multi_group_fit_state"]

    def test_clear_all_state_clears_browser_and_panel(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        window._current_dataset = _make_dataset(42)

        window._clear_all_state()

        assert window._current_dataset is None
        assert window._data_browser._was_cleared

    def test_restore_project_state_with_missing_file_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        # Suppress the "locate directory" dialog – user clicks No.
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No)
        )

        window = mw_module.MainWindow()

        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 42,
                "source_file": "/nonexistent/path/run42.nxs",
                "metadata_overrides": {"field": 100.0},
            }
        ]
        project_path = str(tmp_path / "test.asymp")
        window.restore_project_state(state, project_path)

        # A warning should have been logged
        warnings = [m for m in window._log_panel.messages if "WARNING" in m]
        assert warnings, "Expected at least one WARNING log message for missing file"

    def test_restore_project_state_locates_moved_files(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        """When files are missing, choosing a fallback directory should load them."""
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        # Create a real data file in a "new" directory (simulating a moved file).
        new_data_dir = tmp_path / "moved_data"
        new_data_dir.mkdir()
        fake_nexus = new_data_dir / "run42.nxs"
        fake_nexus.write_bytes(b"\x00")  # empty placeholder

        # Suppress the QMessageBox (user clicks Yes to locate directory).
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        monkeypatch.setattr(
            QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes)
        )
        # Return the new directory from the dialog.
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: str(new_data_dir))
        )
        # Stub _load_file so we don't need real NeXus parsing – just record the path.
        loaded_paths: list[str] = []

        def _stub_load_file(self_inner, path):
            loaded_paths.append(path)
            return None  # return None so nothing is added to the browser

        monkeypatch.setattr(mw_module.MainWindow, "_load_file", _stub_load_file)

        window = mw_module.MainWindow()
        state = _minimal_state()
        state["datasets"] = [
            {
                "run_number": 42,
                "source_file": "/original/data/run42.nxs",
                "metadata_overrides": {"field": 100.0},
            }
        ]
        project_path = str(tmp_path / "test.asymp")
        window.restore_project_state(state, project_path)

        # _load_file should have been called with the resolved path in the new directory.
        assert len(loaded_paths) == 1
        assert os.path.basename(loaded_paths[0]) == "run42.nxs"
        assert str(new_data_dir) in loaded_paths[0]

    def test_restore_project_state_opens_fit_and_params_docks_when_results_exist(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsClear)

        window = mw_module.MainWindow()
        assert window._dock_fit.isHidden()
        assert window._dock_fit_parameters.isHidden()

        state = _minimal_state()
        state["single_fit_state"]["result_html"] = "<b>Saved fit result</b>"
        state["fit_parameters_state"] = {
            "rows": [
                {"run_number": 1, "field": 100.0, "temperature": 5.0, "values": {}, "errors": {}}
            ]
        }

        project_path = str(tmp_path / "test.asymp")
        window.restore_project_state(state, project_path)

        assert not window._dock_fit.isHidden()
        assert not window._dock_fit_parameters.isHidden()

    def test_restore_project_state_restores_global_parameter_fit_window_state(
        self, monkeypatch: pytest.MonkeyPatch, qapp: QApplication, tmp_path
    ) -> None:
        monkeypatch.setattr(mw_module, "DataBrowserPanel", _StubDataBrowserWithState)
        monkeypatch.setattr(mw_module, "FitPanel", _StubFitPanelWithState)
        monkeypatch.setattr(mw_module, "PlotPanel", _StubPlotPanelWithState)
        monkeypatch.setattr(mw_module, "LogPanel", _StubLogPanel)
        monkeypatch.setattr(mw_module, "FourierPanel", _StubFourierWithState)
        monkeypatch.setattr(mw_module, "FitParametersPanel", _StubFitParamsWithCrossGroup)

        class _StubGlobalParameterFitWindow:
            def __init__(self, _parent):
                self.restored_state = None

            def set_results(self, **_kwargs):
                return

            def restore_state(self, state):
                self.restored_state = state

            def show(self):
                return

            def raise_(self):
                return

            def activateWindow(self):
                return

            def has_result(self):
                return True

            def close(self):
                return

        monkeypatch.setattr(mw_module, "GlobalParameterFitWindow", _StubGlobalParameterFitWindow)

        window = mw_module.MainWindow()
        state = _minimal_state()
        state["global_parameter_fit_window_state"] = {
            "show_components": True,
            "fit_share_x": True,
        }

        project_path = str(tmp_path / "test.asymp")
        window.restore_project_state(state, project_path)

        assert window._global_parameter_fit_window is not None
        assert window._global_parameter_fit_window.restored_state == {
            "show_components": True,
            "fit_share_x": True,
        }


class TestNonFiniteJsonSafety:
    """Non-finite floats (NaN / ±Infinity) must round-trip through the project
    file as valid *strict* JSON, while in-app load behaviour is unchanged."""

    def _state_with_non_finite(self) -> dict:
        state = _minimal_state()
        # Unbounded parameter bounds (the pervasive ±inf case) + a NaN
        # uncertainty + a non-finite value inside a list.
        state["fit_parameters_state"] = {
            "rows": [],
            "model_fits": {
                "A": {
                    "parameter_name": "A",
                    "x_key": "temperature",
                    "active": True,
                    "ranges": [
                        {
                            "x_min": None,
                            "x_max": None,
                            "windows": None,
                            "model": {"component_names": ["Linear"], "operators": []},
                            "parameters": [
                                {
                                    "name": "m",
                                    "value": 0.5,
                                    "min": -float("inf"),
                                    "max": float("inf"),
                                    "fixed": False,
                                }
                            ],
                            "result": {
                                "success": True,
                                "chi_squared": 1.0,
                                "reduced_chi_squared": 1.0,
                                "message": "",
                                "error_mode": "column",
                                "n_points": 5,
                                "parameters": [
                                    {
                                        "name": "m",
                                        "value": 0.5,
                                        "min": -float("inf"),
                                        "max": float("inf"),
                                        "fixed": False,
                                    }
                                ],
                                "uncertainties": {"m": float("nan")},
                            },
                        }
                    ],
                }
            },
        }
        state["_non_finite_list"] = [1.0, float("nan"), float("inf"), -float("inf"), 2.0]
        return state

    def test_saved_file_is_strict_json(self, tmp_path):
        path = tmp_path / "nonfinite.asymp"
        save_project(self._state_with_non_finite(), path)
        text = path.read_text(encoding="utf-8")

        # A strict parser (one that rejects the NaN/Infinity barewords) must
        # accept the file.
        def _reject(token):
            raise AssertionError(f"non-standard JSON token written: {token}")

        json.loads(text, parse_constant=_reject)
        # And it re-serialises under allow_nan=False (no non-finite floats left).
        json.dumps(json.loads(text), allow_nan=False)

    def test_non_finite_round_trips_to_exact_floats(self, tmp_path):
        path = tmp_path / "nonfinite.asymp"
        save_project(self._state_with_non_finite(), path)
        loaded = load_project(path)

        rng = loaded["fit_parameters_state"]["model_fits"]["A"]["ranges"][0]
        pmin = rng["parameters"][0]["min"]
        pmax = rng["parameters"][0]["max"]
        unc = rng["result"]["uncertainties"]["m"]
        assert math.isinf(pmin) and pmin < 0  # -inf bound preserved
        assert math.isinf(pmax) and pmax > 0  # +inf bound preserved
        assert math.isnan(unc)  # NaN uncertainty preserved as a real float

        lst = loaded["_non_finite_list"]
        assert lst[0] == 1.0 and lst[4] == 2.0
        assert math.isnan(lst[1])
        assert math.isinf(lst[2]) and lst[2] > 0
        assert math.isinf(lst[3]) and lst[3] < 0

    def test_legitimate_string_tokens_are_not_converted(self, tmp_path):
        """A genuine string value equal to 'NaN'/'Infinity' must survive as a
        string (the wrapper is a uniquely-keyed object, not a bare string)."""
        state = _minimal_state()
        state["browser_state"]["filters"] = {"title": "NaN", "note": "Infinity"}
        path = tmp_path / "strings.asymp"
        save_project(state, path)
        loaded = load_project(path)
        assert loaded["browser_state"]["filters"]["title"] == "NaN"
        assert loaded["browser_state"]["filters"]["note"] == "Infinity"
