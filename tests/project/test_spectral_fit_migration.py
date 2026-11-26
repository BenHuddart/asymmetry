"""Historical spectral results retain their values and an unknown weight policy."""

import pytest

from asymmetry.core.fitting.fit_log import FitLog
from asymmetry.core.fitting.spectral import default_frequency_model
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel


@pytest.mark.parametrize("schema_version", [42, 43])
def test_unknown_historical_spectral_weights_preserve_saved_statistics_and_parameters(
    tmp_path, schema_version
):
    record = {
        "objective": "gaussian",
        "success": True,
        "chi_squared": 17.0,
        "reduced_chi_squared": 1.7,
        "parameters": {"nu0": 2.7},
        "uncertainties": {"nu0": 0.125},
        "quality": {"verdict": "good", "dof": 10},
        "ndof": 10,
    }
    model = default_frequency_model().to_dict()
    historical = {
        "schema_version": schema_version,
        "datasets": [
            {
                "run_number": 77,
                "representations": {
                    "freq_fft": {
                        "single_fits": {
                            "": {
                                "open_id": "s1",
                                "fits": [
                                    {
                                        "fit_id": "s1",
                                        "model": model,
                                        "result": record.copy(),
                                        "provenance": "single",
                                        "ui_state": {"result_html": "χ²/ν 1.7 [good]"},
                                    }
                                ],
                            }
                        }
                    }
                },
            }
        ],
        "batches": [
            {
                "batch_id": "b1",
                "canonical_model": model,
                "member_run_numbers": [77],
                "rep_type": "freq_fft",
                "results_by_run": {"77": record.copy()},
            }
        ],
    }

    project = ProjectModel.from_project_state(historical)
    slot = project.representation(77, RepresentationType.FREQ_FFT).fit
    assert "Spectral weighting unknown" in slot.ui_state["result_html"]
    assert "[good]" not in slot.ui_state["result_html"]
    assert slot.result["historical_result_html"] == "χ²/ν 1.7 [good]"
    assert slot.data_key() is None
    path = tmp_path / "old-fit.asymp"
    state = {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [{"run_number": 77}]}
    project.write_to_project_state(state)
    save_project(state, path)

    reopened = ProjectModel.from_project_state(load_project(path))
    slot = reopened.representation(77, RepresentationType.FREQ_FFT).fit
    member = reopened.batch("b1").results_by_run[77]
    for result in (slot.result, member):
        assert result["input_attribution"]["spectral_weighting"]["policy"] == "unknown"
        assert result["quality"] is None
        for key in ("parameters", "uncertainties", "chi_squared", "reduced_chi_squared"):
            assert result[key] == record[key]
        report = FitLog().format_record(result)
        assert "Spectral weighting unknown" in report
        assert "Historical fit statistic = 17" in report
        assert "2.7 +/- 0.125" in report


def test_historical_fft_report_retains_a_reduced_value_with_ambiguous_degrees_of_freedom():
    result = {
        "objective": "gaussian",
        "success": True,
        "chi_squared": 8.0,
        "reduced_chi_squared": 8.0,
        "parameters": {"nu0": 2.7},
        "ndof": 1,
        "quality": {"verdict": "poor", "dof": 1},
    }
    older = {
        "schema_version": 41,
        "datasets": [{"run_number": 77}],
        "batches": [
            {
                "batch_id": "b1",
                "canonical_model": default_frequency_model().to_dict(),
                "member_run_numbers": [77],
                "rep_type": "freq_fft",
                "results_by_run": {"77": result},
            }
        ],
    }
    migrated = ProjectModel.from_project_state(older).batch("b1").results_by_run[77]
    assert migrated["ndof"] is None
    assert migrated["reduced_chi_squared"] is None
    assert migrated["historical_reduced_chi_squared"] == 8
    assert migrated["quality"] is None
    report = FitLog().format_record(migrated)
    assert "Historical fit statistic/nu = 8.0000" in report
    assert "Spectral weighting unknown" in report
