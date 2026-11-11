"""Master labels and independently evidenced correspondence at public entries."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.combine import CombineError, coadd_compatibility_error, combine_runs
from asymmetry.core.transform.projections import reduction_identity
from tests.core.test_linear_run_arithmetic import source


def acquisition(number, temperature=None, labels=None):
    run = source(number)
    run.metadata.update(instrument="EMU", facility="ISIS", beamline="R80", title="sample")
    if temperature is not None:
        run.metadata["temperature"] = temperature
    if labels is not None:
        run.grouping["histogram_labels"] = labels
    return run


@pytest.mark.parametrize("scales", [[1, 1], [0, 2], [-1, 0.5]])
def test_first_authored_master_and_active_finite_ranges(scales):
    a, b = acquisition(1, 10), acquisition(2, 20)
    a.grouping["good_frames"], b.grouping["good_frames"] = 100, 300
    c = combine_runs([a, b], scales=scales)
    assert c.metadata["temperature"] == 10
    assert c.metadata["instrument"] == "EMU"
    assert c.metadata["facility"] == "ISIS"
    assert c.metadata["beamline"] == "R80"
    assert c.metadata["title"] == "sample"
    assert c.metadata["temperature_spread"] == ((20, 20) if scales[0] == 0 else (10, 20))
    assert [r["coefficient"] for r in c.metadata["combination"]["constituents"]] == scales
    assert c.source_file == ""
    assert "source_file" not in c.metadata


@pytest.mark.parametrize("missing", [None, "invalid", np.nan, np.inf])
def test_missing_master_stays_unknown(missing):
    c = combine_runs([acquisition(1, missing), acquisition(2, 20)])
    assert c.metadata.get("temperature") is None
    assert c.metadata["temperature_spread"] == (20, 20)


@pytest.mark.parametrize("sign", [1, -1])
def test_known_detector_role_conflict_in_core_and_preflight(sign):
    a = acquisition(1, labels=["Forw", "Back"])
    b = acquisition(2, labels=["Backward", "Forward"])
    with pytest.raises(CombineError, match="detector.*correspondence"):
        combine_runs([a, b], sign=sign)
    assert "correspondence" in coadd_compatibility_error([a, b])
    combine_runs([a, b], scales=[1, 0])


@pytest.mark.parametrize(
    "labels", [["Forward", "Backward"], ["sample", "holder"], ["", "Back"], None]
)
def test_matching_cosmetic_partial_or_unknown_labels_do_not_veto(labels):
    a, b = acquisition(1, labels=["Forw", "Back"]), acquisition(2, labels=labels)
    b.metadata["instrument"] = "different descriptive name"
    combine_runs([a, b])


def test_original_facts_cannot_be_hidden_by_composite_inspection_labels():
    a, b, c = [acquisition(i, labels=["Forward", "Backward"]) for i in range(3)]
    nested = combine_runs([a, b])
    b.grouping["histogram_labels"] = ["Backward", "Forward"]
    with pytest.raises(CombineError, match="correspondence"):
        combine_runs([nested, c])


def test_nested_zero_and_actual_reuse():
    a, b = acquisition(1), acquisition(2)
    combine_runs([combine_runs([a, b], scales=[1, 0]), b])
    with pytest.raises(CombineError, match="independent|reused"):
        combine_runs([combine_runs([a, b]), a])


def test_repeated_subtracted_reference_is_not_independent():
    a, b, r = [acquisition(i) for i in range(3)]
    with pytest.raises(CombineError, match="independent|reused"):
        combine_runs([combine_runs([a, r], sign=-1), combine_runs([b, r], sign=-1)])


def test_equal_numbers_and_values_do_not_establish_overlap():
    a = acquisition(1)
    combine_runs([a, acquisition(1)])
    b = replace(acquisition(1), source_file=a.source_file)
    combine_runs([a, b])


def test_nested_ranges_use_ordered_leaf_conditions_and_effective_coefficients():
    a, b, c = [acquisition(i, 10 * i) for i in range(1, 4)]
    nested = combine_runs([combine_runs([a, b], scales=[1, -2]), c], scales=[0.5, 0])
    records = nested.metadata["combination"]["constituents"]
    assert [r["run_number"] for r in records] == [1, 2, 3]
    assert [r["coefficient"] for r in records] == [0.5, -1, 0]
    assert nested.metadata["temperature"] == 10
    assert nested.metadata["temperature_spread"] == (10, 20)


def test_condition_labels_do_not_change_numerical_identity_but_roles_do():
    a, b = acquisition(1), acquisition(2)
    run = combine_runs([a, b])
    before = reduction_identity(run.grouping)
    a.metadata.update(temperature=17, title="new display label")
    assert reduction_identity(run.grouping) == before
    a.grouping["histogram_labels"] = ["Forward", "Backward"]
    assert reduction_identity(run.grouping) != before


def test_editing_retained_sources_cannot_hide_internal_known_overlap():
    a, b, c = [acquisition(i) for i in range(3)]
    a.source_file, b.source_file = "original.nxs", "other.nxs"
    nested = combine_runs([a, b])
    b.source_file = a.source_file
    with pytest.raises(CombineError, match="independent|reused"):
        combine_runs([nested, c])


def test_shared_target_reference_request_is_applied_once():
    from asymmetry.core.data.combine import reduce_combined_run
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    a, b, r = [acquisition(i) for i in range(3)]
    r.histograms = [replace(h, counts=np.full(12, 2)) for h in r.histograms]
    for run in (a, b):
        run.grouping.update(background_correction=True, background_mode="reference_run")
    reference = BackgroundReference(r.histograms, ReferenceScaleRequest("manual", 1).resolve())
    result = reduce_combined_run(combine_runs([a, b]), reference=reference)
    # One shared reference: F=20-2, B=40-2, rather than subtracting it twice.
    np.testing.assert_allclose(result.asymmetry, 100 * (18 - 38) / 56)


def test_workflow_source_uses_master_without_copying_acquisition_caches(monkeypatch):
    from asymmetry.core.data.combine import reduce_combined_run
    from asymmetry.core.workflow.reduction import load_reduction_source

    runs = [acquisition(1), acquisition(2, 20)]
    runs[0].metadata.update(source_file="original.nxs", stale_cached_diagnostic=17)
    datasets = [reduce_combined_run(run) for run in runs]
    monkeypatch.setattr("asymmetry.core.workflow.reduction.load", lambda path: datasets[int(path)])
    combined = load_reduction_source(["0", "1"], None)
    assert combined.metadata["temperature"] is None
    assert combined.metadata["instrument"] == "EMU"
    assert "source_file" not in combined.metadata
    assert "stale_cached_diagnostic" not in combined.metadata
