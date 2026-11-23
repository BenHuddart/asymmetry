"""Disk reference replay supplies its own captured authority before quick-look."""

import pytest

from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io import load_background_run, nexus
from tests.io.test_load_source_defaults import source_file

pytestmark = pytest.mark.io


@pytest.mark.parametrize(
    "source",
    [
        None,
        SourceAssessment(None, "absent", "project"),
        SourceAssessment(False, "known", "explicit"),
    ],
)
def test_reference_load_prepares_capture_before_quicklook(tmp_path, monkeypatch, source):
    path = tmp_path / "reference.nxs"
    source_file(path, file_anchor=2)
    observed = []
    compute = nexus.quick_asymmetry

    def quick(histograms, grouping, *, metadata):
        observed.append(metadata["source_assessment"])
        return compute(histograms, grouping, metadata=metadata)

    monkeypatch.setattr(nexus, "quick_asymmetry", quick)
    timing = {"t0_policy": {"mode": "from_file"}}
    if source is not None:
        timing["source"] = source.to_dict()
    dataset = load_background_run(
        {"source_file": str(path), "run_number": 42, "period_number": 1, "timing": timing}
    )
    expected = source or SourceAssessment(None, "absent", "legacy:metadata")
    assert observed == [expected]
    assert dataset.run.source_assessment == expected
    assert dataset.n_points > 0


def test_invalid_present_reference_capture_rejects_before_file_decode(tmp_path, monkeypatch):
    path = tmp_path / "reference.nxs"
    source_file(path)
    observed = []
    monkeypatch.setattr(nexus, "quick_asymmetry", lambda *args, **kwargs: observed.append(True))
    with pytest.raises(ValueError, match="source|Source"):
        load_background_run({"source_file": str(path), "timing": {"source": None}})
    assert observed == []
