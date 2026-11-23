"""Opaque extension decoders replay only decisions that leave their output unchanged."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    resolve_instrument,
)
from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io import load
from asymmetry.core.io.base import BaseLoader, LoaderRegistry
from asymmetry.core.io.run_preparation import ReplayState

pytestmark = pytest.mark.io


@pytest.fixture(params=["old", "kwargs"])
def extension(monkeypatch, request):
    decoded = []

    class OldLoader(BaseLoader):
        def load(self, filepath):
            run = Run(
                run_number=42,
                source_file=str(filepath),
                metadata=dict(metadata),
                histograms=[Histogram(counts=np.array([2.0, 3.0, 5.0]), bin_width=0.02, t0_bin=1)],
                grouping={"groups": {7: [1]}, "t0_bin": 1, "first_good_bin": 1},
                source_assessment=capture[0],
            )
            dataset = MuonDataset(
                time=np.array([0.01, 0.03]),
                asymmetry=np.array([12.0, 13.0]),
                error=np.array([0.3, 0.4]),
                run=run,
                metadata=run.metadata,
            )
            decoded.append(dataset)
            return dataset

    class KeywordLoader(OldLoader):
        def load(self, filepath, **kwargs):
            return super().load(filepath)

    metadata = {"instrument": "EMU"}
    capture = [None]
    monkeypatch.setitem(
        LoaderRegistry._loaders, ".old", KeywordLoader if request.param == "kwargs" else OldLoader
    )
    return metadata, capture, decoded


@pytest.mark.parametrize("pulsed", [None, True, False])
def test_old_extension_fresh_capture_and_semantic_noop_reopen(extension, pulsed):
    metadata, _, decoded = extension
    if pulsed is not None:
        metadata["source_pulsed"] = pulsed
    fresh = load("run.old")
    assert fresh.run.source_assessment.pulsed is pulsed
    assert fresh.run.source_assessment.origin != "profile"
    saved = ReplayState(
        source=fresh.run.source_assessment,
        instrument_choice=fresh.run.instrument_choice,
        run_number=42,
    )
    reopened = load("run.old", replay=saved)
    assert reopened.run.source_assessment == saved.source
    assert reopened.run.instrument_choice == saved.instrument_choice
    for field in ("time", "asymmetry", "error"):
        assert vars(reopened)[field] is vars(decoded[-1])[field]
    np.testing.assert_array_equal(reopened.run.histograms[0].counts, [2.0, 3.0, 5.0])
    assert reopened.run.histograms[0].t0_bin == 1
    assert reopened.run.grouping == {"groups": {7: [1]}, "t0_bin": 1, "first_good_bin": 1}


def test_old_extension_legacy_replay_and_equal_value_historical_provenance(extension):
    metadata, capture, _ = extension
    metadata["source_pulsed"] = True
    legacy = load("run.old", replay=ReplayState(run_number=42))
    assert legacy.run.source_assessment.pulsed is True
    assert legacy.run.source_assessment.origin == "legacy:explicit"
    historical = SourceAssessment(True, "known", "profile", "isis.emu", "historical")
    choice = legacy.run.instrument_choice
    old_choice = replace(choice, selection=replace(choice.selection, profile_revision="historical"))
    replayed = load("run.old", replay=ReplayState(source=historical, instrument_choice=old_choice))
    assert replayed.run.source_assessment is historical
    assert replayed.run.instrument_choice is old_choice
    capture[0] = SourceAssessment(None, "unsupported", "decoder")
    typed = load("run.old")
    assert typed.run.source_assessment is capture[0]
    unknown = SourceAssessment(None, "absent", "project")
    assert load("run.old", replay=ReplayState(source=unknown)).run.source_assessment is unknown


@pytest.mark.parametrize("contrary", ["source", "identity", "acquisition", "ambiguous"])
def test_old_extension_rejects_decisions_that_require_redecoding(extension, contrary):
    metadata, _, _ = extension
    metadata["source_pulsed"] = True
    fresh = load("run.old")
    selected = resolve_instrument(fresh.run).selection
    source = fresh.run.source_assessment
    if contrary == "source":
        source = SourceAssessment(False, "known", "project")
    elif contrary == "identity":
        selected = InstrumentSelection("psi.gps", selected.acquisition, "1", "project")
    elif contrary == "acquisition":
        selected = replace(selected, acquisition="other")
    else:
        metadata["beamline"] = "GPS"
    choice = RecordedInstrumentChoice(selected, ())
    with pytest.raises(ValueError, match="loader.*captured.*replay|captured.*replay.*loader"):
        load("run.old", replay=ReplayState(source=source, instrument_choice=choice))
