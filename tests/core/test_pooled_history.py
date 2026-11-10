"""Changed G+R numerical identities keep saved arrays separate from fresh replay."""

import json
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.preparation import (
    WIZARD_PREPARATION,
    histogram_observation,
    prepare_wizard_dataset,
    wizard_observation_identity,
    wizard_observation_matches,
    wizard_rebuild_dataset,
)
from tests.core.test_pooled_periods import two_periods


def test_changed_pooled_identity_blocks_historical_score_reuse_and_reconstruction(monkeypatch):
    import asymmetry.core.transform.preparation as preparation

    run = two_periods()
    with monkeypatch.context() as old:
        old.setattr(preparation, "period_statistic_policy", lambda grouping: {})
        source = histogram_observation(run)
        historical = MuonDataset(
            np.arange(8) * 0.1, np.full(8, 100.0), np.ones(8), run=run, histogram_observation=source
        )
        signature = json.dumps(
            {
                "wizard_preparation": WIZARD_PREPARATION,
                "observation": wizard_observation_identity(historical),
            }
        )
    assert not wizard_observation_matches(historical, signature)
    with pytest.raises(ValueError, match="observation changed"):
        wizard_rebuild_dataset(historical, 2, signature)
    np.testing.assert_array_equal(historical.asymmetry, 100)
    # Curve-only historical arrays remain readable and merge as explicit curves.
    curves = replace(historical, histogram_observation=None)
    np.testing.assert_array_equal(prepare_wizard_dataset(curves, 2).asymmetry, 100)
    current = histogram_observation(run)
    assert current.identity != source.identity


@pytest.mark.parametrize("mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_scoped_policy_in_observation_workdir_fourier_and_projection(mode, monkeypatch, tmp_path):
    import asymmetry.core.fourier.spectrum as spectrum
    import asymmetry.core.transform.projections as projections
    import asymmetry.core.workflow.workdir as workdir
    from asymmetry.core.workflow.reduction import ReductionSettings

    run = two_periods()
    run.grouping["period_mode"] = mode
    path = tmp_path / "source"
    path.write_bytes(b"synthetic source identity")

    def identities():
        return (
            spectrum.fourier_grouping_digest(run, "fb_asymmetry"),
            projections.reduction_identity(run.grouping),
            workdir.reduction_digest(
                source_files=[path], grouping={}, settings=ReductionSettings(period=mode)
            ),
        )

    current = identities()
    for owner in (spectrum, projections, workdir):
        monkeypatch.setattr(owner, "period_statistic_policy", lambda grouping: {})
    old = identities()
    assert [a != b for a, b in zip(old, current, strict=True)] == [mode == "green_plus_red"] * 3


def test_pooled_baseline_selector_and_historical_linear_snapshot_remain_distinct():
    from asymmetry.core.transform.rotating_frame import (
        FrameSnapshot,
        Provenance,
        RotatingFrame,
        period_weights,
    )

    frame = RotatingFrame.typed_frequency(1, 2)
    frame = frame.with_baseline(0, Provenance.TYPED, 2, 3).with_baseline(1, Provenance.TYPED, 2, 3)
    old = FrameSnapshot(frame, (1.0, 1.0), "old reduction")
    current = FrameSnapshot(frame, period_weights("green_plus_red", 2), "current reduction")
    restored = FrameSnapshot.from_dict(json.loads(json.dumps(old.to_dict())))
    assert restored.rotation == old.rotation
    assert restored.rotation[5] == (4.0, 6.0)
    assert current.rotation[5] == (2.0, 3.0)
    assert not old.same_rotation(current)


@pytest.mark.parametrize(
    "field,value",
    [("t0_bin", 2), ("t0_time_us", 0.025), ("good_bin_start", 2), ("good_bin_end", 5)],
)
def test_ordinary_green_support_invalidates_fourier_and_projection(field, value):
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest
    from asymmetry.core.transform.projections import reduction_identity

    run = two_periods()
    before = (fourier_grouping_digest(run, "fb_asymmetry"), reduction_identity(run.grouping))
    run.grouping["period_histograms"][1] = [
        replace(h, **{field: value}) for h in run.grouping["period_histograms"][1]
    ]
    after = (fourier_grouping_digest(run, "fb_asymmetry"), reduction_identity(run.grouping))
    assert all(a != b for a, b in zip(before, after, strict=True))


def test_historical_project_and_workdir_outputs_load_without_reduction(tmp_path, monkeypatch):
    import asymmetry.core.workflow.workdir as workdir
    from asymmetry.core.project import CURRENT_SCHEMA_VERSION, load_project, save_project
    from asymmetry.core.transform.rotating_frame import FrameSnapshot, RotatingFrame
    from asymmetry.core.workflow.reduction import ReductionSettings

    frame = FrameSnapshot(RotatingFrame.typed_frequency(1, 2), (1.0, 1.0), "historical sum")
    historical = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [],
        "plot_state": {"fit_curve": {"time": [0.0, 1.0], "asymmetry": [100.0, 100.0]}},
        "saved_science": {
            "chi_squared": 12.5,
            "frame": frame.to_dict(),
            "period_mode": "green_plus_red",
        },
    }
    path = tmp_path / "historical.asymp"
    save_project(historical, path)
    assert load_project(path) == historical
    run = two_periods()
    ds = MuonDataset(np.arange(8) * 0.1, np.full(8, 100.0), np.ones(8), run=run)
    settings = ReductionSettings(period="green_plus_red")
    source = tmp_path / "source"
    source.write_bytes(b"synthetic historical source")
    with monkeypatch.context() as old:
        old.setattr(workdir, "period_statistic_policy", lambda grouping: {})
        digest = workdir.reduction_digest(source_files=[source], grouping={}, settings=settings)
    entry = workdir.ReducedEntry(
        1,
        digest,
        str(source),
        8,
        settings,
        {"run_number": 1, "chi_squared": 12.5},
        1.0,
        "off",
        1,
        2,
        [],
    )
    directory = workdir.WorkDir(tmp_path / "work")
    directory.write_reduced(ds, entry)
    restored = directory.reduced(1)
    np.testing.assert_array_equal(restored.asymmetry, 100)
    assert restored.metadata["chi_squared"] == 12.5
    assert restored.run is None
    current = workdir.reduction_digest(source_files=[source], grouping={}, settings=settings)
    assert not directory.is_current(1, current)
