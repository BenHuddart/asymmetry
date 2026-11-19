"""Synthetic preset preparation agrees with the existing profile and reduction owners."""

import pickle
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instrument import get_instrument_layout, layout_detector_labels
from asymmetry.core.project.profiles import (
    GroupingProfile,
    profile_fingerprint_for_run,
    resolve_effective_grouping,
)
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
from asymmetry.core.transform.projections import reduce_selected_projections
from asymmetry.core.workflow import (
    CalibrationOptions,
    ReductionSettings,
    apply_grouping_preset,
    calibrate_run,
    reduce_run,
    resolve_reduction_grouping,
)


def source(instrument="GPS", count=None):
    layout = get_instrument_layout(instrument)
    labels = layout_detector_labels(layout)
    run = Run(
        run_number=42,
        histograms=[
            Histogram(np.arange(32, dtype=float) + 100 + i, 0.1, 2, 3, 29, 0.24)
            for i in range(count if count is not None else layout.n_detectors)
        ],
        metadata={"instrument": layout.display, "title": "Synthetic", "sample": {"name": "Ag"}},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "histogram_labels": [labels.get(i + 1, "") for i in range(layout.n_detectors)],
            "good_frames": 100000,
        },
    )
    return MuonDataset(np.array([]), np.array([]), np.array([]), metadata=run.metadata, run=run)


@pytest.mark.parametrize(
    "instrument,preset_name",
    [
        ("GPS", "Longitudinal"),
        ("GPS-RD", "Transverse (Vector)"),
        ("GPS-RD15", "Transverse (Vector)"),
        ("HiFi", "Transverse (Vector)"),
        ("EMU", "Vector Polarization"),
        ("FLAME", "Transverse"),
    ],
)
def test_complete_preset_matches_supported_profile_preparation(instrument, preset_name):
    data = source(instrument)
    data.run.grouping["excluded_detectors"] = [1] if instrument in {"HiFi", "EMU"} else []
    before = pickle.dumps(data)
    preset = get_instrument_layout(instrument).presets[preset_name]
    profile = GroupingProfile(
        "expected",
        profile_fingerprint_for_run(data.run),
        groups={gid: sorted(g.detector_ids) for gid, g in preset.groups.items() if g.detector_ids},
        group_names={gid: g.name for gid, g in preset.groups.items() if g.detector_ids},
        included_groups={gid: True for gid, g in preset.groups.items() if g.detector_ids},
        forward_group=preset.forward_group,
        backward_group=preset.backward_group,
        projections=[p.to_payload() for p in preset.projections],
        excluded_detectors=data.run.grouping["excluded_detectors"],
        extra={"grouping_preset": preset_name},
    )
    expected_grouping = resolve_effective_grouping(profile, data.run)
    expected_run = replace(data.run, grouping=expected_grouping)
    expected = prepare_histogram_asymmetry(expected_run, merge=1, facility=None, reference=None)
    actual = apply_grouping_preset(data, preset_name)
    assert actual.run.grouping == expected_grouping
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert actual.metadata["sample"] == {"name": "Ag"}
    assert actual.run.histograms is data.run.histograms
    assert actual.histogram_observation is not None
    assert actual.effective_recipe is not None
    assert pickle.dumps(data) == before
    if preset.projections:
        curves = reduce_selected_projections(actual.run)
        direct = reduce_selected_projections(expected_run)
        assert set(curves) == {p.label for p in preset.projections}
        for label in curves:
            np.testing.assert_array_equal(curves[label].asymmetry, direct[label].asymmetry)
            np.testing.assert_array_equal(curves[label].error, direct[label].error)


def test_gps_pair_orientation_and_calibration_reduction():
    prepared = apply_grouping_preset(source(), "Longitudinal")
    assert prepared.run.grouping["groups"] == {1: [1], 2: [2]}
    assert (prepared.run.grouping["forward_group"], prepared.run.grouping["backward_group"]) == (
        2,
        1,
    )
    outcome = calibrate_run(prepared, ReductionSettings(), CalibrationOptions("ratio"))
    reduced = reduce_run(prepared.run, ReductionSettings(alpha=outcome.result))
    assert reduced.n_points > 0
    assert np.all(np.isfinite(reduced.error))
    calibrated_run = replace(
        prepared.run,
        grouping=resolve_reduction_grouping(prepared.run, ReductionSettings(alpha=outcome.result)),
    )
    calibrated = replace(prepared, run=calibrated_run)
    same = apply_grouping_preset(calibrated, "Longitudinal")
    assert same.run.grouping["alpha_policy"] == calibrated.run.grouping["alpha_policy"]
    with pytest.raises(ValueError, match="active captured calibration"):
        apply_grouping_preset(calibrated, "Spin-rotated (B+U/F+D)")


@pytest.mark.parametrize(
    "instrument,count,preset,error",
    [
        ("GPS", 5, "Transverse (Vector)", "(channel|detector|histogram)"),
        ("GPS", None, "unknown", "unknown"),
    ],
)
def test_bad_preset_or_missing_channels_fail_without_mutation(instrument, count, preset, error):
    data = source(instrument, count)
    before = pickle.dumps(data)
    with pytest.raises((KeyError, ValueError), match=error):
        apply_grouping_preset(data, preset)
    assert pickle.dumps(data) == before


def test_conflicting_labels_and_empty_excluded_group_fail():
    data = source()
    data.run.grouping["histogram_labels"][:2] = ["Up", "Down"]
    with pytest.raises(ValueError, match="labels"):
        apply_grouping_preset(data, "Longitudinal")
    data = source()
    data.run.grouping["excluded_detectors"] = [1]
    with pytest.raises(ValueError, match="no detectors after exclusion"):
        apply_grouping_preset(data, "Longitudinal")


def test_unknown_instrument_and_curve_only_fail():
    data = source()
    data.run.metadata = {
        "instrument": "unknown",
        "facility": "PSI",
        "histogram_labels": ["unknown"] * 6,
    }
    with pytest.raises(ValueError, match="unrecognised instrument"):
        apply_grouping_preset(data, "Longitudinal")
    with pytest.raises(ValueError, match="acquired histograms"):
        apply_grouping_preset(replace(data, run=None), "Longitudinal")


@pytest.mark.parametrize("coefficient", ["alpha", "beta"])
def test_legacy_calibration_is_recognised_by_profile_policy(coefficient):
    data = source()
    data.run.grouping.update({coefficient: 1.1, f"{coefficient}_reference_run": 41})
    with pytest.raises(ValueError, match="active captured calibration"):
        apply_grouping_preset(data, "Longitudinal")


def test_projection_label_is_replaced_and_rotating_frame_requires_original():
    vector = apply_grouping_preset(source(), "Transverse (Vector)")
    projected = reduce_selected_projections(vector.run)["Up-Down"]
    assert projected.metadata["projection"] == "Up-Down"
    scalar = apply_grouping_preset(projected, "Longitudinal")
    assert "projection" not in scalar.metadata
    assert "projection" not in scalar.run.metadata
    assert projected.metadata["projection"] == "Up-Down"
    with pytest.raises(ValueError, match="laboratory-frame source"):
        apply_grouping_preset(
            replace(projected, metadata=dict(projected.metadata, rotating_frame={})),
            "Longitudinal",
        )


def test_invalid_exclusion_cannot_bypass_membership_validation():
    data = source()
    data.run.grouping["excluded_detectors"] = [99]
    with pytest.raises(ValueError, match="(channel|detector|histogram)"):
        apply_grouping_preset(data, "Longitudinal")


def test_held_period_caches_do_not_survive_new_membership():
    from asymmetry.core.io.periods import select_period

    data = source()
    stale = (np.arange(3.0), np.full(3, 99.0), np.ones(3))
    data.run.grouping.update(
        period_histograms=[data.run.histograms, data.run.histograms],
        period_reduced=[stale, stale],
    )
    prepared = apply_grouping_preset(data, "Longitudinal")
    assert "period_reduced" not in prepared.run.grouping
    red = select_period(prepared, "red")
    np.testing.assert_array_equal(red.asymmetry, prepared.asymmetry)
    assert "period_reduced" in data.run.grouping
    del data.run.grouping["period_histograms"]
    with pytest.raises(ValueError, match="raw histograms for all held periods"):
        apply_grouping_preset(data, "Longitudinal")


def test_automatic_alpha_retains_loaded_background_reference():
    from functools import partial

    from asymmetry.core.io import background_reference_for_grouping
    from asymmetry.core.project.profiles import profile_from_payload
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.preparation import histogram_observation
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    data = source()
    data.run.grouping.update(
        alpha_policy={"mode": "per_run_estimate"},
        background_correction=True,
        background_mode="reference_run",
        background_run={
            "run_number": 43,
            "scale_request": {"mode": "manual", "coefficient": 1.0},
        },
    )
    reference = BackgroundReference(
        [replace(h, counts=np.full(32, 10.0)) for h in data.run.histograms],
        ReferenceScaleRequest("manual", 1.0).resolve(),
        run_number=43,
        source_file="synthetic-reference.nxs",
    )
    data.histogram_observation = histogram_observation(data.run, reference=reference)
    prepared = apply_grouping_preset(data, "Longitudinal")
    expected = profile_from_payload(
        data.run.grouping, "expected", profile_fingerprint_for_run(data.run)
    )
    expected.forward_group, expected.backward_group = 2, 1
    grouping = resolve_effective_grouping(
        expected, data.run, reference_resolver=partial(background_reference_for_grouping, reference)
    )
    direct = prepare_histogram_asymmetry(
        replace(data.run, grouping=grouping), merge=1, facility=None, reference=reference
    )
    assert prepared.run.grouping["alpha"] == grouping["alpha"]
    np.testing.assert_array_equal(prepared.asymmetry, direct.asymmetry)
    np.testing.assert_array_equal(prepared.error, direct.error)
    assert prepared.histogram_observation.reference.run_number == 43
