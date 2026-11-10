"""Independent F/B uncertainties require disjoint acquired observations."""

from fractions import Fraction

import numpy as np
import pytest

from asymmetry.core.data.dataset import Run
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import InvalidGroupingError, group_forward_backward
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import corrected_grouped_counts, reduce_grouped_asymmetry
from tests.core.test_group_membership import histograms


@pytest.mark.parametrize(
    "groups", [{1: [1], 2: [1]}, {1: [1, 2], 2: [2, 3]}, {"1": [2, 1], "2": [3, 2]}]
)
@pytest.mark.parametrize(
    "boundary",
    ["group", "correction", "reduction", "profile_fixed", "profile_estimate", "integral"],
)
def test_overlap_is_unavailable_at_production_boundaries(groups, boundary):
    hist = histograms()
    grouping = {"groups": groups, "group_names": {1: "North", 2: "South"}, "alpha": 1.7}
    with pytest.raises(InvalidGroupingError, match="shared.*channel|channel.*shared") as error:
        if boundary == "group":
            group_forward_backward(hist, grouping)
        elif boundary.startswith("profile"):
            profile = GroupingProfile(
                name="sample",
                fingerprint=ProfileFingerprint("test", 3),
                groups={int(k): v for k, v in groups.items()},
                alpha_policy=AlphaPolicy(
                    mode="per_run_estimate" if boundary.endswith("estimate") else "fixed", value=1.7
                ),
            )
            resolve_effective_grouping(profile, Run(histograms=hist))
        elif boundary == "integral":
            from asymmetry.core.transform.integral import integrate_run

            integrate_run(Run(histograms=hist, grouping=grouping))
        else:
            options = dict(
                histograms=hist,
                grouping=grouping,
                forward_idx=[d - 1 for d in groups.get(1, groups.get("1"))],
                backward_idx=[d - 1 for d in groups.get(2, groups.get("2"))],
                use_deadtime=False,
                deadtime_mode="off",
                use_background=False,
            )
            if boundary == "correction":
                corrected_grouped_counts(**options, allow_incomplete=True)
            else:
                reduce_grouped_asymmetry(**options, binning=binning_spec(grouping), alpha=1.7)
    expected = {int(k): v for k, v in groups.items()} if boundary.startswith("profile") else groups
    assert error.value.grouping["groups"] == expected


def test_covariance_counterexample_from_independent_source_counts():
    """First-order asymmetry variance includes covariance from the shared channel."""
    # A=(F-B)/(F+B), F=X+Y, B=Y+Z; derivatives are with respect to X,Y,Z.
    x, y, z = map(Fraction, (10, 30, 50))
    f, b = x + y, y + z
    df, db = 2 * b / (f + b) ** 2, -2 * f / (f + b) ** 2
    independent = df**2 * f + db**2 * b
    source_variance = df**2 * x + (df + db) ** 2 * y + db**2 * z
    assert independent == Fraction(1, 135)
    assert source_variance == Fraction(1, 270)
    with pytest.raises(InvalidGroupingError):
        group_forward_backward(histograms(), {"groups": {1: [1, 2], 2: [2, 3]}})


def test_exclusion_repairs_overlap_without_changing_raw_counts_or_variances():
    hist = histograms()
    grouping = {"groups": {1: [1, 2], 2: [2, 3]}, "excluded_detectors": [2]}
    fb = group_forward_backward(hist, grouping)
    corrected = corrected_grouped_counts(
        histograms=hist,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[2],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    for result in (fb.counts, corrected):
        np.testing.assert_array_equal(result.forward.counts, np.full(20, 10.0))
        np.testing.assert_array_equal(result.backward.counts, np.full(20, 50.0))
        np.testing.assert_array_equal(result.forward.variance, np.full(20, 2.0))
        np.testing.assert_array_equal(result.backward.variance, np.full(20, 11.0))
    for h, c, v in zip(hist, [10, 30, 50], [2, 7, 11]):
        np.testing.assert_array_equal(h.counts, np.full(20, c))
        np.testing.assert_array_equal(h.count_variance, np.full(20, v))


def test_equal_valued_independent_channels_and_overlapping_auxiliary_spectra_work():
    hist = histograms()
    hist[1].counts = hist[0].counts.copy()
    grouping = {"groups": {1: [1], 2: [2], 3: [1, 2]}, "first_good_bin": 0, "last_good_bin": 19}
    fb = group_forward_backward(hist, grouping)
    np.testing.assert_array_equal(fb.forward, fb.backward)
    dataset = build_group_signal_dataset(Run(histograms=hist, grouping=grouping), 3)
    assert dataset.time.size


def test_emu_vector_pairs_allow_reuse_across_all_three_projections():
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.instrument import get_instrument_layout

    preset = get_instrument_layout("EMU").presets["Vector Polarization"]
    groups = {gid: list(group.detector_ids) for gid, group in preset.groups.items()}
    hist = [Histogram(np.full(20, float(i + 1)), 0.1) for i in range(96)]
    assert set(groups[1]).intersection(groups[3])
    assert len(preset.projections) == 3
    for projection in preset.projections:
        grouping = {
            "groups": groups,
            "forward_group": projection.forward_group,
            "backward_group": projection.backward_group,
        }
        fb = group_forward_backward(hist, grouping)
        assert fb.forward[0] == sum(groups[projection.forward_group])
        assert fb.backward[0] == sum(groups[projection.backward_group])


def test_count_inspection_does_not_invent_an_independent_pair_for_background():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.grouped_time_domain import build_count_group

    run = Run(
        histograms=histograms(),
        grouping={
            "groups": {1: [1], 2: [1, 2]},
            "first_good_bin": 0,
            "last_good_bin": 19,
            "background_correction": True,
            "background_mode": "fixed",
            "background_fixed_values": [1.0, 2.0],
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    group = build_count_group(dataset, 1, lifetime_corrected=False)
    np.testing.assert_array_equal(group.counts, np.full(20, 10.0))


def test_public_workflow_rejects_overlapping_custom_pair():
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    run = Run(
        histograms=histograms(),
        grouping={
            "groups": {1: [1, 2], 2: [2, 3]},
            "first_good_bin": 0,
            "last_good_bin": 19,
        },
    )
    with pytest.raises(InvalidGroupingError, match="shared acquired channels 2"):
        reduce_run(run, ReductionSettings())


def test_resolved_index_boundary_checks_supplied_observations():
    # The dictionary describes a different pair; the explicit selection is the input here.
    grouping = {"groups": {1: [1], 2: [3]}}
    with pytest.raises(InvalidGroupingError, match="shared acquired channels 2"):
        corrected_grouped_counts(
            histograms=histograms(),
            grouping=grouping,
            forward_idx=[0, 1],
            backward_idx=[1, 2],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )


@pytest.mark.parametrize("group_names", [None, [], {1: "North", 2: "South"}])
def test_optional_group_labels_do_not_prevent_actionable_pair_error(group_names):
    grouping = {"groups": {1: [1, 2], 2: [2, 3]}, "group_names": group_names}
    with pytest.raises(InvalidGroupingError, match="shared acquired channels 2"):
        group_forward_backward(histograms(), grouping)


def test_string_selected_group_ids_keep_integer_key_compatibility():
    grouped = group_forward_backward(
        histograms(),
        {
            "groups": {1: [1], 2: [3]},
            "forward_group": "1",
            "backward_group": "2",
        },
    )
    assert grouped.forward[0] == 10
    assert grouped.backward[0] == 50


def test_projection_error_names_the_selected_pair():
    from asymmetry.core.transform.projections import reduce_run_projections

    grouping = {
        "groups": {1: [1], 2: [3], 5: [1, 2], 6: [2, 3]},
        "group_names": {5: "Left", 6: "Right"},
        "projections": [{"label": "custom", "forward_group": 5, "backward_group": 6}],
    }
    with pytest.raises(InvalidGroupingError, match=r"Left \(5\) and Right \(6\).*channels 2"):
        reduce_run_projections(Run(histograms=histograms(), grouping=grouping), 0)
