"""Shared hardware event content cannot be treated as independent observations."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.count_inputs import build_count_groups
from asymmetry.core.fourier.group_preparation import prepare_group_counts
from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
from asymmetry.core.transform.reduce import corrected_grouped_counts


def gps_run(groups, *, excluded=()):
    return Run(
        histograms=[
            Histogram(
                counts=np.full(40, 100.0 + i),
                bin_width=0.1,
                t0_bin=0,
                good_bin_start=0,
                good_bin_end=39,
            )
            for i in range(15)
        ],
        metadata={"instrument": "GPS", "facility": "PSI"},
        grouping={
            "instrument": "GPS-RD15",
            "groups": groups,
            "forward_group": 1,
            "backward_group": 2,
            "excluded_detectors": list(excluded),
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 39,
        },
    )


@pytest.mark.parametrize(
    "groups", [{1: [6, 13, 14], 2: [1]}, {1: [6], 2: [13, 14]}, {1: [5], 2: [15]}]
)
@pytest.mark.parametrize("consumer", ["pair", "correction", "asymmetry", "counts", "fourier"])
def test_shared_events_fail_before_scientific_output(groups, consumer):
    run = gps_run(groups)
    saved = deepcopy(run.grouping)
    with pytest.raises(ValueError, match="shared events|overlap|Mobile"):
        if consumer == "pair":
            group_forward_backward(run.histograms, run.grouping, metadata=run.metadata)
        elif consumer == "correction":
            corrected_grouped_counts(
                histograms=run.histograms,
                grouping=run.grouping,
                forward_idx=[i - 1 for i in groups[1]],
                backward_idx=[i - 1 for i in groups[2]],
                use_deadtime=False,
                deadtime_mode="off",
                use_background=False,
                metadata=run.metadata,
            )
        elif consumer == "asymmetry":
            prepare_histogram_asymmetry(run, merge=1, facility="PSI", reference=None)
        elif consumer == "counts":
            build_count_groups(
                MuonDataset(np.array([]), np.array([]), np.array([]), run=run), [1, 2]
            )
        else:
            compute_average_group_spectrum(
                run,
                GroupSpectrumConfig(
                    selected_group_ids=[1, 2], remove_diamag=False, subtract_average_signal=False
                ),
            )
    assert run.grouping == saved
    for index, hist in enumerate(run.histograms):
        np.testing.assert_array_equal(hist.counts, np.full(40, 100.0 + index))


@pytest.mark.parametrize(
    "groups",
    [{1: [13, 14], 2: [11, 12]}, {1: [6], 2: [5]}, {1: [3], 2: [15]}, {1: [13, 14], 2: [15]}],
)
def test_valid_readout_choices_keep_counts_and_variances(groups):
    run = gps_run(groups)
    result = group_forward_backward(run.histograms, run.grouping, metadata=run.metadata)
    for actual, gid in ((result.forward, 1), (result.backward, 2)):
        expected = sum(100.0 + ordinal - 1 for ordinal in groups[gid])
        np.testing.assert_array_equal(actual, np.full(40, expected))


def test_effective_exclusions_and_unrequested_diagnostic_groups_do_not_block():
    run = gps_run({1: [6, 13, 14], 2: [1], 3: [6, 13, 14], 4: [5, 15]}, excluded=(6,))
    result = group_forward_backward(run.histograms, run.grouping, metadata=run.metadata)
    np.testing.assert_array_equal(result.forward, np.full(40, 225.0))
    groups = build_count_groups(
        MuonDataset(np.array([]), np.array([]), np.array([]), run=run), [1, 2]
    )
    assert len(groups) == 2
    assert run.grouping["groups"][3] == [6, 13, 14]


def test_captured_unknown_is_authoritative_and_raw_evidence_is_not_reused():
    from asymmetry.core.instruments.readouts import require_independent_readouts

    run = gps_run({1: [6], 2: [13, 14]})
    require_independent_readouts(
        15, run.grouping, ([5], [12, 13]), run.metadata | {"instrument_context": None}
    )
    with pytest.raises(ValueError, match="share events"):
        require_independent_readouts(15, run.grouping, ([5], [12, 13]), run.metadata)


def test_transient_context_keeps_explicit_retained_choice_and_no_stored_metadata_mutation():
    from asymmetry.core.instruments.resolution import (
        InstrumentSelection,
        RecordedInstrumentChoice,
        assess_instrument,
    )
    from asymmetry.core.transform.corrections import reference_observation_metadata

    run = gps_run({1: [6], 2: [13, 14]})
    run.metadata = {"instrument": "EMU", "facility": "ISIS"}
    observed = assess_instrument(15, metadata=run.metadata)
    selection = InstrumentSelection("psi.gps", "root_combined_split", "1", "manual")
    run.instrument_choice = RecordedInstrumentChoice(selection, observed.evidence)
    context = reference_observation_metadata(run)
    assert context["instrument_context"].selection is selection
    assert "instrument_context" not in run.metadata
    with pytest.raises(ValueError, match="share events"):
        group_forward_backward(run.histograms, run.grouping, metadata=context)


def test_reference_guard_uses_reference_identity_not_sample():
    from asymmetry.core.instruments.resolution import resolve_instrument
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    reference = BackgroundReference(
        histograms=gps_run({1: [1], 2: [2]}).histograms,
        scaling=ReferenceScaleRequest("manual", 1.0).resolve(),
        instrument_context=resolve_instrument(gps_run({1: [1], 2: [2]})),
    )
    reference.require_readouts([12, 13], [14])
    with pytest.raises(ValueError, match="share events"):
        reference.require_readouts([5], [12, 13])
    with pytest.raises(ValueError, match="Mobile"):
        reference.require_readouts([4], [14])


def test_each_acquired_period_is_checked_before_ordinal_flattening():
    from asymmetry.core.instruments.readouts import require_run_readouts

    first = gps_run({1: [6], 2: [13, 14]})
    second = gps_run({1: [6], 2: [13, 14]})
    outer = gps_run({1: [6], 2: [13, 14]})
    outer.metadata = {}
    outer.grouping.pop("instrument")
    outer.grouping.update(
        period_inputs=(first, second),
        period_mapping={"1": "red", "2": "green"},
        period_mode="green_plus_red",
    )
    with pytest.raises(ValueError, match="share events"):
        require_run_readouts(outer, ([5], [12, 13]))


def test_safe_period_pool_keeps_each_set_choice_and_uses_flattened_coordinates_only_locally():
    from asymmetry.core.transform.constituent_counts import resolve_constituent_counts
    from asymmetry.core.transform.corrections import reference_observation_metadata

    run = gps_run({1: [7, 8], 2: [9, 10]})
    run.grouping["readout_set"] = "individual"
    first = gps_run(run.grouping["groups"])
    second = gps_run(run.grouping["groups"])
    run.grouping.update(
        period_inputs=(first, second),
        period_mapping={"1": "red", "2": "green"},
        period_mode="green_plus_red",
    )
    prepared = resolve_constituent_counts(
        run.histograms, run.grouping, ((6, 7), (8, 9)), reference_observation_metadata(run)
    )
    assert prepared.pair == ((6, 7, 21, 22), (8, 9, 23, 24))
    assert "readout_set" not in prepared.grouping
    assert all(m.grouping["readout_set"] == "individual" for m in prepared.members)
    assert prepared.layout.alignment.length == 40


def test_separate_fourier_traces_are_not_an_independent_averaging_request():
    run = gps_run({1: [6], 2: [13, 14]})
    prepared = prepare_group_counts(run, [1, 2])
    assert set(prepared.layouts) == {1, 2}
    with pytest.raises(ValueError, match="share events"):
        compute_average_group_spectrum(
            run, GroupSpectrumConfig(selected_group_ids=[1, 2]), prepared=prepared
        )
    generic = gps_run({1: [1], 2: [1, 2]})
    generic.metadata = {}
    generic.grouping.pop("instrument")
    assert set(prepare_group_counts(generic, [1, 2]).layouts) == {1, 2}


def test_conflicting_authority_remains_a_conflict_at_raw_and_reference_boundaries():
    from asymmetry.core.instruments.resolution import (
        InstrumentConflictError,
        RecordedInstrumentChoice,
        resolve_instrument,
    )
    from asymmetry.core.transform.corrections import (
        BackgroundReference,
        reference_observation_metadata,
    )
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    run = gps_run({1: [6], 2: [13, 14]})
    context = resolve_instrument(run)
    run.instrument_choice = RecordedInstrumentChoice(context.selection, context.observed.evidence)
    run.metadata = {"instrument": "EMU", "facility": "ISIS"}
    context = resolve_instrument(run)
    assert context.requires_choice
    with pytest.raises(InstrumentConflictError):
        group_forward_backward(
            run.histograms, run.grouping, metadata=reference_observation_metadata(run)
        )
    reference = BackgroundReference(
        run.histograms, ReferenceScaleRequest("manual", 1.0).resolve(), instrument_context=context
    )
    with pytest.raises(InstrumentConflictError):
        reference.require_readouts([5], [12, 13])


@pytest.mark.parametrize("unsafe_green", [False, True])
def test_green_minus_red_keeps_original_acquisition_eligibility(unsafe_green):
    from asymmetry.core.instruments.resolution import (
        GenericInstrumentIdentity,
        RecordedInstrumentChoice,
        resolve_instrument,
    )

    run = gps_run({1: [13, 14], 2: [11, 12]})
    red = gps_run(run.grouping["groups"])
    green = gps_run(run.grouping["groups"])
    if unsafe_green:
        # Same ordinals are disjoint readouts on retained generic Red, but overlap
        # on GPS Green. The original owners, not the outer display, decide.
        run.grouping["groups"] = {1: [6], 2: [13, 14]}
        red.instrument_choice = RecordedInstrumentChoice(
            GenericInstrumentIdentity(), resolve_instrument(red).observed.evidence
        )
        run.instrument_choice = RecordedInstrumentChoice(
            GenericInstrumentIdentity(), resolve_instrument(run).observed.evidence
        )
    run.grouping.update(
        period_inputs=(red, green),
        period_histograms=[red.histograms, green.histograms],
        period_mapping={"1": "red", "2": "green"},
        period_mode="green_minus_red",
    )
    if unsafe_green:
        with pytest.raises(ValueError, match="share events"):
            prepare_histogram_asymmetry(run, merge=1, facility="PSI", reference=None)
    else:
        result = prepare_histogram_asymmetry(run, merge=1, facility="PSI", reference=None)
        assert result.asymmetry.size == 40


def test_actual_nexus_cli_rejects_shared_events_and_retains_raw_acquisition(tmp_path, capsys):
    from asymmetry import cli
    from asymmetry.core.io import load
    from asymmetry.core.io.nexus_writer import write_nexus_v1

    run = gps_run({1: [6], 2: [13, 14]})
    run.run_number = 680
    path = tmp_path / "GPS00000680.nxs"
    write_nexus_v1(run, path)
    loaded = load(path)
    assert len(loaded.run.histograms) == 15
    for original, decoded in zip(run.histograms, loaded.run.histograms, strict=True):
        np.testing.assert_array_equal(decoded.counts, original.counts)
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            ["reduce", str(tmp_path), "--runs", "680", "--json", "--workdir", str(tmp_path / "wd")]
        )
    assert stopped.value.code == 1
    assert "Readouts L and LB share events" in capsys.readouterr().err


def test_cached_standalone_fourier_rejects_changed_canonical_authority():
    from asymmetry.core.fourier.grouped import build_group_signal_dataset
    from asymmetry.core.instruments.resolution import (
        GenericInstrumentIdentity,
        RecordedInstrumentChoice,
        resolve_instrument,
    )

    run = gps_run({1: [6, 13, 14], 2: [1]})
    context = resolve_instrument(run)
    run.instrument_choice = RecordedInstrumentChoice(
        GenericInstrumentIdentity(), context.observed.evidence
    )
    prepared = prepare_group_counts(run, [1])
    assert build_group_signal_dataset(run, 1, prepared=prepared).n_points == 40
    run.instrument_choice = RecordedInstrumentChoice(context.selection, context.observed.evidence)
    with pytest.raises(ValueError, match="stale|match|shared|source|Prepared"):
        build_group_signal_dataset(run, 1, prepared=prepared)
    with pytest.raises(ValueError, match="share events"):
        prepare_group_counts(run, [1])


def test_per_channel_timing_remains_available_for_retained_unsafe_groups():
    from asymmetry.core.transform.corrections import reference_observation_metadata
    from asymmetry.core.transform.t0 import resolve_selected_timing

    run = gps_run({1: [6], 2: [13, 14]})
    resolve_selected_timing(run.histograms, run.grouping, reference_observation_metadata(run))
    assert run.grouping["t0_bin"] == 0
    with pytest.raises(ValueError, match="share events"):
        prepare_histogram_asymmetry(run, merge=1, facility="PSI", reference=None)


def test_safe_fourier_metadata_preserves_json_reference_identity_contract():
    import json

    from asymmetry.core.fourier.grouped import build_group_signal_dataset
    from asymmetry.core.maxent import MaxEntConfig, build_maxent_input

    run = gps_run({1: [1], 2: [2]})
    prepared = prepare_group_counts(run, [1, 2])
    signal = build_group_signal_dataset(run, 1, prepared=prepared)
    assert prepared.identity[0] is None
    assert signal.metadata["reference_file_identity"] is None
    assert json.loads(json.dumps(signal.metadata))["group_id"] == 1
    joint = build_maxent_input(
        run,
        MaxEntConfig(
            auto_steer=False, n_spectrum_points=32, auto_window=False, f_min_mhz=0, f_max_mhz=2
        ),
        prepared=prepared,
    )
    assert json.loads(json.dumps(joint.metadata))["reference_file_identity"] is None


@pytest.mark.parametrize("cached", [False, True])
def test_joint_maxent_rejects_actual_selected_overlap_but_not_separate_diagnostics(cached):
    from asymmetry.core.maxent import MaxEntConfig, build_maxent_input

    run = gps_run({1: [6], 2: [13, 14], 3: [1]})
    prepared = prepare_group_counts(run, [1, 2]) if cached else None
    config = MaxEntConfig(
        selected_group_ids=[1, 2],
        auto_steer=False,
        n_spectrum_points=32,
        auto_window=False,
        f_min_mhz=0,
        f_max_mhz=2,
    )
    with pytest.raises(ValueError, match="share events"):
        build_maxent_input(run, config, prepared=prepared)
    config.selected_group_ids = [3]
    safe = build_maxent_input(run, config)
    assert [g.group_id for g in safe.groups] == [3]


@pytest.mark.parametrize("cached", [False, True])
def test_joint_maxent_uses_background_reference_own_shared_event_identity(tmp_path, cached):
    from asymmetry.core.instruments.resolution import (
        GenericInstrumentIdentity,
        RecordedInstrumentChoice,
        resolve_instrument,
    )
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.maxent import MaxEntConfig, build_maxent_input

    sample = gps_run({1: [6], 2: [13, 14]})
    sample.instrument_choice = RecordedInstrumentChoice(
        GenericInstrumentIdentity(), resolve_instrument(sample).observed.evidence
    )
    reference = gps_run({1: [1], 2: [2]})
    reference.run_number = 681
    path = tmp_path / "GPS00000681.nxs"
    write_nexus_v1(reference, path)
    sample.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={
            "source_file": str(path),
            "run_number": 681,
            "period_number": 1,
            "scale_request": {"mode": "manual", "coefficient": 0.1},
        },
    )
    prepared = prepare_group_counts(sample, [1, 2]) if cached else None
    config = MaxEntConfig(
        selected_group_ids=[1, 2],
        auto_steer=False,
        n_spectrum_points=32,
        auto_window=False,
        f_min_mhz=0,
        f_max_mhz=2,
    )
    with pytest.raises(ValueError, match="share events"):
        build_maxent_input(sample, config, prepared=prepared)
