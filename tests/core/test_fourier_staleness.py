"""Tests for the grouped-FFT staleness helpers (digest + config diff)."""

from __future__ import annotations

from dataclasses import replace

from asymmetry.core.data.dataset import Run
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    config_differences,
    fourier_grouping_digest,
)


def _run(grouping: dict) -> Run:
    return Run(run_number=1, grouping=grouping)


def _base_grouping() -> dict:
    return {
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Fwd", 2: "Bwd"},
        "first_good_bin": 0,
        "last_good_bin": 100,
        "t0_bin": 5,
        "bunching_factor": 1,
    }


# --------------------------------------------------------------------------- #
# fourier_grouping_digest
# --------------------------------------------------------------------------- #


def test_digest_none_run_is_empty_string():
    assert fourier_grouping_digest(None) == ""


def test_digest_stable_across_dict_key_order():
    g1 = {"groups": {1: [1, 2], 2: [3]}, "t0_bin": 5}
    g2 = {"t0_bin": 5, "groups": {2: [3], 1: [1, 2]}}
    assert fourier_grouping_digest(_run(g1)) == fourier_grouping_digest(_run(g2))


def test_digest_stable_int_vs_str_group_ids():
    g1 = {"groups": {1: [1, 2]}}
    g2 = {"groups": {"1": ["1", "2"]}}
    assert fourier_grouping_digest(_run(g1)) == fourier_grouping_digest(_run(g2))


def test_digest_changes_when_detector_moves_group():
    base = _base_grouping()
    moved = _base_grouping()
    moved["groups"] = {1: [1], 2: [2, 3, 4]}
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(moved))


def test_digest_changes_with_excluded_detectors():
    base = _base_grouping()
    excluded = _base_grouping()
    excluded["excluded_detectors"] = [2]
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(excluded))


def test_digest_changes_with_t0_bin():
    base = _base_grouping()
    changed = _base_grouping()
    changed["t0_bin"] = 6
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(changed))


def test_digest_changes_with_dead_time_us_when_correction_enabled():
    base = _base_grouping()
    base["deadtime_correction"] = True
    base["dead_time_us"] = [0.01, 0.01, 0.01, 0.01]
    changed = _base_grouping()
    changed["deadtime_correction"] = True
    changed["dead_time_us"] = [0.02, 0.01, 0.01, 0.01]
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(changed))


def test_digest_unchanged_dead_time_us_when_correction_disabled():
    base = _base_grouping()
    base["deadtime_correction"] = False
    base["dead_time_us"] = [0.01, 0.01, 0.01, 0.01]
    changed = _base_grouping()
    changed["deadtime_correction"] = False
    changed["dead_time_us"] = [0.02, 0.01, 0.01, 0.01]
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(changed))


def test_digest_changes_with_deadtime_correction_flag():
    base = _base_grouping()
    base["deadtime_correction"] = False
    changed = _base_grouping()
    changed["deadtime_correction"] = True
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(changed))


def test_digest_changes_with_background_mode_when_correction_enabled():
    base = _base_grouping()
    base["background_correction"] = True
    base["background_mode"] = "tail_fit"
    changed = _base_grouping()
    changed["background_correction"] = True
    changed["background_mode"] = "range"
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(changed))


def test_digest_unchanged_background_mode_when_correction_disabled():
    base = _base_grouping()
    base["background_correction"] = False
    base["background_mode"] = "tail_fit"
    changed = _base_grouping()
    changed["background_correction"] = False
    changed["background_mode"] = "range"
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(changed))


def test_digest_uses_resolved_background_mode():
    # The consumer resolves its mode via resolve_background_mode: an explicit
    # "range" and an absent background_mode (which infers "range") must digest
    # identically, while a GUI-hint background_method must not perturb it.
    explicit = _base_grouping()
    explicit["background_correction"] = True
    explicit["background_mode"] = "range"
    inferred = _base_grouping()
    inferred["background_correction"] = True
    inferred["background_method"] = "estimated"
    assert fourier_grouping_digest(_run(explicit)) == fourier_grouping_digest(_run(inferred))


def test_digest_unchanged_with_group_names_change():
    base = _base_grouping()
    renamed = _base_grouping()
    renamed["group_names"] = {1: "Left", 2: "Right"}
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(renamed))


def test_digest_unchanged_with_projections_change():
    base = _base_grouping()
    with_proj = _base_grouping()
    with_proj["projections"] = [{"label": "Px", "forward_group": 1, "backward_group": 2}]
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(with_proj))


def test_digest_unchanged_axis_switch_background_disabled():
    """A polarisation-axis switch rewrites forward/backward_group on every dataset;
    with background correction off, that rewrite must not perturb the digest."""
    base = _base_grouping()
    base["background_correction"] = False
    base["forward_group"] = 1
    base["backward_group"] = 2
    switched = _base_grouping()
    switched["background_correction"] = False
    switched["forward_group"] = 2
    switched["backward_group"] = 1
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(switched))


def test_digest_unchanged_axis_switch_dict_shaped_background_values():
    """Dict-shaped background_values/ranges key by group id directly, so they do
    not need forward_group/backward_group to route entries."""
    base = _base_grouping()
    base["background_correction"] = True
    base["background_values"] = {1: 10.0, 2: 20.0}
    base["forward_group"] = 1
    base["backward_group"] = 2
    switched = _base_grouping()
    switched["background_correction"] = True
    switched["background_values"] = {1: 10.0, 2: 20.0}
    switched["forward_group"] = 2
    switched["backward_group"] = 1
    assert fourier_grouping_digest(_run(base)) == fourier_grouping_digest(_run(switched))


def test_digest_changes_axis_switch_list_shaped_background_values():
    """List-shaped background_values are positional [forward, backward]; an axis
    switch that flips forward/backward_group changes what each entry means."""
    base = _base_grouping()
    base["background_correction"] = True
    base["background_values"] = [10.0, 20.0]
    base["forward_group"] = 1
    base["backward_group"] = 2
    switched = _base_grouping()
    switched["background_correction"] = True
    switched["background_values"] = [10.0, 20.0]
    switched["forward_group"] = 2
    switched["backward_group"] = 1
    assert fourier_grouping_digest(_run(base)) != fourier_grouping_digest(_run(switched))


# --------------------------------------------------------------------------- #
# config_differences
# --------------------------------------------------------------------------- #


def test_config_differences_identical_is_empty():
    config = GroupSpectrumConfig()
    assert config_differences(config, config) == []


def test_config_differences_default_vs_default_is_empty():
    assert config_differences(GroupSpectrumConfig(), GroupSpectrumConfig()) == []


def test_config_differences_tau_inert_when_both_windows_none():
    current = GroupSpectrumConfig(window="none", filter_time_constant_us=1.5)
    recorded = GroupSpectrumConfig(window="none", filter_time_constant_us=3.0)
    assert config_differences(current, recorded) == []


def test_config_differences_tau_reported_when_window_active():
    current = GroupSpectrumConfig(window="lorentzian", filter_time_constant_us=1.5)
    recorded = GroupSpectrumConfig(window="lorentzian", filter_time_constant_us=3.0)
    assert config_differences(current, recorded) == ["apodisation filter τ"]


def test_config_differences_filter_start_reported_when_window_active():
    current = GroupSpectrumConfig(window="gaussian", filter_start_us=0.0)
    recorded = GroupSpectrumConfig(window="gaussian", filter_start_us=1.0)
    assert config_differences(current, recorded) == ["apodisation filter start"]


def test_config_differences_phase_reported_in_phase_display():
    current = GroupSpectrumConfig(display="Phase", group_phase_degrees={1: 0.0})
    recorded = GroupSpectrumConfig(display="Phase", group_phase_degrees={1: 10.0})
    assert config_differences(current, recorded) == ["group phases"]


def test_config_differences_phase_not_reported_in_power_sqrt_display():
    current = GroupSpectrumConfig(display="(Power)^1/2", group_phase_degrees={1: 0.0})
    recorded = GroupSpectrumConfig(display="(Power)^1/2", group_phase_degrees={1: 10.0})
    assert config_differences(current, recorded) == []


def test_config_differences_t0_offset_gated_on_phase_mode():
    current = GroupSpectrumConfig(display="(Power)^1/2", t0_offset_us=0.0)
    recorded = GroupSpectrumConfig(display="(Power)^1/2", t0_offset_us=1.0)
    assert config_differences(current, recorded) == []
    current = replace(current, display="Phase")
    recorded = replace(recorded, display="Phase")
    assert config_differences(current, recorded) == ["t0 offset"]


def test_config_differences_burg_orders_only_in_burg_mode():
    current = GroupSpectrumConfig(display="(Power)^1/2", burg_order_min=2, burg_order_max=40)
    recorded = GroupSpectrumConfig(display="(Power)^1/2", burg_order_min=4, burg_order_max=20)
    assert config_differences(current, recorded) == []

    current = replace(current, display="Resolution (Burg)")
    recorded = replace(recorded, display="Resolution (Burg)")
    assert config_differences(current, recorded) == ["Burg pole scan"]


def test_config_differences_correlation_settings_only_in_correlation_mode():
    current = GroupSpectrumConfig(display="(Power)^1/2", correlation_reference_field_gauss=100.0)
    recorded = GroupSpectrumConfig(display="(Power)^1/2", correlation_reference_field_gauss=200.0)
    assert config_differences(current, recorded) == []

    current = replace(current, display="Correlation (radical)")
    recorded = replace(recorded, display="Correlation (radical)")
    assert config_differences(current, recorded) == ["correlation settings"]


def test_config_differences_correlation_field_none_vs_value_is_difference():
    current = GroupSpectrumConfig(
        display="Correlation (radical)", correlation_reference_field_gauss=None
    )
    recorded = GroupSpectrumConfig(
        display="Correlation (radical)", correlation_reference_field_gauss=100.0
    )
    assert config_differences(current, recorded) == ["correlation settings"]


def test_config_differences_selected_group_ids_ordering_insensitive():
    current = GroupSpectrumConfig(selected_group_ids=[2, 1, 3])
    recorded = GroupSpectrumConfig(selected_group_ids=[1, 3, 2])
    assert config_differences(current, recorded) == []


def test_config_differences_none_vs_list_selected_group_ids_reported():
    current = GroupSpectrumConfig(selected_group_ids=None)
    recorded = GroupSpectrumConfig(selected_group_ids=[1, 2])
    assert config_differences(current, recorded) == ["included groups"]


def test_config_differences_t_min_t_max_none_vs_value():
    current = GroupSpectrumConfig(t_min_us=None, t_max_us=None)
    recorded = GroupSpectrumConfig(t_min_us=0.5, t_max_us=None)
    assert config_differences(current, recorded) == ["time window"]


def test_config_differences_label_ordering_display_first_no_duplicates():
    current = GroupSpectrumConfig(
        display="Phase",
        t_min_us=0.0,
        t_max_us=None,
        window="lorentzian",
        filter_start_us=0.0,
        filter_time_constant_us=1.0,
        t0_offset_us=0.0,
        baseline_mode="sigma_clip",
        baseline_kappa=2.0,
        exclude_enabled=True,
        exclusion_ranges=[(1.0, 0.1)],
        remove_diamag=False,
        diamag_exclusion=True,
        diamag_half_width_mhz=0.3,
    )
    recorded = GroupSpectrumConfig(
        display="(Power)^1/2",
        t_min_us=1.0,
        t_max_us=5.0,
        window="none",
        filter_start_us=1.0,
        filter_time_constant_us=2.0,
        t0_offset_us=1.0,
        baseline_mode="none",
        baseline_kappa=3.0,
        exclude_enabled=False,
        exclusion_ranges=[(2.0, 0.2)],
        remove_diamag=True,
        diamag_exclusion=False,
        diamag_half_width_mhz=0.5,
    )
    labels = config_differences(current, recorded)
    assert labels[0] == "display mode"
    assert labels[1] == "time window"
    assert len(labels) == len(set(labels))
    # "baseline offset" collapses baseline_mode + baseline_kappa into one label.
    assert labels.count("baseline offset") == 1
    # "diamagnetic handling" collapses remove_diamag + diamag_exclusion(+ half-width).
    assert labels.count("diamagnetic handling") == 1
    # window differs, so the filter fields are active for at least one side.
    assert "apodisation" in labels


def test_config_differences_baseline_kappa_inert_when_both_none():
    current = GroupSpectrumConfig(baseline_mode="none", baseline_kappa=2.0)
    recorded = GroupSpectrumConfig(baseline_mode="none", baseline_kappa=5.0)
    assert config_differences(current, recorded) == []


def test_config_differences_exclusion_ranges_inert_when_both_disabled():
    current = GroupSpectrumConfig(exclude_enabled=False, exclusion_ranges=[(1.0, 0.1)])
    recorded = GroupSpectrumConfig(exclude_enabled=False, exclusion_ranges=[(2.0, 0.2)])
    assert config_differences(current, recorded) == []


def test_config_differences_exclusion_ranges_order_insensitive():
    current = GroupSpectrumConfig(exclude_enabled=True, exclusion_ranges=[(1.0, 0.1), (2.0, 0.2)])
    recorded = GroupSpectrumConfig(exclude_enabled=True, exclusion_ranges=[(2.0, 0.2), (1.0, 0.1)])
    assert config_differences(current, recorded) == []


def test_config_differences_diamag_half_width_inert_when_both_disabled():
    current = GroupSpectrumConfig(diamag_exclusion=False, diamag_half_width_mhz=0.3)
    recorded = GroupSpectrumConfig(diamag_exclusion=False, diamag_half_width_mhz=0.9)
    assert config_differences(current, recorded) == []


def test_config_differences_pulse_settings_inert_when_both_disabled():
    current = GroupSpectrumConfig(pulse_compensation=False, pulse_half_width_us=0.1)
    recorded = GroupSpectrumConfig(pulse_compensation=False, pulse_half_width_us=0.9)
    assert config_differences(current, recorded) == []


def test_config_differences_pulse_settings_reported_when_active():
    current = GroupSpectrumConfig(pulse_compensation=True, pulse_half_width_us=0.1)
    recorded = GroupSpectrumConfig(pulse_compensation=True, pulse_half_width_us=0.9)
    assert config_differences(current, recorded) == ["pulse settings"]
