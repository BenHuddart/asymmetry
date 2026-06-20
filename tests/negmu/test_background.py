"""Tests for WP3 — Set-as-BG capture-component subtraction.

Verification-plan §6: Phase 3 acceptance criteria.

All synthetic histograms use simulate_capture_run (no inline generators).
The round-trip test checks that the derived Run produced by
capture_background_run reconstructs the same grouped, subtracted counts as
the array-level subtract_capture_background call.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.negmu.background import capture_background_run, subtract_capture_background
from asymmetry.core.negmu.fit import CaptureModelSpec, fit_capture_group
from asymmetry.core.negmu.model import evaluate_capture_model
from asymmetry.core.simulate import simulate_capture_run
from tests.negmu.helpers import make_dataset

# ---------------------------------------------------------------------------
# Template geometry (matches test_fit_single.py for consistency)
# ---------------------------------------------------------------------------

N_BINS = 1024
BIN_WIDTH = 0.016  # µs
GROUP_ID = 1


def _make_template():
    from asymmetry.core.data.dataset import Histogram, Run

    histograms = [
        Histogram(
            counts=np.zeros(N_BINS, dtype=float),
            bin_width=BIN_WIDTH,
            t0_bin=0,
            good_bin_start=0,
            good_bin_end=N_BINS - 1,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {GROUP_ID: [1, 2]},
        "group_names": {GROUP_ID: "Group 1"},
        "forward_group": GROUP_ID,
        "backward_group": GROUP_ID,
        "alpha": 1.0,
        "t0_bin": 0,
        "t_good_offset": 0,
        "first_good_bin": 0,
        "last_good_bin": N_BINS - 1,
        "bin_index_base": 1,
        "bunching_factor": 1,
        "good_frames": 1.0,
        "deadtime_correction": False,
        "dead_time_us": [0.0, 0.0],
        "included_groups": {GROUP_ID: True},
    }
    return Run(
        run_number=0,
        histograms=histograms,
        metadata={"title": "BG subtraction test template", "synthetic": True},
        grouping=grouping,
        source_file="",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cf_fit():
    """C + Fe only (no decayBG), background_per_bin=0 — well-separated τ values.

    C has τ=2.030 µs, Fe has τ=0.206 µs (~10× apart).  No flat background so
    the residual after subtracting Fe is a clean C signal.  Used for the
    Poisson-tolerance plausibility check.
    """
    template = _make_template()
    spec = CaptureModelSpec(elements=("C", "Fe"), include_decay_background=False)
    components = spec.components()
    weights = {"C": 6.0, "Fe": 4.0}
    run = simulate_capture_run(
        template,
        components,
        weights,
        total_events=5.0e6,
        seed=42,
        background_per_bin=0.0,
    )
    dataset = make_dataset(run)
    fit = fit_capture_group(dataset, GROUP_ID, spec, cost="poisson")
    return dataset, fit, spec, components, weights


@pytest.fixture(scope="module")
def cfe_bg_fit():
    """C + Fe + decayBG with flat background — used for mechanics and round-trip."""
    template = _make_template()
    spec = CaptureModelSpec(elements=("C", "Fe"), include_decay_background=True)
    components = spec.components()
    weights = {"C": 5.0, "Fe": 3.0, "decayBG": 2.0}
    run = simulate_capture_run(
        template,
        components,
        weights,
        total_events=5.0e6,
        seed=0,
        background_per_bin=2.0,
    )
    dataset = make_dataset(run)
    fit = fit_capture_group(dataset, GROUP_ID, spec, cost="poisson")
    return dataset, fit, spec, components, weights


# ---------------------------------------------------------------------------
# subtract_capture_background — array-level tests
# ---------------------------------------------------------------------------


class TestSubtractCaptureBackground:
    def test_identity_no_unwanted(self, cfe_bg_fit):
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        result = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=[])
        np.testing.assert_array_equal(result, group.counts)

    def test_mechanics_exact(self, cfe_bg_fit):
        """result == counts - Σ unwanted_exponentials (no flat background)."""
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        unwanted = ["C"]

        result = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=unwanted)

        # Manual reference: evaluate unwanted exponentials only (no flat bg term)
        params = {p.name: float(p.value) for p in fit.parameters}
        params_no_bg = {k: v for k, v in params.items() if k != "background"}
        c_comp = [c for c in spec.components() if c.label == "C"]
        expected_bg = evaluate_capture_model(c_comp, params_no_bg, group.time)
        expected_result = group.counts - expected_bg

        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)

    def test_flat_background_not_subtracted(self, cfe_bg_fit):
        """Subtracting Fe should leave the flat background intact in the residual."""
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)

        result = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=["Fe"])
        # At late times (after Fe has decayed), only C + decayBG + flat_bg remain.
        # The flat background should NOT have been subtracted.
        # Late 10 % of the window: Fe is negligible, residual ≈ C + decayBG + flat_bg
        n_late = N_BINS // 10
        late_mean = float(np.mean(result[-n_late:]))
        # There should be some signal left (not driven to zero by over-subtraction)
        assert late_mean > 0.0, "flat background was incorrectly subtracted"

    def test_retained_component_within_poisson_tolerance(self, cf_fit):
        """After subtracting Fe, the C residual is within 5σ Poisson of the C curve.

        Uses C + Fe only (no decayBG), background_per_bin=0, so the residual is
        a clean C signal.  The tolerance is computed from the fit-evaluated C curve
        which, for a well-conditioned fit, closely tracks the generating curve.
        """
        dataset, fit, spec, components, *_ = cf_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)

        # Subtract Fe; C should remain.
        result = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=["Fe"])

        # Expected residual: C exponential from fitted params (no flat bg).
        params = {p.name: float(p.value) for p in fit.parameters}
        params_no_bg = {k: v for k, v in params.items() if k != "background"}
        c_comp = [c for c in components if c.label == "C"]
        expected_c = evaluate_capture_model(c_comp, params_no_bg, group.time)

        diff = np.abs(result - expected_c)
        # 5σ Poisson tolerance based on total grouped counts per bin (generous,
        # accounts for fit-parameter uncertainty in the subtracted Fe component).
        tolerance = 5.0 * np.sqrt(np.maximum(group.counts, 1.0))
        assert np.all(diff < tolerance), (
            f"C residual deviates beyond 5σ Poisson; max diff = {diff.max():.1f}, "
            f"max tolerance = {tolerance[np.argmax(diff)]:.1f}"
        )

    def test_subtract_returns_float64(self, cfe_bg_fit):
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        result = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=["C"])
        assert result.dtype == np.float64

    def test_unknown_label_raises(self, cfe_bg_fit):
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        with pytest.raises(ValueError, match="not in spec"):
            subtract_capture_background(group.time, group.counts, fit, spec, unwanted=["Au"])

    def test_spec_fit_mismatch_raises(self, cfe_bg_fit):
        """Requesting a component absent from fit.parameters raises ValueError (not silent no-op).

        build_capture_count_model defaults amp to 0.0 for missing keys, so without
        this guard the subtraction would be a silent no-op.
        """
        dataset, fit, spec, *_ = cfe_bg_fit
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        # Al is a valid element but was NOT in the fit's spec → amp_Al absent.
        mismatch_spec = CaptureModelSpec(elements=("C", "Fe", "Al"), include_decay_background=False)
        with pytest.raises(ValueError, match="fit has no amplitude for"):
            subtract_capture_background(
                group.time, group.counts, fit, mismatch_spec, unwanted=["Al"]
            )


# ---------------------------------------------------------------------------
# capture_background_run — derived Run tests
# ---------------------------------------------------------------------------


class TestCaptureBackgroundRun:
    def test_round_trip_counts_equal_array_level(self, cfe_bg_fit):
        """Derived Run's group histogram equals the array-level subtraction."""
        dataset, fit, spec, *_ = cfe_bg_fit

        # Array-level reference
        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        array_level = subtract_capture_background(
            group.time, group.counts, fit, spec, unwanted=["Fe"]
        )

        # Derived Run
        derived_run = capture_background_run(dataset, GROUP_ID, fit, spec, unwanted=["Fe"])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        derived_group = build_count_group(derived_dataset, GROUP_ID, lifetime_corrected=False)

        np.testing.assert_array_almost_equal(derived_group.counts, array_level)

    def test_identity_no_unwanted_round_trip(self, cfe_bg_fit):
        """Empty unwanted → derived Run's group histogram equals original grouped counts."""
        dataset, fit, spec, *_ = cfe_bg_fit

        group = build_count_group(dataset, GROUP_ID, lifetime_corrected=False)
        derived_run = capture_background_run(dataset, GROUP_ID, fit, spec, unwanted=[])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        derived_group = build_count_group(derived_dataset, GROUP_ID, lifetime_corrected=False)

        np.testing.assert_array_almost_equal(derived_group.counts, group.counts)

    def test_provenance_metadata_present(self, cfe_bg_fit):
        """Derived Run carries background_subtraction provenance."""
        dataset, fit, spec, *_ = cfe_bg_fit
        derived_run = capture_background_run(dataset, GROUP_ID, fit, spec, unwanted=["Fe"])
        assert "background_subtraction" in derived_run.metadata
        prov = derived_run.metadata["background_subtraction"]
        assert prov["group_id"] == GROUP_ID
        assert "Fe" in prov["unwanted"]
        assert "spec_elements" in prov

    def test_synthetic_metadata_inherited(self, cfe_bg_fit):
        """Derived Run inherits synthetic marker from the source run."""
        dataset, fit, spec, *_ = cfe_bg_fit
        derived_run = capture_background_run(dataset, GROUP_ID, fit, spec, unwanted=["Fe"])
        assert derived_run.metadata.get("synthetic") is True

    def test_run_number_override(self, cfe_bg_fit):
        dataset, fit, spec, *_ = cfe_bg_fit
        derived_run = capture_background_run(
            dataset, GROUP_ID, fit, spec, unwanted=[], run_number=99
        )
        assert derived_run.run_number == 99

    def test_time_axis_preserved_with_bunching(self, cfe_bg_fit):
        """Derived Run axis_start_bins is read from source integers, not float inversion.

        With bunching_factor=2 the post-rebin time axis uses bin midpoints:
        time[0] = (axis_start + 0.5) * bin_width_orig.  Dividing by bin_width_post
        = 2 * bin_width_orig gives (axis_start + 0.5) / 2, which rounds to the wrong
        integer whenever axis_start is even.  The integer-based fix avoids this.
        """
        dataset, fit, spec, *_ = cfe_bg_fit
        source_run = dataset.run

        # Inject bunching_factor=2 into the grouping.
        bunched_grouping = {**source_run.grouping, "bunching_factor": 2}
        bunched_run = Run(
            run_number=source_run.run_number,
            histograms=source_run.histograms,
            metadata=source_run.metadata,
            grouping=bunched_grouping,
            source_file=source_run.source_file,
        )
        bunched_dataset = MuonDataset(
            time=dataset.time,
            asymmetry=dataset.asymmetry,
            error=dataset.error,
            metadata=dataset.metadata,
            run=bunched_run,
        )

        src_group = build_count_group(bunched_dataset, GROUP_ID, lifetime_corrected=False)
        derived_run = capture_background_run(bunched_dataset, GROUP_ID, fit, spec, unwanted=[])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        derived_group = build_count_group(derived_dataset, GROUP_ID, lifetime_corrected=False)

        # With the integer fix, the error is bounded by 0.5 * bin_width_orig (midpoint
        # shift inherent to rebinning) rather than up to 1 full post-rebin bin width
        # (= 2 * bin_width_orig) that the float round-trip would produce.
        bin_width_orig = float(source_run.histograms[0].bin_width)
        err = abs(float(derived_group.time[0]) - float(src_group.time[0]))
        assert err <= 0.5 * bin_width_orig + 1e-9, (
            f"Time axis off by more than half a pre-rebin bin with bunching: "
            f"err={err:.6f} µs (src={src_group.time[0]:.4f}, "
            f"derived={derived_group.time[0]:.4f})"
        )

    def test_time_axis_preserved_with_bunching_and_offset(self, cfe_bg_fit):
        """Combined regime: bunching_factor>1 AND first_good_bin>common_t0.

        Regression guard. The earlier hand-derived geometry computed
        axis_start_bins = first_good − common_t0 in *pre-bunch* (fine) bins but
        applied it as the good_bin_start of a histogram whose bin_width was the
        *post-bunch* width, stretching the time-axis offset by the bunch factor.
        The two existing tests exercised bunching (offset 0) and offset
        (bunch 1) separately and missed this combination.
        """
        dataset, fit, spec, *_ = cfe_bg_fit
        source_run = dataset.run

        offset = 10
        pad = np.zeros(offset, dtype=np.float64)
        shifted_hists = [
            Histogram(
                counts=np.concatenate([pad, h.counts]),
                bin_width=h.bin_width,
                t0_bin=0,
                good_bin_start=offset,
                good_bin_end=offset + len(h.counts) - 1,
            )
            for h in source_run.histograms
        ]
        grouping = {
            **source_run.grouping,
            "first_good_bin": offset,
            "last_good_bin": offset + N_BINS - 1,
            "bunching_factor": 2,
        }
        run = Run(
            run_number=source_run.run_number,
            histograms=shifted_hists,
            metadata=source_run.metadata,
            grouping=grouping,
            source_file=source_run.source_file,
        )
        ds = MuonDataset(
            time=dataset.time,
            asymmetry=dataset.asymmetry,
            error=dataset.error,
            metadata=dataset.metadata,
            run=run,
        )

        src_group = build_count_group(ds, GROUP_ID, lifetime_corrected=False)
        derived_run = capture_background_run(ds, GROUP_ID, fit, spec, unwanted=[])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        derived_group = build_count_group(derived_dataset, GROUP_ID, lifetime_corrected=False)

        # Tolerance is the inherent ≤ ½ pre-rebin-bin midpoint shift, NOT the
        # factor-of-bunch error the old code produced (~0.15 µs here).
        bin_width_fine = float(source_run.histograms[0].bin_width)
        err = abs(float(derived_group.time[0]) - float(src_group.time[0]))
        assert err <= 0.5 * bin_width_fine + 1e-9, (
            f"Time axis off by more than half a pre-rebin bin with bunching+offset: "
            f"err={err:.6f} µs (src={src_group.time[0]:.4f}, derived={derived_group.time[0]:.4f})"
        )

    def test_time_axis_preserved_with_nonzero_offset(self, cfe_bg_fit):
        """Derived Run preserves time[0] when source group has first_good_bin > 0."""
        dataset, fit, spec, *_ = cfe_bg_fit
        source_run = dataset.run

        # Re-encode the source histograms with a 10-bin dead-zone prefix so
        # build_count_group returns time[0] = 10 * bin_width instead of 0.
        offset = 10
        pad = np.zeros(offset, dtype=np.float64)
        shifted_hists = [
            Histogram(
                counts=np.concatenate([pad, h.counts]),
                bin_width=h.bin_width,
                t0_bin=0,
                good_bin_start=offset,
                good_bin_end=offset + len(h.counts) - 1,
            )
            for h in source_run.histograms
        ]
        shifted_grouping = {
            **source_run.grouping,
            "first_good_bin": offset,
            "last_good_bin": offset + N_BINS - 1,
        }
        shifted_run = Run(
            run_number=source_run.run_number,
            histograms=shifted_hists,
            metadata=source_run.metadata,
            grouping=shifted_grouping,
            source_file=source_run.source_file,
        )
        shifted_dataset = MuonDataset(
            time=dataset.time,
            asymmetry=dataset.asymmetry,
            error=dataset.error,
            metadata=dataset.metadata,
            run=shifted_run,
        )

        src_group = build_count_group(shifted_dataset, GROUP_ID, lifetime_corrected=False)
        expected_t0 = float(src_group.time[0])

        derived_run = capture_background_run(shifted_dataset, GROUP_ID, fit, spec, unwanted=[])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        derived_group = build_count_group(derived_dataset, GROUP_ID, lifetime_corrected=False)

        assert abs(float(derived_group.time[0]) - expected_t0) < 1e-9, (
            f"Time axis shifted: expected t[0]={expected_t0:.4f} µs, "
            f"got {derived_group.time[0]:.4f} µs"
        )

    def test_derived_run_refittable(self, cf_fit):
        """The derived Run can be re-fitted with fit_capture_group (no crash)."""
        dataset, fit, spec, *_ = cf_fit

        # Subtract Fe; re-fit remaining C signal
        derived_run = capture_background_run(dataset, GROUP_ID, fit, spec, unwanted=["Fe"])
        derived_dataset = MuonDataset(
            time=np.array([]),
            asymmetry=np.array([]),
            error=np.array([]),
            metadata={},
            run=derived_run,
        )
        # Re-fit with C only
        c_spec = CaptureModelSpec(elements=("C",), include_decay_background=False)
        refit = fit_capture_group(derived_dataset, GROUP_ID, c_spec, cost="poisson")
        assert refit is not None
        # Recovered C amplitude should be positive and meaningful
        c_amp = float(refit.parameters["amp_C"].value)
        assert c_amp > 0.0
