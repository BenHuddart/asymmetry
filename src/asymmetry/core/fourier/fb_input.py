"""F/B Fourier input through the canonical corrected period reduction."""

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io import resolve_background_reference
from asymmetry.core.transform.corrections import reference_observation_metadata
from asymmetry.core.transform.grouping import (
    grouping_balances,
    independent_pair_indices,
)
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import correction_flags_from_grouping, reduce_period_asymmetry
from asymmetry.core.transform.reference_scale import ReferenceExposure


def build_fb_signal_dataset(run, *, cancel_check=None):
    """Preserve F/B coordinates, correction outcomes and atomic held-period fallback."""
    grouping = run.grouping
    histograms = list(run.histograms)
    pair = independent_pair_indices(grouping, n_histograms=len(histograms))
    alpha, beta = grouping_balances(grouping)
    flags = correction_flags_from_grouping(grouping)
    cache = {}

    def reference(effective):
        return resolve_background_reference(
            effective.get("background_run"),
            sample_exposure=ReferenceExposure.from_grouping(effective),
            cache=cache,
        )

    result = reduce_period_asymmetry(
        histograms,
        grouping,
        str(grouping.get("period_mode", "red")),
        binning=binning_spec(grouping),
        forward_idx=list(pair[0]),
        backward_idx=list(pair[1]),
        alpha=alpha,
        beta=beta,
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        metadata=reference_observation_metadata(run),
        reference_resolver=reference,
        cancel_check=cancel_check,
    )
    return MuonDataset(
        effective_recipe=result.effective_recipe,
        time=result.time,
        asymmetry=result.asymmetry,
        error=result.error,
        metadata=result.metadata_with_alignment(run.metadata),
        run=run,
        computation_revision=result.computation_revision,
    )
