"""EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.

Set-as-BG capture-component subtraction.

Adapts WiMDA's ``SetBgButtonClick`` (NegMuAnalyse.pas) as a histogram-level
model subtraction: evaluate the unwanted components from the fitted parameters
and subtract from the counts, leaving the signal of interest. The derived
:class:`~asymmetry.core.data.dataset.Run` produced by
:func:`capture_background_run` can be re-fitted with
:func:`~asymmetry.core.negmu.fit.fit_capture_group`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.negmu.fit import CaptureModelSpec
from asymmetry.core.negmu.model import CaptureComponent, evaluate_capture_model


def subtract_capture_background(
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    fit: FitResult,
    spec: CaptureModelSpec,
    *,
    unwanted: Sequence[str],
) -> NDArray[np.float64]:
    """Return ``counts − Σ_{label in unwanted} amp_label·exp(−t/tau_label)``.

    The unwanted components are evaluated from the fitted parameters via
    :func:`~asymmetry.core.negmu.model.evaluate_capture_model`.  The result is
    the residual signal of interest.  Passing an empty ``unwanted`` sequence is
    the identity — the counts are returned unchanged.  (WiMDA SetBgButtonClick
    adapted as a histogram-level model subtraction, comparison.md §7.)

    .. note::

       The unwanted components are evaluated from the **non-polarised** model
       (plain ``amp·exp(−t/τ)``).  If the fit was run with a Phase-4
       ``polarisation`` multiplier, that ``(1 + P_pol(t))`` modulation is *not*
       reproduced here, so the subtraction is only exact for the unpolarised
       capture model.

    Parameters
    ----------
    time
        Time axis (μs) — the same array used for the fit.
    counts
        Raw count histogram on that time axis.
    fit
        :class:`~asymmetry.core.fitting.engine.FitResult` from a preceding
        :func:`~asymmetry.core.negmu.fit.fit_capture_group` /
        :func:`~asymmetry.core.negmu.fit.fit_capture_histogram` call.
    spec
        :class:`~asymmetry.core.negmu.fit.CaptureModelSpec` used for the fit —
        provides the component ordering and lifetime seeds.
    unwanted
        Labels of the components to subtract.  Each must appear in ``spec``.
        Duplicates are silently ignored.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    counts_arr = np.asarray(counts, dtype=np.float64)

    unwanted_set = set(unwanted)
    if not unwanted_set:
        return counts_arr.copy()

    all_components = {c.label: c for c in spec.components()}
    unknown = unwanted_set - all_components.keys()
    if unknown:
        raise ValueError(f"subtract_capture_background: labels {sorted(unknown)!r} are not in spec")

    unwanted_comps: list[CaptureComponent] = [
        all_components[lbl] for lbl in spec.labels() if lbl in unwanted_set
    ]

    params: dict[str, float] = {p.name: float(p.value) for p in fit.parameters}
    # Exclude the flat background from the unwanted sum; the plan specifies only
    # Σ amp_label·exp(−t/τ_label) per component, not the flat background term.
    # The flat background remains in the residual for downstream re-fitting.
    params_no_bg = {k: v for k, v in params.items() if k != "background"}

    # Guard against spec/fit mismatch.  build_capture_count_model defaults
    # amp to 0.0 when a key is absent, so a missing amp_<label> would silently
    # produce a zero-subtraction no-op with no error.
    missing = [lbl for lbl in unwanted_set if f"amp_{lbl}" not in params_no_bg]
    if missing:
        raise ValueError(
            f"subtract_capture_background: fit has no amplitude for "
            f"{sorted(missing)!r} — was fit run with a different spec?"
        )

    background_model = evaluate_capture_model(unwanted_comps, params_no_bg, time_arr)
    return counts_arr - background_model


def capture_background_run(
    dataset: MuonDataset,
    group_id: int,
    fit: FitResult,
    spec: CaptureModelSpec,
    *,
    unwanted: Sequence[str],
    run_number: int | None = None,
) -> Run:
    """Derived :class:`~asymmetry.core.data.dataset.Run` with unwanted components subtracted.

    Builds the raw grouped histogram for ``group_id`` via
    :func:`~asymmetry.core.fitting.grouped_time_domain.build_count_group`
    (``lifetime_corrected=False``), subtracts the unwanted components, and
    returns a minimal :class:`~asymmetry.core.data.dataset.Run` suitable for
    re-fitting with :func:`~asymmetry.core.negmu.fit.fit_capture_group`.

    The derived Run contains a single detector histogram (the grouped, subtracted
    counts) with ``group_id`` mapped to that detector.  This preserves the
    time-bin structure of the original fit exactly.

    Parameters
    ----------
    dataset
        Source dataset with a :class:`~asymmetry.core.data.dataset.Run` and
        grouping (same dataset passed to the original fit).
    group_id
        Group whose histogram is to be cleaned.
    fit
        :class:`~asymmetry.core.fitting.engine.FitResult` from the preceding fit
        of this group.
    spec
        :class:`~asymmetry.core.negmu.fit.CaptureModelSpec` used for that fit.
    unwanted
        Labels of the components to subtract.  Empty sequence → identity.
    run_number
        Run number for the derived Run.  Defaults to the source run's number.
    """
    source_run = dataset.run

    # build_count_group owns the t0 alignment, bunching, good-bin window and the
    # detector resolution (excluded_detectors, string-keyed group ids). Derive the
    # derived-run geometry from the trace it returns rather than re-deriving it by
    # hand from the raw grouping. The hand path mixed pre-bunch bin indices
    # (first_good − common_t0, in fine bins) with a post-bunch bin width, giving a
    # time axis off by a factor of the bunching whenever bunching_factor > 1 and
    # the good window started after t0; it also read the raw group detector list
    # (ignoring excluded_detectors) and used an int-keyed .get that silently fell
    # back to detector [1] for string-keyed groupings.
    group = build_count_group(dataset, group_id, lifetime_corrected=False)
    corrected = subtract_capture_background(group.time, group.counts, fit, spec, unwanted=unwanted)
    n_corrected = len(corrected)

    # Post-rebin bin width straight from the trace (group.time is evenly spaced).
    if len(group.time) > 1:
        bin_width = float(group.time[1] - group.time[0])
    else:
        bunch = max(1, int(source_run.grouping.get("bunching_factor", 1)))
        bin_width = float(source_run.histograms[0].bin_width) * bunch

    # group.time[0] is the post-rebin bin-centre offset from t0; rounding to the
    # nearest whole post-rebin bin carries at most the ½-bin midpoint shift that
    # rebinning already imposes (and that the original trace itself carries).
    axis_start_bins = max(0, int(round(float(group.time[0]) / bin_width))) if bin_width > 0 else 0

    last_good = axis_start_bins + n_corrected - 1
    if axis_start_bins > 0:
        padded = np.concatenate([np.zeros(axis_start_bins, dtype=np.float64), corrected])
    else:
        padded = corrected.copy()  # own the buffer — Histogram stores counts by reference

    histogram = Histogram(
        counts=padded,
        bin_width=bin_width,
        t0_bin=0,
        good_bin_start=axis_start_bins,
        good_bin_end=last_good,
    )

    grouping: dict = {
        "groups": {group_id: [1]},
        "group_names": {group_id: f"Group {group_id}"},
        "included_groups": {group_id: True},
        "first_good_bin": axis_start_bins,
        "last_good_bin": last_good,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "dead_time_us": [0.0],
        "bin_index_base": 1,
    }
    base_meta: dict = dict(source_run.metadata) if source_run is not None else {}
    base_meta["background_subtraction"] = {
        "group_id": int(group_id),
        "unwanted": list(unwanted),
        "spec_elements": list(spec.elements),
        "include_decay_background": bool(spec.include_decay_background),
    }

    number: int
    if run_number is not None:
        number = int(run_number)
    elif source_run is not None:
        number = int(source_run.run_number)
    else:
        number = 0

    return Run(
        run_number=number,
        histograms=[histogram],
        metadata=base_meta,
        grouping=grouping,
        source_file="",
    )
