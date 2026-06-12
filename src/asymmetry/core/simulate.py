"""Synthetic μSR run generation and statistics degradation.

Counts in each detector time bin are Poisson distributed; the expected counts
for detector *d* follow the muon lifetime envelope modulated by the asymmetry
signal seen by that detector:

    N_d(t) = N0_d · exp(−t/τ_μ) · [1 + a_d(t)] + b_d        (t ≥ 0)
    N_d(t) = b_d                                            (t < 0)

Simulation draws Poisson variates of these *expected counts* — never Gaussian
noise added to an asymmetry curve — so per-bin errors propagate correctly
through the real reduction chain (grouping → α-balanced asymmetry → error
formula) by construction. The α calibration enters as the relative efficiency
of the forward and backward detector groups: forward-group detectors receive
the weight 2α/(1+α) and backward-group detectors 2/(1+α), so the reduction
(F − αB)/(F + αB) recovers a(t) with α restored.

The functional behaviour follows WiMDA's ``Simulate.pas``/``DegradeStats.pas``
(study: ``docs/porting/simulate-mode/``), with the divergences documented in
the study's comparison.md: an optional flat background, background-only bins
before t0, deterministic seeding, and exact binomial thinning for degrade
factors below one.

This module is Qt-free and must stay importable without the GUI.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asymmetry.core.negmu.model import CaptureComponent

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import (
    combine_period_asymmetry,
    select_period_histograms,
)
from asymmetry.core.transform.asymmetry import compute_asymmetry, slice_to_good_window
from asymmetry.core.transform.grouping import (
    group_forward_backward,
    resolve_group_indices,
)
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.utils.constants import MUON_LIFETIME_US, PeriodMode

#: A per-group asymmetry signal: a callable evaluated on the time axis in
#: microseconds (returning the *fractional* asymmetry), or an array on the
#: detector's post-t0 bin grid (short arrays are zero-padded).
GroupSignal = Callable[[NDArray[np.float64]], NDArray[np.float64]] | NDArray[np.float64]

#: Grouping keys that describe two-period source files; a synthetic run is
#: single-period, so these must not be inherited from the template.
_PERIOD_KEYS = (
    "period_histograms",
    "period_reduced",
    "period_mode",
    "period_good_frames",
    "period_dead_time_us",
)


# ---------------------------------------------------------------------------
# Built-in ideal-instrument templates (simulate with no run loaded)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentTemplate:
    """A built-in idealised instrument that can stand in for a loaded run.

    :func:`simulate_run` only reads *structure* from its template — detector
    count, bin width, per-detector t0, the good-bin window and the
    forward/backward grouping — never the (zero) counts, so a built-in
    template carries empty histograms and a complete grouping. The
    :attr:`default_total_events` and :attr:`default_background_per_bin` are
    teaching-sensible starting points the dialog seeds its spinners with; the
    continuous instrument carries a non-zero flat background (the
    time-independent uncorrelated background characteristic of continuous
    sources — textbook Ch. 14), the pulsed one does not.
    """

    key: str
    label: str
    description: str
    n_detectors: int
    n_bins: int
    bin_width_us: float
    t0_bin: int
    forward_detectors: tuple[int, ...]
    backward_detectors: tuple[int, ...]
    alpha: float = 1.0
    good_frames: float = 1.0
    good_bin_start: int | None = None
    good_bin_end: int | None = None
    default_total_events: float = 10.0e6
    default_background_per_bin: float = 0.0
    instrument_name: str = ""
    field_state: str = "ZF"
    detector_orientation: str = "Longitudinal"

    def build(self) -> Run:
        """Materialise an empty-histogram :class:`Run` with this geometry."""
        first_good = self.t0_bin if self.good_bin_start is None else self.good_bin_start
        last_good = self.n_bins - 1 if self.good_bin_end is None else self.good_bin_end
        histograms = [
            Histogram(
                counts=np.zeros(self.n_bins, dtype=float),
                bin_width=self.bin_width_us,
                t0_bin=self.t0_bin,
                good_bin_start=first_good,
                good_bin_end=last_good,
            )
            for _ in range(self.n_detectors)
        ]
        groups = {
            1: list(self.forward_detectors),
            2: list(self.backward_detectors),
        }
        grouping = {
            "groups": groups,
            "group_names": {1: "Forward", 2: "Backward"},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": float(self.alpha),
            "t0_bin": self.t0_bin,
            "t_good_offset": 0,
            "first_good_bin": first_good,
            "last_good_bin": last_good,
            "bin_index_base": 1,
            "bunching_factor": 1,
            "good_frames": float(self.good_frames),
            "deadtime_correction": False,
            "dead_time_us": [0.0] * self.n_detectors,
            "included_groups": {1: True, 2: True},
        }
        return Run(
            run_number=0,
            histograms=histograms,
            metadata={
                "title": self.label,
                "instrument": self.instrument_name,
                "field_state": self.field_state,
                "detector_orientation": self.detector_orientation,
                "builtin_template": self.key,
            },
            grouping=grouping,
            source_file="",
        )


#: Built-in idealised instruments, keyed for the dialog template combo. The
#: pulsed F/B template mirrors an ISIS-style spectrometer (32 + 32 detectors,
#: 16 ns bins, a 32 μs window); the continuous template a PSI-style F/B pair
#: with fine 1 ns binning, a short 10 μs window and a flat background. The two
#: are the source archetypes the textbook contrasts (Ch. 14).
BUILTIN_TEMPLATES: dict[str, InstrumentTemplate] = {
    "ideal_pulsed_fb": InstrumentTemplate(
        key="ideal_pulsed_fb",
        label="Ideal pulsed F/B (ISIS-style)",
        description=(
            "Pulsed-source spectrometer: 32 forward + 32 backward detectors, "
            "16 ns bins over a 32 μs window, no uncorrelated background."
        ),
        n_detectors=64,
        n_bins=2000,
        bin_width_us=0.016,
        t0_bin=100,
        forward_detectors=tuple(range(1, 33)),
        backward_detectors=tuple(range(33, 65)),
        alpha=1.0,
        good_frames=1.0,
        default_total_events=40.0e6,
        default_background_per_bin=0.0,
        instrument_name="IDEAL-PULSED",
    ),
    "ideal_continuous_fb": InstrumentTemplate(
        key="ideal_continuous_fb",
        label="Ideal continuous F/B (PSI-style)",
        description=(
            "Continuous-source F/B pair: 1 ns bins over a 10 μs window with a "
            "flat uncorrelated background (10 counts/bin/detector)."
        ),
        n_detectors=2,
        n_bins=10000,
        bin_width_us=0.001,
        t0_bin=1000,
        forward_detectors=(1,),
        backward_detectors=(2,),
        alpha=1.0,
        good_frames=1.0,
        default_total_events=20.0e6,
        default_background_per_bin=10.0,
        instrument_name="IDEAL-CONTINUOUS",
    ),
}


def build_builtin_template(key: str) -> Run:
    """Build an empty-histogram :class:`Run` for a named built-in instrument.

    Raises :class:`KeyError` for an unknown key. See :data:`BUILTIN_TEMPLATES`
    for the available instruments.
    """
    try:
        template = BUILTIN_TEMPLATES[key]
    except KeyError:
        raise KeyError(
            f"Unknown built-in instrument template {key!r}; available: {sorted(BUILTIN_TEMPLATES)}."
        ) from None
    return template.build()


# ---------------------------------------------------------------------------
# Promoted screenshot-synthesis helpers (docs/screenshots/data/archetypes.py)
# ---------------------------------------------------------------------------


def poisson_asymmetry_errors(
    asymmetry: NDArray[np.float64],
    counts_per_bin: float = 5e4,
) -> NDArray[np.float64]:
    """Per-bin uncertainty (percent) derived from a target counts-per-bin.

    The asymmetry σ for a two-detector experiment scales as
    ``sqrt((1 - A^2) / N)``. Picking a counts-per-bin gives a realistic noise
    envelope without doing a histogram round-trip; use the full
    :func:`simulate_run` pipeline when the errors must come from actual
    Poisson histograms.
    """
    variance = np.clip(1.0 - (np.asarray(asymmetry) / 100.0) ** 2, 1e-3, 1.0) / counts_per_bin
    return np.sqrt(variance) * 100.0


def build_run_from_detector_asymmetries(
    *,
    run_number: int,
    detector_asymmetries: list[dict],
    title: str,
    temperature_k: float,
    field_g: float,
    bin_width_us: float = 0.005,
    n_bins: int = 2400,
    t0_bin: int = 100,
    n0_per_detector: float = 1.0e6,
    lifetime_us: float = MUON_LIFETIME_US,
    rng: np.random.Generator | None = None,
) -> tuple[Run, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Synthesise a full :class:`Run` from per-detector asymmetry signals.

    Each detector histogram is built as

        N_d(t) = N_{0,d} · exp(-t/τ) · (1 + A_d(t))   for t > 0
              = N_{0,d}                               for t ≤ 0

    with Poisson counting noise applied. The grouping payload puts one
    detector per group so the GUI's *Individual Groups* domain view shows a
    trace per detector. The function also returns the F–B asymmetry trace
    (using the first two groups) and a per-bin error estimate so a wrapper
    :class:`MuonDataset` carries a defensible time-domain view.

    This is the screenshot-archetype builder promoted into core: the flat
    pre-t0 plateau at N₀ is its historical convention (kept for byte-stable
    documentation data). :func:`simulate_run` is the template-driven
    instrument-faithful path, with background-only bins before t0.

    Parameters
    ----------
    detector_asymmetries
        One dict per detector. Each dict must define ``"asymmetry"`` (a
        ``(n_bins - t0_bin,)`` array giving the fractional A_d(t)) plus
        optional ``"label"`` and ``"n0"``.
    lifetime_us
        Decay constant of the envelope. Defaults to the canonical muon
        lifetime; the screenshot archetypes pass their legacy rounded value
        to keep documentation data byte-stable.
    """
    rng = rng if rng is not None else np.random.default_rng()
    bins = np.arange(n_bins)
    time_full = (bins - t0_bin) * bin_width_us
    decay = np.exp(-np.maximum(time_full, 0.0) / lifetime_us)

    histograms: list[Histogram] = []
    for det in detector_asymmetries:
        amplitude = np.zeros(n_bins, dtype=float)
        amplitude[t0_bin:] = det["asymmetry"][: n_bins - t0_bin]
        n0 = float(det.get("n0", n0_per_detector))
        # Pre-trigger flat counts come from the implantation rate alone.
        clean = np.full(n_bins, n0, dtype=float)
        clean[t0_bin:] = n0 * decay[t0_bin:] * (1.0 + amplitude[t0_bin:])
        clean = np.clip(clean, 1e-3, None)
        noisy = rng.poisson(clean).astype(float)
        histograms.append(
            Histogram(
                counts=noisy,
                bin_width=bin_width_us,
                t0_bin=t0_bin,
                good_bin_start=t0_bin,
                good_bin_end=n_bins - 1,
            )
        )

    n_groups = len(histograms)
    # Grouping entries are 1-based detector numbers (the repo-wide convention
    # decoded by resolve_group_indices); group g holds detector g.
    groups = {gid: [gid] for gid in range(1, n_groups + 1)}
    group_names = {
        gid: det.get("label", f"Group {gid}")
        for gid, det in zip(range(1, n_groups + 1), detector_asymmetries, strict=True)
    }
    grouping = {
        "groups": groups,
        "group_names": group_names,
        "forward_group": 1,
        "backward_group": min(2, n_groups),
        "alpha": 1.0,
        "t0_bin": t0_bin,
        "t_good_offset": 0,
        "first_good_bin": t0_bin,
        "last_good_bin": n_bins - 1,
        "bin_index_base": 0,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "included_groups": {gid: True for gid in groups},
    }

    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"title": title, "temperature": temperature_k, "field": field_g},
        grouping=grouping,
    )

    # F-B asymmetry from the first two groups for the wrapper MuonDataset.
    fwd = histograms[0].counts.astype(float)
    bwd = histograms[min(1, n_groups - 1)].counts.astype(float)
    fwd_post = fwd[t0_bin:]
    bwd_post = bwd[t0_bin:]
    denom = fwd_post + bwd_post
    raw_asym = np.where(denom > 0, (fwd_post - bwd_post) / denom, 0.0) * 100.0
    time_post = time_full[t0_bin:]
    # Exact Poisson propagation of (F−B)/(F+B): var = 4FB/(F+B)³ = (1−A²)/N.
    variance = np.clip(1.0 - (raw_asym / 100.0) ** 2, 1e-3, 1.0)
    error_post = np.where(denom > 0, np.sqrt(variance / np.clip(denom, 1.0, None)), 0.0) * 100.0
    return run, time_post, raw_asym, error_post


# ---------------------------------------------------------------------------
# Template-driven forward model
# ---------------------------------------------------------------------------


def _detector_group_map(grouping: dict, n_detectors: int) -> dict[int, int]:
    """Map 0-based detector index → group id, first containing group wins."""
    groups = grouping.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            "Simulation requires a detector grouping definition (grouping['groups']) "
            "on the template run."
        )
    mapping: dict[int, int] = {}
    for key in groups:
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        for det in resolve_group_indices(groups, gid):
            if 0 <= det < n_detectors:
                mapping.setdefault(det, gid)
    return mapping


def _signal_values(
    signal: GroupSignal | None,
    t_post: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a group signal on the detector's post-t0 time axis."""
    if signal is None:
        return np.zeros_like(t_post)
    if callable(signal):
        return np.asarray(signal(t_post), dtype=float)
    values = np.zeros_like(t_post)
    arr = np.asarray(signal, dtype=float)
    n = min(arr.size, t_post.size)
    values[:n] = arr[:n]
    return values


def expected_counts(
    template: Run,
    group_signals: Mapping[int, GroupSignal],
    *,
    total_events: float,
    group_weights: Mapping[int, float] | None = None,
    background_per_bin: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Expected (noise-free) per-detector count histograms.

    The deterministic core of :func:`simulate_run_from_group_signals`,
    exposed so tests and diagnostics can compare the sampled histograms (or a
    reduction of them) against the exact expectation.

    ``total_events`` is the expected number of detected decay events summed
    over all detectors and the post-t0 histogram window — the per-bin rate at
    t = 0 uses the exact telescoping normalisation
    ``n0 = N_d · (1 − exp(−Δt/τ_μ))``, so the window sum equals
    ``total_events · (1 − exp(−T/τ_μ))`` up to the (simulated) truncation of
    the lifetime envelope at the histogram end. Background counts are *in
    addition to* this budget.

    ``group_weights`` rescales each group's per-detector rate; weights are
    normalised over the assigned detectors so the total event budget is
    independent of the weighting (this is how the α split is applied without
    distorting the run-level rate).
    """
    if not template.histograms:
        raise ValueError("Simulation requires a template run with detector histograms.")
    if not np.isfinite(total_events) or total_events <= 0:
        raise ValueError("total_events must be a positive, finite event budget.")
    if background_per_bin < 0:
        raise ValueError("background_per_bin must be non-negative.")

    histograms = template.histograms
    n_det = len(histograms)
    det_group = _detector_group_map(template.grouping, n_det)

    weights = np.ones(n_det, dtype=float)
    if group_weights is not None:
        for det in range(n_det):
            gid = det_group.get(det)
            if gid is not None and gid in group_weights:
                weights[det] = float(group_weights[gid])
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("group_weights must be finite and non-negative.")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("group_weights must leave at least one detector with rate.")

    # Detectors in the same group with the same post-t0 grid share one model
    # evaluation — a CompositeModel on a 64-detector template is evaluated
    # once per group, not once per detector.
    signal_cache: dict[tuple[int | None, int, float], NDArray[np.float64]] = {}

    expected: list[NDArray[np.float64]] = []
    for det, hist in enumerate(histograms):
        n_bins = hist.n_bins
        t0_bin = max(0, int(hist.t0_bin))
        bin_width = float(hist.bin_width)
        n_post = max(0, n_bins - t0_bin)
        t_post = np.arange(n_post, dtype=float) * bin_width

        clean = np.full(n_bins, float(background_per_bin), dtype=float)
        if n_post:
            # Exact per-bin envelope normalisation: summing
            # n0·exp(−i·Δt/τ) over the window telescopes to
            # N_d·(1 − exp(−T/τ)) with n0 = N_d·(1 − exp(−Δt/τ)).
            # (WiMDA uses the first-order N_d·Δt/τ.)
            n_events_det = total_events * weights[det] / total_weight
            n0 = n_events_det * (1.0 - np.exp(-bin_width / MUON_LIFETIME_US))
            gid = det_group.get(det)
            cache_key = (gid, n_post, bin_width)
            signal = signal_cache.get(cache_key)
            if signal is None:
                signal = _signal_values(group_signals.get(gid), t_post)
                signal_cache[cache_key] = signal
            envelope = n0 * np.exp(-t_post / MUON_LIFETIME_US) * (1.0 + signal)
            clean[t0_bin:] += np.clip(envelope, 0.0, None)
        expected.append(clean)
    return expected


def simulate_run_from_group_signals(
    template: Run,
    group_signals: Mapping[int, GroupSignal],
    *,
    total_events: float,
    seed: int = 0,
    group_weights: Mapping[int, float] | None = None,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
    simulation_metadata: Mapping[str, Any] | None = None,
) -> Run:
    """Simulate a run from per-group fractional asymmetry signals.

    The scriptable multi-group seam: each entry of ``group_signals`` assigns a
    signal a_g(t) to one detector group of the template's grouping; detectors
    in unlisted groups receive the bare lifetime envelope. Bin structure,
    per-detector t0, the good-bin window and the grouping are all taken from
    ``template``. Sampling is a single seeded :class:`numpy.random.Generator`
    drawing ``poisson(expected)`` per detector, in detector order — a fixed
    seed reproduces the run bit-for-bit.

    The returned :class:`Run` carries ``metadata["synthetic"] = True`` and a
    ``metadata["simulation"]`` provenance dict; deadtimes are zeroed in the
    grouping (the synthetic counts contain no deadtime distortion, so zero is
    the true instrument description).
    """
    expected = expected_counts(
        template,
        group_signals,
        total_events=total_events,
        group_weights=group_weights,
        background_per_bin=background_per_bin,
    )

    return _sample_and_build_run(
        template,
        expected,
        seed=seed,
        total_events=total_events,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title,
        default_title="Simulated run",
        simulation_metadata=simulation_metadata,
    )


#: Template metadata keys a synthetic run inherits (they parameterise models —
#: precession frequencies, etc. — or label the run).
_INHERITED_METADATA_KEYS = (
    "instrument",
    "temperature",
    "field",
    "field_state",
    "field_direction",
    "detector_orientation",
)


def _synthetic_run_grouping(template: Run, n_histograms: int) -> dict[str, Any]:
    """Base grouping for a synthetic run: template grouping with deadtimes off.

    The period payload (which can hold full per-detector histogram arrays for a
    combined two-period template) is stripped before the deep copy, and the
    deadtimes are zeroed — the synthetic counts carry no deadtime distortion, so
    zero is the true instrument description. Callers add an ``alpha`` override or
    a fresh period payload on top.
    """
    grouping: dict[str, Any] = copy.deepcopy(
        {k: v for k, v in template.grouping.items() if k not in _PERIOD_KEYS}
    )
    grouping["deadtime_correction"] = False
    grouping["dead_time_us"] = [0.0] * n_histograms
    return grouping


def _synthetic_run_metadata(
    template: Run,
    *,
    number: int,
    seed: int,
    total_events: float,
    background_per_bin: float,
    title: str | None,
    default_title: str,
    simulation_metadata: Mapping[str, Any] | None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity + provenance metadata shared by every synthetic-run builder.

    Carries the inherited template metadata plus the ``synthetic`` marker and a
    ``simulation`` provenance block (seed, event budget, background, template
    run/source, then any ``simulation_metadata`` the caller adds). This is the
    single source of the synthetic-run identity contract.
    """
    provenance: dict[str, Any] = {
        "seed": int(seed),
        "total_events": float(total_events),
        "background_per_bin": float(background_per_bin),
        "template_run_number": template.run_number,
        "template_source_file": template.source_file,
    }
    if simulation_metadata:
        provenance.update(dict(simulation_metadata))

    metadata: dict[str, Any] = {
        key: template.metadata[key] for key in _INHERITED_METADATA_KEYS if key in template.metadata
    }
    metadata.update(
        {
            "run_number": number,
            "run_label": f"SIM {number}",
            "title": title if title is not None else default_title,
            "synthetic": True,
            "simulation": provenance,
        }
    )
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return metadata


def _sample_and_build_run(
    template: Run,
    expected: list[NDArray[np.float64]],
    *,
    seed: int,
    total_events: float,
    background_per_bin: float,
    run_number: int | None,
    title: str | None,
    default_title: str,
    simulation_metadata: Mapping[str, Any] | None,
) -> Run:
    """Poisson-sample expected per-detector counts and assemble a synthetic Run.

    Shared by :func:`simulate_run_from_group_signals` and
    :func:`simulate_double_pulse_run` so both carry the same provenance and
    metadata (template run/source, detector orientation, deadtime zeroing) via
    the shared :func:`_synthetic_run_grouping` / :func:`_synthetic_run_metadata`
    contract. The seeded sampling draws ``poisson(expected)`` per detector in
    detector order.
    """
    rng = np.random.default_rng(seed)
    histograms = [
        Histogram(
            counts=rng.poisson(clean).astype(float),
            bin_width=float(hist.bin_width),
            t0_bin=int(hist.t0_bin),
            good_bin_start=int(hist.good_bin_start),
            good_bin_end=int(hist.good_bin_end),
        )
        for hist, clean in zip(template.histograms, expected, strict=True)
    ]

    number = template.run_number if run_number is None else int(run_number)
    grouping = _synthetic_run_grouping(template, len(histograms))
    metadata = _synthetic_run_metadata(
        template,
        number=number,
        seed=seed,
        total_events=total_events,
        background_per_bin=background_per_bin,
        title=title,
        default_title=default_title,
        simulation_metadata=simulation_metadata,
    )

    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="",
    )


def _bind_model_signal_percent(
    model: Any,
    params: Mapping[str, float],
) -> tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], str]:
    """Bind a model + parameters to an ``a(t)``-in-percent callable.

    ``model`` is either a ``CompositeModel`` (its ``function`` bound with
    ``params``) or a plain callable ``a(t, **params)`` returning the asymmetry
    in percent — the same convention the fit panel uses. Returns the bound
    callable and a human-readable expression for provenance.
    """
    params = dict(params)
    if hasattr(model, "function") and callable(model.function):
        base_fn = model.function

        def signal_percent(t: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(base_fn(t, **params), dtype=float)

        expression = model.formula_string() if hasattr(model, "formula_string") else repr(model)
    elif callable(model):

        def signal_percent(t: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(model(t, **params), dtype=float)

        expression = getattr(model, "__name__", "custom callable")
    else:
        raise TypeError("model must be a CompositeModel or a callable a(t) in percent.")
    return signal_percent, expression


def _fb_group_signals_and_weights(
    grouping: dict,
    n_histograms: int,
    *,
    forward_signal: GroupSignal,
    backward_signal: GroupSignal,
    forward_gid: int,
    backward_gid: int,
    alpha: float,
) -> tuple[dict[int, GroupSignal], dict[int, float]]:
    """Forward/backward group signals and α-split weights for a template.

    Backward-group detectors see ``−a(t)`` and forward-group (every
    non-backward) detectors ``+a(t)``. The α split fixes the GROUP TOTALS at
    F:B = α; each side's budget share is divided by its detector count so
    unequal group sizes cannot skew the ratio (a per-detector weight alone
    would give F/B = α·n_F/n_B). Shared by :func:`simulate_run` and the
    per-period synthesis so both apply the identical balance.
    """
    det_groups = _detector_group_map(grouping, n_histograms)
    group_ids = sorted(set(det_groups.values()))
    n_backward = sum(1 for gid in det_groups.values() if gid == backward_gid)
    n_forward = len(det_groups) - n_backward
    group_signals: dict[int, GroupSignal] = {}
    group_weights: dict[int, float] = {}
    for gid in group_ids:
        if gid == backward_gid:
            group_signals[gid] = backward_signal
            group_weights[gid] = 2.0 / (1.0 + alpha) / max(1, n_backward)
        else:
            group_signals[gid] = forward_signal
            group_weights[gid] = 2.0 * alpha / (1.0 + alpha) / max(1, n_forward)
    if forward_gid not in group_signals or backward_gid not in group_signals:
        raise ValueError(
            "Template grouping must assign detectors to both the forward and backward groups."
        )
    return group_signals, group_weights


def simulate_run(
    template: Run,
    model: Any,
    parameters: Mapping[str, float] | None = None,
    *,
    total_events: float,
    seed: int = 0,
    alpha: float | None = None,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Simulate a forward/backward run from a fit model (WiMDA Simulate).

    ``model`` is either a ``CompositeModel`` (its ``function`` is bound with
    ``parameters``) or a plain callable returning the asymmetry **in percent**
    on a time axis in microseconds — the same convention the fit panel uses.
    Forward-group detectors see ``+a(t)``, backward-group detectors ``−a(t)``
    (every non-backward group counts as forward, as in WiMDA). The α split
    fixes the forward/backward group *totals* at 2α/(1+α) : 2/(1+α) of the
    event budget, divided equally among each side's detectors, so the
    reduction recovers a(t) with α restored regardless of group sizes.

    α defaults to the template grouping's balance factor. See
    :func:`simulate_run_from_group_signals` for the sampling and provenance
    contract; the generating model expression, parameter values and α are
    recorded in ``metadata["simulation"]``.
    """
    if not template.histograms:
        raise ValueError("Simulation requires a template run with detector histograms.")

    params = dict(parameters or {})
    signal_percent, expression = _bind_model_signal_percent(model, params)

    grouping = template.grouping if isinstance(template.grouping, dict) else {}
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    if alpha is None:
        try:
            alpha = float(grouping.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite.")

    def forward_signal(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return signal_percent(t) / 100.0

    def backward_signal(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return -signal_percent(t) / 100.0

    group_signals, group_weights = _fb_group_signals_and_weights(
        grouping,
        len(template.histograms),
        forward_signal=forward_signal,
        backward_signal=backward_signal,
        forward_gid=forward_gid,
        backward_gid=backward_gid,
        alpha=alpha,
    )

    return simulate_run_from_group_signals(
        template,
        group_signals,
        total_events=total_events,
        seed=seed,
        group_weights=group_weights,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title if title is not None else f"Simulated: {expression}",
        simulation_metadata={
            "model": expression,
            "parameters": {name: float(value) for name, value in params.items()},
            "alpha": float(alpha),
        },
    )


def simulate_count_run(
    template: Run,
    model: Any,
    parameters: Mapping[str, float] | None = None,
    *,
    total_events: float,
    seed: int = 0,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Simulate per-group single-histogram count data (WiMDA evfactor path).

    The count-domain counterpart of :func:`simulate_run`. Where ``simulate_run``
    imprints the antisymmetric ``±a(t)`` of a forward/backward asymmetry
    experiment, this imprints the **same** ``+a(t)`` on every detector group, so
    each group is an independent single-histogram measurement

        N_g(t) = N₀_g · exp(−t/τ_μ) · [1 + a(t)] + b        (t ≥ 0)

    with no balancing backward detector — the non-FB convention WiMDA's
    ``Simulate.pas`` evfactor branch (``nn = a(t)/ghists·evfactor·e^{−t/τ}``)
    produces. The event budget is divided equally over the detectors (no α
    split), so each group's ``N₀`` follows its detector count.

    The result is fittable by :func:`asymmetry.core.fitting.count_domain.fit_single_histogram`
    (``side="forward"`` on any group), recovering the generating ``N₀``, ``a(t)``
    amplitude and flat background. Provenance records ``count_mode = True``. Use
    :func:`simulate_run` when the data must instead exercise the α-free
    forward/backward count fit (``fit_fb_alpha``), which needs the antisymmetric
    pair.
    """
    if not template.histograms:
        raise ValueError("Simulation requires a template run with detector histograms.")

    params = dict(parameters or {})
    signal_percent, expression = _bind_model_signal_percent(model, params)

    def forward_signal(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return signal_percent(t) / 100.0

    grouping = template.grouping if isinstance(template.grouping, dict) else {}
    det_groups = _detector_group_map(grouping, len(template.histograms))
    group_ids = sorted(set(det_groups.values()))
    if not group_ids:
        raise ValueError("Template grouping must assign detectors to at least one group.")
    group_signals: dict[int, GroupSignal] = {gid: forward_signal for gid in group_ids}

    return simulate_run_from_group_signals(
        template,
        group_signals,
        total_events=total_events,
        seed=seed,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title if title is not None else f"Simulated counts: {expression}",
        simulation_metadata={
            "model": expression,
            "parameters": {name: float(value) for name, value in params.items()},
            "count_mode": True,
        },
    )


def simulate_double_pulse_run(
    template: Run,
    model: Any,
    parameters: Mapping[str, float] | None = None,
    *,
    total_events: float,
    dpsep_us: float,
    alpha: float | None = None,
    background_per_bin: float = 0.0,
    seed: int = 0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Simulate an ISIS double-pulse run (WiMDA double-pulse count model).

    Two muon pulses separated by ``dpsep_us`` each carry the polarization,
    evaluated at ``t ± dpsep/2`` and weighted by ``exp(∓dpsep/2τ_μ)``; the
    envelope stays at ``t`` and the single-pulse limit is ``dpsep_us → 0``. Used
    to round-trip the double-pulse single-histogram fit (the generating model
    matches :func:`asymmetry.core.fitting.count_domain.fit_single_histogram`'s
    ``dpsep`` model).
    """
    if not template.histograms:
        raise ValueError("Simulation requires a template run with detector histograms.")
    if not np.isfinite(dpsep_us) or dpsep_us < 0:
        raise ValueError("dpsep_us must be non-negative and finite.")
    if not np.isfinite(total_events) or total_events <= 0:
        raise ValueError("total_events must be a positive, finite event budget.")

    params = dict(parameters or {})
    base_fn = model.function if hasattr(model, "function") and callable(model.function) else model
    expression = model.formula_string() if hasattr(model, "formula_string") else repr(model)

    def signal_fraction(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(base_fn(t, **params), dtype=float) / 100.0

    grouping = template.grouping if isinstance(template.grouping, dict) else {}
    backward_gid = int(grouping.get("backward_group", 2))
    if alpha is None:
        try:
            alpha = float(grouping.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite.")

    det_groups = _detector_group_map(grouping, len(template.histograms))
    n_backward = sum(1 for gid in det_groups.values() if gid == backward_gid)
    n_forward = len(det_groups) - n_backward

    def _weight(gid: int | None) -> float:
        if gid == backward_gid:
            return 2.0 / (1.0 + alpha) / max(1, n_backward)
        return 2.0 * alpha / (1.0 + alpha) / max(1, n_forward)

    weights = np.array([_weight(det_groups.get(det)) for det in range(len(template.histograms))])
    total_weight = float(weights.sum()) or 1.0

    tau = float(MUON_LIFETIME_US)
    dpsep2 = float(dpsep_us) / 2.0
    c1, c2 = np.exp(-dpsep2 / tau), np.exp(dpsep2 / tau)

    expected: list[NDArray[np.float64]] = []
    for det, hist in enumerate(template.histograms):
        n_bins = hist.n_bins
        t0_bin = max(0, int(hist.t0_bin))
        bin_width = float(hist.bin_width)
        n_post = max(0, n_bins - t0_bin)
        clean = np.full(n_bins, float(background_per_bin), dtype=float)
        if n_post:
            u = np.arange(n_post, dtype=float) * bin_width
            sign = -1.0 if det_groups.get(det) == backward_gid else 1.0
            a1 = signal_fraction(u + dpsep2)
            # Clamp the gated-out times (u <= dpsep2) to >= 0 before evaluating
            # the model so a model that raises / returns non-finite for t < 0
            # cannot poison the zero-weighted early bins (see the matching guard
            # in count_domain._double_pulse_single_model).
            a2 = signal_fraction(np.maximum(u - dpsep2, 0.0))
            gate = (u > dpsep2).astype(float)
            factor = 0.5 * (c1 * (1.0 + sign * a1) + gate * c2 * (1.0 + sign * a2))
            n_events_det = total_events * weights[det] / total_weight
            n0 = n_events_det * (1.0 - np.exp(-bin_width / tau))
            clean[t0_bin:] += np.clip(n0 * np.exp(-u / tau) * factor, 0.0, None)
        expected.append(clean)

    return _sample_and_build_run(
        template,
        expected,
        seed=seed,
        total_events=total_events,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title,
        default_title=f"Double-pulse: {expression}",
        simulation_metadata={
            "dpsep_us": float(dpsep_us),
            "alpha": float(alpha),
            "model": expression,
            "parameters": {name: float(value) for name, value in params.items()},
        },
    )


# ---------------------------------------------------------------------------
# Two-period (red/green) synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodSpec:
    """One period's forward/backward signal for a two-period synthetic run.

    Each period is an independent measurement: its own model (or the same model
    with different parameters), event budget and α balance, all imprinted with
    the standard forward/backward antisymmetry so the period reduces exactly
    like a real single-period run. ``total_events`` and ``alpha`` fall back to
    the run-level budget and the template grouping's balance when ``None``.
    ``scale`` multiplies the whole signal amplitude (``0`` makes the period a
    flat reference — the usual light-off / RF-off green — so G−R recovers the
    other period); it keeps the model's own provenance intact rather than
    wrapping it in an opaque closure.
    """

    model: Any
    parameters: Mapping[str, float] | None = None
    total_events: float | None = None
    alpha: float | None = None
    scale: float = 1.0
    label: str = ""


def simulate_two_period_run(
    template: Run,
    periods: list[PeriodSpec],
    *,
    total_events: float,
    seed: int = 0,
    background_per_bin: float = 0.0,
    period_mode: PeriodMode | str = PeriodMode.RED,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Simulate a two-period (red/green) run from per-period models.

    Pulsed-source runs can be recorded in period mode (light-on/off, RF-on/off,
    ALC steps): one file holds several period histograms. This synthesises the
    two-period case — ``periods[0]`` is *red* (period 1), ``periods[1]`` *green*
    (period 2) — through the same Poisson forward model as :func:`simulate_run`,
    one independent draw per period from a single seeded generator (a fixed seed
    reproduces both periods bit-for-bit).

    The returned :class:`Run` carries the loader's two-period payload
    (``period_histograms``, ``period_reduced``, ``period_good_frames``,
    ``period_dead_time_us``, ``period_mode``) with ``run.histograms`` cloned from
    the red period, so :func:`asymmetry.core.io.periods.select_period`, the
    green∓red combinations in :func:`reduce_run_to_dataset`, and
    :func:`degrade_run` all operate on it exactly as on a loaded period-mode
    file. The NeXus writer still emits a single period (study decision); a
    two-period synthetic run lives in-memory and through the project.
    """
    if not template.histograms:
        raise ValueError("Simulation requires a template run with detector histograms.")
    if len(periods) != 2:
        raise ValueError("simulate_two_period_run needs exactly two PeriodSpecs (red, green).")
    if not np.isfinite(total_events) or total_events <= 0:
        raise ValueError("total_events must be a positive, finite event budget.")

    grouping = template.grouping if isinstance(template.grouping, dict) else {}
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    default_alpha = grouping.get("alpha", 1.0)
    try:
        default_alpha = float(default_alpha)
    except (TypeError, ValueError):
        default_alpha = 1.0
    good_frames = grouping.get("good_frames", 1.0)
    try:
        good_frames = float(good_frames)
    except (TypeError, ValueError):
        good_frames = 1.0
    n_det = len(template.histograms)

    # One generator draws red then green in order, so a fixed seed is
    # reproducible and the two periods are independent.
    rng = np.random.default_rng(seed)

    # One deadtime-zeroed base copy (shared synthetic-run contract); the
    # reduction grouping at the loader default α = 1 and the final run grouping
    # both derive from it by shallow override, so each period reduces — and the
    # synthetic counts (which carry no deadtime distortion) read back — exactly
    # like a freshly loaded single-period run.
    base_grouping = _synthetic_run_grouping(template, n_det)
    reduce_grouping = {**base_grouping, "alpha": 1.0}

    period_histograms: list[list[Histogram]] = []
    period_reduced: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
    period_provenance: list[dict[str, Any]] = []
    for spec in periods:
        signal_percent, expression = _bind_model_signal_percent(spec.model, spec.parameters or {})
        scale = float(spec.scale)

        def forward_signal(
            t: NDArray[np.float64],
            _fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] = signal_percent,
            _scale: float = scale,
        ) -> NDArray[np.float64]:
            return _scale * _fn(t) / 100.0

        def backward_signal(
            t: NDArray[np.float64],
            _fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] = signal_percent,
            _scale: float = scale,
        ) -> NDArray[np.float64]:
            return -_scale * _fn(t) / 100.0

        alpha = default_alpha if spec.alpha is None else float(spec.alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("PeriodSpec.alpha must be positive and finite.")
        group_signals, group_weights = _fb_group_signals_and_weights(
            grouping,
            n_det,
            forward_signal=forward_signal,
            backward_signal=backward_signal,
            forward_gid=forward_gid,
            backward_gid=backward_gid,
            alpha=alpha,
        )
        events = total_events if spec.total_events is None else float(spec.total_events)
        expected = expected_counts(
            template,
            group_signals,
            total_events=events,
            group_weights=group_weights,
            background_per_bin=background_per_bin,
        )
        hists = [
            Histogram(
                counts=rng.poisson(clean).astype(float),
                bin_width=float(hist.bin_width),
                t0_bin=int(hist.t0_bin),
                good_bin_start=int(hist.good_bin_start),
                good_bin_end=int(hist.good_bin_end),
            )
            for hist, clean in zip(template.histograms, expected, strict=True)
        ]
        period_histograms.append(hists)
        period_reduced.append(_reduce_histograms(hists, reduce_grouping))
        period_provenance.append(
            {
                "model": expression,
                "parameters": {
                    name: float(value) for name, value in dict(spec.parameters or {}).items()
                },
                "alpha": float(alpha),
                "scale": scale,
                "total_events": float(events),
                "label": spec.label,
            }
        )

    run_grouping = {
        **base_grouping,
        "good_frames": good_frames,
        "period_histograms": period_histograms,
        "period_reduced": period_reduced,
        "period_good_frames": [good_frames, good_frames],
        "period_dead_time_us": [[0.0] * n_det, [0.0] * n_det],
        "period_mode": str(period_mode),
    }

    number = template.run_number if run_number is None else int(run_number)
    metadata = _synthetic_run_metadata(
        template,
        number=number,
        seed=seed,
        total_events=total_events,
        background_per_bin=background_per_bin,
        title=title,
        default_title="Simulated two-period run",
        simulation_metadata={"two_period": True, "periods": period_provenance},
        extra_metadata={"period_count": 2},
    )

    # run.histograms mirror the red period (the loader convention for a combined
    # two-period run); clone so later edits cannot alias the period payload.
    histograms = [
        Histogram(
            counts=hist.counts.copy(),
            bin_width=hist.bin_width,
            t0_bin=hist.t0_bin,
            good_bin_start=hist.good_bin_start,
            good_bin_end=hist.good_bin_end,
        )
        for hist in period_histograms[0]
    ]
    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=run_grouping,
        source_file="",
    )


# ---------------------------------------------------------------------------
# Multi-group simulation (per-group amplitude / phase, from a grouped fit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSignalSpec:
    """Per-group amplitude, relative phase and count-rate weight for a TF ring.

    The simulation analogue of a grouped time-domain fit's per-group nuisance
    block (``amplitude``, ``relative_phase``, ``N0`` in
    :mod:`asymmetry.core.fitting.grouped_time_domain`): each detector group sees
    the *shared* normalised polarisation scaled by its own ``amplitude`` and
    phase-shifted by its own ``relative_phase``, with ``n0_weight`` setting its
    relative count rate (detector efficiency).
    """

    group_id: int
    amplitude: float
    relative_phase: float = 0.0
    n0_weight: float = 1.0
    label: str = ""


def _phase_parameter_names(parameter_names: list[str]) -> list[str]:
    """Names of the model's phase parameter(s).

    Uses the repo-wide :func:`~asymmetry.core.fitting.parameters.split_parameter_name`
    convention (base name ``"phase"``, with or without a numeric component
    suffix) so simulation and grouped fitting agree on which parameters carry a
    phase — rather than a private string-prefix heuristic that would drift.
    """
    from asymmetry.core.fitting.parameters import split_parameter_name

    return [name for name in parameter_names if split_parameter_name(str(name))[0] == "phase"]


def build_group_signals(
    model: Any,
    specs: Mapping[int, GroupSignalSpec] | list[GroupSignalSpec],
    *,
    base_parameters: Mapping[str, float] | None = None,
) -> tuple[dict[int, GroupSignal], dict[int, float]]:
    """Build ``(group_signals, group_weights)`` for a multi-group simulation.

    ``model`` is a *normalised* polarisation model (amplitude 1, background 0 —
    the grouped-fit contract), so its output is the dimensionless polarisation
    P(t) ≈ [−1, 1] directly (no percent conversion: the per-group ``amplitude``
    owns the scale, exactly as in
    :func:`~asymmetry.core.fitting.grouped_time_domain.build_grouped_count_model`).
    Each spec forms ``a_g(t) = amplitude · P(t)`` and adds ``relative_phase``
    (radians) to the model's phase parameter(s); ``base_parameters`` supplies
    the shared model values. The returned weights are the per-group
    ``n0_weight`` values, fed to :func:`simulate_run_from_group_signals` which
    normalises them over the assigned detectors.

    Raises :class:`ValueError` if a non-zero ``relative_phase`` is requested for
    a model with no phase parameter.
    """
    spec_list = list(specs.values()) if isinstance(specs, Mapping) else list(specs)
    base = dict(base_parameters or {})
    if hasattr(model, "function") and callable(model.function):
        model_fn = model.function
        param_names = list(getattr(model, "param_names", []))
    elif callable(model):
        model_fn = model
        import inspect

        param_names = [
            name
            for name, parameter in inspect.signature(model).parameters.items()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ][1:]  # drop the time axis (first positional)
    else:
        raise TypeError("model must be a CompositeModel or a callable a(t) in percent.")
    phase_names = _phase_parameter_names(param_names)

    group_signals: dict[int, GroupSignal] = {}
    group_weights: dict[int, float] = {}
    for spec in spec_list:
        params = dict(base)
        if not np.isclose(spec.relative_phase, 0.0):
            if not phase_names:
                raise ValueError(
                    "A non-zero relative_phase requires a phase-capable model "
                    f"(group {spec.group_id} requested {spec.relative_phase})."
                )
            for phase_name in phase_names:
                params[phase_name] = float(params.get(phase_name, 0.0)) + float(spec.relative_phase)

        def signal(
            t: NDArray[np.float64],
            _amp: float = float(spec.amplitude),
            _params: dict[str, float] = params,
        ) -> NDArray[np.float64]:
            return _amp * np.asarray(model_fn(t, **_params), dtype=float)

        group_signals[int(spec.group_id)] = signal
        group_weights[int(spec.group_id)] = float(spec.n0_weight)
    return group_signals, group_weights


def simulate_multi_group_run(
    template: Run,
    model: Any,
    specs: Mapping[int, GroupSignalSpec] | list[GroupSignalSpec],
    *,
    total_events: float,
    seed: int = 0,
    base_parameters: Mapping[str, float] | None = None,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Simulate a run with a distinct amplitude/phase per detector group.

    The multi-group counterpart of :func:`simulate_run`: instead of the F/B α
    split it drives :func:`simulate_run_from_group_signals` with per-group
    signals built by :func:`build_group_signals` from ``specs`` and the shared
    normalised ``model``. Use it to synthesise a TF ring whose groups differ in
    phase, seeded from a grouped time-domain fit's nuisance parameters.
    """
    group_signals, group_weights = build_group_signals(
        model, specs, base_parameters=base_parameters
    )
    spec_list = list(specs.values()) if isinstance(specs, Mapping) else list(specs)
    expression = (
        model.formula_string()
        if hasattr(model, "formula_string")
        else getattr(model, "__name__", "custom callable")
    )
    return simulate_run_from_group_signals(
        template,
        group_signals,
        total_events=total_events,
        seed=seed,
        group_weights=group_weights,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title if title is not None else f"Simulated multi-group: {expression}",
        simulation_metadata={
            "model": expression,
            "parameters": {
                name: float(value) for name, value in dict(base_parameters or {}).items()
            },
            "multi_group": True,
            "group_specs": [
                {
                    "group_id": int(spec.group_id),
                    "amplitude": float(spec.amplitude),
                    "relative_phase": float(spec.relative_phase),
                    "n0_weight": float(spec.n0_weight),
                }
                for spec in spec_list
            ],
        },
    )


def group_specs_from_grouped_fit(grouped_result: Any) -> list[GroupSignalSpec]:
    """Extract per-group :class:`GroupSignalSpec`\\ s from a grouped fit result.

    Reads the ``amplitude``/``relative_phase``/``N0`` nuisance values from each
    group's :class:`~asymmetry.core.fitting.engine.FitResult` in a
    :class:`~asymmetry.core.fitting.grouped_time_domain.GroupedTimeDomainFitResult`
    (duck-typed: any object exposing ``group_results``). Groups with no fitted
    amplitude fall back to a 0.2 default. The shared model values are *not*
    captured here — pass them as ``base_parameters`` to
    :func:`build_group_signals` / :func:`simulate_multi_group_run`.
    """
    specs: list[GroupSignalSpec] = []
    group_results = getattr(grouped_result, "group_results", {})
    for group_id, fit_result in group_results.items():
        values = {p.name: float(p.value) for p in getattr(fit_result, "parameters", [])}
        try:
            gid = int(group_id)
        except (TypeError, ValueError):
            continue
        specs.append(
            GroupSignalSpec(
                group_id=gid,
                amplitude=values.get("amplitude", 0.2),
                relative_phase=values.get("relative_phase", 0.0),
                n0_weight=values.get("N0", 1.0),
                label=str(group_id),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Degrade statistics (WiMDA DegradeStats)
# ---------------------------------------------------------------------------


def degrade_run(
    run: Run,
    factor: float,
    *,
    seed: int = 0,
    run_number: int | None = None,
) -> Run:
    """Resample a run's histograms to a different statistics level.

    For ``factor < 1`` each recorded count survives independently with
    probability ``factor`` (binomial thinning) — thinning a Poisson process
    is exactly Poisson, so the result is statistically indistinguishable
    from a run measured for ``factor`` times the beam time. For
    ``factor > 1`` no exact construction from recorded data exists; the
    WiMDA convention ``Poisson(counts · factor)`` is used, which is
    over-dispersed relative to a genuinely longer measurement (variance
    ``λf(1+f)`` rather than ``λf``).

    Returns a **new** :class:`Run` (the source run is untouched) with
    ``metadata["degraded"]`` provenance. A fixed seed reproduces the
    resampling bit-for-bit. ``grouping["good_frames"]`` (and the per-period
    frame counts) are scaled by ``factor`` — thinning by f *is* a measurement
    f times shorter, and an inherited deadtime correction only stays exact
    when counts and frames scale together. Combined two-period runs keep
    their period payload: each period's histograms are thinned with the same
    generator and the stored per-period reductions are recomputed, so period
    selection keeps working on the derived run.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Degrade factor must be positive and finite.")
    if not run.histograms:
        raise ValueError("Degrade statistics requires a run with detector histograms.")

    rng = np.random.default_rng(seed)

    def thin(hist: Histogram) -> Histogram:
        counts = np.clip(np.rint(np.asarray(hist.counts, dtype=float)), 0, None)
        if factor < 1.0:
            new_counts = rng.binomial(counts.astype(np.int64), factor)
        elif factor == 1.0:
            new_counts = counts
        else:
            new_counts = rng.poisson(counts * factor)
        return Histogram(
            counts=np.asarray(new_counts, dtype=float),
            bin_width=float(hist.bin_width),
            t0_bin=int(hist.t0_bin),
            good_bin_start=int(hist.good_bin_start),
            good_bin_end=int(hist.good_bin_end),
        )

    source_grouping = run.grouping if isinstance(run.grouping, dict) else {}
    period_lists = source_grouping.get("period_histograms")
    has_periods = (
        isinstance(period_lists, list)
        and len(period_lists) >= 2
        and all(isinstance(period, list) and period for period in period_lists)
    )

    # The period payload holds full histogram arrays; exclude it from the
    # deepcopy and rebuild it from the thinned periods below.
    grouping = copy.deepcopy({k: v for k, v in source_grouping.items() if k not in _PERIOD_KEYS})

    if has_periods:
        # run.histograms on a combined two-period run are clones of period 0
        # (the loader convention) — thin the periods, then mirror that.
        thinned_periods = [[thin(hist) for hist in period] for period in period_lists]
        histograms = [
            Histogram(
                counts=hist.counts.copy(),
                bin_width=hist.bin_width,
                t0_bin=hist.t0_bin,
                good_bin_start=hist.good_bin_start,
                good_bin_end=hist.good_bin_end,
            )
            for hist in thinned_periods[0]
        ]
        grouping["period_histograms"] = thinned_periods
        if "period_mode" in source_grouping:
            grouping["period_mode"] = source_grouping["period_mode"]
        period_good_frames = source_grouping.get("period_good_frames")
        if isinstance(period_good_frames, list):
            grouping["period_good_frames"] = [float(value) * factor for value in period_good_frames]
        if "period_dead_time_us" in source_grouping:
            grouping["period_dead_time_us"] = copy.deepcopy(source_grouping["period_dead_time_us"])
        # Recompute the stored per-period reductions (loader convention:
        # default reduction at α = 1) from the thinned histograms.
        grouping["period_reduced"] = [
            _reduce_histograms(period, {**grouping, "alpha": 1.0}) for period in thinned_periods
        ]
    else:
        histograms = [thin(hist) for hist in run.histograms]

    try:
        grouping["good_frames"] = float(grouping.get("good_frames", 1.0)) * factor
    except (TypeError, ValueError):
        pass

    number = run.run_number if run_number is None else int(run_number)
    metadata = dict(run.metadata)
    source_label = metadata.get("run_label") or str(run.run_number)
    metadata.pop("nexus_fields", None)
    metadata.pop("nexus_time_series", None)
    # The derived run was not loaded from the source's file; leaving the path
    # in place would make project save/reload and the file-overwrite prompt
    # treat it as the original run.
    metadata["source_file"] = ""
    metadata.update(
        {
            "run_number": number,
            "run_label": f"{source_label} ×{factor:g}",
            "title": f"{metadata.get('title', '')} (degraded ×{factor:g})".strip(),
            "degraded": {
                "factor": float(factor),
                "seed": int(seed),
                "source_run_number": run.run_number,
                "source_run_label": str(source_label),
            },
        }
    )

    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="",
    )


# ---------------------------------------------------------------------------
# Reduction to a browser-ready dataset
# ---------------------------------------------------------------------------


def _reduce_histograms(
    histograms: list[Histogram],
    grouping: dict,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Loader-convention F-B reduction of one histogram set (percent)."""
    fb = group_forward_backward(histograms, grouping)

    n = min(fb.forward.size, fb.backward.size)
    asymmetry, error = compute_asymmetry(fb.forward[:n], fb.backward[:n], fb.alpha)
    asymmetry = asymmetry * 100.0
    error = error * 100.0

    bin_width = float(histograms[0].bin_width) if histograms else 1.0
    return slice_to_good_window(
        asymmetry, error, grouping, common_t0=fb.common_t0, bin_width=bin_width
    )


def reduce_run_to_dataset(run: Run) -> MuonDataset:
    """Reduce a run to its F-B asymmetry :class:`MuonDataset` (percent).

    Mirrors the loader reduction (``NexusLoader``): align and sum the
    forward/backward groups, form the α-balanced asymmetry in percent, slice
    to the good-bin window and build the time axis from the common t0. Runs
    carrying a two-period payload are reduced according to their
    ``period_mode`` (red, green, or the green∓red combinations), and a
    ``bunching_factor`` above one is applied, so a derived run surfaces in
    the Data Browser looking exactly like its source. Used for synthetic and
    degraded runs.
    """
    if not run.histograms:
        raise ValueError("Reduction requires a run with detector histograms.")
    grouping = run.grouping if isinstance(run.grouping, dict) else {}

    period_lists = grouping.get("period_histograms")
    has_periods = (
        isinstance(period_lists, list)
        and len(period_lists) >= 2
        and all(isinstance(period, list) and period for period in period_lists)
    )
    mode = str(grouping.get("period_mode", PeriodMode.RED))
    if has_periods and mode in {
        str(PeriodMode.GREEN_MINUS_RED),
        str(PeriodMode.GREEN_PLUS_RED),
    }:
        red_hists, red_grouping = select_period_histograms(run.histograms, grouping, 0)
        green_hists, green_grouping = select_period_histograms(run.histograms, grouping, 1)
        red = _reduce_histograms(red_hists, red_grouping)
        green = _reduce_histograms(green_hists, green_grouping)
        time, asymmetry, error = combine_period_asymmetry(*red, *green, mode)
    elif has_periods:
        index = 1 if mode == str(PeriodMode.GREEN) else 0
        hists, period_grouping = select_period_histograms(run.histograms, grouping, index)
        time, asymmetry, error = _reduce_histograms(hists, period_grouping)
    else:
        time, asymmetry, error = _reduce_histograms(run.histograms, grouping)

    try:
        bunch_factor = max(1, int(grouping.get("bunching_factor", 1)))
    except (TypeError, ValueError):
        bunch_factor = 1
    if bunch_factor > 1 and asymmetry.size > 0:
        time, asymmetry, error = rebin(time, asymmetry, error, bunch_factor)

    metadata = dict(run.metadata)
    metadata.setdefault("run_number", run.run_number)
    metadata.setdefault("run_label", str(run.run_number))
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata=metadata,
        run=run,
    )


# ---------------------------------------------------------------------------
# Run statistics helpers (matched-statistics re-simulation, dialog defaults)
# ---------------------------------------------------------------------------


def total_events_of(run: Run) -> float:
    """Realised event total of a run: the sum of all histogram counts.

    The gross count (signal + background). Use it as the event-budget default
    for a re-simulation or a dialog spinner; for a *signal-only* budget that
    excludes a run's flat background, subtract
    ``estimate_background_per_bin(run) * total bins`` (see
    :func:`matched_statistics`).
    """
    return float(sum(float(h.counts.sum()) for h in run.histograms))


def estimate_background_per_bin(run: Run) -> float:
    """Estimate a run's flat background from its pre-t0 bins (counts/bin).

    The bins before t0 see no implanted-muon decay signal, so their mean count
    is the time-independent background level (zero for an ideal pulsed source).
    Returns the median across detectors for robustness, or ``0.0`` when no
    detector has a pre-t0 region.
    """
    levels: list[float] = []
    for hist in run.histograms:
        t0 = max(0, int(hist.t0_bin))
        if t0 > 0:
            pre = np.asarray(hist.counts[:t0], dtype=float)
            if pre.size:
                levels.append(float(pre.mean()))
    return float(np.median(levels)) if levels else 0.0


def matched_statistics(run: Run) -> tuple[float, float]:
    """Signal event budget and flat background for re-simulating a run.

    Returns ``(total_signal_events, background_per_bin)`` such that feeding them
    to :func:`simulate_run` reproduces the run's statistics: the flat background
    (estimated from the pre-t0 bins) is split off the gross count so it is *not*
    counted as decay signal. For an ideal pulsed run (no background) this is
    just the gross total and zero.
    """
    background_per_bin = estimate_background_per_bin(run)
    total_bins = sum(int(hist.n_bins) for hist in run.histograms)
    background_total = background_per_bin * total_bins
    signal_events = max(total_events_of(run) - background_total, 1.0)
    return signal_events, background_per_bin


# ---------------------------------------------------------------------------
# μ⁻ capture-lifetime run synthesis
# ---------------------------------------------------------------------------


def simulate_capture_run(
    template: Run,
    components: Sequence[CaptureComponent],
    weights: Mapping[str, float],
    *,
    total_events: float,
    group_id: int | None = None,
    seed: int = 0,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Synthesise a μ⁻ capture-lifetime run.

    Per detector in the signal group (or all detectors if ``group_id`` is
    ``None``):

        N_d(t) = Σ_i N_{i,d}·exp(−(t−t0)/τ_i) + b      (t ≥ t0)
        N_d(t) = b                                        (t < t0)

    with the component populations split by ``weights`` (relative, normalised
    over the components) and the per-bin envelope using the same exact
    telescoping normalisation as :func:`expected_counts`::

        n0_i = N_{i,d}·(1−exp(−Δt/τ_i))

    so the post-t0 window sum ≈ ``total_events``. Detectors in ``group_id``
    (or all detectors if ``None``) carry the signal; ``background_per_bin`` is
    added in addition to the event budget. Poisson-sampled with ``seed`` and
    assembled (provenance, deadtime-zeroing) via :func:`_sample_and_build_run`.
    Provenance records ``capture_mode=True``, the components/τ and weights,
    and the seed.

    Parameters
    ----------
    template
        Geometry template — bin structure, t0, grouping. Counts are ignored.
    components
        Ordered sequence of :class:`~asymmetry.core.negmu.model.CaptureComponent`
        objects (duck-typed: need ``.label`` and ``.tau_us``).
    weights
        Relative weight per component label. Normalised over the components
        present; missing labels default to zero weight.
    total_events
        Expected number of detected capture events in the post-t0 window,
        summed over all signal detectors.
    group_id
        Only detectors in this group carry the signal. If ``None``, all
        detectors are signal detectors.
    seed
        RNG seed for bit-for-bit reproducibility.
    background_per_bin
        Flat background counts per bin added to every detector (signal and
        non-signal alike).
    run_number, title
        Provenance metadata for the returned :class:`Run`.
    """
    if not np.isfinite(total_events) or total_events <= 0:
        raise ValueError("total_events must be a positive, finite event budget.")
    if background_per_bin < 0:
        raise ValueError("background_per_bin must be non-negative.")

    histograms = template.histograms
    n_det = len(histograms)
    if n_det == 0:
        raise ValueError("simulate_capture_run requires a template run with detector histograms.")

    # Normalise component weights
    labels = [str(c.label) for c in components]
    weight_vals = [max(0.0, float(weights.get(lbl, 0.0))) for lbl in labels]
    total_w = sum(weight_vals)
    if total_w <= 0:
        raise ValueError(
            "weights must have at least one positive entry for the components present."
        )
    weights_norm = [w / total_w for w in weight_vals]

    # Identify signal detectors (0-based indices)
    if group_id is not None:
        det_group = _detector_group_map(template.grouping, n_det)
        signal_dets = {det for det, gid in det_group.items() if gid == group_id}
        if not signal_dets:
            raise ValueError(f"simulate_capture_run: no detectors found for group_id={group_id!r}.")
    else:
        signal_dets = set(range(n_det))

    n_signal = len(signal_dets)

    # Build expected per-detector count arrays
    expected: list[NDArray[np.float64]] = []
    for det_idx, hist in enumerate(histograms):
        n_bins = hist.n_bins
        t0_bin = max(0, int(hist.t0_bin))
        bin_width = float(hist.bin_width)
        n_post = max(0, n_bins - t0_bin)

        clean = np.full(n_bins, float(background_per_bin), dtype=float)

        if det_idx in signal_dets and n_post > 0:
            t_post = np.arange(n_post, dtype=float) * bin_width
            n_det_total = total_events / n_signal
            envelope = np.zeros(n_post, dtype=float)
            for comp, w_norm in zip(components, weights_norm):
                tau = float(comp.tau_us)
                n_i_det = n_det_total * w_norm
                # Same telescoping normalisation as expected_counts
                n0_i = n_i_det * (1.0 - np.exp(-bin_width / tau))
                envelope += n0_i * np.exp(-t_post / tau)
            clean[t0_bin:] += np.clip(envelope, 0.0, None)

        expected.append(clean)

    sim_metadata: dict[str, Any] = {
        "capture_mode": True,
        "components": [{"label": str(c.label), "tau_us": float(c.tau_us)} for c in components],
        "weights": {str(c.label): float(weights.get(str(c.label), 0.0)) for c in components},
        "group_id": group_id,
    }

    return _sample_and_build_run(
        template,
        expected,
        seed=seed,
        total_events=total_events,
        background_per_bin=background_per_bin,
        run_number=run_number,
        title=title,
        default_title="Simulated μ⁻ capture run",
        simulation_metadata=sim_metadata,
    )
