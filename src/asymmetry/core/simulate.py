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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import (
    combine_period_asymmetry,
    select_period_histograms,
)
from asymmetry.core.transform.asymmetry import compute_asymmetry
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

    # Filter the period payload (which can hold full per-detector histogram
    # arrays for combined two-period templates) BEFORE deep-copying.
    grouping = copy.deepcopy({k: v for k, v in template.grouping.items() if k not in _PERIOD_KEYS})
    grouping["deadtime_correction"] = False
    grouping["dead_time_us"] = [0.0] * len(histograms)

    number = template.run_number if run_number is None else int(run_number)
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
        key: template.metadata[key]
        for key in (
            "instrument",
            "temperature",
            "field",
            "field_state",
            "field_direction",
            "detector_orientation",
        )
        if key in template.metadata
    }
    metadata.update(
        {
            "run_number": number,
            "run_label": f"SIM {number}",
            "title": title if title is not None else "Simulated run",
            "synthetic": True,
            "simulation": provenance,
        }
    )

    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="",
    )


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

    det_groups = _detector_group_map(grouping, len(template.histograms))
    group_ids = sorted(set(det_groups.values()))
    # The α split fixes the GROUP TOTALS at F:B = α; each side's budget share
    # is divided by its detector count so unequal group sizes cannot skew the
    # ratio (a per-detector weight alone would give F/B = α·n_F/n_B).
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

    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", asymmetry.size - 1))
    except (TypeError, ValueError):
        last_good = asymmetry.size - 1
    last_good = min(last_good, asymmetry.size - 1)
    if first_good > last_good:
        first_good, last_good = 0, asymmetry.size - 1

    asymmetry = asymmetry[first_good : last_good + 1]
    error = error[first_good : last_good + 1]
    bin_width = float(histograms[0].bin_width) if histograms else 1.0
    time = (np.arange(asymmetry.size, dtype=float) + first_good - fb.common_t0) * bin_width
    return time, asymmetry, error


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
