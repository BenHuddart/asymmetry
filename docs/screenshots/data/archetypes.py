"""Deterministic synthetic μSR datasets for the documentation screenshots.

Each generator returns one or more :class:`~asymmetry.core.data.MuonDataset`
instances (or, for parameter-domain models, plain numpy arrays). The data are
grounded in real materials and parameters drawn from Blundell, De Renzi,
Lancaster, Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022) and the
classic μSR literature, but the numerical values are rounded for clean,
deterministic documentation. All randomness flows through an explicit
:class:`numpy.random.Generator` seeded per scenario to keep PNG outputs
byte-stable across CI runs.

Material legend
---------------

==========  =================================  ==========================
Material    Used by                            Textbook reference
==========  =================================  ==========================
**EuO**     gui_usage, composite_models,        Ch 6, Fig 6.6 (Tc=69 K
            logbook                             ferromagnet)
**Ag**      fitting, fit_wizard, lf_kubo,       Ch 5.2 (canonical nuclear
            global_fit_wizard                   dipolar host, Δ≈0.4 μs⁻¹)
**MgB₂**    parameter_trending,                 Ch 8 (two-gap SC,
            sc_penetration_depth                Tc=36 K)
**YBCO**    grouped_time_domain_fitting,        Ch 8/9 (Knight shift &
            fourier_analysis                    vortex lattice, Tc=90 K)
**PbF₂**    muon_fluorine                       Ch 4.6 (F-μ-F entanglement)
==========  =================================  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.models import MODELS

MUON_LIFETIME_US = 2.197

# Physical constants and material parameters --------------------------------

GAMMA_MU_MHZ_PER_G = 0.01355   # γ_μ / (2π) in MHz/G ≈ 13.55 kHz/G
GAMMA_MU_MHZ_PER_T = 135.5     # γ_μ / (2π) in MHz/T

TC_EUO_K = 69.0                # EuO Curie temperature
DELTA_AG_PER_US = 0.39         # Ag nuclear dipolar Δ (μs⁻¹), Ch 5.2
R_MUF_ANG = 1.17               # F-μ-F equilibrium distance (Å)
TC_MGB2_K = 36.0               # MgB₂ critical temperature
TC_YBCO_K = 90.0               # YBa₂Cu₃O₇₋δ critical temperature
LAMBDA_YBCO_NM = 130.0         # YBCO ab-plane penetration depth
XI_YBCO_NM = 2.0               # YBCO coherence length

_DEFAULT_NPOINTS = 480
_DEFAULT_TMAX_US = 8.0


def _time_axis(n_points: int = _DEFAULT_NPOINTS, t_max: float = _DEFAULT_TMAX_US):
    return np.linspace(0.0, t_max, n_points)


def _poisson_errors(asymmetry: np.ndarray, counts_per_bin: float = 5e4) -> np.ndarray:
    """Per-bin uncertainty derived from a target counts-per-bin.

    The asymmetry σ for a two-detector experiment scales as
    ``sqrt((1 - A^2) / N)``. Picking a counts-per-bin gives a realistic noise
    envelope without doing a histogram round-trip.
    """
    variance = np.clip(1.0 - (asymmetry / 100.0) ** 2, 1e-3, 1.0) / counts_per_bin
    return np.sqrt(variance) * 100.0


def _build_run_with_detector_asymmetries(
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
    rng: np.random.Generator | None = None,
) -> tuple[Run, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a full :class:`Run` from per-detector asymmetry signals.

    Each detector histogram is built as

        N_d(t) = N_{0,d} · exp(-t/τ_μ) · (1 + A_d(t))   for t > 0
              = N_{0,d}                                  for t ≤ 0

    with Poisson counting noise applied. The grouping payload puts one
    detector per group so the GUI's *Individual Groups* domain view shows a
    trace per detector. The function also returns the F–B asymmetry trace
    (using the first two groups) and a per-bin error estimate so the wrapper
    :class:`MuonDataset` carries a defensible time-domain view.

    Parameters
    ----------
    detector_asymmetries
        One dict per detector. Each dict must define ``"asymmetry"`` (a
        ``(n_bins - t0_bin,)`` array giving A_d(t)) plus optional
        ``"label"``, ``"amplitude_scale"``, and ``"n0"``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    bins = np.arange(n_bins)
    time_full = (bins - t0_bin) * bin_width_us
    decay = np.exp(-np.maximum(time_full, 0.0) / MUON_LIFETIME_US)

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
    groups = {gid: [gid - 1] for gid in range(1, n_groups + 1)}
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
    error_post = np.where(denom > 0, np.sqrt(2.0 / denom), 0.0) * 100.0
    return run, time_post, raw_asym, error_post


# ---------------------------------------------------------------------------
# Static-field thread: Ag polycrystal (nuclear dipolar Kubo–Toyabe)
# ---------------------------------------------------------------------------

def make_ag_zf_gkt(seed: int = 23) -> MuonDataset:
    """ZF Ag polycrystal — static Gaussian Kubo–Toyabe (Δ=0.39 μs⁻¹).

    Canonical nuclear-dipolar reference sample at PSI/ISIS/TRIUMF — every
    facility runs Ag as part of routine calibrations. Maps to textbook eq 5.13
    and Fig 5.5.
    """
    rng = np.random.default_rng(seed)
    gkt = MODELS["StaticGKT_ZF"].function
    time = _time_axis()
    clean = gkt(time, A0=24.0, Delta=DELTA_AG_PER_US, baseline=0.3)
    error = _poisson_errors(clean, counts_per_bin=1.2e5)
    asymmetry = clean + rng.normal(0.0, error)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": 4101,
            "title": "ZF Ag polycrystal 20K",
            "temperature": 20.0,
            "field": 0.0,
        },
    )


def make_ag_lf_decoupling(
    seed: int = 41, fields_g: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)
) -> list[MuonDataset]:
    """LF Kubo–Toyabe decoupling series on Ag with shared Δ=0.39 μs⁻¹.

    Spans the textbook units γ_μB_L/Δ ∈ {0, 1, 2, 5, 10} (cf. Fig 5.6 of
    Blundell et al. and the original Hayano PRB 20, 850 (1979) curve). The
    default 5-field sweep is used for the time-domain overlay; pass a 4-field
    subset (e.g. (0, 15, 50, 100)) for global-fit scenarios.
    """
    rng = np.random.default_rng(seed)
    lfkt = MODELS["LFKuboToyabe"].function
    time = _time_axis()
    datasets: list[MuonDataset] = []
    for index, field_g in enumerate(fields_g):
        clean = lfkt(
            time, A0=24.0, Delta=DELTA_AG_PER_US, B_L=float(field_g), baseline=0.3
        )
        error = _poisson_errors(clean, counts_per_bin=1.0e5)
        asymmetry = clean + rng.normal(0.0, error)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=asymmetry,
                error=error,
                metadata={
                    "run_number": 5201 + index,
                    "title": f"LF Ag {field_g:g} G",
                    "temperature": 20.0,
                    "field": float(field_g),
                },
            )
        )
    return datasets


# ---------------------------------------------------------------------------
# Magnetism thread: EuO ferromagnet through Tc (Ch 6, Fig 6.6)
# ---------------------------------------------------------------------------

def make_euo_tf_tscan(seed: int = 17) -> list[MuonDataset]:
    """EuO ZF temperature scan crossing the Curie point Tc=69 K.

    Below Tc the spontaneous local field at the muon site drives precession
    whose frequency tracks the magnetic order parameter, ν(T) ∝ (1−T/Tc)^β
    with β ≈ 0.4 (Blundell PRB 81, 092407, 2010 — textbook Fig 6.6). Above
    Tc the signal is paramagnetic exponential relaxation with damping that
    peaks in the critical region due to fluctuations of the staggered
    magnetization.
    """
    rng = np.random.default_rng(seed)
    osc = MODELS["Oscillatory"].function
    exp_fn = MODELS["ExponentialRelaxation"].function
    time = _time_axis(n_points=600, t_max=6.0)

    nu_0_mhz = 28.0     # ν(T=0) extrapolated frequency (textbook Fig 6.6)
    beta = 0.40
    damping_floor = 0.10
    lambda_peak = 4.0   # critical damping near Tc
    delta_t_k = 6.0     # critical-region width

    temps_k = [30.0, 50.0, 65.0, 69.0, 73.0, 90.0]
    datasets: list[MuonDataset] = []
    for index, t in enumerate(temps_k):
        if t < TC_EUO_K - 0.5:
            order = (1.0 - t / TC_EUO_K) ** beta
            frequency_mhz = nu_0_mhz * order
        else:
            frequency_mhz = 0.0
        damping = damping_floor + lambda_peak * np.exp(
            -((t - TC_EUO_K) / delta_t_k) ** 2
        )

        if frequency_mhz > 0.01:
            clean = osc(
                time,
                A0=22.0,
                frequency=frequency_mhz,
                phase=0.0,
                Lambda=damping,
                baseline=0.4,
            )
        else:
            clean = exp_fn(time, A0=22.0, Lambda=damping, baseline=0.4)

        error = _poisson_errors(clean, counts_per_bin=1.0e5)
        asymmetry = clean + rng.normal(0.0, error)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=asymmetry,
                error=error,
                metadata={
                    "run_number": 3001 + index,
                    "title": f"EuO ZF {t:g}K",
                    "temperature": t,
                    "field": 0.0,
                },
            )
        )
    return datasets


def make_euo_composite(seed: int = 71) -> MuonDataset:
    """EuO at T=70 K (just above Tc) — damped Larmor + slow exponential tail.

    Sits inside the critical region where the system shows both a damped
    oscillating component from residual short-range correlations and a slow
    exponential background. Useful for illustrating the composite-model
    fraction-group machinery.
    """
    rng = np.random.default_rng(seed)
    osc = MODELS["Oscillatory"].function
    exp_fn = MODELS["ExponentialRelaxation"].function
    time = _time_axis(n_points=600, t_max=8.0)

    osc_part = osc(time, A0=14.0, frequency=2.5, phase=0.0, Lambda=1.5, baseline=0.0)
    exp_part = exp_fn(time, A0=6.0, Lambda=0.3, baseline=0.0)
    clean = osc_part + exp_part + 0.4
    error = _poisson_errors(clean, counts_per_bin=1.0e5)
    asymmetry = clean + rng.normal(0.0, error)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": 3501,
            "title": "EuO ZF 70K (critical)",
            "temperature": 70.0,
            "field": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Superconductor thread: MgB₂ σ(T) two-gap (already in examples/)
# ---------------------------------------------------------------------------

def make_mgb2_sigma_t(seed: int = 105, n_points: int = 28) -> dict:
    """MgB₂ σ(T) data set produced by the SC_TwoGap_SS evaluator.

    Returns a dict with arrays ``T_K``, ``sigma``, and ``sigma_err`` suitable
    for the parameter-trending workflow (Niedermayer et al. PRB 65, 094512
    (2002); Sonier RMP 72, 769 (2000)). Parameters chosen to match the
    canonical MgB₂ alpha-model decomposition (small/large gap ratios
    1.1 / 2.3, weight ≈ 0.55).
    """
    from asymmetry.core.fitting.sc.models import sc_two_gap_ss

    rng = np.random.default_rng(seed)
    temperatures_k = np.linspace(1.5, TC_MGB2_K - 1.0, n_points)
    sigma_clean = sc_two_gap_ss(
        temperatures_k,
        sigma_0=1.25,
        Tc=TC_MGB2_K,
        gap_ratio_1=1.1,
        gap_ratio_2=2.3,
        weight=0.55,
        sigma_bg=0.03,
    )
    sigma_err = np.full_like(temperatures_k, 0.015)
    sigma = sigma_clean + rng.normal(0.0, sigma_err)
    return {
        "T_K": temperatures_k,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "Tc_K": TC_MGB2_K,
    }


# ---------------------------------------------------------------------------
# Cuprate thread: YBa₂Cu₃O₇₋δ above and below Tc
# ---------------------------------------------------------------------------

def make_ybco_knight_grouped(seed: int = 101) -> MuonDataset:
    """Normal-state YBa₂Cu₃O₇₋δ TF run with 4 detector groups for grouped fitting.

    T = 100 K (just above Tc=90 K), TF = 200 G. Returns a single
    :class:`MuonDataset` whose underlying :class:`Run` carries four detector
    histograms with their own amplitudes and relative phases (mimicking a
    4-detector ring geometry) and a grouping payload that puts one detector
    per group. The GUI's *Individual Groups* domain therefore shows four
    per-group traces, and the multi-group time-domain fit window engages
    automatically: per-group amplitudes, baselines, N₀, and relative phases
    fit locally while the Larmor frequency and damping are shared (Sonier
    RMP 72, 769, 2000).

    The TF of 200 G (rather than the literature-typical 600 mT) is chosen so
    that the time-domain plot shows a few cycles cleanly; the docs caption
    notes the trade-off.
    """
    rng = np.random.default_rng(seed)
    bin_width_us = 0.005
    n_bins = 2400
    t0_bin = 100

    # Per-detector asymmetry signal: a Knight-shifted Larmor precession with
    # detector-specific amplitude and phase. The +0.005 Knight shift is baked
    # into the frequency itself (rather than fit as a separate parameter) so
    # the fit machinery exercises the shared-frequency code path naturally.
    frequency_mhz = GAMMA_MU_MHZ_PER_G * 200.0 * (1.0 + 0.005)
    damping = 0.08
    time_post = (np.arange(n_bins - t0_bin)) * bin_width_us

    detectors = []
    for index, params in enumerate(
        [
            {"label": "Det 1 (0°)", "amplitude": 0.20, "phase": 0.0},
            {"label": "Det 2 (90°)", "amplitude": 0.21, "phase": np.pi / 2.0},
            {"label": "Det 3 (180°)", "amplitude": 0.19, "phase": np.pi},
            {"label": "Det 4 (270°)", "amplitude": 0.20, "phase": 3 * np.pi / 2.0},
        ]
    ):
        asym = (
            params["amplitude"]
            * np.cos(2 * np.pi * frequency_mhz * time_post + params["phase"])
            * np.exp(-damping * time_post)
        )
        detectors.append(
            {
                "label": params["label"],
                "asymmetry": asym,
                "n0": 1.0e6 + index * 5e4,
            }
        )

    run, time_axis, raw_asym, raw_err = _build_run_with_detector_asymmetries(
        run_number=7101,
        detector_asymmetries=detectors,
        title="YBCO TF 200G 100K (Knight shift)",
        temperature_k=100.0,
        field_g=200.0,
        bin_width_us=bin_width_us,
        n_bins=n_bins,
        t0_bin=t0_bin,
        rng=rng,
    )
    return MuonDataset(
        time=time_axis,
        asymmetry=raw_asym,
        error=raw_err,
        metadata={
            "run_number": run.run_number,
            "title": run.metadata["title"],
            "temperature": 100.0,
            "field": 200.0,
        },
        run=run,
    )


def make_ybco_vortex_lattice(seed: int = 103) -> MuonDataset:
    """YBa₂Cu₃O₇₋δ in the vortex state — TF μSR with asymmetric P(B).

    T = 10 K (below Tc=90 K), TF = 200 mT (mixed state). The synthesised
    time-domain polarization is the inverse-cosine-transform of a target
    asymmetric vortex-lattice field distribution P(B − B_app) with a sharp
    low-field peak at the saddle-point van Hove singularity and a long
    high-field tail toward the vortex cores (Brandt PRB 37, 2349, 1988;
    Sonier RMP 72, 769, 2000).

    Returns a single :class:`MuonDataset` with a full :class:`Run` carrying
    two detector histograms (forward / backward of an F–B pair). The
    grouping payload exposes the F and B groups so the GUI's *Compute FFT*
    path can build the grouped Fourier spectrum and the *Frequency* domain
    plot shows the canonical asymmetric P(B) line shape.
    """
    rng = np.random.default_rng(seed)
    bin_width_us = 0.005
    n_bins = 3200
    t0_bin = 100
    time_post = (np.arange(n_bins - t0_bin)) * bin_width_us

    # Synthesise the *demodulated* asymmetry envelope from the target P(B−B_app).
    # In the lab frame the muon precesses at γ_μ·B_app, which would be too fast
    # to resolve in the time domain at this bin width. We synthesise the lab-
    # frame signal here so the GUI's FFT pipeline sees the full precession and
    # renders it correctly in the frequency domain.
    b_app_g = 2000.0      # 200 mT in gauss
    sigma_vl_per_us = 0.62
    # Asymmetric P(ΔB) construction (skewed Gaussian + exponential tail).
    delta_b = np.linspace(-1.5, 4.5, 4096) * sigma_vl_per_us
    peak = -0.3 * sigma_vl_per_us
    p_b = np.zeros_like(delta_b)
    rise = delta_b >= peak - 0.6 * sigma_vl_per_us
    tail = delta_b >= peak
    p_b[rise] = np.exp(-((delta_b[rise] - peak) / (0.18 * sigma_vl_per_us)) ** 2)
    p_b[tail] += 0.55 * np.exp(-(delta_b[tail] - peak) / (1.1 * sigma_vl_per_us))
    p_b = np.clip(p_b, 0.0, None)
    kernel = np.exp(-np.linspace(-3, 3, 61) ** 2)
    kernel /= kernel.sum()
    p_b = np.convolve(p_b, kernel, mode="same")
    p_b /= np.trapezoid(p_b, delta_b)

    omega_app_per_us = 2.0 * np.pi * GAMMA_MU_MHZ_PER_G * b_app_g
    omega_delta = 2.0 * np.pi * delta_b
    amplitude = 0.20
    asym_envelope = np.trapezoid(
        p_b[None, :] * np.cos((omega_app_per_us + omega_delta[None, :]) * time_post[:, None]),
        delta_b,
        axis=1,
    )
    asym_envelope = asym_envelope * np.exp(-0.04 * time_post)
    asym_fwd = +amplitude * asym_envelope
    asym_bwd = -amplitude * asym_envelope  # opposite-sign asymmetry for the F-B pair

    detectors = [
        {"label": "Forward", "asymmetry": asym_fwd, "n0": 1.2e6},
        {"label": "Backward", "asymmetry": asym_bwd, "n0": 1.2e6},
    ]

    run, time_axis, raw_asym, raw_err = _build_run_with_detector_asymmetries(
        run_number=7301,
        detector_asymmetries=detectors,
        title="YBCO TF 200mT 10K (vortex state)",
        temperature_k=10.0,
        field_g=b_app_g,
        bin_width_us=bin_width_us,
        n_bins=n_bins,
        t0_bin=t0_bin,
        rng=rng,
    )
    return MuonDataset(
        time=time_axis,
        asymmetry=raw_asym,
        error=raw_err,
        metadata={
            "run_number": run.run_number,
            "title": run.metadata["title"],
            "temperature": 10.0,
            "field": b_app_g,
        },
        run=run,
    )


# ---------------------------------------------------------------------------
# Muon-fluorine thread: PbF₂ F-μ-F entanglement
# ---------------------------------------------------------------------------

def make_pbf2_fmuf(seed: int = 89) -> MuonDataset:
    """PbF₂ F-μ-F entanglement signal at low T with r_μF=1.17 Å.

    PbF₂ provides a clean F-μ-F demonstration: the heavy Pb host carries no
    significant nuclear moment, so the polarization is dominated by the
    analytical F-μ-F dipolar pattern (Brewer et al. PRB 33, 7813, 1986;
    textbook Ch 4.6). t_max = 20 μs is required to resolve the slow beat
    envelope.
    """
    rng = np.random.default_rng(seed)
    fmuf_model = CompositeModel(["FmuF_Linear", "Constant"], operators=["+"])
    time = _time_axis(n_points=1200, t_max=20.0)

    # CompositeModel exposes a single `.function` callable that takes the
    # mangled per-component parameter names. For a flat [FmuF_Linear, Constant]
    # composite the names are A_1 (the FmuF amplitude), r_muF, and A_bg.
    clean = fmuf_model.function(time, A_1=22.0, r_muF=R_MUF_ANG, A_bg=0.2)
    error = _poisson_errors(clean, counts_per_bin=1.0e5)
    asymmetry = clean + rng.normal(0.0, error)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": 8101,
            "title": "PbF₂ ZF 5K (F-μ-F)",
            "temperature": 5.0,
            "field": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Vector polarization thread: EMU 3-axis projections
# ---------------------------------------------------------------------------

def make_emu_vector(seed: int = 97) -> list[MuonDataset]:
    """Three EMU-style polarization projections P_x, P_y, P_z.

    Generic display dataset for the vector-polarization page — not tied to a
    specific real material since the focus is the detector-grouping geometry
    rather than the underlying physics. Z component carries a slow
    exponential decay; X has a weak transverse oscillation; Y is near-zero
    with statistical noise.
    """
    rng = np.random.default_rng(seed)
    osc = MODELS["Oscillatory"].function
    exp_fn = MODELS["ExponentialRelaxation"].function
    time = _time_axis(n_points=480, t_max=8.0)

    p_z_clean = exp_fn(time, A0=18.0, Lambda=0.25, baseline=0.3)
    p_x_clean = osc(
        time, A0=8.0, frequency=0.6, phase=0.0, Lambda=0.4, baseline=0.0
    )
    p_y_clean = np.full_like(time, 0.3)

    datasets: list[MuonDataset] = []
    for index, (axis, clean) in enumerate(
        zip(("Pz", "Px", "Py"), (p_z_clean, p_x_clean, p_y_clean), strict=True)
    ):
        error = _poisson_errors(clean, counts_per_bin=6e4)
        asymmetry = clean + rng.normal(0.0, error)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=asymmetry,
                error=error,
                metadata={
                    "run_number": 9001 + index,
                    "title": f"EMU vector ZF — {axis}",
                    "temperature": 25.0,
                    "field": 0.0,
                    "polarization_axis": axis,
                },
            )
        )
    return datasets


# ---------------------------------------------------------------------------
# Data processing thread: low-statistics TF for rebin demo
# ---------------------------------------------------------------------------

def make_generic_tf_for_processing(seed: int = 107) -> MuonDataset:
    """Low-statistics 100 G TF dataset for the data-processing rebin demo.

    counts_per_bin is intentionally low so that rebinning (×8) produces a
    visibly cleaner trace — the standard pedagogical demonstration of
    statistics scaling with bin width.
    """
    rng = np.random.default_rng(seed)
    osc = MODELS["Oscillatory"].function
    time = _time_axis(n_points=480, t_max=8.0)
    clean = osc(
        time,
        A0=22.0,
        frequency=GAMMA_MU_MHZ_PER_G * 100.0,
        phase=0.0,
        Lambda=0.1,
        baseline=0.4,
    )
    error = _poisson_errors(clean, counts_per_bin=1.5e4)
    asymmetry = clean + rng.normal(0.0, error)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": 9501,
            "title": "TF 100G low-statistics demo",
            "temperature": 20.0,
            "field": 100.0,
        },
    )


__all__ = [
    "DELTA_AG_PER_US",
    "GAMMA_MU_MHZ_PER_G",
    "GAMMA_MU_MHZ_PER_T",
    "LAMBDA_YBCO_NM",
    "R_MUF_ANG",
    "TC_EUO_K",
    "TC_MGB2_K",
    "TC_YBCO_K",
    "XI_YBCO_NM",
    "make_ag_lf_decoupling",
    "make_ag_zf_gkt",
    "make_emu_vector",
    "make_euo_composite",
    "make_euo_tf_tscan",
    "make_generic_tf_for_processing",
    "make_mgb2_sigma_t",
    "make_pbf2_fmuf",
    "make_ybco_knight_grouped",
    "make_ybco_vortex_lattice",
]
