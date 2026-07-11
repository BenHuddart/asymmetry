# NOTES — A spin-Peierls transition (Magnetism)

Module: `spin_peierls.py`. Example: `Magnetism/A spin-Peierls transition`.
Spec: that example's `GROUND_TRUTH.md` (KTCNQF₄, quasi-1D S=½; guide
"Fluctuations in a spin-Peierls compound"). No reference paper in-folder; no
numeric fit targets in the guide — the ground truth is the **qualitative
static↔dynamic identification at ≈150 K** (GT §6/§9).

## Scenarios registered

| Scenario | Render | Intended docs use |
|---|---|---|
| `corpus_spinpeierls_zf_spectra` | GUI MainWindow overlay of five ZF spectra 185→60 K (bunch 6, 0–10 µs). | **Core teaching image** — the ZF line-shape changes from a slow static-Gaussian decay (185/145/130 K) to a fast dynamic decay (90/60 K). Answers guide Q1 ("difference between the shape of the spectra at high and low T", GT §5). |
| `corpus_spinpeierls_model_discrimination` | Standalone matplotlib, 2 columns (185 K \| 90 K), each = data + static Gaussian KT + dynamic exponential over a σ-residual panel, χ²ᵣ annotated. | Answers guide Q2/Q3 ("different ways to fit the data" / "physical significance", GT §5). Shows the model that describes the data **flips** across the transition. |
| `corpus_spinpeierls_beta_t` | GUI FitParametersPanel: stretched exponent β(T), ref lines β=2/β=1, T_SP≈150 K marker. | The line-shape crossover as a single number: β≈2 (static) → ≈0.6 (dynamic). |
| `corpus_spinpeierls_lambda_t` | GUI FitParametersPanel: relaxation rate λ(T), T_SP≈150 K marker. | The dynamic relaxation **rate peak** (~0.94 µs⁻¹ at 60 K) as fluctuations enter the µSR window. |

Requires-fit: all but `zf_spectra` (which is a plain overlay render).

## Run selection & workflow (GT §2–4)

- **Data map** taken from the GT §2 file-header audit (2026-07-11), confirmed
  live from the on-disk `.nxs` headers: sample `KTCNQF4`, instrument `EMU`.
- **ZF (0 G) series used** (headers agree with the guide's table for 130–185 K,
  GT §3): 29944(185) 29943(180) 29941(175) 29940(170) 29939(165) 29938(160)
  29937(155) 29936(150) 29934(145) 29933(140) 29932(135) 29931(130), plus the
  three lower-T ZF runs 29924(90) 29923(60) 29921(30). Run 29920 is 300 K ZF
  (kept out of the trends — a large 185→300 K gap — but part of the static
  high-T limit). TF calibration 29919 (300 K / 100 G) not needed for the ZF
  story.
- **Model (GT §4 — deliberately open-ended; models left to the analyst):**
  candidate static = `StaticGKT_ZF` + `Constant` (nuclear-dipolar Gaussian KT);
  candidate dynamic = `Exponential` + `Constant` (discrimination fig) and
  `StretchedExponential` + `Constant` (continuous crossover via β).
- **Fit window** 0.1–12 µs (skip pulse/deadtime; stop before the ZF F−B error
  fan). No α / t0 / good-bin / deadtime values are supplied by the guide (GT §9)
  — the loader defaults are used.
- **Amplitude protocol** (wave-1 lesson: fix the amplitude for real
  stretched-exponential series): the t=0 asymmetry A_1 is calibrated once from
  the highest-T ZF run (185 K, β & A_1 free → **A_1 = 17.9**) and then fixed for
  the whole β(T)/λ(T) batch; A_bg floats per run. Left free, A_1 runs away to
  60–120 at 60–90 K (β/λ/A_1 degeneracy).
- **Warm start** descending in temperature (wave-1 lesson): each ZF fit seeds
  from the previous higher-T minimum. Cold seeds walk to wrong β/λ minima.

## Fitted values (stretched exponential, A_1 fixed = 17.9)

| T (K) | β | λ (µs⁻¹) | regime |
|---|---|---|---|
| 185 | 1.96 | 0.167 | static (Gaussian) |
| 175–150 | ~2.0 | ~0.167 | static (Gaussian) |
| 145 | 1.88 | 0.168 | line-shape starts changing |
| 140 | 1.88 | 0.170 | " |
| 135 | 1.76 | 0.174 | " |
| 130 | 1.71 | 0.178 | " |
| 90 | 0.67 | 0.563 | dynamic |
| 60 | 0.58 | 0.938 | dynamic (λ peak) |
| 30 | 1.08 | 0.214 | static recovers (singlet gap) |

Discrimination fig χ²ᵣ (independent per-model fits, A_1 free):
185 K → static KT **1.12**, exponential 2.03 (static wins);
90 K → static KT **8.03**, exponential 2.04 (dynamic wins).

There is **no ground-truth number to compare against** (GT §6/§9): the guide
supplies only the qualitative result. These β/λ magnitudes reproduce the corpus
working note (`ANALYSIS_asymmetry.md`, excluded from ground truth): "high T
β ≈ 2 (Gaussian / static nuclear dipolar); cooling β drops and λ develops a
feature (spike near 60 K)." Confirmed.

## Transition location vs 150 / 160 K

- Guide value **T_SP ≈ 150 K** (GT §6); literature **T_SP ≈ 160 K** (Berlie
  *et al.*, PRB **93**, 054422 (2016), GT §9). Both marked/discussed.
- In the on-disk ZF data the **line-shape change onsets on cooling through
  ~145–130 K** (β departs from 2 at 145 K; 1.71 by 130 K) — consistent with the
  electronic fluctuations coming into the µSR window as the ≈150 K transition is
  approached. This is the honest signature: onset ~145 K, i.e. right at the
  guide's ≈150 K and a little below the literature 160 K.
- The **λ peak sits at ~60 K**, well below the transition — the fluctuations are
  slowest (deepest in the µSR window) below T_SP, not at it. This is *not* a
  critical-divergence-at-T_c signature; it is a fluctuation peak in the
  paramagnetic-into-singlet crossover.

## Static-vs-dynamic conclusion (the deliverable, GT §6)

Correctly identified either side of ≈150 K:
- **Above (≳150 K): static.** Static Gaussian Kubo–Toyabe fits (χ²ᵣ≈1.1), β≈2,
  weak λ. The fast paramagnetic S=½ fluctuations are motionally averaged, so the
  muon senses only the static nuclear-dipolar field.
- **Cooling toward/below (~130→60 K): dynamic.** Static KT fails badly (χ²ᵣ≈8 at
  90 K — it predicts a ⅓-recovery plateau the data don't show); β→0.5–0.6 and λ
  rises to a peak — the electronic correlation time has entered the µSR window.
- **Deep singlet phase (~30 K): static recovers** (β up to ~1.1, λ drops) as the
  moments gap into singlets. A nice bonus the on-disk low-T ZF runs reveal.

Net: a **dynamic-fluctuation regime sandwiched between two static limits**
(high-T motional-narrowing and low-T singlet-gap), with the crossover onset at
the ≈150 K transition.

## Feature-demonstration opportunities

- **Overlay / line-shape comparison** (`zf_spectra`) — the multi-run overlay
  with per-run bunching reads well for a shape-vs-T story.
- **Model-comparison with residuals + χ²** (`model_discrimination`) — done as a
  standalone matplotlib figure (mgb2 pattern) because the GUI single-fit panel
  shows one model at a time; a side-by-side "which model, which side" panel has
  no native GUI view. Good candidate if a two-model overlay view is ever added.
- **Parameter-trending panel** (`beta_t`, `lambda_t`) — β(T) and λ(T) via
  `FitParametersPanel` with in-`settle` reference lines / transition markers.

## Problems hit (honest)

1. **The 100 G "110–125 K ZF" block does NOT extend the story.** The guide's
   low-T ZF rows (29945–29951) are absent on disk; the only 110–125 K data on
   disk are 29926–29930, which carry a **100 G transverse** field (GT §2). Fit
   with any ZF relaxation model they give χ²ᵣ ≈ 300–440 (the signal precesses at
   ~1.36 MHz). So they cannot bridge the ZF gap honestly and are **excluded**.
   Consequence: the ZF trends have an **unavoidable 90 K → 130 K gap**, right
   where the crossover happens; the crossover is bracketed, not densely traced.
2. **Stretched-exponential A_1 degeneracy** at the fast low-T runs — fixed by the
   A_1 calibration/fix protocol above (wave-1 lesson applied).
3. **λ is a single-model rate across a changing β**, so its absolute meaning
   drifts (Gaussian-width-like at high β, exponential-rate-like at low β). The
   *rise and peak* are the robust, reportable signature; magnitudes are
   model-dependent (as the working note also notes). Stated in the λ(T) prose.
4. **The brief's physics sketch has the T-direction inverted** relative to this
   dataset. The sketch says "above = dynamic, below = static (singlets)"; the
   data (and the working note, and GT §9's anticipated β≈2-high-T / β≈0.2-low-T
   finding) show **above = static Gaussian, dynamic emerging on cooling toward
   the transition, static again deep in the singlet phase**. Followed the data
   and the direction-agnostic GT §6 deliverable ("static- vs dynamic-relaxation
   either side of the ≈150 K transition").
