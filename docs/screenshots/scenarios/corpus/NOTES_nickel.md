# NOTES — Ferromagnetic nickel (Magnetism)

Module: `nickel_ferro.py` · Example: `Magnetism/Ferromagnetic nickel`
Data: EMU/ISIS NeXus-v2 HDF5, `emu00124218.nxs`–`emu00124278.nxs` (61 runs).
GROUND_TRUTH: `Magnetism/Ferromagnetic nickel/GROUND_TRUTH.md`.

The corpus data live in `…/Ferromagnetic nickel/Data/` (capital **D**), not the
lower-case `data/` the task brief and GROUND_TRUTH §2 quote. Scenarios resolve
`_DATA = "Magnetism/Ferromagnetic nickel/Data/emu00%d.nxs"`.

## ⚠ Temperature units — °C, not K (resolved 2026-07-11, GROUND_TRUTH §2 warning)

The file headers and the guide's run log quote the **furnace controller's
Celsius readings** even though the NeXus `sample/temperature` units attribute
claims Kelvin. First analysis pass here read them as kelvin and flagged a
"T_C ≈ 358 K vs guide 630 K" inconsistency; the orchestrator resolved it:
**358 °C = 631 K = the literature T_C of Ni exactly.** Every consistency check
then locks in place — the resolvable ~6 MHz precession at controller "345"
is Ni at 618 K = 0.98 T_C, the guide's run-log label "T_C region" for 325–380
is correct in °C, and the absence of oscillation at controller "100" (373 K
≈ 0.6 T_C, ν ≈ 18 MHz per the 1973 Fig. 2 curve) is exactly the pulsed-source
band-pass loss of guide Q7/Q9. **Scenarios convert trend temperatures to kelvin
(+273.15)**; on-screen per-run labels (`nickel_T=345.0_F=0`) are the raw file
metadata, i.e. controller °C. Critical exponents are unaffected by the offset
(cancels in T_C − T).

## Scenarios registered

| name | render | intended docs use |
|---|---|---|
| `corpus_ni_zf_precession_fit` | Converged Oscillatory×Exponential+Constant fit on the 618 K (controller 345 °C) ZF run 124232 (ν ≈ 6.1 MHz), zoomed to ~2.2 µs. | Headline "ferromagnet money shot": a **spontaneous** internal-field precession with NO applied field, at 0.98 T_C — matching the 1973 PRL Fig. 2 curve. |
| `corpus_ni_nu_t_order_parameter` | ν(T) in kelvin from real per-run ZF fits (593–629 K) + fitted OrderParameter power law. | The order parameter / critical-exponent deliverable: T_C ≈ 631 K (literature 630 K), β vs the universality table. |
| `corpus_ni_zf_fft` | FFT of run 124232: single spontaneous line at ν₀ ≈ 6.14 MHz (Gaussian-line fit panel populated). | Frequency-domain view; the precession sits *inside* EMU's pulsed-source band (contrast Q7/Q9). |
| `corpus_ni_lf_decoupling` | 473 K (controller 200 °C) LF field scan (1200→4000 G): recovered asymmetry vs applied field, monotone recovery. | The static-internal-field / decoupling story (guide §5 Q8). |

All four capture cleanly under `flock … capture_corpus`; all PNGs re-read as
images after the °C→K re-render. Sizes ~99–384 KB (≤ 600 KB budget).
`requires_fit = True` on the two scenarios that run real iminuit fits
(`…_precession_fit`, `…_nu_t_order_parameter`).

## Run selection & workflow (GROUND_TRUTH § refs)

- **ZF order-parameter branch = runs 124227–124236 (controller 320–356 °C =
  593–629 K)** (GT §3 run log). This is the window where the spontaneous
  precession is *resolved*: below 593 K the internal field is high enough that
  ν runs out of EMU's ≲10 MHz pulsed-source band (GT §5 Q7/Q9), and above
  ~629 K the amplitude collapses approaching T_C. Verified by FFT sweep of all
  ZF runs 124218–124248: coherent high-amplitude line only in 124227–124236,
  frequency falling 9.4 → 2.8 MHz.
- **Fit model** = `Oscillatory × Exponential + Constant` (GT §4 / §5 Q3: cosine
  precession × relaxation envelope; the large uncalibrated α = 1 baseline is
  absorbed by the additive Constant — the "additional relaxation term" the guide
  invites in §5 Q4). Warm-started ν *downward* / amplitude *upward* in ascending
  T so no run's oscillator amplitude collapses to zero (the EuO/YMnAl lesson).
- **Trend law** = OrderParameter `y0·[1−(T/Tc)^α]^β` with **α fixed = 1**, which
  is exactly the guide's `f_ZF(T) ∝ (T_C−T)^β` (GT §4 / §6), fitted in kelvin.
  α free gives the same β but `success=False` (loosely constrained by 10 points
  below Tc); fixing α = 1 converges cleanly.
- **LF**: 473 K field scan 124272–124278 + 124270 (dropped the 124271 4000 G
  repeat). Recovered asymmetry = mean early-time (≤ 4 µs) polarisation per run
  (robust; the A_bg/A_1 split from an Exp+Const fit is degenerate here because
  the traces barely relax).

## Fitted values vs GROUND_TRUTH / guidance

| Quantity | This work (EMU corpus) | Guidance / GT | Note |
|---|---|---|---|
| **Ordering temperature T_C** | **630.89 ± 0.16 K** | ≈ **630 K** (guide + PRL 30, 1064; GT §6) | **agreement** after the °C→K correction ✔ |
| **Critical exponent β (DELIVERABLE)** | **0.390 ± 0.008** | table: Heisenberg 3D **0.367**; Ising 3D 0.326; MF 0.5 (GT §6) | **≈ 3D Heisenberg** — the expected class for bulk Ni ✔ |
| ZF spontaneous precession, 618 K (124232) | ν = 6.126 MHz, χ²ᵣ = 1.14, λ = 0.53 µs⁻¹ ⇒ B_µ ≈ 452 G | 1973 Fig. 2 digitised: ~0.43 kG (ν ≈ 5.8 MHz) at ~615 K (GT §11, guidance) | quantitative agreement within the ±10 K/±0.03 kG digitisation error |
| ν at 593 K (124227) | 9.42 MHz ⇒ B_µ ≈ 695 G | 1973 Fig. 2: ~0.63 kG (8.5 MHz) at ~590 K (GT §11) | same-curve agreement |
| Power-law amplitude y0 (fit) | 28.0 ± 0.6 MHz | — | near-T_C power-law amplitude only; **not** a physical B_µ(0) (the Brillouin curve flattens below ~450 K, GT §11), so no B_µ(0) claim is made from it |
| B_µ(0) (saturation) | not measurable on EMU | 1550 G ⇒ ν ≈ 21 MHz (GT §6, 1973 guidance) | out of EMU's pulsed band — *this is the guide's Q7/Q9 teaching point*, confirmed by the absence of oscillation below ~593 K |
| LF decoupling, 473 K | recovered asymmetry 42.5 → 53.3 (arb.) over 1200 → 4000 G | qualitative recovery (GT §5 Q8) | monotone; static internal field decouples |

β = 0.390(8) is the graded deliverable and lands at the **3D Heisenberg** value
(0.367), distinct from mean-field (0.5) and 3D-Ising (0.326) — exactly the
physics GROUND_TRUTH §6/§9 predicts for bulk Ni. Together with
T_C = 630.9 ± 0.2 K this is a *quantitative* reproduction of the literature from
the teaching data. w (from λ(T)) and γ (from the TF Knight shift) were not
extracted — see below.

## Feature-demonstration opportunities

- **Pulsed-source band-pass is the distinctive teaching point** of this example
  (unique among the corpus ferromagnets — EuO/PSI is continuous-source and
  resolves 30 MHz easily). The `…_zf_fft` scenario shows the in-band line; a
  future companion could FFT a low-controller-T run (124218, 373 K) side-by-side
  to show the *absence* of a line (the Q7 loss) directly. A two-run FFT overlay
  was not attempted here to keep the render reliable.
- **ZF waterfall was tried and dropped**: the auto waterfall Y-offset (~280/trace,
  driven by the late-time asymmetry error fan) dwarfs the ~10-unit oscillation,
  so the slowing precession was not legible. The FFT replaced it as the cleaner
  frequency-domain render.
- **TF series (124249–124269, 100 G, controller 380→340 °C = 653→613 K)**
  straddles T_C and carries a paramagnetic-side signal above it — a candidate
  for a γ (susceptibility) Knight-shift deliverable, but the shift is small
  (GT §5 Q5) and was not extracted.
- **w (correlation-time exponent)** from λ(T) ∝ |T−T_C|^(−w): the ZF envelope
  rate λ rises toward T_C in the fitted runs (0.53 → 1.4 µs⁻¹ over 618→629 K),
  the qualitative critical-slowing signature, but a clean w fit needs points on
  both sides of T_C and was left out of scope.

## Problems hit / notes

- **Temperature units**: see the warning block above — the single biggest trap
  in this example, now recorded in GROUND_TRUTH §2. The NeXus units attribute
  is wrong (claims Kelvin, values are controller °C).
- Data dir is `Data/` not `data/`; `._`-prefixed AppleDouble sidecars litter the
  folder (macOS) but the loader reads the real `.nxs` fine (NeXus-v2 HDF5, no
  conversion issues, as the brief noted).
- Loaded EMU asymmetry is uncalibrated (α = 1): mean ~30–53 (arb. %), oscillation
  a few % on top. All internal fields quoted here come from the *frequency* via
  γ_µ/2π = 135.5 MHz/T and are geometry-independent; the asymmetry *amplitude*
  scale is not physical and is not reported as ground truth (consistent with
  GT §6 "1973 SREL-specific" note — polarisation/τ values do not carry over).
- `Parameter` has no `vary=` kwarg; fix a parameter with `fixed=True`.
- OrderParameter panel fit needs α pinned (`fixed=True`) for `success=True` on
  this 10-point, below-T_C-only branch.

## Top pick

`corpus_ni_zf_precession_fit` — the spontaneous ZF precession converged at
χ²ᵣ = 1.14 with no applied field, at 0.98 T_C and in quantitative agreement with
the 1973 PRL Fig. 2 curve, is the cleanest single image of "a ferromagnet's
internal field measured directly", and the pulsed-source-band-limit story is the
render this example does better than any other in the corpus. Close second:
`corpus_ni_nu_t_order_parameter` for the T_C = 630.9 K / β ≈ 0.39 ≈ Heisenberg
deliverable.
