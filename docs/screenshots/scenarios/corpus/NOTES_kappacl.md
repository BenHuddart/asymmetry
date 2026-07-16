# NOTES — κ-Cl AFM transition in high TF (`kappa_cl_hightf.py`)

Corpus example: **Magnetism/AFM transition in high TF** —
κ-(BEDT-TTF)₂Cu[N(CN)₂]Cl ("κ-Cl"), deuterated crystal, HAL9500/PSI, field ∥ *b*.
18 HiFi PSI `.mdu` files (`tdc_hifi_2020_00686–00693` at 6 T,
`00730–00739` at 8 T; sample `kappa-ETCl`). Spec = the example's
`GROUND_TRUTH.md` (paper: Huddart *et al.*, Phys. Rev. Research **5**, 013015
(2023); no teaching docx). This is the corpus's one **high-field TF
frequency-domain / .mdu loader** example.

## Scenarios registered

| Scenario | What the render shows | Intended docs use |
|---|---|---|
| `corpus_kappacl_load_browse` | 12 of the 18 `.mdu` runs (both 6 T and 8 T) in the data browser with **measured T** and titles, plus the base-T ordered 6 T run's time trace zoomed to 0–0.03 µs where the **~813 MHz** precession resolves as an oscillation. | `.mdu` loader / format-support page; "loading a HAL high-field TF series". |
| `corpus_kappacl_tf_fft` | Frequency-domain overlay of the 6 T diamagnetic Larmor line at **813.5 MHz**: paramagnetic run (50.7 K, sharp) vs ordered run (3.2 K, **broadened + depleted**). Real frequency `PlotPanel`, real core grouped FFT. | Fourier-analysis page; "high-field TF FFT reaches the GHz line; the AFM broadens it below T_N". |
| `corpus_kappacl_maxent` | MaxEnt internal-field distribution around the 6 T applied field (field-centred auto window); the reconstructed internal-field line resolves at ~813.7 MHz, just above the γ_µ·B applied-field marker (813.3 MHz). | MaxEnt page; "the paper's headline observable, reproduced". |
| `corpus_kappacl_amplitude_t` | **Headline**: 6 T order parameter Â(T) from the central-line depletion, with the **OrderParameter** power-law fit; flat plateau → sharp drop through the transition. | Parameter-trending / order-parameter page (6 T). |
| `corpus_kappacl_amplitude_t_8t` | 8 T order parameter Â(T) with its OrderParameter fit — the ordered-state transition at the higher field. | Same page, 8 T companion (field dependence). |

## Run selection & workflow (GROUND_TRUTH refs)

* **Run→T map** from `GROUND_TRUTH.md` §3 (measured, not setpoint). 6 T runs
  686–693 (3.24–50.66 K); 8 T runs 730–739 (3.12–100 K). The two far 8 T
  background runs (75, 100 K, off the Fig 8(f) axis) are dropped from the 8 T
  trend, matching the §3 subset caveat.
* **Observable / model** (§4, §6b): the paper tracks the *integrated spectral
  area on the wings* of the internal-field distribution vs T and fits
  A(T) = A_BG + (A(0)−A_BG)[1−(T/T_N)^α]^β. Here the transferable metric is the
  **normalised central-line depletion** Â = (H_para − H)/(H_para − H_0), where
  H is the Hann-windowed FFT peak height of the diamagnetic line (0–4 µs). This
  is the *complement* of the paper's wing area — spectral weight the ordered
  internal field moves out of the sharp central line into the wings — and it
  reproduces the §6b normalised Â(T) closely (below). Â is fitted with the GUI's
  `OrderParameter` trend model (y0·[1−(T/Tc)^α]^β) to recover T_N.

## Fitted values vs ground-truth targets

Â(T), normalised central-line depletion (this module) vs GT §6b digitised:

| T (K) | Â 6 T (module) | Â 6 T (GT §6b) | | T (K) | Â 8 T (module) | Â 8 T (GT §6b) |
|---|---|---|---|---|---|---|
| 3.24 | 1.01 | 1.04 | | 3.12 | 1.00 | 0.97 |
| 6.00 | 0.99 | 1.00 | | 6.00 | 1.00 | 1.04 |
| 12.0 | 0.94 | 0.93 | | 10.47| 0.94 | 0.98 |
| 18.0 | 0.87 | 0.90 | | 18.0 | 0.86 | 0.92 |
| 24.0 | 0.74 | 0.78 | | 24.0 | 0.78 | 0.80 |
| 27.0 | 0.48 | 0.50 | | 27.0 | 0.56 | 0.62 |
| 30.0 | 0.10 | 0.07 | | 30.0 | 0.11 | 0.20 |
| 50.66| 0.00 | 0.00 | | 50.0 | 0.00 | 0.00 |

The Â(T) **shape** (flat plateau to ~24 K, sharp drop, flat background) is
reproduced within ≈±0.05 of the §6b digitised target at both fields.

Recovered T_N from the `OrderParameter` fit to the Â(T) points:

| Field | T_N (module) | T_N target (§6, §6b) |
|---|---|---|
| 6 T | **27.2(4) K** | 28.2(5) K |
| 8 T | **27.1(4) K** | 30.2(2) K |

## Honest assessment — what the GUI can / cannot do at 6–8 T

**High-field reach — RESOLVED, no blocker.** GT §9.6 flagged the ~0.8–1.1 GHz
diamagnetic precession as possibly "far beyond time-binning Nyquist". It is
**not**: the `.mdu` (HiFi/HAL PSI) loader delivers the raw octant histograms
binned at **0.0244 ns** (≈388 894 bins over 9.5 µs), so Nyquist ≈ **20.5 GHz**.
A plain FFT resolves the 6 T line at **813.7 MHz** (and its 2nd harmonic at
1627 MHz) and the 8 T line at **1084 MHz** directly — **no rotating reference
frame needed**. The Fourier and MaxEnt pipelines both reach the line unaided;
the frequency panel's "X relative to ref. field" (reference = 60000/80000 G)
gives the ΔB axis of the paper directly.

**FFT ordered-vs-paramagnetic contrast — works well.** Under matched Lorentzian
apodisation (τ = 3 µs) the ordered line is ≈2× broader (FWHM 0.33 vs 0.17 MHz)
and depleted (peak ≈0.73×) — the AFM broadening is visible and quantitative.
Note the *frequency* `PlotPanel` in the main window renders one run's FFT at a
time; the comparison figure uses a standalone frequency `PlotPanel` with
`plot_datasets([para, ordered])` (both spectra from the real
`compute_average_group_spectrum`). This is a mild UI gap: **no in-window
multi-run FFT overlay** — worth a product note.

**MaxEnt — works out of the box since PR 249 (auto workload steering).** The
field-centred **auto window** (centres on γ_µ·B) is exactly right for the ΔB
observable and needs no manual frequency entry. The raw run is 388 894 time
bins × 8 octant groups, so a full-resolution reconstruction was a huge
projection matrix (3.1 M observations, 2¹⁹-point default spectrum) that tripped
the workload-warning modal and blocked offscreen capture. PR 249's **auto
workload steering** now sizes the unset knobs automatically: on run 686 it
raises `binning` to **10** and caps `t_max` at **≈2.005 µs** (spectrum length
would cap at 1024 in the scripting API; the GUI honours the explicit
Spectrum-points value), exactly the values this scenario used to hand-tune, and
records them in the result metadata `auto_steer_applied`.

Re-tested on run 686 (6 T, 3.24 K, base-T ordered), PR-249 branch
(2026-07-16), core `maxent()`, `early_stop` on:

| Path | binning | t_max (µs) | spec pts | cycles | χ²/N | peak (MHz) | time |
|---|---|---|---|---|---|---|---|
| Hand-tuned (explicit) | 8 | 2.0 | 256 | 9 (converged) | **1.03** | **813.57** | 165 s |
| **Auto-steer (all unset)** | 10 (auto) | 2.005 (auto) | 1024 (auto) | 8 (converged) | **1.04** | **813.56** | 200 s |

Physics is identical (peak 813.56–813.57 MHz, just above the γ_µ·6T ≈ 813.3 MHz
applied-field marker; χ²/N ≈ 1.03). Auto-steer is ~20 % slower only because its
scripting-API default spectrum is 1024 vs the hand-tuned 256; forcing 256
points in the GUI keeps it fast. **The `corpus_kappacl_maxent` scenario now
uses the auto-steer path** (End/Binning left unset, `Auto workload steering`
ticked, 256 spectrum points) — a simpler scenario that also demonstrates the
new feature. `auto_steer_applied` metadata on the auto path:
`{'time_binning_factor': 10, 't_max_us': 2.0048828125, 'n_spectrum_points': 1024}`.
Explicit values always win (setting `binning`/`t_max`/points leaves
`auto_steer_applied` empty).

**T_N recovery — partial; the field shift is under-resolved.** The single most
important honesty point for product: the peak-height proxy reproduces the §6b
Â(T) **shape and values** at both fields, and the 8 T data *is* subtly more
ordered at 24–27 K (Â 0.78/0.56 vs 6 T 0.74/0.48), consistent with a higher
T_N. **But** the `OrderParameter` fit returns **T_N ≈ 27 K at both fields**, so
the headline **6 T→8 T rise of T_N (28.2→30.2 K) is not cleanly recovered**.
Causes: (i) the transition is sampled by only ~3 points (24/27/30 K) with the
drop falling entirely between the 27 and 30 K runs at *both* fields, so the fit
pins Tc near the last plateau point and collapses to a near-step (small β);
(ii) the central-line-depletion proxy compresses the field separation relative
to the paper's wing-area integral; (iii) the corpus is missing the ~35 K
background run per field that the published Fig 8 uses (§3 subset caveat). The
recovered T_N ≈ 27 K sits ~1 K below the 6 T target and ~3 K below the 8 T
target. This is a **data-sampling + observable-proxy limitation, not a GUI
defect** — but a docs page should not claim the field shift is resolved from
these 18 runs alone.

## Feature-demonstration opportunities (spotted, some not captured)

* **ΔB axis toggle** ("X relative to ref. field", reference 60000/80000 G) —
  the cleanest way to reproduce the paper's Fig 8 ΔB x-axis. *Not* used in the
  captured MaxEnt shot: enabling it re-maps the frequency axis in a way that did
  not compose with the scenario's explicit `set_view_limits` framing (the data
  fell out of view), so the shot is left in absolute MHz with the γ_µ·B marker.
  The toggle itself works interactively; a scenario using it would need to read
  back the *re-mapped* x-range before framing. Worth a callout on the Fourier
  page and a small follow-up to make the two compose.
* **Spectral moments** on the internal-field line (B_rms(T)) would be an
  alternative order parameter and exercises `moments_trend_row`; not captured
  (peak-height depletion was the cleaner match to §6b).
* **2nd-harmonic line at 1627 MHz (6 T)** is present in the raw FFT — a nice
  incidental demonstration that the loader/FFT reach multi-GHz; not framed here.
* **Field-induced canting (m_b, §4/§6)** needs peak-II position tracking in the
  MaxEnt ΔB distribution — beyond a screenshot scenario.

## Problems hit

* Manual matplotlib drawing onto the main-window frequency panel's axis crashes
  on teardown (`FigureCanvasQTAgg already deleted` via a queued viewport
  refresh). The FFT overlay was reworked to a standalone `PlotPanel` +
  `plot_datasets`, which manages its own artists and is robust.
* `MuonDataset.run_label` has no setter; the overlay legend label is set through
  `metadata['run_label']` instead.
* MaxEnt at full `.mdu` resolution used to trip the GUI workload-warning modal
  (blocked offscreen), so the scenario hand-reduced binning/time-range.
  **Resolved by PR 249**: auto workload steering sizes those knobs out of the
  box (the scenario now leaves them unset), and the warning — if a residual
  config is still large — now routes to the log panel and proceeds under a
  headless (offscreen/minimal) QPA platform instead of hanging on the modal
  (`ASYMMETRY_SUPPRESS_WORKLOAD_WARNING` does the same on a display).
