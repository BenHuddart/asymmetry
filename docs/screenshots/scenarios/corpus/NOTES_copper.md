# NOTES — Muon diffusion and QLCR in copper

Module: `copper_diffusion.py`
Example: `Nuclear magnetism and ionic motion/Muon diffusion and QLCR in copper`
Spec: that folder's `GROUND_TRUTH.md` (GT). Corpus is read-only.

The classic muon-diffusion teaching set: muons at octahedral interstitial sites
in fcc Cu dephasing in the ⁶³Cu/⁶⁵Cu nuclear-dipolar field, motionally narrowing
as the muon hops. Headline physics = the **quantum-diffusion hop-rate curve**
(mobility minimum ~50 K; Luke *et al.*, PRB 43, 3284). Data: 2010 EMU set
(20882–20917) + 2024 ARGUS set (76924–76961).

## Scenarios registered

| Scenario | Render | Intended docs use |
|---|---|---|
| `corpus_cu_zf_static_kt` | Static Gaussian KT fit on ARGUS 76935 (40 K ZF): textbook dip-and-⅓-recovery. | ZF core / KT lineshape on real data. |
| `corpus_cu_zf_quantum_diffusion` | EMU ZF 40 K (20886) vs ~5 K (20887) overlay, 8× bunched: the ⅓ tail relaxes at low T. | Guide Q2/Q3 — static KT vs low-T quantum diffusion. |
| `corpus_cu_tf_abragam` | Abragam fit on EMU 20885 (100 K, 100 G TF): precession damped by the Abragam envelope. | TF core / Abragam → hop rate; Δ cross-check. |
| `corpus_cu_hop_rate_arrhenius` | **Headline** ν(T) 5–200 K (log-y): quantum-diffusion minimum + Arrhenius high-T branch. | Hop rate, mobility minimum, E_a deliverable. |
| `corpus_cu_qlcr_scan` | EMU 40 K LF integral-asymmetry field scan (20888–20900): QLCR dip ~78 G. | Unique feature — integral-counting QLCR by level crossing. |

Workflow follows GT §4 (per-geometry) and §5 (guide questions). Run selection
per GT §3; set-temperatures used where read-T is flagged (GT §9).

## Fitted values vs GT anchors

| Quantity | Asymmetry (this module) | GT anchor | Verdict |
|---|---|---|---|
| ZF static-KT Δ (ARGUS 40 K, 76935) | **0.394 µs⁻¹** (χ²ᵣ 1.41) | 0.38–0.39 µs⁻¹ (textbook, GT §6/§10) | ✓ on the anchor |
| TF Abragam Δ (EMU 100 K, 20885) | **0.385 µs⁻¹**, ν 0.267 µs⁻¹, f 1.395 MHz (χ²ᵣ 1.01) | Δ cross-checks ZF; TF-derived ~10% lower (GT §10) | ✓ consistent with ZF Δ |
| ZF dynamic-KT Δ (EMU series) | ~0.37 µs⁻¹, stable to ~180 K | 0.37 µs⁻¹ (program, GT §7) | ✓ |
| ZF dynamic-KT ν(T) | 0.10 (5 K) → **min ~0.018 (60 K)** → 2.27 µs⁻¹ (200 K) | ν ~0.01–0.06 (<90 K) → 2.15 MHz (200 K) (program, GT §7) | ✓ matches; low-T rise present |
| Arrhenius E_a (high-T, 90–200 K) | **73 meV** (a 161, c 0.06) | 62 ± 3 meV (program, GT §7); method-dependent | ✓ same ballpark; higher fit window ⇒ higher E_a |
| QLCR resonance field (EMU integral) | dip at **~78 G** (dense 75–90 G) | ~80 G region (deliverable, GT §6) | ✓ |

All values are (deliverable)/(textbook)/(program) tier — the guide states no
target numbers (GT §6, §9), so grading is against the loose literature anchors.

## Feature-demonstration opportunities

- **Integral-asymmetry field scan** (ALC/integral-scan view) doubles as the QLCR
  render — reused the `tcnq_alc.py` path (`_plot_workspace.set_active_view
  ("integral_scan")` → `_alc_fit_panel.build_requested`). The EMU dense scan
  gives a cleaner localized dip than ARGUS, so EMU 20888–20900 was chosen.
- **Per-parameter log-y** in the trending panel (`_y_controls["nu"].log`) is what
  makes the two-decade ν(T) span (and thus the quantum-diffusion minimum) legible
  in one frame — a nicer showcase than the linear llz ν(T) render.
- **Multiplicative composite** `Oscillatory × Abragam` collapses to
  `A_1·cos(2πft+φ)·G_Abragam(Δ,ν)` (dedup'd amplitude/baseline) — a clean way to
  fit a TF envelope with a ZF-family relaxation function.
- Not captured but available: ZF dynamic-KT single-run fit at high T (motional
  narrowing the static model can't hold); ARGUS QLCR scan (76941–76955);
  TF/ZF hop-rate cross-check plot (Q4).

## Problems / honest notes

- **Quantum-diffusion low-T rise is real but single-point.** Only one sub-40 K ZF
  point per instrument (EMU 20887, ARGUS 76938/76936), and no sub-K data, so the
  mobility *minimum* is resolved (5 K ν≈0.10 > 60 K ν≈0.018) but the full
  tunnelling upturn Luke saw below ~1 K is out of range. Framed honestly: minimum
  + activated branch, not a complete quantum-diffusion curve.
- **20887 temperature is ambiguous** — set 1 K / read 5.80 K (GT §9). Plotted at
  its set value (labelled ~5 K in prose). Either placement keeps it the lowest,
  elevated-ν point; the qualitative story is unchanged.
- **ARGUS ZF is messy** for a T-trend (frozen/suspect read-T on 76937/76956 and
  the LF block, GT §3b/§9), so the ν(T)/Arrhenius trend uses the **EMU** ZF series
  (20886, 20901–20917 + 20887) which is dense and clean. ARGUS 76935 is used only
  for the single static-KT render (highest-stats 40 K ZF, Δ=0.394).
- **EMU TF is sparse** (only 20883/84/85 at 280/250/100 K, 100 G — GT §9). The TF
  Abragam single-fit uses 20885 (100 K), where Δ is still near-static and cleanly
  cross-checks the ZF width. ARGUS 20 G TF fits poorly (0.27 MHz precession, ~2
  cycles in window; χ²ᵣ 4–10) so it was not used.
- **Static-KT χ²ᵣ = 1.41 flags "poor"** in the GUI quality badge, but the fit is
  visually excellent (2010 ndof); the label is the program's strict threshold.
- `set_bunch_factor(8)` was needed on the quantum-diffusion overlay — the raw ZF
  error fan buries the tail contrast otherwise (README noise lesson).

## Capture status

All 5 captured green under `flock`, PNGs read back and verified (all < 400 KB):
`corpus_cu_zf_static_kt`, `corpus_cu_zf_quantum_diffusion`, `corpus_cu_tf_abragam`,
`corpus_cu_hop_rate_arrhenius`, `corpus_cu_qlcr_scan`.
Top pick: **`corpus_cu_hop_rate_arrhenius`** (the quantum-diffusion minimum +
Arrhenius on log-y is the example's signature result), closely followed by
`corpus_cu_qlcr_scan` (unique level-crossing feature).
