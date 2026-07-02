# Fit / trend / representation consolidation — GUI audit findings and decisions

Companion to [README.md](README.md) (the five-representation review). This file
records the **live GUI audit** (2026-07-01/02, WiMDA Muon School corpus, EuO
`deltat_pta_gps_29xx.bin` runs 2923–2973) and the decision Ben took on each
question. The chosen designs are folded into
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

Audit method: the app was driven end-to-end from a worktree build for **each**
representation — load → single fit → batch fit → trend → trend model-fit →
save/reload → view switches. F-B asymmetry used the ZF temperature scan
2923–2960 (T_C = 69.05(1) K ground truth); grouped/raw-counts, FFT, MaxEnt and
integral scan used the TF60G runs 2961–2973 and ZF run 2960 (ν ≈ 30 MHz).

Finding numbers F1–F11 come from the F-B pass, F12+ from the
grouped/FFT/MaxEnt/scan/round-trip pass. README §-references are to the
five-representation review.

---

## 1. Decided design questions

### D1 — DataGroup ↔ FitSeries coupling (README §6) → **Option B: Linked**

Evidence: browser DataGroup "B = 60 G" and grouped batch series "B = 60 G"
share an auto-name but nothing else; batches are built from the live browser
selection, which can include runs the browser is not even displaying (F9 made
runs 2961–2967 invisible, and the grouped batch silently fit exactly those 7
invisible runs); re-runs spawn duplicate identically-named series (F13).

Decision: DataGroup owns an order key and back-references to series built from
it; batch surfaces gain "fit this group" and record provenance (group id +
member snapshot); series remain independent objects. Additive schema change.

### D2 — Model presentation on run selection (README §7) → **Carry forward + visible provenance badge**

Evidence (F6): selecting unseen TF run 2963 presented the ZF precession model
*and 2960's fitted values* with nothing indicating it had never been fitted.
After reload the F-B batch form even held the frequency-domain Gaussian model
(F21). Carry-forward itself is useful for run series and is retained.

Decision: keep carry-forward, but the panel must state it explicitly
("model carried from 2960 — not fitted for this run"), cleared when a fit is
recorded; form state keyed per representation (fixes the F21 restore mix-up at
the same time).

### D3 — Trend quality gating (F2/F3/F5/F12/F14) → **Flag + one-click exclude, no auto-exclusion**

Evidence: F-B batch dropped 5/25 members with only a log line ("20 datasets");
garbage members (run 2949: A₁ = 967 ± 316 %, f ≈ 0.012 MHz; run 2947:
A₁ ≈ 1.4e-5) plotted unflagged and pulled the OrderParameter trend fit to
Tc = 70.9 K vs 69.05 K ground truth; grouped fits report **no χ²ᵣ and no
parameter errors anywhere**; grouped members table has no quality column and no
include-in-trend toggles; trend points cannot be excluded by click (the
integral-scan panel already has this interaction).

Decision: every series member carries χ²ᵣ + sanity flags (relative-error
magnitude, bound-pinning); flagged points render distinctly in trends; member
tables across all representations gain quality + include checkboxes; trend
plots gain click-to-exclude with provenance. Exclusion is never automatic.

### D4 — Series identity and re-run semantics (F13/F22) → **Replace-in-place + one naming scheme**

Evidence: re-running the grouped batch created a second identically-named
"B = 60 G" chip; F-B re-runs replace; four naming conventions coexist
("Model · 2923–2960", "B = 60 G", "GaussianPeak + ConstantBackground ·
2952–29…", "Integral scan 1" / "Model fit (single): frequency vs temperature").

Decision: a batch series is keyed by (representation, member set, model);
re-running updates that series in place. Unified name
`<model> · <member-range> [· <representation>]`, user-renamable. Duplication
only via explicit "Duplicate series". Migration rule needed for projects
already containing duplicates.

### D5 — Default parameter classification (F10) → **All Local by default**

Evidence: the first amplitude-like parameter defaults to Global in every batch
(F-B A₁, grouped B (G), frequency height) → either a silent global fit or a
trend-less batch: fitting 35 grouped traces ended in "No varying fit
parameters" with an explanatory footnote instead of a trend.

Decision: all free parameters default to Local; Global is explicit opt-in
(wizard may *suggest* Global). Behaviour change is called out in the plan's
test updates.

### D6 — Frequency-domain fitting defaults (F15/F16) → **Editable range + non-DC peak seeding + DC guard band**

Evidence: the frequency Fit tab's FIT RANGE fields are disabled placeholders
showing the *time-domain* fit range (0.006–9.839) as MHz and typing into them
does nothing; the actual fit always spans the full spectrum; default ν₀ seeding
locked onto the DC/apodisation spike (ν₀ = 0.97 MHz, χ²ᵣ = 137, flagged
"poor") while the physical 30 MHz peak was in plain view.

Decision: make the frequency fit range editable (default = displayed X
window), seed ν₀ from the dominant non-DC peak, exclude a small DC guard band
by default.

### D7 — MaxEnt ZF window (F19) → **Data-aware window when B ≈ 0 + actionable divergence message**

Evidence: "Auto window from field" centres the reconstruction window on the
applied field; for ZF run 2960 (internal field ν ≈ 30 MHz) the window sat at
0–≈10 MHz, guaranteeing the observed divergence ("stopped early at cycle 15 as
χ² began rising"); the message suggests adjusting "the time/frequency window"
without naming the Window control.

Decision: when B ≈ 0, derive the window from the data (FFT peak scan /
Nyquist fraction); divergence message names the Window control and the current
window bounds.

**Live real-data verification (2026-07-02, scripted against the actual
`deltat_pta_gps_2960.bin`).** The Phase 6 data-aware branch fires as intended on
the real run (B = 0 G, T = 1.5 K): it returns a data-derived window (≈0–5.9 MHz)
rather than the trivial `(0, 10)` field fallback, and the reconstruction
converges (χ² falls monotonically, no divergence). But the real 2960 signal has
**no dominant ~30 MHz peak** — the "ν ≈ 30 MHz" figure above was a mis-estimate.
After lifetime correction the only coherent content sits at ≈0.5–2 MHz (verified
by direct FFT of both grouped signals; everything above ~10 MHz is noise), the
peak finder correctly locks onto ≈2 MHz, and the reconstructed spectrum puts its
weight near-DC. So the mechanism is correct; the acceptance target should read
"the window tracks the data-derived dominant peak instead of collapsing to the
near-DC field fallback," not literally "contains 30 MHz." Left the original
figure above intact for provenance rather than rewriting the audit after the
fact.

### D8 — Plan scope → **Everything in one plan, bugs first**

Phase 0 = data-integrity bugs (F21 round-trip mix-up, F9 sorted-group vanish)
with regression tests; then the design work; small state/UX bugs (F17, F18,
F20) as standalone steps. Phases must be independently landable.

---

## 2. Full findings ledger

Bugs (fix, no design input needed):

| # | Finding | Evidence |
|---|---------|----------|
| F9 | "Form Data Group" while the browser is column-sorted hides the group header **and** its member rows until project reload; selection of the hidden runs persists invisibly and batches operate on them. Data survives (.asymp reload shows the group). | Group "B = 60 G" (2961–2967) vanished in both sort directions; reappeared after reload. |
| F21 | Save/reload representation mix-up: (a) F-B batch series chip absent from the F-B Parameters view after reload; (b) "Integral scan 1" chip appears in the F-B trend view; (c) the frequency Gaussian model (`height*exp(-4*ln(2)*((nu-nu0)/fwhm)^2)+bg`, ν in MHz) restored into the F-B **time-domain** Batch form incl. its Global/Local classification; (d) FFT plot restores with a legend entry ("2960 Average") but no data. Grouped and frequency series chips + trends survive correctly. | audit-roundtrip.asymp, 2026-07-02. |
| F17 | Frequency "Run Batch Fit" enabled-state is stale: stayed disabled after FFTs were available and enabled only after an unrelated browser-selection change. Same class as the PR #89 `_update_fit_block_state` fix. | 8-run selection, 2026-07-02. |
| F18 | "More… → Add to Series…" after a completed frequency single fit silently does nothing (no dialog, no log, no chip). | Run 2960 Gaussian fit. |
| F20 | Wheel-scrolling right-dock panels mutates unfocused spin boxes: scrolling the MaxEnt panel changed Spectrum points 1024 → 512 silently. | MaxEnt panel. |
| F2 | F-B batch silently dropped 5 of 25 members; only evidence is the log line "20 datasets"; no failure list or member markers. | ZF 2923–2960 batch. |
| F8 | Browser default insertion order + OS-scrambled file-dialog order → shift-range selections silently non-contiguous. Header-click sorting exists but is undiscoverable (and triggers F9). | 51-run load. |
| F12 | Grouped fit completion reports no χ²ᵣ (log: "Grouped time-domain fit completed: 5 groups"); no parameter uncertainties shown anywhere in the grouped panel. Terminology mislabels: plot title "Grouped time-domain — 5 runs" for 5 detector groups of one run; log "35 groups" for 35 run×group traces. | Runs 2967, 2961–2967. |
| F14 | Fitted-members table: metadata column `B (G)` and fitted parameter column `B (G)` distinguishable only by italics; no per-member χ²ᵣ; no include-in-trend toggles (F-B table has them). | Grouped series table. |

Design-adjacent frictions folded into D1–D8 or the plan's smaller steps:

| # | Finding |
|---|---------|
| F1 | Single/Batch tabs hold independent model state; F-B view lacks the grouped view's "Share with Group"/"Send to Batch" bridges. |
| F3 | Garbage fits pass into trends unflagged (see D3). |
| F4 | Trend model-fit defaults to Linear `m*x+b` with nonsense seeds even for obvious order-parameter data; OrderParameter must be typed by name (its seeds are then data-aware). Dialog's "Quality of fit: poor" band is good and kept. |
| F5 | No click-to-exclude/context menu on parameter-plot trend points (integral scan has one). |
| F6 | Silent model+values carry-forward to unseen runs (see D2). |
| F7 | Single-fitting a batch member with a different model silently overwrites the primary slot; series table/trend keep old values; no diverged/excluded state visible anywhere (README §5.3 confirmed live). |
| F10 | Global-by-default classification (see D5). |
| F11 | F-B batch does not inherit seeds from a just-completed matching single fit; grouped "Send to Batch" does (inconsistent). |
| F13 | Duplicate identically-named series on re-run (see D4). |
| F15/F16 | Frequency range/seeding traps (see D6). |
| F19 | MaxEnt ZF window (see D7). |
| F22 | Four series-naming conventions (see D4). |

Cross-representation observations:

- FFT and MaxEnt fit panels share one in-session form: the FFT view's fitted
  values (incl. the DC-spike ν₀ = 0.970123) silently appeared as the MaxEnt
  batch seeds. Whether this sharing is intended needs an explicit decision in
  code; the plan keys form state per representation (D2) and treats FFT/MaxEnt
  as two representations.
- Stale axis limits persist across representation switches: entering the FFT
  view kept X = 7000–8600 MHz, so a successfully computed spectrum rendered as
  an empty plot — a plausible mechanism behind the overnight "FFT never
  renders" report (PR #89 context).
- The frequency trend presents fitted ν₀/FWHM as B₀ (G)/B_wid (G) via γ_μ —
  useful, but the members table then shows two same-named field columns (F14
  aggravation).
- Integral scan fitting (baseline regions + Gaussian/Lorentzian peaks) is a
  fourth, fully separate fitting surface with no connection to the trend
  Model Fit dialog.

Positives to preserve: auto-switch to Parameters after a batch; explanatory
text when a Global parameter yields no trend; honest "Fit converged … poor"
χ² banding; MaxEnt divergence guard (message wording aside); series tint on
browser rows; the grouped classification "?" help button; typed expressions in
the Build Fit Function dialog.
