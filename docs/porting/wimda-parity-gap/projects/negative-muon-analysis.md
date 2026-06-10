# Project brief: negative-muon-analysis (DEFERRED)

Umbrella: `wimda-parity-gap` · **Deferred** (decision 2026-06-10) — not in
the parallel batch · Size L if promoted · adapt, don't port

## Motivation

μ⁻ elemental analysis (capture-lifetime spectroscopy) is a real and growing
ISIS programme, and WiMDA is the only one of the reference programs with any
support. It is, however, a different community and observable from μSR spin
relaxation, and WiMDA's implementation is visibly unfinished (large
commented blocks, hard-coded parameter slots p[213..223]). Deferred until
there's a concrete user/collaboration pull; this brief records what to
salvage so the knowledge isn't lost.

## WiMDA reference

`NegMuAnalyse.pas` (2474 lines) + hooks in `Analyse.pas` ("Muon Polarity"
radio) and `PlotPar.pas` (μ⁻ lifetime / polarity in the decay-corrected plot
mode):
- **Element lifetime table**: built-in μ⁻ capture lifetimes for 67 elements
  H→Pb (`mystrings`/`myelements`, lines 104–120) — the most directly
  salvageable asset.
- **Multi-exponential lifetime fitting**: up to 5 elemental components +
  decay background, each `N·exp(−t/τ)` with τ seeded from the table;
  separate F/B amplitudes; optional polarisation function (None/LorGau/
  Diamagnetic) multiplying the fit for μ⁻SR-style work.
- **Capture-ratio report** (`Ratio`, lines 455–911): amplitude ratios of the
  element of interest vs others, separately for F and B — relative capture
  probabilities → elemental composition.
- **Set-as-BG**: evaluate fitted unwanted components and subtract from
  displayed data.
- Dedicated GLE export (~700-line generator).

## If promoted: adaptation sketch (not a port)

- Element table → a data module in `core/` (with literature citation for the
  lifetime values — verify against a current compilation, e.g. Suzuki,
  Measday & Roalsvig, before trusting WiMDA's numbers).
- Multi-exponential capture model → a composite component family in
  `core/fitting` (sum of `N·e^{−t/τ}` with element-pinned τ presets) fitted
  on raw single-histogram counts — which **depends on the single-histogram
  count-fit mode from `count-domain-fit-modes`** (natural sequencing if
  promoted).
- Capture ratios → a derived-quantities report, not a bespoke form.
- μ⁻ polarity (lifetime in the decay-correction) → small plot/reduction
  option, shared with `workflow-visualisation`'s raw/decay-corrected views.
- Phasing: (1) element table + multi-exp fitting on counts; (2) ratio
  report + docs; (3) polarisation-function mode (μ⁻SR) if wanted.

## Prerequisites for promotion

- A user or collaboration with concrete μ⁻ runs and questions.
- Test data: no μ⁻ corpus exists locally — acquiring ISIS elemental-analysis
  runs is a hard prerequisite.
- Cross-check target: Mantid's Elemental Analysis interface (μ-XRF side is
  out of scope, but its lifetime handling is a useful comparison; GPL —
  oracle only).
