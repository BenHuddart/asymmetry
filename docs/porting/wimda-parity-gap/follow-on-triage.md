# Follow-on triage — post-Wave B (PRs #54–#58)

Date: 2026-06-12. Compiled from a sweep of every study's recorded
follow-ons after Wave B merged (main at `eaeeacf`), verified against code
where cheap. Successor to [wave-a-closeout.md](wave-a-closeout.md)'s
status tracking. Ben's triage decisions 2026-06-12.

Already closed without doc updates (for the record): MINOS-on-α (#56's
shared drive_minuit), arbitrary-X + cross-group error modes/windows +
generic quadrature (= `⊕`, PR #38/#39), α-free-fit promote path (PR #50),
both Wave A strays (PR #44). Pure decision-records, not open work:
no-sandboxing, project-local plugins REFUSED, Set-BG out of scope,
per-detector FFT / FB t=0 extrapolation out-of-scope records.

## NOW — four sessions

| # | Session | Items | Why now |
|---|---|---|---|
| ① | physics-sign-off + quick wins (S) | Kadono transcription check + non-textbook source sign-off incl. MuLFRelax (1−δ) (fit-function-parity); fit-log provenance population (fit-workflow); testing-worktree golden regeneration post-#35 (error-propagation); configurable MINOS confidence; fit-wizard opt-in for the 12 parity components | The Kadono item is a flagged **possible physics bug in shipped code** — highest priority in the whole list; the rest are verified-open small wins |
| ② | RRF finish + engine pass (M) | RRF fit GUI exposure + engine-level frequency_offset (rrf follow-ons — held back only by Wave B fit-panel ownership, now moot); fgAll→Poisson cost-factory unification (twice deferred; last statistics-quality gap) | WiMDA users fit in the RRF from the GUI — without this, RRF fitting is API-only by accident, not decision |
| ③ | batch arithmetic (M) | In-batch co-add + re-fit-coadded selections (WiMDA BatchFit Smooth/Bin + fit-table coadd-refit; `combine_runs` dependency satisfied by #55); symmetric N-run co-subtract surface; MaxEnt batch-reconstruct-then-send (makes moments usable for B_rms(T) series) | The remaining WiMDA batch-workflow parity + the moments headline use case |
| ④ | Wave C workflow-visualisation (M) | Run stepping, ASCII export + batch, events columns, B-from-log, deadtime auto-discovery + stale warning, log-count view (raw shipped in #53), F/B overlay, snapped cursor; live-run monitoring stays an optional beamline-gated phase | The bulk of remaining parity. GOVERNING TEST (Ben, 2026-06-12): each item is weighed against the established Asymmetry workflow — we adopt WiMDA functionality that *benefits* that workflow, we do not make Asymmetry behave like WiMDA; per-item adapt-or-reject verdicts expected |

After ①–④, every WiMDA-parity item in the programme is closed except the
deliberately dormant set (negative-muon in motion via its committed plan,
fc5e9f7; live-run awaiting beamline access; Sconv awaiting a regularised
formulation).

Sequencing: sessions ② and ④ touch fit_panel/data_browser/mainwindow —
launch only after the in-flight local work on the hub's main (ahead of
origin, uncommitted fit_panel.py edits; the responsiveness pass with its
TaskRunner) has landed on origin/main. The TaskRunner likely supersedes
the BaseFitWorker follow-on — re-check at session ② start. Sessions ① and
③ are unaffected. The negative-muon implementation session (Sonnet, plan
committed) is disjoint from all four.

## DEFER — trigger-parked (triggers as recorded in the owning studies)

- **User-demand**: per-group AvCorr bit-parity; hyperfine-axis controls;
  low-field correlation; probe-γ GUI exposure; N6 MaxEnt-phase pull on the
  Fourier panel; single-fit cross-accumulation; `⊕` in the time-domain
  grammar; per-detector / MaxEnt RRF; auto-ν₀ tracking; period mapping in
  the integral/ALC path; Mantid error-model escape hatch; GUI average-seed
  batch mode; JSONL fit-log export; symmetric multi-period co-subtract
  beyond the reference case; scaled co-add surface.
- **Hard prerequisites**: negative-muon GUI promotion (real μ⁻ data + a
  user); live-run monitoring (beamline); F10 stateless-window refactor
  (next contract change); Sconv deconvolution (safe adjoint); getresults
  derived-quantities hook (concrete user need — owns an uncertainties
  story); hot reload (+ the plugins reentrancy item that travels with it).
- **Opportunistic niceties**: simulate extras (deadtime-distortion
  injection, two-period NeXus writer, event-mode/logs/PSI-ROOT writers,
  more instrument templates, archetype gallery additions, pull-batch
  mode); N0 single-histogram FFT input; field-scan moment windows;
  demodulation caching; TF-phase fine-t0 (beyond parity); WiMDA live
  spot-checks; MaxEnt reconstruction view-band abstraction; loader pulse
  metadata; stale-series auto-removal; fit_index default-x.
- **Hygiene basket** (bundle into a future cleanup or the UI-polish
  pass): t0-shift-loop unification; series-order-key into core;
  `_safe_float` consolidation; shared value±error formatter; moments
  capability Protocol; eligibility-predicate reuse; workspace-view seam;
  shared synthetic-signal test helper; grouping-dialog resolver tidy;
  chain-seeding fallback logging; strict from_dict on recoverable caches;
  core-level missing-component guard; plot-side quick binning control
  (explicitly UI-polish territory).
