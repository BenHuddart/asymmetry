# WiMDA parity programme — COMPLETE

Date: 2026-06-13. This is the authoritative final status of the WiMDA
functionality-parity programme. It supersedes the per-document status
notes (the [README](README.md) status block, [wave-a-closeout.md](wave-a-closeout.md),
[follow-on-triage.md](follow-on-triage.md)) for the question "is it done?".
The [decision-record.md](decision-record.md) remains the durable reference
for *what was decided and why*; this document records *that the programme
closed and on what terms*.

## Verdict

**Functional parity is complete.** Every capability the gap study identified
as portable WiMDA functionality is now in Asymmetry, plus several
beyond-WiMDA improvements the ports surfaced. No parity work remains open.
What is not shipped is the **trigger-parked tier** — items deliberately
deferred against a recorded trigger (user demand, beamline access, a safe
numerical formulation, or a concrete second use case). "Deferred" here means
*intentionally awaiting a trigger*, not *unfinished*.

The governing rule held throughout: **parity of functionality, not of
implementation** — modern numerics and physical correctness over literal
ports, every divergence documented with both behaviours.

## What "complete" rests on

- All planned projects merged; no open parity PRs at closeout.
- The one physics-correctness risk the programme flagged in shipped code —
  the MuoniumLFRelax `(1−δ)` prefactor, carried over from WiMDA but absent
  from Kadono — was checked against the source and **removed as spurious**
  (#71). `muonium_lf_relaxation` is now the clean BPP/Redfield rate
  `λ(B) = δ_ex²τ_c/(1+(ω₁₂τ_c)²)`. The remaining non-textbook sources were
  signed off in the same pass.
- The asymmetry-error testing-worktree goldens were audited against the
  exact-Poisson change and regenerated where stale (recorded on
  `testing/wimda-eval`; decision-record §asymmetry-error-propagation).
- WiMDA-side bugs found during the programme are catalogued in
  decision-record §3 as "do not oracle against" — they outlived the port as
  a verification hazard list.

## PR ledger

Foundations and Wave A (analysis-capability parity):

| PR | Project |
|---|---|
| #29 | Muonium oscillation components (groundwork) |
| #30 | WiMDA time-domain fit-function parity |
| #31 | Umbrella gap study (12-project portfolio) |
| #32 | model-function-parity |
| #33 | simulate-mode |
| #34 | data-reduction-parity |
| #35 | asymmetry-error-propagation (emergent: exact Poisson σ_A) |
| #36 | data-reduction follow-ons |
| #37 | simulate follow-ons |
| #38, #39 | model-fit follow-ons (arbitrary-X, error modes, ⊕, recursion) |
| #40 | maxent-completion |
| #41 | count-domain-fit-modes |
| #42 | frequency-domain-finishers (incl. Burg diagnostic) |
| #43 | radical-correlation-spectrum (emergent, promoted from #42) |
| #44 | Wave A strays (two-period/count-mode simulate; Fourier tail-fit) |
| #45 | Wave A closeout docs |

Collision reconciliation:

| PR | Phase |
|---|---|
| #46 | reconciliation study (19-flag verdicts + plan) |
| #48 | Phase 1 — mechanical UNIFYs (N1/N2/F2/F12) |
| #50 | Phase 2 — calibration promote family (core/transform/promote.py) |
| #49 | Phase 3 — frequency-panel UX |
| #47 | Phase 4 — documentation package |
| #51 | Phase 5 — trending decorations into FitSeries.extra |

Wave B (workflow machinery):

| PR | Project |
|---|---|
| #54 | spectral-moments |
| #55 | run-arithmetic (histogram-level co-add/co-subtract) |
| #56 | fit-workflow-diagnostics (MINOS, χ² verdict, chain-seeding, abort) |
| #57 | rrf (complex demodulation + frequency-offset fits) |
| #58 | python-user-functions (plugin registration + discovery) |

Post-Wave-B follow-on tranche (the NOW tier of [follow-on-triage.md](follow-on-triage.md)):

| PR | Session |
|---|---|
| #69 | negative-muon-analysis (API-only, experimental/WIP) |
| #71 | physics sign-off + quick wins (MuoniumLFRelax fix + 4) |
| #72 | RRF finish + engine pass (Advanced-gated RRF fit, fgAll→Poisson) |
| #73 | workflow-visualisation (Wave C; ADOPT/ADAPT verdicts) |
| #75 | batch-workflow parity (in-batch co-add, re-fit-coadded, MaxEnt batch) |

Adjacent work that ran within the same window but is *not* parity-programme
scope (recorded so the ledger is unambiguous): #52 public-release prep,
#53/#60–#68 GUI appearance/responsiveness/packaging, #59/#67 docs
infrastructure, #70 async project-result load, #74 unified asymmetry
projections.

## What remains — trigger-parked by design

The full ledger with triggers is in
[follow-on-triage.md](follow-on-triage.md) §DEFER. Summary of the trigger
classes (none is a parity gap):

- **User-demand**: per-group AvCorr bit-parity; hyperfine-axis controls;
  low-field correlation; probe-γ GUI exposure; N6 MaxEnt-phase pull;
  single-fit cross-accumulation; ⊕ in the time-domain grammar; per-detector
  / MaxEnt RRF; auto-ν₀ tracking; period mapping in the integral/ALC path;
  Mantid error-model escape hatch; GUI average-seed batch mode; JSONL
  fit-log export; symmetric multi-period co-subtract beyond reference;
  scaled co-add surface.
- **Hard prerequisite**: negative-muon GUI promotion (real μ⁻ data + a
  user — the API-only core shipped in #69 is the WIP placeholder); live
  current-run monitoring (beamline access); F10 stateless-window refactor
  (next time the window's data contract changes); Sconv deconvolution (a
  regularised adjoint); getresults derived-quantities hook (a concrete user
  need to fix its uncertainties story); hot reload.
- **Opportunistic niceties**: simulate extras (deadtime-distortion
  injection, two-period NeXus writer, event-mode/log/PSI-ROOT writers, more
  instrument templates, archetype-gallery additions, pull-batch mode); N0
  single-histogram FFT input; field-scan moment windows; demodulation
  caching; TF-phase fine-t0; WiMDA live spot-checks; MaxEnt reconstruction
  view-band abstraction; loader pulse metadata; stale-series auto-removal;
  fit_index default-x.
- **Hygiene basket** (bundle into a future cleanup / UI-polish pass):
  t0-shift-loop unification; series-order-key into core; `_safe_float`
  consolidation; shared value±error formatter; moments capability Protocol;
  eligibility-predicate reuse; workspace-view seam; shared synthetic-signal
  test helper; grouping-dialog resolver tidy; chain-seeding fallback
  logging; strict from_dict on recoverable caches; core-level
  missing-component guard; plot-side quick binning control.

## Permanent non-goals (settled)

From decision-record §1 — recorded so they are never re-litigated as gaps:
Eigen.pas eigensolvers (mislabel; superseded by `np.linalg.eigh`);
Kramers–Kronig (optical spectroscopy, not μSR); HDF4 `.nxs` (coverage
boundary); legacy file formats / zip loading / printing / in-app GLE editor
/ ARGUS-KEK hardware fixers / dead WiMDA forms; TRIUMF MUD (WiMDA's own
support is a non-functional stub — stays on the general roadmap, not a
parity item).

## If the programme is ever reopened

A trigger fires (a user asks, μ⁻ data arrives, a beamline test slot opens).
The path is unchanged: pull the item from the §DEFER ledger, run a
study-first pass under `docs/porting/<slug>/`, and follow the same
worktree-isolation + review + verification discipline the programme used
throughout. The decision-record's divergence ledger and WiMDA-bug list are
the standing context any such session should read first.
