# Maximum Entropy Verification Plan

Staged to match the selected direction in `implementation-options.md`.

## Stage 0: Pre-Implementation Research Checks

Resolve before writing the core engine (open questions 1–3 in
`comparison.md`):

1. confirm `scripts/Muon/MaxentTools/` inner modules import with numpy alone;
   if not, generate golden `.npz` files once inside a Mantid environment
2. confirm Asymmetry's loaders/grouping preserve per-group raw counts,
   frames, t0, and good-bin metadata end to end
3. pin the provenance citations and the phase-sign decision (BACK vs MODAMP
   quirk) in an update to `comparison.md`

## Stage 1: Inner-Loop Parity Against The Oracle

- unit tests for the forward/adjoint pair: `tropus(opus(x))` adjointness
  check `⟨opus(x), y⟩ = ⟨x, tropus(y)⟩` on random vectors
- run the inner Skilling–Bryan loop on oracle golden inputs with identical
  iteration schedule, constants pinned to Mantid's, phases/backgrounds
  frozen: per-bin agreement of `f`, χ², entropy, TEST trajectories to tight
  tolerance (1e-8 relative — both are float64 numpy on the same formulas)
- property checks: spectrum non-negative (≥ negativity clamp), χ² decreases
  toward target across iterations, entropy finite

## Stage 2: Outer-Loop Behavior

- synthetic cases 1–6 from `test-data.md` with seeded RNG:
  phase recovery, two-peak resolution, fixed-phase identity contract,
  constant-background and deadtime recovery, dead-group safety
- continuation semantics: running N cycles then M more equals running N+M
  cycles (state resumability); changing a restart-required parameter raises
- diagnostics record contains per-cycle χ², entropy, TEST, sconv, phases;
  MOVE σ-tightening events are surfaced, not silent

## Stage 3: Reference Cross-Checks

- Mantid doc-test smoke values (five MUSR00022725 spectrum points at 5e-2)
  if the run is obtainable; otherwise oracle golden data stands in
- WiMDA `.max`/`.mlog` golden comparison: peak positions and widths on the
  shared test run; per-cycle log quantities follow the same trajectory
  (loose tolerance — constants and bunching differ)

## Stage 4: GUI And Harness

- panel smoke test: MaxEnt view token renders, run/+N/converge buttons drive
  the resumable state, spectrum lands in the `FrequencyMaxEnt`
  representation, time-domain overlay appears on the time tab
  (`QT_QPA_PLATFORM=offscreen`)
- project persistence round-trip of MaxEnt settings and (if stored) results
- `python tools/harness.py structural`, `lint`, targeted
  `test -- tests/test_maxent*.py`, then full `validate` before merge

## Completion Criteria For The Implementation Pass

- all five study docs updated with the final decisions and verification
  outcomes; `docs/porting/index.json` status advanced from `study`
- WiMDA-side map mislabels do not propagate: Asymmetry docs/tests must
  anchor on `Wimdamax.pas`, never WiMDA's `maxent-spectrum` FEATURE_MAP entry
