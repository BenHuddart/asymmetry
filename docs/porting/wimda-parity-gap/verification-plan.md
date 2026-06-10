# Verification plan — WiMDA parity gap study

Two layers: verifying this study's *claims* (the inventory is the product),
and the standing verification frame each project inherits.

## 1. Verifying the inventory itself

The inventory's risk is a **missed capability** (a missed project) or a
**wrong status** (porting something that exists / skipping something that
doesn't). Checks performed:

- **Coverage**: every unit in `wimda.dpr`'s `uses` clause was assigned to
  exactly one research slice; a dedicated sweep confirmed no unit was left
  uncovered (`PlotModel`, `SaveRange`, `FromTo`, `SetALCthresh`,
  `Rebinning`, `Resampling`, `BGform` were caught this way). The WiMDA menu
  tree (`WiMDA_Main.dfm`) was enumerated item-by-item as an independent
  feature surface.
- **Status claims**: every PRESENT/PARTIAL is cited to an Asymmetry file;
  ABSENT claims were grep-verified (e.g. no `rrf`, no `find_t0`, no Burg/AR
  code anywhere in `src/asymmetry`).
- **Known-stale-source guard**: the old comparison matrix was not trusted;
  Asymmetry status came from a fresh code scan on the synced branch
  (`474534e`). Two prior-roadmap errors were corrected this way (Eigen.pas
  misclassification; "no ALC GUI" staleness).
- **Residual risks, accepted**: Delphi event-handler spaghetti means a
  behaviour reachable only through an obscure UI path could still hide;
  each project's study pass re-reads its WiMDA slice at full depth, which is
  the backstop. The `__history`/`__recovery` exclusion is deliberate (backup
  noise, not reachable code).

## 2. Standing verification frame for the projects

Each project's own study pass produces its detailed verification-plan.md.
Portfolio-wide rules they inherit:

1. **Behavioural contract**: WiMDA arithmetic transcribed into tests where
   parity is the goal (`tests/test_wimda_parity_components.py` precedent);
   deliberate deviations (physically-correct replacements) documented in the
   project's comparison.md with both curves plotted.
2. **Oracles**: Mantid/musrfit GPL code as oracle only, never vendored.
   WiMDA itself has known bugs to avoid oracling against (`KTBArray`
   low-field branch; vestigial dead forms).
3. **Self-consistency**: synthetic data with known ground truth wherever the
   quantity has a closed form (moments, simulate round-trip, RRF
   demodulation, alpha estimators).
4. **Gate**: `python tools/harness.py validate` green per phase; GUI work
   additionally `gui-smoke`; docs-only changes `docs`.
5. **Corpus regression**: the previously-verified corpus results (CdS
   χ²ᵣ=1.35, EuO β, EMU repolarisation, photo-μSR periods) must not regress
   when reduction-layer projects (1, 2) change shared kernels.

## 3. Refresh / bookkeeping

- When a project starts: promote its brief to a full
  `docs/porting/<slug>/` study, add its `index.json` entry, link back here.
- When a project ships: update the portfolio table in README.md and the
  relevant rows of comparison.md (this study's tables are dated and
  versioned by branch — keep them honest the same way the old matrix was
  supposed to be).
- The old `docs/porting/comparison-matrix.md` WiMDA column is superseded by
  this study; a pointer note was added there rather than rewriting it.
