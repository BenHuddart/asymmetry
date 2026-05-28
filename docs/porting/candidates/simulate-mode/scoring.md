# Simulate mode: scoring

## Impact (1-5)

**Score: 4**

- *Breadth of user benefit:* very broad. Students, teachers, fit
  validation, cross-tool benchmarking — multiple distinct user
  modes are unlocked.
- *Pedagogical value:* the strongest in this roadmap. Asymmetry
  becomes a teaching platform without any external scripting.
- *Alignment with Asymmetry strengths:* the synthesis pipeline
  already exists for the screenshot infrastructure — promotion
  to first-class GUI feature is pure exposure.
- *Marketing value:* WiMDA's simulate mode is a well-known
  feature; Mantid and musrfit users routinely ask for it.

## Ease (1-5)

**Score: 4**

- *Registry / API readiness:* the synthesis helpers are already in
  `docs/screenshots/data/archetypes.py` and ready to promote into
  `core/simulate.py`.
- *Model complexity:* zero new physics. Pure UI + plumbing.
- *GUI surface required:* one new modal dialog. Reuses the existing
  Fit Function Builder for model selection.
- *Test-data availability:* round-trip validation is its own oracle
  (simulate → fit → recover parameters).
- *Risk:* low. Worst case is a clumsy dialog UX that needs polish.

## Score = impact × ease = **16**

Tier: **Now**. Recommend promoting alongside Dynamic KT, MINOS, and
the theory library expansion — the four most impactful low-friction
roadmap items.
