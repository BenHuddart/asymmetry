# RRF transform: scoring

## Impact (1-5)

**Score: 3**

- *Breadth of user benefit:* moderate. Vortex-lattice, Knight-shift,
  and high-TF studies use RRF routinely; low-TF and ZF studies do not.
- *Pedagogical value:* high in its niche — the textbook chapter 8/9
  vortex-lattice discussion assumes the reader understands RRF.
- *Alignment with Asymmetry strengths:* RRF is a transform applied to
  asymmetry, which `core/transform/` already hosts (asymmetry, rebin,
  background, deadtime, grouping). Fits naturally.

## Ease (1-5)

**Score: 4**

- *Registry / API readiness:* clean drop into `core/transform/rrf.py`.
- *Model complexity:* low — ~30 lines numpy plus a low-pass.
- *GUI surface required:* small. One toolbar toggle and a reference-
  frequency spinner; reuse the existing reference-field metadata.
- *Test-data availability:* cross-validate against Mantid's `RRFMuon`
  on a synthetic vortex-lattice signal.
- *Risk:* low. Phase-sign convention is the main pitfall.

## Score = impact × ease = **12**

Tier: **Next**. Lands well after the higher-impact items but before
the heavier ones.
