# Phase auto-calibration: scoring

## Impact (1-5)

**Score: 3**

- *Breadth of user benefit:* moderate. Reduces friction for TF
  workflows but doesn't unlock fundamentally new science.
- *Pedagogical value:* low-moderate. Phase calibration is plumbing,
  not physics.
- *Alignment with Asymmetry strengths:* the Multi-Group Fit
  infrastructure is already in place; this is a specialised
  application of it.

## Ease (1-5)

**Score: 3**

- *Registry / API readiness:* fit engine, multi-group fit, and
  Oscillatory model all exist.
- *Model complexity:* the algorithm itself is simple — the
  complexity is in robust handling of the failure modes (no clear
  frequency, near-zero amplitudes, mixed-frequency datasets).
- *GUI surface required:* moderate (button + result dialog +
  consumer wiring).
- *Risk:* moderate. Phase auto-calibration is a place where users
  can silently get wrong answers; need good diagnostics.

## Score = impact × ease = **9**

Tier: **Next**. Lands after MINOS / dynamic-KT / simulate / theory
library; a quality-of-life win once those are in.
