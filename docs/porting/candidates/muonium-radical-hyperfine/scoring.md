# Muonium-radical hyperfine: scoring

## Impact (1-5)

**Score: 3**

- High for the chemistry / semiconductor μSR community
  (Amato-Morenzoni Ch 7 is mainly about muonium in
  semiconductors).
- Low for magnetism / superconductivity users.
- Mantid is the only reference tool with full coverage; porting
  closes a real gap.

## Ease (1-5)

**Score: 3**

- Each model is independently portable; the analytical Breit-
  Rabi forms are well-documented.
- Depends on the `theory-library-expansion` candidate landing
  first so the registry has the right shape.
- No new GUI surface needed.
- Risk: hyperfine constant conventions vary across the
  literature; document the chosen convention explicitly.

## Score = impact × ease = **9**

Tier: **Next**. Lands after the theory-library expansion.
