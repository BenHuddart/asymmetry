# Python user functions: scoring

## Impact (1-5)

**Score: 4**

- Removes a class of "Asymmetry can't do X" complaints — users no
  longer need to fork the codebase to fit unusual models.
- Marketing value: this is a genuinely better extensibility story
  than musrfit's or Mantid's; could attract contributors.
- Aligns with Asymmetry's "modern Python stack" positioning.

## Ease (1-5)

**Score: 3**

- Decorator + filesystem scan is ~100 lines.
- Validation harness needs careful design (failure modes are
  surprising — wrong signature, missing param, returns wrong
  shape).
- Documentation lift is significant (a worked example, a
  troubleshooting section, a security note).
- GUI integration is trivial (categories already supported).

## Score = impact × ease = **12**

Tier: **Next**. The decorator design has to land *after* MINOS,
dynamic-KT, simulate, and theory-library expansion so the user-facing
example library is rich enough to make the plugin API attractive.
