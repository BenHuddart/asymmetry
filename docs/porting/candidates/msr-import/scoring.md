# `.msr` import: scoring

## Impact (1-5)

**Score: 3**

- Opens Asymmetry to the entire musrfit-using community without
  forcing them to redo their setup.
- Cross-tool benchmarking story is materially stronger once
  identical inputs can be loaded into both tools.
- Limited by the fact that users don't switch tools daily; this is
  a one-time conversion benefit per user.

## Ease (1-5)

**Score: 3**

- `.msr` format is plain text with a modest grammar — parsing is
  ~200 lines.
- Theory-name mapping is the bulk of the work and depends on the
  theory-library-expansion candidate landing first (otherwise
  imports drop several musrfit functions).
- No GUI work needed: just a File → Open Project that detects
  `.msr` extension.
- Risk: undocumented `.msr` syntax variants in the wild. Mitigate by
  validating against the musrfit test corpus before claiming
  general support.

## Score = impact × ease = **9**

Tier: **Next**. Best landed after theory-library-expansion so the
import doesn't silently lose functionality.
