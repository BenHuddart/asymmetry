# LEM depth profiling

**Status:** candidate. Surfaced by practical-workflow #12.

## What

Helpers for low-energy muon (LEM) depth-resolved measurements:
implantation-energy → mean-depth conversion plus a parameter
trending mode where the *x* axis is depth rather than temperature
or field.

## Why

- LEM is PSI's flagship technique for thin-film and surface μSR.
- The Amato-Morenzoni textbook devotes all of Ch 8 to LEM.
- Without the depth conversion, users currently have to do this
  step externally (TrimSP or Monte Carlo).

## Prior art

- **PSI Mantid plugins**: external integration with TrimSP.
- **musrfit**: external workflow.
- **WiMDA**: ❌.

## Roadmap position

The depth conversion is small. Coupling to TrimSP is larger and
needs a clear LEM data path. Suggest **Later** tier; revisit if
LEM users show interest.
