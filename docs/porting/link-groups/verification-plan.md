# Link groups — verification plan

## Automated (in-repo, CI)

Synthetic damped-cosine triplet (see test-data.md). All in
`tests/test_link_groups.py` (core/engine/serialization) and
`tests/test_fit_panel_tabs.py` (GUI single-fit table):

1. **Equality after fit** — followers equal their group main; follower σ equals
   main σ.
2. **Free set excludes followers** — `ParameterSet.free_parameters` and the
   engine's fitted count both exclude followers; reduced-χ² uses the reduced
   count.
3. **Frequencies recovered** — three free frequencies return symmetric about
   `f₀`; `f₊ − f₋ ≈ 0.242` MHz; χ²ᵣ ≈ 1.
4. **`.asymp` round-trip** — a project with link groups saves and reloads with
   `link_group` preserved.
5. **GUI** — assigning a Link group in the single-fit table feeds through to the
   fit (follower drops out of the free set; follower row shows main's value/σ).
6. **Backward compatibility** — a project/parameter dict without `link_group`
   loads as `None`.

Run: `python tools/harness.py validate` from the worktree venv.

## Manual acceptance (CdS real data, corpus only)

On the CdS 5.2 K run, fit three `Oscillatory * Exponential + … + Constant`
lines; link the three relaxation rates (and satellite amplitudes/phases as
appropriate); leave frequencies free.

Pass criteria:

- converged fit, χ²ᵣ ≈ 1.3
- satellites symmetric about the central line; central `f ≈ 1.389` MHz
- splitting `2δ ≈ 0.242` MHz (the hyperfine constant)
- linked parameters absent from the free-fit set, reporting propagated
  uncertainties

Recorded against the engine harness in the testing worktree; corpus files are
never committed.

## Outcome

Implemented: `Parameter.link_group` + `ParameterSet.link_groups/link_main/
link_followers`; `FitEngine.fit` substitutes followers and propagates the main's
uncertainty; a "Link" column in the single-fit table; `link_group` round-trips
through `.asymp`. Synthetic suite (`tests/test_link_groups.py`) + GUI/serialization
tests green.

CdS engine acceptance reproduced on the real 5.12 K run (EMU00020721, TF 100 G),
fitting three free-frequency `Oscillatory*Exponential` lines + `Constant`, with
the three relaxation rates linked (group 1), the two satellite amplitudes linked
(group 2), and the three phases linked (group 3), window 0.1–10 µs:

- converged, **χ²ᵣ = 1.35**
- central **f = 1.3889 ± 0.0003 MHz**
- satellites at 1.2684 / 1.5099 MHz → **splitting 2δ = 0.2416 MHz**, centre
  1.3892 MHz (symmetric about the central line)
- 8 free parameters of 13 (five followers dropped out); each follower equals its
  group main and reports the main's propagated uncertainty

Equal spacing is recovered from the data with free frequencies, exactly as WiMDA
does it — no offset tie required.
