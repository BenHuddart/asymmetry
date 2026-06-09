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

_Filled in during the implementation pass:_ engine + GUI + serialization landed;
synthetic suite green under `validate`; CdS engine acceptance reproduced
(central f, 2δ, χ²ᵣ as above).
