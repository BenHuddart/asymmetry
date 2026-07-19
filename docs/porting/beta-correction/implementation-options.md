# Beta Correction — Implementation Options

Decisions marked **CHOSEN** were agreed with Ben in the 2026-07-19 planning
discussion (see README decision log).

## 1. Formula seam

**Option A — explicit `beta` keyword beside `alpha` (CHOSEN).**
`compute_asymmetry(forward, backward, alpha=1.0, beta=1.0)` and the same on
`compute_asymmetry_with_count_errors`, `binned_fb_asymmetry`,
`reduce_grouped_asymmetry`. Every call site that threads `alpha` explicitly
threads `beta` beside it. Greppable, symmetric with α, and lets the preview
override β independently for the β=1 compare ghost.

**Option B — read β from the grouping dict inside `binned_fb_asymmetry`.**
Least caller churn (the function already receives `grouping`), but creates an
asymmetric API (α explicit, β implicit), and the preview's β=1 ghost would
need a payload mutation — the exact bug class the correction-order study
banned. Rejected.

Call sites threading β (from `GroupedForwardBackward.beta` or the caller's
grouping read): `reduce.py::reduce_grouped_asymmetry`,
`representation/time.py::TimeFBAsymmetry`, `simulate.py::_reduce_histograms`,
`data/combine.py::reduce_combined_run`, `transform/integral.py`
(`integrate_asymmetry` + callers), `gui/mainwindow.py`
`_reduce_grouped_histograms_to_asymmetry`, `gui/windows/grouping/preview_pane.py`.
The loaders' initial reduction (`io/nexus.py:763`, `io/psi.py:898`,
`io/root.py:610`) hard-codes α=1 before any profile resolves and needs no β.

## 2. Persistence

**Option A — plain `beta: float` on `GroupingProfile` + `"beta"` payload key
(CHOSEN).** Emitted only when ≠ 1.0, so existing profiles/projects round-trip
byte-identically and lenient `from_dict` needs no schema bump — the
`t0_policy` precedent (`profiles.py:680–685`). Lenient read clamps
non-finite/≤0 to 1.0, mirroring the α read in `group_forward_backward`.

**Option B — a `BetaPolicy` dataclass mirroring `AlphaPolicy`.** Deferred:
with only a `fixed` mode it is ceremony without behaviour. Revisit when the
estimator lands (`calibrated` mode with method/source_run/error provenance
would then mirror α exactly). The payload key chosen now (`"beta"`, plus
future `beta_method`/`beta_error`/`beta_reference_run` companions) is
forward-compatible with that upgrade.

Instrument self-healing: scalar `beta` is instrument-independent (like scalar
`alpha`) and is **kept** on a heal — it is not added to
`_INSTRUMENT_DEPENDENT_STRUCTURE_KEYS`.

## 3. GUI

**Option A — separate blue β card (CHOSEN).** `STAGE_BETA` (+`_SOFT`) tokens,
a small `BetaSectionWidget` (value spin + explanation + compare toggle),
registered as a fourth `CorrectionCard` after α. Justified by
determination-independence: β shares nothing with α's estimate machinery in
v1. The pipeline strip gains a β chip after α (the strip already teaches
order; α and β both act at the asymmetry-formation step).

**Option B — extend the α card to "α / β (detector balance)".** Rejected for
v1: the α section is already dense (calibration picker, methods, vector
table, staleness), and β is a rarely-used default-1 knob. Revisit only if the
future estimator (which yields α and β together) makes a shared surface
compelling.

Preview compare: `compare_stage` gains `"beta"` — ghost = same corrected
counts re-binned at β = 1 (α as configured), exactly parallel to the α
compare's α = 1 ghost. β is scalar-only, so unlike α the compare stays
available in vector mode only if β applies there; v1 applies β to the F-B
asymmetry reduction (scalar path). Per-projection vector reductions keep
β = 1 — recorded limitation, revisit with the estimator.

## 4. Fitting (deferred)

Fittable β belongs in the count-domain models
(`core/fitting/count_domain.py::build_fb_count_model`,
`grouped_time_domain.py::FBCountModel`): forward `∝ N₀√α(1 + A·P(t))`,
backward `∝ (N₀/√α)(1 − β·A·P(t))`, `beta` a structural parameter defaulting
to fixed 1. Running that fit on a calibration run with β free **is** the β
estimator (β̂ = A₀,b/A₀,f), so the deferred estimator and deferred fittable-β
are one feature. Do not build until the expert consult on estimation
methodology returns.
