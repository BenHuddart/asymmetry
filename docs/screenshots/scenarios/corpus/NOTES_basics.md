# NOTES — Basics (calibration & data-handling primer)

Module: `basics_calibration.py` (auto-discovered). Corpus example: `Basics`.
Spec: `Basics/GROUND_TRUTH.md`. Corpus is read-only; data resolved through
`_corpus.load_corpus_datasets` / `ASYMMETRY_CORPUS_ROOT`.

Basics is *the* data-handling on-ramp, so these renders prioritise the
data-processing UI (grouping / α / dead-time / t0) over fitting, plus the B1
steering deliverable which exercises the fit-table + manual-column trend.

## Scenarios registered

| Name | Run(s) | What the render shows | GT § | Intended docs use |
|---|---|---|---|---|
| `corpus_basics_grouping` | MUSR00044989 | Grouping window: 64 detectors → 2 groups (32 each), Forward/Backward assignment, t0 "From file", live F/B asymmetry preview | A3 | Detector-grouping reference / "how to set F/B groups" |
| `corpus_basics_alpha` | EMU00018854 (Ag TF 100 G) | Alpha calibration dialog: Estimate → α = 0.885, before (α=1, grey) vs after (α̂, blue) asymmetry balanced about zero | A4 | α-calibration reference / "estimating detector balance" |
| `corpus_basics_deadtime` | emu00034998 (Ag, high rate) | F/B asymmetry, dead-time Off vs Auto-Load (file, silver-derived); ~5 % early-time correction | A2 | Dead-time correction concept ("create a plot to show the effect") |
| `corpus_basics_t0` | EMU00018850 (Ag TF, pulsed EMU) | Raw summed counts show the muon pulse; t0 (mid-pulse, 0.224 µs) and tgood (0.336 µs) marked from the file, ~112 ns tgood offset shaded | A1 / B3 | t0 / tgood concept ("timing origin & good-data window") |
| `corpus_basics_steering` | EMU 44989–44997 | Fit Parameters trending panel: Ag-mask a₀ (a-relaxin) vs steering current (**manual column**) with the fitted polynomial (**Model Fit** `Cubic`) overlaid — curve minimum at **I = −0.060 A**, the beam-centred current | B1 | Fit-table + manual-column trend + Model Fit; steering-curve worked example |

## Workflow followed (with GT references)

- **Grouping (A3, GT §A3/§C).** Opened `GroupingDialog` directly on
  MUSR00044989 (the pattern from `grouping_window_profile_editor.py`). The MuSR
  default is 2 groups of 32 detectors; the dialog shows the group table,
  Forward/Backward combos and the live asymmetry preview. Bunched the preview
  (factor 20) so the fine-binned late-time tail is legible instead of a wall of
  exploding error bars.
- **α (A4, GT §A4/§C).** Drove the real `AlphaCalibrationDialog` on
  EMU00018854 (Ag TF 100 G — a clean silver TF calibration run, the classic α
  reference). Corpus `groups` are 1-based; the dialog wants 0-based indices
  (it re-adds 1 internally), so I shift them down. Clicked **Estimate** and
  blocked on the worker thread until it lands (per `alpha_calibration_dialog.py`).
- **Dead-time (A2, GT §A2/§C).** The guide asks to load the silver run, view the
  asymmetry with correction **Off**, then **Auto Load** and "create a plot to
  show the effect." Implemented exactly that as a before/after overlay via the
  real `reduce_grouped_asymmetry` pipeline (`use_deadtime=False, mode="off"` vs
  `use_deadtime=True, mode="file"`). "Auto Load" ≡ file mode: emu00034998 carries
  per-detector silver-derived `dead_time_us` (96 values) in its NeXus header.
- **t0 / tgood (A1, GT §A1/§A0/§B3).** Plotted the summed raw counts near the
  pulse for EMU00018850 with markers at the stored `t0_bin`/`first_good_bin`.
  Values match GT §A0: t0_bin 14 → 0.224 µs, first_good 21 → 0.336 µs, offset
  ≈ 112 ns (GT: stored t0 ≈ 0.24 µs, "close to the guide's 0.3 µs guess").
- **Steering (B1, GT §B1/§C).** Loaded the **WiMDA reference output**
  `Basics/data/steering_curve.dat` (the a-relaxin ≈ a₀ per run) into the real
  `FitParametersPanel` as a trend series, and registered the transcribed
  steering-magnet current — *not logged in the EMU files* — as a **custom trend
  column** (`set_custom_x_fields` + select it on the X-axis combo). This is the
  fit-table/manual-column feature the guide's "Tip" describes. The guide's
  "fit a parabola; the minimum is the beam-centred current" step is then done
  with the panel's **Model Fit** machinery: a weighted iminuit fit of the
  `Cubic` polynomial component (injected via the `parameter_trending_mgb2.py`
  route) overlays the fitted curve on the points. `Cubic` rather than a pure
  quadratic because WiMDA's own reference curve (`steering_curve_fits.tab`)
  is a cubic (refit gives c3 ≈ 0.31) — a pure quadratic of the same 9 points
  puts the minimum at −0.095 A, while the cubic reproduces WiMDA's −0.06 A.

## Results vs ground-truth targets

| Quantity | GT target | Rendered value | Match |
|---|---|---|---|
| Grouping groups | 2 groups, 32 detectors each (1–32 / 33–64) | 2 groups × 32 detectors | ✓ (assignment order note below) |
| α (any TF cal run) | no printed target ("note the value") | α = 0.88487(33), diamagnetic, run 18854 | ✓ (qualitatively balances about 0) |
| Dead-time | no printed value; Off→Auto-Load "changes the plot" | visible ~5.2 % early-time shift (Off ≈ 17.8 %, Auto ≈ 23 %) | ✓ |
| t0 / tgood (EMU 18850) | t0_bin 15 / first_good 22 (§A0, 1-based); offset ≈ 0.1 µs | t0 0.224 µs (bin 14, 0-based), tgood 0.336 µs (bin 21), offset ≈ 112 ns | ✓ |
| B1 a₀(I) curve | parabola, min a₀ = 5.18 at I = −0.06 A; ≈7–8 at ±1 A | plotted points = steering_curve.dat exactly (min ≈5.17–5.19 near 0 A, 6.98/7.96 at ∓1/+1 A) | ✓ |
| B1 curve minimum (beam-centred current) | I = −0.06 A, min a₀ = 5.18 | fitted `Cubic` minimum **I = −0.060 A**, a₀(min) = 5.12 (fit c0=5.133, c1=0.280, c2=2.354, c3=0.233; χ²ᵣ = 0.31) | ✓ current exact; a₀(min) −0.06 low (weighting differences vs WiMDA) |

Only the steering scenario runs iminuit at capture time (the polynomial trend
fit → `requires_fit = True`); the α estimate is the algebraic diamagnetic
estimator and dead-time is a pure-core reduction, so the other four are
fit-free. All five ran fine on the env's numpy 2.2.6.

## Feature-demonstration opportunities spotted

- **Manual fit-table column** (`FitParametersPanel.set_custom_x_fields`) — used
  for steering; the single best demo of "add a column not logged in the files".
- **α before/after preview** — the calibration dialog draws the α=1 vs α̂ curves,
  a self-evident "what α does" visual (used).
- **Dead-time before/after** via `reduce_grouped_asymmetry`'s `use_deadtime`
  toggle — the cleanest way to reproduce the guide's "show the effect" plot (used).
- **B2 range curve** (quartz + Ti foils, EMU 18888–18899) and **B4 frequency
  response** (quartz TF, EMU 19626–19643) are *not* captured — both are
  deliverables with no worked answer on disk (GT §E) and would need 12–18 real
  per-run fits. They are natural follow-ups if per-run batch fitting is wired in;
  B4's muonium-amplitude roll-off would make a striking trend render.
- **Model Fit polynomial trend overlay** — used for steering (the `Cubic`
  component, injected as a real `ParameterModelFit` per
  `parameter_trending_mgb2.py`). Note: the Model Fit registry has no dedicated
  *Quadratic* component (Linear, Cubic, Quartic, quintic Polynomial, Sextic);
  a pure parabola needs `Polynomial` with c3–c5 fixed at 0. A first-class
  Quadratic would be a small, teachable addition — recorded as a feature
  opportunity (worked around here, and the cubic is what WiMDA itself used).

## Problems hit / caveats

- **Grouping group-order.** The dialog shows Group 1 = detectors 33–64 and
  Group 2 = 1–32, i.e. the reverse of GT §A3's "Group 1 = 1–32 = Backward".
  This is the loader/preset's forward/backward convention for this file, not a
  scenario bug; the F/B *pairing* (and thus the asymmetry) is still correct. The
  ID lists are the teaching point, so this is cosmetic — flagged for the doc author.
- **MUSR grouping run reads as ZF but the dialog shows a transverse-field nudge**
  banner ("Transverse-field run: … apply 'Transverse (Vector)'"). GT §A0 already
  flags that MUSR00044989 is zero-field (field = 0 G) even though the guide calls
  it "a transverse field measurement"; the app's TF heuristic misfires here. Left
  as-is (honest program output); worth a caption note in docs.
- **Concurrent captures don't mix.** Running two `capture_corpus` processes at
  once (offscreen QPA) deadlocks/stalls; capture the Basics set in a single
  process. Not a scenario bug — an offscreen-Qt / shared-output-dir contention.
- **Corpus filename casing.** The dead-time and steering EMU runs are lowercase
  on disk (`emu00034998.nxs`, `emu00044989.nxs`); the t0/α/grouping runs are
  uppercase. Paths are spelled to match the actual files.
