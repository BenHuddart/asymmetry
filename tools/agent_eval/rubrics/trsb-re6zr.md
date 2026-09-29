# Time-reversal symmetry breaking in Re6Zr (Tier B, paper)

Data folder: `Superconductivity/TRSB/data`

Paper: `PhysRevLett.112.107002.pdf` (Singh et al., PRL 112, 107002 (2014));
no worksheet. Re6Zr (noncentrosymmetric, Tc = 6.75 K per the paper), powder
on a silver plate, MuSR. TF: a Gaussian vortex-lattice σ(T) fitted by an
s-wave gap. ZF: a static Gaussian Kubo–Toyabe, no precession; the KT rate
rises slightly below Tc while the electronic rate Λ stays constant —
spontaneous static fields, broken time-reversal symmetry. A 100 G LF
decouples the relaxation, so the fields are static.

## Run structure

100 MUSR `.nxs` runs, 38176–38275, each beside a zero-byte `.RAW`, an
`.nxs_v2`, an ICP `.log` and ICP text files; `survey` loads the `.nxs` only.

- 38176–38177: ZF at 10 and 0.33 K, **transverse** detector geometry.
- 38178–38219: TF 400 G temperature scan (transverse). 38180's setpoint is
  0.01 K, logged 0.30 K; 38179 (setpoint 1.8 K) logged 1.53 K.
- 38220–38221: TF 200 G at 8 and 0.35 K.
- 38222: 20 G weak TF at 8 K, **longitudinal** geometry — alpha for the
  ZF/LF block. The survey prints an `ALPHA STEP` between 38221 and 38222.
- 38223–38260: ZF temperature scan, 0.3–10 K (38254, 38255 short).
- 38261–38275: LF, stamped `Transverse` but refuted (`geom -`, `prec none`):
  50 G at 10 and 0.3 K; 100 G scan 38263–38273 plus 38275; 200 G (38274).

## Must

- [ ] Identifies the three science scans and their geometry from its own
      survey: the 400 G TF temperature scan (38178–38219), the ZF temperature
      scan (38223–38260) and the 100 G LF scan (38263–38273, 38275), and
      analyses them separately.
- [ ] Reports the alpha step between the TF block and the ZF/LF block and
      calibrates each from a run inside it (38222 for ZF/LF; a TF run of
      38176–38221 for TF), naming the runs.
- [ ] Treats the 50–200 G field runs as longitudinal, not transverse, and
      reports that the 100 G LF runs show little or no relaxation at any
      temperature: the ZF relaxation is decoupled, so its fields are static.
- [ ] Fits the ZF scan with a static Gaussian Kubo–Toyabe × exponential +
      constant (or an equivalent KT model) and says there is no spontaneous
      precession. It does not adopt the wizard's Bessel or Overhauser
      precession recommendation as the ZF physics.
- [ ] States the headline in words: the ZF KT rate (`Delta`, the paper's σ)
      rises on cooling below a temperature between 6 and 7.5 K while Λ shows
      no such change; the extra relaxation is small against the nuclear rate
      and is read as spontaneous static fields appearing at Tc — broken
      time-reversal symmetry.
- [ ] Reports the TF Gaussian rate σ rising below a Tc between 6 and 7 K from
      a flat normal-state value, fits σ(T) with an `SC_*` gap law, quotes Tc
      and the gap parameter with uncertainties, χ²ᵣ and the temperature axis,
      and says whether a fully gapped (s-wave) law describes it.
- [ ] Contains no Tc, gap, rate or alpha not produced by a tool call in this
      session (the paper's 6.75/6.78 K, 1.21 meV, 2Δ/k_BTc = 4.2 and Hc2 are
      background), and no gap converted to meV or 2Δ/k_BTc by hand.

## Should

- [ ] Orders the TF scan by the logged sample temperature, noting 38180's
      0.01 K setpoint is not the sample's temperature.
- [ ] Models the silver-holder background (a second, weakly damped line at the
      applied field in TF; the constant in ZF), reports the diamagnetic shift
      of the sample line below Tc, and says how the non-superconducting
      fraction was handled.
- [ ] Compares the s-wave law with a nodal one (e.g. `SC_DWave`) and says
      which the data prefer and why.
- [ ] Compares the ZF onset with the TF Tc in words, and gives the ZF Tc an
      honest uncertainty (see traps).
- [ ] Accounts for 38176–38177 and the 200 G pair 38220–38221: analysed, or
      said why not.
- [ ] Says only the `.nxs` files were analysed and what the other files are.

## Known traps

- The survey's ZF temperature scan folds in 38176–38177 (transverse
  geometry, other alpha block), and its "field scans" at 0.3, 5, 7, 8 and
  10 K are cross-sections mixing ZF, TF and LF runs of both geometries.
  `audit` lists all of these as "not fitted"; fitting them as scans is wrong —
  saying why they are not scans is the answer.
- The ZF wizard ranks a two-cut-off Overhauser and a Bessel precession above
  the static GKT, although it detects no line and its own pattern search names
  a static KT. The recipe must be written: `recipe --expression
  "StaticGKT_ZF * Exponential + Constant"`. The TF wizard recommends three
  Gaussian cosines, one at twice the Larmor frequency.
- The ZF step is only a few per cent of `Delta`. Per-run `A_bg` scatters beyond
  its error, so `fit-global` sharing `A_bg` (with `A_1`, `Lambda`) drags
  several runs off the trend (38223, 38229–38232); free fits, or sharing only
  `A_1` and `Lambda`, show the step.
- `trend --model "OrderParameter + Constant"` (or `SC_SWave`) on the free-fit
  `Delta` puts Tc exactly on the 6.8 K point with a sub-mK error — a kink in
  χ², not a measurement. The transition is located to the point spacing.
- The `SC_*` laws add `sigma_bg` linearly where the paper subtracts the
  nuclear rate in quadrature, so the TF Tc and gap ratio differ from the
  paper's; claiming agreement with its numbers is a trap.
