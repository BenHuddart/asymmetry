# Muoniated-radical correlation spectrum — comparison

WiMDA → Asymmetry, feature by feature, for the muonium-radical correlation
spectrum (WiMDA's `Corr` / `AvCorr` Fourier modes and the `rmatch` Breit–Rabi
pair-matching). Each section quotes the WiMDA Pascal (file:line, from
`$WIMDA_SRC/src`, ignoring `__history/`, `__recovery/`),
states the Asymmetry target, and records every divergence with **both**
behaviours. Parity of *functionality*, not implementation: modern numerics and
physical correctness win where they tension with WiMDA.

This study **promotes** the deferred follow-on recorded in
[`frequency-domain-finishers/comparison.md` §9](../frequency-domain-finishers/comparison.md)
and confirms that brief's transcription against the Pascal source directly,
against the textbook Breit–Rabi relations (§4.4), and against the modern
radical-µSR literature (McKenzie 2013; Pratt 2000).

References (APS style, collected):

- B. D. Blundell, A. De Renzi, T. Lancaster and F. L. Pratt (eds.), *Muon
  Spectroscopy: An Introduction*, Oxford University Press (2022). [§4.4 two-spin
  muonium / Breit–Rabi; §12.4 muoniated radicals; §19.4 muon-spin (RF)
  resonance.]
- I. McKenzie, *Annu. Rep. Prog. Chem. Sect. C* **109**, 65 (2013). [Modern
  radical-µSR review: TF line pair → A_µ, TF-vs-ALC complementarity.]
- F. L. Pratt, *Physica B* **289–290**, 710 (2000). [WiMDA, by its author; names
  the radical correlation spectrum as a Fourier-window option.]
- S. J. Blundell *et al.*, *Nat. Rev. Methods Primers* **1**, 89 (2021).
  [Breit–Rabi / radical orientation.]
- I. McKenzie *et al.*, *J. Phys. Chem. B* **117**, 13614 (2013). [Cyclohexadienyl
  hyperfine parameters, used as the worked example.]

---

## 1. What the correlation spectrum is, and why it exists

A muon that **adds to an unsaturated bond** (a C=C double bond, an aromatic
ring, a C=O group) produces a **muoniated radical**: the muon sits at a
β-position and its spin couples, through the molecule's unpaired electron, to
that electron with an **isotropic hyperfine coupling A_µ** (McKenzie 2013, §1.1,
Fig. 1; textbook §12.4). In a **transverse field** the radical precesses, and
because the muon–electron system is a two-spin (Breit–Rabi) system the radical
appears in the FFT spectrum as a **pair of precession lines**, not one. The
spacing of that pair encodes A_µ.

The correlation spectrum is a transform of the ordinary TF FFT power spectrum
that **collapses each genuine radical line-pair onto a single peak at the
hyperfine-coupling value A_µ**. It is, in effect, a matched filter for
"two lines that are a Breit–Rabi pair", swept over the spectrum. It is the
standard frequency-domain route to identifying a muoniated radical and pinning
its A_µ.

Pratt 2000 (the WiMDA paper, by the feature's author) records only that the
Fourier window provides "diamagnetic subtraction, filtering and radical
frequency correlation spectra … to assist in muonium radical spectroscopy"
(p. 710, point 12). The mathematics is in the WiMDA Pascal and is reconstructed
and verified below.

---

## 2. The Breit–Rabi line pair — the physics behind `rmatch`

### 2.1 Three independent sources agree

For an isotropic two-spin (muon+electron) system with hyperfine coupling `A`
(in frequency units, MHz) in a transverse field `B`, the **two high-field
("Paschen–Back") precession frequencies** ν₁₂ and ν₃₄ obey the exact relations:

**Textbook §4.4, eqn 4.54** (here `x = B/B₀`, `B₀ = ω₀/(2π·2γ₊)`,
`ζ = γ₋/γ₊ = 0.99037`):

```
ν₁₂ = (A/2)·| 1 + ζx − √(1+x²) |
ν₃₄ = (A/2)·( 1 − ζx + √(1+x²) )
```

**Textbook §4.4, eqn 4.65 (high TF):**  `A = ν₁₂ + ν₃₄`  — the pair **sum** is
the hyperfine coupling.

**McKenzie 2013, eqns 6–8** (signed convention; ν_e, ν_µ the electron and muon
Larmor frequencies):

```
ν₁₂ = ν_mid − ½A_µ
ν₄₃ = ν_mid + ½A_µ              ⇒  A_µ = ν₄₃ − ν₁₂   (pair difference)
ν_mid = ½[ √(A_µ² + (ν_e+ν_µ)²) − ν_e + ν_µ ]
```

These look different (sum vs difference) but are the **same relation**: in
McKenzie's signed convention **ν₁₂ is negative** at the fields used (she notes
"the ν₁₂ line is negative" at 2.9 and 14.5 kG, Fig. 3), so
`ν₄₃ − ν₁₂ = ν₄₃ + |ν₁₂| = A_µ`. The textbook's "sum" uses `|ν₁₂|`.

**`core/fitting/muonium.py`** already ports textbook eqn 4.54 exactly:
`_tf_levels(field, A_hf)` returns the four levels with
`w12 = |E₁−E₂|`, `w34 = |E₃−E₄|`, the mixing `δ = x/√(1+x²)`, and
`x = (g_e+g_µ)·B/A_hf`. Its `ζ`-equivalent is
`(g_e−g_µ)/(g_e+g_µ) = 0.99037` — **identical to WiMDA's `dg`** and the
textbook's `ζ`. The constant `g_e+g_µ` (CODATA, via `units.py`) equals WiMDA's
`gg = 2.8024 + 0.01355342`.

**Numerical verification** (this study, `.venv` + `muonium.py`):

| B (G) | A (MHz) | w₁₂ (MHz) | w₃₄ (MHz) | w₁₂+w₃₄ |
|------:|--------:|----------:|----------:|--------:|
| 1000  | 500     | 214.4241  | 285.5759  | 500.0000 |
| 2000  | 514     | 218.1893  | 295.8107  | 514.0000 |
| 3000  | 330     | 121.1170  | 208.8830  | 330.0000 |
| 5000  | 1200    | 506.7091  | 693.2909  | 1200.0000 |

`w12 + w34 = A` to machine precision in every case. This is the relation the
correlation hyperfine axis is built on.

### 2.2 WiMDA's `rmatch` is an approximate inverse

`rmatch(freq, field)` (`Plot.pas:515-523`) maps an observed line to its
Breit–Rabi partner in closed form:

```pascal
function rmatch(freq, field: double): double;
begin
  wemp := sqr(2.81555 * field);     { (≈(g_e+g_µ)·B)²       }
  wemn := 1.394225 * field;         { ≈ ½(g_e−g_µ)·B        }
  wplus := 4 * (wemn - freq);
  rmatch := -(wemn - wemp / wplus);
end;
```

WiMDA also defines `w12`/`w34` (`Plot.pas:525-551`) as cursor read-outs — these
are textbook eqn 4.54 verbatim (`dg = 0.99037`, `gg = 2.8024 + 0.01355342`).

**What `rmatch` does** (this study, reconciled numerically). Fed the *upper*
line of a pair (the more positive |ν|, i.e. `−w34` in WiMDA's negative-frequency
scan), it returns the *lower* partner `≈ −w12`, so the hyperfine axis
`|f₁+f₂| ≈ w12 + w34 = A`:

| input            | `rmatch` output | `|f₁+f₂|` | true A |
|------------------|----------------:|----------:|-------:|
| −w34 (B=1000,A=500) | −214.4242     | 500.0001  | 500    |
| −w34 (B=5000,A=1200)| −506.7366     | 1200.0275 | 1200   |

Fed the *lower* line (`−w12`) it returns a partner that does **not** match the
upper line — `|f₁+f₂|` lands away from A and into the spectral noise. This
one-sidedness is **intentional**: WiMDA's generation loop scans every bin above
the diamagnetic line as a *candidate upper line*, so a real pair contributes a
peak at A while a non-pair contributes background.

**Divergence (rounded constants).** `rmatch` uses `2.81555` and `1.394225`,
which differ from the CODATA values WiMDA itself uses elsewhere
(`g_e+g_µ = 2.81605`, `½(g_e−g_µ) = 1.394471`) at the **5th significant figure**.
Combined with `rmatch` being a high-field expansion, the recovered A drifts by
~0.01–0.03 MHz over A ≈ 500–1200 MHz. **Asymmetry:** does not transliterate
`rmatch`; it builds the spectrum by the **exact forward map** (§4 below), so A
is recovered to machine precision and the constants come from the single
`units.py` source. WiMDA's `rmatch` is retained only as a transcribed test
oracle (verification-plan §3) documenting the divergence.

### 2.3 High-field scope boundary

The pair relation `A = ν₁₂ + ν₃₄` is the **high transverse field** result
(textbook eqn 4.65; the "Paschen–Back region", textbook §19.4 Example 19.8). At
*low* field the observable pair is (ν₁₂, ν₂₃) and A is recovered by a different
expression (textbook eqn 4.64). For a radical with A ≈ 514 MHz the crossover
field `B₀ = (A/A_vac)·0.15853 T ≈ 183 G` (textbook Example 4.9), so the kG
fields at which radicals are actually measured (2.9–14.5 kG, McKenzie Fig. 3)
sit firmly in the high-field regime. The correlation spectrum is therefore a
**high-TF tool**; this is stated in the user docs and is *not* a divergence from
WiMDA (whose `rmatch` is likewise high-field).

---

## 3. `CorrFn` — the order-weighted line-pair combiner

`Plot.pas:1387-1394`:

```pascal
function CorrFn(y1, y2: double; order: integer): double;
begin
  if (order > 0) and (y1 <> 0) and (y2 <> 0) then
    CorrFn := 2 * abs(y1 * y2) /
      (power(abs(y1 / y2), order) + power(abs(y2 / y1), order))
  else
    CorrFn := abs(y1 * y2);
end;
```

`CorrFn(y₁, y₂, n)` combines the spectral amplitudes at the two paired
frequencies. It is the product `|y₁·y₂|` weighted by an **order-`n` ratio
penalty** `2 / (r^n + r^{−n})` with `r = |y₁/y₂|`. The penalty equals 1 when
`y₁ = y₂` and falls toward 0 as the two amplitudes diverge — so a genuine pair
(both lines present, comparable amplitude) is **rewarded** and a spurious pair
(one line strong, one in noise) is **suppressed**, increasingly so for larger
`order`. `order = 0` reduces to the plain product `|y₁·y₂|`. WiMDA's default is
`CorrOrder = 2` (`FFTPar.dfm:817`).

**Asymmetry:** ported verbatim as a small pure function in the new
`core/fourier/correlation.py`, default order 2, with the same `order = 0`
fallback. No divergence.

---

## 4. The generation loop — `Corr` / `AvCorr`

`Plot.pas:2149-2230` (power mode shown; `field`/`cfield`/`corder` from
`1487-1490`, `2154`):

```pascal
i0 := 2 + trunc(0.01355342 * field / fint);   { first bin above diamag line }
corder := FFTparams.CorrOrder.Value;
if field <> 0 then
  for i1 := i0 to nf2 do begin
    f1 := -(i1 - 1) * fint;                    { candidate (negative) freq    }
    f2 := rmatch(f1, cfield);                  { Breit–Rabi partner           }
    i2 := trunc(abs(f2) / fint);
    ifrac := frac(abs(f2) / fint);             { linear-interp partner bin    }
    if i2 <= nf then begin
      extrp := (1-ifrac)*fd^[abs(i2)] + ifrac*fd^[abs(i2+1)];      { S(f2)    }
      if gindex = 1 then
        dd^[ii,2] := CorrFn(fd^[abs(i1-1)], extrp, corder)         { group 1  }
      else
        dd^[ii,2] := dd^[ii,2] + CorrFn(fd^[abs(i1-1)], extrp, corder);
      data^[ii,1] := abs(f1 + f2);             { hyperfine axis = |f1+f2|     }
      ii := ii + 1;
    end;
  end;
{ later: dd^[i,2] := dd^[i,2] / g1;  — average over g1 groups }
```

Reading it out:

1. **Start above the diamagnetic line.** `i0` is the first bin above the
   diamagnetic muon frequency `γ_µ·field` (WiMDA's `0.01355342·field`). The
   diamagnetic peak is excluded from the scan.
2. **For each candidate bin** at frequency `f1`, find its Breit–Rabi partner
   `f2 = rmatch(f1, cfield)`, **linearly interpolate** the power spectrum at the
   (generally non-integer) partner bin, and combine the two amplitudes with
   `CorrFn`.
3. **Plot at the hyperfine axis** `|f1 + f2|`, which equals A at a genuine pair.
4. **`AvCorr`** accumulates `CorrFn` across groups then divides by the group
   count `g1` (`Plot.pas:2244`); **`Corr`** is the single-group spectrum.

**Two fields, by design.** `field` (run header / diamag-fit field, `Plot.pas:1487`)
sets the diamagnetic start bin; `cfield` (the `CorrField` text box,
`Plot.pas:1490`) sets the field used by `rmatch`. They are normally equal; the
split lets the user nudge the matching field independently.

**`CorrField` control** (`FFTPar.pas:512-526`): a field in Gauss; on change it
sets `CorrFreq = field·0.01355342` (the diamagnetic frequency read-out) **and**
`RangeMid[1]` — i.e. the diamagnetic exclusion slot tracks the same field.

### 4.1 Asymmetry's approach — exact forward map through the shared seam

The Asymmetry pipeline has already collapsed every group's spectrum to a single
**averaged display channel** in `compute_average_group_spectrum`
([`spectrum.py:322`](../../../src/asymmetry/core/fourier/spectrum.py)) — the same
architecture that let the frequency-domain-finishers exclusions/baseline operate
"mode-correctly" on one averaged channel rather than per-group pre-derivation
arrays. The correlation spectrum slots in as **one more derived display mode**,
exactly like `Resolution (Burg)` (`spectrum.py:354,403-426`): when the display
mode is `correlation`, take the averaged power spectrum `S(ν)` and transform it.

Rather than transliterate `rmatch`'s approximate inverse, Asymmetry uses the
**exact forward map**:

```
for each A on the hyperfine axis:
    ν₁₂, ν₃₄ = _tf_levels(B_ref, A)          # muonium.py, exact eqn 4.54
    # ν₁₂ + ν₃₄ = A  by construction (eqn 4.65)
    corr(A) = CorrFn( S(ν₁₂), S(ν₃₄), order ) # S interpolated at each line
```

This **reuses `muonium.py`** (requirement (1)), needs no inversion, carries no
rounded constants, and places each peak at the true A_µ to machine precision.
The hyperfine axis is sampled directly (uniform A grid) instead of being read
out as `|f₁+f₂|` from a scan, which also removes WiMDA's non-uniform-axis
artefact (its `data[ii,1] = |f1+f2|` points are unequally spaced).

**Divergence (average-then-correlate vs correlate-then-average).** WiMDA's
`AvCorr` averages the per-group `CorrFn` results; Asymmetry correlates the
**already-averaged** power spectrum. Because `CorrFn` is non-linear (a product),
these are not identical. **Behaviour:** WiMDA = `mean_g CorrFn(S_g(ν₁₂),
S_g(ν₃₄))`; Asymmetry = `CorrFn(mean_g S_g(ν₁₂), mean_g S_g(ν₃₄))`. The
Asymmetry form is consistent with how every other conditioning step already
operates on the averaged channel, has better noise behaviour (averaging the
linear spectra before the non-linear combine), and is the recommended default.
Per-group `Corr` (single selected group) is reachable by selecting one group;
true per-group-then-average `AvCorr` is recorded as a follow-on
(implementation-options §follow-ons) should a user need bit-parity.

---

## 5. The hyperfine axis is a coupling axis, not a field axis

WiMDA plots against `|f1+f2|` in MHz and offers a "shift units" toggle, but the
quantity **is the muon hyperfine coupling A_µ** (MHz), not a precession
frequency and not a field. Converting it to Gauss/Tesla through `units.py`'s
`γ_µ` (as the ordinary frequency axis does) would be physically meaningless —
A_µ is not `γ_µ·B`. **Asymmetry:** the correlation dataset carries its own
x-label ("Muon hyperfine coupling A_µ (MHz)", checkpoint-3 wording) and is
**excluded from the MHz/G/T field-unit selector**. This is a deliberate
departure from treating the correlation x-axis like the FFT frequency axis.

---

## 6. TF correlation vs ALC — complementary routes to radical hyperfine couplings

(Verified against McKenzie 2013 §1.2.1–1.2.2 and textbook §19.4; this content is
the basis of the mandatory user-doc subsection, not implemented code.)

Both TF correlation and ALC (avoided level crossing) measure a radical's
hyperfine couplings, from orthogonal directions:

- **TF correlation** (this feature): high **transverse** field, FFT line pair
  → the **isotropic muon coupling A_µ**. Best in **liquids, at high field, with
  resolvable precession and good radical yield**. Needs a **continuous source**
  (the spectrum runs to hundreds of MHz, beyond a pulsed source's bandwidth) and
  a **promptly formed** radical (textbook §19.4 Example 19.8; McKenzie 2013
  §1.2.1).
- **ALC** (already in Asymmetry — see §8 out-of-scope): swept **longitudinal**
  field, time-integral asymmetry, resonance **dips** (McKenzie 2013 §1.2.2,
  eqns 11–13). Three resonance types by the selection rule Δ|M| = 0, ±1, ±2
  (M = Σ m_z of muon, electron, nuclear spins):
    - **Δ₁** (ΔM = ±1, muon spin flip): one per radical, resonance field
      ≈ `A_µ/(2γ_µ)` → gives **A_µ**, the same coupling as TF correlation (a
      cross-check).
    - **Δ₀** (ΔM = 0, muon–nucleus flip-flop): "as many … as there are nuclei
      with I > 0"; field depends on **both** A_µ and the **nuclear** hfcc
      `A_k` (eqn 12) → maps the **other (nuclear) couplings**, and in
      anisotropic media the **dipolar** part → molecular **orientation and
      dynamics**. Observed in solids, liquids and gases.
    - **Δ₂** (ΔM = ±2): extremely weak, rarely observed.
  Best in **solids, liquid crystals, polymers and oriented/complex media**, and
  where precession is too broad to resolve.

**Practical workflow:** TF correlation first to **identify the radical and pin
A_µ**; then ALC to **map the rest of the coupling network** and the
anisotropy/dynamics (the Δ₀ dipolar resonance → molecular orientation). The user
docs cross-reference Asymmetry's existing ALC mode
([`core/transform/integral.py`](../../../src/asymmetry/core/transform/integral.py),
[`core/fitting/field_scan.py`](../../../src/asymmetry/core/fitting/field_scan.py),
the ALC user-guide page) so the reader can act on the second step.

---

## 7. Worked example for the user docs (cyclohexadienyl)

The canonical muoniated radical: **Mu addition to benzene → the cyclohexadienyl
radical** (McKenzie 2013 Fig. 1b; textbook §19.4 Example 19.8). Measured muon
hyperfine coupling **A_µ = 514.4(1) MHz**, proton hfcc **A_p = 128.5(3) MHz**
(McKenzie *et al.*, *J. Phys. Chem. B* **117**, 13614 (2013)). The textbook
states the method in one sentence: "A_µ can be directly calculated from the
splitting of the ν₁₂ and ν₃₄ lines of the frequency spectrum recorded in the
Paschen–Back region of the Breit–Rabi diagram" — precisely the correlation
principle. A second concrete spectrum (McKenzie Fig. 3): the muoniated
1,2-dicarboxyvinyl radical dianion, **A_µ = 493.8(2) MHz** at 2.9 and 14.5 kG.

These anchor both the documentation pedagogy and the synthetic verification
target (a simulated radical with A_µ = 514.4 MHz at, say, 2.9 kG must produce a
correlation peak at 514.4 MHz).

---

## Reused-API reconciliation (no duplication)

| Need | Reused API | Status |
|---|---|---|
| Breit–Rabi pair frequencies | [`muonium.py`](../../../src/asymmetry/core/fitting/muonium.py) `_tf_levels`, `G_MU_MHZ_PER_G`, `G_E_MHZ_PER_G` | **Reuse** — exact eqn 4.54; `w12+w34=A`. Do not re-derive. |
| Spectrum builder seam | [`spectrum.py:322`](../../../src/asymmetry/core/fourier/spectrum.py) `compute_average_group_spectrum` | **Extend** — add `correlation` as a derived mode beside `burg` (`spectrum.py:354`). |
| Display-mode registry | [`fft.py:11`](../../../src/asymmetry/core/fourier/fft.py) `_DISPLAY_ALIASES` / `canonical_fourier_display_mode` | **Extend** — add `correlation` alias (append at end of block). |
| Axis units | [`units.py`](../../../src/asymmetry/core/fourier/units.py) | **Reuse for B (Gauss) only**; the A_µ axis is *not* field-converted. |
| GUI display-mode + badge | `fourier_panel.py` Burg radio (`:239-248`) + reveal-on-toggle (`:414-415,539-541`) | **Follow pattern** — a badged specialist radio with a revealed control group. |
| Line-pair combiner | new `core/fourier/correlation.py` `CorrFn` | **New** — verbatim port of `Plot.pas:1387-1394`. |
| Project recipe | [`spectrum.py:107`](../../../src/asymmetry/core/fourier/spectrum.py) `GroupSpectrumConfig.to_dict/from_dict` | **Extend** — additive keys (reference field, corr order); round-trip must not regress. |

## Out of scope (recorded with rationale)

- **ALC / avoided-level-crossing analysis** — already in Asymmetry via the
  time-integral-asymmetry field-scan observable
  ([`core/transform/integral.py`](../../../src/asymmetry/core/transform/integral.py))
  + resonance fitting ([`core/fitting/field_scan.py`](../../../src/asymmetry/core/fitting/field_scan.py)),
  ported in PR #23 ([`time-integral-asymmetry`](../time-integral-asymmetry/)).
  The user docs discuss the TF/ALC **complementarity** (§6) but no ALC code is
  touched or duplicated.
- **Anisotropic / dipolar hyperfine** beyond the isotropic A_µ the correlation
  method targets. (The anisotropic two-spin solver exists in `muonium.py`
  `high_tf_muonium_aniso` for *fitting*, but the correlation matched-filter
  assumes the isotropic pair.)
- **DFT hyperfine prediction** — McKenzie compares measured couplings to DFT;
  out of scope here.
- **Low-field correlation** (textbook eqn 4.64) — the method is high-TF by
  construction (§2.3); no low-field variant is built.
- **Muoniated-radical *fit* functions / workflow** — a separate candidate
  ([`muonium-radical-hyperfine`](../candidates/muonium-radical-hyperfine/)),
  partly addressed by the `muonium.py` components; this study is the
  frequency-domain correlation transform only.
