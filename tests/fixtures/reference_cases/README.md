# Recorded synthetic reference cases

These cases contain owned synthetic input and safe native numerical output.
They are regressions against recorded execution, not fresh runs in ordinary CI.
The same format extends the existing `../psi_root/reference.json`; it is test
evidence, with no production registry or runner framework.

`format_version: 1` names the case and evidence class. `identity` records program
version/build and binary SHA256 (Mantid names its `_api.so`; the release archive
hash is separate). `inputs` hashes complete consumed files. `settings` declares
selection, timing, grouping/corrections, packing/variance/objective, units, and
phase/window/padding or their inapplicability. `native_output` contains native
values or a hashed numerical file, separate from the independent `anchor`.
`mapping`, `native_status`, `tolerances`, and `limits` preserve convention and
scope. Input hashes, required provenance, finite JSON round-trip, and tolerance
bounds are checked at the fixture boundary; operation tests check the numerical
contract. Private setup logs and reference implementations are excluded.

`constant/` preserves all three supplied-error observations and final outputs
from Mantid6.16.1 and musrfit1.12.1/root6/708f777b/ROOT6.40.04. Independent
normal equations give mean40/9, SE2/3, chi-square113/9 with NDF2. Mantid's
stationary status remains verbatim. musrfit's fit[-.5,2.5] retains all three
points; a fit[0,2] in this native path omits the terminal point and is unsuitable
for that comparison. Output errors use absolute supplied sigmas, without
residual rescaling. The6-significant-digit objective12.5556 motivates absolute
tolerance5e-5; Mantid uses1e-7. Fresh musrfit invocation in a copied directory:

```sh
musrfit owned.msr --dump ascii --keep-mn2-output --use-no-of-threads 1
```

`manual-phase/` uses owned rounded1001-bin cosine counts from the bounded
frequency control. Full type0 musrview requests at0°/37° have identical input,
support, t0, background, packing, and normalisation. The public ROOT-canvas
extractor reads observed TH1F real/imaginary coefficients, excluding theory.
All495 native bins are included. Bins0–493 rotate by exp(-i*37*pi/180) within
`4*float32eps*(abs(z0)+abs(z37))`; bin494 remains unchanged and is tested as a
difference. This is a same-bin phase test, with no native absolute DFT-scale or
error agreement claim. No endpoint is removed to make a whole-output assertion
pass. CLI musrFT data-file phase options do not change the tested dump/graphic
output, so those flags alone are not a manual-phase oracle. In a copied directory:

```sh
musrview phase0.msr -f --root --timeout 15
musrview phase37.msr -f --root --timeout 15
root -n -l -b -q 'export_canvas.C("phase0_0_F.root","phase0.csv")'
root -n -l -b -q 'export_canvas.C("phase37_0_F.root","phase37.csv")'
```

`pulse-source.json` instead pins an external Mantid checkout/module and START
settings. It is the configuration contract for optional source-module execution,
not a recorded installed Mantid algorithm result. No third-party implementation
is included. The independent analytic pulse tests remain installation-free.
