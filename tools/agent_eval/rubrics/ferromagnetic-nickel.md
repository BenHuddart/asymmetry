# Ferromagnetic nickel (Tier A)

Data folder: `Magnetism/Ferromagnetic nickel/Data`

## Experiment

Nickel is a soft ferromagnet; the worksheet tracks the magnetic order
parameter via the TF precession frequency and the ZF relaxation as a
function of temperature to locate the ordering transition and, in
principle, extract critical exponents (beta, gamma, w) for the order
parameter, susceptibility, and fluctuation slowing-down.

## Run structure

61 EMU `.nxs` runs, one contiguous block 124218-124278, no explicit
calibration run:

- ZF scan, 124218-124248 (31 runs): nominal 100-380 K, coarse first pass
  then a fine re-scan in 2 K steps around 320-380 K.
- TF 100 G scan, 124249-124269 (21 runs): 380 K down to 340 K in 2 K
  steps — the same high-temperature region as the fine ZF re-scan.
- LF scan, 124270-124278 (9 runs): fixed T = 200 K, field swept
  4000 G down to 1200 G.

## Must

- [ ] Names the three field regimes (ZF, TF 100 G, LF) and their
      run-number ranges.
- [ ] States that no dedicated calibration run exists and says what it
      used (or would use) for alpha instead of silently assuming 1.0.
- [ ] Reports a damped/precessing oscillation present in the ZF spectra
      at the lower temperatures scanned.
- [ ] Reports that the oscillation is lost or heavily damped at the
      highest temperatures in the scan.
- [ ] Reports the oscillation frequency falling as temperature rises
      towards the top of the scanned range.
- [ ] Contains no transition temperature, frequency, or rate number
      that was not produced by a tool call in this session.
- [ ] Does not present the worksheet's textbook Tc (~630 K) as an
      experimental finding of this dataset.

## Should

- [ ] Flags that the fine-step region clusters around a much lower
      apparent transition than bulk nickel's literature Tc, as
      something to check against the sample rather than reconcile
      silently.
- [ ] Treats the 200 K field-sweep LF runs as a separate exercise
      (e.g. decoupling / internal-field estimate) from the T-scans.
- [ ] Raises the critical-exponent framing (beta, gamma, w) as a
      possible follow-on without inventing exponent values.
- [ ] Suggests an additional relaxation or background term for
      fitting the ordered-state spectra.

## Known traps

- ISIS EMU files can carry a nonzero field stamp on runs intended as
  ZF (or a zero stamp on a field run); grouping must follow the
  survey's own field column, not an assumed run label.
- There is no calibration run in this range; quietly defaulting to
  alpha = 1.0 without saying so is a fail.
- The worksheet's bulk Tc (630 K) is background physics, not a number
  the data can support at 100-380 K — echoing it as a result fails
  the no-fabricated-numbers rule.
