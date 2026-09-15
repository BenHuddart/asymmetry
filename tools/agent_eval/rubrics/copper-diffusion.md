# Muon diffusion and QLCR in copper (Tier B)

Data folder: `Nuclear magnetism and ionic motion/Muon diffusion and QLCR in copper/Data`

## Experiment

Muons mimic interstitial hydrogen in copper metal. The worksheet uses
three separate field regimes on the same sample to probe hopping
dynamics: TF line-shape change (Lorentzian to Gaussian) with cooling to
extract a hop rate and an Arrhenius activation energy, ZF Kubo-Toyabe
behaviour whose dynamics are compared with the TF hop rate, and an
LF quadrupolar level-crossing resonance (QLCR) scan.

## Run structure

Two instrument eras/prefixes in one folder, no single calibration run
called out by name (the low-field TF runs serve as the alpha/lineshape
reference):

- EMU (2010), 20882-20917 (36 runs): TF 20-100 G cooling series
  280-100 K (20882-20885), ZF at 40 K and 1 K (20886-20887), an LF
  field sweep at fixed 40 K from 40-150 G (20888-20900, the QLCR set),
  then a ZF temperature series 60-200 K (20901-20917).
- ARGUS (2024), 76924-76961 (38 runs): TF 20 G decoupling series
  300-50 K (76924-76934), ZF runs at 40 K/4 K/100-140 K, and a second
  LF field sweep at fixed 40-44 K from 10-150 G (76941-76955, the QLCR
  set).

## Must

- [ ] Splits the folder into its TF, ZF, and LF run groups rather than
      treating it as one undifferentiated series.
- [ ] Separates the EMU (2010) and ARGUS (2024) runs as two distinct
      measurement campaigns on the same sample.
- [ ] Reports a change in TF line shape (Lorentzian-like to
      Gaussian-like) as temperature decreases, from at least one of
      the two TF series.
- [ ] Reports the ZF spectra as showing Kubo-Toyabe-like behaviour and
      notes any qualitative change between the lowest-temperature run
      and the next-lowest.
- [ ] Identifies the fixed-temperature, swept-field runs as the LF/QLCR
      data, distinct from the temperature scans.
- [ ] Contains no hop rate, activation energy, or rate number not
      produced by a tool call in this session.

## Should

- [ ] Compares a hop rate or dynamics trend derived from TF against one
      derived from ZF, as the worksheet asks.
- [ ] Notes that copper's own field/temperature labels in EMU and
      ARGUS logs disagree by campaign (different step sizes and
      ranges), rather than merging both eras' runs into one T-axis.
- [ ] Flags any run whose logged sample temperature deviates
      substantially from its nominal set-point.

## Known traps

- This is a single folder holding three field regimes and two
  instrument eras; grouping by run label alone (rather than field and
  timestamp) will merge unrelated scans.
- The QLCR field sweeps are at fixed temperature, not a function of T
  — plotting them against temperature instead of field is a common
  mis-grouping.
- Neither era has a labelled "calibration" run; TF runs at fields that
  behave close to full-asymmetry precession are the implicit
  reference, so alpha handling should be stated rather than assumed.
