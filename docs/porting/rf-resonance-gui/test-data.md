# Test data — RF-µSR resonance GUI surface

Reuses the corpus the core port was verified on (see
[`../rf-musr-resonance-fit/test-data.md`](../rf-musr-resonance-fit/test-data.md)).

## Benzene RF scan (paper-graded)
- Corpus: `Chemistry/Muon spectroscopy of benzene/` (WiMDA muon-school corpus).
- DEVA runs **56426–56462**, **ν_RF = 218.5 MHz**, 293 K (from the core study).
- Observable: **(Red − Green) integral asymmetry vs swept static field**; W-shaped
  double dip with resonances near **865 G / 773 G** (digitised).
- Targets (paper, McKenzie 2013, Table 1): **A_µ = 514.78(4) MHz**, **A_p = 124.6(14) MHz**.
  Core port recovers A_µ=516.0, A_p=125.4, resonances 866/772 G, χ²/dof=1.6.
- **TODO (study pass):** confirm the on-disk run list for the RF technique (the
  benzene folder also holds ALC/repolarisation/high-TF runs); map which runs are
  RF-on vs RF-off (or which detector groups are "Red"/"Green").

## Headless test fixtures
- Prefer a **core scan-builder unit test**: synthetic or corpus-conditional
  (Red − Green) field series → fit `RFResonanceMuP` → assert A_µ within tolerance.
  Mirror the corpus-conditional skip pattern in `tests/test_maxent_corpus_smoke.py`
  and `tests/test_fft_axis_corpus.py` (this round) so CI stays green.
- GUI smoke (offscreen): the RF-scan panel (option A2) builds a scan and the model
  appears in its picker — mirror `tests/test_*_gui.py` panel patterns.

## Notes
- Corpus root on the test machine: `C:\Users\benhu\Source\wimda-corpus`
  (also `~/Documents/WiMDA muon school`; honor `WIMDA_CORPUS_ROOT`).
- No reference RF scan numbers are needed beyond the paper values above; the core
  study already established the numeric oracle.
