# Al-LLZ captures

The longitudinal-field datasets use the standard EMU forward and backward
groups, file timing and good bins, and the diamagnetic calibration from run
51315. The calibration screenshot waits for the Grouping preview after the
estimate has completed.

The Batch setup and result share the same 160 K triplet and form. The result
capture runs the Batch fit with the 0–12 µs window and uses the zero-field run
for the overlay. The temperature trend uses one shared fluctuation rate and
one marginal rate error per three-field fit, with the previous temperature's
fit supplying the next seeds. The five coldest setpoints supply the mean rate
subtracted in the natural logarithm. The remaining eight setpoints enter the
line fit.

To replay the captures, set `ASYMMETRY_CORPUS_ROOT` to the extracted corpus and
run `python -m docs.screenshots.capture_corpus --module ionic_motion_llz`.
