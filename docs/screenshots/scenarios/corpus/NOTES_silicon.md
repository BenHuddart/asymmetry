# Silicon photo-μSR captures

The Grouping and overlay captures use run 103277. The overlay opens its two
period reductions as separate browser entries. For each run the time-domain
fit uses the file grouping, nominal detector balance and good bins from the
first microsecond. The Green fit supplies the fixed amplitude for the Red
fit. The trend figures and downloadable silicon lifetime script use the same
weighted, zero-offset calibration, and the decay figure fits the first six
delay points.

To replay the captures, set `ASYMMETRY_CORPUS_ROOT` to the extracted corpus
and run `python -m docs.screenshots.capture_corpus --module silicon_photo`.
