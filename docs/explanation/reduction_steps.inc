.. container:: reduction-guide

   .. container:: reduction-step
      :name: guide-acquire

      .. rubric:: Start with detector counts

      A detector histogram counts positrons in successive time bins after a
      muon arrival or beam pulse. The total rate falls as muons decay. The
      imbalance between opposite directions carries information about the
      ensemble spin polarisation.

      :Input → output: Acquired histograms → a selected run, period,
         detector set, and time interval. Selecting runs or periods does not
         add their counts together.
      :How it works: For independent raw counts that follow Poisson statistics,
         a bin containing :math:`n` counts has the usual estimated variance
         :math:`\sigma_n^2=n`.
      :Uncertainty: Corrected counts, shared events, and subtracted reference
         runs need the variance information carried through those operations.
         Treating their results as fresh Poisson counts gives incorrect errors.
      :When this applies: Loading raw data keeps the timing and usable time
         intervals supplied by the file. The file format alone does not tell
         you whether the muon source is pulsed or continuous.
      :What to keep with the result: Acquisition identity, period, detector
         membership, exclusions, and file timing. See :doc:`/reference/loading_data`
         and :doc:`/explanation/musr_primer` for the physical measurement.

   .. container:: reduction-step
      :name: guide-deadtime

      .. rubric:: Correct for missed detector events

      A detector cannot record every closely spaced event. Deadtime correction
      estimates these missed events separately for each detector, before its
      counts are added to a group. It needs the appropriate detector calibration
      and the acquisition exposure used to convert counts to rates.

      :Input → output: Each selected detector's raw counts → corrected counts
         and variances, or a recorded reason why correction was not applied.
      :How it works: Apply the detector's correction formula to its own
         histogram. Correcting the group sum generally gives a different result.
      :Uncertainty: Counting errors pass through the correction formula.
         These errors assume the supplied correction coefficients and exposure
         are fixed; their uncertainties are not included.
      :When this applies: If required **File** coefficients are missing,
         correction is left off for the whole requested result. Invalid exposure
         or a count rate at which the correction formula breaks down raises an error.
      :What to keep with the result: Requested correction, whether it was
         applied, the coefficient source, and exposure. See
         :doc:`/reference/detector_grouping` and the
         :ref:`deadtime comparison <comparison-deadtime-scope>` for the reviewed
         program versions and evidence.

   .. container:: reduction-step
      :name: guide-align

      .. rubric:: Align and sum detector groups

      Forward (F) and backward (B) counts must describe the same times. Shift each
      selected histogram by whole bins using that detector's time zero, keep
      the time interval measured by all selected detectors, and add the counts
      within each group.

      :Input → output: Selected detector histograms → aligned group counts.
      :How it works: :math:`F_i=\sum_{d\in F}n_{di}` and
         :math:`B_i=\sum_{d\in B}n_{di}` for disjoint detector groups.
      :Uncertainty: Independent detector variances add. Where groups share
         counting noise, that shared contribution must also be carried through
         the calculation. Different group names do not establish independence.
      :When this applies: Alignment does not interpolate counts or fill gaps
         with zeros. Timestamps use the exact time-zero value of the reference
         detector. Recorded fractional-bin offsets describe the remaining
         alignment approximation, not an uncertainty in time zero.
      :What to keep with the result: Detector membership, each detector's time
         zero, selected time interval, whole-bin shifts, and fractional offsets.
         See :doc:`/reference/detector_grouping`, :doc:`/reference/run_arithmetic`,
         and the :ref:`timing comparison <comparison-timing>` for the reviewed
         program versions and evidence.

   .. container:: reduction-step
      :name: guide-background

      .. rubric:: Account for accidental background counts

      Accidental counts dilute the directional signal. Choose one background
      treatment for the grouped counts: estimate a level from a selected range,
      fit the late-time tail, supply a fixed level, or subtract a separately
      prepared reference run. Polarisation from muons stopping in the sample
      holder is a different contribution, described by the physical fit model.

      :Input → output: Aligned group counts → background-treated group counts.
      :How it works: Reference subtraction uses :math:`F'=F-sF_{\rm ref}` with
         a fixed scale :math:`s`. For independent sample and reference counts,
         :math:`V_{F'}=V_F+s^2V_{F_{\rm ref}}`.
      :Uncertainty: A background estimated from the same data introduces shared
         noise between bins. Its counting uncertainty is carried through the
         calculation. Uncertainty in supplied fixed levels and in the reference scale
         is not included in these errors. Separate error bars do not describe all correlations needed
         for a fit to correlated bins.
      :When this applies: The reference run needs its own timing, period,
         and detector corrections, with compatible bin widths and enough
         measured time in common. Count-domain fits reject reference subtraction
         and fit their own background levels.
      :What to keep with the result: Background method, whether it was
         applied, selected range or reference run, timing, exposure, and scale.
         See :doc:`/reference/data_reduction/backgrounds` and the
         :ref:`reference-error comparison <comparison-reference-errors>` for
         the reviewed program versions and evidence.

   .. container:: reduction-step
      :name: guide-bin

      .. rubric:: Add counts before calculating asymmetry

      Wider bins give more counts per estimate at the cost of time resolution.
      Add the forward counts together and the backward counts together, then
      calculate the asymmetry from those two sums.

      :Input → output: Prepared F/B acquisition bins → combined F/B counts
         covering the bins that actually contribute.
      :How it works: :math:`F_I=\sum_{i\in I}F_i`,
         :math:`B_I=\sum_{i\in I}B_i`. With both balance coefficients equal to
         one and no corrections, :math:`(F,B)=(9,1),(1,1)` becomes
         :math:`(10,2)`, giving :math:`A_I=2/3` (about 66.7 %).
      :Uncertainty: Combine the counts' variance information before taking
         the ratio. Include shared background uncertainty; adding the separate
         variances alone would lose that contribution.
      :When this applies: Only whole acquisition bins contribute. For variable-width
         binning, each requested width selects the nearest whole number of
         acquisition bins. Half-bin ties round upwards, and every output bin
         contains at least one acquisition bin. Initial variable-width binning
         keeps a partial last bin; fixed-size packing and later count merges
         drop incomplete groups. Timestamps are midpoints
         of the contributing bins, including offsets from the exact time zero.
         For vector rotation, this initial count binning comes before rotation;
         extra display binning averages the rotated signals afterwards.
      :What to keep with the result: Contributing bin indices, actual edges,
         binning settings, and selected time interval. See
         :doc:`/reference/data_reduction/binning` and the
         :ref:`binning comparison <comparison-binning>` for the reviewed
         program versions and evidence.

      .. admonition:: Why averaging the two ratios gives a different answer

         The original ratios are :math:`(9-1)/(9+1)=0.8` and
         :math:`(1-1)/(1+1)=0`. Their arithmetic mean is :math:`0.4`,
         or 40 %, rather than 66.7 %. Averaging the ratios gives the sparse
         second bin equal weight. Adding counts first gives it only two of
         the twelve counts. These estimates answer different questions;
         choose the one that matches the quantity you want to measure.

   .. container:: reduction-step
      :name: guide-ratio

      .. rubric:: Calculate the forward/backward asymmetry

      The difference between forward and backward counts still follows the
      falling muon population. Dividing by a balanced sum gives a dimensionless
      measure of their directional imbalance. The count balance α accounts for
      unequal group rates; β allows unequal asymmetry amplitudes. Both refer
      to the selected detector pair.

      :Input → output: Prepared F/B counts and chosen α/β → asymmetry values
         and their counting errors.
      :How it works: :math:`A=(F-\alpha B)/(\beta F+\alpha B)`.
         With :math:`\alpha=\beta=1`, the example :math:`F=10,B=2`
         gives fractional :math:`A=2/3`. APIs that report percent return
         :math:`100A`, with errors in percentage points.
      :Uncertainty: First-order error propagation includes the F and B
         variances and any retained covariance, which describes their shared
         fluctuations. For independent raw Poisson counts and both balance
         coefficients equal to one, :math:`\sigma_A^2=4FB/(F+B)^3`.
         These errors hold calibration and supplied correction quantities
         fixed; uncertainty in those quantities is not included. Sparse bins and unusable ratios need the treatment described
         in :ref:`asymmetry-counting-errors`.
      :When this applies: Detector labels do not determine α or β. Estimating
         α from each run's count integrals is a different method from a
         diamagnetic calibration fit. Signed corrected counts and general β
         can give fractional asymmetries outside −1 to 1. A ratio with an
         unusable denominator cannot be used as a valid measurement.
      :What to keep with the result: Balance coefficients, their sources and
         available calibration records, units, and the recorded reduction
         settings. See :doc:`/reference/asymmetry_units`,
         :doc:`/reference/data_reduction/alpha_calibration`, and
         :doc:`/reference/data_reduction/beta_calibration`.

   .. container:: reduction-step
      :name: guide-fit

      .. rubric:: Fit a model to the measured asymmetry

      A relaxation or precession model describes the spins of muons stopping
      in the sample and holder. Compare its prediction with the measured
      asymmetry to estimate physical parameters.

      :Input → output: Laboratory-frame asymmetry and a physical model →
         fitted parameters and diagnostics.
      :How it works: Add sample and holder contributions with separate
         amplitudes, for example :math:`A(t)=A_sP_s(t)+A_hP_h(t)`.
      :Uncertainty: Use the fitting method appropriate to the selected data
         and their errors. An error bar for each point does not describe all
         correlations between points. Uncertainty in fixed calibration values
         is not included in these errors.
      :When this applies: A selected ordinary double-pulse response acts on
         the model prediction after reduction. It leaves measured counts and
         their reduction unchanged. Exposure alone does not determine the
         sample-holder polarisation amplitude.
      :What to keep with the result: Identity of the fitted data, model,
         fixed assumptions, and selected time interval. See
         :doc:`/reference/composite_models`, :doc:`/reference/fitting`, and
         :ref:`ordinary-time-response`.

   .. container:: reduction-step
      :name: guide-lifetime

      .. rubric:: Remove the decay envelope from group signals

      Fourier and maximum-entropy analysis can start from detector groups
      without forming an F/B ratio. Multiplying by the inverse of the muon
      decay envelope exposes the oscillatory group signal.

      :Input → output: Prepared group counts, before subtracting a mean →
         lifetime-corrected group signals and errors.
      :How it works: Multiply counts and errors by :math:`\exp(t/\tau_\mu)`,
         where :math:`\tau_\mu` is the muon lifetime.
      :Uncertainty: The same factor amplifies late-time errors. Shared noise
         from acquisitions or backgrounds must also be carried through the
         calculation where those sources are tracked.
      :When this applies: FFT and MaxEnt each estimate a normalisation level from
         their selected time interval before further processing. That level does not
         establish an absolute polarisation calibration. The alternative
         F−B Fourier input transforms an asymmetry ratio.
      :What to keep with the result: Group signals have their own processing
         records. They do not include the complete reduction record attached
         to asymmetry calculated directly from counts. See
         :doc:`/reference/fourier_analysis` and the
         :ref:`spectral comparison <comparison-spectral-conventions>` for
         the reviewed program versions and evidence.

   .. container:: reduction-step
      :name: guide-fourier

      .. rubric:: Find oscillation frequencies with a Fourier transform

      A Fourier spectrum reveals oscillation frequencies. Its peaks also
      depend on the measured time interval, sampling, window, and display
      convention. A displayed peak width is therefore not automatically
      the intrinsic width of a field distribution.

      :Input → output: Lifetime-corrected groups, or the selected F−B
         asymmetry → complex Fourier coefficients and a chosen display.
      :How it works: The discrete transform sums weighted samples against
         :math:`\exp(-2\pi i f t)`. Padding gives more closely spaced frequency
         samples without adding resolving information to the measured signal.
      :Uncertainty: Spectral errors assume the stated calibration and
         processing settings are fixed. Windowing and shared input noise can
         correlate spectral bins; their error bars do not establish independence.
      :When this applies: Grouped FFT estimates its fractional signal level
         with :math:`1/\sigma` weights and offers mean removal. Correcting for
         window gain does not calibrate absolute polarisation. Pulse-amplitude
         compensation changes the displayed amplitudes and which frequency
         bins are retained; it leaves measured counts and response phase unchanged.
      :What to keep with the result: Input signal, selected time interval,
         apodisation window, padding, phase, normalisation, and response
         assumptions. See :doc:`/reference/fourier_analysis` and the
         :ref:`spectral comparison <comparison-spectral-conventions>` for
         the reviewed program versions and evidence.

   .. container:: reduction-step
      :name: guide-maxent

      .. rubric:: Reconstruct a field distribution with maximum entropy

      MaxEnt finds a field distribution whose predicted signal matches the
      prepared group signals, using an entropy term to regularise the noisy
      inverse problem. The reconstruction depends on the model and on the
      strength of this regularisation.

      :Input → output: Prepared group signals → a reconstructed distribution
         and diagnostics comparing its prediction with the data.
      :How it works: The fit balances agreement with the data against
         an entropy term. This is a fitted reconstruction rather than an FFT
         of the displayed curve.
      :Uncertainty: Input errors assume the stated calibration and processing
         settings are fixed. They do not provide a joint statistical model for
         overlapping detector groups. Reconstructed widths also depend on the
         reconstruction choices and are not automatically intrinsic physical widths.
      :When this applies: MaxEnt estimates its starting level with
         inverse-variance weights and reconstructs phase and amplitude in its
         own way. An enabled pulse response acts on the prediction using the
         observation's resolved time origin.
      :What to keep with the result: Input groups, measured time interval,
         field grid, regularisation, phase, and response. See
         :doc:`/reference/fourier_analysis` and the
         :ref:`MaxEnt comparison <comparison-spectral-conventions>` for
         the reviewed program versions and evidence.

   .. container:: reduction-step
      :name: guide-integral

      .. rubric:: Add counts over time, then calculate a ratio

      A field, temperature, or run scan needs one value from each acquisition
      or condition. Sum the prepared counts within the selected time window,
      then form a ratio to measure the total counting signal in that window.

      :Input → output: Prepared counts in selected whole acquisition bins →
         one count-integral asymmetry for each chosen period operation.
      :How it works: :math:`A_{\rm int}=(\sum F-\alpha\sum B)/(\beta\sum F+\alpha\sum B)`.
      :Uncertainty: Combine the counts and their retained shared-noise
         contributions before calculating the final uncertainty. Periods that
         share counting noise with each other or a reference run cannot be
         treated as independent.
      :When this applies: Include bins whose centres lie on or between the
         window endpoints. Omitted endpoints use the good window. Display
         binning does not change the selected acquisition bins. G+R pools
         prepared counts first; G−R subtracts the prepared period ratios.
      :What to keep with the result: Time window, estimator, period operation,
         balance coefficients, correction outcomes, and any shared count sources.
         See :doc:`/reference/alc_mode`,
         :doc:`/reference/data_reduction/period_mapping`, and
         :ref:`condition-scan-observables`.

   .. container:: reduction-step
      :name: guide-native-mean

      .. rubric:: Average the ratios from acquisition bins

      A time-bin mean gives equal weight to each retained acquisition-bin
      ratio, regardless of its total counts. It measures a different quantity
      from a ratio formed after adding counts over the window.

      :Input → output: Prepared acquisition-bin ratios or G−R differences →
         their time-bin mean.
      :How it works: :math:`\overline{A}=N^{-1}\sum_i A_i` over the bins
         retained by the estimator. This generally differs from a ratio of sums.
      :Uncertainty: Check the status of each acquisition-bin ratio or G−R
         difference before averaging. Shared counting noise contributes to
         the mean's uncertainty and must be carried through the calculation.
      :When this applies: The same time-window and period-correction rules
         apply as for count integrals. The calculation starts from acquisition
         bins rather than the combined bins of a displayed curve.
      :What to keep with the result: Estimator, selected acquisition bins,
         point status, and period operation. See :doc:`/reference/alc_mode`,
         :doc:`/reference/data_reduction/period_mapping`, and
         :ref:`condition-scan-observables`.

   .. container:: reduction-step
      :name: guide-scan

      .. rubric:: Plot and fit the scan

      Associate each reduced value with its field, temperature, or run number.
      Fit those original values. A numerical derivative is available separately
      for display and export.

      :Input → output: Reduced values, their status, and source records →
         a condition scan and optional fit to the original values.
      :How it works: The derivative uses adjacent increasing coordinates,
         :math:`(A_{i+1}-A_i)/(x_{i+1}-x_i)`, placed at their midpoint.
      :Uncertainty: Fit selection checks known covariance, or shared
         fluctuations, within the selected window. Derivatives of known
         dependent input pairs keep their values but have unavailable errors.
         Separate derivative error bars do not establish a suitable fit likelihood.
      :When this applies: Derivatives are for display and export only, with
         no automatic rule for skipping gaps. A combined run's condition label
         is a label, not a weighted estimate of its physical conditions.
      :What to keep with the result: Condition coordinates, reduction
         settings, shared count sources, selected points, and operation.
         See :doc:`/reference/alc_mode` and :doc:`/reference/run_arithmetic`.

   .. container:: reduction-step
      :name: guide-vector

      .. rubric:: Rotate two measured transverse components

      Two measured transverse spin components allow a point-by-point rotation
      of coordinates. Their physical geometry must be established: a detector
      label or fitted phase alone does not determine it or supply a missing
      component.

      :Input → output: Two measured components, adjusted for baseline, gain,
         rotation sense, and phase → rotated components.
      :How it works: Apply a rotation matrix to the two components using the
         chosen frame convention. Initial count binning comes before rotation;
         extra display binning averages the rotated signals afterwards.
      :Uncertainty: Shared counting noise from acquisitions, references, or
         backgrounds passes through rotation and display binning. These errors
         treat the chosen frame and calibration quantities as fixed.
      :When this applies: Two-component rotation and single-signal demodulation
         are alternative operations. Pooled G+R rotation requires the Red and
         Green transverse baselines to agree; it is unavailable if they differ.
      :What to keep with the result: Both input signals and the chosen frame.
         See :doc:`/reference/rotating_frame_projection`,
         :ref:`vector-geometry-assumptions`, and
         :ref:`vector-rotation-counting-errors`.

   .. container:: reduction-step
      :name: guide-scalar

      .. rubric:: Show the slowly varying part of one signal

      Complex demodulation and a finite-impulse-response low-pass filter can
      reveal a slowly varying envelope in one measured asymmetry. The result
      is a processed display; it does not supply a second measured transverse
      component.

      :Input → output: One laboratory-frame asymmetry → a filtered complex
         display and optional magnitude.
      :How it works: Multiply by a complex oscillation at the reference
         frequency, then low-pass filter. That frequency sets the display frame.
      :Uncertainty: Real and imaginary parts can share noise within each bin;
         this covariance enters first-order magnitude errors. Filtered bins
         remain correlated. At low signal, the magnitude is biased and need
         not follow an exact Rician distribution.
      :When this applies: Fits use the original laboratory-frame observations,
         rather than treating filtered points as independent measurements.
      :What to keep with the result: Original signal, reference frequency,
         and filter. See :doc:`/reference/rotating_frame` and
         :ref:`scalar-rrf-magnitude-errors`.

   .. container:: reduction-step
      :name: guide-countfit

      .. rubric:: Fit a model to detector-group counts

      A full-count fit predicts the falling detector-group rates and their
      spin-dependent signal together. It starts from counts, rather than an
      F/B ratio, and fits its own accidental-background levels.

      :Input → output: Counts suitable for the chosen statistical method →
         a fit using Gaussian errors or Poisson count statistics.
      :How it works: Predict full group counts, including muon decay, the
         spin signal, group normalisations, and flat accidental backgrounds.
      :Uncertainty: Raw-count Poisson fitting and lifetime-corrected Gaussian
         fitting are separate choices. Use the variance-aware Gaussian method
         for deadtime-corrected counts.
      :When this applies: Both methods reject active reference-run subtraction.
         Other background settings are checked for compatibility before fitting;
         they do not replace the background levels fitted by the count model.
         A supplied pulse response changes the prediction, not measured counts.
      :What to keep with the result: Selected count input, applied corrections,
         fitting method, fitted background model, and response. See
         :doc:`/reference/count_domain_fitting`,
         :doc:`/reference/grouped_time_domain_fitting`, and the
         :ref:`count-background comparison <comparison-count-backgrounds>` for
         the reviewed program versions and evidence.
