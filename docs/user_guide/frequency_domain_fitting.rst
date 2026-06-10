Frequency-Domain Fitting
========================

Frequency-domain fitting extends the same single-fit, global-fit, and
parameter-trending workflow used in the time domain to displayed Fourier
spectra.  The V1 workflow fits the real-valued spectrum currently shown in the
Frequency view.  It does not fit the complex FFT directly.

Workflow
--------

1. Compute a Fourier spectrum from the **Fourier** panel.
2. Switch to the **Frequency** workspace.
3. Open the **Fit** dock.
4. Fit the displayed spectrum with a Gaussian or Lorentzian peak plus a
   constant or linear background.
5. For a run series, select multiple runs with cached spectra and use the
   **Global** tab.
6. Inspect ``nu0`` and ``fwhm`` in the **Parameters** dock, alongside derived
   ``B0`` and ``Bwid`` field equivalents.

The fitting x axis is stored internally as absolute frequency in MHz.  Plotting
controls may show field in gauss or a reference-relative frequency axis, but fit
parameters remain canonical: ``nu0`` and ``fwhm`` are MHz quantities.

Available Components
--------------------

The fit-function builder is filtered by analysis domain: when fitting a
spectrum it offers only the frequency-domain components below (as a flat
list), and these components do not appear when fitting in the time domain.
Typing a component name from the other domain gives an explanatory error.

``GaussianPeak``
    Peak height, centre ``nu0``, and full width at half maximum ``fwhm``.

``LorentzianPeak``
    Peak height, centre ``nu0``, and full width at half maximum ``fwhm``.

``ConstantBackground``
    Flat spectral background ``bg``.

``LinearBackground``
    Background ``bg + slope * nu``.

Global Fits And Trends
----------------------

Global frequency-domain fitting uses the same parameter-role table as
time-domain global fitting.  Mark peak centre or width as ``Local`` to trend
them across a field or temperature series, or mark background terms as
``Global`` when they should be shared.

Successful global frequency fits are sent to the **Parameters** dock under the
``Frequency Domain`` group.  The parameter-trending tools can then fit
``nu0(T)``, ``fwhm(B)``, ``B0(T)``, or ``Bwid(B)`` using the usual trend-model
workflow.

Project Files
-------------

Project files store frequency-fit state separately from time-domain fit state.
This lets a project reopen with both a time-domain model and a spectral peak
model intact.  Cached Fourier spectra are still stored in the Fourier spectrum
state; raw detector arrays remain referenced by source-file path rather than
embedded in the project.
