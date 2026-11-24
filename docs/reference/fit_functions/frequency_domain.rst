.. _fit-frequency-domain:

Frequency domain
================

These components fit the real-valued Fourier spectrum displayed in the
Frequency workspace, as a function of frequency :math:`\nu` in MHz (see
:doc:`../frequency_domain_fitting` for the workflow). They are only offered
when fitting in the frequency domain, where the picker presents them as a
flat list.

Both peak models describe the displayed spectrum. Their width parameter
``fwhm`` is the **full width at half maximum (FWHM)**, measured in MHz.

For an isolated, correctly phased absorptive line with an effectively infinite
record, an exponential envelope :math:`e^{-\lambda t}` gives a Lorentzian
:math:`\mathrm{FWHM}=\lambda/\pi`, while
:math:`e^{-(\sigma t)^2}` gives a Gaussian
:math:`\mathrm{FWHM}=2\sigma\sqrt{\ln2}/\pi`.
Here frequency is cyclic (MHz), while :math:`\lambda` and :math:`\sigma`
are rates in μs⁻¹. The Gaussian rate uses the convention in
:ref:`fit-gaussian`; it is not generally the KT :math:`\Delta`.

These ideal transform pairs do not automatically describe a finite,
apodised, modulus/power, or MaxEnt spectrum. The fitted ``fwhm`` is the width
of the **displayed processed peak**. Instrument response and the estimator
must be accounted for before interpreting it as an intrinsic relaxation rate
or field width. The peak functions include no automatic response convolution
or deconvolution. The ideal rate conversions above apply only when the stated
line-shape assumptions hold. See :doc:`../fourier_analysis`.

Peak centre and width convert to field equivalents using the cyclic
:math:`\gamma_\mu/(2\pi)`: :math:`B_0=\nu_0/(\gamma_\mu/2\pi)` and
:math:`B_{\mathrm{wid}}=\mathrm{FWHM}/(\gamma_\mu/2\pi)`. This conversion
changes units, not the estimator's physical meaning. Peak heights remain in
the arbitrary units of the displayed spectrum and depend on processing and
normalisation.

.. image:: /_generated/screenshots/spectral_peak_component_help.png
   :alt: GaussianPeak component help distinguishing processed FWHM from an intrinsic rate
   :width: 100%

.. _fit-gaussian-peak:

GaussianPeak
------------

.. math::

   S(\nu) = h\,\exp\!\left[-4\ln 2\,
   \frac{(\nu-\nu_0)^2}{w^2}\right],
   \qquad w \equiv \mathrm{FWHM}

A Gaussian spectral line — appropriate when the underlying time-domain
envelope is Gaussian (static, dense field distribution): the spectral-domain
counterpart of :ref:`fit-gaussian`. The width :math:`w` is the literal full
width at half maximum, :math:`S(\nu_0 \pm w/2) = h/2` exactly.

==========  ==============  =====  ============================================
Name        Symbol          Unit   Description
==========  ==============  =====  ============================================
``height``  :math:`h`       a.u.   Peak height.
``nu0``     :math:`\nu_0`   MHz    Peak centre.
``fwhm``    :math:`w`       MHz    Full width at half maximum.
==========  ==============  =====  ============================================

.. _fit-lorentzian-peak:

LorentzianPeak
--------------

.. math::

   S(\nu) = \frac{h}{1 + 4\,(\nu-\nu_0)^2/w^2},
   \qquad w \equiv \mathrm{FWHM}

A Lorentzian spectral line — appropriate when the time-domain envelope is
exponential (dynamic broadening, or a dilute static field distribution): the
spectral-domain counterpart of :ref:`fit-exponential`. Parameters as for
``GaussianPeak``, with the same half-height convention. Note the heavy
Lorentzian tails: fit windows should extend several FWHM beyond the peak or
the width can become weakly constrained. A non-Lorentzian shape can reflect
overlap, processing, or a different field distribution; two fitted peaks do
not by themselves establish two stopping sites.

.. _fit-constant-background:

ConstantBackground
------------------

.. math::

   S(\nu) = b_g

A flat spectral baseline, absorbing white noise and the flat part of any
apodisation pedestal. Parameter: ``bg`` (a.u.).

.. _fit-linear-background:

LinearBackground
----------------

.. math::

   S(\nu) = b_g + m\,\nu

A linearly sloped baseline for spectra with a gentle trend across the fit
window, e.g. the shoulder of a distant intense line or low-frequency
1/f-like leakage. Parameters: ``bg`` (a.u.), ``slope`` (a.u./MHz). Prefer
``ConstantBackground`` unless the slope is clearly visible — the slope is
strongly correlated with the peak parameters in narrow windows.
