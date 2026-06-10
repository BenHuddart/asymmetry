.. _fit-frequency-domain:

Frequency domain
================

These components fit the real-valued Fourier spectrum displayed in the
Frequency workspace, as a function of frequency :math:`\nu` in MHz (see
:doc:`../frequency_domain_fitting` for the workflow). They are only offered
when fitting in the frequency domain, where the picker presents them as a
flat list. The natural pairing of line shape to physics mirrors the time
domain: an exponentially relaxing (dynamically broadened) signal gives a
Lorentzian line, while a static Gaussian field distribution gives a Gaussian
line, with the FWHM related to the time-domain rate by
:math:`\mathrm{FWHM} = \lambda/\pi` (Lorentzian, rate :math:`\lambda`) and
:math:`\mathrm{FWHM} = 2\sigma\sqrt{\ln 2}/\pi` (Gaussian, rate
:math:`\sigma` in the :math:`e^{-(\sigma t)^2}` convention).

.. _fit-gaussian-peak:

GaussianPeak
------------

.. math::

   S(\nu) = h\,\exp\!\left[-4\ln 2\,\frac{(\nu-\nu_0)^2}{w^2}\right]

A Gaussian spectral line of height :math:`h`, centre :math:`\nu_0` and full
width at half maximum :math:`w` — appropriate when the underlying time-domain
envelope is Gaussian (static, dense field distribution). The centre converts
to a local field through :math:`B_0 = 2\pi\nu_0/\gamma_\mu`.

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

   S(\nu) = \frac{h}{1 + 4\,(\nu-\nu_0)^2/w^2}

A Lorentzian spectral line — appropriate when the time-domain envelope is
exponential (dynamic broadening, or a dilute static field distribution).
Parameters as for ``GaussianPeak``. Note the heavy Lorentzian tails: fit
windows should extend several FWHM beyond the peak or the width will be
underestimated.

.. _fit-constant-background:

ConstantBackground
------------------

.. math::

   S(\nu) = b_g

A flat spectral baseline, absorbing white noise and the flat part of any
apodization pedestal. Parameter: ``bg`` (a.u.).

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
