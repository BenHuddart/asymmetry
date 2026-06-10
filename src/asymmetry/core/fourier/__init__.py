"""Frequency-domain analysis: FFT, grouped inputs, and apodization."""

from asymmetry.core.fourier.fft import (
    average_fourier_display_values,
    canonical_fourier_display_mode,
    estimate_fft_phase,
    exclude_frequency_ranges,
    fft_asymmetry,
    fft_complex_asymmetry,
    fourier_display_values,
    fourier_mode_uses_entropy_optimizer,
    fourier_mode_uses_phase_correction,
    optimize_phase_entropy,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
    fourier_display_ylabel,
    precompute_group_fourier_inputs,
)
from asymmetry.core.fourier.units import (
    FieldUnit,
    axis_label,
    convert,
    frequency_resolution_mhz,
    gauss_to_mhz,
    gauss_to_tesla,
    mhz_to_gauss,
    mhz_to_tesla,
    tesla_to_gauss,
    tesla_to_mhz,
)
from asymmetry.core.fourier.window import apply_window

__all__ = [
    "FieldUnit",
    "axis_label",
    "convert",
    "frequency_resolution_mhz",
    "gauss_to_mhz",
    "gauss_to_tesla",
    "mhz_to_gauss",
    "mhz_to_tesla",
    "tesla_to_gauss",
    "tesla_to_mhz",
    "fft_asymmetry",
    "fft_complex_asymmetry",
    "exclude_frequency_ranges",
    "average_fourier_display_values",
    "canonical_fourier_display_mode",
    "estimate_fft_phase",
    "fourier_display_values",
    "fourier_mode_uses_entropy_optimizer",
    "fourier_mode_uses_phase_correction",
    "optimize_phase_entropy",
    "build_group_signal_dataset",
    "GroupSpectrumConfig",
    "compute_average_group_spectrum",
    "fourier_display_ylabel",
    "precompute_group_fourier_inputs",
    "apply_window",
]
