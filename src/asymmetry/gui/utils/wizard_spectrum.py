"""Render the single wizard's captured spectrum without numerical replay."""

import numpy as np

from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.plot_decimation import decimate_for_preview

_MAX_FINGERPRINT_POINTS = 2000


def populate_fingerprint_plot(self, *, scan_label: str) -> None:
    if self._dataset is None:
        self._fft_ax = None
        self._user_peak_artists = []
        self._draw_figure(self._fingerprint_plot_widget, None)
        return
    widget = self._fingerprint_plot_widget
    figure = widget._figure
    canvas = widget._canvas
    if figure is None or canvas is None:
        return

    figure.clear()
    ax_time = figure.add_subplot(2, 1, 1)
    ax_fft = figure.add_subplot(2, 1, 2)
    # Display decimation changes neither the science observation nor the
    # captured diagnostic spectrum; its frequencies retain the prepared µs grid.
    plot_time, plot_asymmetry, plot_error = decimate_for_preview(
        np.asarray(self._dataset.time, dtype=float),
        np.asarray(self._dataset.asymmetry, dtype=float),
        np.asarray(self._dataset.error, dtype=float),
        _MAX_FINGERPRINT_POINTS,
    )
    ax_time.errorbar(
        plot_time,
        plot_asymmetry,
        yerr=plot_error,
        fmt=".",
        markersize=3,
        color=tokens.PLOT_DATA,
    )
    ax_time.set_xlabel("Time (µs)")
    ax_time.set_ylabel("Asymmetry")
    ax_time.set_title("Time Domain")

    preview = self._spectral_preview
    if preview is not None:
        ax_fft.plot(preview.frequency, preview.magnitude, color=tokens.ACCENT)
    ax_fft.set_xlabel("Frequency (MHz)")
    ax_fft.set_ylabel("|FFT|")
    fixed = (
        preview is not None
        and preview.observation.attribution.to_dict()["preparation"] == "fixed_native_support"
    )
    ax_fft.set_title("Windowed FFT (fixed native diagnostic)" if fixed else "Windowed FFT")

    self._fft_ax = ax_fft
    self._user_peak_artists = []
    analysis = self._recommendation.peak_analysis if self._recommendation else None
    if analysis is not None:
        scan_label: str | None = scan_label
        for peak in analysis.peaks:
            if peak.source in ("fft", "residual_fft"):
                ax_fft.axvline(
                    peak.frequency_mhz,
                    color=tokens.ACCENT,
                    alpha=0.35,
                    linewidth=1.2,
                )
            elif peak.source in ("early_fft", "damped_scan"):
                # This plot is the Hann-windowed transform, which is exactly
                # the view a heavily damped line is absent from. Mark it in a
                # distinct style so the seed is visible and obviously not a
                # feature of the spectrum drawn underneath it: dotted for the
                # early-window crop ladder, dashed for the matched-apodisation
                # scan, which also names itself in the legend.
                is_scan = peak.source == "damped_scan"
                ax_fft.axvline(
                    peak.frequency_mhz,
                    color=tokens.ACCENT,
                    alpha=0.55,
                    linewidth=1.2,
                    linestyle="--" if is_scan else ":",
                    label=scan_label if is_scan else None,
                )
                if is_scan:
                    scan_label = None  # one legend entry, not one per line
        if scan_label is None:
            ax_fft.legend(loc="upper right", fontsize="x-small", framealpha=0.8)
    self._refresh_peak_overlays()
    canvas.draw_idle()
