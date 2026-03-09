"""Fourier analysis panel — window selection, FFT controls, display options.

Mirrors WiMDA's Analyse → Fourier dialog.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FourierPanel(QWidget):
    """Controls for frequency-domain analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Transform settings
        settings = QGroupBox("Settings")
        form = QFormLayout(settings)

        self._window_combo = QComboBox()
        self._window_combo.addItems(["none", "gaussian", "hann", "cosine", "lorentzian"])
        form.addRow("Window:", self._window_combo)

        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(1, 16)
        self._padding_spin.setValue(1)
        form.addRow("Zero-pad factor:", self._padding_spin)

        self._display_combo = QComboBox()
        self._display_combo.addItems(["Real", "Magnitude", "Power"])
        form.addRow("Display:", self._display_combo)

        layout.addWidget(settings)

        # Action button
        self._fft_btn = QPushButton("Compute FFT")
        layout.addWidget(self._fft_btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        layout.addStretch()
