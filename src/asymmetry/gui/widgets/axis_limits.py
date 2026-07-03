"""Shared numeric text field for axis/fit-range limits.

Converges the two ``_FloatLimitField`` implementations that grew up
independently in ``fit_panel.py`` (fit-range min/max) and ``plot_panel.py``
(plot axis min/max, later reused by ``alc_panel.py``): a plain ``QLineEdit``
with no spin arrows, in the design's compact limit-field style, exposing the
small spinbox-compatible surface (``value``/``setValue``/``decimals``/
``setDecimals``/``setRange``) that callers already depend on.

The fit_panel variant was the featured one — it clamps out-of-range values
(programmatic ``setValue``/``setRange`` and focus-out of "Intermediate"
input, which ``QDoubleValidator`` alone does not reject) and forces a commit
on Return/Enter and on focus-out, so a value the validator only rates
Intermediate (not yet Acceptable) still reaches ``editingFinished`` instead of
silently reverting on the next external refresh. :class:`FloatLimitField`
adopts that behavior for every call site.

Per-call-site validator ranges are preserved rather than converged: fit-range
fields keep their historical ±1000 (later widened to ±1e6 for the frequency
domain via ``setRange``), while plot/ALC axis-limit fields keep their
historical ±1e6 so a legitimate axis limit (e.g. a frequency axis in
thousands of MHz) is never silently clamped.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QKeyEvent
from PySide6.QtWidgets import QLineEdit, QSizePolicy, QWidget

from asymmetry.gui.styles.fonts import mono_font


class FloatLimitField(QLineEdit):
    """Compact text field for a numeric axis/fit-range limit (min/max).

    Replaces ``QDoubleSpinBox`` for axis and fit-range limits: a plain typed
    field (the design's limit-field style) with no spin arrows and no
    reserved arrow padding, so a min/max pair stays narrow in a dock or
    toolbar. A ``QDoubleValidator`` keeps entries numeric and in range, and
    out-of-range values are clamped the way a spin box would clamp them
    (``QDoubleValidator`` on its own only rejects out-of-range *keystrokes*,
    not a programmatic ``setValue``/``setRange`` or an "Intermediate" value
    committed on focus-out).

    Exposes the small spinbox-compatible surface (``value``/``setValue``/
    ``decimals``/``setDecimals``/``setRange``/``set_unset``) that the shared
    fit-range and axis-limit plumbing relies on, and keeps ``editingFinished``
    (a built-in ``QLineEdit`` signal) as the commit signal external handlers
    already connect to.

    Parameters
    ----------
    value:
        Initial value.
    decimals:
        Display precision. Defaults to 3 (fit_panel's default).
    minimum_width, maximum_width:
        Pixel width bounds. ``maximum_width=None`` leaves the field
        unbounded above (plot_panel/alc_panel do not cap width; fit_panel
        does, to keep the min/max pair compact on a 13" dock).
    value_range:
        ``(minimum, maximum)`` passed to the ``QDoubleValidator`` and used to
        clamp ``setValue``/typed input. Defaults to fit_panel's historical
        ±1000; plot_panel/alc_panel call sites pass ``(-1e6, 1e6)`` to
        preserve their historical, much wider axis range.
    """

    def __init__(
        self,
        value: float = 0.0,
        *,
        decimals: int = 3,
        minimum_width: int = 56,
        maximum_width: int | None = 88,
        value_range: tuple[float, float] = (-1000.0, 1000.0),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = max(0, int(decimals))
        self._value = float(value)
        self._validator = QDoubleValidator(value_range[0], value_range[1], self._decimals, self)
        self._validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(self._validator)
        self.setFont(mono_font(11.0))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setClearButtonEnabled(False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # A bare QLineEdit sizes to ~17 chars; cap it so a min/max pair stays
        # compact (fits "-1000.000" with room to spare).
        self.setMinimumWidth(minimum_width)
        if maximum_width is not None:
            self.setMaximumWidth(maximum_width)
        self._value = self._clamp(self._value)
        self.setText(self._format(self._value))
        # Normalise the display after a manual edit (e.g. "1" -> "1.000"). This
        # connects first, so an external editingFinished handler reads the
        # already-normalised text via value().
        self.editingFinished.connect(self._normalise_text)

    def _format(self, value: float) -> str:
        return f"{float(value):.{self._decimals}f}"

    def _clamp(self, value: float) -> float:
        """Clamp to the validator's range, matching QDoubleSpinBox.

        ``QDoubleValidator`` only rejects out-of-range *keystrokes*; it does
        not bound a programmatic ``setValue``/``setRange`` or an Intermediate
        entry committed on focus-out. The spin boxes this field replaces
        clamped both, so do the same here — otherwise an out-of-range limit
        could reach the engine or the plot.
        """
        return min(max(float(value), self._validator.bottom()), self._validator.top())

    def _normalise_text(self) -> None:
        self.setText(self._format(self.value()))

    def _commit(self) -> None:
        """Normalise the text and fire ``editingFinished`` unconditionally.

        ``QLineEdit`` only emits ``editingFinished`` when ``hasAcceptableInput()``
        is true, so a value the ``QDoubleValidator`` rates *Intermediate* (e.g.
        a digit just over the soft range, or a half-typed entry committed with
        Return) never reaches external plumbing and the field silently reverts
        on the next external refresh. Driving the signal ourselves makes a
        commit land reliably regardless of how the text was entered.
        """
        self._normalise_text()
        self.editingFinished.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override
        """Commit on Return/Enter, consuming the key.

        Returning early (rather than calling super) both guarantees the
        commit and stops the key bubbling to any default button that would
        push the pre-edit value back into the field.
        """
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._commit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Guarantee a commit on focus-out even for Intermediate input.

        For Acceptable input the base class already emits ``editingFinished``;
        only force our own commit when it would otherwise be suppressed, so
        the commit fires exactly once either way.
        """
        forced = not self.hasAcceptableInput()
        super().focusOutEvent(event)
        if forced:
            self._commit()

    def value(self) -> float:
        """Current value (clamped to range), or the last set value if blank."""
        try:
            return self._clamp(float(self.text()))
        except ValueError:
            return self._value

    def setValue(self, value: float) -> None:  # noqa: N802 — spinbox-API shim
        self._value = self._clamp(value)
        self.setText(self._format(self._value))

    def set_unset(self, placeholder: str) -> None:
        """Blank the field and show *placeholder* text (no value has been set)."""
        self.clear()
        self.setPlaceholderText(placeholder)

    def decimals(self) -> int:
        return self._decimals

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 — spinbox-API shim
        self._decimals = int(decimals)
        self._validator.setDecimals(self._decimals)
        self.setText(self._format(self.value()))

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802 — spinbox-API shim
        self._validator.setRange(minimum, maximum, self._decimals)
