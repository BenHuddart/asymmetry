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
thousands of MHz) is never silently clamped. The one exception is the
frequency plot panel's Y pair (``AxisLimitControls(y_value_range=...)``): a
grouped-count FFT magnitude scales with the event count and legitimately
exceeds 1e6 on a high-statistics run, so clamping there hid the spectrum
above the top of the axis.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QWidget

from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.widgets import build_nav_button_qss


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
    commit_on_set_value:
        When true, a programmatic ``setValue`` also fires ``editingFinished``
        (still suppressed by ``QSignalBlocker`` like any other signal), so a
        driven value change reaches the same commit plumbing as a typed entry.
        The fit-range fields opt in: their value is *state owned elsewhere*
        (the plot panel's fit range), so a silent ``setValue`` leaves the fit
        running over the old range while the field displays the new one. The
        display-mirror path (``_apply_fit_range_display``) wraps its
        ``setValue`` in ``QSignalBlocker``, so mirroring the owner's value back
        into the field stays silent and there is no feedback loop. Axis-limit
        fields keep the default (off): their many mirror call sites write
        ``setValue`` unblocked and must not re-trigger an axis commit.
    """

    def __init__(
        self,
        value: float = 0.0,
        *,
        decimals: int = 3,
        minimum_width: int = 56,
        maximum_width: int | None = 88,
        value_range: tuple[float, float] = (-1000.0, 1000.0),
        commit_on_set_value: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._commit_on_set_value = bool(commit_on_set_value)
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
        # Unset support (see set_unset): once a field has been given an unset
        # placeholder it is "unset-capable", and blank text on it means "no
        # value" rather than a half-finished edit — _normalise_text must then
        # leave the blank alone instead of rewriting the stored value back,
        # or clearing the field could never stick (it would visibly snap back
        # after any earlier setValue, e.g. a project restore).
        self._unset_capable = False
        self._unset = False
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

        NaN needs its own guard: ``min``/``max`` pass it straight through
        (every comparison is False), so without one a programmatic
        ``setValue(nan)`` would take residence in the field, round-trip
        through QSettings at shutdown, and crash the next startup when the
        restored limit reaches ``Axes.set_ylim``. Fall back to the last good
        value instead. ±Inf already clamps to the range ends.
        """
        value = float(value)
        if math.isnan(value):
            value = self._value if math.isfinite(self._value) else 0.0
        return min(max(value, self._validator.bottom()), self._validator.top())

    def _normalise_text(self) -> None:
        if self._unset_capable and not self.text().strip():
            # A committed blank on an unset-capable field is the unset state
            # (placeholder shows); rewriting the stored value here would make
            # clearing the field impossible.
            self._unset = True
            return
        self._unset = False
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
        self._unset = False
        self.setText(self._format(self._value))
        # Opt-in two-way binding: a driven value change commits like a typed
        # one so it reaches the owner of the underlying state. Callers that
        # only want to mirror external state blank the signal with
        # ``QSignalBlocker`` (as the fit-range display path does).
        if self._commit_on_set_value:
            self.editingFinished.emit()

    def set_unset(self, placeholder: str) -> None:
        """Blank the field and show *placeholder* text (no value has been set)."""
        self.clear()
        self.setPlaceholderText(placeholder)
        self._unset_capable = True
        self._unset = True

    def is_unset(self) -> bool:
        """True when the field is blank in its explicit unset state.

        Only meaningful on unset-capable fields (those given a placeholder via
        :meth:`set_unset`); plain limit fields always return False.
        """
        return self._unset_capable and self._unset and not self.text().strip()

    def decimals(self) -> int:
        return self._decimals

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 — spinbox-API shim
        self._decimals = int(decimals)
        self._validator.setDecimals(self._decimals)
        self.setText(self._format(self.value()))

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802 — spinbox-API shim
        self._validator.setRange(minimum, maximum, self._decimals)


class AxisLimitControls(QWidget):
    """Builds and owns the standard X/Y axis-limit row (fields + Auto buttons).

    Converges the ``row0``/``row`` assembly duplicated in
    ``PlotPanel._create_limit_controls`` and ``ALCPanel._build_limit_controls``:
    an ``X:`` label, min/max :class:`FloatLimitField` pair, a ``Y:`` label,
    min/max pair, and checkable "Auto X"/"Auto Y" buttons in the nav-button
    style, all in one row.

    This is a pure builder/holder — it constructs the widgets, exposes them as
    attributes, and lays them out on itself, but does **not** wire any
    ``editingFinished``/``clicked``/``toggled`` handlers. The two call sites
    connect their own handlers to the exposed widgets, which is deliberate:
    ``PlotPanel`` reacts to the Auto buttons' ``clicked`` signal while
    ``ALCPanel`` reacts to ``toggled`` (different semantics, both preserved by
    leaving the wiring to the caller).

    Parameters
    ----------
    field_width:
        ``minimum_width`` passed to each :class:`FloatLimitField` (fields are
        otherwise unbounded above, i.e. ``maximum_width=None``).
    show_units:
        When true, also builds ``.x_unit_label``/``.y_unit_label`` (empty
        ``QLabel``s placed after the x/y max fields) for the caller to set
        text on. When false, those attributes are ``None`` and no unit label
        appears in the row.
    auto_checked:
        Initial ``setChecked`` state for both Auto buttons.
    value_range:
        ``(minimum, maximum)`` forwarded to every :class:`FloatLimitField`.
    initial_values:
        ``(x_min, x_max, y_min, y_max)`` seed values for the four fields.
        Defaults to ``PlotPanel``'s historical defaults; ``ALCPanel`` passes
        its own historical ``(0.0, 1.0, 0.0, 1.0)`` to preserve its distinct
        initial display.
    """

    def __init__(
        self,
        *,
        field_width: int = 76,
        show_units: bool = False,
        auto_checked: bool = False,
        value_range: tuple[float, float] = (-1e6, 1e6),
        y_value_range: tuple[float, float] | None = None,
        initial_values: tuple[float, float, float, float] = (0.0, 10.0, -30.0, 30.0),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # The Y pair can carry a wider range than X: the frequency plot
        # panel's spectrum magnitude scales with the event count, while its
        # x-axis (MHz) never approaches the historical ±1e6 guard.
        y_range = y_value_range if y_value_range is not None else value_range
        x_min_v, x_max_v, y_min_v, y_max_v = initial_values
        self.x_min = FloatLimitField(
            x_min_v, minimum_width=field_width, maximum_width=None, value_range=value_range
        )
        self.x_max = FloatLimitField(
            x_max_v, minimum_width=field_width, maximum_width=None, value_range=value_range
        )
        self.y_min = FloatLimitField(
            y_min_v, minimum_width=field_width, maximum_width=None, value_range=y_range
        )
        self.y_max = FloatLimitField(
            y_max_v, minimum_width=field_width, maximum_width=None, value_range=y_range
        )

        self.x_unit_label: QLabel | None = QLabel("") if show_units else None
        self.y_unit_label: QLabel | None = QLabel("") if show_units else None

        nav_qss = build_nav_button_qss()
        self.auto_x_btn = QPushButton("Auto X")
        self.auto_x_btn.setCheckable(True)
        self.auto_x_btn.setStyleSheet(nav_qss)
        self.auto_x_btn.setMaximumWidth(65)
        self.auto_x_btn.setChecked(auto_checked)

        self.auto_y_btn = QPushButton("Auto Y")
        self.auto_y_btn.setCheckable(True)
        self.auto_y_btn.setStyleSheet(nav_qss)
        self.auto_y_btn.setMaximumWidth(65)
        self.auto_y_btn.setChecked(auto_checked)

        row = QHBoxLayout(self)
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("X:"))
        row.addWidget(self.x_min)
        row.addWidget(QLabel("–"))
        row.addWidget(self.x_max)
        if self.x_unit_label is not None:
            row.addWidget(self.x_unit_label)
        row.addWidget(QLabel("Y:"))
        row.addWidget(self.y_min)
        row.addWidget(QLabel("–"))
        row.addWidget(self.y_max)
        if self.y_unit_label is not None:
            row.addWidget(self.y_unit_label)
        row.addWidget(self.auto_x_btn)
        row.addWidget(self.auto_y_btn)
        row.addStretch()
