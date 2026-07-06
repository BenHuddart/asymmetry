"""Dialog for fitting parameter trends vs field/temperature."""

from __future__ import annotations

import os
import re
import traceback
from collections.abc import Callable, Sequence

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.composite import QUADRATURE_OPERATOR, UnknownComponentError
from asymmetry.core.fitting.experiment_design import (
    NextPointSuggestion,
    SuggestionCalibration,
    aic_weights,
    calibrate_suggestion,
    cost_weighted_utility,
    suggest_discriminating_point,
    suggest_next_point,
)
from asymmetry.core.fitting.fit_quality import assess_fit_quality
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ErrorMode,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    apply_error_mode,
    carve_window_gap,
    component_names_for_x,
    fit_parameter_model,
    included_intervals,
    is_order_parameter_observable,
    sample_parameter_model,
    set_included_intervals,
    suggest_model_seeds,
    suggest_trend_seeds,
    validate_fit_windows,
    windows_mask,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.fit_settings import fit_quality_confidence
from asymmetry.gui.styles import metrics, tokens
from asymmetry.gui.styles.widgets import (
    _FIT_VERDICT_COLOURS,
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    apply_param_table_style,
    clear_layout,
    error_html,
    fit_quality_chip_html,
    info_html,
    make_formula_box,
    make_section,
    success_html,
    warning_html,
)
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.axis_limits import FloatLimitField
from asymmetry.gui.widgets.function_builder.dialog import (
    FunctionBuilderDialog,
    make_component_expression_parser,
)
from asymmetry.gui.widgets.range_card import RangeCard, RangeCardView
from asymmetry.gui.widgets.screen_sizing import resize_to_available
from asymmetry.gui.widgets.trend_preview import (
    PreviewRange,
    PreviewSeries,
    SuggestionOverlay,
    TrendPreviewCanvas,
    range_span_color,
)
from asymmetry.gui.windows.new_user_function_dialog import NewUserFunctionDialog

#: D-optimal combo entry (userData sentinel; see _on_suggest_target_changed).
_SUGGEST_ALL_PARAMS = "__all__"

#: Compare-against combo entry for a composite built via Edit… (userData
#: sentinel; see _adopt_custom_compare_model). The selected entry is the
#: single source of truth for what "Fit & compare" fits.
_COMPARE_CUSTOM = "__custom__"

#: Operators offered by the parameter/trending builder — the base arithmetic
#: set plus the quadrature combinator ``⊕`` (``f ⊕ g = sqrt(f**2 + g**2)``),
#: e.g. ``PowerLaw ⊕ Constant`` reproduces ``PowerLawQuadBG``.
_PARAMETER_MODEL_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/", QUADRATURE_OPERATOR)

_OPERATOR_OPTIONS = ["+", "-", "*", "/"]

#: (label, ErrorMode, value-field meaning) for the dialog-level error selector.
_ERROR_MODE_OPTIONS: list[tuple[str, ErrorMode, str | None]] = [
    ("Column (propagated errors)", ErrorMode.COLUMN, None),
    ("Percent of y", ErrorMode.PERCENT, "%"),
    ("Absolute", ErrorMode.ABSOLUTE, "σ"),
    ("None (unit weights)", ErrorMode.NONE, None),
    ("Estimate from scatter", ErrorMode.SCATTER, None),
]

_ERROR_MODE_TOOLTIP = (
    "How each point is weighted in the fit.\n"
    "Column: the propagated errors of the trended parameter (default).\n"
    "Percent of y: σᵢ = (value/100)·|yᵢ|.\n"
    "Absolute: one constant σ for every point.\n"
    "None: unit weights — χ² loses its absolute meaning.\n"
    "Estimate from scatter: unit-weight fit whose parameter errors are\n"
    "rescaled by √(χ²/ν), i.e. errors estimated from the scatter of the\n"
    "points themselves. Use it when the trend has no trustworthy error bars."
)

_QUALITY_TOOLTIP = (
    "For a correct model with correct error bars, χ² follows the chi-squared\n"
    "distribution with ν = N − N_free degrees of freedom; at 95 % confidence a\n"
    "good fit's reduced χ² falls inside the band shown (the band tightens\n"
    "toward 1 as ν grows). Above the band (poor): the model misses real\n"
    "structure, or the errors are underestimated. Below it (overdone): the fit\n"
    "reproduces the data better than the errors allow — usually overestimated\n"
    "errors or too many free parameters. The verdict assumes real (Column-mode)\n"
    "errors; with unit weights or scatter-estimated errors χ²ᵣ carries no\n"
    "goodness information and no verdict is shown."
)

_Y_PARAM_UNITS = {
    "A": "%",
    "A0": "%",
    "A_bg": "%",
    "baseline": "%",
    "Lambda": "us^-1",
    "sigma": "us^-1",
    "Delta": "us^-1",
    "frequency": "MHz",
    "phase": "rad",
}

_PARAM_UNITS = {
    "a": None,
    "b": None,
    "c": None,
    "f": "us^-1",
    "A": "MHz",
    "D": "MHz",
    "nu": "MHz",
    "m": None,
    "tau": "(x units)",
    "B0": "G",
    "Bwid": "G",
    "Tc": "K",
    "Ea": "meV",
    "C": "MHz",  # legacy alias used in older saved model-fit states
    "D_2D": "us^-1",
    "D_nD": "us^-1",
    "D_perp": "us^-1",
    "lambda_BG": "us^-1",
    "lambda_0D": "us^-1",
}

_NON_NEGATIVE_PARAMS = {"D", "D_2D", "D_nD", "D_perp", "lambda_BG", "lambda_0D", "f"}
_STRICTLY_POSITIVE_PARAMS = {"tau", "B0", "Bwid", "nu", "m"}
_POSITIVE_EPS = 1e-12


def _base_param_name(name: str) -> str:
    match = re.match(r"^(.+)_\d+$", name)
    return match.group(1) if match else name


def _x_unit(x_key: str) -> str | None:
    if x_key == "field":
        return "G"
    if x_key == "temperature":
        return "K"
    if x_key == "run":
        return "run"
    return None


def _y_unit(parameter_name: str) -> str | None:
    return _Y_PARAM_UNITS.get(_base_param_name(parameter_name))


def _format_param_label(name: str, x_key: str, parameter_name: str) -> str:
    """Return display label with units for range-parameter table."""
    base = _base_param_name(name)
    y_unit = _y_unit(parameter_name)
    x_unit = _x_unit(x_key)

    if base == "tau":
        unit = x_unit or "(x units)"
    elif base == "m":
        if y_unit and x_unit:
            unit = f"{y_unit} / {x_unit}"
        elif y_unit:
            unit = f"{y_unit} / x"
        else:
            unit = "(y units / x unit)"
    elif base in {"a", "b", "c"}:
        unit = y_unit or "(y units)"
    else:
        unit = _PARAM_UNITS.get(base)

    return f"{name} [{unit}]" if unit else name


def _format_model_param_label(
    model: ParameterCompositeModel,
    name: str,
    x_key: str,
    parameter_name: str,
) -> str:
    """Return display label for a specific model parameter.

    Keeps Redfield exponent ``m`` unitless while using unit-aware labels for
    all other parameters.
    """
    component_for_param: dict[str, str] = {}
    for mapping, component in zip(model._param_mappings, model.components, strict=True):
        for unique_name in mapping.values():
            component_for_param[unique_name] = component.name

    if _base_param_name(name) == "m" and component_for_param.get(name) == "Redfield":
        return name
    return _format_param_label(name, x_key, parameter_name)


def _component_name_for_param(model: ParameterCompositeModel, name: str) -> str | None:
    """Return component name that owns a unique model parameter name."""
    for mapping, component in zip(model._param_mappings, model.components, strict=True):
        for unique_name in mapping.values():
            if unique_name == name:
                return component.name
    return None


def _should_reset_param_on_model_change(model: ParameterCompositeModel, name: str) -> bool:
    """Return True when model changes should prefer defaults over name-based carryover."""
    return _base_param_name(name) == "m" and _component_name_for_param(model, name) == "Redfield"


def _component_pool_for_context(x_key: str, parameter_name: str) -> list[str]:
    """Return component pool with context-specific redundancy filtering."""
    available = component_names_for_x(x_key)
    if x_key != "field":
        return available

    base = _base_param_name(parameter_name).strip().lower()
    is_lambda_like = base.startswith("lambda") or base.startswith("λ")

    if is_lambda_like:
        return [name for name in available if name != "Constant"]
    return [name for name in available if name != "Lambda_bg"]


def _default_component_for_context(x_key: str, parameter_name: str, available: list[str]) -> str:
    """Choose the default trend component for a fresh range (F4).

    Linear is the safe default in every ordinary context. When the trend is a
    magnetic order-parameter observable (precession frequency / internal field)
    versus temperature, default to ``OrderParameter`` instead — its data-aware
    ``T_c``/amplitude seeds (see :func:`suggest_trend_seeds`) then make the fit
    converge out of the box, so the user need not type the component by name.
    The "is this an order-parameter observable" judgement lives in core
    (:func:`is_order_parameter_observable`), beside the seed logic.
    """
    if (
        x_key == "temperature"
        and is_order_parameter_observable(parameter_name)
        and "OrderParameter" in available
    ):
        return "OrderParameter"
    if "Linear" in available:
        return "Linear"
    return available[0] if available else "Constant"


def _normalize_parameter_limits(
    name: str,
    value: float,
    p_min: float,
    p_max: float,
) -> tuple[float, float, float, list[str]]:
    """Normalize start/min/max based on model-domain expectations."""
    notes: list[str] = []
    base = _base_param_name(name)

    if p_min > p_max:
        p_min, p_max = p_max, p_min
        notes.append(f"{name}: swapped min/max")

    if base in _NON_NEGATIVE_PARAMS:
        if p_min < 0.0:
            p_min = 0.0
            notes.append(f"{name}: min clamped to 0")
        if p_max < p_min:
            p_max = p_min + 1.0
            notes.append(f"{name}: max raised above min")

    if base in _STRICTLY_POSITIVE_PARAMS:
        if p_min < _POSITIVE_EPS:
            p_min = _POSITIVE_EPS
            notes.append(f"{name}: min clamped to >0")
        if p_max < p_min:
            p_max = p_min * 10.0
            notes.append(f"{name}: max raised above positive min")

    clamped_value = min(max(value, p_min), p_max)
    if clamped_value != value:
        value = clamped_value
        notes.append(f"{name}: start value clamped to bounds")

    return value, p_min, p_max, notes


def _in_test_mode() -> bool:
    """Return True when running under pytest to avoid modal popups."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _show_info(parent: QWidget, title: str, text: str) -> None:
    """Show informational message in interactive mode only."""
    if _in_test_mode():
        return
    QMessageBox.information(parent, title, text)


def _show_warning(parent: QWidget, title: str, text: str) -> None:
    """Show warning message in interactive mode only."""
    if _in_test_mode():
        return
    QMessageBox.warning(parent, title, text)


def _last_model_memory_key(parameter_name: str, x_key: str) -> str:
    """Memory-dict key for the remembered model of ``(base_param_name, x_key)``.

    The parameter name is reduced to its base (``Lambda_2`` → ``Lambda``) so a
    per-group index does not fragment the memory across otherwise-identical
    trends.
    """
    return f"{_base_param_name(parameter_name)}|{x_key}"


def _store_last_model_expression(
    parameter_name: str,
    x_key: str,
    expression: str,
    memory: dict[str, str],
) -> None:
    """Persist *expression* as the last-used model for ``(parameter, x_key)``.

    Entirely best-effort: it confirms the string round-trips through
    ``ParameterCompositeModel.from_expression`` before storing, and swallows any
    failure — a broken store must never disrupt the fit UI. *memory* is a
    plain ``dict[str, str]`` owned by the caller (e.g. ``FitParametersPanel`` /
    ``GlobalParameterFitWindow``), which persists it into project state — so
    the memory is project-scoped rather than a global per-user setting.
    """
    if not expression:
        return
    try:
        # Only remember expressions that parse back, so restore never trips on
        # something we wrote.
        ParameterCompositeModel.from_expression(expression)
    except Exception:
        return
    try:
        memory[_last_model_memory_key(parameter_name, x_key)] = expression
    except Exception:
        pass


def _load_last_model(
    parameter_name: str,
    x_key: str,
    component_pool: set[str] | frozenset[str] | list[str],
    memory: dict[str, str],
) -> ParameterCompositeModel | None:
    """Return the remembered model for ``(parameter, x_key)``, or None.

    Restricts the parse to *component_pool* (so a component valid in another
    context but not offered here falls back), and returns None on any failure —
    a missing key, an unparseable string, or an out-of-pool component.
    """
    try:
        raw = memory.get(_last_model_memory_key(parameter_name, x_key))
    except Exception:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return _pool_restricted_model_parser(set(component_pool))(raw)
    except Exception:
        return None


def _pool_restricted_model_parser(
    pool: set[str] | frozenset[str],
) -> Callable[[str], ParameterCompositeModel]:
    """Wrap ``ParameterCompositeModel.from_expression`` to reject components
    outside *pool*.

    ``ParameterCompositeModel.from_expression`` parses against the full
    ``PARAMETER_MODEL_COMPONENTS`` registry (and always allows ``⊕``), so a
    registered component that is simply not offered in this context (e.g. a
    field-only basis function while trending vs temperature) would otherwise
    parse successfully if the user typed it in text mode. Re-validate the
    parsed component names against the context pool and raise the same
    ``UnknownComponentError``-style message the shared grammar uses elsewhere,
    so a mis-typed/out-of-context name gets a helpful, suggestion-bearing
    error instead of silently succeeding.

    *pool* is read at call time (not snapshotted), so passing the dialog's
    live ``set`` (rather than a ``frozenset`` copy) lets a component created
    mid-session (see :meth:`ParameterModelBuilderDialog._create_user_function`)
    become acceptable to this parser without rebuilding it.
    """

    def _parse(expression: str) -> ParameterCompositeModel:
        model = ParameterCompositeModel.from_expression(expression)
        offending = [name for name in model.component_names if name not in pool]
        if offending:
            raise UnknownComponentError(offending[0], allowed=pool)
        return model

    return _parse


class ParameterModelBuilderDialog(FunctionBuilderDialog):
    """Compose a parameter model from basis components."""

    def __init__(
        self,
        component_pool: list[str],
        parent: QWidget | None = None,
        initial_model: ParameterCompositeModel | None = None,
    ) -> None:
        self._component_pool = sorted(component_pool)
        # A live set (not a frozen snapshot): both the model parser and the
        # expression parser below capture *this same object*, so a component
        # authored mid-session via _create_user_function (which does
        # self._pool.add(name)) becomes acceptable to both without rebuilding
        # either closure. parse_component_expression / the model parser only
        # ever do membership checks against it, so mutating it in place is
        # safe — see _pool_restricted_model_parser / make_component_expression_parser.
        self._pool: set[str] = set(self._component_pool)
        component_definitions = {
            name: PARAMETER_MODEL_COMPONENTS[name]
            for name in self._component_pool
            if name in PARAMETER_MODEL_COMPONENTS
        }
        initial_expression = (
            initial_model.component_expression_string()
            if initial_model is not None
            else (self._component_pool[0] if self._component_pool else "Constant")
        )
        super().__init__(
            title="Build Parameter Model",
            expression_prefix="y(x)",
            component_definitions=component_definitions,
            model_parser=_pool_restricted_model_parser(self._pool),
            expression_parser=make_component_expression_parser(
                allowed_components=self._pool,
                allowed_operators=set(_PARAMETER_MODEL_OPERATORS),
            ),
            initial_expression=initial_expression,
            operators=_PARAMETER_MODEL_OPERATORS,
            enable_fraction_groups=False,
            on_create_user_function=self._create_user_function,
            parent=parent,
        )

    # ------------------------------------------------------------ authoring
    def _create_user_function(self) -> object | None:
        """Open the authoring dialog for a new parameter-trend component.

        The created component always registers with ``scopes=("common",)``
        (see ``NewUserFunctionDialog``/``create_user_function``), so it is
        valid in every trending context; add it to the live pool so both the
        model parser and the expression parser accept it immediately.
        """
        dialog = NewUserFunctionDialog("parameter", parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return None
        created = dialog.created()
        if created is None:
            return None
        self._pool.add(created.definition.name)
        return created.definition

    def get_model(self) -> ParameterCompositeModel | None:
        model = self.built_model()
        return model if isinstance(model, ParameterCompositeModel) else None


class ModelFitDialog(QDialog):
    """Configure and run model fits for one Y parameter vs selected X variable."""

    #: Subclasses whose fit backend does not honour the error-mode selector
    #: or per-range fit windows set these to False so the controls are never
    #: shown promising semantics the fit would silently ignore.
    _supports_error_modes: bool = True
    _supports_windows: bool = True
    #: Subclasses whose fit backend does not honour x-uncertainty (e.g. the
    #: cross-group fit) set this to False so the effective-variance toggle is
    #: never shown.
    _supports_x_errors: bool = True

    def __init__(
        self,
        parameter_name: str,
        x_key: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_errors: np.ndarray,
        existing_fit: ParameterModelFit | None = None,
        parent: QWidget | None = None,
        x_errors: np.ndarray | None = None,
        x_label: str | None = None,
        model_memory: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)

        # Caller-owned "remember last-used model per (parameter, x_key)" store
        # (item 4.2). A None caller (e.g. CrossGroupFitDialog, which does not
        # participate) gets a session-local dict that is never persisted, so
        # it simply never remembers across dialogs. Assigned EARLY — before
        # ``_create_default_range`` (below) reads it.
        self._model_memory: dict[str, str] = model_memory if model_memory is not None else {}

        # ``x_key`` is the internal abscissa id (e.g. ``custom:84576a7e``) used
        # by the fit backend / persistence; ``x_label`` is the friendly column
        # name shown to the user ("Current (A)"). Fall back to the key when no
        # label is supplied so older callers keep working.
        x_display = x_label or x_key
        self.setWindowTitle(f"Model Fit: {parameter_name} vs {x_display}")
        # Cap the default to the available screen — the wider default leaves room
        # for the preview pane on the right; the smaller floor keeps it usable on
        # a small display (P2-3).
        resize_to_available(self, 1280, 700, min_width=560, min_height=420)

        self._parameter_name = parameter_name
        self._x_key = x_key
        self._x_display = x_display
        self._component_pool = _component_pool_for_context(x_key, parameter_name)
        self._x = np.asarray(x_values, dtype=float)
        self._y = np.asarray(y_values, dtype=float)
        self._yerr = np.asarray(y_errors, dtype=float)
        self._xerr = None if x_errors is None else np.asarray(x_errors, dtype=float)
        self._removed = False
        # One RangeCard per fit range (recreated each _rebuild_ranges_ui).
        self._range_cards: list[RangeCard] = []
        # (min, max) spin pair per INCLUDED INTERVAL of the ACTIVE range, keyed by
        # interval index. The details pane only ever shows the active range's
        # fit-region intervals; rebuilt by ``_rebuild_fit_region_rows``.
        self._region_row_spins: dict[int, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        # Per-interval "Remove" buttons + the "Exclude region…" button, tracked
        # so ``_set_fit_ui_busy`` can disable them during a fit.
        self._region_remove_btns: list[QPushButton] = []
        self._exclude_region_btn: QPushButton | None = None
        self._active_range_idx: int | None = None
        self._fit_in_progress = False
        # Own busy flag for the user-initiated "Guess seeds" data-aware seeder
        # (item 3.3). Deliberately SEPARATE from ``_fit_in_progress`` so the
        # lightweight off-thread seed suggestion never drives ``_set_fit_ui_busy``
        # / locks the whole dialog the way a real fit does.
        self._guess_in_progress = False
        self._guess_target_idx: int | None = None
        # Background fits run on the shared TaskRunner (gui/tasks.py), which owns
        # the QThread/worker lifecycle and a bounded, Windows-safe shutdown.
        self._tasks = TaskRunner(self)
        self._fit_done_callback: Callable[[object], None] | None = None

        # ── "Suggest next point" state (BED, Phase 2 — §5.4) ─────────────────
        # The last suggestion computed for the active range (cleared on refit /
        # range switch / a fresh Suggest click); calibration runs off-thread on
        # the shared TaskRunner and its own busy flag (never _fit_in_progress).
        self._last_suggestion: NextPointSuggestion | None = None
        self._last_suggestion_calibration: SuggestionCalibration | None = None
        self._calibration_in_progress = False
        self._pending_calibration: tuple | None = None

        # ── Model discrimination state (BED, Phase 3 — §8.1) ──────────────────
        # Per-range list of fitted alternative candidates, keyed by range index.
        # Deliberately session-local (not on ModelFitRange / not serialized):
        # cleared whenever the range's primary fit reruns or its model
        # expression changes (see _clear_discrimination_candidates).
        # Each entry: (model, parameters, chi2, n_free).
        self._discrimination_candidates: dict[
            int, list[tuple[ParameterCompositeModel, ParameterSet, float, int]]
        ] = {}
        self._last_discrimination: NextPointSuggestion | None = None
        self._compare_fit_in_progress = False
        self._pending_compare: tuple | None = None

        if existing_fit is not None and existing_fit.ranges:
            self._fit = existing_fit
        else:
            self._fit = ParameterModelFit(parameter_name=parameter_name, x_key=x_key, ranges=[])
            self._fit.ranges.append(self._create_default_range())

        # The dialog body is a horizontal splitter: the LEFT pane carries every
        # existing control (unchanged order) while the RIGHT pane is a preview
        # host that a later work item fills with a plot canvas. Full-width
        # footer-slot + button box stay directly on the dialog's top layout,
        # beneath the splitter, so both span the whole dialog and remain direct
        # items of self.layout() (contract C6 / the cross-group footer test).
        #
        # The left pane's content (in particular each non-wrapping range row —
        # see _rebuild_ranges_ui) can force a minimum width well past 1000px.
        # Wrapping it in a QScrollArea decouples that content minimum from the
        # splitter's size negotiation: the scroll area itself only demands
        # _LEFT_PANE_MIN_WIDTH, and any wider row scrolls horizontally instead
        # of starving (or clipping) the preview pane on the right.
        #
        # Sized via metrics.dialog_width (not a class constant): it reads the
        # live application font, so it must be computed per-instance once a
        # QApplication is guaranteed to exist, not at class-definition time.
        self._LEFT_PANE_MIN_WIDTH = metrics.dialog_width(58)  # ~420px at default scale
        self._PREVIEW_PANE_MIN_WIDTH = metrics.dialog_width(47)  # ~340px at default scale
        top_layout = QVBoxLayout(self)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_pane)
        left_scroll.setMinimumWidth(self._LEFT_PANE_MIN_WIDTH)
        self._left_scroll = left_scroll

        right_pane = QWidget()
        right_pane.setMinimumWidth(self._PREVIEW_PANE_MIN_WIDTH)
        self._preview_host = QVBoxLayout(right_pane)
        self._preview_host.setContentsMargins(0, 0, 0, 0)
        # Live, off-thread candidate-curve preview (work item 1.3). Drag is
        # wired here (work item 2.2): dragging a range/window edge or right-drag
        # excluding a region routes back through the handlers below.
        self._preview = TrendPreviewCanvas()
        self._preview.enable_drag(True)
        self._preview.range_edge_dragged.connect(self._on_preview_range_edge_dragged)
        self._preview.window_edge_dragged.connect(self._on_preview_window_edge_dragged)
        self._preview.exclude_region_requested.connect(self._on_preview_exclude_region)
        # Add/select gestures go through an overridable hook so single-range
        # subclasses (cross-group) can leave them inert (item 3.2/3.3).
        self._connect_plot_range_signals()

        # "Show residuals" (item 4.1, relocated by P2): opt-in residual strip
        # beneath the preview plot. Lives in the preview pane itself — right
        # above the canvas it controls — rather than in the top-of-dialog
        # toggle row, so it reads as attached to the plot instead of orphaned.
        # Default UNCHECKED — the user opts in; keeping it off by default also
        # matches the narrow-screen "don't crowd the pane" stance. When the
        # preview pane auto-collapses on a narrow screen this checkbox rides
        # with it (hidden), which is correct — residuals are meaningless with
        # no visible preview.
        self._show_residuals_check = QCheckBox("Show residuals")
        self._show_residuals_check.setChecked(False)
        self._show_residuals_check.setToolTip(
            "Show a residual strip below the preview plot: (data − model) / σ for "
            "the selected range's in-fit points, with a ±1σ guide band. Uses the "
            "seed curve before a fit and the fitted curve after."
        )
        self._show_residuals_check.toggled.connect(self._on_show_residuals_toggled)
        residuals_row = QHBoxLayout()
        residuals_row.addWidget(self._show_residuals_check)
        residuals_row.addStretch()
        self._preview_host.addLayout(residuals_row)

        self._preview_host.addWidget(self._preview, 1)

        # ── Preview threading state (work item 1.3) ──────────────────────────
        # A dedicated generation token guards against stale worker results;
        # it is deliberately SEPARATE from ``_fit_in_progress`` so a preview
        # tick never drives ``_set_fit_ui_busy`` (which would lock the dialog).
        # ``_preview_active``/``_preview_pending`` enforce "at most one sample
        # in flight plus one coalesced request".
        self._preview_generation = 0
        self._preview_active = False
        self._preview_pending = False
        # Set in closeEvent before the TaskRunner is shut down, so a debounce
        # timer that fires during teardown cannot start a fresh worker on a
        # runner the dialog is done with (TaskRunner.shutdown has no re-entry
        # guard against a later start()).
        self._shutting_down = False
        self._last_preview_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._launch_preview_sample)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(left_scroll)
        self._splitter.addWidget(right_pane)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        # Placeholder sizes; _maybe_collapse_preview (called from showEvent)
        # replaces these with a proportional split computed from the dialog's
        # real width once it is known, so the preview opens at a usable width
        # instead of whatever this constructor-time guess happens to be.
        self._splitter.setSizes([560, 620])
        top_layout.addWidget(self._splitter)

        # First-show auto-collapse bookkeeping (see _maybe_collapse_preview).
        self._preview_collapsed = False
        self._preview_auto_managed = True
        self._preview_expanded_sizes = [560, 620]

        summary = QLabel(f"Y parameter: <b>{parameter_name}</b> | X variable: <b>{x_display}</b>")
        left_layout.addWidget(summary)

        x_min_data = float(np.nanmin(self._x)) if np.any(np.isfinite(self._x)) else 0.0
        x_max_data = float(np.nanmax(self._x)) if np.any(np.isfinite(self._x)) else 1.0
        self._x_min_data = x_min_data
        self._x_max_data = x_max_data
        self._data_range_label = QLabel(f"Data range: {x_min_data:.6g} – {x_max_data:.6g}")
        left_layout.addWidget(self._data_range_label)

        # "Show preview" toggle, hidden until a narrow width auto-collapses the
        # preview pane (see _maybe_collapse_preview). Checking it restores the
        # pane; unchecking re-collapses it.
        self._show_preview_toggle = QPushButton("Show preview")
        self._show_preview_toggle.setCheckable(True)
        self._show_preview_toggle.setChecked(True)
        self._show_preview_toggle.setVisible(False)
        self._show_preview_toggle.toggled.connect(self._on_show_preview_toggled)
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self._show_preview_toggle)
        toggle_row.addStretch()
        left_layout.addLayout(toggle_row)

        # Named insertion point (contract C6): directly under the summary/
        # data-range labels, at the top of the dialog body. Subclasses (e.g.
        # CrossGroupFitDialog's inherited-source banner) add widgets here
        # instead of doing index-based layout.insertWidget arithmetic. Empty by
        # default, so it must add no visual space.
        self._header_slot = QVBoxLayout()
        self._header_slot.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(self._header_slot)

        self._error_mode_combo: QComboBox | None = None
        self._error_value_label: QLabel | None = None
        self._error_value_spin: QDoubleSpinBox | None = None
        if self._supports_error_modes:
            error_row = QHBoxLayout()
            error_row.addWidget(QLabel("Errors:"))
            self._error_mode_combo = QComboBox()
            for label, mode, _meaning in _ERROR_MODE_OPTIONS:
                self._error_mode_combo.addItem(label, userData=mode.value)
            self._error_mode_combo.setToolTip(_ERROR_MODE_TOOLTIP)
            self._error_mode_combo.currentIndexChanged.connect(self._on_error_mode_changed)
            error_row.addWidget(self._error_mode_combo)
            self._error_value_label = QLabel("Value:")
            error_row.addWidget(self._error_value_label)
            self._error_value_spin = QDoubleSpinBox()
            self._error_value_spin.setRange(1e-12, 1e12)
            self._error_value_spin.setDecimals(6)
            self._error_value_spin.setValue(1.0)
            self._error_value_spin.setToolTip(
                "Percent of |y| (Percent mode) or the constant σ (Absolute mode)."
            )
            error_row.addWidget(self._error_value_spin)
            error_row.addStretch()
            left_layout.addLayout(error_row)
            self._on_error_mode_changed(0)

        # Effective-variance x-uncertainty toggle (item 1): only meaningful when
        # the abscissa is itself a fitted parameter and carries usable errors.
        self._x_error_check: QCheckBox | None = None
        x_has_err = self._xerr is not None and bool(
            np.any(np.isfinite(self._xerr) & (self._xerr > 0))
        )
        if self._supports_x_errors and x_key.startswith("param:") and x_has_err:
            xerr_row = QHBoxLayout()
            self._x_error_check = QCheckBox("Account for x uncertainty")
            self._x_error_check.setToolTip(
                "Weight the fit by the x-parameter's per-point uncertainty using "
                "the Orear/York effective-variance method (errors-in-variables). "
                "Unchecked = the x-axis is treated as exact (ordinary least "
                "squares)."
            )
            self._x_error_check.setChecked(bool(getattr(self._fit, "use_x_errors", False)))
            xerr_row.addWidget(self._x_error_check)
            xerr_row.addStretch()
            left_layout.addLayout(xerr_row)

        # Flat BENCH section (make_section) instead of a QGroupBox — the app
        # deliberately avoids QGroupBox chrome in inspector panels. The uppercase
        # header lives INSIDE the section container (above ``_ranges_host``), so a
        # ``_rebuild_ranges_ui`` clear_layout of ``_ranges_host`` never destroys it.
        #
        # ONE merged "Fit ranges" section (Step 2): the RangeCard stack + "Add
        # Range" button sit on top, then the "Selected range" fit-region editor
        # (the included-interval rows + "Exclude region…"), Guess row,
        # formula/result boxes, param table) directly below. The card IS the
        # range selector — clicking a card activates it and the details pane
        # below follows via ``_set_active_range``.
        params_group, params_layout = make_section("Fit ranges")

        self._ranges_host = QVBoxLayout()
        self._ranges_host.setSpacing(4)
        params_layout.addLayout(self._ranges_host)

        add_row = QHBoxLayout()
        add_btn = QPushButton("Add Range")
        add_btn.clicked.connect(self._add_range)
        add_row.addWidget(add_btn)
        add_row.addStretch()
        params_layout.addLayout(add_row)

        self._range_hint_label = QLabel("Editing the highlighted range.")
        params_layout.addWidget(self._range_hint_label)

        # P6: the plot's drag-to-add / click-to-select gestures are the least
        # discoverable part of this dialog, so spell them out in a muted,
        # always-visible hint beside the range controls they affect.
        self._empty_state_hint_label = QLabel(
            "Drag on the plot to add a fit range, or click a range to edit it."
        )
        self._empty_state_hint_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11px;")
        params_layout.addWidget(self._empty_state_hint_label)

        # ── Fit region for the ACTIVE range (contract C-REGIONROW) ────────────
        # A flat "Selected range" BENCH sub-header separates the card stack /
        # Add-Range controls ABOVE from the fit-region editor BELOW (P1
        # grouping). The fit region is ONE region the user sculpts by carving
        # gaps out of it: it is shown as a list of the included intervals it is
        # made of. Editing a spin, removing an interval, or "Exclude region…"
        # all funnel through set_included_intervals (the single storage source
        # of truth) so a down-to-1 result collapses back to a plain range.
        region_section, region_layout = make_section("Selected range")
        params_layout.addWidget(region_section)

        region_label = QLabel("Fit region")
        region_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        region_layout.addWidget(region_label)

        # The interval rows are rebuilt in place by _rebuild_fit_region_rows on
        # every active-range / interval change; each row is [Interval N] [min]
        # – [max] [Remove] (Remove hidden when there is only one interval).
        self._region_rows_block = QWidget()
        self._region_rows_layout = QVBoxLayout(self._region_rows_block)
        self._region_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._region_rows_layout.setSpacing(2)
        region_layout.addWidget(self._region_rows_block)

        # "Exclude region…" is the ONLY region action — there is deliberately no
        # "+ Add interval". Carving is what splits the one region into multiple
        # included intervals.
        exclude_row = QHBoxLayout()
        self._exclude_region_btn = QPushButton("Exclude region…")
        self._exclude_region_btn.setToolTip(
            "Carve a gap out of the fit region (or drag across a region on the "
            "plot). The remaining included intervals are listed above."
        )
        self._exclude_region_btn.clicked.connect(self._on_exclude_region_clicked)
        exclude_row.addWidget(self._exclude_region_btn)
        exclude_row.addStretch()
        region_layout.addLayout(exclude_row)

        # "Guess seeds" (item 3.3): user-initiated, off-thread data-aware seeding
        # of the ACTIVE range's free parameters. Never fires automatically — the
        # seed-preserve behaviour is pinned by test_model_fit_seed_preserve.py.
        guess_row = QHBoxLayout()
        self._guess_seeds_btn = QPushButton("Guess seeds")
        self._guess_seeds_btn.setToolTip(
            "Derive data-aware starting values for the selected range's free "
            "parameters from the fitted region's x/y data. Fixed parameters are "
            "never changed. Runs off the GUI thread."
        )
        self._guess_seeds_btn.clicked.connect(self._on_guess_seeds_clicked)
        guess_row.addWidget(self._guess_seeds_btn)
        self._guess_status_label = QLabel("")
        self._guess_status_label.setTextFormat(Qt.TextFormat.RichText)
        guess_row.addWidget(self._guess_status_label)
        guess_row.addStretch()
        params_layout.addLayout(guess_row)

        self._formula_box, self._formula_label = make_formula_box()
        params_layout.addWidget(self._formula_box)

        # Result box (item 4.3): the χ² + quality lines live inside a BENCH
        # result frame so a successful fit tints green inline — replacing the
        # per-fit "Fit complete" modal. The tint is swapped in _select_range
        # based on the active range's result state.
        self._result_box = QFrame()
        self._result_box.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._result_box.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        result_box_layout = QVBoxLayout(self._result_box)
        result_box_layout.setContentsMargins(8, 6, 8, 6)
        result_box_layout.setSpacing(2)

        self._chi2_label = QLabel("")
        self._chi2_label.setTextFormat(Qt.TextFormat.RichText)
        result_box_layout.addWidget(self._chi2_label)

        self._quality_label = QLabel("")
        self._quality_label.setTextFormat(Qt.TextFormat.RichText)
        self._quality_label.setToolTip(_QUALITY_TOOLTIP)
        result_box_layout.addWidget(self._quality_label)

        params_layout.addWidget(self._result_box)

        self._fit_progress_label = QLabel("")
        self._fit_progress_label.setStyleSheet(f"color: {tokens.WARN};")
        self._fit_progress_label.setVisible(False)
        params_layout.addWidget(self._fit_progress_label)

        self._suggest_section = self._build_suggest_section()
        params_layout.addWidget(self._suggest_section)

        self._param_table = QTableWidget(0, 6)
        self._param_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Min", "Max", "Fixed", "Error"]
        )
        apply_param_table_style(self._param_table)
        self._param_table.itemChanged.connect(self._on_param_table_edited)
        params_layout.addWidget(self._param_table)

        left_layout.addWidget(params_group)

        # Named insertion point (contract C6): directly above the OK/Cancel
        # button box. Subclasses (e.g. CrossGroupFitDialog's "Suggest roles…"
        # controls + rationale panel) add widgets here instead of scanning the
        # layout for ``self._buttons``. Empty by default, so it must add no
        # visual space.
        self._footer_slot = QVBoxLayout()
        self._footer_slot.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(self._footer_slot)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        remove_fit_btn = QPushButton("Remove Fit")
        remove_fit_btn.clicked.connect(self._on_remove_fit)
        buttons.addButton(remove_fit_btn, QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        self._remove_fit_btn = remove_fit_btn
        self._add_range_btn = add_btn
        top_layout.addWidget(buttons)

        self._rebuild_ranges_ui()
        self._select_range(0)

        # Initial preview paint (cheap synchronous spans + a debounced sample).
        self._request_preview_update()

    # -- preview-pane auto-collapse -------------------------------------------
    #: Below this usable width the right-hand preview pane auto-collapses and a
    #: "Show preview" toggle appears so it can be restored.
    _PREVIEW_NARROW_THRESHOLD = 900
    #: Share of the dialog's usable width given to the left pane when
    #: computing the proportional initial/expanded split (see
    #: ``_expanded_split_sizes``).
    _LEFT_PANE_WIDTH_FRACTION = 0.55

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().showEvent(event)
        # Decide once on first show, then let the user resize the splitter freely.
        self._maybe_collapse_preview(first_show=True)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._maybe_collapse_preview(first_show=False)

    def _expanded_split_sizes(self, width: int) -> list[int]:
        """Return ``[left, right]`` splitter sizes for an expanded preview.

        Splits *width* proportionally (left pane gets
        ``_LEFT_PANE_WIDTH_FRACTION``), then floors each side at its minimum
        width so neither the scrollable controls pane nor the preview pane is
        ever squashed below a usable size — even if that means the two floors
        sum to more than *width* (the splitter/scroll area absorb the excess).
        """
        left = max(self._LEFT_PANE_MIN_WIDTH, round(width * self._LEFT_PANE_WIDTH_FRACTION))
        right = max(self._PREVIEW_PANE_MIN_WIDTH, width - left)
        return [left, right]

    def _maybe_collapse_preview(self, *, first_show: bool) -> None:
        """Collapse/expand the preview pane based on the dialog's usable width.

        The pane is auto-managed only until the user drives it manually
        (``_on_show_preview_toggled``). Guarded so it fires once on first show
        and only when the width actually *crosses* the threshold thereafter,
        never fighting a manual splitter drag on every resize tick. A tiny or
        unknown reported width (e.g. offscreen) is treated as "not narrow" so
        the pane is not spuriously collapsed.
        """
        splitter = getattr(self, "_splitter", None)
        if splitter is None or not getattr(self, "_preview_auto_managed", False):
            return

        width = self.width()
        # A zero/tiny width is what offscreen/unrealized windows report; don't
        # collapse on that — mirror resize_to_available's screen-unknown guard.
        narrow = 0 < width < self._PREVIEW_NARROW_THRESHOLD

        if not first_show and narrow == self._preview_collapsed:
            # Already in the right state; nothing to do (avoids re-collapsing a
            # pane the user just expanded by dragging back above the threshold).
            return

        if not narrow and width > 0:
            # The real dialog width is now known — compute the proportional
            # split from it so the preview opens at a usable width instead of
            # the constructor-time placeholder sizes.
            self._preview_expanded_sizes = self._expanded_split_sizes(width)

        if narrow:
            self._collapse_preview()
        else:
            self._expand_preview()

    def _collapse_preview(self) -> None:
        self._preview_collapsed = True
        self._splitter.setSizes([1, 0])
        self._show_preview_toggle.setVisible(True)
        self._show_preview_toggle.blockSignals(True)
        self._show_preview_toggle.setChecked(False)
        self._show_preview_toggle.blockSignals(False)

    def _expand_preview(self) -> None:
        self._preview_collapsed = False
        self._splitter.setSizes(list(self._preview_expanded_sizes))
        self._show_preview_toggle.setVisible(False)
        self._show_preview_toggle.blockSignals(True)
        self._show_preview_toggle.setChecked(True)
        self._show_preview_toggle.blockSignals(False)

    def _on_show_preview_toggled(self, checked: bool) -> None:
        """User drove the preview toggle — honour it and stop auto-managing.

        Once the user makes an explicit choice we leave the pane under their
        control (a later narrowing does not override it), matching the
        "don't fight manual resizing" invariant.
        """
        self._preview_auto_managed = False
        if checked:
            self._preview_collapsed = False
            self._splitter.setSizes(list(self._preview_expanded_sizes))
            self._show_preview_toggle.setVisible(True)
        else:
            self._preview_collapsed = True
            self._splitter.setSizes([1, 0])
            self._show_preview_toggle.setVisible(True)

    def _on_show_residuals_toggled(self, checked: bool) -> None:
        """User toggled the residual strip: forward it to the preview canvas."""
        preview = getattr(self, "_preview", None)
        if preview is not None:
            preview.set_show_residuals(bool(checked))

    # -- live preview (work item 1.3) -----------------------------------------
    #
    # Lifecycle: a trigger calls _request_preview_update(), which (a) updates the
    # cheap visuals (series + spans + active range) SYNCHRONOUSLY so the plot
    # tracks an edit/drag instantly, and (b) (re)starts a ~120 ms debounce timer.
    # On timeout _launch_preview_sample() snapshots every input as plain data on
    # the GUI thread, bumps _preview_generation, and starts an off-thread worker
    # that samples the candidate curve(s). Results marshal back through
    # _on_preview_ready / _on_preview_error, which drop any result whose
    # generation no longer matches (a late/stale sample) and drain a single
    # coalesced pending request. The worker reads ONLY its snapshots — never
    # self, never a widget, never self._fit.

    def _preview_series(self) -> list[PreviewSeries]:
        """Data traces to draw. Base: ONE series from the dialog's x/y/err.

        Subclasses (cross-group) override to draw one series per group.
        """
        xerr = None if self._xerr is None else np.asarray(self._xerr, dtype=float)
        return [
            PreviewSeries(
                label="data",
                x=np.asarray(self._x, dtype=float),
                y=np.asarray(self._y, dtype=float),
                yerr=np.asarray(self._yerr, dtype=float),
                xerr=xerr,
            )
        ]

    def _request_preview_update(self) -> None:
        """Single entry point for every preview trigger.

        Updates the cheap visuals (series, spans, active range) immediately and
        restarts the debounce that launches the off-thread curve sample. Keep
        this cheap: no fitting, no heavy sampling.
        """
        if self._shutting_down:
            return
        preview = getattr(self, "_preview", None)
        if preview is None:
            return

        preview.set_series(self._preview_series())
        preview.set_active_range(self._active_range_idx)
        # Spans track instantly; the curve for each range reuses the last sampled
        # curve (empty until the first sample completes) so we never fit here.
        preview.set_ranges(self._current_preview_ranges())

        self._preview_timer.start()

    def _current_preview_ranges(self) -> list[PreviewRange]:
        """PreviewRange list for the CURRENT spans, reusing the last curves.

        Cheap: reads the range spans/windows off ``self._fit`` and reuses the
        last off-thread-sampled curve (empty arrays before the first sample).
        The in-mask is left empty here (recomputed on the sampling pass); the
        canvas falls back to "all in" on a length mismatch.
        """
        empty = np.array([], dtype=float)
        ranges: list[PreviewRange] = []
        for idx, fit_range in enumerate(self._fit.ranges):
            curve = self._last_preview_curves.get(idx, (empty, empty))
            windows = list(fit_range.windows) if fit_range.windows else None
            ranges.append(
                PreviewRange(
                    idx=idx,
                    x_min=fit_range.x_min,
                    x_max=fit_range.x_max,
                    windows=windows,
                    in_mask=empty.astype(bool),
                    curve_x=curve[0],
                    curve_y=curve[1],
                    fitted=self._range_is_fitted(idx),
                )
            )
        return ranges

    def _range_is_fitted(self, idx: int) -> bool:
        """Whether range *idx* currently holds a converged fit result.

        Drives the preview curve style (solid vs dashed seed). Reads through
        the ``_result_for_range`` hook so subclasses that cache results outside
        the per-range model (e.g. the cross-group dialog) stay consistent; any
        edit that invalidates the result flips the curve back to dashed on the
        next preview update.
        """
        result = self._result_for_range(idx)
        return result is not None and bool(getattr(result, "success", False))

    def _launch_preview_sample(self) -> None:
        """Snapshot inputs on the GUI thread and start the off-thread sampler.

        Coalescing: if a sample is already in flight, mark one pending request
        and return; the ready/error handler drains it. Otherwise snapshot every
        input as plain data, bump the generation token, and launch the worker.
        """
        # A timer event already dequeued before closeEvent stopped the timer
        # must not spin up a worker on the shut-down runner.
        if self._shutting_down:
            return
        if self._preview_active:
            self._preview_pending = True
            return

        preview = getattr(self, "_preview", None)
        if preview is None:
            return

        series = self._preview_series()
        # Empty when the primary series carries no finite point — nothing to
        # sample; show the empty state and skip the worker entirely.
        primary = series[0] if series else None
        if primary is None or not np.any(np.isfinite(np.asarray(primary.x, dtype=float))):
            preview.set_state("empty")
            return

        # Snapshot each range as plain data: a fresh model, a copy of its params,
        # bounds, and windows. Nothing here is read again off-thread.
        range_snapshots: list[dict[str, object]] = []
        for idx, fit_range in enumerate(self._fit.ranges):
            model_snapshot = ParameterCompositeModel(
                component_names=list(fit_range.model.component_names),
                operators=list(fit_range.model.operators),
            )
            params_snapshot = ParameterSet(
                [
                    Parameter(
                        name=p.name,
                        value=float(p.value),
                        min=float(p.min),
                        max=float(p.max),
                        fixed=bool(p.fixed),
                    )
                    for p in fit_range.parameters
                ]
            )
            range_snapshots.append(
                {
                    "idx": idx,
                    "model": model_snapshot,
                    "params": params_snapshot,
                    "x_min": fit_range.x_min,
                    "x_max": fit_range.x_max,
                    "windows": list(fit_range.windows) if fit_range.windows else None,
                    "fitted": self._range_is_fitted(idx),
                }
            )

        primary_x = np.asarray(primary.x, dtype=float).copy()
        active_idx = self._active_range_idx
        self._preview_generation += 1
        gen = self._preview_generation

        def _worker(_worker: object) -> object:
            # OFF-THREAD: touches only the plain snapshots captured above.
            out_ranges: list[PreviewRange] = []
            for snap in range_snapshots:
                idx = int(snap["idx"])
                curve_x, curve_y = sample_parameter_model(
                    snap["model"],
                    snap["params"],
                    snap["x_min"],
                    snap["x_max"],
                    snap["windows"],
                )
                # The in-mask is computed over the PRIMARY series' x for the
                # active range only; other ranges leave it empty (canvas draws
                # their span/curve without per-point greying).
                if idx == active_idx:
                    in_mask = windows_mask(primary_x, snap["windows"], snap["x_min"], snap["x_max"])
                else:
                    in_mask = np.array([], dtype=bool)
                out_ranges.append(
                    PreviewRange(
                        idx=idx,
                        x_min=snap["x_min"],
                        x_max=snap["x_max"],
                        windows=snap["windows"],
                        in_mask=in_mask,
                        curve_x=curve_x,
                        curve_y=curve_y,
                        fitted=bool(snap["fitted"]),
                    )
                )
            return (gen, out_ranges)

        self._preview_active = True
        preview.set_state("loading")
        self._tasks.start(
            _worker,
            on_finished=self._on_preview_ready,
            on_error=self._on_preview_error,
        )

    def _on_preview_ready(self, payload: object) -> None:
        """GUI thread: apply a fresh sample, or drop it if a newer one exists."""
        gen = -1
        out_ranges: list[PreviewRange] = []
        if isinstance(payload, tuple) and len(payload) == 2:
            gen, out_ranges = payload  # type: ignore[assignment]

        if gen != self._preview_generation:
            # A newer sample was launched after this one started — this result is
            # stale. Do not draw it; still release the in-flight slot and drain.
            self._finish_preview_sample()
            return

        preview = getattr(self, "_preview", None)
        if preview is not None:
            # Cache the sampled curve per range so the next SYNCHRONOUS span
            # update can reuse it (spans move instantly; curve lags one sample).
            self._last_preview_curves = {rng.idx: (rng.curve_x, rng.curve_y) for rng in out_ranges}
            preview.set_series(self._preview_series())
            preview.set_ranges(out_ranges)
            preview.set_active_range(self._active_range_idx)
            has_curve = any(np.asarray(rng.curve_y).size > 0 for rng in out_ranges)
            preview.set_state("ready" if has_curve else "empty")

        self._finish_preview_sample()

    def _on_preview_error(self, message: str) -> None:
        """GUI thread: surface the failure and drain any coalesced request."""
        preview = getattr(self, "_preview", None)
        if preview is not None:
            preview.set_state("error", message)
        self._finish_preview_sample()

    def _finish_preview_sample(self) -> None:
        """Release the in-flight slot; launch the coalesced request if any."""
        self._preview_active = False
        if self._preview_pending:
            self._preview_pending = False
            self._launch_preview_sample()

    def get_model_fit(self) -> ParameterModelFit | None:
        if self._removed:
            return None
        self._fit.use_x_errors = self._use_x_errors()
        return self._fit

    def was_removed(self) -> bool:
        return self._removed

    def _use_x_errors(self) -> bool:
        # Gate on enabled as well as checked: the toggle is disabled under the
        # None/Scatter error modes (whose unit y-weights have no scale to combine
        # with σ_x), so a box left checked from a prior mode must not feed
        # x-errors into the fit — keeps the GUI honest independently of the core
        # guard rather than relying on the two staying in lockstep.
        return (
            self._x_error_check is not None
            and self._x_error_check.isEnabled()
            and self._x_error_check.isChecked()
        )

    def _error_mode(self) -> ErrorMode:
        if self._error_mode_combo is None:
            return ErrorMode.COLUMN
        data = self._error_mode_combo.currentData()
        return ErrorMode(data) if data is not None else ErrorMode.COLUMN

    def _error_value(self) -> float | None:
        if self._error_value_spin is not None and self._error_mode() in (
            ErrorMode.PERCENT,
            ErrorMode.ABSOLUTE,
        ):
            return float(self._error_value_spin.value())
        return None

    def _on_error_mode_changed(self, _idx: int) -> None:
        mode = self._error_mode()
        # The effective-variance toggle has no real σ_y to combine with under
        # unit-weight (None) or scatter modes, so the fit ignores x-errors there
        # — disable the control instead of leaving it promising a no-op.
        if getattr(self, "_x_error_check", None) is not None:
            self._x_error_check.setEnabled(mode not in (ErrorMode.NONE, ErrorMode.SCATTER))
        # A mode switch changes whether the noise model behind "Suggest next
        # point" is meaningful (NONE/SCATTER never are; see
        # _suggest_disabled_reason) AND changes what chi2 means for the
        # existing candidates' AIC weights, so clear both and re-derive the
        # section's enabled state. Guarded: this runs once during __init__
        # before the section exists.
        if getattr(self, "_suggest_section", None) is not None:
            self._clear_suggestion()
            self._clear_discrimination_candidates()
            self._refresh_suggest_section()
        if self._error_value_label is None or self._error_value_spin is None:
            return
        needs_value = mode in (ErrorMode.PERCENT, ErrorMode.ABSOLUTE)
        self._error_value_label.setVisible(needs_value)
        self._error_value_spin.setVisible(needs_value)
        self._error_value_spin.setEnabled(needs_value)
        self._request_preview_update()

    def _create_default_range(self) -> ModelFitRange:
        x_min = float(np.nanmin(self._x)) if np.any(np.isfinite(self._x)) else 0.0
        x_max = float(np.nanmax(self._x)) if np.any(np.isfinite(self._x)) else 1.0

        available = self._component_pool
        # RESTORE (item 4.2): prefer the user's last-used model for this
        # (base_param, x_key) if one was remembered and still parses within this
        # context's pool; otherwise fall back to the context default. Purely
        # non-load-bearing — any failure yields the default.
        model = _load_last_model(self._parameter_name, self._x_key, available, self._model_memory)
        if model is None:
            default_component = _default_component_for_context(
                self._x_key, self._parameter_name, available
            )
            model = ParameterCompositeModel([default_component], [])

        params = ParameterSet()
        y_mean = float(np.nanmean(self._y)) if np.any(np.isfinite(self._y)) else 0.0
        y_span = (
            float(np.nanmax(self._y) - np.nanmin(self._y)) if np.any(np.isfinite(self._y)) else 1.0
        )
        # Data-aware seeds for critical-temperature trend components
        # (OrderParameter / CriticalDivergence): derive T_c and amplitude from the
        # actual x/y so an order-parameter default converges without a manual
        # reseed (F4). Empty for Linear and the other plain components, which keep
        # the inline heuristic below unchanged.
        trend_seeds = suggest_trend_seeds(model, self._x, self._y)

        for pname in model.param_names:
            default_val = model.param_defaults[pname]
            if pname in {"c", "b"}:
                default_val = y_mean
            elif pname in {"m", "a"}:
                default_val = y_span if y_span > 0 else default_val
            elif pname.startswith("B0") or pname.startswith("tau") or pname.startswith("nu"):
                default_val = max(1e-6, (x_max - x_min) / 2.0)
            elif pname.startswith("D_2D"):
                default_val = max(1e-6, default_val)
            elif pname.startswith("D"):
                default_val = max(1e-6, default_val)
            params.add(
                Parameter(
                    name=pname,
                    value=float(trend_seeds.get(pname, default_val)),
                    fixed=(pname == "shape_factor_a"),
                )
            )

        return ModelFitRange(x_min=x_min, x_max=x_max, model=model, parameters=params)

    def _add_range_with_bounds(self, x_min: float | None, x_max: float | None) -> None:
        """Append a new range (default model/params) with the given bounds — or the
        data-extent default when None — then rebuild, activate it, and refresh the
        preview.

        The "Add Range" button and the plot's drag-out gesture both converge here,
        so a drag-created range gets the SAME ``_create_default_range`` model/param
        seeding as the button. Degenerate drag bounds (inverted or zero-width) fall
        back to the default data-extent bounds rather than creating a bad range.
        """
        new_range = self._create_default_range()
        if x_min is not None and x_max is not None:
            lo = float(x_min)
            hi = float(x_max)
            if hi > lo:
                new_range.x_min = lo
                new_range.x_max = hi
            # else: inverted/degenerate span — keep the default data-extent bounds.
        self._fit.ranges.append(new_range)
        self._rebuild_ranges_ui()
        new_idx = len(self._fit.ranges) - 1
        self._set_active_range(new_idx)
        self._request_preview_update()

    def _add_range(self) -> None:
        self._add_range_with_bounds(None, None)

    def _remove_range(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        if len(self._fit.ranges) == 1:
            _show_info(self, "Range required", "At least one range must remain.")
            return
        del self._fit.ranges[idx]
        self._rebuild_ranges_ui()
        self._select_range(max(0, idx - 1))

    def _rebuild_ranges_ui(self) -> None:
        previous_idx = self._active_range_idx if self._active_range_idx is not None else 0

        clear_layout(self._ranges_host)
        self._range_cards = []

        active_idx = min(previous_idx, len(self._fit.ranges) - 1) if self._fit.ranges else 0
        for idx in range(len(self._fit.ranges)):
            card = RangeCard(idx)
            card.set_state(self._range_card_view(idx, show_run=(idx == active_idx)))
            card.set_active(idx == active_idx)
            card.selected.connect(self._select_range)
            card.run_requested.connect(self._run_fit)
            card.edit_model_requested.connect(self._edit_model)
            card.remove_requested.connect(self._remove_range)
            self._ranges_host.addWidget(card)
            self._range_cards.append(card)

        self._post_rebuild_ranges_ui()
        if self._fit.ranges:
            self._select_range(max(0, min(previous_idx, len(self._fit.ranges) - 1)))

    def _range_card_view(self, idx: int, *, show_run: bool) -> RangeCardView:
        """Assemble the plain render payload for range *idx*'s card.

        Bounds text uses the ``∪`` union formatting from ``_range_bounds_text``;
        the status chip reuses the same verdict path (``assess_fit_quality`` /
        ``fit_quality_chip_html``) the details pane uses, so card and result box
        never disagree.
        """
        fit_range = self._fit.ranges[idx]
        result = self._result_for_range(idx)

        status: str
        chip_html = ""
        tooltip = ""
        if result is None:
            status = "not_run"
        elif getattr(result, "success", False):
            status = "success"
            chip_html = self._range_status_chip_html(fit_range, result)
            chi2 = getattr(result, "chi_squared", None)
            rchi2 = getattr(result, "reduced_chi_squared", None)
            if chi2 is not None and rchi2 is not None:
                tooltip = f"chi2 = {float(chi2):.6g}, reduced chi2 = {float(rchi2):.6g}"
        else:
            status = "failed"
            tooltip = getattr(result, "message", "") or "No convergence"

        return RangeCardView(
            idx=idx,
            title=self._range_card_title(idx),
            swatch_color=range_span_color(idx),
            bounds_text=self._range_bounds_text(fit_range),
            formula=fit_range.model.formula_string(),
            status=status,  # type: ignore[arg-type]
            status_chip_html=chip_html,
            status_tooltip=tooltip,
            can_remove=len(self._fit.ranges) > 1,
            show_run=show_run,
        )

    def _range_card_title(self, idx: int) -> str:
        """Card title for range *idx*. Overridable so a subclass with a single,
        always-active range (cross-group mode) can drop the redundant "Range 1"
        label; the base numbering is unchanged for the standard multi-range case.
        """
        return f"Range {idx + 1}"

    def _range_bounds_text(self, fit_range: ModelFitRange) -> str:
        """Compact bounds string for a card, e.g. "[12–40] ∪ [55–88] K".

        The card is now the range selector, so this is the sole authority for a
        range's rendered bounds. Appends the x-unit when the abscissa has one.
        """
        unit = _x_unit(self._x_key)
        suffix = f" {unit}" if unit else ""
        if fit_range.windows:
            union = " ∪ ".join(f"[{lo:.6g}–{hi:.6g}]" for lo, hi in fit_range.windows)
            return f"{union}{suffix}"
        x_min = fit_range.x_min if fit_range.x_min is not None else float("nan")
        x_max = fit_range.x_max if fit_range.x_max is not None else float("nan")
        return f"[{x_min:.6g}–{x_max:.6g}]{suffix}"

    def _range_status_chip_html(self, fit_range: ModelFitRange, result: object) -> str:
        """Verdict chip for a successful range, via the shared quality path.

        Returns "" when no verdict applies (unit/scatter weights, ν < 1, or a
        result without a point count — e.g. a cross-group/legacy bridge result).
        """
        if getattr(result, "error_mode", None) in (ErrorMode.NONE.value, ErrorMode.SCATTER.value):
            return ""
        n_points = getattr(result, "n_points", 0)
        if not n_points or n_points <= 0:
            return ""
        n_free = len(fit_range.parameters.free_parameters)
        quality = assess_fit_quality(
            result.chi_squared, n_points - n_free, fit_quality_confidence()
        )
        if quality.verdict is None:
            return ""
        quality_dict = {
            "verdict": quality.verdict,
            "band_low": float(quality.band_low),
            "band_high": float(quality.band_high),
            "confidence": float(quality.confidence),
            "dof": int(quality.dof),
        }
        params_at_bound = list(getattr(result, "params_at_bound", ()) or [])
        return fit_quality_chip_html(quality_dict, params_at_bound)

    def _post_rebuild_ranges_ui(self) -> None:
        """Hook for subclasses to adjust freshly rebuilt range rows."""

    # ── details-pane fit-region editor (contract C-REGIONROW / C-REGION) ──────

    def _resolved_intervals(self, fit_range: ModelFitRange) -> list[tuple[float, float]]:
        """``included_intervals`` with open (``None``) bounds resolved to data.

        ``included_intervals`` reports an unbounded plain range as
        ``(-inf, +inf)``; the GUI resolves those to the data extent
        (``_x_min_data`` / ``_x_max_data``) before display, mirroring how the
        old window seeding resolved missing bounds.
        """
        resolved: list[tuple[float, float]] = []
        for lo, hi in included_intervals(fit_range):
            r_lo = self._x_min_data if not np.isfinite(lo) else float(lo)
            r_hi = self._x_max_data if not np.isfinite(hi) else float(hi)
            resolved.append((float(r_lo), float(r_hi)))
        return resolved

    def _rebuild_fit_region_rows(self, idx: int) -> None:
        """Rebuild the fit-region interval rows for the ACTIVE range.

        Replaces the old plain-bounds pair AND the exclusion-window sub-block:
        one row per included interval of the active range, keyed in
        ``self._region_row_spins`` by INTERVAL INDEX. The per-row Remove button
        is hidden when there is exactly one interval (a plain range can never be
        emptied below one interval).
        """
        clear_layout(self._region_rows_layout)
        self._region_row_spins = {}
        self._region_remove_btns = []

        fit_range = self._fit.ranges[idx]
        intervals = self._resolved_intervals(fit_range)
        show_remove = len(intervals) > 1

        for iidx, (lo, hi) in enumerate(intervals):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(f"Interval {iidx + 1}"))

            i_min = QDoubleSpinBox()
            i_min.setRange(-1e12, 1e12)
            i_min.setDecimals(8)
            i_min.setValue(float(lo))
            i_min.valueChanged.connect(
                lambda value, i=iidx: self._on_region_interval_edited(i, 0, value)
            )
            row.addWidget(i_min)

            row.addWidget(QLabel("–"))

            i_max = QDoubleSpinBox()
            i_max.setRange(-1e12, 1e12)
            i_max.setDecimals(8)
            i_max.setValue(float(hi))
            i_max.valueChanged.connect(
                lambda value, i=iidx: self._on_region_interval_edited(i, 1, value)
            )
            row.addWidget(i_max)

            self._region_row_spins[iidx] = (i_min, i_max)

            remove_btn = QPushButton("Remove")
            remove_btn.setToolTip("Drop this included interval from the fit region.")
            remove_btn.clicked.connect(
                lambda _checked=False, i=iidx: self._remove_interval(self._active_range_idx, i)
            )
            remove_btn.setVisible(show_remove)
            row.addWidget(remove_btn)
            self._region_remove_btns.append(remove_btn)

            row.addStretch()
            self._region_rows_layout.addWidget(row_widget)

    def _current_region_intervals(self) -> list[tuple[float, float]]:
        """Read the current interval list straight off the visible spin rows."""
        intervals: list[tuple[float, float]] = []
        for iidx in sorted(self._region_row_spins):
            i_min, i_max = self._region_row_spins[iidx]
            intervals.append((float(i_min.value()), float(i_max.value())))
        return intervals

    def _on_region_interval_edited(self, interval_idx: int, bound: int, value: float) -> None:
        """A fit-region interval's min/max spin was edited (numeric path).

        Builds the new interval list from the current rows, clamps the edited
        interval so ``min <= max``, writes it back through
        ``set_included_intervals`` (the collapse rule auto-plains a down-to-1
        result), invalidates the stale result, refreshes the card, and — per the
        FEEDBACK-LOOP RULE — a numeric edit does a full preview update.
        """
        idx = self._active_range_idx
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return
        intervals = self._current_region_intervals()
        if interval_idx < 0 or interval_idx >= len(intervals):
            return
        lo, hi = intervals[interval_idx]
        # Clamp so the edited bound never inverts its partner.
        if bound == 0:
            lo = float(value)
            if lo > hi:
                hi = lo
        else:
            hi = float(value)
            if hi < lo:
                lo = hi
        intervals[interval_idx] = (lo, hi)

        fit_range = self._fit.ranges[idx]
        set_included_intervals(fit_range, intervals)
        self._invalidate_range_result(idx)
        self._refresh_range_card(idx)
        # Numeric edit: full preview update (canvas has no live span to fight).
        self._request_preview_update()

    def _remove_interval(self, idx: int, interval_idx: int) -> None:
        """Drop one included interval from the fit region.

        NEVER-EMPTY guard: removing the last interval is a no-op (a fit region
        can never be emptied). Otherwise the survivors are written through
        ``set_included_intervals`` — its collapse rule plains a down-to-1 result
        back to ``windows is None`` — then the rows/card/preview refresh.
        """
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return
        fit_range = self._fit.ranges[idx]
        intervals = self._resolved_intervals(fit_range)
        if interval_idx < 0 or interval_idx >= len(intervals):
            return
        if len(intervals) <= 1:
            # Never empty: refuse to drop the last interval.
            return
        del intervals[interval_idx]
        set_included_intervals(fit_range, intervals)
        self._invalidate_range_result(idx)
        self._rebuild_fit_region_rows(idx)
        self._refresh_range_card(idx)
        self._request_preview_update()

    def _on_exclude_region_clicked(self) -> None:
        """The details-pane "Exclude region…" button: carve a default gap."""
        self._exclude_default_gap(self._active_range_idx)

    def _exclude_default_gap(self, idx: int | None) -> None:
        """Carve a sensible default gap when there is no drag interval.

        The button has no drag interval, so carve the MIDDLE THIRD of the widest
        current included interval, giving the user two intervals to then
        fine-tune via the spins or by dragging on the plot. Routes through the
        shared ``_exclude_region``.
        """
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return
        fit_range = self._fit.ranges[idx]
        intervals = self._resolved_intervals(fit_range)
        # Widest current interval — the one with the most room to carve.
        lo, hi = max(intervals, key=lambda interval: interval[1] - interval[0])
        span = hi - lo
        if span <= 0:
            return
        gap_lo = lo + span / 3.0
        gap_hi = lo + 2.0 * span / 3.0
        self._exclude_region(idx, gap_lo, gap_hi)

    def _exclude_region(self, idx: int, lo: float, hi: float) -> None:
        """Carve ``[lo, hi]`` out of the fit region (shared button + canvas path).

        Resolves the range's effective bounds (open bounds fall back to the data
        extent), carves the gap out of the current window union, and — only if
        that actually changed the coverage — writes the survivors through
        ``set_included_intervals``, invalidates the stale result, rebuilds the
        interval rows, and refreshes the card + preview.

        NEVER-EMPTY: ``carve_window_gap`` no-ops an all-excluding carve, so the
        result is always non-empty; ``set_included_intervals`` would reject an
        empty list as a second guard.
        """
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        fit_range = self._fit.ranges[idx]
        x_min = fit_range.x_min if fit_range.x_min is not None else self._x_min_data
        x_max = fit_range.x_max if fit_range.x_max is not None else self._x_max_data

        new_windows = carve_window_gap(fit_range.windows, x_min, x_max, lo, hi)

        # NO-OP GUARD: normalise the current windows to what carve_window_gap
        # would have seeded (None -> [(x_min, x_max)]) and compare. A carve that
        # is disjoint from every window (a stray drag/click outside the fitted
        # region) returns the seeded windows unchanged; dropping a good fit for
        # that would be a nasty surprise, so early-return without invalidating.
        current = list(fit_range.windows) if fit_range.windows else [(float(x_min), float(x_max))]
        if new_windows == current:
            return

        set_included_intervals(fit_range, new_windows)
        self._invalidate_range_result(idx)
        self._rebuild_fit_region_rows(idx)
        self._refresh_range_card(idx)
        self._request_preview_update()

    def _refresh_range_card(self, idx: int) -> None:
        """Repaint range *idx*'s card from a freshly-rebuilt view (chip + bounds).

        Preserves the card's current ``show_run`` (active-ness) so a status/bounds
        refresh does not flip which card exposes the Run Fit action.
        """
        if idx < 0 or idx >= len(self._range_cards):
            return
        show_run = idx == self._active_range_idx
        self._range_cards[idx].set_state(self._range_card_view(idx, show_run=show_run))

    def _invalidate_range_result(self, idx: int) -> None:
        """Drop a range's fit result after its mask changed; refresh labels."""
        fit_range = self._fit.ranges[idx]
        if fit_range.result is None:
            return
        fit_range.result = None
        self._refresh_range_card(idx)
        if self._active_range_idx == idx:
            self._chi2_label.setText(info_html("Fitting not yet run for selected range"))
            self._quality_label.setText("")
            self._apply_result_box_style(None)

    def _quality_text_for_range(self, fit_range: ModelFitRange) -> str:
        """χ² quality verdict line for a fitted range (empty when not fitted)."""
        result = fit_range.result
        if result is None or not result.success:
            return ""
        if result.error_mode in (ErrorMode.NONE.value, ErrorMode.SCATTER.value):
            return info_html(
                "No χ² quality verdict: with unit-weight or scatter-estimated "
                "errors χ²ᵣ carries no goodness information."
            )
        if result.n_points <= 0:
            # Results built outside fit_parameter_model (cross-group bridge,
            # legacy saved state) do not carry a point count — say nothing
            # rather than implying the fit had no degrees of freedom.
            return ""
        n_free = len(fit_range.parameters.free_parameters)
        quality = assess_fit_quality(
            result.chi_squared, result.n_points - n_free, fit_quality_confidence()
        )
        if quality.verdict is None:
            return info_html("No χ² quality verdict (no degrees of freedom).")
        # Reuse the shared verdict→colour map (good=green, poor=error,
        # overdone=accent) instead of re-rolling one here.
        color = _FIT_VERDICT_COLOURS.get(quality.verdict, tokens.TEXT_MUTED)
        return (
            f'<span style="color:{color};">Quality of fit: <b>{quality.verdict}</b> '
            f"— χ²ᵣ target band {quality.band_low:.3f} to {quality.band_high:.3f} "
            f"(ν = {quality.dof}, {quality.confidence:.0%} confidence). "
            "Hover for what this means.</span>"
        )

    def _apply_range_bounds(self, idx: int, x_min: float, x_max: float, *, source: str) -> None:
        """Funnel for a range-edge change (details-pane or plot drag of a PLAIN range).

        A plain range's edges ARE interval 0 of its fit region, so this writes
        the single interval ``(x_min, x_max)`` through ``set_included_intervals``
        (which keeps ``windows is None`` for a 1-interval range) and mirrors it
        into the interval-0 spin pair. Only fires for plain ranges — a windowed
        range shows no whole-range edges (only window edges are draggable).
        """
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        fit_range = self._fit.ranges[idx]
        # A plain-range edge move writes interval 0; the collapse rule keeps
        # windows None for the single-interval case.
        set_included_intervals(fit_range, [(float(x_min), float(x_max))])
        self._invalidate_range_result(idx)

        # Mirror into the details-pane interval-0 spin pair, but ONLY for the
        # active range: the canvas only ever drags the active range, so a
        # non-active idx here is a programmatic call and the details pane already
        # shows another range. A non-active card's bounds_text refreshes below.
        if idx == self._active_range_idx:
            spins = self._region_row_spins.get(0)
            if spins is not None:
                i_min, i_max = spins
                with QSignalBlocker(i_min):
                    i_min.setValue(float(x_min))
                with QSignalBlocker(i_max):
                    i_max.setValue(float(x_max))

        self._refresh_range_card(idx)

        # FEEDBACK-LOOP RULE (canvas vs spinbox): during a canvas drag the
        # TrendPreviewCanvas already mutates and redraws its own span artist
        # live, so a full _request_preview_update() here would synchronously
        # set_ranges() and fight the in-progress drag. For a canvas source we
        # therefore only kick the debounce so the OFF-THREAD curve resamples to
        # catch up — the canvas owns its spans mid-drag. For a spinbox (numeric)
        # edit there is no live span, so we call the full update to move the
        # canvas spans to match the typed value.
        if source == "canvas":
            self._preview_timer.start()
        else:
            self._request_preview_update()

    def _connect_plot_range_signals(self) -> None:
        """Connect the plot's add/select gestures. Overridable so single-range
        subclasses (cross-group) can leave them inert."""
        self._preview.range_select_requested.connect(self._on_preview_range_select)
        self._preview.range_add_requested.connect(self._on_preview_range_add)

    def _on_preview_range_select(self, idx: int) -> None:
        """Canvas clicked a non-active range's span: make it the active range.

        The bounds-guard lives inside ``_set_active_range``; ``from_plot=True``
        marks the plot as the selection source (contract C-ACTIVE).
        """
        self._set_active_range(idx, from_plot=True)

    def _on_preview_range_add(self, x_min: float, x_max: float) -> None:
        """Canvas dragged out a new span on empty area: append a seeded range."""
        self._add_range_with_bounds(x_min, x_max)

    def _on_preview_range_edge_dragged(self, idx: int, x_min: float, x_max: float) -> None:
        """Canvas dragged a range edge: mirror it into the model + spinboxes.

        Only the active range's edges are draggable, so a signal for a
        non-active index is defensively ignored.
        """
        if idx != self._active_range_idx:
            return
        self._apply_range_bounds(idx, float(x_min), float(x_max), source="canvas")

    def _on_preview_window_edge_dragged(
        self, idx: int, window_idx: int, lo: float, hi: float
    ) -> None:
        """Canvas dragged a window edge: mirror it into the model + interval spins.

        The window index equals the fit-region interval index by construction,
        so this updates interval ``window_idx`` and writes the whole list back
        through ``set_included_intervals`` (keeping the envelope / collapse rule
        consistent), then mirrors the interval spin pair.
        """
        if idx != self._active_range_idx:
            return
        if idx < 0 or idx >= len(self._fit.ranges):
            return
        fit_range = self._fit.ranges[idx]
        intervals = self._resolved_intervals(fit_range)
        if window_idx < 0 or window_idx >= len(intervals):
            return
        intervals[window_idx] = (float(lo), float(hi))
        set_included_intervals(fit_range, intervals)
        self._invalidate_range_result(idx)

        # Mirror the corresponding interval spinboxes under a signal blocker so
        # they do not re-fire _on_region_interval_edited and loop. The interval
        # index == window index by construction (idx is the active range here).
        spins = self._region_row_spins.get(window_idx)
        if spins is not None:
            i_min, i_max = spins
            with QSignalBlocker(i_min):
                i_min.setValue(float(lo))
            with QSignalBlocker(i_max):
                i_max.setValue(float(hi))

        self._refresh_range_card(idx)
        # Canvas-source: curve resample only (see FEEDBACK-LOOP RULE above).
        self._preview_timer.start()

    def _on_preview_exclude_region(self, idx: int, lo: float, hi: float) -> None:
        """Right-drag exclude gesture: carve ``[lo, hi]`` out of the fit region.

        The plot drag supplies a real interval, so it routes straight through
        the shared ``_exclude_region`` (the button's default-gap path is the
        other caller).
        """
        self._exclude_region(idx, lo, hi)

    def _edit_model(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        fit_range = self._fit.ranges[idx]
        dlg = ParameterModelBuilderDialog(
            component_pool=self._component_pool,
            initial_model=fit_range.model,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        model = dlg.get_model()
        if model is None:
            return

        fit_range.model = model
        # Remember this model for the (base_param, x_key) so the next fresh
        # dialog seeds it as the default (item 4.2). Best-effort / silent.
        _store_last_model_expression(
            self._parameter_name,
            self._x_key,
            model.component_expression_string(),
            self._model_memory,
        )

        # Critical-temperature trend components (CriticalDivergence,
        # OrderParameter) default to an unphysical Tc=10; seed Tc (and a cheap
        # amplitude/baseline) from the actual x/y data so the fit converges
        # without a manual reseed. Only newly reset params adopt these — params
        # carried over from the previous model keep the user's value.
        trend_seeds = suggest_trend_seeds(model, self._x, self._y)

        new_params = ParameterSet()
        for pname in model.param_names:
            if pname in fit_range.parameters and not _should_reset_param_on_model_change(
                model, pname
            ):
                old = fit_range.parameters[pname]
                new_params.add(
                    Parameter(
                        name=pname, value=old.value, min=old.min, max=old.max, fixed=old.fixed
                    )
                )
            else:
                new_params.add(
                    Parameter(
                        name=pname,
                        value=float(trend_seeds.get(pname, model.param_defaults[pname])),
                        fixed=(pname == "shape_factor_a"),
                    )
                )
        fit_range.parameters = new_params
        fit_range.result = None
        self._on_model_edited(idx)
        # The old model's sampled curve no longer applies — drop it so the
        # synchronous span pass does not reuse a stale line until the next
        # off-thread sample lands.
        self._last_preview_curves.pop(idx, None)

        self._rebuild_ranges_ui()
        # _select_range already re-requests the preview; the stale-curve pop
        # above is the only extra work the model edit needs here.
        self._select_range(idx)

    def _on_model_edited(self, idx: int) -> None:
        """Hook: called after a range's model (component add/remove/edit) changes.

        The base dialog already tracks fit state per-range via
        ``fit_range.result`` (cleared just above), so it needs no extra
        invalidation here beyond dropping this range's fitted discrimination
        alternatives (§8.1 item 1: cleared when the range's model expression
        changes — they were fitted/ranked against the previous leader).
        Subclasses that cache a fit result *outside* the per-range model
        (e.g. ``CrossGroupFitDialog``'s ``self._result`` / ``self._last_config``,
        which span the single shared range) override this to drop that cache
        too.
        """
        self._discrimination_candidates.pop(idx, None)

    # -- data-aware "Guess seeds" (item 3.3) ----------------------------------
    #
    # USER-INITIATED ONLY. This must never fire on a model edit or a range
    # selection — the seed-preserve invariant is pinned by
    # tests/gui/test_model_fit_seed_preserve.py. It mirrors the cross-group
    # "Suggest roles…" off-thread pattern: snapshot plain data on the GUI thread,
    # run core off-thread on ``self._tasks`` under its own ``_guess_in_progress``
    # flag (NOT ``_fit_in_progress``), then write the returned values back through
    # the shared commit path so they are normalised/persisted like any edit.

    def _set_guess_busy(self, busy: bool) -> None:
        self._guess_in_progress = busy
        self._guess_seeds_btn.setEnabled(not busy)

    def _on_guess_seeds_clicked(self) -> None:
        """Suggest data-aware seeds for the ACTIVE range, off-thread."""
        if self._guess_in_progress or self._fit_in_progress:
            return
        idx = self._active_range_idx
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return

        # Persist any pending table edit first so the snapshot reflects the
        # current bounds/fixed flags, then snapshot as plain data.
        self._commit_param_table(notify_adjustments=False)
        fit_range = self._fit.ranges[idx]

        model_snapshot = ParameterCompositeModel(
            component_names=list(fit_range.model.component_names),
            operators=list(fit_range.model.operators),
        )

        # Mask x/y/yerr to the active range (same idiom as the preview sampler)
        # so seeds reflect the fitted region. Fall back to full data if the mask
        # leaves fewer than two points to estimate from.
        x_full = np.asarray(self._x, dtype=float).copy()
        y_full = np.asarray(self._y, dtype=float).copy()
        yerr_full = np.asarray(self._yerr, dtype=float).copy()
        windows = list(fit_range.windows) if fit_range.windows else None
        mask = windows_mask(x_full, windows, fit_range.x_min, fit_range.x_max)
        if mask.size == x_full.size and int(np.count_nonzero(mask)) >= 2:
            x_masked = x_full[mask]
            y_masked = y_full[mask]
            yerr_masked = yerr_full[mask]
        else:
            x_masked, y_masked, yerr_masked = x_full, y_full, yerr_full

        self._set_guess_busy(True)
        self._guess_status_label.setText(info_html("Guessing seeds…"))

        def _worker(_worker: object) -> object:
            # OFF-THREAD: reads only the plain snapshots captured above.
            return suggest_model_seeds(model_snapshot, x_masked, y_masked, yerr_masked)

        self._guess_target_idx = idx
        self._tasks.start(
            _worker,
            on_finished=self._on_guess_seeds_done,
            on_error=self._on_guess_seeds_error,
        )

    def _on_guess_seeds_done(self, payload: object) -> None:
        """GUI thread: write returned seeds into the active range's param table."""
        self._set_guess_busy(False)
        seeds = payload if isinstance(payload, dict) else {}
        if not seeds:
            self._guess_status_label.setText(
                info_html("No data-aware seed available for this model")
            )
            return

        # Only touch the range that was active when Guess was launched, and only
        # if it is still the active range (a selection change mid-run makes the
        # result stale for the table currently shown).
        target_idx = getattr(self, "_guess_target_idx", None)
        if target_idx is None or target_idx != self._active_range_idx:
            self._guess_status_label.setText("")
            return

        changed = False
        self._param_table.blockSignals(True)
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            if name_item is None or value_item is None:
                continue
            name_data = name_item.data(Qt.ItemDataRole.UserRole)
            name = name_data.strip() if isinstance(name_data, str) else name_item.text().strip()
            if name not in seeds:
                continue
            # NEVER overwrite a fixed parameter (fixed row or shape_factor_a).
            control = self._param_table.cellWidget(row, 4)
            if bool(self._read_param_row_control(control).get("fixed", False)):
                continue
            if name == "shape_factor_a":
                continue
            value_item.setText(f"{float(seeds[name]):.8g}")
            changed = True
        self._param_table.blockSignals(False)

        if not changed:
            self._guess_status_label.setText(
                info_html("Data-aware seeds apply only to fixed parameters here")
            )
            return

        # Route through the shared commit path so the written values are
        # normalised (via _normalize_parameter_limits) and persisted, then
        # refresh the preview so the dashed seed curve follows.
        self._commit_param_table(notify_adjustments=False)
        self._guess_status_label.setText("")
        self._request_preview_update()

    def _on_guess_seeds_error(self, message: str) -> None:
        self._set_guess_busy(False)
        self._guess_status_label.setText(warning_html("Seed guess failed"))

    def _run_fit(self, idx: int) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Please wait for the current fit to finish.")
            return
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        self._commit_param_table(notify_adjustments=True)
        fit_range = self._fit.ranges[idx]

        if fit_range.windows:
            try:
                validate_fit_windows(fit_range.windows)
            except ValueError as exc:
                _show_warning(self, "Invalid window", str(exc))
                return
        elif (
            fit_range.x_max is not None
            and fit_range.x_min is not None
            and fit_range.x_max <= fit_range.x_min
        ):
            _show_warning(self, "Invalid range", "x max must be greater than x min.")
            return

        model_snapshot = ParameterCompositeModel(
            component_names=list(fit_range.model.component_names),
            operators=list(fit_range.model.operators),
        )
        params_snapshot = ParameterSet(
            [
                Parameter(
                    name=p.name,
                    value=float(p.value),
                    min=float(p.min),
                    max=float(p.max),
                    fixed=bool(p.fixed),
                )
                for p in fit_range.parameters
            ]
        )
        x_vals = np.asarray(self._x, dtype=float).copy()
        y_vals = np.asarray(self._y, dtype=float).copy()
        y_errs = np.asarray(self._yerr, dtype=float).copy()
        x_min = fit_range.x_min
        x_max = fit_range.x_max
        windows = list(fit_range.windows) if fit_range.windows else None
        error_mode = self._error_mode()
        error_value = self._error_value()
        x_errs = (
            np.asarray(self._xerr, dtype=float).copy()
            if self._use_x_errors() and self._xerr is not None
            else None
        )

        self._fit_progress_label.setText(f"Fit in progress for Range {idx + 1}...")

        def _task():
            return fit_parameter_model(
                x=x_vals,
                y=y_vals,
                yerr=y_errs,
                model=model_snapshot,
                parameters=params_snapshot,
                x_min=x_min,
                x_max=x_max,
                error_mode=error_mode,
                error_value=error_value,
                windows=windows,
                xerr=x_errs,
                # Multi-start robustness (item 3.4): 4 extra deterministic starts
                # plus the user's own start; seed fixed so the run is reproducible.
                extra_starts=4,
                seed=0,
            )

        def _on_done(result: object) -> None:
            fit_result = result
            fit_range.result = fit_result
            if fit_result.success:
                fit_range.parameters = fit_result.parameters

            # A refit invalidates any suggestion computed against the previous
            # fit (stale utility curve / result line must never survive it), and
            # invalidates this range's fitted alternatives (they were compared
            # against the previous leader/masked-data fit) — spec item 1.
            # _select_range below re-derives the section's enabled state from
            # the fresh result via _refresh_suggest_section.
            self._discrimination_candidates.pop(idx, None)
            if idx == self._active_range_idx:
                self._clear_suggestion()
                self._compare_candidates_label.setText("")

            # Item 4.3: no per-fit success/failure MODAL. _select_range refreshes
            # the χ² label (inline "Fit successful …" / "Fit failed: …") and tints
            # the result box green on success, so both outcomes read inline.
            self._select_range(idx)

            if fit_result.success:
                # Item 4.2: remember the converged model as the default for the
                # next fresh dialog of this (base_param, x_key).
                _store_last_model_expression(
                    self._parameter_name,
                    self._x_key,
                    fit_range.model.component_expression_string(),
                    self._model_memory,
                )

        self._start_fit_task(_task, _on_done)

    # -- "Suggest next point" section (BED, Phase 2 — §5.4) -------------------
    #
    # Enabled only for the ACTIVE range when it has a successful fit with a
    # covariance AND the dialog's error mode produces a meaningful noise model
    # (i.e. not NONE/SCATTER — those carry no real sigma to interpolate). The
    # suggestion itself runs on the GUI thread (milliseconds); only the
    # Monte-Carlo calibration pass runs off-thread via self._tasks.

    def _build_suggest_section(self) -> QFrame:
        """Build the "Suggest next point" group; returns its outer container."""
        section, layout = make_section("Suggest next point")

        self._suggest_disabled_hint = QLabel("")
        self._suggest_disabled_hint.setWordWrap(True)
        self._suggest_disabled_hint.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._suggest_disabled_hint)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self._suggest_target_combo = QComboBox()
        self._suggest_target_combo.setToolTip(
            "The parameter to minimise the posterior uncertainty of (c-optimal), "
            "or the whole covariance ellipsoid (D-optimal, all parameters)."
        )
        self._suggest_target_combo.currentIndexChanged.connect(self._on_suggest_target_changed)
        target_row.addWidget(self._suggest_target_combo, 1)
        layout.addLayout(target_row)

        goal_row = QHBoxLayout()
        goal_row.addWidget(QLabel("Precision goal:"))
        self._suggest_goal_edit = QLineEdit()
        self._suggest_goal_edit.setPlaceholderText("e.g. 0.1")
        self._suggest_goal_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 12, self))
        self._suggest_goal_edit.setToolTip(
            "Optional target sigma for the selected parameter (same units). "
            "Solves for the event-count factor needed at the suggested x; "
            "only meaningful for a single-parameter (c-optimal) target."
        )
        goal_row.addWidget(self._suggest_goal_edit, 1)
        layout.addLayout(goal_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Candidate range:"))
        # Seeded from the measured x span, which is fixed for the dialog's
        # lifetime (the series is passed at construction) — users widen it to
        # allow extrapolated suggestions.
        self._suggest_min_field = FloatLimitField(
            self._x_min_data, value_range=(-1e12, 1e12), decimals=6
        )
        self._suggest_max_field = FloatLimitField(
            self._x_max_data, value_range=(-1e12, 1e12), decimals=6
        )
        range_row.addWidget(self._suggest_min_field)
        range_row.addWidget(QLabel("–"))
        range_row.addWidget(self._suggest_max_field)
        range_row.addStretch()
        layout.addLayout(range_row)

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Typical run (Mevents):"))
        self._suggest_typical_run_edit = QLineEdit()
        self._suggest_typical_run_edit.setPlaceholderText("optional")
        self._suggest_typical_run_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 6, self))
        self._suggest_typical_run_edit.setToolTip(
            "Display-only: your instrument's typical run size, used to convert "
            "the events factor into an approximate Mevents figure."
        )
        self._suggest_typical_run_edit.textChanged.connect(self._on_suggest_conversion_changed)
        rate_row.addWidget(self._suggest_typical_run_edit)
        rate_row.addWidget(QLabel("Rate (Mevents/h):"))
        self._suggest_rate_edit = QLineEdit()
        self._suggest_rate_edit.setPlaceholderText("optional")
        self._suggest_rate_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 6, self))
        self._suggest_rate_edit.setToolTip(
            "Display-only: your instrument's count rate, used to additionally "
            "show an equivalent counting time. Never affects the computation."
        )
        self._suggest_rate_edit.textChanged.connect(self._on_suggest_conversion_changed)
        rate_row.addWidget(self._suggest_rate_edit)
        layout.addLayout(rate_row)

        # ── Cost weighting (Phase 3, §8.2) ────────────────────────────────────
        # Off by default; a compact grid of four fields beneath the rate row.
        self._cost_weight_check = QCheckBox("Weight by measurement cost")
        self._cost_weight_check.setToolTip(
            "Weight the displayed utility curve(s) by a crude movement/counting "
            "cost model (TAS-AI's IG^0.7 / time): utility**0.7 / (count_time + "
            "move_time(x)). Recomputes best_x as the argmax of the weighted "
            "curve; the analytic post-sigma/events figures still refer to the "
            "unweighted point, so they are dropped from the result line when "
            "the weighted peak differs."
        )
        self._cost_weight_check.toggled.connect(self._on_cost_weight_toggled)
        layout.addWidget(self._cost_weight_check)

        cost_row = QHBoxLayout()
        cost_row.addWidget(QLabel("Count time/pt (h):"))
        self._cost_count_time_edit = QLineEdit()
        self._cost_count_time_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 6, self))
        self._cost_count_time_edit.setPlaceholderText("e.g. 2")
        cost_row.addWidget(self._cost_count_time_edit)
        cost_row.addWidget(QLabel("Move up (h/+1x):"))
        self._cost_up_rate_edit = QLineEdit()
        self._cost_up_rate_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 6, self))
        self._cost_up_rate_edit.setPlaceholderText("e.g. 0.1")
        cost_row.addWidget(self._cost_up_rate_edit)
        cost_row.addWidget(QLabel("Move down (h/-1x):"))
        self._cost_down_rate_edit = QLineEdit()
        self._cost_down_rate_edit.setValidator(QDoubleValidator(0.0, 1.0e12, 6, self))
        self._cost_down_rate_edit.setPlaceholderText("e.g. 0.1")
        cost_row.addWidget(self._cost_down_rate_edit)
        cost_row.addWidget(QLabel("Current x:"))
        self._cost_current_x_edit = QLineEdit()
        self._cost_current_x_edit.setValidator(QDoubleValidator(-1.0e12, 1.0e12, 6, self))
        cost_row.addWidget(self._cost_current_x_edit)
        layout.addLayout(cost_row)
        for edit in (
            self._cost_count_time_edit,
            self._cost_up_rate_edit,
            self._cost_down_rate_edit,
            self._cost_current_x_edit,
        ):
            edit.textChanged.connect(self._on_cost_field_changed)
        self._on_cost_weight_toggled(False)

        button_row = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest")
        self._suggest_btn.clicked.connect(self._on_suggest_clicked)
        button_row.addWidget(self._suggest_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self._suggest_result_label = QLabel("")
        self._suggest_result_label.setTextFormat(Qt.TextFormat.RichText)
        self._suggest_result_label.setWordWrap(True)
        layout.addWidget(self._suggest_result_label)

        self._suggest_warning_label = QLabel("")
        self._suggest_warning_label.setTextFormat(Qt.TextFormat.RichText)
        self._suggest_warning_label.setWordWrap(True)
        self._suggest_warning_label.setVisible(False)
        layout.addWidget(self._suggest_warning_label)

        # ── "Compare against" discrimination subsection (Phase 3, §8.1) ───────
        compare_section, compare_layout = make_section("Compare against")
        layout.addWidget(compare_section)

        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Model:"))
        self._compare_model_combo = QComboBox()
        self._compare_model_combo.setToolTip(
            "Quick-pick a single alternative component, or use Edit… to build a "
            "composite model the same way the primary model is built."
        )
        self._compare_model_combo.currentIndexChanged.connect(self._on_compare_model_combo_changed)
        compare_row.addWidget(self._compare_model_combo, 1)
        self._compare_edit_btn = QPushButton("Edit…")
        self._compare_edit_btn.setToolTip(
            "Open the same model-builder dialog used for the primary model to "
            "compose the alternative candidate."
        )
        self._compare_edit_btn.clicked.connect(self._on_compare_edit_clicked)
        compare_row.addWidget(self._compare_edit_btn)
        compare_layout.addLayout(compare_row)

        compare_button_row = QHBoxLayout()
        self._compare_fit_btn = QPushButton("Fit && compare")
        self._compare_fit_btn.setToolTip(
            "Fit the alternative model over the same masked data as the active "
            "range's fit, off-thread, and add it to the candidate list below."
        )
        self._compare_fit_btn.clicked.connect(self._on_compare_fit_clicked)
        compare_button_row.addWidget(self._compare_fit_btn)
        compare_button_row.addStretch()
        compare_layout.addLayout(compare_button_row)

        self._compare_candidates_label = QLabel("")
        self._compare_candidates_label.setTextFormat(Qt.TextFormat.RichText)
        self._compare_candidates_label.setWordWrap(True)
        compare_layout.addWidget(self._compare_candidates_label)

        self._discrimination_result_label = QLabel("")
        self._discrimination_result_label.setTextFormat(Qt.TextFormat.RichText)
        self._discrimination_result_label.setWordWrap(True)
        compare_layout.addWidget(self._discrimination_result_label)

        return section

    def _suggest_disabled_reason(self, idx: int | None) -> str | None:
        """Return why the section is disabled for range *idx*, or None if enabled."""
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return "Select a range."
        result = self._result_for_range(idx)
        if result is None or not getattr(result, "success", False):
            return "Run a successful fit for this range to suggest a next point."
        if getattr(result, "covariance", None) is None:
            return (
                "No covariance available from this fit (HESSE did not run/"
                "converge) — cannot estimate parameter sensitivities."
            )
        if self._error_mode() in (ErrorMode.NONE, ErrorMode.SCATTER):
            return (
                "Errors are set to None/Estimate-from-scatter — there is no "
                "real noise model to predict a new point's uncertainty from. "
                "Choose Column, Percent, or Absolute errors."
            )
        return None

    def _refresh_suggest_section(self) -> None:
        """Sync the section's enabled state + target combo to the active range."""
        idx = self._active_range_idx
        reason = self._suggest_disabled_reason(idx)
        enabled = reason is None

        for widget in (
            self._suggest_target_combo,
            self._suggest_goal_edit,
            self._suggest_min_field,
            self._suggest_max_field,
            self._suggest_btn,
        ):
            widget.setEnabled(enabled)
        self._suggest_section.setToolTip(reason or "")
        self._suggest_disabled_hint.setText(reason or "")
        self._suggest_disabled_hint.setVisible(not enabled)

        # "Compare against" needs the same successful-fit gate (it fits over
        # the same masked data), but not the error-mode noise-model gate on
        # its own — the AIC ranking only needs chi2, though the
        # discrimination-utility overlay reuses the same noise model as the
        # refinement band, so it is gated identically for simplicity.
        for widget in (
            self._compare_model_combo,
            self._compare_edit_btn,
            self._compare_fit_btn,
        ):
            widget.setEnabled(enabled)

        if not enabled:
            self._compare_candidates_label.setText("")
            self._discrimination_result_label.setText("")
            return

        fit_range = self._fit.ranges[idx]
        free_names = [p.name for p in fit_range.parameters.free_parameters]
        current = self._suggest_target_combo.currentData()
        with QSignalBlocker(self._suggest_target_combo):
            self._suggest_target_combo.clear()
            self._suggest_target_combo.addItem(
                "All parameters (D-optimal)", userData=_SUGGEST_ALL_PARAMS
            )
            for name in free_names:
                self._suggest_target_combo.addItem(name, userData=name)
            # Preserve the previous selection if it is still a free parameter;
            # otherwise default to the first free parameter (spec default).
            restore_idx = 0
            if current in free_names:
                restore_idx = self._suggest_target_combo.findData(current)
            elif free_names:
                restore_idx = self._suggest_target_combo.findData(free_names[0])
            self._suggest_target_combo.setCurrentIndex(max(0, restore_idx))

        self._on_suggest_target_changed()
        self._refresh_compare_model_combo()
        self._refresh_candidates_label()
        self._refresh_cost_current_x_default()

    def _refresh_compare_model_combo(self) -> None:
        """Populate the quick-pick combo with this context's component pool.

        Excludes the active range's own model expression from the default
        selection (best-effort — a composite expression simply won't match
        any single entry) so "Fit & compare" defaults to something
        different from the primary model.
        """
        current = self._compare_model_combo.currentData()
        custom = getattr(self, "_compare_custom_model", None)
        with QSignalBlocker(self._compare_model_combo):
            self._compare_model_combo.clear()
            if custom is not None:
                self._compare_model_combo.addItem(
                    f"{custom.component_expression_string()} (custom)",
                    userData=_COMPARE_CUSTOM,
                )
            for name in self._component_pool:
                self._compare_model_combo.addItem(name, userData=name)
            restore_idx = self._compare_model_combo.findData(current)
            if restore_idx < 0:
                idx = self._active_range_idx
                active_expr = (
                    self._fit.ranges[idx].model.component_expression_string()
                    if idx is not None
                    else ""
                )
                for i in range(self._compare_model_combo.count()):
                    if self._compare_model_combo.itemData(i) != active_expr:
                        restore_idx = i
                        break
                restore_idx = max(restore_idx, 0)
            self._compare_model_combo.setCurrentIndex(restore_idx)
        # A model built via Edit… (composite/custom) is stashed here and takes
        # priority over the quick-pick combo until the user picks a plain
        # component again; a fresh range/model change drops it.
        if not hasattr(self, "_compare_custom_model"):
            self._compare_custom_model: ParameterCompositeModel | None = None

    def _refresh_cost_current_x_default(self) -> None:
        """Default "Current x" to the last measured x; refresh when data changes."""
        if self._cost_current_x_edit.text().strip():
            return
        xs = np.asarray(self._x, dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(xs))
        if finite_idx.size == 0:
            return
        # "Last measured x" means the last finite point in measurement order
        # (position in the series), not the numerically largest x.
        last_x = float(xs[finite_idx[-1]])
        self._cost_current_x_edit.setText(f"{last_x:.6g}")

    def _on_suggest_target_changed(self, *_args: object) -> None:
        """Precision-goal field only makes sense for a single (c-optimal) target."""
        target = self._suggest_target_combo.currentData()
        is_c_optimal = target is not None and target != _SUGGEST_ALL_PARAMS
        self._suggest_goal_edit.setEnabled(is_c_optimal)
        if not is_c_optimal:
            self._suggest_goal_edit.clear()

    def _effective_y_err_for_active_fit(self) -> np.ndarray:
        """The per-point sigma the active range's fit was actually weighted with.

        Reuses the SAME error-mode resolution the fit itself uses
        (``apply_error_mode``) so the suggestion's noise model matches the fit
        that produced its covariance exactly. Falls back to the raw ``yerr``
        column when the mode yields ``None`` (unit weights with no column).
        """
        resolved = apply_error_mode(self._y, self._yerr, self._error_mode(), self._error_value())
        if resolved is None:
            return np.ones_like(self._y)
        return resolved

    def _clear_suggestion(self) -> None:
        """Drop any stale suggestion overlay + result line (spec item 7).

        Also drops the discrimination overlay/state (Part A): both bands
        share the same clearing rules — a refit, range switch, or error-mode
        change invalidates both, never just one.
        """
        self._last_suggestion = None
        self._last_suggestion_calibration = None
        self._pending_calibration = None
        if getattr(self, "_suggest_result_label", None) is not None:
            self._suggest_result_label.setText("")
        if getattr(self, "_suggest_warning_label", None) is not None:
            self._suggest_warning_label.setText("")
            self._suggest_warning_label.setVisible(False)
        preview = getattr(self, "_preview", None)
        if preview is not None:
            preview.set_suggestion(None)
        self._clear_discrimination_display()

    def _clear_discrimination_candidates(self) -> None:
        """Drop the active range's fitted alternative candidates (spec item 1).

        Called whenever the range's primary fit reruns or its model
        expression changes — the candidates were fitted against a specific
        masked dataset/leader and do not survive either.
        """
        idx = self._active_range_idx
        if idx is not None:
            self._discrimination_candidates.pop(idx, None)
        if getattr(self, "_compare_candidates_label", None) is not None:
            self._compare_candidates_label.setText("")
        self._clear_discrimination_display()

    def _clear_discrimination_display(self) -> None:
        """Drop the discrimination suggestion/overlay only (candidates kept)."""
        self._last_discrimination = None
        if getattr(self, "_discrimination_result_label", None) is not None:
            self._discrimination_result_label.setText("")
        preview = getattr(self, "_preview", None)
        if preview is not None:
            preview.set_discrimination(None)

    def _on_suggest_clicked(self) -> None:
        """Compute a fresh suggestion for the active range (GUI thread; ms)."""
        idx = self._active_range_idx
        if self._suggest_disabled_reason(idx) is not None:
            return
        self._clear_suggestion()

        fit_range = self._fit.ranges[idx]
        result = self._result_for_range(idx)
        covariance = result.covariance
        target_data = self._suggest_target_combo.currentData()
        target = None if target_data in (None, _SUGGEST_ALL_PARAMS) else str(target_data)

        goal_text = self._suggest_goal_edit.text().strip()
        sigma_goal: float | None = None
        if target is not None and goal_text:
            try:
                parsed = float(goal_text)
                if np.isfinite(parsed) and parsed > 0.0:
                    sigma_goal = parsed
            except ValueError:
                sigma_goal = None

        y_err = self._effective_y_err_for_active_fit()

        suggestion = suggest_next_point(
            fit_range.model,
            fit_range.parameters,
            covariance,
            self._x,
            y_err,
            self._suggest_min_field.value(),
            self._suggest_max_field.value(),
            target=target,
            sigma_goal=sigma_goal,
        )
        self._last_suggestion = suggestion
        self._apply_suggestion_to_canvas(suggestion)
        self._update_suggest_result_label(suggestion, calibration=None)

        if target is not None and np.isfinite(suggestion.best_x):
            self._launch_calibration(fit_range, result, suggestion, y_err)

    # -- cost weighting (Phase 3, §8.2) ---------------------------------------
    #
    # Pure display recomposition: the checkbox/fields never trigger a
    # re-suggest, only a re-render from the last stored suggestion(s) (spec
    # item 3). "Weighted" means: apply cost_weighted_utility to the RAW curve,
    # recompute best_x as its argmax, and use that x for the canvas marker and
    # result line — the events/sigma tail (which describes the unweighted
    # point) is dropped whenever the weighted argmax differs.

    def _read_cost_model(self) -> tuple[float, float, float, float] | None:
        """Return (count_time, up_rate, down_rate, x_current) iff valid+enabled.

        Validated here (not in core): count_time > 0, rates >= 0, x_current
        finite. Invalid/incomplete input means "not applied" — the caller
        falls back to the unweighted curve, matching the core's own silent
        no-op guard in ``cost_weighted_utility``.
        """
        if not self._cost_weight_check.isChecked():
            return None
        try:
            count_time = float(self._cost_count_time_edit.text())
            up_rate = float(self._cost_up_rate_edit.text())
            down_rate = float(self._cost_down_rate_edit.text())
            x_current = float(self._cost_current_x_edit.text())
        except ValueError:
            return None
        if not (
            np.isfinite(count_time)
            and np.isfinite(up_rate)
            and np.isfinite(down_rate)
            and np.isfinite(x_current)
        ):
            return None
        if count_time <= 0.0 or up_rate < 0.0 or down_rate < 0.0:
            return None
        return count_time, up_rate, down_rate, x_current

    def _display_curve(self, suggestion: NextPointSuggestion) -> tuple[np.ndarray, float, bool]:
        """Return (utility_for_display, best_x_for_display, cost_weighted).

        Applies the cost model to ``suggestion.utility`` when enabled+valid;
        otherwise returns the suggestion's own curve/best_x unchanged.
        """
        cost = self._read_cost_model()
        if cost is None or suggestion.x_candidates.size == 0:
            return suggestion.utility, suggestion.best_x, False
        count_time, up_rate, down_rate, x_current = cost
        weighted = cost_weighted_utility(
            suggestion.x_candidates,
            suggestion.utility,
            x_current,
            count_time=count_time,
            up_rate=up_rate,
            down_rate=down_rate,
        )
        if not np.any(np.isfinite(weighted)):
            return suggestion.utility, suggestion.best_x, False
        best_idx = int(np.nanargmax(weighted))
        weighted_best_x = float(suggestion.x_candidates[best_idx])
        return weighted, weighted_best_x, True

    def _on_cost_weight_toggled(self, checked: bool) -> None:
        for edit in (
            self._cost_count_time_edit,
            self._cost_up_rate_edit,
            self._cost_down_rate_edit,
            self._cost_current_x_edit,
        ):
            edit.setEnabled(checked)
        self._on_cost_field_changed()

    def _on_cost_field_changed(self, *_args: object) -> None:
        """Re-render from the stored suggestion(s) without re-running suggest."""
        if self._last_suggestion is not None:
            self._apply_suggestion_to_canvas(self._last_suggestion)
            self._update_suggest_result_label(
                self._last_suggestion, calibration=self._last_calibration()
            )
        if self._last_discrimination is not None:
            self._apply_discrimination_to_canvas(self._last_discrimination)
            self._update_discrimination_result_label(self._last_discrimination)

    def _apply_suggestion_to_canvas(self, suggestion: NextPointSuggestion) -> None:
        preview = getattr(self, "_preview", None)
        if preview is None:
            return
        utility, best_x, _weighted = self._display_curve(suggestion)
        preview.set_suggestion(
            SuggestionOverlay(
                x=suggestion.x_candidates,
                utility=utility,
                extrapolated=suggestion.extrapolated,
                best_x=best_x,
            )
        )

    def _suggest_conversion_line(self, events_factor: float | None) -> str:
        """Optional "≈ N Mevents" / "≈ N h" tail (spec item 6; display-only)."""
        if events_factor is None:
            return ""
        typical_text = self._suggest_typical_run_edit.text().strip()
        if not typical_text:
            return ""
        try:
            typical = float(typical_text)
        except ValueError:
            return ""
        if not np.isfinite(typical) or typical <= 0.0:
            return ""
        mevents = events_factor * typical
        line = f"≈ {mevents:.3g} Mevents"
        rate_text = self._suggest_rate_edit.text().strip()
        if rate_text:
            try:
                rate = float(rate_text)
            except ValueError:
                rate = float("nan")
            if np.isfinite(rate) and rate > 0.0:
                line += f" ≈ {mevents / rate:.3g} h"
        return line

    def _on_suggest_conversion_changed(self, *_args: object) -> None:
        """Typical-run/rate fields are pure display — re-render the result line."""
        if self._last_suggestion is None:
            return
        self._update_suggest_result_label(
            self._last_suggestion, calibration=self._last_calibration()
        )

    def _last_calibration(self) -> SuggestionCalibration | None:
        return getattr(self, "_last_suggestion_calibration", None)

    def _update_suggest_result_label(
        self,
        suggestion: NextPointSuggestion,
        *,
        calibration: SuggestionCalibration | None,
        calibrating: bool = False,
    ) -> None:
        """Render the result line + warnings for the current suggestion state."""
        if not np.isfinite(suggestion.best_x):
            self._suggest_result_label.setText(
                info_html("No informative candidate found in this range.")
            )
            self._set_suggest_warnings(suggestion.warnings)
            return

        _utility, display_x, weighted = self._display_curve(suggestion)
        # Cost-weighting moves WHERE (display_x) but the analytic post-sigma/
        # events figures below refer only to suggestion.best_x — when the
        # weighted argmax differs, those figures are no longer valid at
        # display_x, so they are dropped and a "(cost-weighted)" note is
        # added instead (spec item 2).
        moved = weighted and not np.isclose(display_x, suggestion.best_x)

        parts = [f"Measure at x = {display_x:.4g}"]

        if suggestion.target is not None and not moved:
            if suggestion.target_unreachable:
                floor = suggestion.floor_sigma
                floor_text = f" (floor sigma({suggestion.target}) ~ {floor:.3g})" if floor else ""
                parts.append(
                    f"the precision goal cannot be reached with a single new point{floor_text}"
                )
            elif suggestion.events_factor_to_target is not None:
                parts.append(
                    f"× {suggestion.events_factor_to_target:.2g} of a typical run's statistics"
                )

            if calibrating:
                parts.append("calibrating…")
            elif calibration is not None and np.isfinite(calibration.realized_post_sigma):
                parts.append(
                    f"→ σ({suggestion.target}) ≈ {calibration.realized_post_sigma:.3g} "
                    "(MC-calibrated)"
                )
            elif suggestion.predicted_post_sigma is not None:
                parts.append(
                    f"→ σ({suggestion.target}) ≈ {suggestion.predicted_post_sigma:.3g} (approximate)"
                )

        if not moved:
            conversion = self._suggest_conversion_line(suggestion.events_factor_to_target)
            if conversion:
                parts.append(conversion)

        if weighted:
            parts.append("(cost-weighted)")

        self._suggest_result_label.setText(" ".join(parts))

        warnings = list(suggestion.warnings)
        if calibration is not None:
            warnings.extend(calibration.warnings)
        self._set_suggest_warnings(warnings)

    def _set_suggest_warnings(self, warnings: Sequence[str]) -> None:
        text = list(dict.fromkeys(w for w in warnings if w))
        if not text:
            self._suggest_warning_label.setText("")
            self._suggest_warning_label.setVisible(False)
            return
        self._suggest_warning_label.setText(warning_html(" ".join(text)))
        self._suggest_warning_label.setVisible(True)

    def _launch_calibration(
        self,
        fit_range: ModelFitRange,
        result: object,
        suggestion: NextPointSuggestion,
        y_err: np.ndarray,
    ) -> None:
        """Off-thread Monte-Carlo calibration of a c-optimal suggestion (§3.5)."""
        if self._calibration_in_progress:
            # A calibration is already running for a superseded suggestion; the
            # completion callbacks drain this pending request so a fresh
            # suggestion never silently stays uncalibrated.
            self._pending_calibration = (fit_range, result, suggestion, y_err)
            return

        # Round-trip through the dict codec: a bare constructor call from the
        # name/operator lists would drop the parentheses (they default to
        # zeros) and silently change precedence for parenthesized composites.
        model_snapshot = ParameterCompositeModel.from_dict(fit_range.model.to_dict())
        params_snapshot = ParameterSet(
            [
                Parameter(
                    name=p.name,
                    value=float(p.value),
                    min=float(p.min),
                    max=float(p.max),
                    fixed=bool(p.fixed),
                )
                for p in fit_range.parameters
            ]
        )
        x_vals = np.asarray(self._x, dtype=float).copy()
        y_vals = np.asarray(self._y, dtype=float).copy()
        y_errs = np.asarray(y_err, dtype=float).copy()
        target = suggestion.target
        best_x = suggestion.best_x
        events_factor = suggestion.events_factor_to_target or 1.0
        predicted_post_sigma = suggestion.predicted_post_sigma
        generation = self._active_range_idx
        request_token = suggestion

        def _worker(_worker: object) -> object:
            return calibrate_suggestion(
                model_snapshot,
                params_snapshot,
                x_vals,
                y_vals,
                y_errs,
                best_x,
                target=target,
                events_factor=events_factor,
                n_trials=30,
                seed=0,
                predicted_post_sigma=predicted_post_sigma,
            )

        def _on_done(calibration: object) -> None:
            self._calibration_in_progress = False
            if self._drain_pending_calibration():
                return
            # Drop a stale result: the active range changed, or a fresh
            # suggestion superseded this one, since the worker was launched.
            if self._active_range_idx != generation or self._last_suggestion is not request_token:
                return
            self._last_suggestion_calibration = calibration
            self._update_suggest_result_label(suggestion, calibration=calibration)

        def _on_error(_message: str) -> None:
            self._calibration_in_progress = False
            if self._drain_pending_calibration():
                return
            # Clear the transient "calibrating…" note; the analytic figure
            # (labelled approximate) remains the best available.
            if self._active_range_idx == generation and self._last_suggestion is request_token:
                self._update_suggest_result_label(suggestion, calibration=None)

        self._calibration_in_progress = True
        self._update_suggest_result_label(suggestion, calibration=None, calibrating=True)
        self._tasks.start(_worker, on_finished=_on_done, on_error=_on_error)

    def _drain_pending_calibration(self) -> bool:
        """Launch a calibration queued while another was in flight.

        Returns True when a pending request was (re)launched — the caller's
        own result is then stale by construction and must be dropped. The
        pending request is only honoured if it still matches the live
        suggestion; anything else (range switch, cleared suggestion) is
        discarded.
        """
        pending = getattr(self, "_pending_calibration", None)
        self._pending_calibration = None
        if pending is None:
            return False
        fit_range, result, suggestion, y_err = pending
        if self._last_suggestion is not suggestion:
            return False
        self._launch_calibration(fit_range, result, suggestion, y_err)
        return True

    # -- "Compare against" model discrimination (BED, Phase 3 — §8.1) ---------
    #
    # UX: a model picker (component-pool quick-pick combo, or Edit… reusing the
    # SAME ParameterModelBuilderDialog affordance the primary model uses) plus
    # "Fit & compare". The chosen alternative is fitted over the SAME masked
    # data as the active range's fit, off-thread, following _launch_calibration's
    # snapshot/token/pending pattern. Successful fits accumulate in
    # self._discrimination_candidates[range_idx]; the AIC display and the
    # discrimination suggestion (suggest_discriminating_point, leader vs ALL
    # alternatives) refresh from that list.

    def _selected_compare_model(self) -> ParameterCompositeModel | None:
        """The model "Fit & compare" would fit — exactly what the combo shows:
        the "(custom)" entry maps to the Edit… composite, any other entry to
        that single component. The displayed selection is the sole truth (a
        hidden stash silently overriding the combo caused fits of something
        other than what was on screen)."""
        data = self._compare_model_combo.currentData()
        if data == _COMPARE_CUSTOM:
            return getattr(self, "_compare_custom_model", None)
        if not data:
            return None
        try:
            return ParameterCompositeModel([str(data)], [])
        except Exception:
            return None

    def _on_compare_edit_clicked(self) -> None:
        """Open the same model-builder dialog the primary model uses (§8.1
        item 2: "reuse whatever affordance the dialog already uses"), seeded
        from the current quick-pick/custom selection."""
        initial = self._selected_compare_model()
        dlg = ParameterModelBuilderDialog(
            component_pool=self._component_pool,
            initial_model=initial,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model = dlg.get_model()
        if model is None:
            return
        self._adopt_custom_compare_model(model)

    def _adopt_custom_compare_model(self, model: ParameterCompositeModel) -> None:
        """Surface an Edit…-built composite as a selected "(custom)" combo entry."""
        self._compare_custom_model = model
        self._refresh_compare_model_combo()
        with QSignalBlocker(self._compare_model_combo):
            self._compare_model_combo.setCurrentIndex(
                max(0, self._compare_model_combo.findData(_COMPARE_CUSTOM))
            )
        self._compare_model_combo.setToolTip(
            f"Custom (via Edit…): {model.component_expression_string()}"
        )

    def _on_compare_model_combo_changed(self, *_args: object) -> None:
        """Picking a plain component drops the Edit… composite and its entry."""
        if self._compare_model_combo.currentData() == _COMPARE_CUSTOM:
            return
        self._compare_custom_model = None
        custom_idx = self._compare_model_combo.findData(_COMPARE_CUSTOM)
        if custom_idx >= 0:
            with QSignalBlocker(self._compare_model_combo):
                self._compare_model_combo.removeItem(custom_idx)
        self._compare_model_combo.setToolTip(
            "Quick-pick a single alternative component, or use Edit… to build a "
            "composite model the same way the primary model is built."
        )

    def _on_compare_fit_clicked(self) -> None:
        """Fit the selected alternative over the active range's masked data."""
        idx = self._active_range_idx
        if self._suggest_disabled_reason(idx) is not None:
            return
        alt_model = self._selected_compare_model()
        if alt_model is None:
            return

        fit_range = self._fit.ranges[idx]
        # Identity token for the leader fit this comparison is against: a
        # refit of the SAME range (possibly with changed bounds/windows)
        # replaces the result object, and a candidate fitted against the old
        # masked data must not join the new leader's AIC ranking — the range
        # index alone cannot catch that.
        leader_token = self._result_for_range(idx)

        # Seed the alternative the SAME way a fresh model of that type is
        # seeded (_create_default_range's heuristic): data-aware trend seeds
        # merged over per-component defaults, mirroring _edit_model's reseed
        # path for a newly-adopted model.
        params = ParameterSet()
        y_mean = float(np.nanmean(self._y)) if np.any(np.isfinite(self._y)) else 0.0
        y_span = (
            float(np.nanmax(self._y) - np.nanmin(self._y)) if np.any(np.isfinite(self._y)) else 1.0
        )
        x_min_data, x_max_data = self._x_min_data, self._x_max_data
        trend_seeds = suggest_trend_seeds(alt_model, self._x, self._y)
        for pname in alt_model.param_names:
            default_val = alt_model.param_defaults[pname]
            if pname in {"c", "b"}:
                default_val = y_mean
            elif pname in {"m", "a"}:
                default_val = y_span if y_span > 0 else default_val
            elif pname.startswith("B0") or pname.startswith("tau") or pname.startswith("nu"):
                default_val = max(1e-6, (x_max_data - x_min_data) / 2.0)
            elif pname.startswith("D"):
                default_val = max(1e-6, default_val)
            params.add(
                Parameter(
                    name=pname,
                    value=float(trend_seeds.get(pname, default_val)),
                    fixed=(pname == "shape_factor_a"),
                )
            )

        # Snapshot exactly like _run_fit: same x/y/err, same range bounds/
        # windows/error mode as the active (leader) fit, so the comparison is
        # over the identical masked dataset.
        model_snapshot = ParameterCompositeModel.from_dict(alt_model.to_dict())
        params_snapshot = ParameterSet(
            [
                Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=p.fixed)
                for p in params
            ]
        )
        x_vals = np.asarray(self._x, dtype=float).copy()
        y_vals = np.asarray(self._y, dtype=float).copy()
        y_errs = np.asarray(self._yerr, dtype=float).copy()
        x_min = fit_range.x_min
        x_max = fit_range.x_max
        windows = list(fit_range.windows) if fit_range.windows else None
        error_mode = self._error_mode()
        error_value = self._error_value()
        generation = idx

        def _worker(_worker: object) -> object:
            return fit_parameter_model(
                x=x_vals,
                y=y_vals,
                yerr=y_errs,
                model=model_snapshot,
                parameters=params_snapshot,
                x_min=x_min,
                x_max=x_max,
                error_mode=error_mode,
                error_value=error_value,
                windows=windows,
                extra_starts=4,
                seed=0,
            )

        def _on_done(result: object) -> None:
            self._compare_fit_in_progress = False
            if self._drain_pending_compare():
                return
            if (
                self._active_range_idx != generation
                or self._result_for_range(generation) is not leader_token
            ):
                return
            if getattr(result, "success", False):
                n_free = len(result.parameters.free_parameters)
                self._discrimination_candidates.setdefault(generation, []).append(
                    (model_snapshot, result.parameters, float(result.chi_squared), n_free)
                )
                self._refresh_candidates_label()
                self._refresh_discrimination_suggestion()
            else:
                self._compare_candidates_label.setText(
                    warning_html(f"Alternative fit failed: {result.message or 'No convergence'}")
                )

        def _on_error(message: str) -> None:
            self._compare_fit_in_progress = False
            if self._drain_pending_compare():
                return
            if self._active_range_idx == generation:
                self._compare_candidates_label.setText(warning_html(f"Fit error: {message}"))

        if self._compare_fit_in_progress:
            self._pending_compare = (_worker, _on_done, _on_error)
            return
        self._compare_fit_in_progress = True
        self._tasks.start(_worker, on_finished=_on_done, on_error=_on_error)

    def _drain_pending_compare(self) -> bool:
        """Launch a compare-fit queued while another was in flight."""
        pending = getattr(self, "_pending_compare", None)
        self._pending_compare = None
        if pending is None:
            return False
        worker, on_done, on_error = pending
        self._compare_fit_in_progress = True
        self._tasks.start(worker, on_finished=on_done, on_error=on_error)
        return True

    def _refresh_candidates_label(self) -> None:
        """Render the fitted-alternatives list with AIC evidence (spec item 3)."""
        idx = self._active_range_idx
        if idx is None:
            self._compare_candidates_label.setText("")
            return
        candidates = self._discrimination_candidates.get(idx, [])
        if not candidates:
            self._compare_candidates_label.setText("")
            return

        result = self._result_for_range(idx)
        leader_chi2 = float(getattr(result, "chi_squared", float("nan")))
        fit_range = self._fit.ranges[idx]
        leader_n_free = len(fit_range.parameters.free_parameters)

        chi2s = [leader_chi2] + [c[2] for c in candidates]
        n_frees = [leader_n_free] + [c[3] for c in candidates]
        weights = aic_weights(chi2s, n_frees)
        leader_weight = weights[0] if weights else 0.0

        lines = [
            f"Leader ({fit_range.model.component_expression_string()}): weight {leader_weight:.3g}"
        ]
        for (model, _params, chi2, n_free), weight in zip(candidates, weights[1:], strict=True):
            ratio = leader_weight / weight if weight > 0.0 else float("inf")
            decisive = " — decisive" if ratio > 100.0 else ""
            lines.append(
                f"{model.component_expression_string()}: weight {weight:.3g} "
                f"/ ratio {ratio:.3g} vs leader{decisive}"
            )
        self._compare_candidates_label.setText("<br>".join(lines))

    def _refresh_discrimination_suggestion(self) -> None:
        """Recompute the discrimination suggestion from the current candidates."""
        idx = self._active_range_idx
        if idx is None:
            return
        candidates = self._discrimination_candidates.get(idx, [])
        if not candidates:
            self._clear_discrimination_display()
            return

        fit_range = self._fit.ranges[idx]
        result = self._result_for_range(idx)
        if result is None or not getattr(result, "success", False):
            self._clear_discrimination_display()
            return

        y_err = self._effective_y_err_for_active_fit()
        alternatives = [(model, params) for model, params, _chi2, _n in candidates]
        suggestion = suggest_discriminating_point(
            fit_range.model,
            fit_range.parameters,
            alternatives,
            self._x,
            y_err,
            self._suggest_min_field.value(),
            self._suggest_max_field.value(),
        )
        self._last_discrimination = suggestion
        self._apply_discrimination_to_canvas(suggestion)
        self._update_discrimination_result_label(suggestion)

    def _apply_discrimination_to_canvas(self, suggestion: NextPointSuggestion) -> None:
        preview = getattr(self, "_preview", None)
        if preview is None:
            return
        utility, best_x, _weighted = self._display_curve(suggestion)
        preview.set_discrimination(
            SuggestionOverlay(
                x=suggestion.x_candidates,
                utility=utility,
                extrapolated=suggestion.extrapolated,
                best_x=best_x,
            )
        )

    def _update_discrimination_result_label(self, suggestion: NextPointSuggestion) -> None:
        """Render the discrimination result line (spec item 4 of §8.1)."""
        if not np.isfinite(suggestion.best_x):
            text = "; ".join(suggestion.warnings) or "No discriminating point found in this range."
            self._discrimination_result_label.setText(info_html(text))
            return

        _utility, display_x, weighted = self._display_curve(suggestion)
        parts = [f"Best discriminating point: x = {display_x:.4g}"]
        if weighted:
            parts.append("(cost-weighted)")
        self._discrimination_result_label.setText(" ".join(parts))
        if suggestion.warnings:
            self._discrimination_result_label.setText(
                self._discrimination_result_label.text()
                + "<br>"
                + warning_html(" ".join(suggestion.warnings))
            )

    # -- template-method hooks: per-row control, error cell, result source -----
    #
    # ``_select_range`` and ``_commit_param_table`` below are the single shared
    # implementations for both this dialog and ``CrossGroupFitDialog``. The two
    # dialogs differ only in a handful of small, well-scoped ways — the param
    # table's editable "Type" column, where a range's fitted result is stored,
    # and the surrounding status text. Those differences are isolated behind the
    # hooks below so the ~150-line table/status flow lives here exactly once. The
    # base implementations reproduce this dialog's own behaviour (a Fixed
    # checkbox, a single-uncertainty error cell, ``fit_range.result`` as the
    # result source); the subclass overrides the hooks, not the flow.

    def _make_param_row_control(self, param: Parameter, row: int) -> QWidget:
        """Build the editable control for the param table's Fixed/Type column.

        Base: a "Fixed" checkbox in a centered container, wired to
        ``_on_param_table_edited``. Subclasses that expose a richer per-row role
        (e.g. Global/Local/Fixed) return their own widget instead. The returned
        widget is installed as the cell widget of column 4; ``row`` is provided
        for subclasses that need it.
        """
        fixed = QCheckBox()
        fixed.setChecked(bool(param.fixed))
        fixed_container = QWidget()
        fixed_layout = QHBoxLayout(fixed_container)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.addWidget(fixed)
        fixed_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fixed.stateChanged.connect(lambda _state: self._on_param_table_edited())
        return fixed_container

    def _read_param_row_control(self, widget: QWidget) -> dict[str, object]:
        """Read the per-row control back into a plain dict for committing.

        Base returns ``{"fixed": bool}`` from the checkbox. Subclasses return the
        extra keys they carry (e.g. ``{"role": str, "fixed": bool}``). ``widget``
        is whatever ``_make_param_row_control`` produced (the cell widget of
        column 4); a mismatched/legacy widget yields ``fixed=False``.
        """
        fixed = False
        if widget is not None and widget.layout() is not None and widget.layout().count() > 0:
            inner = widget.layout().itemAt(0).widget()
            if isinstance(inner, QCheckBox):
                fixed = inner.isChecked()
        return {"fixed": fixed}

    def _result_for_range(self, idx: int) -> object | None:
        """The fitted result to display for range *idx*, or None if not run.

        Base reads the per-range ``fit_range.result``. Subclasses that cache
        results elsewhere (e.g. the cross-group dialog's ``_range_results`` map)
        override this so the shared status/error flow reads the right source.
        """
        return self._fit.ranges[idx].result

    def _error_cell_for_param(
        self, param_name: str, row_control: QWidget, result: object | None
    ) -> QTableWidgetItem:
        """The Error-column cell for one parameter row.

        Base shows the single fitted uncertainty from ``result.uncertainties``.
        Subclasses whose result carries per-group uncertainties (cross-group)
        override this to summarise them. ``row_control`` is the column-4 widget
        for rows whose error presentation depends on the role/type.
        """
        err = np.nan
        if result is not None:
            err = result.uncertainties.get(param_name, np.nan)
        return QTableWidgetItem(f"{err:.4g}" if np.isfinite(err) else "")

    def _set_formula_display(self, fit_range: ModelFitRange) -> None:
        """Render the selected range's formula through the shared pan/zoom box.

        Uses ``FormulaBox.set_formula`` so the expression picks up
        ``insert_formula_break_points`` + a height re-measure. Both this dialog
        and the cross-group subclass inherit this — the subclass no longer writes
        into the bare label, so long global expressions wrap/scroll like the rest.
        """
        self._formula_box.set_formula(f"y(x) = {fit_range.model.formula_string()}")

    def _range_hint_text(self, idx: int) -> str:
        """Hint shown above the param table for the selected range."""
        return (
            f"Editing parameters for Range {idx + 1}. "
            "Run Fit to update result values/uncertainties."
        )

    def _chi2_status_text(self, result: object | None) -> str:
        """Rich-text χ² status line for the selected range's result."""
        if result is None:
            return info_html("Fitting not yet run for selected range")
        if result.success:
            return success_html(
                "Fit successful",
                detail=(
                    f"chi2 = {result.chi_squared:.6g}, "
                    f"reduced chi2 = {result.reduced_chi_squared:.6g}"
                ),
            )
        return error_html(f"Fit failed: {result.message or 'No convergence'}")

    def _quality_status_text(self, fit_range: ModelFitRange, result: object | None) -> str:
        """χ² quality-verdict line for the selected range (empty when none).

        Also appends the item-3.4 bad-minimum signals (parameters pinned at a
        bound; a data-aware start beating the user's start) INLINE beneath the
        χ² verdict. Both are guarded on ``getattr`` of fields that only the
        single-fit ``ParameterModelFitResult`` carries, so the cross-group path
        (whose ``CrossGroupFitResult`` lacks them, and which overrides this hook
        anyway) is unaffected.
        """
        lines = [self._quality_text_for_range(fit_range)]
        lines.extend(self._bad_minimum_status_lines(result))
        return "<br>".join(line for line in lines if line)

    def _bad_minimum_status_lines(self, result: object | None) -> list[str]:
        """Inline warning/info lines for multi-start bad-minimum signals (3.4).

        Returns an empty list when *result* is None, unsuccessful, or lacks the
        new multi-start fields — so a cross-group / legacy result adds nothing.
        """
        if result is None or not getattr(result, "success", False):
            return []
        lines: list[str] = []
        params_at_bound = getattr(result, "params_at_bound", ())
        if params_at_bound:
            names = ", ".join(str(name) for name in params_at_bound)
            lines.append(
                warning_html(
                    f"Parameters at their limits: {names} — the fit may be "
                    "constrained; widen bounds or re-seed."
                )
            )
        if getattr(result, "seed_beat_user_start", False):
            lines.append(
                info_html(
                    "A data-aware start improved the fit — seeds were re-derived from the data."
                )
            )
        return lines

    def _apply_result_box_style(self, result: object | None) -> None:
        """Tint the result box green on a successful result, neutral otherwise.

        Replaces the per-fit "Fit complete" modal (item 4.3): a converged fit
        reads inline via the same green success surface the rest of the app uses.
        """
        box = getattr(self, "_result_box", None)
        if box is None:
            return
        success = result is not None and bool(getattr(result, "success", False))
        box.setStyleSheet(RESULT_BOX_SUCCESS_STYLE if success else RESULT_BOX_NEUTRAL_STYLE)

    def active_range_index(self) -> int | None:
        """Read-only accessor for the current active range (contract C-ACTIVE)."""
        return self._active_range_idx

    def _set_active_range(self, idx: int | None, *, from_plot: bool = False) -> None:
        """The single source of truth for the active range (contract C-ACTIVE).

        The ONLY writer of ``self._active_range_idx``. Idempotent; bounds-guards
        ``idx``; fans out to every mirror — each card's ``set_active(idx == i)``,
        the details pane (formula/result box/Guess/param table + the C-BOUNDS
        bounds pair + window sub-block), and the canvas ``set_active_range(idx)``
        (via ``_request_preview_update``). No mirror is ever the source.

        ``from_plot`` is reserved for Step 3 (plot-driven selection); both paths
        currently perform the same fan-out, but the parameter freezes the
        signature now.
        """
        if idx is None or idx < 0 or idx >= len(self._fit.ranges):
            return
        self._active_range_idx = idx
        fit_range = self._fit.ranges[idx]

        # Move the active highlight + the Run Fit action to the selected card,
        # and repoint the details-pane fit-region editor at it.
        for card_idx, card in enumerate(self._range_cards):
            is_active = card_idx == idx
            card.set_active(is_active)
            card.set_state(self._range_card_view(card_idx, show_run=is_active))
        self._rebuild_fit_region_rows(idx)

        self._set_formula_display(fit_range)
        self._range_hint_label.setText(self._range_hint_text(idx))

        result = self._result_for_range(idx)
        self._chi2_label.setText(self._chi2_status_text(result))
        self._quality_label.setText(self._quality_status_text(fit_range, result))
        self._apply_result_box_style(result)

        self._param_table.blockSignals(True)
        self._param_table.setRowCount(0)
        for row, param in enumerate(fit_range.parameters):
            self._param_table.insertRow(row)
            display_name = _format_model_param_label(
                fit_range.model,
                param.name,
                self._x_key,
                self._parameter_name,
            )
            name_item = QTableWidgetItem(display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, param.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._param_table.setItem(row, 0, name_item)
            self._param_table.setItem(row, 1, QTableWidgetItem(f"{param.value:.8g}"))
            self._param_table.setItem(row, 2, QTableWidgetItem(f"{param.min:.8g}"))
            self._param_table.setItem(row, 3, QTableWidgetItem(f"{param.max:.8g}"))

            row_control = self._make_param_row_control(param, row)
            self._param_table.setCellWidget(row, 4, row_control)

            err_item = self._error_cell_for_param(param.name, row_control, result)
            self._param_table.setItem(row, 5, err_item)

        self._param_table.blockSignals(False)
        self._param_table.resizeColumnsToContents()
        self._post_select_range(idx)

        # Range switch invalidates any suggestion computed for the previously
        # active range (spec item 7); only clear the LIVE display state
        # (canvas/labels) here — the candidates dict is keyed per range index,
        # so a range's fitted alternatives survive switching away and back —
        # then rebuild the section (incl. the candidates label + discrimination
        # suggestion) for whichever range is now active.
        if getattr(self, "_suggest_section", None) is not None:
            self._clear_suggestion()
            self._refresh_suggest_section()
            self._refresh_discrimination_suggestion()

        self._request_preview_update()

    def _select_range(self, idx: int) -> None:
        """Thin alias for :meth:`_set_active_range` (contract C-ACTIVE).

        Kept so existing callers/tests keep the ``_select_range`` name; all
        active-range writes funnel through the single source of truth.
        """
        self._set_active_range(idx)

    def _post_select_range(self, idx: int) -> None:
        """Hook: called at the end of ``_select_range`` after the table is built.

        Base does nothing. Subclasses use it for post-build cleanup (e.g.
        removing stray legacy cell widgets)."""

    def _on_param_table_edited(self, *_args: object) -> None:
        """Persist parameter edits immediately and invalidate stale fit results."""
        if self._active_range_idx is None:
            return

        fit_range = self._fit.ranges[self._active_range_idx]
        self._commit_param_table()
        fit_range.result = None
        self._chi2_label.setText(
            f'<span style="color:{tokens.ACCENT};">Fitting not yet run for selected range</span>'
        )
        self._quality_label.setText("")
        self._apply_result_box_style(None)
        self._request_preview_update()

    def _commit_param_table(self, *, notify_adjustments: bool = False) -> None:
        if self._active_range_idx is None:
            return

        fit_range = self._fit.ranges[self._active_range_idx]
        new_params = ParameterSet()
        adjustments: list[str] = []
        self._param_table.blockSignals(True)
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            value_item = self._param_table.item(row, 1)
            min_item = self._param_table.item(row, 2)
            max_item = self._param_table.item(row, 3)
            fixed_widget = self._param_table.cellWidget(row, 4)

            if name_item is None or value_item is None:
                continue

            name_data = name_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(name_data, str) and name_data.strip():
                name = name_data.strip()
            else:
                name = name_item.text().strip()
            if not name:
                continue

            seed_rejected = False
            try:
                value = float(value_item.text())
            except (TypeError, ValueError):
                # An unparseable / mangled seed edit (e.g. the Windows input
                # layer turning "-10" into "--0"). Keep the parameter's
                # previous value rather than silently committing 0, which
                # would destroy the user's seed with no feedback.
                if name in fit_range.parameters:
                    value = fit_range.parameters[name].value
                else:
                    value = 0.0
                seed_rejected = True

            try:
                p_min = float(min_item.text()) if min_item is not None else -float("inf")
            except (TypeError, ValueError):
                p_min = -float("inf")

            try:
                p_max = float(max_item.text()) if max_item is not None else float("inf")
            except (TypeError, ValueError):
                p_max = float("inf")

            value, p_min, p_max, notes = _normalize_parameter_limits(name, value, p_min, p_max)
            adjustments.extend(notes)

            if value_item is not None:
                value_item.setText(f"{value:.8g}")
                if seed_rejected:
                    value_item.setBackground(QBrush(QColor(tokens.ACCENT_RED_SOFT)))
                    value_item.setToolTip("Unrecognised number — kept the previous value.")
                else:
                    value_item.setBackground(QBrush())
                    value_item.setToolTip("")
            if min_item is not None:
                min_item.setText(f"{p_min:.8g}")
            if max_item is not None:
                max_item.setText(f"{p_max:.8g}")

            control_state = self._read_param_row_control(fixed_widget)
            fixed = bool(control_state.get("fixed", False))

            new_params.add(Parameter(name=name, value=value, min=p_min, max=p_max, fixed=fixed))

        self._param_table.blockSignals(False)
        fit_range.parameters = new_params

        if adjustments:
            self._range_hint_label.setText(
                "Adjusted parameter values to satisfy model-domain requirements "
                "(e.g. positive tau/B0/nu and non-negative diffusion rates)."
            )
            if notify_adjustments:
                _show_info(self, "Parameter limits adjusted", "; ".join(dict.fromkeys(adjustments)))

    def _on_remove_fit(self) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Cannot remove fit while fitting is in progress.")
            return
        self._removed = True
        self.accept()

    def _refuse_close_while_fitting(self) -> bool:
        """Notify and return True while a fit is running (blocks reject/close)."""
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Please wait for the current fit to finish.")
            return True
        return False

    def reject(self) -> None:
        if self._refuse_close_while_fitting():
            return
        super().reject()

    def closeEvent(self, event) -> None:
        """Refuse to tear down mid-fit; otherwise shut the TaskRunner down."""
        if self._refuse_close_while_fitting():
            event.ignore()
            return
        # Stop the debounce and flag teardown BEFORE shutting the runner down,
        # so a pending preview timer cannot start a fresh worker on the
        # shut-down runner (TaskRunner.shutdown has no re-entry guard).
        self._shutting_down = True
        self._preview_timer.stop()
        self._tasks.shutdown()
        super().closeEvent(event)

    def _start_fit_task(
        self, task: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        if self._fit_in_progress:
            return

        self._fit_in_progress = True
        self._set_fit_ui_busy(True)
        self._fit_done_callback = on_done

        def _run(worker, task=task):
            try:
                return task()
            except Exception as exc:
                # TaskWorker would surface only str(exc); wrap the full
                # traceback so the failure dialog keeps the same detail the
                # old _FitWorker.failed signal carried.
                raise RuntimeError(traceback.format_exc()) from exc

        self._tasks.start(
            _run,
            on_finished=self._on_fit_worker_finished,
            on_error=self._on_fit_worker_failed,
        )

    def _on_fit_worker_finished(self, result: object) -> None:
        """Handle fit completion on the dialog (UI) thread."""
        callback = self._fit_done_callback
        try:
            if callback is not None:
                callback(result)
        except Exception:
            _show_warning(
                self,
                "Fit failed",
                "Unexpected error while applying fit results.\n\n" + traceback.format_exc(),
            )
        finally:
            self._fit_done_callback = None
            self._fit_in_progress = False
            self._set_fit_ui_busy(False)

    def _on_fit_worker_failed(self, trace: str) -> None:
        """Handle fit failure on the dialog (UI) thread."""
        try:
            _show_warning(self, "Fit failed", f"Unexpected error during fitting.\n\n{trace}")
        finally:
            self._fit_done_callback = None
            self._fit_in_progress = False
            self._set_fit_ui_busy(False)

    def _set_fit_ui_busy(self, busy: bool) -> None:
        self._fit_progress_label.setVisible(busy)
        if not busy:
            self._fit_progress_label.setText("")

        # Suppress plot dragging while a fit runs so the user cannot mutate the
        # range/windows underneath an in-flight fit; re-enabled when it settles.
        preview = getattr(self, "_preview", None)
        if preview is not None:
            preview.enable_drag(not busy)

        self._param_table.setEnabled(not busy)
        if hasattr(self, "_guess_seeds_btn"):
            # Don't let a Guess launch race a real fit; leave it disabled while
            # its own guess is in flight too.
            self._guess_seeds_btn.setEnabled(not busy and not self._guess_in_progress)
        if hasattr(self, "_add_range_btn"):
            self._add_range_btn.setEnabled(not busy)
        if hasattr(self, "_remove_fit_btn"):
            self._remove_fit_btn.setEnabled(not busy)

        for button in self._buttons.buttons():
            button.setEnabled(not busy)

        # Cards: disable each card's Run Fit + overflow controls.
        for card in self._range_cards:
            card.set_enabled(not busy)

        # Details-pane fit-region editor: every interval spin pair, each
        # per-interval Remove button, and the "Exclude region…" button are
        # disabled while a fit runs and re-enabled when it settles.
        for i_min, i_max in self._region_row_spins.values():
            i_min.setEnabled(not busy)
            i_max.setEnabled(not busy)
        for btn in self._region_remove_btns:
            btn.setEnabled(not busy)
        if self._exclude_region_btn is not None:
            self._exclude_region_btn.setEnabled(not busy)
