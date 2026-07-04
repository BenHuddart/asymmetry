"""Pure helpers for the Global Parameter Fit results window.

Split out of ``global_parameter_fit_window.py`` to keep the (large) window
module focused on Qt wiring. Everything here is widget-free except
:class:`CorrelationMatrixDialog`, a small self-contained dialog, so the value
formatting and table-export builders can be unit-tested without a window.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from asymmetry.core.fitting.parameters import get_param_info

__all__ = [
    "format_value_with_error",
    "build_global_table_rows",
    "count_free_parameters",
    "information_criteria",
    "global_table_tsv",
    "global_table_csv",
    "global_table_latex",
    "CorrelationMatrixDialog",
]


def count_free_parameters(result) -> int:
    """Return the number of *free* parameters in a cross-group fit result.

    ``k = len(global_parameters) + Σ len(local_parameters[group])`` — the shared
    globals plus every group's local parameters. Fixed parameters are excluded
    *by construction*: :class:`~asymmetry.core.fitting.parameter_models.CrossGroupFitResult`
    keeps fixed values in a separate ``fixed_parameters`` set that is never part
    of ``global_parameters``/``local_parameters``, so they simply do not appear
    in this sum. ``0`` when *result* is ``None``.
    """
    if result is None:
        return 0
    k = len(result.global_parameters)
    for pset in result.local_parameters.values():
        k += len(pset)
    return k


def information_criteria(result) -> dict | None:
    """Return likelihood-based information criteria for a cross-group fit.

    Returns ``{"aic", "aicc", "bic", "k", "n"}`` computed from the fit's total
    χ² and free-parameter count ``k`` (:func:`count_free_parameters`) over
    ``n = result.n_points`` fitted points::

        AIC  = χ² + 2k
        AICc = AIC + 2k(k+1)/(n - k - 1)   (``inf`` when ``n - k - 1 <= 0``)
        BIC  = χ² + k·ln(n)

    Returns ``None`` when ``n <= 0`` or χ² is non-finite (no meaningful
    criterion can be formed).

    .. caution::
       These absolute values are likelihood-based **only for the column /
       percent / absolute error modes**, where χ² is a genuine
       ``Σ (residual/σ)²``. Under ``"none"``/``"scatter"`` the χ² has no
       likelihood interpretation, so the absolute numbers are not meaningful —
       and *any* comparison of two fits' criteria is valid only when they were
       fit to the **same data with the same error mode** (mirroring the caveat
       on :mod:`asymmetry.core.fitting.cross_group_roles`). AIC/AICc/BIC
       *differences* are the comparable quantity, and only under those
       conditions.
    """
    if result is None:
        return None
    n = int(result.n_points)
    if n <= 0:
        return None
    chi2 = float(result.chi_squared)
    if not math.isfinite(chi2):
        return None
    k = count_free_parameters(result)
    aic = chi2 + 2.0 * k
    denom = n - k - 1
    if denom <= 0:
        aicc = float("inf")
    else:
        aicc = aic + (2.0 * k * (k + 1)) / denom
    bic = chi2 + k * math.log(n)
    return {"aic": aic, "aicc": aicc, "bic": bic, "k": k, "n": n}


def format_value_with_error(value: float, err: float | None) -> str:
    """Format ``value(err)`` with the uncertainty rounded to 1–2 sig figs.

    The convention (matching Wu et al. Table I, and the wider μSR literature):
    the value is rounded to the same decimal place as the last significant
    digit of the uncertainty, and the uncertainty is written in parentheses as
    that many digits *in units of the last decimal place*. Examples::

        (63.4, 2.1)      -> "63(2)"
        (0.0674, 0.0031) -> "0.067(3)"      (leading digit ≥ 2 → one sig fig)
        (1.23, 0.012)    -> "1.230(12)"     (leading digit 1 → two sig figs)
        (5.0, 0.0)       -> "5"             (no/degenerate error → plain value)
        (5.0, nan)       -> "5"

    The uncertainty is rounded to one significant figure, except when its
    leading digit is 1 (two significant figures) — the standard PDG rule.
    When *err* is non-finite or non-positive the plain value is returned
    (formatted with a compact ``%g``). When *value* is non-finite it is
    rendered as ``"nan"``/``"inf"`` verbatim.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(v):
        return f"{v:g}"

    if err is None:
        return f"{v:g}"
    try:
        e = float(err)
    except (TypeError, ValueError):
        return f"{v:g}"
    if not math.isfinite(e) or e <= 0.0:
        return f"{v:g}"

    # Determine the exponent of the leading digit of the error.
    exp = math.floor(math.log10(e))
    # Two significant figures when the leading digit is 1 (e.g. 1.2 -> "12"),
    # else one significant figure — the standard PDG rounding rule.
    lead = e / (10.0**exp)
    sig = 2 if lead < 2.0 else 1
    # Decimal place of the last kept error digit.
    last_place = exp - (sig - 1)

    # Round the error to that place and express its digits.
    factor = 10.0**last_place
    err_rounded = round(e / factor) * factor
    # Rounding can bump the leading digit up an order (e.g. 9.6 -> 10); recompute.
    if err_rounded > 0:
        exp2 = math.floor(math.log10(err_rounded))
        lead2 = err_rounded / (10.0**exp2)
        sig2 = 2 if lead2 < 2.0 else 1
        last_place = exp2 - (sig2 - 1)
        factor = 10.0**last_place

    err_digits = int(round(err_rounded / factor))
    value_rounded = round(v / factor) * factor

    if last_place >= 0:
        # Integer-scale uncertainty: no decimals in the value.
        value_text = f"{value_rounded:.0f}"
    else:
        decimals = -last_place
        value_text = f"{value_rounded:.{decimals}f}"

    return f"{value_text}({err_digits})"


def build_global_table_rows(result) -> list[dict[str, object]]:
    """Return the ordered rows backing the global-params table & exports.

    Each row is ``{name, symbol, unit, value, err, fixed}``: ``symbol`` is the
    unicode label (unit stripped — the unit lives in its own column), ``value``
    a float, ``err`` a float or ``None`` (``None`` for fixed params). Shared
    global parameters come first, then fixed ones (flagged ``fixed=True``).
    """
    rows: list[dict[str, object]] = []
    if result is None:
        return rows
    for p in result.global_parameters:
        info = get_param_info(p.name)
        err = result.global_uncertainties.get(p.name)
        rows.append(
            {
                "name": p.name,
                "symbol": info.unicode_label(include_unit=False),
                "unit": info.unit or "",
                "value": float(p.value),
                "err": (float(err) if err is not None and np.isfinite(err) and err >= 0 else None),
                "fixed": False,
            }
        )
    for p in result.fixed_parameters:
        info = get_param_info(p.name)
        rows.append(
            {
                "name": p.name,
                "symbol": info.unicode_label(include_unit=False),
                "unit": info.unit or "",
                "value": float(p.value),
                "err": None,
                "fixed": True,
            }
        )
    return rows


def global_table_tsv(result) -> str:
    """Tab-separated global-parameter table (with header) for the clipboard."""
    lines = ["Parameter\tValue\tUncertainty\tUnits"]
    for row in build_global_table_rows(result):
        name = str(row["symbol"])
        if row["fixed"]:
            name = f"{name} (fixed)"
        value = f"{float(row['value']):.6g}"
        err = row["err"]
        err_text = "" if err is None else f"{float(err):.3g}"
        unit = str(row["unit"])
        lines.append(f"{name}\t{value}\t{err_text}\t{unit}")
    return "\n".join(lines) + "\n"


def global_table_csv(result) -> str:
    """Comma-separated global-parameter table (with header) for CSV export."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Parameter", "Value", "Uncertainty", "Units"])
    for row in build_global_table_rows(result):
        name = str(row["name"])
        if row["fixed"]:
            name = f"{name} (fixed)"
        err = row["err"]
        writer.writerow(
            [
                name,
                f"{float(row['value']):.8g}",
                "" if err is None else f"{float(err):.6g}",
                str(row["unit"]),
            ]
        )
    return buf.getvalue()


def global_table_latex(result, *, parameter_name: str | None = None) -> str:
    """Return a booktabs-style LaTeX ``tabular`` of the global parameters.

    Values are formatted ``value(err)`` via :func:`format_value_with_error`
    (matching the paper's Table I), fixed parameters get a ``[fixed]`` marker,
    and the parameter symbol is the ``get_param_info`` LaTeX label (unit split
    into its own column).
    """
    caption = parameter_name or "global parameter fit"
    lines: list[str] = []
    lines.append("\\begin{tabular}{lll}")
    lines.append("\\toprule")
    lines.append("Parameter & Value & Units \\\\")
    lines.append("\\midrule")
    for row in build_global_table_rows(result):
        info = get_param_info(str(row["name"]))
        symbol = info.latex_label(include_unit=False)
        # Wrap in math mode if the symbol is not already delimited.
        if "$" not in symbol:
            symbol = f"${symbol}$"
        unit = str(row["unit"])
        if row["fixed"]:
            value_text = f"{float(row['value']):g} [fixed]"
        else:
            value_text = format_value_with_error(float(row["value"]), row["err"])
        lines.append(f"{symbol} & {value_text} & {unit} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% global parameter fit: {caption}")
    return "\n".join(lines) + "\n"


class CorrelationMatrixDialog(QDialog):
    """A small dialog showing the free-global-parameter correlation matrix.

    Cells are coloured by ``|ρ|`` (light → saturated red) and labelled to two
    decimal places, mirroring the way correlation matrices are read in fitting
    reports. Built from ``result.global_correlations`` = ``(names, matrix)``.
    """

    def __init__(self, names: list[str], matrix: list[list[float]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global parameter correlations")
        layout = QVBoxLayout(self)

        n = len(names)
        symbols = [get_param_info(name).unicode_label(include_unit=False) for name in names]

        table = QTableWidget(n, n, self)
        table.setHorizontalHeaderLabels(symbols)
        table.setVerticalHeaderLabels(symbols)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i in range(n):
            row = matrix[i] if i < len(matrix) else []
            for j in range(n):
                value = float(row[j]) if j < len(row) else float("nan")
                item = QTableWidgetItem("" if not np.isfinite(value) else f"{value:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if np.isfinite(value):
                    item.setBackground(_correlation_color(value))
                table.setItem(i, j, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table)
        self._table = table

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # Size to the content but keep it modest.
        self.resize(min(120 + 90 * n, 720), min(120 + 40 * n, 560))


def _correlation_color(value: float) -> QColor:
    """Map ``ρ`` to a background colour: white at 0, saturated red at |ρ|=1.

    The diagonal (|ρ|=1) reads strongly; near-zero off-diagonal cells stay
    near-white so a glance surfaces the strongly-correlated pairs.
    """
    mag = min(1.0, abs(float(value)))
    # Interpolate white → a warm red as |ρ| grows.
    r = 255
    g = int(round(255 * (1.0 - mag) + 90 * mag))
    b = int(round(255 * (1.0 - mag) + 70 * mag))
    return QColor(r, g, b)


def _label_indicates_no_verdict(error_mode: str) -> bool:
    """Return ``True`` when χ²ᵣ carries no goodness verdict for *error_mode*."""
    return str(error_mode).lower() in {"none", "scatter"}
