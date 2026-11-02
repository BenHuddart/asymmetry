"""Table cells for core likelihood criteria and their explicit unavailable reasons."""

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem

from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric


class _CriterionItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


def criterion_item(
    criteria: InformationCriteria | None,
    metric: SelectionMetric,
    *,
    unavailable_reason: str | None = None,
) -> QTableWidgetItem:
    """Render a core criterion, with unavailable cells sorted after finite values."""
    metric = SelectionMetric.from_value(metric)
    value = None if criteria is None else criteria.value(metric)
    item = _CriterionItem()
    item.setData(Qt.ItemDataRole.UserRole, math.inf if value is None else value)
    if value is not None:
        item.setData(Qt.ItemDataRole.DisplayRole, float(value))
        return item
    reason = unavailable_reason if criteria is None else criteria.unavailable_reason(metric)
    if not reason:
        raise ValueError("an unavailable criterion requires a reason")
    item.setText("—")
    text = f"{metric.value} undefined: {reason}"
    if criteria is not None:
        n = "unknown" if criteria.sample_count is None else str(criteria.sample_count)
        text += f" (n = {n} points, k = {criteria.parameter_count} free parameters)"
    item.setToolTip(text)
    return item
