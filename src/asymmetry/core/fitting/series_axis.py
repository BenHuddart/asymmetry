"""Recorded physical axes for ordered fitting series."""

from asymmetry.core.data.dataset import MuonDataset


def axis_value(dataset: MuonDataset, axis_key: str) -> float:
    """Read the selected physical coordinate without inventing missing values."""
    if axis_key == "field":
        return field_value(dataset)
    if axis_key == "temperature":
        return temperature_value(dataset)
    return float(dataset.run_number)


def field_value(dataset: MuonDataset) -> float:
    """Read the recorded field in gauss."""
    field = dataset.metadata.get("field", dataset.run.field if dataset.run is not None else 0.0)
    return float(field) if field is not None else float("nan")


def temperature_value(dataset: MuonDataset) -> float:
    """Read the recorded temperature in kelvin."""
    value = dataset.metadata.get(
        "temperature", dataset.run.temperature if dataset.run is not None else 0.0
    )
    return float(value) if value is not None else float("nan")
