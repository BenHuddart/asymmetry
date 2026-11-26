"""Fit the EuO walkthrough's order parameter from a Fit Parameters TSV export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

EUO_RUNS = (2960, 2925, *range(2928, 2944))


def read_euo_export(path: Path) -> list[tuple[int, float, float, float]]:
    """Read one selected temperature series, preserving the exported errors."""
    columns = ("Run", "T (K)", "frequency (MHz)", "err_frequency (MHz)")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(
            (line for line in stream if not line.startswith("#")), delimiter="\t"
        )
        if not set(columns).issubset(reader.fieldnames or ()) or "Series" in reader.fieldnames:
            raise ValueError(f"Export one Fit Parameters series with columns {', '.join(columns)}")
        by_run = {}
        for row in reader:
            run = int(row["Run"])
            if run not in EUO_RUNS:
                continue
            if run in by_run:
                raise ValueError(f"Run {run} occurs twice in the export")
            values = [row[column] for column in columns[1:]]
            if any(value in ("", None) for value in values):
                raise ValueError(
                    f"Run {run} is missing a temperature, frequency or frequency error"
                )
            temperature, frequency, error = map(float, values)
            if any(
                not np.isfinite(value) or value <= 0 for value in (temperature, frequency, error)
            ):
                raise ValueError(
                    f"Run {run} needs finite positive temperature, frequency and error"
                )
            by_run[run] = (run, temperature, frequency, error)
    missing = set(EUO_RUNS) - by_run.keys()
    if missing:
        raise ValueError(f"Export lacks walkthrough runs {', '.join(map(str, sorted(missing)))}")
    return sorted(by_run.values(), key=lambda row: row[1])


def order_parameter(temperature, nu0, tc, alpha, beta):
    """Return the precession frequency in MHz, zero at and above the transition."""
    return nu0 * np.maximum(1.0 - (temperature / tc) ** alpha, 0.0) ** beta


def fit_euo_export(path: Path):
    """Fit the exported rows using the trend dialog's Column weights."""
    rows = read_euo_export(path)
    temperature, frequency, sigma = np.array(rows, dtype=float)[:, 1:].T
    parameters, _ = curve_fit(
        order_parameter,
        temperature,
        frequency,
        p0=(30.0, 70.0, 1.5, 0.4),
        bounds=((0.0, temperature.max(), 0.01, 0.01), np.inf),
        sigma=sigma,
        absolute_sigma=True,
    )
    residuals = (frequency - order_parameter(temperature, *parameters)) / sigma
    return rows, parameters, float(residuals @ residuals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Fit Parameters TSV with logged temperatures")
    path = parser.parse_args().export
    rows, (nu0, tc, alpha, beta), chi_squared = fit_euo_export(path)
    print("Run    Sample T (K)  Frequency (MHz)  Measured error (MHz)")
    for run, temperature, frequency, error in rows:
        print(f"{run:4}   {temperature:>12g}  {frequency:>15g}  {error:>20g}")
    print(f"{len(rows)} runs, {rows[0][1]:.2f}–{rows[-1][1]:.2f} K")
    print(f"nu0 = {nu0:.1f} MHz, Tc = {tc:.1f} K, alpha = {alpha:.2f}, beta = {beta:.2f}")
    print(f"Weighted chi-square / dof = {chi_squared:.1f} / {len(rows) - 4}")


if __name__ == "__main__":
    main()
