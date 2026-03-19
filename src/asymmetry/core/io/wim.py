"""Loader for WiMDA *.wim* files.

The .wim format is a plain-text file produced by the WiMDA program.  It
contains three marked sections (run information, grouping information,
data-set information) followed by columnar asymmetry data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.io.base import BaseLoader


class WimLoader(BaseLoader):
    """Read a ``.wim`` file and return a :class:`MuonDataset`."""

    extensions = [".wim"]
    format_name = "WiMDA (.wim)"

    def load(self, filepath: str) -> MuonDataset:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = path.read_text(encoding="iso-8859-1")

        run_meta = _parse_run_info(content)
        grouping = _parse_group_info(content)
        dataset_info = _parse_dataset_info(content)
        time, asymmetry, error = _parse_data(content)

        run = Run(
            run_number=run_meta.get("run_number", 0),
            metadata={**run_meta, **dataset_info, "source_file": str(path)},
            grouping=grouping,
            source_file=str(path),
        )

        return MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=error,
            metadata=run.metadata,
            run=run,
        )


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _extract_section(content: str, start_marker: str, end_marker: str) -> str:
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start == -1 or end == -1:
        return ""
    return content[start + len(start_marker) : end].strip()


def _parse_run_info(content: str) -> dict:
    section = _extract_section(
        content, "! START OF RUN INFORMATION", "! END OF RUN INFORMATION"
    )
    if not section:
        return {}

    parsed: dict[str, str] = {}
    for line in section.splitlines():
        line = line.lstrip("!").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip().lower().replace(" ", "_")] = value.strip()

    result: dict = {}

    # Run number
    result["run_number"] = int(parsed.get("run_number", "0"))
    result["title"] = parsed.get("title", "")
    result["comment"] = parsed.get("comment", "")
    result["started"] = parsed.get("started", "")
    result["stopped"] = parsed.get("stopped", "")

    # Temperature  e.g. "10.515 K (label 10 K)"
    temp_str = parsed.get("temperature", "")
    m = re.match(r"([\d.]+)\s*K\s*(?:\(label\s*([\d.]+)\s*K\))?", temp_str)
    if m:
        result["temperature"] = float(m.group(1))
        result["temperature_label"] = m.group(2)
    elif temp_str:
        try:
            result["temperature"] = float(temp_str.split()[0])
        except ValueError:
            result["temperature"] = 0.0

    # Field  e.g. "30.00 G"
    field_str = parsed.get("field", "")
    parsed_field = None
    try:
        parsed_field = float(field_str.split()[0]) if field_str else None
    except ValueError:
        parsed_field = None

    # Some instruments write field only in comment, e.g. "LF 32G Bz".
    comment_field = _extract_field_from_comment(result.get("comment", ""))
    result["field_header"] = parsed_field
    result["field_comment_candidate"] = comment_field
    # Default field stays with explicit header value; GUI can optionally apply
    # comment-derived value after prompting user.
    result["field"] = parsed_field if parsed_field is not None else 0.0

    # Histograms  e.g. "8 (25600 bins of 390.625 ps = 10.00 µs)"
    hist_str = parsed.get("histograms", "")
    m = re.match(
        r"(\d+)\s*\((\d+)\s*bins\s*of\s*([\d.]+)\s*ps\s*=\s*([\d.]+)", hist_str
    )
    if m:
        result["histograms_count"] = int(m.group(1))
        result["bin_count"] = int(m.group(2))
        result["bin_width_ps"] = float(m.group(3))
        result["bin_width_us"] = float(m.group(4))

    # Events  e.g. "2.91 MEv grouped in range (raw = 6.76)"
    events_str = parsed.get("events", "")
    m = re.match(
        r"([\d.]+)\s*MEv\s*grouped\s*in\s*range\s*\(\s*raw\s*=\s*([\d.]+)",
        events_str,
    )
    if m:
        result["events_mev"] = float(m.group(1))
        result["events_raw"] = float(m.group(2))

    return result


def _extract_field_from_comment(comment: str) -> float | None:
    """Extract magnetic field in Gauss from comment text.

    Examples matched: "LF 32G", "Bz=150 G", "field: 10.5 gauss".
    """
    if not comment:
        return None

    # Number followed by G/Gauss, optionally with context labels.
    patterns = [
        r"(?i)\b(?:field|bx|by|bz|lf|tf|zf)?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*(?:g|gauss)\b",
        r"(?i)\b([+-]?\d+(?:\.\d+)?)\s*(?:g|gauss)\b",
    ]
    for pat in patterns:
        m = re.search(pat, comment)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _parse_group_info(content: str) -> dict:
    section = _extract_section(
        content,
        "! START OF GROUPING INFORMATION",
        "! END OF GROUPING INFORMATION",
    )
    if not section:
        return {}

    groups: dict[int, list[tuple[int, int]]] = {}
    result: dict = {}

    for line in section.splitlines():
        line = line.lstrip("!").strip()
        if not line:
            continue

        # Group definitions
        m = re.search(r"Group#(\d+)\s*Hist\(t0\):\s*(.+)", line)
        if m:
            gnum = int(m.group(1))
            entries = re.findall(r"(\d+)\((\d+)\)", m.group(2))
            groups[gnum] = [(int(h), int(d)) for h, d in entries]
            continue

        # Forward / Backward / Alpha
        m = re.search(
            r"Forward Group\s*=\s*(\d+),\s*Backward Group\s*=\s*(\d+),\s*Alpha\s*=\s*([\d.]+)",
            line,
        )
        if m:
            result["forward_group"] = int(m.group(1))
            result["backward_group"] = int(m.group(2))
            result["alpha"] = float(m.group(3))
            continue

        # Good bins
        m = re.search(
            r"Offset to first good bin\s*=\s*(\d+),\s*Last good bin\s*=\s*(\d+)",
            line,
        )
        if m:
            result["first_good_bin"] = int(m.group(1))
            result["last_good_bin"] = int(m.group(2))
            continue

        # Bunching
        m = re.search(r"bunching factor\s*=\s*(\d+)", line)
        if m:
            bunching_factor = int(m.group(1))
            result["bunching_factor"] = bunching_factor
            # Preserve the file-provided bunching baseline so the GUI can
            # allow only integer multiples without losing the original value.
            result["source_bunching_factor"] = bunching_factor
        if "Fixed binning" in line:
            result["fixed_binning"] = True

        # Dead-time
        if "Deadtime" in line:
            result["deadtime_correction"] = " on" in line.lower() or line.lower().endswith("on")

        # Count rates
        if "Count rates" in line:
            rest = line.split(":", 1)[1] if ":" in line else ""
            rates = re.findall(r"([\d.]+)", rest)
            result["count_rates"] = [float(r) for r in rates]

    result["groups"] = groups
    return result


def _parse_dataset_info(content: str) -> dict:
    section = _extract_section(
        content,
        "! START OF DATA SET INFORMATION",
        "! END OF DATA SET INFORMATION",
    )
    if not section:
        return {}

    parsed: dict[str, str] = {}
    for line in section.splitlines():
        line = line.lstrip("!").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip().lower()] = value.strip()

    return {
        "datarow_format": parsed.get("datarow", ""),
        "dataset_title": parsed.get("title", ""),
        "xlabel": parsed.get("xlabel", ""),
        "ylabel": parsed.get("ylabel", ""),
    }


def _parse_data(
    content: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker = "! END OF DATA SET INFORMATION"
    idx = content.find(marker)
    if idx == -1:
        return np.array([]), np.array([]), np.array([])

    data_text = content[idx + len(marker) :]
    rows: list[list[float]] = []
    for line in data_text.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if not rows:
        return np.array([]), np.array([]), np.array([])

    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]
