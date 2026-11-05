"""Normalise supported serialized ROOT headers before shared unit conversion."""

from __future__ import annotations

import re
from math import isfinite
from typing import Any

from asymmetry.core.io.physical_units import UnitConvention

# Supported serialized layouts: timing slots index active channels within a bank.
LEM_HEADER_BANKS = {4: (range(4), range(20, 24)), 5: (range(32),)}
LEM_HEADER_UNITS = {
    f"TLemRunHeader v{version}": {
        name: UnitConvention(unit, f"TLemRunHeader v{version}; musrfit 1.12.1 ROOT-NPP")
        for name, unit in (
            ("Time Resolution", "ns"),
            ("Sample Temperature", "K"),
            ("Sample Magnetic Field", "G"),
        )
    }
    for version in LEM_HEADER_BANKS
}


def read_directory_header(run_header) -> tuple[dict[str, str], str | None]:
    """Read a TDirectory ``RunHeader`` into header entries plus a run summary.

    New-format (2026 DAQ) files write each leaf as a TObjString whose TKey
    name and payload are both the encoded ``NNN - Label: Value -@type``
    string. The ``:`` in those names breaks ``root_file[path]`` lookup
    (uproot parses ``:`` as an object-path separator), so this walks
    ``(name, object)`` pairs instead of re-fetching by key path — and the
    names can also contain ``/`` (``N/A``, ``MeV/c``, URLs), so recursion
    is per-directory to keep each leaf name intact.
    """
    header: dict[str, str] = {}
    summary_lines: list[tuple[int, str]] = []

    def walk(directory, section: str) -> None:
        for key, obj in directory.items(recursive=False):
            name = re.sub(r";\d+$", "", str(key))
            if obj.__class__.__name__ == "ReadOnlyDirectory":
                if not section and name == "RunSummary":
                    # Free-text lines ("NNNN text", no -@type); some
                    # contain ":" and would false-parse, so they stay out
                    # of the header dict and become one provenance blob.
                    for line_key, line_obj in obj.items(recursive=False):
                        line = str(line_obj) if line_obj.classname == "TObjString" else None
                        if line is not None:
                            summary_lines.append(
                                _run_summary_line(re.sub(r";\d+$", "", str(line_key)), line)
                            )
                    continue
                walk(obj, f"{section}/{name}" if section else name)
                continue
            value = str(obj) if obj.classname == "TObjString" else None
            if value is None:
                continue
            # Only encoded keys discriminate encoded leaves; clean values
            # such as timestamps can contain colons without encoding a label.
            parsed = parse_musrroot_string(name) if section else None
            if parsed is not None:
                payload = parse_musrroot_string(value)
                header[f"{section}/{parsed[0]}"] = payload[1] if payload else parsed[1]
            else:
                header[f"{section}/{name}" if section else name] = parse_header_value(value)

    walk(run_header, "")
    if not summary_lines:
        return header, None
    summary_lines.sort(key=lambda item: item[0])
    return header, "\n".join(text for _, text in summary_lines)


def _run_summary_line(key_name: str, payload: str) -> tuple[int, str]:
    order = re.match(r"\s*(\d+)", key_name)
    # Strip the "NNNN " / "NNNN - " numeric prefix only; the remaining
    # text (including indentation) is kept verbatim.
    text = re.sub(r"^\d+(?: - | )", "", payload, count=1)
    return int(order.group(1)) if order else 1 << 30, text


def parse_musrroot_string(text: str) -> tuple[str, str] | None:
    match = re.match(r"\s*\d+\s*-\s*(.*?)\s*(?:-@\d+)?\s*$", text)
    if match is None:
        return None
    body = match.group(1)
    if ":" not in body:
        return None
    key, value = body.split(":", 1)
    return key.strip(), parse_header_value(value)


def parse_header_value(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+-@\d+\s*$", "", text).strip()
    return text


def read_lem_header(record) -> dict[str, str]:
    """Decode supported TLemRunHeader numeric calibration, not its display strings.

    LEM stores resolution in ns, field in G, temperature in K, and t0 in
    zero-based active-channel bins. Version 4 banks 00/20 share slots; v5
    retains direct numbering. Negative or nonfinite
    t0 entries supply no usable calibration; histogram axes do not supply calibration.
    """
    header = {
        f"RunInfo/{key}": str(record.member(member))
        for key, member in (
            ("Run Number", "fRunNumber"),
            ("Time Resolution", "fTimeResolution"),
            ("Sample Temperature", "fSampleTemperature"),
            ("Sample Magnetic Field", "fSampleBField"),
        )
    }
    header["RunInfo/Header Class"] = f"TLemRunHeader v{record.class_version}"
    for key, member in (
        ("Run Title", "fRunTitle"),
        ("Setup", "fLemSetup"),
        ("Run Start Time", "fStartTimeString"),
        ("Run Stop Time", "fStopTimeString"),
    ):
        header[f"RunInfo/{key}"] = str(record.member(member)).partition(":")[2].strip()
    for bank in LEM_HEADER_BANKS[record.class_version]:
        for number, t0 in zip(bank, record.member("fTimeZero")[: record.member("fNHist")]):
            prefix = f"DetectorInfo/Detector{number:03d}"
            header[f"{prefix}/Histo Number"] = str(number)
            if isfinite(t0) and t0 >= 0:
                header[f"{prefix}/Time Zero Bin"] = str(t0)
    return header
