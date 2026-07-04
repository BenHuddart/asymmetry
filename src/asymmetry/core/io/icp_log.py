"""Best-effort parser for ISIS ICP instrument ``.log`` files.

Every ISIS muon run recorded through the ICP DAE control software writes a
plain-text sidecar log next to the data file (``MUSR00038241.RAW`` /
``.nxs`` / ``.nxs_v2`` alongside ``MUSR00038241.log``). Each line is a
tab-separated ``timestamp \\t channel \\t value`` sample of an instrument
"selog" channel — magnet power supplies, steering coils, temperatures, beam
current, and so on — polled throughout the run.

This module extracts the *applied field* from that log: which magnet supply
was selected (``a_selected_magnet``) and the corresponding field-magnitude
channel's last reading. Both NeXus ``sample/magnetic_field*`` fields and this
``.log`` ultimately come from the same instrument control system, but the
NeXus file sometimes omits or blanks the field (see
:mod:`asymmetry.core.io.nexus`); the ``.log`` is a second, independent read
of the same hardware state that is often present even then.

Key correctness point — **every** ``Field_*`` channel is logged continuously
regardless of which magnet is actually powering the beamline (a ZF run still
shows a non-zero ``Field_Hifi`` reading from an unrelated, unselected magnet
several metres away). Only the channel named by the currently-selected magnet
(``a_selected_magnet``) is a trustworthy reading of the *applied* field;
aggregating or maximising across channels would badly misread a ZF run as a
many-thousand-gauss run. See :data:`_MAGNET_TO_FIELD_CHANNEL`.

This module deliberately extracts **magnitude and a very narrow "zero field"
direction signal only** — never a TF/LF direction from the magnet name. A
magnet named ``Danfysik`` or ``T20 Coils`` does not by itself say whether the
field it drives is applied transverse or longitudinal to the initial muon
spin; guessing that would be exactly the forbidden orientation-from-hardware
inference the field-geometry study rejected (see
``docs/porting/field-geometry/``). The one direction claim this parser will
make is reading the literal string ``a_selected_magnet == "Active ZF"``
(or any value containing ``"ZF"``) as "zero field selected" — that is an
explicit recorded token, the same kind of signal ``field_direction_from_text``
already trusts, not a magnitude-based guess.

Parsing is defensive throughout: a malformed, truncated, or unexpected log
must never raise out of :func:`parse_icp_log_text` — a best-effort ``None``
is a fully valid outcome, exactly as an absent NeXus field would be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Maps a known ``a_selected_magnet`` value (or substring) to the channel
#: whose readings represent that magnet's field, in gauss. Names are matched
#: case-insensitively, longest/most-specific key first is not required: only
#: one magnet is ever selected at a time.
_MAGNET_TO_FIELD_CHANNEL: dict[str, str] = {
    "danfysik": "Field_Danfysik",
    "t20": "Field_T20",
    "hifi": "Field_Hifi",
    "emu": "Field_Emu",
}

#: Substring in ``a_selected_magnet`` that marks "no magnet selected" — the
#: field-magnitude channel to trust is the dedicated ZF monitor instead.
_ZF_MAGNET_TOKEN = "zf"

#: Channel carrying the residual/ambient field magnitude when ZF is selected.
_ZF_MAGNITUDE_CHANNEL = "Field_ZF_Magnitude"


@dataclass(frozen=True)
class IcpFieldReading:
    """Best-effort applied-field reading recovered from an ICP ``.log``.

    ``field_gauss`` is the last sample of the field channel corresponding to
    the selected magnet (or the ZF monitor when ZF was selected).
    ``field_direction`` is only ever ``"Zero field"`` (an explicit recorded
    token) or ``""`` (unknown) — this parser never infers TF/LF from a magnet
    name or field magnitude. ``selected_magnet`` is the raw recorded value,
    kept for provenance/debugging.
    """

    field_gauss: float | None
    field_direction: str
    selected_magnet: str


def _channel_for_magnet(selected_magnet: str) -> tuple[str, bool]:
    """Return ``(channel_name, is_zero_field)`` for a recorded magnet name."""
    lowered = selected_magnet.strip().lower()
    if _ZF_MAGNET_TOKEN in lowered:
        return _ZF_MAGNITUDE_CHANNEL, True
    for token, channel in _MAGNET_TO_FIELD_CHANNEL.items():
        if token in lowered:
            return channel, False
    return "", False


def parse_icp_log_text(text: str) -> IcpFieldReading | None:
    """Parse ICP ``.log`` content and recover the applied-field reading.

    Scans every ``timestamp \\t channel \\t value`` line, tracking the most
    recent ``a_selected_magnet`` value and the most recent reading of every
    ``Field_*`` channel. Once the whole log has been scanned, the channel
    matching the *final* selected magnet is used — an instrument log can
    record magnet changes mid-run (e.g. before/after a field-setting step),
    and the last selection is the one that applied to the bulk of the run.

    Returns ``None`` when no usable ``a_selected_magnet`` / field-channel pair
    is found (e.g. an empty file, a non-ICP log, or a magnet name this parser
    does not recognise). Never raises — any malformed line is skipped.
    """
    last_magnet: str | None = None
    field_values: dict[str, float] = {}

    for raw_line in text.splitlines():
        parts = raw_line.split("\t")
        if len(parts) < 3:
            continue
        channel = parts[1].strip()
        value_text = parts[2].strip()
        if not channel:
            continue

        if channel == "a_selected_magnet":
            if value_text:
                last_magnet = value_text
            continue

        if channel.startswith("Field_"):
            try:
                field_values[channel] = float(value_text)
            except ValueError:
                continue

    if last_magnet is None:
        return None

    channel, is_zf = _channel_for_magnet(last_magnet)
    if not channel or channel not in field_values:
        return None

    return IcpFieldReading(
        field_gauss=field_values[channel],
        field_direction="Zero field" if is_zf else "",
        selected_magnet=last_magnet,
    )


def parse_icp_log_file(path: str | Path) -> IcpFieldReading | None:
    """Read and parse an ICP ``.log`` file at *path*.

    Best-effort: a missing file, permission error, or decode failure returns
    ``None`` rather than raising, so a caller can unconditionally probe for a
    sidecar log without a separate existence check.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_icp_log_text(text)


def sibling_icp_log_path(data_file_path: str | Path) -> Path:
    """Return the conventional ``.log`` sidecar path for a data file.

    ISIS ICP writes the log beside the run file with the same stem, e.g.
    ``MUSR00038241.nxs`` -> ``MUSR00038241.log`` (also true for ``.RAW`` and
    ``.nxs_v2``). This only computes the *expected* path; callers should check
    existence (or rely on :func:`parse_icp_log_file`'s best-effort ``None``).
    """
    path = Path(data_file_path)
    return path.with_suffix(".log")
