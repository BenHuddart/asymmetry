"""Which numbers in a piece of prose were printed by a command.

The agent skill's number rule is that every number in a summary must appear in
a command's output. Every command's printed output is appended to
``<workdir>/cli-output.log`` (see :func:`asymmetry.cli.main`), and
:func:`unverified_numbers` holds a draft against it.

A match is deliberately loose — a number written with *d* decimals matches any
printed value it rounds from — so a match says only that the number appears in
some output, not that it is the right one. Numbers written as a multiple or a
significance or a whole-number percentage (``10×``, ``4.3σ``, ``32 %``) are almost always
arithmetic on printed values, so they match only when a command printed that
exact token, and a number after "a factor of" is always listed.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

#: A signed decimal or scientific number not glued to a word or a dot on its
#: left, so a run range ``9031-9051`` reads as two numbers, not a negative one.
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: Suffixes that make a number a derived multiple, significance or percentage.
_DERIVED_SUFFIX = re.compile(r"\s?(?:×|x|σ|sigma|%|percent|-fold|fold|\s?times)(?![a-zA-Z])")

#: Phrases that make the number after them a ratio ("a factor of 3").
_RATIO_PREFIX = re.compile(r"(?:factor of|times|fold)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Unverified:
    """A number in the draft that no logged output printed."""

    text: str
    line_number: int
    line: str


def _value(token: str) -> float:
    return float(token.replace("−", "-"))


def _decimals(token: str) -> int:
    mantissa = re.split(r"[eE]", token)[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def printed_values(log_text: str) -> list[float]:
    """Every number in the logged command output, sorted."""
    return sorted(_value(match.group()) for match in _NUMBER.finditer(log_text))


def unverified_numbers(draft: str, log_text: str) -> list[Unverified]:
    """The numbers in *draft* that no output in *log_text* printed (see above)."""
    values = printed_values(log_text)
    found: list[Unverified] = []
    for line_number, line in enumerate(draft.splitlines(), start=1):
        for match in _NUMBER.finditer(line):
            token = match.group()
            if _RATIO_PREFIX.search(line[: match.start()]):
                found.append(Unverified(token, line_number, line.strip()))
                continue
            suffix = _DERIVED_SUFFIX.match(line, match.end())
            # A decimal percentage ("A(0) 16.42 %") is a printed asymmetry in
            # its unit; a whole-number one ("32 %") is almost always a ratio.
            if suffix is not None and "%" in suffix.group() and "." in token:
                suffix = None
            if suffix is not None:
                if token + suffix.group() not in log_text:
                    found.append(Unverified(token + suffix.group(), line_number, line.strip()))
                continue
            value = _value(token)
            tolerance = 0.5 * 10.0 ** -_decimals(token) + 1e-12
            lo = bisect.bisect_left(values, abs(value) - tolerance)
            hi = bisect.bisect_right(values, abs(value) + tolerance)
            neg_lo = bisect.bisect_left(values, -abs(value) - tolerance)
            neg_hi = bisect.bisect_right(values, -abs(value) + tolerance)
            if lo == hi and neg_lo == neg_hi:
                found.append(Unverified(token, line_number, line.strip()))
    return found


__all__ = ["Unverified", "printed_values", "unverified_numbers"]
