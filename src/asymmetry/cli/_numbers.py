"""Which numbers in a piece of prose were printed by a command.

The agent skill's number rule is that every number in a summary must appear in
a command's output. Every command's printed output is appended to
``<workdir>/cli-output.log`` (see :func:`asymmetry.cli.main`), and
:func:`unverified_numbers` holds a draft against it.

A match is deliberately loose — a number written with *d* decimals matches any
printed value it rounds from — so a match says only that the number appears in
some output, not that it is the right one. Numbers written as a multiple or a
significance or a whole-number or ``±`` percentage (``10×``, ``4.3σ``, ``32 %``,
``±0.6 %``), or named as a difference (``ΔAICc 11``), match only when a command
printed that exact token. A number after "a factor of" or a difference phrase
("agree to about 0.02 MHz", "falls by 0.03 MHz"), or named as a comparison ("4
points better", "a 1.2–1.3 % spread"), is always listed. A range ``a–b`` takes
the comparison after ``b`` for both ends.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

#: A signed decimal or scientific number not glued to a word or a dot on its
#: left, so a run range ``9031-9051`` reads as two numbers, not a negative one.
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: The second end of a range ``a–b`` (or ``a to b``) after a number.
_RANGE_END = re.compile(r"\s?(?:[–-]|to)\s?[-+−]?\d+(?:\.\d+)?")

#: Suffixes that make a number a derived multiple, significance or percentage.
_DERIVED_SUFFIX = re.compile(
    r"\s?(?:×|x|σ|sigma|%|percent|-fold|fold|\s?times"
    r"|\s?(?:standard|combined) errors?|\s?error bars?)(?![a-zA-Z])"
)

_HEDGE = r"(?:about|roughly|approximately|around|some|only|up to|~|≈)?"

#: Verbs whose "by N" is a change, not a time ("falls by 0.03" against
#: "disappears by 6 K").
_CHANGE_VERB = (
    r"(?:differ|fall|fell|rise|rose|drop|shift|move|change|var(?:y|ie)|increase|decrease"
    r"|grow|grew|exceed|prefer|depart|deviate|disagree|offset|apart|separat|split)"
)

#: The rest of one clause: no punctuation and no second verb joined by "and".
_CLAUSE = r"(?:(?!\band\b)[^.,;:])*?"

#: Phrases that make the number after them a ratio ("a factor of ~3") or a
#: difference ("within about 2 G", "falls by 0.03", "agree with the survey to
#: about 0.02 MHz"), hedged or not.
_DERIVED_PREFIX = re.compile(
    rf"(?:factor of|times|fold|within|\b{_CHANGE_VERB}\w*\b{_CLAUSE},?\s*\bby"
    rf"|agree\w*\b{_CLAUSE}\bto)\s*{_HEDGE}\s*$",
    re.IGNORECASE,
)

#: A quantity named as a difference (``ΔAICc``, ``Δf =``): arithmetic unless printed.
_DELTA_PREFIX = re.compile(rf"Δ[A-Za-zχν]\S*\s*(?:=|:|of|is)?\s*{_HEDGE}\s*$")

#: A ``±`` with no value before it: what follows is a relative error (``±0.6 %``).
_RELATIVE_ERROR = re.compile(r"(?:^|[^\d\s])\s*±\s*$")

#: Words after a number (and its unit) that name it a comparison of two values.
_COMPARISON_SUFFIX = re.compile(
    r"\s?(?:%|[^\W\d][^\s,;.]*)?\s(?:better|worse|lower|higher|larger|smaller|faster"
    r"|slower|spread|difference|discrepancy|mismatch|disagreement|scatter)\b"
    r"(?!\s+(?:bound|limit|edge))",
    re.IGNORECASE,
)


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


#: A JSON array of ten or more numbers — a time axis, a histogram, a spectrum
#: dumped by ``--json``. Its values were never read by anyone, and a rounded
#: sum or ratio would match one of its thousands of elements by chance.
_NUMBER_ARRAY = re.compile(r"\[(?:\s*[-+\deE.]+\s*,){9,}\s*[-+\deE.]+\s*\]")


def printed_values(log_text: str) -> list[float]:
    """Every number in the logged command output, sorted, bulk arrays left out."""
    readable = _NUMBER_ARRAY.sub("[]", log_text)
    return sorted(_value(match.group()) for match in _NUMBER.finditer(readable))


def unverified_numbers(draft: str, log_text: str) -> list[Unverified]:
    """The numbers in *draft* that no output in *log_text* printed (see above)."""
    values = printed_values(log_text)
    found: list[Unverified] = []
    for line_number, line in enumerate(draft.splitlines(), start=1):
        for match in _NUMBER.finditer(line):
            token = match.group()
            before = line[: match.start()]
            range_end = _RANGE_END.match(line, match.end())
            after = match.end() if range_end is None else range_end.end()
            suffix = _DERIVED_SUFFIX.match(line, match.end())
            if _DERIVED_PREFIX.search(before) or _COMPARISON_SUFFIX.match(line, after):
                text = token if suffix is None else token + suffix.group()
                found.append(Unverified(text, line_number, line.strip()))
                continue
            if _DELTA_PREFIX.search(before):
                printed = re.escape(token.replace("−", "-"))
                if re.search(rf"(?<![\w.-]){printed}(?!\.?\d)", log_text) is None:
                    found.append(Unverified(token, line_number, line.strip()))
                continue
            # A decimal percentage ("A(0) 16.42 %") is a printed asymmetry in
            # its unit; a whole-number or ± one ("32 %", "±0.6 %") is a ratio.
            if (
                suffix is not None
                and "%" in suffix.group()
                and "." in token
                and _RELATIVE_ERROR.search(before) is None
            ):
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


#: What a summary says when it leans on a trend law, by the law's component.
LAW_VOCABULARY: dict[str, tuple[str, ...]] = {
    "CriticalDivergence": ("critical slowing", "critical divergence", "critical fluctuation"),
    "Arrhenius": ("activation energy", "thermally activated", "arrhenius"),
    "Redfield": ("correlation time", "redfield"),
    "OrderParameter": ("critical exponent",),
}

_FIT_HEADER = re.compile(r"^Fit of (?P<expression>.+?) to ", re.MULTILINE)


def unsupported_laws(draft: str, log_text: str) -> list[tuple[str, str]]:
    """``(law, phrase)`` for vocabulary of a law no logged fit established.

    A law counts as established when at least one logged ``trend --model`` fit
    of an expression containing it did not print ``LAW NOT ESTABLISHED``; a law
    never fitted at all is not judged here.
    """
    verdicts: dict[str, bool] = {}
    for block in re.split(r"(?=^Fit of )", log_text, flags=re.MULTILINE):
        header = _FIT_HEADER.match(block)
        if header is None:
            continue
        established = "LAW NOT ESTABLISHED" not in block.split("\n$ asymmetry", 1)[0]
        for law in LAW_VOCABULARY:
            if law in header.group("expression"):
                verdicts[law] = verdicts.get(law, False) or established
    lowered = draft.lower()
    return [
        (law, phrase)
        for law, established in verdicts.items()
        if not established
        for phrase in LAW_VOCABULARY[law]
        if phrase in lowered
    ]


__all__ = [
    "LAW_VOCABULARY",
    "Unverified",
    "printed_values",
    "unsupported_laws",
    "unverified_numbers",
]
