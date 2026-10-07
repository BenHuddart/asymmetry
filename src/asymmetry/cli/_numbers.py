"""Which numbers in a piece of prose were printed by a command.

The agent skill's number rule is that every number in a summary must appear in
a command's output. Every command's printed output is appended to
``<workdir>/cli-output.log`` (see :func:`asymmetry.cli.main`), and
:func:`unverified_numbers` holds a draft against it.

A match is deliberately loose — a number written with *d* decimals matches any
printed value it rounds from — so a match says only that the number appears in
some output, not that it is the right one. Numbers written as a multiple or a
significance or a whole-number percentage (``10×``, ``4.3σ``, ``32 %``), or named
as a difference (``ΔAICc 11``), match only when a command printed that exact
token (``×``, ``x`` and "times" alike). A number whose context makes it
arithmetic — after "a factor of" or a difference phrase ("agree to about 0.02
MHz", "falls by 0.03 MHz", "a margin of 3.6"), named as a comparison ("4 points
better", "a 1.2–1.3 % spread"), or a spread after a bare ``±`` — is listed unless
a command printed it verbatim to three or more significant digits (a run number,
a field on the scan's grid). A range ``a–b`` shares its context between both
ends, and ``4,200`` and ``3.2 × 10⁻⁸`` read as one number.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

#: A signed decimal or scientific number not glued to a word or a dot on its
#: left, so a run range ``9031-9051`` reads as two numbers, not a negative one.
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: A number as prose writes it: also with thousands separators, or times a power of ten.
_DRAFT_NUMBER = re.compile(
    r"(?<![\w.])(?P<mantissa>[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s?[×x]\s?10\^?(?P<exponent>[-−⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
)
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−", "0123456789--")

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
    r"|grow|grew|exceed|prefer|depart|deviate|disagree|offset|apart|separat|split|beat"
    r"|outperform|improv|lower|rais|reduc|narrow|broaden|widen)"
)

#: The rest of one clause: no punctuation but a decimal point, no second verb
#: joined by "and", and no "to" ("falls to 0.2 by 50 K" says when, not how much).
_CLAUSE = r"(?:(?!\b(?:and|to)\b)(?:[^.,;:]|\.(?=\d)))*?"

#: Phrases that make the number after them a ratio ("a factor of ~3") or a
#: difference ("within about 2 G", "falls by 0.03", "agree with the survey to
#: about 0.02 MHz", "a margin of 3.6"), hedged or not.
_DERIVED_PREFIX = re.compile(
    rf"(?:factor of|times|fold|within|\b{_CHANGE_VERB}\w*\b{_CLAUSE},?\s*\bby"
    rf"|agree\w*\b{_CLAUSE}\bto"
    rf"|\b(?:margin|difference|gap|shift|drop|rise|increase|decrease|change|offset"
    rf"|discrepancy|spread|scatter|deviation)s?\s+of)\s*{_HEDGE}\s*$",
    re.IGNORECASE,
)

#: A ratio written in words ("a factor of six", "tenfold").
_RATIO_WORD = re.compile(
    rf"\bfactor of\s*{_HEDGE}\s*(?:two|three|four|five|six|seven|eight|nine|ten|twenty"
    rf"|fifty|a hundred|hundred|a thousand|thousand)\b"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)-?fold\b",
    re.IGNORECASE,
)

#: A quantity named as a difference (``ΔAICc``, ``Δf =``): arithmetic unless printed.
_DELTA_PREFIX = re.compile(rf"Δ[A-Za-zχν]\S*\s*(?:=|:|of|is)?\s*{_HEDGE}\s*$")

#: A ``±`` after a value, its unit and a table's cell border (``| 85.95 K | ±``):
#: a ``±`` without one is a spread or a relative error (``±0.2 MHz``, ``±0.6 %``),
#: not a printed value's error.
_VALUE_PLUS_MINUS = re.compile(r"\d(?:\s*[^\W\d][^\s|±]*)?[\s|]*±\s*$")

#: Words after a number (and its unit) that name it a comparison of two values.
_COMPARISON_SUFFIX = re.compile(
    r"\s?(?:%|[^\W\d][^\s,;.]*)?\s(?:better|worse|lower|higher|larger|smaller|faster"
    r"|slower|spread|difference|discrepancy|mismatch|disagreement|scatter|behind|ahead)\b"
    r"(?!\s+(?:bound|limit|edge))",
    re.IGNORECASE,
)

#: Significant digits a printed token needs to verify a number its context makes
#: arithmetic: a run number or a grid field, not a small value printed by chance.
_VERBATIM_DIGITS = 3


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


def _multiples(text: str) -> str:
    """*text* with every multiple sign written as ``x`` (``3.3×``, ``3.3 times``)."""
    return re.sub(r"\s?(?:×|times\b)", "x", text)


#: A JSON array of ten or more numbers — a time axis, a histogram, a spectrum
#: dumped by ``--json``. Its values were never read by anyone, and a rounded
#: sum or ratio would match one of its thousands of elements by chance.
_NUMBER_ARRAY = re.compile(r"\[(?:\s*[-+\deE.]+\s*,){9,}\s*[-+\deE.]+\s*\]")


def printed_values(log_text: str) -> list[float]:
    """Every number in the logged command output, sorted, bulk arrays left out."""
    readable = _NUMBER_ARRAY.sub("[]", log_text)
    return sorted(_value(match.group()) for match in _NUMBER.finditer(readable))


def _printed_verbatim(token: str, log_text: str) -> bool:
    return re.search(rf"(?<![\w.-]){re.escape(token)}(?!\.?\d)", log_text) is not None


def unverified_numbers(draft: str, log_text: str) -> list[Unverified]:
    """The numbers in *draft* that no output in *log_text* printed (see above)."""
    values = printed_values(log_text)
    multiples = _multiples(log_text)
    found: list[Unverified] = []
    for line_number, line in enumerate(draft.splitlines(), start=1):
        found.extend(
            Unverified(match.group(), line_number, line.strip())
            for match in _RATIO_WORD.finditer(line)
        )
        shared_until = -1
        for match in _DRAFT_NUMBER.finditer(line):
            token = match.group("mantissa").replace(",", "").replace("−", "-")
            before = line[: match.start()]
            range_end = _RANGE_END.match(line, match.end())
            after = match.end() if range_end is None else range_end.end()
            suffix = _DERIVED_SUFFIX.match(line, match.end())
            text = match.group() if suffix is None else match.group() + suffix.group()
            derived = (
                match.end() <= shared_until
                or _DERIVED_PREFIX.search(before) is not None
                or _COMPARISON_SUFFIX.match(line, after) is not None
                or (before.rstrip().endswith("±") and not _VALUE_PLUS_MINUS.search(before))
            )
            if derived:
                shared_until = after
                significant = len(re.sub(r"\D", "", token).lstrip("0"))
                if significant < _VERBATIM_DIGITS or not _printed_verbatim(token, log_text):
                    found.append(Unverified(text, line_number, line.strip()))
                continue
            if _DELTA_PREFIX.search(before):
                if not _printed_verbatim(token, log_text):
                    found.append(Unverified(match.group(), line_number, line.strip()))
                continue
            # A decimal percentage ("A(0) 16.42 %") is a printed asymmetry in
            # its unit; a whole-number one ("32 %") is almost always a ratio.
            if suffix is not None and "%" in suffix.group() and "." in token:
                suffix = None
            if suffix is not None:
                if _multiples(token + suffix.group()) not in multiples:
                    found.append(Unverified(text, line_number, line.strip()))
                continue
            exponent = match.group("exponent")
            scale = 10.0 ** int(exponent.translate(_SUPERSCRIPT)) if exponent else 1.0
            value = _value(token) * scale
            tolerance = 0.5 * 10.0 ** -_decimals(token) * scale + 1e-12 * scale
            lo = bisect.bisect_left(values, abs(value) - tolerance)
            hi = bisect.bisect_right(values, abs(value) + tolerance)
            neg_lo = bisect.bisect_left(values, -abs(value) - tolerance)
            neg_hi = bisect.bisect_right(values, -abs(value) + tolerance)
            if lo == hi and neg_lo == neg_hi:
                found.append(Unverified(match.group(), line_number, line.strip()))
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
