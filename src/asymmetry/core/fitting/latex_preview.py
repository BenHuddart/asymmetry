"""Mathtext rendering of fit-component formula templates.

This module turns a component's ASCII ``formula_template`` (with its ``{param}``
placeholders already substituted by mathtext parameter *symbols*) into a
matplotlib-mathtext fragment, or reports that the template is too complex to
transform confidently so the caller can fall back to a readable function-name
form.

It is deliberately GUI-free: no matplotlib, no Qt. Mathtext-safety is validated
by the core test suite (which *may* import matplotlib) rather than here.

The transformable subset is intentionally conservative. A template is
transformed only when it parses cleanly as arithmetic over:

* numbers, the independent variable (``t`` / ``nu``), and ``pi``,
* the parameter symbols already substituted into it,
* binary ``+ - * /`` and ``^``/``**`` powers,
* a small whitelist of unary functions (``exp``, ``sqrt``, ``abs``, ``cos``,
  ``sin``, ``tan``, ``ln``, ``erfc``),
* parentheses.

Anything else (matrix/Re[...] forms, ``;``-separated argument lists, opaque
kernel names such as ``TFmuonium`` / ``Dz_pair`` / ``rho_s``, stray ``=`` or
``<...>`` powder-average brackets) makes the parse bail out, and the caller
renders the function-name fallback instead. That fallback is the *designed*
output for the gnarly components, not a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LatexTerm:
    """One top-level additive term of a model's typeset (mathtext) preview.

    ``latex`` is a matplotlib-mathtext-safe fragment for the term with no
    leading operator. ``separator`` is ``""`` for the first term and ``" + "``
    / ``" - "`` for subsequent terms, so ``"".join(sep + latex)`` over a term
    list reconstructs the full expression. ``group`` is the ``(start, end)``
    fraction group this term belongs to / contains when exactly one group's
    component range intersects the term, else ``None`` (the GUI maps ``group``
    to an accent colour).
    """

    latex: str
    separator: str
    group: tuple[int, int] | None


#: Unary functions we can render. ``exp``/``sqrt``/``abs`` get special output;
#: the rest map to a mathtext control word applied to a parenthesised argument.
_SPECIAL_FUNCS = frozenset({"exp", "sqrt", "abs"})
_FUNC_COMMANDS: dict[str, str] = {
    "cos": r"\cos",
    "sin": r"\sin",
    "tan": r"\tan",
    "ln": r"\ln",
    "erfc": r"\mathrm{erfc}",
}
_TRANSFORMABLE_FUNCS = _SPECIAL_FUNCS | set(_FUNC_COMMANDS)

#: Independent variables render literally; ``pi`` becomes ``\pi``. ``t``/``nu``
#: are the time/frequency-domain variables; ``x``/``T`` are the parameter-vs-x
#: trend variables (field/temperature/angle abscissa).
_LITERAL_IDENTIFIERS = {
    "t": "t",
    "nu": r"\nu",
    "pi": r"\pi",
    "x": "x",
    "T": "T",
}


class _BailError(Exception):
    """Raised internally when a template is outside the transformable subset."""


# Tokeniser --------------------------------------------------------------------

# A "symbol" token is a mathtext parameter fragment we substituted in, which may
# contain backslashes, braces, carets and underscores (e.g. ``\lambda`` or
# ``A_{bg}``). We keep it as an opaque atom. Bare ASCII identifiers (function
# names, ``t``, ``pi``) are separate so we can classify them.
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+\.\d+|\.\d+|\d+)
  | (?P<pow>\*\*)
  | (?P<ident>[A-Za-z][A-Za-z0-9]*)
  | (?P<op>[+\-*/^(),|])
    """,
    re.VERBOSE,
)


def _tokenize(body: str) -> list[tuple[str, str]]:
    """Return ``(kind, text)`` tokens, or raise :class:`_BailError`.

    ``symbol`` atoms (substituted parameter mathtext) are recognised greedily
    *before* tokenising the arithmetic, so they are passed in already isolated
    via ``symbol_spans``. Here we only see the arithmetic scaffold plus bare
    identifiers; any character the regex cannot classify bails the transform.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(body)
    while pos < n:
        match = _TOKEN_RE.match(body, pos)
        if match is None or match.start() != pos:
            raise _BailError(f"unrecognised character at {pos!r}")
        pos = match.end()
        kind = match.lastgroup
        text = match.group()
        if kind == "ws":
            continue
        tokens.append((kind or "", text))
    return tokens


# Recursive-descent parser over the arithmetic scaffold -----------------------
#
# Grammar (lowest to highest precedence):
#   expr   := term (('+'|'-') term)*
#   term   := factor (('*'|'/') factor)*
#   factor := power
#   power  := unary ('^' unary)?            # right-assoc single exponent
#   unary  := ('-'|'+') unary | atom
#   atom   := number | symbol | literal-ident | func '(' expr ')'
#           | '(' expr ')' | '|' expr '|'


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], symbols: dict[str, str]) -> None:
        self._tokens = tokens
        self._symbols = symbols
        self._i = 0

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _next(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise _BailError("unexpected end of template")
        self._i += 1
        return token

    def _expect(self, text: str) -> None:
        token = self._next()
        if token[1] != text:
            raise _BailError(f"expected {text!r}, got {token[1]!r}")

    def parse(self) -> str:
        result = self._expr()
        if self._peek() is not None:
            raise _BailError("trailing tokens")
        return result

    def _expr(self) -> str:
        result = self._term()
        while True:
            token = self._peek()
            if token is not None and token[0] == "op" and token[1] in "+-":
                op = self._next()[1]
                rhs = self._term()
                result = f"{result} {op} {rhs}"
            else:
                return result

    def _term(self) -> str:
        result = self._power()
        while True:
            token = self._peek()
            if token is not None and token[0] == "op" and token[1] in "*/":
                op = self._next()[1]
                rhs = self._power()
                if op == "*":
                    # Thin space between multiplicands, matching mathtext idiom.
                    result = f"{result}\\,{rhs}"
                else:
                    result = self._make_fraction(result, rhs)
            else:
                return result

    def _make_fraction(self, lhs: str, rhs: str) -> str:
        # Only build \frac when both sides are "simple" (no top-level additive
        # operator), else keep an explicit ``/`` so we never misgroup.
        if _is_simple_operand(lhs) and _is_simple_operand(rhs):
            return rf"\frac{{{_unwrap(lhs)}}}{{{_unwrap(rhs)}}}"
        return f"{lhs}/{rhs}"

    def _power(self) -> str:
        base = self._unary()
        token = self._peek()
        if token is not None and ((token[0] == "op" and token[1] == "^") or token[0] == "pow"):
            self._next()
            exponent = self._unary()
            return f"{base}^{{{_unwrap(exponent)}}}"
        return base

    def _unary(self) -> str:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] in "+-":
            sign = self._next()[1]
            operand = self._unary()
            return f"{sign}{operand}" if sign == "-" else operand
        return self._atom()

    def _atom(self) -> str:
        token = self._next()
        kind, text = token
        if kind == "number":
            return text
        if kind == "symbol":
            return self._symbols[text]
        if kind == "op" and text == "(":
            inner = self._expr()
            self._expect(")")
            return f"({inner})"
        if kind == "op" and text == "|":
            inner = self._expr()
            self._expect("|")
            return rf"\left|{_unwrap(inner)}\right|"
        if kind == "ident":
            return self._identifier(text)
        raise _BailError(f"unexpected token {text!r}")

    def _identifier(self, name: str) -> str:
        if name in _LITERAL_IDENTIFIERS:
            return _LITERAL_IDENTIFIERS[name]
        if name in _TRANSFORMABLE_FUNCS:
            self._expect("(")
            arg = self._expr()
            self._expect(")")
            return self._apply_function(name, arg)
        raise _BailError(f"unknown identifier {name!r}")

    def _apply_function(self, name: str, arg: str) -> str:
        inner = _unwrap(arg)
        if name == "exp":
            return f"e^{{{inner}}}"
        if name == "sqrt":
            return rf"\sqrt{{{inner}}}"
        if name == "abs":
            return rf"\left|{inner}\right|"
        return rf"{_FUNC_COMMANDS[name]}\left({inner}\right)"


# Simple-operand + unwrap helpers ---------------------------------------------

_ADDITIVE_AT_TOP_RE = re.compile(r"[^{(]\s[+\-]\s")


def _is_simple_operand(fragment: str) -> bool:
    """True when ``fragment`` has no *top-level* additive operator.

    A top-level ``+``/``-`` is rendered by ``_expr`` as `` + `` / `` - `` with
    surrounding spaces, so a simple whitespace-flanked operator search is enough
    to detect an additive sum that would misgroup inside a ``\\frac``.
    """
    return _ADDITIVE_AT_TOP_RE.search(fragment) is None


def _unwrap(fragment: str) -> str:
    """Drop one redundant outer ``(...)`` wrapper if the whole fragment is one."""
    if len(fragment) >= 2 and fragment[0] == "(" and fragment[-1] == ")":
        depth = 0
        for i, char in enumerate(fragment):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return fragment[1:-1] if i == len(fragment) - 1 else fragment
    return fragment


# Parameter-symbol + fallback helpers ------------------------------------------

# A bare, mathtext-safe parameter name (letters, digits, single underscores).
_SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$")


def param_symbol_latex(latex: str, name: str) -> str:
    """Return a mathtext symbol (no surrounding ``$``) for a parameter.

    ``latex`` is the ``ParamInfo.latex`` string, typically ``$...$``-wrapped
    mathtext (e.g. ``$\\lambda$`` or ``$f_{\\mathrm{Gaussian}}$``). We strip the
    ``$`` delimiters. When the latex is empty or is not ``$``-wrapped mathtext
    (a free-text quantity), we synthesise a safe ``\\mathrm{...}`` fallback from
    the parameter ``name`` — mathtext chokes on ``\\_`` inside ``\\mathrm``, so
    underscores are rendered as subscripts instead.
    """
    stripped = latex.strip()
    if len(stripped) >= 2 and stripped.startswith("$") and stripped.endswith("$"):
        inner = stripped[1:-1].strip()
        # A doubled backslash (a mis-escaped ``r"$\\nu$"`` in a registry entry)
        # is not a valid mathtext control sequence and raises at draw time —
        # fall back to a safe name-derived symbol rather than propagate it.
        if inner and "\\\\" not in inner:
            return inner
    return _mathrm_from_name(name)


def _mathrm_from_name(name: str) -> str:
    """Render a raw parameter name as a mathtext ``\\mathrm`` symbol.

    Underscores become subscripts (``A_bg`` -> ``\\mathrm{A}_{\\mathrm{bg}}``)
    because ``\\mathrm{A\\_bg}`` is invalid mathtext. Non-safe names are reduced
    to their alphanumeric run so the result always parses.
    """
    if _SAFE_NAME_RE.match(name):
        head, *rest = name.split("_")
        symbol = rf"\mathrm{{{head}}}"
        for part in rest:
            symbol += rf"_{{\mathrm{{{part}}}}}"
        return symbol
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name) or "x"
    return rf"\mathrm{{{cleaned}}}"


def wrap_if_compound(fragment: str) -> str:
    """Parenthesise ``fragment`` unless it is already a single atom/group.

    Used before squaring a quadrature operand: ``a\\,b`` -> ``(a\\,b)`` so the
    ``^{2}`` binds the whole factor, while an already-parenthesised or single-
    symbol fragment is left as-is.
    """
    stripped = fragment.strip()
    if not stripped:
        return "()"
    if _unwrap(stripped) != stripped:
        # It is a fully wrapped ``(...)`` already.
        return stripped
    # A bare symbol (no operators/spaces) needs no parentheses.
    if re.fullmatch(r"[A-Za-z0-9\\{}_^]+", stripped):
        return stripped
    return f"({stripped})"


def fallback_function_latex(display_name: str, param_symbols: list[str]) -> str:
    """Return the ``\\mathrm{Name}(t; sym, ...)`` function-name fallback fragment.

    Used for components whose ``formula_template`` is too complex to transform
    (muonium, F-mu-F, SC lineshapes, Kubo-Toyabe kernels). This is a designed,
    readable output — never broken mathtext. ``display_name`` is the component
    name; ``param_symbols`` are its already-rendered mathtext parameter symbols
    (amplitude excluded by the caller when suppressed/shared).
    """
    safe = re.sub(r"[^A-Za-z0-9]", "", display_name) or "f"
    head = rf"\mathrm{{{safe}}}(t"
    if param_symbols:
        head += r";\," + r",\,".join(param_symbols)
    return head + ")"


# Public entry point -----------------------------------------------------------


def transform_template(template_body: str, symbols: dict[str, str]) -> str | None:
    """Return mathtext for one component body, or ``None`` if not transformable.

    ``template_body`` is a component ``formula_template`` with every ``{param}``
    placeholder already replaced by a unique sentinel token drawn from
    ``symbols`` (mapping sentinel -> mathtext parameter symbol). Substituting a
    sentinel — rather than the raw mathtext, which contains ``\\``/``{}``/``^``
    that would confuse the arithmetic tokeniser — keeps the parser's alphabet
    clean.

    Returns the mathtext fragment on success, or ``None`` when the template
    falls outside the transformable subset (the caller renders a function-name
    fallback for that term).
    """
    try:
        tokens = _tokenize_with_symbols(template_body, symbols)
        return _Parser(tokens, symbols).parse()
    except _BailError:
        return None


def _tokenize_with_symbols(body: str, symbols: dict[str, str]) -> list[tuple[str, str]]:
    """Tokenise ``body``, recognising the substituted symbol sentinels as atoms.

    Sentinels look like ``\x00<n>\x00`` so they never clash with the arithmetic
    alphabet. Everything between sentinels is tokenised as arithmetic scaffold.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(body)
    sentinel_re = re.compile(r"\x00(\d+)\x00")
    while pos < n:
        match = sentinel_re.match(body, pos)
        if match is not None:
            key = match.group(0)
            if key not in symbols:
                raise _BailError("unknown symbol sentinel")
            tokens.append(("symbol", key))
            pos = match.end()
            continue
        # Consume a run of non-sentinel text up to the next sentinel.
        next_sentinel = sentinel_re.search(body, pos)
        end = next_sentinel.start() if next_sentinel else n
        chunk = body[pos:end]
        tokens.extend(_tokenize(chunk))
        pos = end
    return tokens
