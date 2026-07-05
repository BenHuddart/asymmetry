"""Author user fit functions from a math formula (or an advanced Python body).

This module is the core machinery behind the GUI function builder: it turns a
plain-language *draft* — a name, a description, a list of parameters, and a
Python math expression in ``x`` and the parameter names — into

* a live callable for the preview curve (:func:`build_draft_callable`,
  :func:`evaluate_draft`),
* a full pre-flight validation that never mutates a registry
  (:func:`validate_draft`), and
* the complete text of a readable, hand-editable plugin file
  (:func:`generate_plugin_source`),

and finally writes that file into the user-functions directory and registers
it through the ordinary plugin machinery (:func:`create_user_function`).

Design intent: a function created here is indistinguishable from one a user
hand-wrote and dropped in ``~/.asymmetry/user_functions/``. The generated file
is a cleaner cousin of ``docs/reference/examples/keren_user_function.py`` —
``import numpy as np``, ``import asymmetry``, a documented function, and a
single ``asymmetry.register_component`` / ``register_parameter_component``
call. It loads at startup like any other plugin and can be freely edited
afterwards, so nothing here is a private on-disk format.

Every error surfaced from this module is user-facing: it appears verbatim in
the builder dialog, so messages name the offending thing and say how to fix
it. Validation reuses the private helpers of
:mod:`asymmetry.core.fitting.user_functions` so the builder enforces exactly
the same rules as a hand-written registration — there is no second, looser
validation path.

This is a core-layer module: it must not import Qt, matplotlib, or
``asymmetry.gui``.
"""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asymmetry.core.fitting.domain_library import DOMAINS
from asymmetry.core.fitting.user_functions import (
    _PROBE_GRIDS,
    UserFunctionError,
    _check_name,
    _check_params,
    _probe_function,
)

__all__ = [
    "MATH_NAMES",
    "CreatedUserFunction",
    "DraftParameter",
    "UserFunctionDraft",
    "build_draft_callable",
    "create_user_function",
    "detect_parameter_names",
    "evaluate_draft",
    "generate_function_body",
    "generate_plugin_source",
    "validate_draft",
]


#: Bare math names a formula may use, mapped to their numpy attribute. The
#: builder rewrites each to ``np.<attr>`` when it generates the plugin file, so
#: a formula reads like maths but the emitted code is unambiguous. ``pi`` and
#: ``e`` are numpy constants; the rest are ufuncs vectorised over ``x``.
MATH_NAMES: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arctan2": "arctan2",
    "pi": "pi",
    "e": "e",
}

#: Python builtins allowed bare in a formula (they are not imported — they are
#: already in scope when the generated function runs).
_ALLOWED_BUILTINS: frozenset[str] = frozenset({"abs", "min", "max"})

#: Names that are always in scope inside a formula and so are never treated as
#: user parameters: the independent variable, the numpy alias, the bare math
#: names, and the allowed builtins.
_RESERVED_FORMULA_NAMES: frozenset[str] = (
    frozenset({"x", "np"}) | frozenset(MATH_NAMES) | _ALLOWED_BUILTINS
)


@dataclass
class DraftParameter:
    """One fit parameter of a draft: its name and start (default) value."""

    name: str
    value: float


@dataclass
class UserFunctionDraft:
    """A user's in-progress function definition, before validation/registration.

    ``formula`` is a Python math expression in ``x`` and the parameter names
    (e.g. ``"A*exp(-(x/tau)**alpha)"``). When ``advanced_body`` is set it is a
    complete multi-line Python function body (containing a ``return``) that
    replaces the formula-generated body verbatim; ``formula`` is then only used
    to seed the display template.
    """

    kind: str  # "component" | "parameter"
    name: str
    description: str
    formula: str
    parameters: list[DraftParameter] = field(default_factory=list)
    #: Component kind only: "time" | "frequency". Ignored for parameter kind.
    domain: str = "time"
    #: When set, a full Python function body used instead of the formula.
    advanced_body: str | None = None

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def param_defaults(self) -> dict[str, float]:
        return {p.name: float(p.value) for p in self.parameters}


@dataclass
class CreatedUserFunction:
    """The result of :func:`create_user_function`: the file and the definition."""

    path: Path
    #: The freshly registered ``ComponentDefinition`` /
    #: ``ParameterModelComponentDefinition`` (typed loosely to keep this module
    #: free of registry-shape coupling).
    definition: object


# ── formula parsing ─────────────────────────────────────────────────────────


def _parse_formula(formula: str) -> ast.Expression:
    """Parse *formula* as a Python expression, raising a user-facing error."""
    if not isinstance(formula, str) or not formula.strip():
        raise UserFunctionError("Enter a formula in x and the parameter names.")
    # Reject braces up front: no maths expression needs them (they are Python
    # dict/set syntax), and a brace that reached the display template would
    # only be rejected much later, at registration time, with a confusing
    # stray-placeholder error. Advanced bodies are unaffected — their display
    # template is the opaque ``Name(x; {p1}, …)`` form.
    if "{" in formula or "}" in formula:
        raise UserFunctionError(
            "Braces are not valid in a formula — write plain maths in x and "
            "the parameter names, e.g. A*exp(-lam*x)."
        )
    try:
        return ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise UserFunctionError(f"Formula is not valid Python: {exc.msg}.") from exc


def detect_parameter_names(formula: str) -> list[str]:
    """Return the parameter names a formula uses, in first-appearance order.

    Every bare :class:`ast.Name` that is not ``x``, ``np``, a bare math name,
    or an allowed builtin is a parameter. Powers the builder's "detect
    parameters" button. Raises :class:`UserFunctionError` on a syntax error.
    """
    return _ordered_free_names(_parse_formula(formula), skip=set())


def _ordered_free_names(tree: ast.AST, *, skip: set[str]) -> list[str]:
    """Free (non-reserved, non-*skip*) ``Name`` ids in source-appearance order.

    ``ast.walk`` is breadth-first, so it does not preserve the order names
    appear in the source. Sort by (line, column) so the builder's detected
    parameter list matches the reading order of the formula.
    """
    hits: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _RESERVED_FORMULA_NAMES:
            if node.id in skip or node.id in seen:
                continue
            seen.add(node.id)
            hits.append((node.lineno, node.col_offset, node.id))
    hits.sort()
    return [name for _line, _col, name in hits]


def _unknown_formula_names(formula: str, declared: set[str]) -> list[str]:
    """Names used bare in *formula* that are neither reserved nor declared."""
    return _ordered_free_names(_parse_formula(formula), skip=declared)


# ── source generation ───────────────────────────────────────────────────────


def _snake_case(name: str) -> str:
    """A readable snake_case function/file stem derived from a registry name.

    Registry names are already identifier atoms (``[A-Za-z_][A-Za-z0-9_]*``),
    so this only needs to split CamelCase and lower-case the result.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", spaced)
    stem = re.sub(r"__+", "_", spaced).strip("_").lower()
    return stem or "user_function"


def _rewrite_math_names(formula: str) -> str:
    """Rewrite bare math names in *formula* to their ``np.`` equivalents.

    Only whole-word bare :class:`ast.Name` nodes are rewritten, so a parameter
    or attribute access that merely *contains* a math name (``expected``,
    ``np.exp``) is left alone. Comparison is by source segment to preserve the
    author's exact spacing everywhere else.
    """
    tree = _parse_formula(formula)
    # Collect (col-offset span, replacement) edits, then apply right-to-left so
    # earlier offsets stay valid.
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in MATH_NAMES:
            segment = ast.get_source_segment(formula, node)
            if segment == node.id:  # a genuine bare use, not part of np.exp etc.
                start = node.col_offset
                end = node.end_col_offset
                edits.append((start, end, f"np.{MATH_NAMES[node.id]}"))
    rewritten = formula
    for start, end, replacement in sorted(edits, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten


_TEMPLATE_STRIP_NP_RE = re.compile(r"\bnp\.")


def _build_formula_template(draft: UserFunctionDraft) -> str:
    """The display formula rendered with ``str.format(**params)`` at fit time.

    ``ComponentDefinition.formula_template`` is fed through
    ``str.format(**params)`` (composite.py), so any literal ``{``/``}`` from
    the formula must be doubled, and each parameter name is wrapped in ``{}``
    so it is substituted with its value. ``np.`` prefixes are stripped for a
    maths-like display. For advanced mode the body is opaque, so an
    ``Name(x; {p1}, {p2}, …)`` summary is used instead.
    """
    params = draft.param_names
    if draft.advanced_body is not None:
        inner = ", ".join(f"{{{p}}}" for p in params)
        return f"{draft.name}(x; {inner})" if inner else f"{draft.name}(x)"

    text = draft.formula
    # Escape literal braces first so no author brace is mistaken for a
    # placeholder, then strip np. prefixes for display.
    text = text.replace("{", "{{").replace("}", "}}")
    text = _TEMPLATE_STRIP_NP_RE.sub("", text)
    # Wrap each declared parameter name in {} on identifier boundaries so
    # neighbouring text (functions, other params) is untouched.
    for pname in params:
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(pname)}(?![A-Za-z0-9_])", f"{{{pname}}}", text)
    return text


def _indent_body(body: str, indent: str = "    ") -> str:
    """Re-indent a multi-line function body under a ``def`` at *indent*."""
    lines = body.splitlines()
    # Drop leading/trailing blank lines so the emitted def is tidy.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    out = []
    for line in lines:
        out.append(f"{indent}{line}" if line.strip() else "")
    return "\n".join(out)


def _safe_docstring_text(text: str) -> str:
    """Neutralise text that would prematurely close a triple-quoted docstring."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _signature(draft: UserFunctionDraft) -> str:
    """The keyword-only signature ``x, *, p1=v1, p2=v2`` for the def line."""
    params = draft.parameters
    if not params:
        return "x"
    kwargs = ", ".join(f"{p.name}={float(p.value)!r}" for p in params)
    return f"x, *, {kwargs}"


def generate_function_body(draft: UserFunctionDraft) -> str:
    """Return the UN-indented formula-mode function body for *draft*.

    These are the three lines the generated function would contain in formula
    mode — ``x = np.asarray(...)``, ``result = <rewritten formula>``, and the
    broadcast ``return`` — with no leading ``def``-body indent. The builder's
    Advanced editor pre-fills its editor with this text, so an author who
    switches to advanced mode starts from the exact code the formula would have
    produced (and a draft with ``advanced_body`` set to this text validates and
    evaluates identically to the formula draft).

    The broadcast keeps a constant formula returning an array of ``x``'s shape,
    which the load-time probe requires.
    """
    expr = _rewrite_math_names(draft.formula)
    return (
        "x = np.asarray(x, dtype=float)\n"
        f"result = {expr}\n"
        "return np.broadcast_to(np.asarray(result, dtype=float), x.shape).copy()"
    )


def generate_plugin_source(draft: UserFunctionDraft) -> str:
    """Return the complete text of a readable, hand-editable plugin file.

    The file imports numpy and asymmetry, defines the function, and registers
    it with a single facade call — the same shape as a hand-written plugin, so
    it loads at startup and can be edited freely afterwards.
    """
    func_name = f"_{_snake_case(draft.name)}"
    signature = _signature(draft)
    template = _build_formula_template(draft)

    if draft.advanced_body is not None:
        body = _indent_body(draft.advanced_body)
        func_doc = f'"""{draft.name}: user-authored function (advanced body)."""'
    else:
        # Indent the shared formula body under the def; generate_function_body
        # is the un-indented source the Advanced editor pre-fills with.
        body = _indent_body(generate_function_body(draft))
        func_doc = f'"""{draft.name}: {_safe_docstring_text(draft.formula)}"""'

    if draft.kind == "component":
        register_call = _render_component_registration(draft, func_name, template)
    else:
        register_call = _render_parameter_registration(draft, func_name, template)

    module_doc = (
        f'"""{draft.name} — created by the Asymmetry function builder for you.\n'
        f"\n"
        f"{_safe_docstring_text(draft.description)}\n"
        f"\n"
        f"This file is an ordinary Asymmetry plugin: edit it freely and it will\n"
        f"reload the next time the application starts.\n"
        f'"""'
    )

    return (
        f"{module_doc}\n"
        f"\n"
        f"import numpy as np\n"
        f"\n"
        f"import asymmetry\n"
        f"\n"
        f"\n"
        f"def {func_name}({signature}):\n"
        f"    {func_doc}\n"
        f"{body}\n"
        f"\n"
        f"\n"
        f"{register_call}\n"
    )


def _py_str(text: str) -> str:
    """Render *text* as a double-quoted Python string literal (ruff's style)."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _py_list(items: list[str]) -> str:
    """Render a list of identifier strings as Python source, e.g. ``["A", "b"]``."""
    return "[" + ", ".join(_py_str(item) for item in items) + "]"


def _py_defaults(draft: UserFunctionDraft) -> str:
    """Render the ``param_defaults`` dict literal for the registration call."""
    entries = ", ".join(f"{_py_str(p.name)}: {float(p.value)!r}" for p in draft.parameters)
    return "{" + entries + "}"


def _render_component_registration(draft: UserFunctionDraft, func_name: str, template: str) -> str:
    return (
        "asymmetry.register_component(\n"
        f"    {_py_str(draft.name)},\n"
        f"    {func_name},\n"
        f"    {_py_list(draft.param_names)},\n"
        f"    domain={_py_str(draft.domain)},\n"
        f"    description={_py_str(draft.description)},\n"
        f"    formula_template={_py_str(template)},\n"
        f"    param_defaults={_py_defaults(draft)},\n"
        ")"
    )


def _render_parameter_registration(draft: UserFunctionDraft, func_name: str, template: str) -> str:
    return (
        "asymmetry.register_parameter_component(\n"
        f"    {_py_str(draft.name)},\n"
        f"    {func_name},\n"
        f"    {_py_list(draft.param_names)},\n"
        f"    description={_py_str(draft.description)},\n"
        f"    formula_template={_py_str(template)},\n"
        f"    param_defaults={_py_defaults(draft)},\n"
        '    scopes=("common",),\n'
        ")"
    )


# ── callable construction ───────────────────────────────────────────────────


def build_draft_callable(draft: UserFunctionDraft) -> Callable[..., np.ndarray]:
    """Compile the draft's function definition and return the callable.

    Only the ``def`` is executed — never the registration call — so building a
    callable for the preview curve has no registry side effects. Raises
    :class:`UserFunctionError` with an actionable message for unknown names, a
    missing ``return`` in an advanced body, or a syntax error.
    """
    declared = set(draft.param_names)

    if draft.advanced_body is not None:
        if "return" not in draft.advanced_body:
            raise UserFunctionError(
                "The advanced function body must contain a return statement that "
                "yields an array the same shape as x."
            )
    else:
        # Beat a bare NameError with a message that names the culprit and lists
        # what the user actually declared.
        unknown = _unknown_formula_names(draft.formula, declared)
        if unknown:
            declared_list = draft.param_names or ["(none)"]
            raise UserFunctionError(
                f"Formula uses unknown name(s) {unknown}. Declare them as "
                f"parameters, or check the spelling — declared parameters are "
                f"{declared_list}. You may also use x, np.<func>, and the bare "
                f"math functions {sorted(MATH_NAMES)}."
            )

    source = generate_plugin_source(draft)
    # Compile only the function definition (the first def in the generated
    # module), so exec has no registration side effect and no asymmetry import
    # requirement for the preview path.
    func_name = f"_{_snake_case(draft.name)}"
    func_def = _extract_function_def(source, func_name)

    namespace: dict[str, object] = {"np": np}
    try:
        exec(compile(func_def, filename="<user-function>", mode="exec"), namespace)
    except SyntaxError as exc:
        raise UserFunctionError(f"Function body is not valid Python: {exc.msg}.") from exc
    callable_obj = namespace.get(func_name)
    if not callable(callable_obj):
        raise UserFunctionError("The generated function could not be compiled.")
    return callable_obj  # type: ignore[return-value]


def _extract_function_def(source: str, func_name: str) -> str:
    """Return the source of the ``def func_name`` block from *source* alone."""
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                return segment
    # Should never happen for generated source; guard defensively.
    raise UserFunctionError("Could not locate the generated function definition.")


def evaluate_draft(draft: UserFunctionDraft, x: np.ndarray) -> np.ndarray:
    """Evaluate the draft at its parameter defaults on *x* (the preview curve).

    Exceptions propagate as :class:`UserFunctionError` so the builder can show
    the failure inline instead of crashing the preview.
    """
    func = build_draft_callable(draft)
    try:
        with np.errstate(all="ignore"):
            out = func(np.asarray(x, dtype=float), **draft.param_defaults)
    except Exception as exc:
        raise UserFunctionError(
            f"The function could not be evaluated ({type(exc).__name__}: {exc})."
        ) from exc
    return np.asarray(out, dtype=float)


# ── validation ──────────────────────────────────────────────────────────────


def validate_draft(draft: UserFunctionDraft) -> Callable[..., np.ndarray]:
    """Full pre-flight of a draft, mutating no registry; return the callable.

    Runs exactly the checks a real registration would — name grammar and
    global uniqueness, parameter-name rules, a non-empty description, callable
    construction, and a finite probe evaluation on the right grid — but stops
    short of inserting anything. Raises :class:`UserFunctionError` on the first
    failure.
    """
    if draft.kind not in ("component", "parameter"):
        raise UserFunctionError(
            f"Unknown function kind {draft.kind!r}: choose 'component' (a fit "
            "component) or 'parameter' (a parameter-vs-x trend)."
        )

    if draft.kind == "component":
        domain = str(draft.domain).strip().lower()
        if domain not in DOMAINS:
            raise UserFunctionError(
                f"A valid domain is required (one of {list(DOMAINS)}); got "
                f"{draft.domain!r}. The domain places the component in the "
                "matching picker and plots."
            )
        grid = _PROBE_GRIDS[domain]
    else:
        grid = _PROBE_GRIDS["parameter"]

    _check_name(draft.name)
    params, defaults = _check_params(draft.name, draft.param_names, draft.param_defaults)

    # A parameter named x/np/exp/... would silently shadow the reserved formula
    # names, so a formula could never reach the user's parameter. Reject early
    # with a message that says why.
    clashes = sorted(set(params) & _RESERVED_FORMULA_NAMES)
    if clashes:
        raise UserFunctionError(
            f"{draft.name}: parameter name(s) {clashes} clash with a reserved "
            "formula name (x, np, or a built-in math function). Rename them so "
            "the formula can tell parameters from functions."
        )

    if not isinstance(draft.description, str) or not draft.description.strip():
        raise UserFunctionError(f"{draft.name}: a non-empty description is required.")

    callable_obj = build_draft_callable(draft)
    _probe_function(draft.name, callable_obj, defaults, grid)
    return callable_obj


# ── creation (write file + register) ────────────────────────────────────────


def _unique_path(directory: Path, stem: str) -> Path:
    """A ``<stem>.py`` path in *directory*, suffixed ``_2``, ``_3``… if taken."""
    candidate = directory / f"{stem}.py"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = directory / f"{stem}_{index}.py"
        if not candidate.exists():
            return candidate
        index += 1


def create_user_function(
    draft: UserFunctionDraft, directory: str | Path | None = None
) -> CreatedUserFunction:
    """Validate, write, and register a draft as a plugin file.

    Validates the draft (no registry mutation), writes the generated source to
    a uniquely named file under *directory* (default
    :data:`asymmetry.core.plugins.USER_FUNCTIONS_DIR`, resolved at call time),
    then imports it through the plugin machinery so it registers exactly as a
    startup-discovered file would. If the load fails, the file is deleted and a
    :class:`UserFunctionError` carries the loader's error text.
    """
    validate_draft(draft)

    # Resolve the default lazily so tests can monkeypatch USER_FUNCTIONS_DIR.
    from asymmetry.core import plugins

    dir_path = Path(directory) if directory is not None else plugins.USER_FUNCTIONS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    stem = _snake_case(draft.name)
    if keyword.iskeyword(stem):
        stem = f"{stem}_fn"
    path = _unique_path(dir_path, stem)
    path.write_text(generate_plugin_source(draft), encoding="utf-8")

    source = plugins.load_plugin_file(path)
    if not source.ok:
        path.unlink(missing_ok=True)
        raise UserFunctionError(f"The function was written but failed to load: {source.error}")

    definition = _lookup_registered_definition(draft)
    return CreatedUserFunction(path=path, definition=definition)


def _lookup_registered_definition(draft: UserFunctionDraft) -> object:
    """Fetch the just-registered definition from its registry by name."""
    if draft.kind == "component":
        from asymmetry.core.fitting.composite import COMPONENTS

        return COMPONENTS[draft.name]
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    return PARAMETER_MODEL_COMPONENTS[draft.name]
