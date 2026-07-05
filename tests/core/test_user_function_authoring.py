"""Tests for the user-function authoring machinery (the GUI builder's core).

Every test isolates the fit-function registries through the autouse
``_registry_snapshot`` fixture, so a draft registered or loaded here can never
leak into another test's view of ``COMPONENTS`` /
``PARAMETER_MODEL_COMPONENTS``. Nothing here writes to the real
``USER_FUNCTIONS_DIR`` — creation is always directed at ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.core.fitting.user_function_authoring import (
    CreatedUserFunction,
    DraftParameter,
    UserFunctionDraft,
    build_draft_callable,
    create_user_function,
    detect_parameter_names,
    evaluate_draft,
    generate_function_body,
    generate_plugin_source,
    validate_draft,
)
from asymmetry.core.fitting.user_functions import UserFunctionError


@pytest.fixture(autouse=True)
def _registry_snapshot(registry_snapshot):
    """Isolate registry mutations behind the shared conftest fixture."""
    yield


def _stretched_draft(name="UserStretched", **overrides) -> UserFunctionDraft:
    kwargs = dict(
        kind="component",
        name=name,
        description="Stretched exponential test component",
        formula="A*exp(-(x/tau)**alpha)",
        parameters=[
            DraftParameter("A", 25.0),
            DraftParameter("tau", 1.0),
            DraftParameter("alpha", 1.0),
        ],
        domain="time",
    )
    kwargs.update(overrides)
    return UserFunctionDraft(**kwargs)


def _trend_draft(name="UserTrend", **overrides) -> UserFunctionDraft:
    kwargs = dict(
        kind="parameter",
        name=name,
        description="Linear parameter trend",
        formula="a*x+b",
        parameters=[DraftParameter("a", 1.0), DraftParameter("b", 0.0)],
    )
    kwargs.update(overrides)
    return UserFunctionDraft(**kwargs)


# ── detect_parameter_names ──────────────────────────────────────────────────


def test_detect_parameter_names_first_appearance_order():
    assert detect_parameter_names("A*exp(-(x/tau)**alpha)") == ["A", "tau", "alpha"]


def test_detect_parameter_names_excludes_x_np_and_math():
    names = detect_parameter_names("np.exp(-x)*sin(w*x + phi) + pi*C")
    # x, np, exp, sin, pi are all excluded; only w, phi, C remain, in order.
    assert names == ["w", "phi", "C"]


def test_detect_parameter_names_excludes_bare_builtins():
    assert detect_parameter_names("abs(A)*max(x, B)") == ["A", "B"]


def test_detect_parameter_names_syntax_error_is_user_facing():
    with pytest.raises(UserFunctionError, match="Formula is not valid Python"):
        detect_parameter_names("A*exp(-x")


def test_detect_parameter_names_empty_formula_rejected():
    with pytest.raises(UserFunctionError):
        detect_parameter_names("   ")


# ── generate_plugin_source: formula-mode round trip ─────────────────────────


def test_generated_formula_file_round_trips_through_loader(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _stretched_draft()
    (tmp_path / "stretched.py").write_text(generate_plugin_source(draft), encoding="utf-8")
    report = load_user_functions(tmp_path)

    (source,) = report.sources
    assert source.ok, source.error
    definition = COMPONENTS["UserStretched"]
    assert definition.user is True
    assert definition.domain == "time"
    assert definition.param_names == ["A", "tau", "alpha"]
    assert definition.param_defaults == {"A": 25.0, "tau": 1.0, "alpha": 1.0}


def test_generated_formula_values_match_numpy_directly(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _stretched_draft()
    (tmp_path / "stretched.py").write_text(generate_plugin_source(draft), encoding="utf-8")
    load_user_functions(tmp_path)

    func = COMPONENTS["UserStretched"].function
    x = np.linspace(0.0, 12.0, 41)
    amp, tau, alpha = 25.0, 1.0, 1.0
    expected = amp * np.exp(-((x / tau) ** alpha))
    np.testing.assert_allclose(func(x, A=amp, tau=tau, alpha=alpha), expected)


def test_generated_constant_formula_returns_full_shape_array(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _stretched_draft(
        name="UserConst",
        formula="A",
        parameters=[DraftParameter("A", 3.0)],
    )
    (tmp_path / "const.py").write_text(generate_plugin_source(draft), encoding="utf-8")
    report = load_user_functions(tmp_path)
    assert report.sources[0].ok, report.sources[0].error

    func = COMPONENTS["UserConst"].function
    x = np.linspace(0.0, 5.0, 9)
    out = func(x, A=3.0)
    assert out.shape == x.shape
    np.testing.assert_array_equal(out, np.full_like(x, 3.0))


def test_brace_in_formula_rejected_early_with_clear_message():
    # Braces never belong in a maths formula, and letting one through would
    # only surface a confusing stray-placeholder error at registration time,
    # after the plugin file had been written. Both public entry points must
    # reject it up front.
    with pytest.raises(UserFunctionError, match="Braces are not valid in a formula"):
        detect_parameter_names("A*exp(-x) + {a: 2}")

    draft = _stretched_draft(
        name="UserBrace",
        formula="A*exp(-x) + {1, 2}",
        parameters=[DraftParameter("A", 1.0)],
    )
    with pytest.raises(UserFunctionError, match="Braces are not valid in a formula"):
        validate_draft(draft)
    assert "UserBrace" not in COMPONENTS


def test_template_builder_doubles_literal_braces_defensively():
    from asymmetry.core.fitting.user_function_authoring import _build_formula_template

    # The public path rejects braces before they can reach the template, but
    # the builder still doubles any literal brace defensively so that a
    # template rendered with str.format(**params) can never KeyError or eat a
    # brace as a placeholder.
    draft = _stretched_draft(
        name="UserBrace",
        formula="A*exp(-x)  # note {curly literal}",
        parameters=[DraftParameter("A", 1.0)],
    )
    template = _build_formula_template(draft)
    assert "{{curly literal}}" in template  # doubled
    assert "{A}" in template  # parameter still wrapped for substitution

    rendered = template.format(A="1")
    assert "{curly literal}" in rendered  # single braces after format, no KeyError


def test_generated_advanced_body_round_trips(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _stretched_draft(
        name="UserAdvanced",
        formula="A*exp(-rate*x)",  # only seeds the display template
        parameters=[DraftParameter("A", 2.0), DraftParameter("rate", 0.5)],
        advanced_body=(
            "x = np.asarray(x, dtype=float)\nenvelope = np.exp(-rate * x)\nreturn A * envelope"
        ),
    )
    (tmp_path / "advanced.py").write_text(generate_plugin_source(draft), encoding="utf-8")
    report = load_user_functions(tmp_path)
    assert report.sources[0].ok, report.sources[0].error

    func = COMPONENTS["UserAdvanced"].function
    x = np.linspace(0.0, 8.0, 21)
    np.testing.assert_allclose(func(x, A=2.0, rate=0.5), 2.0 * np.exp(-0.5 * x))
    # Advanced templates use the opaque summary form.
    assert COMPONENTS["UserAdvanced"].formula_template == "UserAdvanced(x; {A}, {rate})"


def test_generated_parameter_kind_registers_with_common_scope(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _trend_draft()
    (tmp_path / "trend.py").write_text(generate_plugin_source(draft), encoding="utf-8")
    report = load_user_functions(tmp_path)
    assert report.sources[0].ok, report.sources[0].error

    definition = PARAMETER_MODEL_COMPONENTS["UserTrend"]
    assert definition.user is True
    assert definition.scopes == ("common",)
    x = np.linspace(1.0, 10.0, 7)
    np.testing.assert_allclose(definition.function(x, a=1.0, b=0.0), x)


# ── generate_function_body ──────────────────────────────────────────────────


def test_generate_function_body_is_unindented_and_rewrites_math():
    body = generate_function_body(_stretched_draft())
    lines = body.splitlines()
    # Un-indented (the Advanced editor pre-fills raw source it then re-indents).
    assert all(line == line.lstrip() for line in lines)
    assert lines[0] == "x = np.asarray(x, dtype=float)"
    assert lines[-1].startswith("return np.broadcast_to(")
    # Bare math name rewritten to its numpy attribute.
    assert "np.exp(" in body


def test_advanced_prefill_from_generate_function_body_round_trips():
    # The Advanced editor pre-fill is the formula body verbatim, so a draft
    # whose advanced_body is that text must validate and evaluate identically
    # to the formula draft it came from.
    formula_draft = _stretched_draft()
    advanced_draft = _stretched_draft(
        name="UserAdvancedPrefill",
        advanced_body=generate_function_body(formula_draft),
    )

    validate_draft(formula_draft)
    validate_draft(advanced_draft)

    x = np.linspace(0.0, 12.0, 41)
    np.testing.assert_allclose(evaluate_draft(advanced_draft, x), evaluate_draft(formula_draft, x))


# ── build_draft_callable / evaluate_draft ───────────────────────────────────


def test_build_draft_callable_has_no_registry_side_effect():
    build_draft_callable(_stretched_draft())
    assert "UserStretched" not in COMPONENTS


def test_evaluate_draft_uses_default_values():
    x = np.linspace(0.0, 4.0, 9)
    out = evaluate_draft(_stretched_draft(), x)
    np.testing.assert_allclose(out, 25.0 * np.exp(-((x / 1.0) ** 1.0)))


def test_evaluate_draft_rewrites_bare_math_names():
    draft = _stretched_draft(
        name="UserCos",
        formula="A*cos(w*x)",
        parameters=[DraftParameter("A", 2.0), DraftParameter("w", 3.0)],
    )
    x = np.linspace(0.0, 2.0, 11)
    np.testing.assert_allclose(evaluate_draft(draft, x), 2.0 * np.cos(3.0 * x))


# ── validate_draft failures leave registries untouched ──────────────────────


def _registries_snapshot():
    return (dict(COMPONENTS), dict(MODELS), dict(PARAMETER_MODEL_COMPONENTS))


def test_validate_draft_bad_kind_rejected():
    before = _registries_snapshot()
    with pytest.raises(UserFunctionError, match="Unknown function kind"):
        validate_draft(_stretched_draft(kind="widget"))
    assert _registries_snapshot() == before


def test_validate_draft_bad_domain_rejected():
    before = _registries_snapshot()
    with pytest.raises(UserFunctionError, match="valid domain"):
        validate_draft(_stretched_draft(domain="pressure"))
    assert _registries_snapshot() == before


def test_validate_draft_name_collision_with_builtin_rejected():
    before = _registries_snapshot()
    with pytest.raises(UserFunctionError, match="already registered"):
        validate_draft(_stretched_draft(name="Exponential"))
    assert _registries_snapshot() == before
    assert "Exponential" not in COMPONENTS or COMPONENTS["Exponential"].user is False


def test_validate_draft_invalid_param_name_rejected():
    before = _registries_snapshot()
    draft = _stretched_draft(
        name="UserBadParam",
        formula="A*x",
        parameters=[DraftParameter("bad name", 1.0)],
    )
    with pytest.raises(UserFunctionError, match="invalid parameter name"):
        validate_draft(draft)
    assert _registries_snapshot() == before


@pytest.mark.parametrize("reserved", ["exp", "x", "np"])
def test_validate_draft_param_shadowing_reserved_name_rejected(reserved):
    before = _registries_snapshot()
    draft = _stretched_draft(
        name="UserShadow",
        formula=f"{reserved}*2" if reserved not in {"x", "np"} else "x*2",
        parameters=[DraftParameter(reserved, 1.0)],
    )
    with pytest.raises(UserFunctionError, match="reserved formula name"):
        validate_draft(draft)
    assert _registries_snapshot() == before


def test_validate_draft_unknown_name_names_it_and_lists_params():
    draft = _stretched_draft(
        name="UserUnknown",
        formula="A*exp(-x/taau)",  # typo: taau instead of tau
        parameters=[DraftParameter("A", 1.0), DraftParameter("tau", 1.0)],
    )
    with pytest.raises(UserFunctionError) as excinfo:
        validate_draft(draft)
    message = str(excinfo.value)
    assert "taau" in message
    assert "tau" in message  # declared parameters are listed


def test_validate_draft_syntax_error_rejected():
    draft = _stretched_draft(
        name="UserSyntax",
        formula="A*exp(-x",
        parameters=[DraftParameter("A", 1.0)],
    )
    with pytest.raises(UserFunctionError, match="not valid Python"):
        validate_draft(draft)


def test_validate_draft_non_finite_probe_rejected():
    # log(x - 100) is NaN over the time probe grid (x in [0, 32]).
    draft = _stretched_draft(
        name="UserNonFinite",
        formula="A*log(x - 100)",
        parameters=[DraftParameter("A", 1.0)],
    )
    with pytest.raises(UserFunctionError, match="non-finite"):
        validate_draft(draft)


def test_validate_draft_advanced_body_without_return_rejected():
    draft = _stretched_draft(
        name="UserNoReturn",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body="y = A * np.asarray(x, dtype=float)",
    )
    with pytest.raises(UserFunctionError, match="return statement"):
        validate_draft(draft)


def test_advanced_body_return_only_in_comment_gets_friendly_error():
    # The word "return" in a comment is not a return statement; the friendly
    # early check must still fire (not a downstream probe error).
    draft = _stretched_draft(
        name="UserCommentReturn",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body="# return the scaled array\nA * np.asarray(x, dtype=float)",
    )
    with pytest.raises(UserFunctionError, match="return statement"):
        validate_draft(draft)


def test_advanced_body_return_only_in_string_literal_gets_friendly_error():
    draft = _stretched_draft(
        name="UserStringReturn",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body=(
            '"""This helper should return an array eventually."""\n'
            "y = A * np.asarray(x, dtype=float)"
        ),
    )
    with pytest.raises(UserFunctionError, match="return statement"):
        validate_draft(draft)


def test_advanced_body_return_only_in_nested_def_gets_friendly_error():
    # A helper closure's return belongs to the helper, not the outer body —
    # the outer function still falls off the end returning None.
    draft = _stretched_draft(
        name="UserNestedReturn",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body=(
            "def helper(v):\n    return v * 2\ny = helper(A * np.asarray(x, dtype=float))"
        ),
    )
    with pytest.raises(UserFunctionError, match="return statement"):
        validate_draft(draft)


def test_advanced_body_return_inside_branches_validates():
    draft = _stretched_draft(
        name="UserBranchReturn",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body=(
            "x = np.asarray(x, dtype=float)\n"
            "if A >= 0:\n"
            "    return A * np.exp(-x)\n"
            "else:\n"
            "    return -A * np.exp(-x)"
        ),
    )
    func = validate_draft(draft)
    x = np.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(func(x, A=1.0), np.exp(-x))


def test_advanced_body_syntax_error_reported_as_syntax_not_missing_return():
    # A body that is both syntactically broken AND lacks a return must get the
    # syntax-error message — reporting "missing return" for unparsable code
    # would send the author chasing the wrong problem.
    draft = _stretched_draft(
        name="UserBodySyntax",
        parameters=[DraftParameter("A", 1.0)],
        advanced_body="y = A * (np.asarray(x, dtype=float)",
    )
    with pytest.raises(UserFunctionError, match="not valid Python"):
        validate_draft(draft)


def test_validate_draft_empty_description_rejected():
    with pytest.raises(UserFunctionError, match="description"):
        validate_draft(_stretched_draft(description="   "))


def test_validate_draft_returns_callable_on_success():
    func = validate_draft(_stretched_draft())
    x = np.linspace(0.0, 3.0, 7)
    np.testing.assert_allclose(func(x, A=25.0, tau=1.0, alpha=1.0), 25.0 * np.exp(-x))
    assert "UserStretched" not in COMPONENTS  # still no mutation


# ── create_user_function ────────────────────────────────────────────────────


def test_create_user_function_writes_file_and_registers(tmp_path):
    result = create_user_function(_stretched_draft(), directory=tmp_path)
    assert isinstance(result, CreatedUserFunction)
    assert result.path.exists()
    assert result.path.parent == tmp_path
    assert result.definition is COMPONENTS["UserStretched"]
    assert COMPONENTS["UserStretched"].user is True


def test_create_user_function_parameter_kind(tmp_path):
    result = create_user_function(_trend_draft(), directory=tmp_path)
    assert result.definition is PARAMETER_MODEL_COMPONENTS["UserTrend"]


def test_create_user_function_uniquifies_filename(tmp_path):
    first = create_user_function(_stretched_draft(name="UserStretched"), directory=tmp_path)
    assert first.path.name == "user_stretched.py"
    # A different registry name whose snake-case stem collides gets a _2 suffix.
    (tmp_path / "user_decay.py").write_text("# placeholder\n", encoding="utf-8")
    second = create_user_function(
        _stretched_draft(
            name="UserDecay", formula="A*exp(-x)", parameters=[DraftParameter("A", 1.0)]
        ),
        directory=tmp_path,
    )
    assert second.path.name == "user_decay_2.py"


def test_create_user_function_validation_failure_leaves_no_file(tmp_path):
    with pytest.raises(UserFunctionError, match="valid domain"):
        create_user_function(_stretched_draft(domain="pressure"), directory=tmp_path)
    assert list(tmp_path.glob("*.py")) == []
    assert "UserStretched" not in COMPONENTS


def test_create_user_function_directory_default_is_resolved_at_call_time(tmp_path, monkeypatch):
    from asymmetry.core import plugins

    monkeypatch.setattr(plugins, "USER_FUNCTIONS_DIR", tmp_path / "created")
    result = create_user_function(_stretched_draft(), directory=None)
    assert result.path.parent == tmp_path / "created"
    assert result.path.exists()


# ── load_plugin_file (plugins.py) ───────────────────────────────────────────


def test_load_plugin_file_appends_to_last_report(tmp_path):
    from asymmetry.core import plugins

    plugins.load_user_functions(tmp_path)  # establish a report for this directory
    baseline = len(plugins.last_load_report().sources)

    draft = _stretched_draft()
    path = tmp_path / "stretched.py"
    path.write_text(generate_plugin_source(draft), encoding="utf-8")
    source = plugins.load_plugin_file(path)

    assert source.ok
    assert source.registered_names() == ["UserStretched"]
    report = plugins.last_load_report()
    assert len(report.sources) == baseline + 1
    assert report.sources[-1] is source


def test_load_plugin_file_creates_report_when_none_exists(tmp_path, monkeypatch):
    from asymmetry.core import plugins

    monkeypatch.setattr(plugins, "_last_report", None)
    draft = _stretched_draft()
    path = tmp_path / "stretched.py"
    path.write_text(generate_plugin_source(draft), encoding="utf-8")

    source = plugins.load_plugin_file(path)
    assert source.ok
    report = plugins.last_load_report()
    assert report is not None
    assert report.directory == str(tmp_path)
    assert report.sources[-1] is source


def test_load_plugin_file_reports_error_and_registers_nothing(tmp_path):
    from asymmetry.core import plugins

    broken = tmp_path / "broken.py"
    broken.write_text("this is not valid python :(\n", encoding="utf-8")

    source = plugins.load_plugin_file(broken)
    assert not source.ok
    assert source.error
    assert source.registered == []


# ── restart parity ──────────────────────────────────────────────────────────


def test_created_function_reloads_identically_after_restart(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    draft = _stretched_draft()
    create_user_function(draft, directory=tmp_path)

    x = np.linspace(0.0, 10.0, 33)
    first_def = COMPONENTS["UserStretched"]
    first_values = first_def.function(x, **first_def.param_defaults)
    first_params = list(first_def.param_names)
    first_defaults = dict(first_def.param_defaults)

    # Simulate a restart: drop the registration, then rediscover from disk.
    del COMPONENTS["UserStretched"]
    assert "UserStretched" not in COMPONENTS
    load_user_functions(tmp_path)

    reloaded = COMPONENTS["UserStretched"]
    assert reloaded.param_names == first_params
    assert reloaded.param_defaults == first_defaults
    np.testing.assert_array_equal(reloaded.function(x, **reloaded.param_defaults), first_values)
