"""Tests for the user-function registration facade and plugin discovery.

Every test restores the registries it touches (the ``_registry_snapshot``
autouse fixture), so user registrations made here can never leak into other
tests' view of ``COMPONENTS``/``MODELS``/``PARAMETER_MODEL_COMPONENTS``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fitting.component_docs import (
    get_component_applicability,
    get_component_references,
)
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    component_names_for_x,
)
from asymmetry.core.fitting.user_functions import (
    UserFunctionError,
    register_component,
    register_parameter_component,
)


@pytest.fixture(autouse=True)
def _registry_snapshot(registry_snapshot):
    """Every test here may register user functions; isolate via the shared
    ``registry_snapshot`` fixture from conftest."""
    yield


def _stretched(t, A, tau, alpha):
    return A * np.exp(-((t / tau) ** alpha))


def _register_stretched(name="UserStretched", **overrides):
    kwargs = dict(
        domain="time",
        description="Stretched exponential test component",
        formula_template="{A}*exp(-(t/{tau})^{alpha})",
        param_defaults={"A": 25.0, "tau": 1.0, "alpha": 1.0},
    )
    kwargs.update(overrides)
    return register_component(name, _stretched, ["A", "tau", "alpha"], **kwargs)


# ── registration happy path ────────────────────────────────────────────────


def test_register_component_appears_in_registry_flagged_user():
    definition = _register_stretched()
    assert COMPONENTS["UserStretched"] is definition
    assert definition.user is True
    assert definition.domain == "time"
    assert definition.category == "User"


def test_registered_component_works_in_composite_expressions():
    _register_stretched()
    model = CompositeModel.from_expression("UserStretched + Constant")
    t = np.linspace(0.0, 16.0, 33)
    out = model.function(t, **model.param_defaults)
    assert out.shape == t.shape
    assert np.all(np.isfinite(out))


def test_register_component_documentation_reaches_kind_aware_lookup():
    _register_stretched(
        applicability="Use for distributed relaxation rates in disordered systems.",
        references=("J. Doe, Phys. Rev. B 1, 1 (1970).",),
    )
    assert "disordered" in get_component_applicability("UserStretched", kind="fit")
    assert get_component_references("UserStretched", kind="fit") == (
        "J. Doe, Phys. Rev. B 1, 1 (1970).",
    )


def test_register_parameter_component_appears_in_scope_filter():
    register_parameter_component(
        "UserTrend",
        lambda x, a, b: a * np.asarray(x, dtype=float) + b,
        ["a", "b"],
        description="Linear test trend",
        formula_template="{a}*x+{b}",
        scopes=("temperature",),
    )
    assert PARAMETER_MODEL_COMPONENTS["UserTrend"].user is True
    assert "UserTrend" in component_names_for_x("temperature")
    assert "UserTrend" not in component_names_for_x("field")
    model = ParameterCompositeModel(["UserTrend"])
    out = model.function(np.linspace(1.0, 10.0, 7), **model.param_defaults)
    assert np.all(np.isfinite(out))


def test_param_defaults_fill_with_unity_when_omitted():
    definition = _register_stretched(param_defaults={"A": 25.0})
    assert definition.param_defaults == {"A": 25.0, "tau": 1.0, "alpha": 1.0}


# ── designed load-time failures ────────────────────────────────────────────


def test_bad_signature_rejected_at_registration():
    def two_param_only(t, A):
        return A * np.asarray(t, dtype=float)

    with pytest.raises(UserFunctionError, match="probe evaluation failed"):
        register_component(
            "UserBadArity",
            two_param_only,
            ["A", "tau"],
            domain="time",
            description="x",
            formula_template="{A}",
        )
    assert "UserBadArity" not in COMPONENTS


def test_nan_on_probe_grid_rejected_at_registration():
    def logs_at_zero(t, A):
        return A * np.log(np.asarray(t, dtype=float) - 1.0)

    with pytest.raises(UserFunctionError, match="non-finite"):
        register_component(
            "UserNaN",
            logs_at_zero,
            ["A"],
            domain="time",
            description="x",
            formula_template="{A}",
        )
    assert "UserNaN" not in COMPONENTS


def test_non_vectorised_function_rejected():
    def scalar_only(t, A):
        return float(A)

    with pytest.raises(UserFunctionError, match="vectorised"):
        register_component(
            "UserScalar",
            scalar_only,
            ["A"],
            domain="time",
            description="x",
            formula_template="{A}",
        )


def test_collision_with_builtin_component_rejected():
    with pytest.raises(UserFunctionError, match="already registered"):
        _register_stretched(name="Keren")
    assert COMPONENTS["Keren"].user is False


def test_cross_registry_collisions_rejected():
    # A MODELS name and a PARAMETER_MODEL_COMPONENTS name are both off-limits
    # for new fit components (N4 cross-registry uniqueness), and vice versa.
    with pytest.raises(UserFunctionError, match="built-in model"):
        _register_stretched(name="ExponentialRelaxation")
    with pytest.raises(UserFunctionError, match="parameter-trend"):
        _register_stretched(name="Arrhenius")
    with pytest.raises(UserFunctionError, match="fit component"):
        register_parameter_component(
            "Keren",
            lambda x, a: a * np.asarray(x, dtype=float),
            ["a"],
            description="x",
            formula_template="{a}",
        )


def test_missing_domain_rejected():
    with pytest.raises(UserFunctionError, match="valid domain"):
        register_component(
            "UserNoDomain",
            _stretched,
            ["A", "tau", "alpha"],
            domain="",
            description="x",
            formula_template="{A}",
        )


def test_missing_metadata_rejected():
    with pytest.raises(UserFunctionError, match="description"):
        _register_stretched(description="")
    with pytest.raises(UserFunctionError, match="formula_template"):
        _register_stretched(formula_template="  ")


def test_grammar_incompatible_name_rejected():
    for bad_name in ("My Decay", "1Decay", "My-Decay", "frac", ""):
        with pytest.raises(UserFunctionError):
            _register_stretched(name=bad_name)


def test_stray_formula_placeholder_rejected():
    with pytest.raises(UserFunctionError, match="placeholders"):
        _register_stretched(formula_template="{A}*exp(-t*{lambda_typo})")


def test_invalid_scopes_rejected():
    with pytest.raises(UserFunctionError, match="scopes"):
        register_parameter_component(
            "UserBadScope",
            lambda x, a: a * np.asarray(x, dtype=float),
            ["a"],
            description="x",
            formula_template="{a}",
            scopes=("pressure",),
        )


def test_unknown_default_and_fixed_param_rejected():
    with pytest.raises(UserFunctionError, match="unknown parameter"):
        _register_stretched(param_defaults={"A": 1.0, "zeta": 2.0})
    with pytest.raises(UserFunctionError, match="fixed_params"):
        _register_stretched(fixed_params=("zeta",))


def test_failed_registration_leaves_all_registries_untouched():
    before = (dict(COMPONENTS), dict(MODELS), dict(PARAMETER_MODEL_COMPONENTS))
    with pytest.raises(UserFunctionError):
        _register_stretched(name="UserAtomic", param_defaults={"A": float("nan")})

    # NaN default fails the finiteness probe; nothing may have been inserted.
    assert "UserAtomic" not in COMPONENTS
    after = (dict(COMPONENTS), dict(MODELS), dict(PARAMETER_MODEL_COMPONENTS))
    assert before == after


def test_user_registration_cannot_overwrite_builtin_definition():
    keren_before = COMPONENTS["Keren"]
    with pytest.raises(UserFunctionError):
        _register_stretched(name="Keren")
    assert COMPONENTS["Keren"] is keren_before


# ── guard and docs-test interplay (W17) ────────────────────────────────────


def test_domain_library_filters_include_user_components():
    from asymmetry.core.fitting.domain_library import components_for_domain

    _register_stretched()
    register_component(
        "UserFreqPeak",
        lambda nu, A, width: A * np.exp(-np.square(np.asarray(nu, dtype=float) / width)),
        ["A", "width"],
        domain="frequency",
        description="Test frequency peak",
        formula_template="{A}*exp(-(nu/{width})^2)",
    )
    assert "UserStretched" in components_for_domain("time")
    assert "UserStretched" not in components_for_domain("frequency")
    assert "UserFreqPeak" in components_for_domain("frequency")


def test_docs_enforcement_exemption_is_by_user_flag():
    """User components must not need documentation pages or applicability
    entries: the docs tests iterate built-ins only, selected by flag."""
    _register_stretched()

    from tests.core.test_fit_function_docs import CATEGORY_PAGES

    builtin_categories = {d.category for d in COMPONENTS.values() if not d.user}
    assert builtin_categories <= set(CATEGORY_PAGES)
    # The user component's default category is not a documented built-in
    # category; only the flag keeps it out of the enforcement sweep.
    assert COMPONENTS["UserStretched"].category not in CATEGORY_PAGES


# ── discovery (core/plugins.py) ────────────────────────────────────────────


GOOD_PLUGIN = """
import numpy as np
from asymmetry.core.fitting.user_functions import register_component

def slow_decay(t, A, rate):
    return A * np.exp(-rate * np.asarray(t, dtype=float))

register_component(
    "UserSlowDecay",
    slow_decay,
    ["A", "rate"],
    domain="time",
    description="Test plugin component",
    formula_template="{A}*exp(-{rate}*t)",
    param_defaults={"A": 25.0, "rate": 0.1},
)
"""

CRASHING_PLUGIN = """
raise RuntimeError("boom at import time")
"""

BAD_REGISTRATION_PLUGIN = """
import numpy as np
from asymmetry.core.fitting.user_functions import register_component

register_component(
    "UserNanPlugin",
    lambda t, A: A * np.log(np.asarray(t, dtype=float) - 1.0),
    ["A"],
    domain="time",
    description="x",
    formula_template="{A}",
)
"""


def test_load_user_functions_registers_from_directory(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    (tmp_path / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    report = load_user_functions(tmp_path)

    assert "UserSlowDecay" in COMPONENTS
    assert COMPONENTS["UserSlowDecay"].user is True
    (source,) = report.sources
    assert source.ok and source.kind == "file" and source.name == "good.py"
    assert source.registered_names() == ["UserSlowDecay"]
    assert report.registered_count == 1
    assert not report.failures


def test_load_user_functions_never_raises_on_bad_files(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    (tmp_path / "a_crashes.py").write_text(CRASHING_PLUGIN, encoding="utf-8")
    (tmp_path / "b_bad_registration.py").write_text(BAD_REGISTRATION_PLUGIN, encoding="utf-8")
    (tmp_path / "c_good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    (tmp_path / "_private.py").write_text(CRASHING_PLUGIN, encoding="utf-8")

    report = load_user_functions(tmp_path)

    names = [source.name for source in report.sources]
    assert names == ["a_crashes.py", "b_bad_registration.py", "c_good.py"]
    crash, bad_reg, good = report.sources
    assert "boom at import time" in crash.error
    assert "RuntimeError" in crash.error and crash.detail
    assert "non-finite" in bad_reg.error
    assert good.ok
    # The bad registration mutated nothing; the good plugin still landed.
    assert "UserNanPlugin" not in COMPONENTS
    assert "UserSlowDecay" in COMPONENTS


def test_load_user_functions_missing_directory_is_empty_report(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    report = load_user_functions(tmp_path / "does_not_exist")
    assert report.sources == []
    assert "No user functions found" in report.summary()


def test_reload_reports_duplicates_without_mutating_registries(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    (tmp_path / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    load_user_functions(tmp_path)
    first_definition = COMPONENTS["UserSlowDecay"]

    report = load_user_functions(tmp_path)
    (source,) = report.sources
    assert not source.ok
    assert "already registered" in source.error
    assert COMPONENTS["UserSlowDecay"] is first_definition


def test_last_load_report_tracks_most_recent_call(tmp_path):
    from asymmetry.core import plugins

    (tmp_path / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    report = plugins.load_user_functions(tmp_path)
    assert plugins.last_load_report() is report


def test_entry_point_discovery(tmp_path, monkeypatch):
    from asymmetry.core import plugins

    calls = []

    def hook():
        calls.append("ran")
        register_component(
            "UserFromEntryPoint",
            lambda t, A: A * np.ones_like(np.asarray(t, dtype=float)),
            ["A"],
            domain="time",
            description="Entry-point test component",
            formula_template="{A}",
        )

    class FakeEntryPoint:
        name = "demo"
        value = "demo_pkg:register"

        def load(self):
            return hook

    class NotCallableEntryPoint:
        name = "broken"
        value = "demo_pkg:CONSTANT"

        def load(self):
            return 42

    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda group: [FakeEntryPoint(), NotCallableEntryPoint()],
    )

    report = plugins.load_user_functions(tmp_path / "no_dir")
    assert calls == ["ran"]
    assert "UserFromEntryPoint" in COMPONENTS
    ep_ok, ep_broken = report.sources
    assert ep_ok.ok and ep_ok.kind == "entry_point"
    assert ep_ok.registered_names() == ["UserFromEntryPoint"]
    assert "callable" in ep_broken.error


def test_top_level_lazy_exports():
    import asymmetry

    assert asymmetry.register_component is register_component
    assert asymmetry.register_parameter_component is register_parameter_component
    assert asymmetry.UserFunctionError is UserFunctionError
    from asymmetry.core.plugins import load_user_functions

    assert asymmetry.load_user_functions is load_user_functions
    with pytest.raises(AttributeError):
        asymmetry.not_a_real_attribute  # noqa: B018


# ── named-placeholder degrade (W1) ─────────────────────────────────────────


def _missing_model_dict() -> dict:
    return {
        "component_names": ["UserGoneDecay", "Constant"],
        "operators": ["+"],
        "open_parentheses": [0, 0],
        "close_parentheses": [0, 0],
        "fraction_groups": [],
    }


def test_from_dict_strict_still_raises_for_unknown_components():
    with pytest.raises(ValueError, match="Unknown component"):
        CompositeModel.from_dict(_missing_model_dict())


def test_placeholder_degrade_round_trips_and_evaluates_to_zero():
    data = _missing_model_dict()
    model = CompositeModel.from_dict(data, allow_missing=True)

    assert model.missing_component_names == ("UserGoneDecay",)
    assert model.to_dict() == data  # original names preserved bit-identically
    assert "UserGoneDecay" not in COMPONENTS  # never registered

    t = np.linspace(0.0, 10.0, 21)
    constant = CompositeModel(["Constant"])
    np.testing.assert_array_equal(
        model.function(t, **model.param_defaults),
        constant.function(t, **constant.param_defaults),
    )
    # The placeholder's unknowable domain must not poison the domain check.
    assert model.domains() == {"time"}


def test_placeholder_model_goes_live_once_plugin_returns():
    data = _missing_model_dict()
    degraded = CompositeModel.from_dict(data, allow_missing=True)
    assert degraded.missing_component_names

    register_component(
        "UserGoneDecay",
        lambda t, A: A * np.exp(-np.asarray(t, dtype=float)),
        ["A"],
        domain="time",
        description="Restored plugin component",
        formula_template="{A}*exp(-t)",
    )
    live = CompositeModel.from_dict(data, allow_missing=True)
    assert live.missing_component_names == ()
    t = np.linspace(0.0, 5.0, 11)
    out = live.function(t, **live.param_defaults)
    assert np.all(np.isfinite(out))


# ── Keren example-plugin parity (the tutorial's worked example) ────────────


_EXAMPLE_PLUGIN = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "reference"
    / "examples"
    / "keren_user_function.py"
)


def _load_example_plugin(tmp_path):
    from asymmetry.core.plugins import load_user_functions

    target = tmp_path / "keren_user_function.py"
    target.write_text(_EXAMPLE_PLUGIN.read_text(encoding="utf-8"), encoding="utf-8")
    report = load_user_functions(tmp_path)
    errors = [source.error for source in report.sources if not source.ok]
    assert not errors, errors
    return report


def test_example_plugin_registers_keren_clone(tmp_path):
    report = _load_example_plugin(tmp_path)
    (source,) = report.sources
    assert source.registered_names() == ["KerenUser"]
    definition = COMPONENTS["KerenUser"]
    assert definition.user is True
    assert definition.param_names == COMPONENTS["Keren"].param_names
    assert definition.param_defaults == COMPONENTS["Keren"].param_defaults

    from asymmetry.core.fitting.domain_library import components_for_domain

    assert "KerenUser" in components_for_domain("time")


def test_example_plugin_matches_builtin_keren_bit_for_bit(tmp_path):
    _load_example_plugin(tmp_path)
    builtin = COMPONENTS["Keren"].function
    clone = COMPONENTS["KerenUser"].function

    t = np.linspace(0.0, 32.0, 1025)
    for delta in (0.2, 0.5, 1.5):
        for nu in (0.0, 0.1, 1.0, 10.0):
            for b_l in (0.0, 20.0, 100.0):
                expected = builtin(t, A=25.0, Delta=delta, nu=nu, B_L=b_l)
                actual = clone(t, A=25.0, Delta=delta, nu=nu, B_L=b_l)
                assert np.array_equal(actual, expected), (delta, nu, b_l)


def test_example_plugin_fit_matches_builtin_exactly(tmp_path):
    pytest.importorskip("iminuit")
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    _load_example_plugin(tmp_path)

    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 16.0, 320)
    truth = {"A": 24.0, "Delta": 0.5, "nu": 1.2, "B_L": 20.0}
    y = COMPONENTS["Keren"].function(t, **truth) + rng.normal(0.0, 0.3, t.size)
    dataset = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.3))

    def _fit(component_name):
        model = CompositeModel([component_name])
        start = {"A_1": 20.0, "Delta": 0.3, "nu": 0.8, "B_L": truth["B_L"]}
        params = ParameterSet(
            [
                Parameter(
                    name,
                    value=start.get(name, 1.0),
                    min=0.0,
                    fixed=(name == "B_L"),
                )
                for name in model.param_names
            ]
        )
        result = FitEngine().fit(dataset, model.function, params)
        assert result.success
        return result

    builtin_result = _fit("Keren")
    clone_result = _fit("KerenUser")

    builtin_values = {p.name: p.value for p in builtin_result.parameters}
    clone_values = {p.name: p.value for p in clone_result.parameters}
    assert clone_values == builtin_values
    assert clone_result.uncertainties == builtin_result.uncertainties

    fitted = clone_values
    assert fitted["A_1"] == pytest.approx(truth["A"], abs=0.5)
    assert fitted["Delta"] == pytest.approx(truth["Delta"], abs=0.1)
    assert fitted["nu"] == pytest.approx(truth["nu"], abs=0.4)


def test_example_plugin_persistence_round_trip(tmp_path):
    _load_example_plugin(tmp_path)
    model = CompositeModel.from_expression("KerenUser + Constant")
    data = model.to_dict()

    restored = CompositeModel.from_dict(data)
    assert restored.missing_component_names == ()
    assert restored.component_names == ["KerenUser", "Constant"]
    t = np.linspace(0.0, 16.0, 65)
    np.testing.assert_array_equal(
        restored.function(t, **restored.param_defaults),
        model.function(t, **model.param_defaults),
    )
