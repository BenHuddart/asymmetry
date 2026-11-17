"""Ownership examples reflect panel state, project children, and external boundaries."""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._harness import _load_harness


def probes(source: str, **modules: str):
    sources = {"src/panel.py": textwrap.dedent(source).encode()}
    sources.update({path: textwrap.dedent(code).encode() for path, code in modules.items()})
    return _load_harness()._owned_probes.Ownership(sources).probes()


@pytest.mark.parametrize(
    "declaration",
    [
        "value: int",
        "value = 4",
        '__slots__ = ("value",)',
        "def __init__(self):\n    self.value = 4",
        "@property\ndef value(self):\n    return 4",
        "def value(self):\n    return 4",
    ],
)
def test_declared_members(declaration: str) -> None:
    source = "class Panel:\n" + textwrap.indent(declaration, "    ")
    source += '\n    def update(this):\n        return hasattr(this, "value")\n'
    assert sum(probes(source)["src/panel.py"].values()) == 1


def test_actual_panel_pattern_and_bounded_project_child() -> None:
    found = probes(
        """
        from asymmetry.state import ProjectModel
        class Panel:
            def __init__(self):
                self._project_model = ProjectModel()
                self._datasets = []
            def refresh(self):
                datasets = getattr(self, "_datasets", None)
                return getattr(self._project_model, "datasets", None)
    """,
        **{
            "src/asymmetry/state.py": """
        from dataclasses import dataclass
        @dataclass
        class ProjectModel:
            datasets: list
    """
        },
    )
    assert sum(found["src/panel.py"].values()) == 2


@pytest.mark.parametrize(
    "import_line",
    [
        "from builtins import getattr as probe",
        "import builtins as b\nprobe = b.getattr",
        "probe = getattr",
    ],
)
def test_builtin_aliases(import_line: str) -> None:
    assert probes(
        import_line
        + """
class Panel:
    value = 1
    def inspect(self):
        alias = self
        return probe(alias, "value", None)
"""
    )


@pytest.mark.parametrize(
    "statement",
    [
        "alias = external",
        "alias = other()",
        "del alias",
        "for alias in external:\n    pass",
        "with external as alias:\n    pass",
        "if flag:\n    alias = external",
        "alias += external",
    ],
)
def test_receiver_alias_rebinding(statement: str) -> None:
    assert not probes(
        "class Panel:\n    value = 1\n    def inspect(self):\n        alias = self\n"
        + textwrap.indent(statement, "        ")
        + '\n        return hasattr(alias, "value")\n'
    )


@pytest.mark.parametrize(
    "statement",
    [
        "getattr = external",
        "def getattr(*args):\n    return 1",
        "from plugin import getattr",
        "getattr = lambda *args: None",
    ],
)
def test_shadowed_builtin(statement: str) -> None:
    assert not probes(
        "class Panel:\n    value = 1\n    def inspect(self):\n"
        + textwrap.indent(statement, "        ")
        + '\n        return getattr(self, "value", None)\n'
    )


def test_parameter_shadowing_and_self_rebinding() -> None:
    assert not probes("""
        class Panel:
            value = 1
            def inspect(self, getattr, hasattr):
                getattr(self, "value", None)
                self = external
                return hasattr(self, "value")
    """)


def test_local_function_shadows_builtin_even_before_definition() -> None:
    assert not probes("""
        class Panel:
            value = 1
            def inspect(self):
                getattr(self, "value", None)
                def getattr(*args):
                    return 1
    """)


def test_external_and_unresolved_dynamic_boundaries() -> None:
    assert not probes("""
        from PySide6.QtWidgets import QWidget
        class Panel(QWidget):
            value = 1
            def __init__(self):
                self.child = QWidget()
                self.plugin = load_plugin()
            def inspect(self, external, name):
                hasattr(self, "setWindowTitle")
                hasattr(self.child, "value")
                hasattr(self.plugin, "value")
                hasattr(external, "value")
                getattr(self, name, None)
                getattr(self, "value", 0)
                try:
                    self.value()
                except Exception:
                    pass
    """)


def test_owned_inheritance_and_relative_import_reexport() -> None:
    found = probes(
        """
        from pkg import Child
        class Panel(Child):
            def inspect(self):
                return hasattr(self, "value")
    """,
        **{
            "src/pkg/__init__.py": "from .state import Child\n",
            "src/pkg/state.py": "class Child:\n    value: int\n",
        },
    )
    assert found


def test_child_reassignment_destroys_proof() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            def __init__(self):
                self.child = Child()
            def reset(self):
                self.child = external
            def inspect(self):
                return hasattr(self.child, "value")
    """)


def test_unknown_child_member_is_not_owned_just_because_receiver_is() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            child: Child
            def inspect(self):
                return hasattr(self.child, "external_extension")
    """)


def test_shadowed_constructor_does_not_prove_project_child() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            def __init__(self, Child):
                self.child = Child()
            def inspect(self):
                return hasattr(self.child, "value")
    """)


def test_receiver_and_builtin_aliases_stop_after_shadowing() -> None:
    found = probes("""
        class Panel:
            value: int
            def inspect(this):
                probe = hasattr
                alias = this
                probe(alias, "value")
                probe = external
                probe(alias, "value")
                alias = external
                hasattr(alias, "value")
    """)
    assert sum(found["src/panel.py"].values()) == 1


@pytest.mark.parametrize(
    "source",
    [
        """class Panel:
    value: int
    def inspect(self):
        def callback():
            return hasattr(self, "value")
        self = external
        return callback()
""",
        """from builtins import hasattr as probe
class Panel:
    value: int
    def inspect(self):
        return probe(self, "value")
probe = external
""",
        """class Child:
    value: int
class Panel:
    def __init__(self):
        from panel import Child
        Child = external_factory
        self.child = Child()
    def inspect(self):
        return hasattr(self.child, "value")
""",
        """class Panel:
    def external(self):
        self = plugin
        self.value = 3
    def inspect(self):
        return hasattr(self, "value")
""",
    ],
)
def test_proof_cannot_survive_rebinding_across_lexical_scopes(source: str) -> None:
    assert not probes(source)


def test_base_class_is_bound_at_class_definition() -> None:
    assert probes("""
        class Child:
            value: int
        class Panel(Child):
            def inspect(self):
                return hasattr(self, "value")
        Child = external
    """)


def test_class_attribute_does_not_shadow_method_global_builtin() -> None:
    assert probes("""
        class Panel:
            value: int
            hasattr = lambda *args: True
            def inspect(self):
                return hasattr(self, "value")
    """)


def test_rebound_imported_project_constructor_is_external() -> None:
    assert not probes(
        """
        from state import Child
        class Panel:
            def __init__(self):
                self.child = Child()
            def inspect(self):
                return hasattr(self.child, "value")
    """,
        **{"src/state.py": "class Child:\n    value: int\nChild = external\n"},
    )


def test_local_class_name_collision_does_not_prove_ownership() -> None:
    assert not probes("""
        class Panel:
            value: int
        def factory():
            class Panel:
                def inspect(self):
                    return hasattr(self, "value")
            return Panel()
    """)


def test_unpacking_assignment_invalidates_child_type() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            def __init__(self):
                self.child = Child()
            def reset(self):
                (self.child, other) = (external, 1)
            def inspect(self):
                return hasattr(self.child, "value")
    """)


def test_local_class_constructor_cannot_borrow_top_level_identity() -> None:
    assert not probes("""
        class Panel:
            value: int
        def factory():
            class Panel:
                pass
            panel = Panel()
            return hasattr(panel, "value")
    """)


def test_class_field_constructor_obeys_class_body_shadowing() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            Child = external_factory
            child = Child()
            def inspect(self):
                return hasattr(self.child, "value")
    """)


def test_local_definition_invalidates_imported_child_constructor() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            def __init__(self):
                from panel import Child
                def Child():
                    return external
                self.child = Child()
            def inspect(self):
                return hasattr(self.child, "value")
    """)


def test_later_import_does_not_retroactively_prove_child_constructor() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            def __init__(self):
                from plugin import Child
                self.child = Child()
                from panel import Child
            def inspect(self):
                return hasattr(self.child, "value")
    """)


@pytest.mark.parametrize(
    "declaration",
    [
        "from typing import Protocol\nclass Plugin(Protocol):\n    def feature(self): ...\n",
        "import typing as t\nclass Plugin(t.Protocol):\n    def feature(self): ...\n",
        "from typing import Protocol\nclass Base(Protocol):\n    def feature(self): ...\nclass Plugin(Base):\n    pass\n",
    ],
)
def test_protocol_annotation_is_external_structural_boundary(declaration: str) -> None:
    assert not probes(
        declaration + '\ndef inspect(plugin: Plugin):\n    return hasattr(plugin, "feature")\n'
    )


def test_inherited_protocol_with_declared_member_stays_external() -> None:
    assert not probes("""
        from typing import Protocol
        class Base(Protocol):
            pass
        class Plugin(Base):
            def feature(self): ...
        class Panel:
            plugin: Plugin
            def inspect(self):
                return hasattr(self.plugin, "feature")
        def inspect(plugin: Plugin):
            return hasattr(plugin, "feature")
    """)


def test_static_method_definition_shadows_class_field_constructor() -> None:
    assert not probes("""
        class Child:
            value: int
        class Panel:
            @staticmethod
            def Child():
                return external
            child = Child()
            def inspect(self):
                return hasattr(self.child, "value")
    """)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* requires Python 3.11")
def test_except_star_rebinding_invalidates_receiver_alias() -> None:
    assert not probes("""
        class Panel:
            value: int
            def inspect(self):
                alias = self
                try:
                    alias = external
                    hasattr(alias, "value")
                except* Exception:
                    pass
    """)


def test_optional_spelling_alone_is_not_a_project_type_proof() -> None:
    assert not probes("""
        class Child:
            value: int
        Optional = external
        class Panel:
            child: Optional[Child]
            def inspect(self):
                return hasattr(self.child, "value")
    """)


@pytest.mark.parametrize(
    "import_line,annotation",
    [
        ("from typing import Optional as Maybe", "Maybe[Child]"),
        ("import typing as t", "t.Optional[Child]"),
    ],
)
def test_resolved_optional_alias_or_qualified_annotation(import_line: str, annotation: str) -> None:
    assert probes(
        import_line
        + "\nclass Child:\n    value: int\nclass Panel:\n    child: "
        + annotation
        + '\n    def inspect(self):\n        return hasattr(self.child, "value")\n'
    )


def test_semantic_identity_keeps_empty_fields_across_python_versions() -> None:
    found = probes("""
        class Panel:
            value: int
            def inspect(self):
                return getattr(self, "value", None)
    """)
    assert (
        next(iter(found["src/panel.py"]))
        == "Panel.inspect|panel.Panel|Call(func=Name(id='getattr', ctx=Load()), args=[Name(id='self', ctx=Load()), Constant(value='value'), Constant(value=None)], keywords=[])"
    )


def test_walrus_rebinding_precedes_later_probe_in_same_expression() -> None:
    assert not probes("""
        class Panel:
            value: int
            def inspect(self):
                alias = self
                return ((alias := external), hasattr(alias, "value"))
    """)
