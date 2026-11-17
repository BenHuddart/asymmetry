"""Bounded static ownership proofs for literal built-in attribute probes.

No application imports: class declarations and simple bindings are the authority.
Unknown receivers and control-flow joins lose their type rather than guessing.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

Binding = tuple[str, str]
TRY_STATEMENTS = (ast.Try, ast.TryStar) if sys.version_info >= (3, 11) else (ast.Try,)


@dataclass
class OwnedClass:
    module: str
    node: ast.ClassDef
    definition_env: dict[str, Binding]
    members: set[str] = field(default_factory=set)
    children: dict[str, str] = field(default_factory=dict)
    bases: set[str] = field(default_factory=set)
    protocol: bool = False


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def import_bindings(
    node: ast.Import | ast.ImportFrom, module: str, package: bool
) -> dict[str, Binding]:
    result = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name if alias.asname else alias.name.split(".")[0]
            result[alias.asname or name] = ("module", name)
    else:
        prefix = node.module or ""
        if node.level:
            parents = module.split(".") if package else module.split(".")[:-1]
            prefix = ".".join(
                parents[: len(parents) - node.level + 1] + ([prefix] if prefix else [])
            )
        for alias in node.names:
            if alias.name != "*":
                result[alias.asname or alias.name] = ("symbol", f"{prefix}.{alias.name}")
    return result


class Ownership:
    """One parse/index per source tree; bounded project imports and inheritance."""

    def __init__(self, sources: dict[str, bytes]):
        self.trees = {path: ast.parse(source, filename=path) for path, source in sources.items()}
        self.classes: dict[str, OwnedClass] = {}
        self.modules: dict[str, dict[str, Binding]] = {}
        self.packages = {module_name(p) for p in sources if p.endswith("/__init__.py")}
        for path, tree in self.trees.items():
            module = module_name(path)
            env = {"hasattr": ("builtin", "hasattr"), "getattr": ("builtin", "getattr")}
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    env.update(import_bindings(node, module, module in self.packages))
                elif isinstance(node, ast.ClassDef):
                    key = f"{module}.{node.name}"
                    self.classes[key] = OwnedClass(module, node, dict(env))
                    env[node.name] = ("class", key)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    env[node.name] = ("unknown", "")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    binding = self.symbol(node.value, env) if node.value else ("unknown", "")
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            env[target.id] = binding
                else:
                    for name in bound_names(node):
                        env[name] = ("unknown", "")
            self.modules[module] = env
        for key, info in self.classes.items():
            env = self.modules[info.module]
            info.bases = {
                name
                for base in info.node.bases
                for kind, name in [
                    self.symbol(
                        base.value if isinstance(base, ast.Subscript) else base, info.definition_env
                    )
                ]
                if kind == "class" and name in self.classes
            }
            info.protocol = any(
                self.symbol(
                    base.value if isinstance(base, ast.Subscript) else base, info.definition_env
                )[1]
                in {"typing.Protocol", "typing_extensions.Protocol"}
                for base in info.node.bases
            )
            choices: dict[str, set[str | None]] = {}
            class_env = dict(info.definition_env)
            for node in info.node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info.members.add(node.name)
                    class_env[node.name] = ("unknown", "")
                    args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    if not args or any(
                        isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}
                        for d in node.decorator_list
                    ):
                        continue
                    receiver = args[0].arg
                    if receiver in bound_names(node):
                        # A rebound first argument cannot declare this class's state.
                        continue
                    local = dict(env)
                    for name in bound_names(node) | {a.arg for a in args}:
                        local[name] = ("unknown", "")
                    # Local imports prove a constructor only if no parameter or
                    # assignment can rebind that name anywhere in this method.
                    writes = bound_names(node, imports=False) | {a.arg for a in args}
                    for child in walk_scope(node):
                        if isinstance(child, (ast.Import, ast.ImportFrom)) and child in node.body:
                            local.update(
                                {
                                    name: binding
                                    for name, binding in import_bindings(
                                        child, info.module, info.module in self.packages
                                    ).items()
                                    if name not in writes
                                }
                            )
                        if isinstance(child, (ast.Assign, ast.AnnAssign)):
                            targets = (
                                child.targets if isinstance(child, ast.Assign) else [child.target]
                            )
                            for outer in targets:
                                for target in assignment_targets(outer):
                                    if (
                                        isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == receiver
                                    ):
                                        info.members.add(target.attr)
                                        value = (
                                            self.assigned_type(child, local)
                                            if target is outer
                                            else None
                                        )
                                        choices.setdefault(target.attr, set()).add(value)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            info.members.add(target.id)
                            choices.setdefault(target.id, set()).add(
                                self.assigned_type(node, class_env)
                            )
                            if target.id == "__slots__" and isinstance(
                                node.value, (ast.Tuple, ast.List, ast.Constant)
                            ):
                                slots = (
                                    node.value.elts
                                    if isinstance(node.value, (ast.Tuple, ast.List))
                                    else [node.value]
                                )
                                info.members.update(
                                    s.value
                                    for s in slots
                                    if isinstance(s, ast.Constant) and isinstance(s.value, str)
                                )
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    binding = self.symbol(node.value, class_env) if node.value else ("unknown", "")
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            class_env[target.id] = binding
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    class_env.update(
                        import_bindings(node, info.module, info.module in self.packages)
                    )
                elif isinstance(node, ast.ClassDef):
                    class_env[node.name] = ("unknown", "")
            info.children = {
                name: next(iter(types))
                for name, types in choices.items()
                if len(types) == 1 and None not in types
            }
        # Monotone member closure terminates even for malformed cyclic inheritance.
        for _ in range(len(self.classes)):
            changed = False
            for info in self.classes.values():
                for base in info.bases:
                    parent = self.classes[base]
                    before = len(info.members)
                    info.members.update(parent.members)
                    changed |= before != len(info.members)
                    if parent.protocol and not info.protocol:
                        info.protocol = True
                        changed = True
            if not changed:
                break

    def symbol(self, expr: ast.expr, env: dict[str, Binding]) -> Binding:
        if isinstance(expr, ast.Name):
            binding = env.get(expr.id, ("unknown", ""))
        elif isinstance(expr, ast.Attribute):
            kind, name = self.symbol(expr.value, env)
            if kind == "module":
                binding = ("symbol", f"{name}.{expr.attr}")
            elif kind == "instance" and name in self.classes:
                child = self.classes[name].children.get(expr.attr)
                return (
                    ("instance", child)
                    if child and not self.classes[child].protocol
                    else ("unknown", "")
                )
            else:
                return ("unknown", "")
        else:
            return ("unknown", "")
        kind, name = binding
        if name in ("builtins.hasattr", "builtins.getattr") and kind == "symbol":
            return ("builtin", name.split(".")[-1])
        seen = set()
        while kind == "symbol" and name not in seen:
            seen.add(name)
            module, _, member = name.rpartition(".")
            kind, name = self.modules.get(module, {}).get(member, binding)
            if (kind, name) == binding:
                break
            binding = (kind, name)
        return kind, name

    def class_type(self, expr: ast.expr, env: dict[str, Binding]) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            try:
                return self.class_type(ast.parse(expr.value, mode="eval").body, env)
            except SyntaxError:
                return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            if isinstance(expr.right, ast.Constant) and expr.right.value is None:
                return self.class_type(expr.left, env)
            return None
        if isinstance(expr, ast.Subscript) and self.symbol(expr.value, env) == (
            "symbol",
            "typing.Optional",
        ):
            return self.class_type(expr.slice, env)
        kind, name = self.symbol(expr, env)
        return (
            name
            if kind == "class" and name in self.classes and not self.classes[name].protocol
            else None
        )

    def assigned_type(
        self, node: ast.Assign | ast.AnnAssign, env: dict[str, Binding]
    ) -> str | None:
        if isinstance(node.value, ast.Call):
            return self.class_type(node.value.func, env)
        if isinstance(node, ast.AnnAssign):
            return self.class_type(node.annotation, env)
        return None

    def probes(self) -> dict[str, Counter[str]]:
        result = {}
        for path, tree in self.trees.items():
            scanner = ProbeScanner(self, module_name(path))
            scanner.block(tree.body, dict(self.modules[module_name(path)]), "<module>")
            if scanner.found:
                result[path] = scanner.found
        return result


def walk_scope(node: ast.AST):
    """Walk a function body without mistaking nested receiver bindings for its own."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from walk_scope(child)


def assignment_targets(target: ast.AST):
    """Unpacking assigns each leaf; its value type is deliberately unresolved."""
    if isinstance(target, (ast.Tuple, ast.List)):
        for child in target.elts:
            yield from assignment_targets(child)
    elif isinstance(target, ast.Starred):
        yield from assignment_targets(target.value)
    else:
        yield target


def bound_names(node: ast.AST, *, imports: bool = True) -> set[str]:
    names = set()
    for child in walk_scope(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif imports and isinstance(child, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
            names.add(child.name)
    return names


class ProbeScanner:
    def __init__(self, ownership: Ownership, module: str):
        self.ownership = ownership
        self.module = module
        self.found: Counter[str] = Counter()

    def expression(self, expr: ast.AST, env: dict[str, Binding], scope: str) -> None:
        if isinstance(
            expr, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            # Comprehension/lambda bindings have separate scope; no stale aliases.
            return
        if isinstance(expr, ast.NamedExpr):
            self.expression(expr.value, env, scope)
            for name in bound_names(expr):
                env[name] = ("unknown", "")
            return
        if isinstance(expr, ast.Call):
            kind, builtin = self.ownership.symbol(expr.func, env)
            arity = 2 if builtin == "hasattr" else 3
            if kind == "builtin" and len(expr.args) == arity and not expr.keywords:
                receiver, attr = expr.args[:2]
                default_none = (
                    arity == 2
                    or isinstance(expr.args[2], ast.Constant)
                    and expr.args[2].value is None
                )
                owner_kind, owner = self.ownership.symbol(receiver, env)
                if (
                    default_none
                    and isinstance(attr, ast.Constant)
                    and isinstance(attr.value, str)
                    and owner_kind == "instance"
                    and attr.value in self.ownership.classes[owner].members
                ):
                    # Scope and complete normalized call distinguish replacements; line shifts do not.
                    identity = probe_identity(scope, owner, expr)
                    self.found[identity] += 1
        for child in ast.iter_child_nodes(expr):
            self.expression(child, env, scope)

    def block(
        self,
        nodes: list[ast.stmt],
        env: dict[str, Binding],
        scope: str,
        owner: str | None = None,
        captured_writes: frozenset[str] = frozenset(),
    ) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                key = f"{self.module}.{node.name}"
                if key in self.ownership.classes and self.ownership.classes[key].node is node:
                    self.block(node.body, dict(env), node.name, key)
                    env[node.name] = ("class", key)
                else:
                    env[node.name] = ("unknown", "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Methods use module globals, never the class body's namespace.
                # Closures lose captures that their enclosing function may rebind.
                local = dict(
                    self.ownership.modules[self.module] if owner or scope == "<module>" else env
                )
                for name in captured_writes:
                    local[name] = ("unknown", "")
                args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                args += [a for a in (node.args.vararg, node.args.kwarg) if a]
                for name in bound_names(node) | {a.arg for a in args}:
                    local[name] = ("unknown", "")
                for arg in args:
                    if arg.annotation and (name := self.ownership.class_type(arg.annotation, env)):
                        local[arg.arg] = ("instance", name)
                static = any(
                    isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}
                    for d in node.decorator_list
                )
                if owner and args and not static and not self.ownership.classes[owner].protocol:
                    local[args[0].arg] = ("instance", owner)
                self.block(
                    node.body,
                    local,
                    f"{scope}.{node.name}",
                    captured_writes=frozenset(bound_names(node)),
                )
                env[node.name] = ("unknown", "")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                env.update(
                    import_bindings(node, self.module, self.module in self.ownership.packages)
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value:
                    self.expression(node.value, env, scope)
                binding = ("unknown", "")
                if isinstance(node.value, ast.Call):
                    name = self.ownership.class_type(node.value.func, env)
                    if name:
                        binding = ("instance", name)
                elif node.value:
                    binding = self.ownership.symbol(node.value, env)
                if isinstance(node, ast.AnnAssign) and not node.value:
                    name = self.ownership.class_type(node.annotation, env)
                    if name:
                        binding = ("instance", name)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = binding
                    else:
                        for name in bound_names(target):
                            env[name] = ("unknown", "")
            elif isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    *TRY_STATEMENTS,
                    ast.Match,
                ),
            ):
                # Drop all names rebound anywhere in a compound statement before
                # inspecting branches; this deliberately trades coverage for proof.
                local = dict(env)
                for name in bound_names(node):
                    local[name] = ("unknown", "")
                    env[name] = ("unknown", "")
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr):
                        self.expression(child, local, scope)
                for field in ("body", "orelse", "finalbody"):
                    body = dict(ast.iter_fields(node)).get(field, [])
                    self.block(body, dict(local), scope, captured_writes=captured_writes)
                if isinstance(node, TRY_STATEMENTS):
                    for handler in node.handlers:
                        branch = dict(local)
                        if handler.name:
                            branch[handler.name] = ("unknown", "")
                        self.block(handler.body, branch, scope, captured_writes=captured_writes)
                if isinstance(node, ast.Match):
                    for case in node.cases:
                        branch = dict(local)
                        for child in ast.walk(case.pattern):
                            if isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                                branch[child.name] = ("unknown", "")
                        self.block(case.body, branch, scope, captured_writes=captured_writes)
            else:
                self.expression(node, env, scope)
                for name in bound_names(node):
                    env[name] = ("unknown", "")


def probe_identity(scope: str, owner: str, call: ast.Call) -> str:
    # Empty Call keywords must have the same representation across Python versions.
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    normalized = ast.dump(call, annotate_fields=True, include_attributes=False, **options)
    return f"{scope}|{owner}|{normalized}"
