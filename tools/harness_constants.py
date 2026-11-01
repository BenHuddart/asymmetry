"""Standard-library checks for the physical-constant authority and its consumers."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = Path("src/asymmetry/core/utils/constants.py")
# Counts freeze the remaining source-reproducing lines, including assignments.
PHYSICAL_CONSTANT_BASELINE = {
    "docs/screenshots/scenarios/grouping_window_t0_row.py": (
        1,
        "Unverified lifetime approximation",
    ),
    "docs/screenshots/scenarios/grouping_window_good_window_row.py": (
        1,
        "Unverified lifetime approximation",
    ),
    "docs/screenshots/scenarios/corpus/lifeas_pnictide.py": (1, "Paper's field-width conversion"),
    "docs/screenshots/scenarios/corpus/maleic_kinetics.py": (2, "Muon-school guide's arithmetic"),
    "docs/screenshots/scenarios/corpus/plateau_ca3co2o6.py": (1, "Paper's precession conversion"),
}

LOCAL_CONSTANT_NAMES = frozenset(
    {
        "R_GAS",
        "HBAR_J_S",
        "MU_0_OVER_4PI",
        "FLUX_QUANTUM_WB",
        "BOLTZMANN_CONSTANT_MEV_PER_K",
        "VACUUM_MUONIUM_A_HF_MHZ",
        "kb_mev",
        "K_B_EV",
    }
)
CANONICAL_NAMES = (
    "PLANCK_CONSTANT_J_S",
    "REDUCED_PLANCK_CONSTANT_J_S",
    "ELEMENTARY_CHARGE_C",
    "BOLTZMANN_CONSTANT_J_PER_K",
    "BOLTZMANN_CONSTANT_MEV_PER_K",
    "AVOGADRO_CONSTANT_PER_MOL",
    "MOLAR_GAS_CONSTANT_J_PER_MOL_K",
    "MUON_MAGNETIC_MOMENT_J_PER_T",
    "MUON_GYROMAGNETIC_RATIO_MHZ_PER_T",
    "ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_PER_T",
    "ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G",
    "PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T",
    "VACUUM_PERMEABILITY_OVER_4PI_H_PER_M",
    "MUON_LIFETIME_US",
    "FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T",
    "MUONIUM_VACUUM_HYPERFINE_MHZ",
    "MAGNETIC_FLUX_QUANTUM_WB",
    "GAUSS_TO_TESLA",
)


@dataclass(frozen=True)
class HarnessFailure:
    """A structural harness failure with a location and remediation hint."""

    path: Path
    line: int
    message: str

    def format(self) -> str:
        try:
            relpath = self.path.relative_to(ROOT)
        except ValueError:
            relpath = self.path
        location = f"{relpath}:{self.line}" if self.line else str(relpath)
        return f"{location}: {self.message}"


def _constant_expression(node: ast.expr, values: dict[str, ast.expr]) -> float:
    """Evaluate the authority's numeric definitions without importing application code."""
    match node:
        case ast.Constant(value=value) if type(value) in (int, float):
            return float(value)
        case ast.Name(id="pi"):
            return math.pi
        case ast.Name(id=name):
            return _constant_expression(values[name], values)
        case ast.BinOp(left=left, op=ast.Mult(), right=right):
            return _constant_expression(left, values) * _constant_expression(right, values)
        case ast.BinOp(left=left, op=ast.Div(), right=right):
            return _constant_expression(left, values) / _constant_expression(right, values)
        case ast.UnaryOp(op=ast.USub(), operand=value):
            return -_constant_expression(value, values)
    raise ValueError(f"Physical constant must have a numeric definition: {ast.dump(node)}")


def _constant_mantissa(value: float) -> float | None:
    """Ignore generic coefficients; recognise constants across powers of ten and units."""
    digits = format(abs(value), ".14g").split("e")[0].replace(".", "").lstrip("0").rstrip("0")
    if len(digits) < 4:
        return None
    return abs(value) / 10 ** math.floor(math.log10(abs(value)))


def find_physical_constant_duplicate_violations(
    root: Path = ROOT, *, baseline: dict[str, tuple[int, str]] | None = None
) -> list[HarnessFailure]:
    """Require one authority; retain only sourced paper arithmetic on a shrinking baseline."""
    baseline = PHYSICAL_CONSTANT_BASELINE if baseline is None else baseline
    authority = ast.parse((ROOT / CONSTANTS_PATH).read_text(encoding="utf-8"))
    definitions = {
        target.id: node.value
        for node in authority.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    mantissas = [
        mantissa
        for name in CANONICAL_NAMES
        for multiplier in (
            (1.0, 2 * math.pi) if name == "MUON_GYROMAGNETIC_RATIO_MHZ_PER_T" else (1.0,)
        )
        if (
            mantissa := _constant_mantissa(
                _constant_expression(definitions[name], definitions) * multiplier
            )
        )
        is not None
    ]
    failures: list[HarnessFailure] = []
    found: dict[str, set[int]] = {}
    paths = sorted(
        path
        for folder in ("src/asymmetry", "docs/screenshots", "tools")
        for path in (root / folder).rglob("*.py")
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative in (CONSTANTS_PATH.as_posix(), "tools/harness.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            message = ""
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Store):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in LOCAL_CONSTANT_NAMES:
                    message = (
                        f"Local physical constant `{name}`; import its unit-bearing authority."
                    )
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                mantissa = _constant_mantissa(node.value)
                if mantissa is not None and any(abs(mantissa - ref) <= 5e-4 for ref in mantissas):
                    message = (
                        "Physical-constant literal; import the authority or derive its rounding."
                    )
            if message:
                found.setdefault(relative, set()).add(node.lineno)
                if relative not in baseline:
                    failures.append(HarnessFailure(path, node.lineno, message))
    for path, (count, reason) in baseline.items():
        actual = len(found.get(path, set()))
        if actual < count:
            message = "Physical-constant baseline is stale; lower its count or remove its entry."
        elif actual > count:
            message = f"Physical-constant baseline grew ({actual} > {count}): {reason}."
        else:
            continue
        failures.append(HarnessFailure(root / path, 0, message))
    return failures
