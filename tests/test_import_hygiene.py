"""Static import hygiene of the modules that must stay free of XSPEC.

The cube helpers, the torch helpers and the exceptions are imported by pickled
worker processes and by the restriction/SIXSA layers, so they must not import
`xspec` or `bsixsa.solver` (which imports `xspec`), and must not pay for torch
at import time. The check parses the sources with `ast` instead of importing
them: importing any `bsixsa` submodule runs the package `__init__`, which
imports XSPEC, so a runtime check cannot see a violation there.

Later phases extend the file lists below.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# Files (relative to `src/`) that must not import XSPEC or the solver anywhere,
# not even lazily inside a function.
XSPEC_FREE_FILES = (
    "bsixsa/_cube.py",
    "bsixsa/_torch_utils.py",
    "bsixsa/exceptions.py",
)

# Files that may use torch only lazily (inside a function body).
TORCH_LAZY_FILES = (
    "bsixsa/_cube.py",
    "bsixsa/_torch_utils.py",
    "bsixsa/exceptions.py",
)

FORBIDDEN_MODULES = ("xspec", "bsixsa.solver")
_IMPORT_FUNCTIONS = ("import_module", "__import__")


@dataclass(frozen=True)
class _Import:
    """One imported module name found in a source file."""

    module: str
    lineno: int
    module_level: bool


def _package_parts(relative_path: str) -> list[str]:
    """Dotted package of a source file given relative to `src/`."""
    parts = list(Path(relative_path).with_suffix("").parts)
    # A module's package is its directory; `__init__` is the package itself.
    return parts[:-1]


def _resolve_from(node: ast.ImportFrom, package: list[str]) -> str:
    """Absolute module named by a (possibly relative) `from ... import`."""
    if node.level == 0:
        return node.module or ""
    keep = len(package) - (node.level - 1)
    if keep < 0:
        raise ValueError(f"relative import beyond the top-level package at line {node.lineno}")
    base = package[:keep]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _collect_imports(source: str, relative_path: str) -> list[_Import]:
    """Every module a source file imports, with whether it runs at import time.

    `from pkg import name` reports both `pkg` and `pkg.name`, because `name`
    may be a submodule (`from bsixsa import solver`). String literals passed to
    `importlib.import_module` or `__import__` are reported too. Imports under
    `if TYPE_CHECKING:` never execute and are skipped.
    """
    tree = ast.parse(source, filename=relative_path)
    package = _package_parts(relative_path)
    found: list[_Import] = []

    def visit(node: ast.AST, module_level: bool) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in node.orelse:
                visit(child, module_level)
            return
        if isinstance(node, ast.Import):
            found.extend(_Import(alias.name, node.lineno, module_level) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from(node, package)
            found.append(_Import(module, node.lineno, module_level))
            found.extend(
                _Import(f"{module}.{alias.name}" if module else alias.name, node.lineno, module_level)
                for alias in node.names
                if alias.name != "*"
            )
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _IMPORT_FUNCTIONS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.append(_Import(first.value, node.lineno, module_level))
        # Function bodies only run when called; class bodies run at import.
        inner_level = module_level and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for child in ast.iter_child_nodes(node):
            visit(child, inner_level)

    visit(tree, True)
    return found


def _matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(forbidden + ".")


def _violations(imports: list[_Import], forbidden: tuple[str, ...], *, module_level_only: bool) -> list[str]:
    return sorted(
        {
            f"line {imp.lineno}: {imp.module}"
            for imp in imports
            if (imp.module_level or not module_level_only)
            and any(_matches(imp.module, name) for name in forbidden)
        }
    )


def _read(relative_path: str) -> str:
    path = SRC_DIR / relative_path
    assert path.is_file(), f"{relative_path} is listed for import hygiene but does not exist"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("relative_path", XSPEC_FREE_FILES)
def test_module_does_not_import_xspec_or_solver(relative_path):
    imports = _collect_imports(_read(relative_path), relative_path)

    assert _violations(imports, FORBIDDEN_MODULES, module_level_only=False) == []


@pytest.mark.parametrize("relative_path", TORCH_LAZY_FILES)
def test_module_imports_torch_only_lazily(relative_path):
    imports = _collect_imports(_read(relative_path), relative_path)

    assert _violations(imports, ("torch",), module_level_only=True) == []


# ---------------------------------------------------------------------------
# The scanner itself: a scan that sees nothing would pass vacuously.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "relative_path"),
    [
        ("import xspec\n", "bsixsa/_cube.py"),
        ("import xspec.core as core\n", "bsixsa/_cube.py"),
        ("from xspec import AllData\n", "bsixsa/_cube.py"),
        ("from .solver import Solver\n", "bsixsa/_cube.py"),
        ("from . import solver\n", "bsixsa/_cube.py"),
        ("from bsixsa.solver import Solver\n", "bsixsa/_cube.py"),
        ("from bsixsa import solver\n", "bsixsa/_cube.py"),
        ("from ..solver import Solver\n", "bsixsa/restriction/region.py"),
        ("from ... import solver\n", "bsixsa/backend/sixsa/flows.py"),
        ("import importlib\nimportlib.import_module('xspec')\n", "bsixsa/_cube.py"),
        ("def f():\n    import xspec\n", "bsixsa/_cube.py"),
        ("class A:\n    def f(self):\n        from .solver import Solver\n", "bsixsa/_cube.py"),
        ("try:\n    import xspec\nexcept ImportError:\n    pass\n", "bsixsa/_cube.py"),
    ],
)
def test_scanner_flags_xspec_and_solver_imports(source, relative_path):
    imports = _collect_imports(source, relative_path)

    assert _violations(imports, FORBIDDEN_MODULES, module_level_only=False)


@pytest.mark.parametrize(
    ("source", "relative_path"),
    [
        ("import numpy as np\nfrom scipy.stats import qmc\n", "bsixsa/_cube.py"),
        ("import xspecial\n", "bsixsa/_cube.py"),
        ("from .solverlib import helper\n", "bsixsa/_cube.py"),
        ("from .restriction import solver_free\n", "bsixsa/_cube.py"),
        ("from ..priors import MultipleIndependent\n", "bsixsa/restriction/region.py"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .solver import Solver\n", "bsixsa/_cube.py"),
    ],
)
def test_scanner_accepts_clean_imports(source, relative_path):
    imports = _collect_imports(source, relative_path)

    assert _violations(imports, FORBIDDEN_MODULES, module_level_only=False) == []


def test_scanner_distinguishes_lazy_from_module_level_torch():
    lazy = _collect_imports("def f():\n    import torch\n", "bsixsa/_torch_utils.py")
    eager = _collect_imports("import torch.nn as nn\n", "bsixsa/_torch_utils.py")
    in_class = _collect_imports("class A:\n    import torch\n", "bsixsa/_torch_utils.py")

    assert _violations(lazy, ("torch",), module_level_only=True) == []
    assert _violations(eager, ("torch",), module_level_only=True) == ["line 1: torch.nn"]
    assert _violations(in_class, ("torch",), module_level_only=True) == ["line 2: torch"]
