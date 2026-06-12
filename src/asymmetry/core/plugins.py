"""Discovery and loading of user-function plugins.

Two discovery channels, mirroring WiMDA's plugin DLLs at far lower friction:

* **User directory** — every ``*.py`` file in ``~/.asymmetry/user_functions/``
  (sorted, non-recursive, names starting with ``_`` skipped) is imported; the
  file body calls :func:`asymmetry.register_component` /
  :func:`asymmetry.register_parameter_component` to add its functions.
* **Entry points** — installed packages can expose a callable under the
  ``asymmetry.user_functions`` entry-point group; each is invoked once and
  performs its registrations.

Discovery is **explicit**: nothing is imported as a side effect of
``import asymmetry``. The GUI calls :func:`load_user_functions` once at
startup (before the main window is built); scripts call it themselves when
they want plugins available.

Every source loads under a blanket ``except Exception`` into a structured
:class:`UserFunctionLoadReport` — a broken plugin file can never prevent the
application from starting, exactly as WiMDA tolerated broken DLLs. The report
is surfaced in the startup log and the "User functions…" dialog.

Trust model: plugin files are ordinary Python executed with full interpreter
privileges (the WiMDA DLL trust model). Only install files you trust.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from asymmetry.core.fitting import user_functions as _facade

#: Default plugin directory, scanned by :func:`load_user_functions`.
USER_FUNCTIONS_DIR = Path.home() / ".asymmetry" / "user_functions"

#: Entry-point group for packaged plugins.
ENTRY_POINT_GROUP = "asymmetry.user_functions"

#: Counter making synthetic module names unique across repeated loads.
_import_counter = 0


@dataclass
class UserFunctionSource:
    """One scanned plugin source (a file or an entry point) and its outcome."""

    name: str
    kind: str  # "file" | "entry_point"
    location: str
    #: ``(kind, name)`` pairs recorded by the registration facade, with kind
    #: ``"component"`` or ``"parameter_component"``.
    registered: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    #: Formatted traceback for failures (shown in the load-report dialog).
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def registered_names(self) -> list[str]:
        return [name for _kind, name in self.registered]


@dataclass
class UserFunctionLoadReport:
    """The outcome of one :func:`load_user_functions` call."""

    directory: str
    sources: list[UserFunctionSource] = field(default_factory=list)

    @property
    def failures(self) -> list[UserFunctionSource]:
        return [source for source in self.sources if not source.ok]

    @property
    def registered_count(self) -> int:
        return sum(len(source.registered) for source in self.sources)

    def summary(self) -> str:
        """One log-friendly line describing the load outcome."""
        if not self.sources:
            return f"No user functions found ({self.directory})"
        parts = [
            f"{self.registered_count} user function(s) registered "
            f"from {len(self.sources)} source(s)"
        ]
        if self.failures:
            failed = ", ".join(source.name for source in self.failures)
            parts.append(f"{len(self.failures)} failed: {failed}")
        return "; ".join(parts)


_last_report: UserFunctionLoadReport | None = None


def last_load_report() -> UserFunctionLoadReport | None:
    """Return the report from the most recent :func:`load_user_functions`."""
    return _last_report


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _import_plugin_file(path: Path) -> None:
    """Import *path* under a unique synthetic module name.

    The synthetic name keeps plugin files from shadowing installed packages
    and lets the same file be re-imported on a later load call.
    """
    global _import_counter
    _import_counter += 1
    module_name = f"_asymmetry_user_functions_{_import_counter}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


def _load_one(source: UserFunctionSource, action) -> None:
    """Run one plugin *action* with the facade collector attached to *source*."""
    collector: list[tuple[str, str]] = []
    _facade._active_collector = collector
    try:
        action()
    except Exception as exc:
        source.error = _format_error(exc)
        source.detail = traceback.format_exc()
    finally:
        _facade._active_collector = None
        source.registered = list(collector)


def load_user_functions(directory: str | Path | None = None) -> UserFunctionLoadReport:
    """Discover and load user functions; return the structured load report.

    Scans *directory* (default :data:`USER_FUNCTIONS_DIR`) for ``*.py`` files
    and the :data:`ENTRY_POINT_GROUP` entry points. Never raises for plugin
    failures — inspect the returned report (also available afterwards via
    :func:`last_load_report`).
    """
    global _last_report

    dir_path = Path(directory) if directory is not None else USER_FUNCTIONS_DIR
    report = UserFunctionLoadReport(directory=str(dir_path))

    if dir_path.is_dir():
        for path in sorted(dir_path.glob("*.py")):
            if path.name.startswith("_"):
                continue
            source = UserFunctionSource(name=path.name, kind="file", location=str(path))
            _load_one(source, lambda path=path: _import_plugin_file(path))
            report.sources.append(source)

    try:
        entry_points = tuple(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001 — malformed installed metadata must not crash startup
        entry_points = ()
    for entry_point in entry_points:
        source = UserFunctionSource(
            name=entry_point.name,
            kind="entry_point",
            location=entry_point.value,
        )

        def _run(entry_point=entry_point) -> None:
            hook = entry_point.load()
            if not callable(hook):
                raise TypeError(
                    f"Entry point '{entry_point.name}' must resolve to a callable "
                    "that performs the registrations."
                )
            hook()

        _load_one(source, _run)
        report.sources.append(source)

    _last_report = report
    return report
