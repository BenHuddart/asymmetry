# Python user functions: comparison

| Aspect | musrfit | WiMDA | Mantid | Asymmetry (proposed) |
|---|---|---|---|---|
| Language | C++ inheriting `PUserFcnBase` | DLL `stdcall` / Fortran exports | C++ Algorithm Factory | Python decorator |
| Build step | yes (CMake + ROOT) | yes (Delphi or external DLL) | yes (Mantid algorithm plugin) | none |
| Discovery | ROOT plugin manager | DLL search path | algorithm registry | filesystem scan |
| Hot reload | no | partial (re-attach DLL) | restart Mantid | restart Asymmetry |
| Friction | high | high | high | ★ low (one file, decorator) |

## API contract

```python
def register_component(
    *,
    name: str,
    params: list[str],
    defaults: dict[str, float],
    param_info: dict[str, ParamInfo] | None = None,
    description: str = "",
    formula_template: str | None = None,
    latex_equation: str | None = None,
    category: str = "user",
) -> Callable[[Callable], Callable]:
    """Decorator that registers a callable into the COMPONENTS registry.

    The decorated function must have signature
    `f(t: ArrayLike, **params) -> ArrayLike`.
    """
```

## Discovery flow

1. On startup, Asymmetry reads the `ASYMMETRY_PLUGIN_DIRS` env var
   (colon-separated) and the default `~/.asymmetry/plugins/`.
2. Each `*.py` file in those dirs is imported in lexicographic order.
3. The `@register_component` decorator runs at import time and adds
   entries to COMPONENTS with the user's choice of category.
4. The Fit Function Builder picks up new entries under that category.

## Validation

- The decorator validates that the function signature accepts the
  declared params (via `inspect.signature`).
- A small synthetic input is evaluated at import time to catch
  obvious errors (the function should return a 1-D array of the
  right length).
- Failures log an error and skip the offending plugin without
  crashing the GUI.

## Security note

Plugin code runs in-process with full Python privileges. This is the
same security model as scipy custom callables. Documentation should
warn shared-install admins to restrict the plugin directory.
