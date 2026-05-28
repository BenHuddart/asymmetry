# Python user functions

**Status:** candidate.

## What

A formal user-extensibility API where domain experts can add custom
theory functions to Asymmetry by writing a single Python file. The
function gets discovered, validated, and registered into MODELS /
COMPONENTS automatically; appears in the Fit Function Builder and
Fit Wizard without code changes to the core.

## Why

- musrfit users today have to write C++ inheriting from
  `PUserFcnBase`, compile against ROOT, manage `.so` paths. High
  friction.
- Mantid plugins are similar (C++ algorithm framework).
- Asymmetry's Python stack means user functions should be a
  one-file Python decorator at most.
- Asymmetry's composite-model expression syntax already lets users
  combine existing components. The missing piece is letting them
  contribute novel components.

## Prior art

- **musrfit:** C++ plugin via `PUserFcnBase` + ROOT plugin manager.
  Reference: `src/tests/userFcn/`.
- **WiMDA:** DLL with `stdcall` exports for muSR functions + a
  separate DLL convention for model functions
  (`src/MusrFunctionUnit.pas`).
- **Mantid:** algorithm/function plugin via the Mantid framework.
- **Asymmetry:** ❌ no formal user-plugin API; users currently
  edit the MODELS / COMPONENTS registries in-source.

## Proposed approach

A decorator-based plugin discovery mechanism:

```python
# User writes ~/.asymmetry/plugins/my_function.py

from asymmetry.plugin import register_component
import numpy as np

@register_component(
    name="MyCustomDecay",
    params=["A", "tau", "alpha"],
    defaults={"A": 25.0, "tau": 1.0, "alpha": 1.0},
    category="user",
)
def my_custom_decay(t, A, tau, alpha):
    return A * np.exp(-(t / tau) ** alpha)
```

On Asymmetry startup, the GUI scans `~/.asymmetry/plugins/`,
imports each module, and the `@register_component` decorator
adds entries to COMPONENTS. The Fit Function Builder picks them
up automatically under the "user" category.

## Out of scope

- Compile-time C++ plugins (Asymmetry is Python-first).
- Sandboxed / signed plugins for shared installations.
- Auto-discovery of plugins on a network filesystem (let users
  point at directories explicitly).
