# LEM depth profiling: comparison

| Aspect | PSI / Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| TrimSP integration | ✅ | ◐ external | ❌ | ❌ |
| Depth-axis parameter trending | ◐ workspace groups | ◐ via `msr2data` | ❌ | ◐ generic trending |
| Implant-energy → depth conversion | ✅ | external | ❌ | ❌ |

## Implementation

Two pieces:

1. **Depth conversion** — a small numeric helper that takes
   implantation energy + sample density + composition and
   returns mean depth and depth spread. Standard tables for
   common substrates (Si, SiO₂, Y BaCu₃O₇, ...) can be bundled.

2. **Depth-axis trending** — a small extension to the
   parameter-trending panel allowing depth as an x-axis key
   alongside temperature and field.

The TrimSP integration is the larger / harder piece; defer.
