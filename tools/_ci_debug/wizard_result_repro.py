"""Temporary CI probe for the global_fit_wizard_result flake (removed before merge)."""

import sys
import time

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

import asymmetry.core.fitting.global_fit_wizard as gfw  # noqa: E402
from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass  # noqa: E402
from asymmetry.core.fitting.composite import COMPONENTS  # noqa: E402
from asymmetry.core.fitting.global_search.trend_objective import SelectionObjective  # noqa: E402
from asymmetry.core.fitting.wizard_scope import WizardScope  # noqa: E402
from docs.screenshots.data import make_ag_lf_decoupling  # noqa: E402

_orig = gfw._fit_separable_assignment


def _traced(datasets, state, local_names, **kw):
    out = _orig(datasets, state, local_names, **kw)
    chi2 = sum(r.chi_squared for r in out.fit_results_by_run.values())
    print(
        f"  STEP local={local_names} ok={out.is_successful} chi2={chi2:.6f} "
        f"ic={out.metric_value(gfw.SelectionMetric.AICC):.6f}",
        flush=True,
    )
    return out


gfw._fit_separable_assignment = _traced


def main() -> None:
    not_lf = {n for n, d in COMPONENTS.items() if FieldGeometry.LF not in d.field_geometries}
    exclude = frozenset(not_lf) | frozenset(
        {
            "StaticGKT_ZF", "DynamicGaussianKT", "DynamicLorentzianKT", "GaussianBroadenedKT",
            "ExponentialRelaxation", "GaussianRelaxation", "StretchedExponential", "RischKehr",
            "MuoniumLF", "Oscillatory",
        }
    )
    scope = WizardScope(
        physics=frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM}),
        exclude_components=exclude,
    )
    ds = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
    print("data digest", float(np.sum([d.asymmetry.sum() for d in ds])).hex())
    inst: dict = {}
    t = time.monotonic()
    rec = gfw.build_global_fit_wizard_recommendation(
        ds,
        scope=scope,
        selected_template_keys=("lf_kt_constant",),
        objective=SelectionObjective.STATISTICAL,
        progress_callback=lambda m: print("  LOG", m, flush=True),
        instrumentation=inst,
    )
    print("elapsed", round(time.monotonic() - t, 2))
    for a in rec.sorted_optimized_assessments():
        print("  RANK", a.global_param_names, a.local_param_names, a.metric_value(rec.metric))
    print("  COUNTERS", inst.get("counters"))
    ok = any(
        set(a.local_param_names) == {"Delta", "B_L"} for a in rec.sorted_optimized_assessments()
    )
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
