"""Temporary CI probe: per-run all-local anchor values and surrogate scores."""

import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

import asymmetry.core.fitting.global_fit_wizard as gfw  # noqa: E402
from asymmetry.core.fitting.global_search import surrogate  # noqa: E402

_orig_est = gfw._run_estimates_from_results


def _traced_estimates(datasets, results_by_run, free_param_names):
    out = _orig_est(datasets, results_by_run, free_param_names)
    for est in out:
        print(
            f"  EST run={est.run_number} chi2={est.chi_squared:.6f} "
            + " ".join(
                f"{n}={v:.9g}±{u:.3g}" for n, v, u in zip(est.names, est.values, est.uncertainties)
            )
            + f" at_bound={sorted(est.at_bound)}",
            flush=True,
        )
        res = results_by_run[est.run_number]
        print(
            "      bounds "
            + " ".join(f"{p.name}[{p.min:g},{p.max:g}]" for p in res.parameters),
            flush=True,
        )
    for name in free_param_names:
        print(
            f"  SURR share={name} ic={surrogate.surrogate_ic(out, (name,), gfw.SelectionMetric.AICC):.6f}",
            flush=True,
        )
    return out


gfw._run_estimates_from_results = _traced_estimates

sys.argv = sys.argv[:1]
import runpy  # noqa: E402

runpy.run_path("tools/_ci_debug/wizard_result_repro.py", run_name="__main__")
_ = np
