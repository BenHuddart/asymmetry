"""Present captured scan executions without calling scientific optimizers."""

import numpy as np

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.fitting.scan_history import ScanFitHistory, ScanFitRecord

MISSING_RESULT = (
    "Saved fit settings have no captured numerical result. "
    "Use Fit baseline, Fit peaks, or Fit RF resonance explicitly to obtain a new result."
)


def fit_settings(view, kind):
    """Select the controls that determine this fit's applicability."""
    state = view.analysis_state()
    keys = ["x_key", "derivative"]
    keys += {
        "baseline": ["regions", "baseline_model"],
        "peaks": ["peaks"],
        "rf": ["rf_nu", "rf_a_mu", "rf_a_p"],
    }[kind]
    return {key: state[key] for key in keys}


def present_fit(window, record):
    """Derive readouts from retained results and draw the sampled prediction."""
    view = window._alc_scan_view
    fit = record.fit_result()
    x = np.asarray(record.support["x"], dtype=float)
    curve = np.asarray(record.prediction, dtype=float)
    if record.kind == "baseline":
        window._alc_baseline = record.baseline_result()
        view.show_baseline_overlay(x, curve)
        view.set_rf_results("")
        return
    if record.kind == "rf":
        a_mu = fit.parameters["A_mu"].value
        a_p = fit.parameters["A_p"].value
        summary = (
            f"A_µ = {a_mu:.2f} ± {fit.uncertainties.get('A_mu', 0.0):.2f} MHz, "
            f"A_p = {a_p:.2f} ± {fit.uncertainties.get('A_p', 0.0):.2f} MHz "
            f"(χ²/dof = {fit.reduced_chi_squared:.2g})"
        )
        window._alc_baseline = None
        view._baseline_curve = None
        view._peaks_results.setText("")
        view.set_rf_results(summary)
    else:
        model = ParameterCompositeModel.from_dict(record.model)
        names, covariance = record.covariance
        uncertainties = dict(zip(names, np.sqrt(np.diag(covariance)), strict=True))
        results, lines = [], []
        for i, component in enumerate(model.component_names):
            values = {
                key: fit.parameters[model.component_param_name(i, key)].value
                for key in ("f", "B0", "Bwid")
            }
            values["Bwid"] = abs(values["Bwid"])
            errors = {
                key: uncertainties[model.component_param_name(i, key)]
                for key in ("f", "B0", "Bwid")
            }
            factor = model.components[i].fwhm_factor
            assert factor is not None, "ALC peak models must define an FWHM factor"
            label = {"GaussianLCR": "Gaussian", "LorentzianLCR": "Lorentzian"}[component]
            results.append(values)
            lines.append(
                f"Peak {i + 1} ({label}): B₀ = {values['B0']:.4g} ± {errors['B0']:.2g} G, "
                f"FWHM = {factor * values['Bwid']:.4g} ± {factor * errors['Bwid']:.2g} G, "
                f"amp = {values['f']:.3g} ± {errors['f']:.2g} %"
            )
        lines.append("Uncertainties include the fitted baseline.")
        view.set_peak_results(results, "\n".join(lines))
        view.set_rf_results("")
    view.show_fit_overlay(x, curve)


def capture_fit(window, kind, scan, model, fit, prediction, **kwargs):
    """Record one explicit solve and use that same record for display."""
    from dataclasses import replace

    view = window._alc_scan_view
    request = fit_settings(view, kind)
    record = ScanFitRecord.capture(
        kind, scan, request, model, fit, prediction, request=request, **kwargs
    )
    present_fit(window, record)
    # The peak table uses fitted values as the next request's seeds.
    record = replace(record, settings=fit_settings(view, kind))
    view.fit_history.capture(record)
    view.set_fit_history_status(MISSING_RESULT if view.fit_history.missing else "")


def restore_fits(window, data):
    """Restore applicable captured history; missing numbers remain unavailable."""
    view = window._alc_scan_view
    history = ScanFitHistory.from_dict(data)
    view.fit_history = history
    scan = window._alc_display_scan(view.x_key())
    for kind in ("baseline", "peaks", "rf"):
        index = history.active.get(kind)
        if index is None:
            continue
        record = history.records[index]
        if (
            scan is None
            or not record.matches(scan, fit_settings(view, kind))
            or (kind == "peaks" and record.baseline_index != history.active.get("baseline"))
        ):
            history.invalidate(kind)
            continue
        present_fit(window, record)
    view.set_fit_history_status(MISSING_RESULT if history.missing else "")
