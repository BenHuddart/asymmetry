"""Spectrum export tables, with a captured frequency display context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asymmetry.core.transform.grouping import (
    grouping_balances,
)
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.gui.tasks import TaskCancelledError
from asymmetry.gui.utils.recipe_export import write_recipe_header

_DENSITY_COLUMN_NAME = {
    "mhz": "density_per_MHz",
    "gauss": "density_per_G",
    "tesla": "density_per_T",
}
_FREQUENCY_X_COLUMN_NAME = {
    "frequency_mhz": "frequency_MHz",
    "field_gauss": "field_G",
    "field_tesla": "field_T",
}
_FREQUENCY_SHIFT_X_COLUMN_NAME = {
    "frequency_mhz": "shift_MHz",
    "field_gauss": "shift_G",
    "field_tesla": "shift_T",
}


@dataclass(frozen=True)
class PlotExportData:
    cancelled: Callable[[], bool] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise TaskCancelledError

    frequency_unit: str = "frequency_mhz"
    axis_mode: str = "absolute"
    correlation: bool = False
    frequency_panel: bool = False
    reference_mhz: float | None = None
    x_label: str = "Frequency (MHz)"

    def _display_x_label(self):
        return self.x_label

    def _convert_frequency_axis_for_display(self, values):
        from asymmetry.core.fourier.units import convert as convert_field_unit
        from asymmetry.core.fourier.units import relative_shift_ppm

        array = np.asarray(values, dtype=float)
        if not self.frequency_panel or self.correlation:
            return array
        unit = {"field_gauss": "gauss", "field_tesla": "tesla"}.get(self.frequency_unit, "mhz")
        if (
            self.axis_mode == "relative_ppm"
            and self.reference_mhz is not None
            and self.reference_mhz > 0
        ):
            return np.asarray(relative_shift_ppm(array, self.reference_mhz))
        if self.axis_mode == "shift":
            array = array - (self.reference_mhz or 0.0)
        return np.asarray(convert_field_unit(array, "mhz", unit)) if unit != "mhz" else array

    def _write_data_file(
        self,
        dat_path: Path,
        payload: dict,
        *,
        label_text: object | None = None,
        x_range: tuple[float, float] | None = None,
        axis_key: str | None = None,
        bake_waterfall_offset: bool = True,
    ) -> None:
        """Write a .dat file with spectra data and metadata header.

        Row selection and baked display offsets are recorded with the captured
        effective recipe; GLE script offsets leave the written values unchanged.
        """
        data = payload.get("data") or {}
        t_data = data.get("t")
        y_data = data.get("y")
        y_err = data.get("err")
        if t_data is None or y_data is None:
            return

        # Only baked offsets change the written values.
        waterfall_offset = float(payload.get("waterfall_offset", 0.0))
        if waterfall_offset and bake_waterfall_offset:
            y_data = np.asarray(y_data, dtype=float) + waterfall_offset

        display_label = label_text if label_text is not None else payload.get("label", "dataset")
        run_metadata = (
            payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
        )
        histogram_info = (
            payload.get("histogram_info") if isinstance(payload.get("histogram_info"), dict) else {}
        )
        grouping = payload.get("grouping") if isinstance(payload.get("grouping"), dict) else {}

        def _fmt_float(value: object, decimals: int) -> str:
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return ""

        def _fmt_mev(value: object) -> str:
            try:
                return f"{float(value) / 1.0e6:.2f}"
            except (TypeError, ValueError):
                return ""

        def _safe_int(value: object) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        title = str(run_metadata.get("title", "") or "")
        comment = str(run_metadata.get("comment", "") or "")
        started = str(run_metadata.get("started", "") or "")
        stopped = str(run_metadata.get("stopped", "") or "")
        temperature = _fmt_float(run_metadata.get("temperature"), 3)
        temperature_label = str(
            run_metadata.get("temperature_label") or run_metadata.get("temperature_setpoint") or ""
        ).strip()
        field = _fmt_float(run_metadata.get("field"), 2)

        n_hist = _safe_int(histogram_info.get("count"))
        n_bins = _safe_int(histogram_info.get("n_bins"))
        bin_width_us = histogram_info.get("bin_width_us")
        bin_width_ps_text = ""
        bin_width_us_text = ""
        if bin_width_us is not None:
            bin_width_ps_text = _fmt_float(float(bin_width_us) * 1.0e6, 6)
            bin_width_us_text = _fmt_float(bin_width_us, 2)

        events_grouped = _fmt_mev(histogram_info.get("events_grouped"))
        events_raw = _fmt_mev(histogram_info.get("events_total"))

        groups_raw = grouping.get("groups")
        groups: dict[int, list[int]] = {}
        if isinstance(groups_raw, dict):
            for key, values in groups_raw.items():
                self.check_cancelled()
                gid = _safe_int(key)
                if gid is None or not isinstance(values, list):
                    continue
                detectors: list[int] = []
                for v in values:
                    self.check_cancelled()
                    vv = _safe_int(v)
                    if vv is not None:
                        detectors.append(vv)
                if detectors:
                    groups[gid] = sorted(set(detectors))

        # Backward-compatible fallback for payloads that store detectors as
        # direct forward/backward lists rather than a groups mapping.
        if not groups:
            f_list = grouping.get("forward")
            b_list = grouping.get("backward")
            if isinstance(f_list, list):
                f_det = [v for v in (_safe_int(x) for x in f_list) if v is not None]
                if f_det:
                    groups[1] = sorted(set(f_det))
            if isinstance(b_list, list):
                b_det = [v for v in (_safe_int(x) for x in b_list) if v is not None]
                if b_det:
                    groups[2] = sorted(set(b_det))

        t0_bins = (
            histogram_info.get("t0_bins") if isinstance(histogram_info.get("t0_bins"), list) else []
        )
        forward_group = _safe_int(grouping.get("forward_group"))
        backward_group = _safe_int(grouping.get("backward_group"))
        alpha = _fmt_float(grouping_balances(grouping)[0], 4)
        t0_bin = _safe_int(grouping.get("t0_bin"))
        if t0_bin is None:
            t0_bin = 0
        t_good_offset = _safe_int(grouping.get("t_good_offset"))
        first_good_bin = _safe_int(grouping.get("first_good_bin"))
        if t_good_offset is None and first_good_bin is not None:
            t_good_offset = first_good_bin - t0_bin
        last_good_bin = _safe_int(grouping.get("last_good_bin"))
        binning = binning_spec(grouping)
        bunching_factor = binning.factor
        deadtime_correction = bool(grouping.get("deadtime_correction", False))

        with open(dat_path, "w", encoding="utf-8") as f:
            f.write("! START OF RUN INFORMATION\n")
            write_recipe_header(
                f, payload, x_range=x_range, bake_waterfall_offset=bake_waterfall_offset
            )
            f.write(f"!  Run number  : {payload.get('run_number', '')}\n")
            if title:
                f.write(f"!  Title       : {title}\n")
            if temperature:
                if temperature_label:
                    f.write(f"!  Temperature : {temperature} K (label {temperature_label})\n")
                else:
                    f.write(f"!  Temperature : {temperature} K\n")
            if field:
                f.write(f"!  Field       : {field} G\n")
            if comment:
                f.write(f"!  Comment     : {comment}\n")
            if started:
                f.write(f"!  Started     : {started}\n")
            if stopped:
                f.write(f"!  Stopped     : {stopped}\n")
            if (
                n_hist is not None
                and n_bins is not None
                and bin_width_ps_text
                and bin_width_us_text
            ):
                f.write(
                    "!  Histograms  : "
                    f"{n_hist} ({n_bins} bins of {bin_width_ps_text} ps = {bin_width_us_text} us)\n"
                )
            if events_grouped and events_raw:
                f.write(
                    f"!  Events      : {events_grouped} MEv grouped in range (raw = {events_raw})\n"
                )
            elif events_raw:
                f.write(f"!  Events      : {events_raw} MEv\n")
            f.write("! END OF RUN INFORMATION\n")
            f.write("!\n")

            f.write("! START OF GROUPING INFORMATION\n")
            for gid in sorted(groups):
                self.check_cancelled()
                det_tokens: list[str] = []
                for det in groups[gid]:
                    self.check_cancelled()
                    idx = det - 1
                    if 0 <= idx < len(t0_bins):
                        det_tokens.append(f"{det:02d}({int(t0_bins[idx])})")
                    else:
                        det_tokens.append(f"{det:02d}")
                f.write(f"!  Group#{gid:02d}  Hist(t0): {', '.join(det_tokens)}\n")

            if forward_group is not None or backward_group is not None:
                fwd_text = "" if forward_group is None else str(forward_group)
                bwd_text = "" if backward_group is None else str(backward_group)
                f.write(
                    f"!  Forward Group = {fwd_text}, Backward Group = {bwd_text}, Alpha = {alpha}\n"
                )
            elif isinstance(grouping.get("forward"), list) or isinstance(
                grouping.get("backward"), list
            ):
                f.write(f"!  Forward Group = forward, Backward Group = backward, Alpha = {alpha}\n")
            if t_good_offset is not None or last_good_bin is not None:
                fg_text = "" if t_good_offset is None else str(t_good_offset)
                lg_text = "" if last_good_bin is None else str(last_good_bin)
                f.write(f"!  Offset to first good bin = {fg_text}, Last good bin = {lg_text}\n")
            binning_mode = binning.mode
            if binning_mode == "variable":
                f.write(
                    f"!  Variable binning, initial bin = {binning.bin0_us} us, "
                    f"bin at 10 us = {binning.bin10_us} us\n"
                )
            elif binning_mode == "constant_error":
                f.write(f"!  Constant-error binning, initial bin = {binning.bin0_us} us\n")
            elif bunching_factor is not None:
                f.write(f"!  Fixed binning, bunching factor = {bunching_factor}\n")
            f.write(f"!  Deadtime correction {'on' if deadtime_correction else 'off'}\n")
            f.write("! END OF GROUPING INFORMATION\n")
            f.write("!\n")

            if payload.get("domain") == "frequency":
                self._write_frequency_data_body(
                    f, payload, display_label=display_label, x_range=x_range
                )
                return

            if axis_key in {"P_x", "P_y", "P_z"}:
                suffix = axis_key.split("_", 1)[1]
                datarow_label = f"a_0 P_{suffix}(t)"
                ylabel = f"a_0 P_{suffix}(t) (%)"
            else:
                datarow_label = "FB asymmetry"
                ylabel = "% Asymmetry"

            f.write("! START OF DATA SET INFORMATION\n")
            f.write(f"!  Datarow: (Time) ({datarow_label}) (Error)\n")
            if axis_key in {"P_x", "P_y", "P_z"}:
                f.write(f"!  Projection: {axis_key}\n")
            f.write(f"!  Title: {display_label}\n")
            f.write("!  Xlabel: Time in microseconds\n")
            f.write(f"!  Ylabel: {ylabel}\n")
            if waterfall_offset:
                f.write(f"!  waterfall offset: {waterfall_offset:.10g}\n")
            f.write("! END OF DATA SET INFORMATION\n")
            f.write("! time  asymmetry  error\n")
            err_arr = y_err if y_err is not None else np.zeros_like(y_data)
            lo, hi = (None, None) if x_range is None else (min(x_range), max(x_range))
            for t_val, y_val, e_val in zip(t_data, y_data, err_arr):
                self.check_cancelled()
                tf = float(t_val)
                if lo is not None and (tf < lo or tf > hi):
                    continue
                f.write(f"{tf:.10g} {float(y_val):.10g} {float(e_val):.10g}\n")

    def _write_fit_file(
        self,
        fit_path: Path,
        payload: dict,
        *,
        x_range: tuple[float, float] | None = None,
        bake_waterfall_offset: bool = True,
    ) -> None:
        """Write a .fit file with fit-curve data and metadata header.

        ``x_range`` optionally restricts the written rows to ``[lo, hi]``;
        ``None`` (the GLE-export default) writes the whole curve.

        ``bake_waterfall_offset`` adds the payload's waterfall offset to the
        written curve. The plain-text export leaves it ``True`` (the text is the
        final artifact, so it must carry the stack). The GLE export passes
        ``False`` when the installed gleplot applies the offset in the script
        (``let``) instead, keeping the sidecar values raw. The header records
        the offset either way.
        """
        fit = payload.get("fit") or {}
        t_fit = fit.get("t")
        y_fit = fit.get("y")
        if t_fit is None or y_fit is None:
            return

        # Match the drawn fit curve's waterfall offset (see _write_data_file).
        waterfall_offset = float(payload.get("waterfall_offset", 0.0))
        if waterfall_offset and bake_waterfall_offset:
            y_fit = np.asarray(y_fit, dtype=float) + waterfall_offset

        meta = payload.get("fit_metadata") or {}
        with open(fit_path, "w", encoding="utf-8") as f:
            f.write(f"! Fit curve for {payload.get('label', 'dataset')}\n")
            f.write(f"! run_number: {payload.get('run_number', '')}\n")
            if waterfall_offset:
                f.write(f"! waterfall offset: {waterfall_offset:.10g}\n")
            fit_function = meta.get("fit_function") or fit.get("label") or "Fit"
            f.write(f"! fit_function: {fit_function}\n")
            chi2 = meta.get("chi_squared")
            red_chi2 = meta.get("reduced_chi_squared")
            if chi2 is not None:
                f.write(f"! chi_squared: {chi2:.8g}\n")
            if red_chi2 is not None:
                f.write(f"! reduced_chi_squared: {red_chi2:.8g}\n")
            params = meta.get("parameters")
            if params:
                f.write("! fitted_parameters:\n")
                for p in params:
                    self.check_cancelled()
                    err = p.get("error", float("nan"))
                    if np.isfinite(err):
                        f.write(f"!   {p['name']} = {p['value']:.8g} +/- {err:.4g}\n")
                    else:
                        f.write(f"!   {p['name']} = {p['value']:.8g}\n")
            f.write("!\n")
            if payload.get("domain") == "frequency":
                x_unit = str(payload.get("x_unit", "frequency_mhz"))
                x_column = self._frequency_x_column_name(
                    x_unit,
                    str(payload.get("axis_mode", "absolute")),
                    bool(payload.get("is_correlation")),
                )
                x_display = fit.get("x_display")
                x_fit = (
                    np.asarray(x_display, dtype=float)
                    if x_display is not None
                    else self._convert_frequency_axis_for_display(t_fit)
                )
                density_unit = payload.get("density_unit")
                y_fit_column = (
                    f"{_DENSITY_COLUMN_NAME.get(str(density_unit), 'density_per_MHz')}_fit"
                    if density_unit is not None
                    else "amplitude_fit"
                )
                f.write(f"! {x_column}  {y_fit_column}\n")
            else:
                x_fit = t_fit
                f.write("! time  asymmetry_fit\n")
            lo, hi = (None, None) if x_range is None else (min(x_range), max(x_range))
            for t_val, y_val in zip(x_fit, y_fit):
                self.check_cancelled()
                tf = float(t_val)
                if lo is not None and (tf < lo or tf > hi):
                    continue
                f.write(f"{tf:.10g} {float(y_val):.10g}\n")

    def _write_frequency_data_body(
        self,
        f,
        payload: dict,
        *,
        display_label: object,
        x_range: tuple[float, float] | None,
    ) -> None:
        """Write the frequency-view DATA SET / FOURIER header and data rows.

        Columns are ``c1`` = x in the current display unit, ``c2`` = amplitude,
        ``c3`` = error, and — only when the display unit differs from canonical
        MHz — ``c4`` = the canonical ``frequency_MHz`` axis, so the file stays
        self-describing while the GLE script's ``d1=c1,c2`` mapping (plotted x,
        amplitude) remains valid. The x-range filter is applied to the
        display-unit column, matching the on-screen window and the text
        export's control-value conversion.
        """
        data = payload.get("data") or {}
        t_data = data.get("t")
        y_data = data.get("y")
        y_err = data.get("err")
        if t_data is None or y_data is None:
            return

        mhz_arr = np.asarray(t_data, dtype=float)
        x_display = payload.get("x_display")
        x_disp_arr = (
            np.asarray(x_display, dtype=float)
            if x_display is not None
            else self._convert_frequency_axis_for_display(t_data)
        )

        x_unit = str(payload.get("x_unit", "frequency_mhz"))
        axis_mode = str(payload.get("axis_mode", "absolute"))
        is_correlation = bool(payload.get("is_correlation"))
        x_column = self._frequency_x_column_name(x_unit, axis_mode, is_correlation)
        # Shift/ppm columns are not self-describing on their own (they need the
        # reference), so always keep the canonical frequency_MHz column beside
        # them; absolute field columns keep it only when the unit differs from MHz.
        if is_correlation:
            include_mhz = False
        elif axis_mode in ("shift", "relative_ppm"):
            include_mhz = True
        else:
            include_mhz = x_unit in ("field_gauss", "field_tesla")

        x_label = str(payload.get("x_label") or self._display_x_label())
        y_label = str(payload.get("y_label") or "FFT (%)")
        # A unit-area field distribution is a density in the displayed x unit
        # (the payload carries the exported Jacobian's unit), so name its column
        # distinctly from an amplitude spectrum. The label sniff is a fallback
        # for payloads without the explicit key.
        density_unit = payload.get("density_unit")
        if density_unit is None and "1/mhz" in y_label.lower():
            density_unit = "mhz"
        amplitude_column = (
            _DENSITY_COLUMN_NAME.get(str(density_unit), "density_per_MHz")
            if density_unit is not None
            else "amplitude"
        )

        f.write("! START OF DATA SET INFORMATION\n")
        f.write(f"!  Datarow: ({x_label}) (Amplitude) (Error)\n")
        f.write(f"!  Title: {display_label}\n")
        f.write(f"!  Xlabel: {x_label}\n")
        f.write(f"!  Ylabel: {y_label}\n")
        f.write("! END OF DATA SET INFORMATION\n")
        f.write("!\n")

        f.write("! START OF FOURIER INFORMATION\n")
        fourier_metadata = payload.get("fourier_metadata")
        if isinstance(fourier_metadata, dict):
            for key in sorted(fourier_metadata):
                self.check_cancelled()
                f.write(f"!  {key}: {fourier_metadata[key]}\n")
        f.write(f"!  Display mode: {y_label}\n")
        f.write(f"!  X unit: {x_unit}\n")
        f.write(f"!  Axis mode: {axis_mode}\n")
        if density_unit is not None:
            f.write(
                f"!  Density unit: {_DENSITY_COLUMN_NAME.get(str(density_unit), str(density_unit))}\n"
            )
        reference_gauss = payload.get("reference_field_gauss")
        if reference_gauss is not None:
            f.write(f"!  Reference field: {float(reference_gauss):.4g} G\n")
        f.write("! END OF FOURIER INFORMATION\n")

        column_names = [x_column, amplitude_column, "error"]
        if include_mhz:
            column_names.append("frequency_MHz")
        f.write("! " + "  ".join(column_names) + "\n")

        err_arr = y_err if y_err is not None else np.zeros_like(y_data)
        lo, hi = (None, None) if x_range is None else (min(x_range), max(x_range))
        for x_val, y_val, e_val, mhz_val in zip(x_disp_arr, y_data, err_arr, mhz_arr):
            self.check_cancelled()
            xf = float(x_val)
            if lo is not None and (xf < lo or xf > hi):
                continue
            if include_mhz:
                f.write(
                    f"{xf:.10g} {float(y_val):.10g} {float(e_val):.10g} {float(mhz_val):.10g}\n"
                )
            else:
                f.write(f"{xf:.10g} {float(y_val):.10g} {float(e_val):.10g}\n")

    @staticmethod
    def _frequency_x_column_name(x_unit: str, axis_mode: str, is_correlation: bool) -> str:
        """Return the self-describing ``.dat`` x-column name for the display mode."""
        if is_correlation:
            return "coupling_MHz"
        if axis_mode == "relative_ppm":
            return "relative_shift_ppm"
        if axis_mode == "shift":
            return _FREQUENCY_SHIFT_X_COLUMN_NAME.get(x_unit, "shift_MHz")
        return _FREQUENCY_X_COLUMN_NAME.get(x_unit, "frequency_MHz")
