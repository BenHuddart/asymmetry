"""Worker-safe GLE figure generation from captured spectrum payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from asymmetry.gui.utils.connector_geometry import ConnectorGeometry
from asymmetry.gui.utils.gle_export import (
    GleExportBuild,
    dedup_export_token,
    safe_file_token,
)
from asymmetry.gui.utils.gle_figure import gle_figure, gle_subplots, save_gle_figure
from asymmetry.gui.utils.plot_export_data import PlotExportData, fit_sidecar_stems


@dataclass(frozen=True)
class PlotGleSnapshot(PlotExportData):
    axes: tuple = ()
    x_limits: tuple[float, float] = (0.0, 10.0)
    y_limits: dict = field(default_factory=dict)
    figsize: tuple[float, float] = (6.4, 4.8)
    stacked: bool = False
    connect_points: bool = field(default=False, kw_only=True)

    def build(self, glp, gle_path, export_dir, *, cancelled=None, output_format="pdf"):
        snapshot = replace(self, cancelled=cancelled)
        snapshot.check_cancelled()
        return snapshot.build_figure(glp, gle_path, export_dir, output_format=output_format)

    def build_figure(self, glp, gle_path, export_dir, *, output_format="pdf"):
        if self.stacked:
            fig, subplot_axes = gle_subplots(
                glp, nrows=len(self.axes), ncols=1, figsize=self.figsize, sharex=True
            )
        else:
            fig = gle_figure(glp, figsize=self.figsize)
            subplot_axes = [fig.add_subplot(111)]
        if self.connect_points:
            fig.style.dashlen = 0.02
        files = [gle_path]
        dat_writes = []
        tokens = set()
        colors = [
            "black",
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "brown",
            "gray",
        ]
        last_axis = None
        last_payload = None
        for index, (axis_key, payloads, title) in enumerate(self.axes):
            ax = subplot_axes[index]
            if not payloads:
                continue
            self._plot_export_payloads_on_axis(
                ax,
                payloads,
                axis_key=axis_key,
                written_files=files,
                dat_writes=dat_writes,
                gle_path=gle_path,
                colors=colors,
                show_legend=(
                    len(payloads) > 1
                    or any(
                        len(p["fits"]) > 1 or (self.connect_points and p["fits"]) for p in payloads
                    )
                )
                and index == 0,
                used_tokens=tokens,
            )
            if title:
                ax.set_title(self._sanitize_gle_text(title))
            last_axis, last_payload = ax, payloads[0]
        if last_axis is not None:
            last_axis.set_xlabel(
                self._sanitize_gle_text(
                    self._strip_matplotlib_math(last_payload.get("x_label") or self.x_label)
                )
                if self.frequency_panel
                else "Time (µs)"
            )
        if self.stacked:
            fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.12, hspace=0.08)
        # Independent frequency bands can overlap even on offset curves.
        preserve_alpha = any(
            sum(
                p.get("domain") == "frequency"
                and p.get("data", {}).get("err") is not None
                and bool(np.any(np.isfinite(p["data"]["err"]) & (np.asarray(p["data"]["err"]) > 0)))
                for p in payloads
            )
            > 1
            for _, payloads, _ in self.axes
        )
        cairo = save_gle_figure(
            fig, gle_path, output_format=output_format, preserve_alpha=preserve_alpha
        )
        for path, payload, label, axis_key in dat_writes:
            self._write_data_file(
                path,
                payload,
                label_text=label,
                axis_key=axis_key,
                bake_waterfall_offset=False,
            )
        return GleExportBuild(files=files, cairo=cairo)

    def _plot_export_payloads_on_axis(
        self,
        ax,
        payloads: list[dict],
        *,
        axis_key: str | None,
        written_files: list[Path],
        dat_writes: list[tuple[Path, dict, object, str | None]],
        gle_path: Path,
        colors: list[str],
        show_legend: bool,
        used_tokens: set[str] | None = None,
    ) -> None:
        """Draw export payloads on a provided axis with axis-specific naming."""
        is_multi = len(payloads) > 1
        axis_suffix = self._export_axis_suffix(axis_key)
        for i, payload in enumerate(payloads):
            label_text = payload.get("label", f"dataset_{i}")
            safe_label = self._sanitize_gle_text(
                label_text,
                fallback=f"Run {payload.get('run_number', i)}",
            )
            # Writer and GLE references share ASCII lowercase stems before deduplication.
            # A letter-led token keeps unquoted GLE data references unambiguous.
            raw_token = safe_file_token(f"{safe_label}_{axis_suffix}").lower()
            if raw_token[:1].isdigit():
                raw_token = f"run_{raw_token}"
            token = dedup_export_token(raw_token, used_tokens)
            data_color = colors[i % len(colors)] if is_multi else "black"
            suppress_subplot_labels = axis_key in {"P_x", "P_y", "P_z"} and not show_legend
            data_label = None if suppress_subplot_labels else safe_label

            data = payload.get("data") or {}
            t_data = data.get("t")
            y_data = data.get("y")
            y_err = data.get("err")
            # Script offsets preserve raw scientific values in the sidecars.
            gle_offset = float(payload.get("waterfall_offset", 0.0))

            dat_path = gle_path.parent / f"{token}.dat"
            if t_data is not None and y_data is not None:
                dat_writes.append((dat_path, payload, label_text, axis_key))
                written_files.append(dat_path)
                if payload.get("domain") == "frequency":
                    # Mirror the on-screen renderer (_plot_frequency_line_masked):
                    # a solid line in display-unit x, plus a shaded ±1σ band
                    # when the error array carries any positive finite value.
                    x_display = payload.get("x_display")
                    x_plot = (
                        np.asarray(x_display, dtype=float)
                        if x_display is not None
                        else np.asarray(t_data, dtype=float)
                    )
                    ax.plot(
                        x_plot,
                        y_data,
                        "-",
                        color=data_color,
                        linewidth=1.2,
                        label=data_label,
                        offset=gle_offset,
                        data_name=token,
                    )
                    if y_err is not None:
                        err = np.asarray(y_err, dtype=float)
                        if bool(np.any(np.isfinite(err) & (err > 0.0))):
                            band_err = np.where(np.isfinite(err), err, 0.0)
                            y_arr = np.asarray(y_data, dtype=float)
                            ax.fill_between(
                                x_plot,
                                y_arr - band_err,
                                y_arr + band_err,
                                color=data_color,
                                alpha=0.25,
                                label=None,
                                offset=gle_offset,
                                data_name=f"{token}_band",
                            )
                else:
                    if self.connect_points:
                        self._plot_connected_time_data(
                            ax, payload, token, data_color, data_label, gle_offset
                        )
                    else:
                        ax.errorbar(
                            t_data,
                            y_data,
                            yerr=y_err,
                            linestyle="none",
                            marker="o",
                            color=data_color,
                            markersize=4,
                            capsize=2,
                            label=data_label,
                            offset=gle_offset,
                            data_name=token,
                        )

            fits = payload["fits"]
            for order, (stem, fit) in enumerate(fit_sidecar_stems(token, fits)):
                fit_path = gle_path.parent / f"{stem}.fit"
                self._write_fit_file(fit_path, payload, fit, bake_waterfall_offset=False)
                written_files.append(fit_path)

                fit_label = fit["label"]
                if suppress_subplot_labels:
                    fit_label = None
                elif self.connect_points or (is_multi and order > 0):
                    identity = f"{safe_label} - {fit_label}" if is_multi else fit_label
                    fit_label = self._sanitize_gle_text(f"Fit: {identity}")
                elif is_multi:
                    fit_label = None
                else:
                    fit_label = self._sanitize_gle_text(fit_label, fallback="Fit")
                fit_color = (
                    (data_color if is_multi else "red")
                    if order == 0
                    else colors[(i + order + 1) % len(colors)]
                )
                ax.plot(
                    fit.get("x_display", fit["t"]),
                    fit["y"],
                    color=fit_color,
                    linewidth=2.0 if self.connect_points else 1.6,
                    zorder=2,
                    label=fit_label,
                    offset=gle_offset,
                    data_name=f"{stem}_fit",
                )

        annotations = payloads[0].get("annotations") or []
        for ann in annotations:
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            text = str(ann.get("text", "")).strip()
            text = self._sanitize_gle_text(text)
            if text:
                ax.text(x, y, text, color="black", ha="left")

        frequency_payload = next((p for p in payloads if p.get("domain") == "frequency"), None)
        if frequency_payload is not None:
            ax.set_ylabel(
                self._sanitize_gle_text(
                    self._strip_matplotlib_math(frequency_payload.get("y_label")),
                    fallback="FFT (%)",
                )
            )
        elif "group_id" in payloads[0]:
            ax.set_ylabel(self._sanitize_gle_text(payloads[0]["y_label"]))
        else:
            ax.set_ylabel(self._export_axis_ylabel(axis_key))
        ax.set_xlim(*self.x_limits)
        ax.set_ylim(*self.y_limits[axis_key])

        if show_legend:
            if self.connect_points:
                ax.legend(loc="best", compact=True)
            else:
                ax.legend(loc="best")

    def _plot_connected_time_data(self, ax, payload, token, color, label, offset):
        """Keep source tables separate from gapped, confidence-coloured display paths."""
        data = payload["data"]
        time, values = np.asarray(data["t"]), np.asarray(data["y"])
        error = np.zeros_like(values) if data["err"] is None else np.asarray(data["err"])
        low = np.asarray(payload.get("connector_low_mask", np.zeros_like(values, dtype=bool)))
        for mask, colour in [(~low, color), (low, "gray")]:
            geometry = ConnectorGeometry(time, values, error, mask)
            vertices = geometry.vertices(self.x_limits, None)
            if not vertices.size:
                continue
            # The labelled line carries the combined key glyph; opaque observations
            # are drawn above fits. GLE's line API has no independent alpha channel.
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
                "--",
                color=colour,
                linewidth=0.65,
                marker="o",
                markersize=4,
                label=label,
                offset=offset,
                zorder=1.6,
                data_name=f"{token}_guide_{colour}",
            )
            label = None
            indices = geometry.indices
            ax.errorbar(
                time[indices],
                values[indices],
                yerr=error[indices],
                linestyle="none",
                marker="o",
                color=colour,
                markersize=4,
                capsize=2,
                label=None,
                offset=offset,
                zorder=3,
                data_name=f"{token}_points_{colour}",
            )

    @staticmethod
    def _sanitize_gle_text(value: object, *, fallback: str = "") -> str:
        """Return text that is safe for GLE string rendering."""
        text = str(value)
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.replace("μ", "u").replace("µ", "u")
        text = text.replace("χ", "chi").replace("²", "^{2}")
        text = "".join(ch for ch in text if ch.isprintable())
        text = " ".join(text.split())
        text = text.encode("ascii", "ignore").decode("ascii")
        return text or fallback

    @staticmethod
    def _strip_matplotlib_math(value: object) -> str:
        """Strip matplotlib math markup so an axis label renders as plain text.

        Frequency y-labels can carry matplotlib mathtext (e.g. the
        ``$|F|$ (arb.)`` fallback). GLE has no use for the ``$…$`` delimiters,
        ``\\mathrm``-style commands, or their braces, so they are removed here —
        locally, on the frequency labels only — rather than by widening the
        shared :meth:`_sanitize_gle_text` (which the byte-identical time-domain
        export also flows through).
        """
        text = str(value)
        text = text.replace("$", "")
        text = re.sub(r"\\(?:mathrm|mathit|mathsf|mathbf|text|rm|it|bf)\b", "", text)
        text = text.replace("{", "").replace("}", "")
        return text

    def _export_axis_suffix(self, axis_key: str | None) -> str:
        """Return filename-safe suffix for per-axis exports."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            return axis_key.lower().replace("_", "")
        return "main"

    def _export_axis_ylabel(self, axis_key: str | None) -> str:
        """Return plain-text y-axis label for plot exports."""
        if axis_key in {"P_x", "P_y", "P_z"}:
            suffix = axis_key.split("_", 1)[1]
            return f"a_{{0}} P_{{{suffix}}}(t) (%)"
        return "Asymmetry (%)"
