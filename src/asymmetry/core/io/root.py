"""Read documented MusrRoot and LEM RunHeader/hDecay objects using uproot."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core import computation
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.base import BaseLoader, comment_field_gauss, field_direction_from_text
from asymmetry.core.io.physical_units import (
    CANONICAL_UNITS,
    Dimension,
    UnitDeclaration,
    canonical,
    parse_quantity,
    quantity_parts,
)
from asymmetry.core.io.quick_asymmetry import quick_asymmetry
from asymmetry.core.io.root_header import (
    LEM_HEADER_BANKS,
    LEM_HEADER_UNITS,
    parse_header_value,
    parse_musrroot_string,
    read_directory_header,
    read_lem_header,
)
from asymmetry.core.transform.fb_counts import validated_count_arrays
from asymmetry.core.transform.reference_scale import ReferenceExposure
from asymmetry.core.utils.perf import perf_timer

_PHYSICAL_HEADER_FIELDS: dict[str, Dimension] = {
    "Time Resolution": "time",
    "Sample Magnetic Field": "field",
    "Sample Temperature": "temperature",
}


@dataclass
class _RootHistogram:
    histo_number: int
    counts: np.ndarray
    title: str


@dataclass
class _RootSlowControlLogs:
    source_file: str
    channels: list[str]
    time_series: dict[str, dict[str, Any]]


class RootLoader(BaseLoader):
    """Read MusrRoot and PSI LEM ROOT files."""

    extensions = [".root"]
    format_name = "MusrRoot / LEM ROOT (.root)"

    def load(self, filepath: str) -> MuonDataset:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            import uproot
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "Loading ROOT files requires the optional 'root' extra: "
                "install with asymmetry[root]."
            ) from exc

        with perf_timer("core.load.root", filename=path.name) as perf:
            with uproot.open(path) as root_file:
                header, header_kind, run_summary = self._read_header(root_file)
                root_histograms = self._read_histograms(root_file)
                conventions = LEM_HEADER_UNITS.get(header.get("RunInfo/Header Class"), {})
                quantities = {
                    name: parse_quantity(
                        header.get(f"RunInfo/{name}"),
                        dimension,
                        context=f"{path}: RunInfo/{name}",
                        convention=conventions.get(name),
                        default=None if dimension == "time" else 0.0,
                    )
                    for name, dimension in _PHYSICAL_HEADER_FIELDS.items()
                }
                slow_control_logs = self._read_slow_control_logs(
                    root_file, path, header, quantities["Sample Temperature"][0]
                )

            if not root_histograms:
                raise ValueError(f"ROOT file does not contain hDecay histograms: {filepath}")
            dataset = self._build_dataset(
                path,
                header,
                header_kind,
                root_histograms,
                quantities,
                slow_control_logs,
                run_summary,
            )
            perf.detail(detectors=len(root_histograms), bins=dataset.n_points)
            return dataset

    def _read_header(self, root_file) -> tuple[dict[str, str], str, str | None]:
        if "RunHeader" in root_file:
            run_header = root_file["RunHeader"]
            if self._is_directory(run_header):
                header, run_summary = read_directory_header(run_header)
                return header, "musr-root-directory", run_summary
            return self._read_folder_header(run_header), "musr-root-folder", None

        if "RunInfo" in root_file:
            run_info = root_file["RunInfo"]
            if self._is_directory(run_info):
                header = self._read_directory_tree(root_file, "RunInfo")
            else:
                header = self._read_folder_section(run_info, "RunInfo")
            kind = header.get("RunInfo/Header Class", "").replace("TLemRunHeader v", "lem-root-v")
            return header, kind or "lem-root-folder", None

        raise ValueError("ROOT file does not contain a RunHeader or RunInfo header")

    def _read_directory_tree(self, root_file, prefix: str) -> dict[str, str]:
        header: dict[str, str] = {}
        for key in root_file.keys(recursive=True):
            path = self._clean_key(key)
            if not path.startswith(f"{prefix}/"):
                continue
            try:
                obj = root_file[key]
            except Exception:
                continue
            if self._is_directory(obj):
                continue
            value = self._object_string(obj)
            if value is not None:
                header[path] = parse_header_value(value)
        return header

    def _read_folder_header(self, run_header) -> dict[str, str]:
        header: dict[str, str] = {}
        for section in self._children(run_header):
            section_name = self._object_name(section)
            if not section_name:
                continue
            header.update(self._read_folder_section(section, section_name))
        return header

    def _read_folder_section(self, section, prefix: str) -> dict[str, str]:
        header: dict[str, str] = {}
        for item in self._children(section):
            item_class = self._object_class(item)
            if item_class == "TLemRunHeader" and item.class_version in LEM_HEADER_BANKS:
                header.update(read_lem_header(item))
                continue
            if item_class == "TObjString":
                parsed = parse_musrroot_string(str(item))
                if parsed is not None:
                    key, value = parsed
                    header[f"{prefix}/{key}"] = value
                continue

            name = self._object_name(item)
            if not name:
                continue
            child_prefix = f"{prefix}/{name}"
            header.update(self._read_folder_section(item, child_prefix))
        return header

    def _read_histograms(self, root_file) -> list[_RootHistogram]:
        histograms: dict[int, _RootHistogram] = {}

        for key in root_file.keys(recursive=True):
            clean = self._clean_key(key)
            match = re.search(r"(?:^|/)hDecay(\d+)$", clean)
            if match is None:
                continue
            try:
                obj = root_file[key]
            except Exception:
                continue
            if self._is_histogram(obj):
                histograms[int(match.group(1))] = self._root_histogram(match, obj)

        if histograms:
            return [histograms[number] for number in sorted(histograms)]

        if "histos" in root_file:
            for obj in self._walk_objects(root_file["histos"]):
                match = re.fullmatch(r"hDecay(\d+)", self._object_name(obj))
                if match is None or not self._is_histogram(obj):
                    continue
                histograms[int(match.group(1))] = self._root_histogram(match, obj)

        return [histograms[number] for number in sorted(histograms)]

    def _read_slow_control_logs(
        self,
        root_file,
        path: Path,
        header: dict[str, str],
        temperature_kelvin: float,
    ) -> _RootSlowControlLogs | None:
        """Read MusrRoot slow-control histograms from ``SCAnaModule``."""
        time_series: dict[str, dict[str, Any]] = {}
        seen_paths: set[str] = set()
        sensor_roles = self._slow_control_sensor_roles(header)

        for key in root_file.keys(recursive=True):
            clean = self._clean_key(key)
            if "SCAnaModule" not in clean:
                continue
            try:
                obj = root_file[key]
            except Exception:
                continue
            series = self._slow_control_series_from_histogram(
                clean,
                obj,
                str(path),
                sensor_roles,
            )
            if series is None:
                continue
            series_path, info = series
            time_series[series_path] = info
            seen_paths.add(series_path)

        if "histos" in root_file:
            for obj, ancestors in self._walk_objects_with_ancestors(root_file["histos"]):
                if not self._is_histogram(obj):
                    continue
                name = self._object_name(obj)
                if re.fullmatch(r"hDecay\d+", name):
                    continue
                title = self._object_title(obj)
                inside_sc_module = any(
                    self._object_name(parent) == "SCAnaModule" for parent in ancestors
                )
                if not inside_sc_module and not self._looks_like_slow_control_label(name, title):
                    continue
                series = self._slow_control_series_from_histogram(
                    name,
                    obj,
                    str(path),
                    sensor_roles,
                )
                if series is None:
                    continue
                series_path, info = series
                if series_path in seen_paths:
                    continue
                time_series[series_path] = info
                seen_paths.add(series_path)

        if not time_series:
            return None
        self._resolve_primary_sample_temperature(time_series, temperature_kelvin)
        return _RootSlowControlLogs(
            source_file=str(path),
            channels=[key.rsplit("/", 1)[-1] for key in sorted(time_series)],
            time_series=time_series,
        )

    def _resolve_primary_sample_temperature(
        self,
        time_series: dict[str, dict[str, Any]],
        scalar: float,
    ) -> None:
        """Choose the finite canonical-K series closest to the scalar temperature.

        Zero padding outside the recorded range must not bias the comparison.
        """
        candidates = [
            info for info in time_series.values() if info.get("role") == "sample_temperature"
        ]
        if not candidates:
            return
        if len(candidates) == 1:
            # A lone sample-temperature series is the primary by definition;
            # label-matched channels arrive with primary=False.
            candidates[0]["primary"] = True
            return

        def typical(info: dict[str, Any]) -> float:
            values = np.asarray(info.get("values", ()), dtype=np.float64)  # finite-filtered
            nonzero = values[values != 0.0]
            if nonzero.size:
                return float(np.mean(nonzero))
            return float(np.mean(values)) if values.size else float("inf")

        if scalar > 0.0:
            best = min(candidates, key=lambda info: abs(typical(info) - scalar))
        else:
            # Without a usable header scalar, respect an existing Sens=-based
            # primary designation rather than demoting it by dict order.
            best = next((info for info in candidates if info.get("primary")), candidates[0])
        for info in candidates:
            info["primary"] = info is best

    def _slow_control_series_from_histogram(
        self,
        key: str,
        obj,
        source_file: str,
        sensor_roles: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        if not self._is_histogram(obj):
            return None

        name = self._object_name(obj) or str(key).rsplit("/", 1)[-1]
        title = self._object_title(obj)
        label = self._slow_control_label(name, title)
        if not label:
            return None
        sensor_role = self._slow_control_sensor_role(label, title, sensor_roles or {})

        values = np.asarray(obj.values(flow=False), dtype=np.float64)
        if values.size == 0:
            return None
        try:
            time_axis = np.asarray(obj.axis().centers(), dtype=np.float64)
        except Exception:
            time_axis = np.arange(values.size, dtype=np.float64)
        if time_axis.size != values.size:
            time_axis = np.arange(values.size, dtype=np.float64)

        raw_units = self._slow_control_units(label, title, sensor_role)
        dimension = "temperature" if sensor_role.get("role") == "sample_temperature" else "field"
        declaration = None
        if sensor_role.get("header_key"):
            values, declaration = canonical(
                values, raw_units, dimension, context=f"{source_file}: {key}"
            )
            raw_units = CANONICAL_UNITS[dimension]

        finite = np.isfinite(values)
        if not np.any(finite):
            return None
        valid_values = values[finite]
        valid_time = time_axis[finite]
        series_path = f"musrroot_slow_control/{label}"
        info: dict[str, Any] = {
            "path": series_path,
            "units": raw_units,
            "time": [float(v) for v in valid_time],
            "values": [float(v) for v in valid_values],
            "mean": float(np.mean(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "source_file": source_file,
            "source_format": "MusrRoot SCAnaModule",
            "reader_provenance": "MusrRoot slow-control histogram",
        }
        if declaration is not None:
            info["unit_declaration"] = asdict(declaration)
        if sensor_role:
            info.update(
                {
                    "role": sensor_role.get("role", ""),
                    "sensor": sensor_role.get("sensor", ""),
                    "primary": bool(sensor_role.get("primary", False)),
                    "header_key": sensor_role.get("header_key", ""),
                }
            )
        return series_path, info

    def _slow_control_label(self, name: str, title: str) -> str:
        for candidate in (title, name):
            label = str(candidate).strip()
            if not label:
                continue
            label = re.sub(r"\s+Run\s+\S+.*$", "", label)
            label = re.sub(r"^\[\d+\]\s*", "", label)
            label = re.sub(r"^h(?=[A-Z])", "", label)
            label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
            label = re.sub(r"[_/]+", " ", label).strip()
            if label and not re.fullmatch(r"hDecay\d+", label):
                return label
        return ""

    def _slow_control_units(
        self,
        label: str,
        title: str,
        sensor_role: dict[str, Any] | None = None,
    ) -> str:
        if sensor_role and sensor_role.get("units"):
            return str(sensor_role["units"])
        combined = f"{label} {title}".lower()
        if "temp" in combined or "(k)" in combined or " kelvin" in combined:
            return "K"
        if "field" in combined or "magnet" in combined:
            return "G"
        return ""

    def _looks_like_slow_control_label(self, name: str, title: str) -> bool:
        combined = f"{name} {title}".lower()
        return any(
            token in combined
            for token in (
                "temp",
                "field",
                "magnet",
                "pressure",
                "voltage",
                "current",
                "cryo",
                "heater",
                "variox",
                "sam_ts",
                "dil_t",
                "(k)",
                "mag_",
            )
        )

    def _slow_control_sensor_roles(self, header: dict[str, str]) -> dict[str, dict[str, Any]]:
        roles: dict[str, dict[str, Any]] = {}
        for key, value in header.items():
            if not key.startswith("RunInfo/"):
                continue
            role = self._slow_control_role_for_header_key(key)
            if role is None:
                continue
            units = quantity_parts(value)[1]
            for sensor in self._slow_control_sensors_from_value(value):
                normalized = self._normalize_sensor_text(sensor)
                if not normalized:
                    continue
                existing = roles.get(normalized, {})
                primary = bool(role == "sample_temperature" and key == "RunInfo/Sample Temperature")
                if existing.get("primary") and not primary:
                    continue
                roles[normalized] = {
                    "sensor": sensor,
                    "role": role,
                    "units": units,
                    "primary": primary,
                    "header_key": key,
                }
        return roles

    def _slow_control_role_for_header_key(self, key: str) -> str | None:
        label = key.rsplit("/", 1)[-1].lower()
        if "sample temperature" in label:
            return "sample_temperature"
        if "sample magnetic field" in label:
            return "sample_field"
        return None

    def _slow_control_sensors_from_value(self, value: str) -> list[str]:
        sensors: list[str] = []
        for match in re.finditer(r"\bSens\s*=\s*([^;]+?)(?=\s+-@\d+\s*$|;|$)", str(value)):
            sensor = match.group(1).strip()
            if sensor:
                sensors.append(sensor)
        return sensors

    def _slow_control_sensor_role(
        self,
        label: str,
        title: str,
        sensor_roles: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_label = self._normalize_sensor_text(f"{label} {title}")
        for sensor, role in sensor_roles.items():
            if sensor and sensor in normalized_label:
                return dict(role)
        # New-format files name channels with human-readable labels and carry
        # no "Sens=..." hints in the header, so match the role on the label
        # itself. Primary stays False here; when several channels claim the
        # sample-temperature role, _resolve_primary_sample_temperature picks
        # the one whose readings agree with the header scalar.
        if "sampletemperature" in normalized_label:
            return {
                "role": "sample_temperature",
                "sensor": "",
                "units": "K",
                "primary": False,
                "header_key": "",
            }
        if "samplemagneticfield" in normalized_label:
            return {
                "role": "sample_field",
                "sensor": "",
                "units": "G",
                "primary": False,
                "header_key": "",
            }
        return {}

    def _normalize_sensor_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text).lower())

    def _root_histogram(self, match: re.Match[str], obj) -> _RootHistogram:
        values = np.asarray(obj.values(flow=False), dtype=np.float64)
        return _RootHistogram(
            histo_number=int(match.group(1)),
            counts=values,
            title=self._object_title(obj),
        )

    def _build_dataset(
        self,
        path: Path,
        header: dict[str, str],
        header_kind: str,
        root_histograms: list[_RootHistogram],
        quantities: dict[str, tuple[float, UnitDeclaration | None]],
        slow_control_logs: _RootSlowControlLogs | None = None,
        run_summary: str | None = None,
    ) -> MuonDataset:
        bin_width_us = quantities["Time Resolution"][0]
        if not np.isfinite(bin_width_us) or bin_width_us <= 0:
            raise ValueError(f"{path}: RunInfo/Time Resolution must be finite and positive.")
        selected = self._select_histograms(header, root_histograms)
        counts = validated_count_arrays(*(hist.counts for hist in selected))
        max_bin = max(0, counts[0].size - 1)

        histograms: list[Histogram] = []
        labels: list[str] = []
        root_numbers: list[int] = []
        detector_t0_bins: list[int] = []
        first_good_bins: list[int] = []
        last_good_bins: list[int] = []

        t0_sources = []
        for hist, counts_array in zip(selected, counts, strict=True):
            detector = self._detector_info(header, hist)
            label = detector.get("Name") or hist.title or f"hDecay{hist.histo_number:03d}"
            t0_value = detector.get("Time Zero Bin")
            t0_sources.append("file" if t0_value is not None else "missing")
            t0 = self._int_from_value(t0_value, default=0)
            first_good = self._int_from_value(detector.get("First Good Bin"), default=t0)
            last_good = self._int_from_value(detector.get("Last Good Bin"), default=max_bin)
            t0 = max(0, min(max_bin, t0))
            first_good = max(0, min(max_bin, first_good))
            last_good = max(first_good, min(max_bin, last_good))
            # ROOT fractional anchors stamp the named bin centre at t = 0.
            t0_time_us = (
                None
                if t0_value is None
                else ((quantity_parts(t0_value)[0] or 0.0) + 0.5) * bin_width_us
            )

            histograms.append(
                Histogram(
                    counts=counts_array,
                    bin_width=bin_width_us,
                    t0_bin=t0,
                    good_bin_start=first_good,
                    good_bin_end=last_good,
                    t0_time_us=t0_time_us,
                )
            )
            labels.append(str(label))
            root_numbers.append(int(hist.histo_number))
            detector_t0_bins.append(t0)
            first_good_bins.append(first_good)
            last_good_bins.append(last_good)

        groups, group_names, forward_gid, backward_gid = self._default_groups(
            labels, instrument=str(header.get("RunInfo/Instrument", ""))
        )
        metadata = self._metadata(
            path,
            header,
            header_kind,
            labels,
            root_numbers,
            quantities,
            slow_control_logs,
            run_summary,
        )
        quick = quick_asymmetry(
            histograms,
            {
                "groups": groups,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "detector_t0_sources": t0_sources,
            },
            metadata=metadata,
        )
        common_t0 = quick.common_t0
        alpha = 1.0
        first_good, last_good = first_good_bins[0], last_good_bins[0]
        if quick.window is not None:
            first_good = quick.window.first + quick.alignment.first_bin
            last_good = quick.window.last + quick.alignment.first_bin
        window = quick.window.slice if quick.window is not None else slice(0, 0)
        time_axis, asymmetry, error = (
            a[window] for a in (quick.time, quick.asymmetry, quick.error)
        )

        metadata.update(quick.metadata)
        grouping = {
            "groups": groups,
            "group_names": group_names,
            "forward_group": forward_gid,
            "backward_group": backward_gid,
            "alpha": alpha,
            **quick.file_grouping,
            "t0_bin": int(common_t0),
            "t_good_offset": int(first_good) - int(common_t0),
            "first_good_bin": int(first_good),
            "last_good_bin": int(last_good),
            "bin_index_base": 0,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "good_frames": 1.0,
            "reference_exposure": ReferenceExposure(origin="whole_run_deadtime").to_dict(),
            "detector_t0_bins": detector_t0_bins,
            "detector_first_good_bins": first_good_bins,
            "detector_last_good_bins": last_good_bins,
            "t0_source": "file" if "file" in t0_sources else "missing",
            "detector_t0_sources": t0_sources,
            "histogram_labels": labels,
            "root_histo_numbers": root_numbers,
            "instrument": str(metadata.get("instrument", "")),
        }

        run = Run(
            run_number=int(metadata.get("run_number", 0) or 0),
            histograms=histograms,
            metadata=metadata,
            grouping=grouping,
            source_file=str(path),
        )
        from asymmetry.core.transform.preparation import histogram_observation

        return MuonDataset(
            effective_recipe=quick.effective_recipe,
            histogram_observation=histogram_observation(run),
            ratio_validity=quick.ratio_validity.select(window)
            if quick.ratio_validity is not None
            else None,
            time=time_axis,
            asymmetry=asymmetry,
            error=error,
            metadata=metadata,
            run=run,
            computation_revision=computation.COMPUTATION_REVISION,
        )

    def _select_histograms(
        self,
        header: dict[str, str],
        root_histograms: list[_RootHistogram],
    ) -> list[_RootHistogram]:
        by_number = {hist.histo_number: hist for hist in root_histograms}
        no_of_histos = self._int_from_value(header.get("RunInfo/No of Histos"), default=0)
        offsets = self._int_list_from_value(header.get("RunInfo/RedGreen Offsets"))
        if no_of_histos > 0 and offsets:
            selected = [
                by_number[number]
                for offset in offsets
                for number in range(offset + 1, offset + no_of_histos + 1)
                if number in by_number
            ]
            if selected:
                return selected

        if no_of_histos > 0:
            selected = [hist for hist in root_histograms if 1 <= hist.histo_number <= no_of_histos]
            if len(selected) == no_of_histos:
                return selected

        return root_histograms

    def _detector_info(self, header: dict[str, str], hist: _RootHistogram) -> dict[str, str]:
        for key, value in header.items():
            match = re.fullmatch(r"(DetectorInfo/Detector\d+/)Histo Number", key)
            if match is None:
                continue
            if self._int_from_value(value, default=-1) == hist.histo_number:
                prefix = match.group(1)
                return {
                    item_key.removeprefix(prefix): item_value
                    for item_key, item_value in header.items()
                    if item_key.startswith(prefix)
                }

        if header.get("RunInfo/Header Class") == "TLemRunHeader v4":
            return {}
        candidates = [
            f"DetectorInfo/Detector{hist.histo_number:03d}/",
            f"DetectorInfo/Detector{hist.histo_number:02d}/",
        ]
        if hist.histo_number > 0:
            candidates.append(f"DetectorInfo/Detector{hist.histo_number - 1:03d}/")

        for prefix in candidates:
            values = {
                key.removeprefix(prefix): value
                for key, value in header.items()
                if key.startswith(prefix)
            }
            if values:
                return values
        return {}

    #: Known PSI instrument names; header values matching one of these
    #: case-insensitively (the 2026 FLAME DAQ writes "flame") are normalized
    #: to the canonical upper-case form. Anything else is kept verbatim.
    _PSI_INSTRUMENTS = ("LEM", "FLAME", "GPS", "DOLLY", "GPD", "HAL", "LTF")

    def _metadata(
        self,
        path: Path,
        header: dict[str, str],
        header_kind: str,
        labels: list[str],
        root_numbers: list[int],
        quantities: dict[str, tuple[float, UnitDeclaration | None]],
        slow_control_logs: _RootSlowControlLogs | None = None,
        run_summary: str | None = None,
    ) -> dict[str, Any]:
        run_number = self._int_from_value(header.get("RunInfo/Run Number"), default=0)
        if run_number == 0:
            match = re.search(r"(\d+)", path.stem)
            run_number = int(match.group(1)) if match else 0

        field = quantities["Sample Magnetic Field"][0]
        instrument = header.get("RunInfo/Instrument", "").strip()
        if not instrument and "flame" in path.stem.lower():
            instrument = "FLAME"
        if instrument.upper() in self._PSI_INSTRUMENTS:
            instrument = instrument.upper()
        title = header.get("RunInfo/Run Title", "")
        comment = header.get("RunInfo/Comment", "")
        field_comment_candidate = comment_field_gauss(f"{title} {comment}")

        metadata: dict[str, Any] = {
            "run_number": run_number,
            "title": title,
            "sample": header.get("RunInfo/Sample Name", ""),
            "temperature": quantities["Sample Temperature"][0],
            "declared_units": {
                dimension: {f"RunInfo/{name}": asdict(quantities[name][1])}
                for name, dimension in _PHYSICAL_HEADER_FIELDS.items()
                if quantities[name][1] is not None
            },
            "field": field,
            "field_header": field,
            "field_comment_candidate": field_comment_candidate,
            "orientation": header.get("RunInfo/Sample Orientation", ""),
            "setup": header.get("RunInfo/Setup", ""),
            "comment": comment,
            # MusrRoot has no structured field-state; trust an explicit TF/LF/ZF
            # token in the free-text setup/comment/title (see
            # field_direction_from_text). "" when absent or ambiguous.
            "field_direction": field_direction_from_text(
                comment,
                header.get("RunInfo/Setup", ""),
                title,
                header.get("RunInfo/Sample Orientation", ""),
            ),
            "started": header.get("RunInfo/Run Start Time", ""),
            "stopped": header.get("RunInfo/Run Stop Time", ""),
            "instrument": instrument,
            "beamline": header.get("BeamlineInfo/Name", ""),
            "facility": header.get("RunInfo/Laboratory", ""),
            "muon_source": header.get("RunInfo/Muon Source", ""),
            "root_format": header_kind,
            "histogram_labels": list(labels),
            "root_histo_numbers": list(root_numbers),
            "source_file": str(path),
        }
        if not metadata["beamline"]:
            metadata["beamline"] = "muE4" if metadata["instrument"] == "LEM" else ""
        if not metadata["facility"] and metadata["instrument"] in {"LEM", "FLAME"}:
            metadata["facility"] = "PSI"
        if slow_control_logs is not None:
            metadata["nexus_time_series"] = slow_control_logs.time_series
            metadata["musrroot_slow_control_log"] = {
                "source_file": slow_control_logs.source_file,
                "source_format": "MusrRoot SCAnaModule",
                "reader_provenance": "MusrRoot slow-control histogram",
                "channels": list(slow_control_logs.channels),
            }
        if run_summary:
            metadata["musrroot_run_summary"] = run_summary
        return metadata

    def _default_groups(
        self,
        labels: list[str],
        *,
        instrument: str = "",
    ) -> tuple[dict[int, list[int]], dict[int, str], int, int]:
        groups, group_names = self._merge_subdetector_groups(labels, instrument=instrument)
        beam_forward_gid = self._first_explicit_group_matching(group_names, "forward")
        beam_backward_gid = self._first_explicit_group_matching(group_names, "backward")
        if beam_forward_gid is not None and beam_backward_gid is not None:
            return groups, group_names, int(beam_backward_gid), int(beam_forward_gid)

        forward_gid = self._first_group_matching(group_names, "forward")
        backward_gid = self._first_group_matching(group_names, "backward")
        if forward_gid is None:
            forward_gid = 1
        if backward_gid is None:
            backward_gid = 2 if len(groups) >= 2 and forward_gid != 2 else 1
        return groups, group_names, int(forward_gid), int(backward_gid)

    #: Transverse direction bases whose split ``_B``/``_F`` sub-detectors are
    #: combined into one group by default (matches the GPS-RD instrument
    #: presets U=(Up_B, Up_F), D, L, R and the six-group PSI-BIN GPS default).
    _TRANSVERSE_BASES = ("up", "down", "left", "right", "top", "bottom")

    def _paired_transverse_bases(self, labels: list[str]) -> set[str]:
        """Recognise complete transverse B/F plate pairs independent of instrument labels.

        A lone half stays a distinct detector; only complete pairs are merged
        into physical groups under the GPS split-half convention.
        """
        half_suffixes: dict[str, set[str]] = {}
        for label in labels:
            token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
            match = re.fullmatch(rf"({'|'.join(self._TRANSVERSE_BASES)})(b|f)", token)
            if match is not None:
                half_suffixes.setdefault(match.group(1), set()).add(match.group(2))
        return {base for base, suffixes in half_suffixes.items() if {"b", "f"} <= suffixes}

    def _combined_transverse_base(self, label: str) -> str | None:
        """Return the transverse direction a *hardware-combined* counter covers.

        A bare transverse-base label (any ``_TRANSVERSE_BASES`` member — ``Up``,
        ``Down``, ``Left``, ``Right``, ``Top``, ``Bottom`` — with no ``_B``/``_F``
        suffix) is a combined counter; anything else — a half, a beam counter,
        the GPS Mobile detector — returns ``None``.
        """
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        return token if token in self._TRANSVERSE_BASES else None

    def _transverse_base(self, label: str) -> str | None:
        """Return the transverse direction a split sub-detector belongs to.

        GPS ROOT (MusrRoot) files expose each transverse plate as a backward
        (``_B``, upstream) and forward (``_F``, downstream) half — e.g.
        ``Up_B``/``Up_F``. These belong in one physical group, so the default
        grouping combines them. Returns the lower-case base (``"up"``…) for a
        ``<base>_B``/``<base>_F`` label, else ``None`` (the label stays in its
        own group, so beam ``Forw``/``Back`` and single-letter ``R_F`` FLAME
        sub-detectors are left untouched).
        """
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        match = re.fullmatch(rf"({'|'.join(self._TRANSVERSE_BASES)})(b|f)", token)
        return match.group(1) if match else None

    def _merge_subdetector_groups(
        self,
        labels: list[str],
        *,
        instrument: str = "",
    ) -> tuple[dict[int, list[int]], dict[int, str]]:
        """Build default detector groups, combining split transverse sub-detectors.

        When the labels carry the GPS split-half convention (a transverse base
        present as both ``_B`` and ``_F`` halves) — or the instrument name
        reports GPS, which enables the merge explicitly — those halves are
        merged so ROOT GPS data loads with the same six-group default (Forward,
        Backward, Up, Down, Left, Right) as the PSI-BIN export; the ungrouped
        GPS Mobile detector stays on its own. Otherwise the merge is off (one
        group per histogram), so a lone ``<base>_F`` or genuinely distinct
        detectors are never silently collapsed.

        Where a standalone ``<base>`` label coexists with both of its halves —
        the pre-2026 GPS DAQ writes hardware-combined transverse counters that
        track the summed halves (plus the Mobile detector, on the port it
        occupies) alongside the split halves — the combined counter gets no
        default group, since keeping it would duplicate the merged pair.
        Demotion requires the pair to be present, never the instrument name
        alone.
        """
        paired = self._paired_transverse_bases(labels)
        combine = bool(paired) or str(instrument).strip().upper().startswith("GPS")
        base_to_gid: dict[str, int] = {}
        groups: dict[int, list[int]] = {}
        ordered_names: list[str] = []
        next_gid = 1
        for detector_id, raw_label in enumerate(labels, start=1):
            base = self._transverse_base(raw_label) if combine else None
            if base is not None and base in base_to_gid:
                groups[base_to_gid[base]].append(detector_id)
                continue
            if combine and base is None and self._combined_transverse_base(raw_label) in paired:
                continue
            gid = next_gid
            next_gid += 1
            groups[gid] = [detector_id]
            if base is not None:
                base_to_gid[base] = gid
                ordered_names.append(base.capitalize())
            else:
                label = str(raw_label).strip()
                ordered_names.append(label or f"Detector {detector_id}")
        return groups, self._unique_names(ordered_names)

    def _unique_names(self, labels: list[str]) -> dict[int, str]:
        seen: dict[str, int] = {}
        result: dict[int, str] = {}
        for gid, label in enumerate(labels, start=1):
            name = str(label).strip() or f"Detector {gid}"
            key = re.sub(r"[^a-z0-9]+", "", name.lower()) or f"detector{gid}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                name = f"{name} {seen[key]}"
            result[gid] = name
        return result

    def _first_group_matching(self, group_names: dict[int, str], direction: str) -> int | None:
        for gid, name in group_names.items():
            if self._label_direction(name) == direction:
                return gid
        return None

    def _first_explicit_group_matching(
        self,
        group_names: dict[int, str],
        direction: str,
    ) -> int | None:
        for gid, name in group_names.items():
            if self._explicit_label_direction(name) == direction:
                return gid
        return None

    def _explicit_label_direction(self, label: str) -> str | None:
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if token.startswith(("forw", "fwd")) or "forward" in token:
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token:
            return "backward"
        return None

    def _label_direction(self, label: str) -> str | None:
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if token.startswith(("forw", "fwd")) or "forward" in token or "left" in token:
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token or "right" in token:
            return "backward"
        return None

    def _int_from_value(self, value: str | None, *, default: int = 0) -> int:
        if value is None:
            return default
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        return int(round(float(match.group(0)))) if match else default

    def _int_list_from_value(self, value: str | None) -> list[int]:
        if value is None:
            return []
        return [
            int(round(float(match.group(0))))
            for match in re.finditer(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        ]

    def _clean_key(self, key: str) -> str:
        return re.sub(r";\d+$", "", str(key))

    def _is_directory(self, obj) -> bool:
        return obj.__class__.__name__ == "ReadOnlyDirectory"

    def _is_histogram(self, obj) -> bool:
        return hasattr(obj, "values") and self._object_class(obj).startswith("TH1")

    def _object_class(self, obj) -> str:
        return str(getattr(obj, "classname", ""))

    def _object_name(self, obj) -> str:
        try:
            return str(obj.member("fName"))
        except Exception:
            return str(getattr(obj, "name", ""))

    def _object_title(self, obj) -> str:
        try:
            return str(obj.member("fTitle"))
        except Exception:
            return str(getattr(obj, "title", ""))

    def _object_string(self, obj) -> str | None:
        if self._object_class(obj) == "TObjString":
            return str(obj)
        return None

    def _children(self, obj) -> list:
        if self._is_directory(obj) or self._is_histogram(obj):
            return []
        if self._object_class(obj) == "TObjString":
            return []
        if hasattr(obj, "member") and "fFolders" in getattr(obj, "member_names", []):
            try:
                return [child for child in obj.member("fFolders") if child is not None]
            except Exception:
                return []
        try:
            return [child for child in obj if child is not None]
        except Exception:
            return []

    def _walk_objects(self, obj):
        for child in self._children(obj):
            yield child
            yield from self._walk_objects(child)

    def _walk_objects_with_ancestors(self, obj, ancestors=()):
        for child in self._children(obj):
            yield child, ancestors
            yield from self._walk_objects_with_ancestors(child, (*ancestors, child))
