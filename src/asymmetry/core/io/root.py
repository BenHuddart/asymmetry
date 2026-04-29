"""Load MusrRoot/LEM ROOT raw histogram data.

The supported layout follows musrfit's ``PRunDataHandler::ReadRootFile``:
newer MusrRoot files store a ``RunHeader`` plus ``hDecay%03d`` histograms,
while older LEM files use ROOT folders and ``hDecay%02d`` histograms. The
implementation uses ``uproot`` rather than PyROOT, so it reads the documented
ROOT objects directly into Asymmetry's normal raw ``Run`` model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.base import BaseLoader
from asymmetry.core.transform import (
    apply_grouping_aligned,
    common_t0_for_groups,
    compute_asymmetry,
)


@dataclass
class _RootHistogram:
    histo_number: int
    counts: np.ndarray
    title: str


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

        with uproot.open(path) as root_file:
            header, header_kind = self._read_header(root_file)
            root_histograms = self._read_histograms(root_file)

        if not root_histograms:
            raise ValueError(f"ROOT file does not contain hDecay histograms: {filepath}")
        return self._build_dataset(path, header, header_kind, root_histograms)

    # ------------------------------------------------------------------
    # ROOT object extraction
    # ------------------------------------------------------------------

    def _read_header(self, root_file) -> tuple[dict[str, str], str]:
        if "RunHeader" in root_file:
            run_header = root_file["RunHeader"]
            if self._is_directory(run_header):
                return self._read_directory_header(root_file), "musr-root-directory"
            return self._read_folder_header(run_header), "musr-root-folder"

        if "RunInfo" in root_file:
            run_info = root_file["RunInfo"]
            if self._is_directory(run_info):
                header = self._read_directory_tree(root_file, "RunInfo")
            else:
                header = self._read_folder_section(run_info, "RunInfo")
            return header, "lem-root-folder"

        raise ValueError("ROOT file does not contain a RunHeader or RunInfo header")

    def _read_directory_header(self, root_file) -> dict[str, str]:
        header: dict[str, str] = {}
        for key in root_file.keys(recursive=True):
            path = self._clean_key(key)
            if not path.startswith("RunHeader/"):
                continue
            try:
                obj = root_file[key]
            except Exception:
                continue
            if self._is_directory(obj):
                continue
            value = self._object_string(obj)
            if value is None:
                continue
            rel_path = path.removeprefix("RunHeader/")
            header[rel_path] = self._parse_header_value(value)
        return header

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
                header[path] = self._parse_header_value(value)
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
            if self._object_class(item) == "TObjString":
                parsed = self._parse_musrroot_string(str(item))
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

    def _root_histogram(self, match: re.Match[str], obj) -> _RootHistogram:
        values = np.asarray(obj.values(flow=False), dtype=np.float64)
        return _RootHistogram(
            histo_number=int(match.group(1)),
            counts=values,
            title=self._object_title(obj),
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(
        self,
        path: Path,
        header: dict[str, str],
        header_kind: str,
        root_histograms: list[_RootHistogram],
    ) -> MuonDataset:
        bin_width_us = self._time_resolution_us(header)
        selected = self._select_histograms(header, root_histograms)
        n_bins = min(len(hist.counts) for hist in selected)
        max_bin = max(0, n_bins - 1)

        histograms: list[Histogram] = []
        labels: list[str] = []
        root_numbers: list[int] = []
        detector_t0_bins: list[int] = []
        first_good_bins: list[int] = []
        last_good_bins: list[int] = []

        for hist in selected:
            detector = self._detector_info(header, hist)
            label = detector.get("Name") or hist.title or f"hDecay{hist.histo_number:03d}"
            t0 = self._int_from_value(detector.get("Time Zero Bin"), default=0)
            first_good = self._int_from_value(detector.get("First Good Bin"), default=t0)
            last_good = self._int_from_value(detector.get("Last Good Bin"), default=max_bin)
            t0 = max(0, min(max_bin, t0))
            first_good = max(0, min(max_bin, first_good))
            last_good = max(first_good, min(max_bin, last_good))

            histograms.append(
                Histogram(
                    counts=np.asarray(hist.counts[:n_bins], dtype=np.float64),
                    bin_width=bin_width_us,
                    t0_bin=t0,
                    good_bin_start=first_good,
                    good_bin_end=last_good,
                )
            )
            labels.append(str(label))
            root_numbers.append(int(hist.histo_number))
            detector_t0_bins.append(t0)
            first_good_bins.append(first_good)
            last_good_bins.append(last_good)

        groups, group_names, forward_gid, backward_gid = self._default_groups(labels)
        forward_idx = [det - 1 for det in groups[forward_gid]]
        backward_idx = [det - 1 for det in groups[backward_gid]]
        common_t0 = common_t0_for_groups(histograms, forward_idx, backward_idx)
        forward = apply_grouping_aligned(histograms, forward_idx, common_t0_bin=common_t0)
        backward = apply_grouping_aligned(histograms, backward_idx, common_t0_bin=common_t0)
        n = min(len(forward), len(backward))
        alpha = 1.0
        asymmetry, error = compute_asymmetry(forward[:n], backward[:n], alpha=alpha)
        asymmetry = asymmetry * 100.0
        error = error * 100.0

        first_good = min(
            n - 1,
            int(common_t0)
            + max(
                (max(0, first - t0) for first, t0 in zip(first_good_bins, detector_t0_bins)),
                default=0,
            ),
        )
        last_good = min(
            n - 1,
            int(common_t0)
            + min(
                (max(0, last - t0) for last, t0 in zip(last_good_bins, detector_t0_bins)),
                default=n - 1,
            ),
        )
        if last_good < first_good:
            last_good = first_good

        time_axis = (np.arange(n, dtype=np.float64) - float(common_t0)) * bin_width_us
        time_axis = time_axis[first_good : last_good + 1]
        asymmetry = asymmetry[first_good : last_good + 1]
        error = error[first_good : last_good + 1]

        metadata = self._metadata(path, header, header_kind, labels, root_numbers)
        grouping = {
            "groups": groups,
            "group_names": group_names,
            "forward_group": forward_gid,
            "backward_group": backward_gid,
            "alpha": alpha,
            "t0_bin": int(common_t0),
            "t_good_offset": max(0, int(first_good) - int(common_t0)),
            "first_good_bin": int(first_good),
            "last_good_bin": int(last_good),
            "bin_index_base": 0,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "detector_t0_bins": detector_t0_bins,
            "detector_first_good_bins": first_good_bins,
            "detector_last_good_bins": last_good_bins,
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
        return MuonDataset(
            time=np.asarray(time_axis, dtype=np.float64),
            asymmetry=np.asarray(asymmetry, dtype=np.float64),
            error=np.asarray(error, dtype=np.float64),
            metadata=metadata,
            run=run,
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

    def _metadata(
        self,
        path: Path,
        header: dict[str, str],
        header_kind: str,
        labels: list[str],
        root_numbers: list[int],
    ) -> dict[str, Any]:
        run_number = self._int_from_value(header.get("RunInfo/Run Number"), default=0)
        if run_number == 0:
            match = re.search(r"(\d+)", path.stem)
            run_number = int(match.group(1)) if match else 0

        field = self._field_gauss(header.get("RunInfo/Sample Magnetic Field"))
        metadata: dict[str, Any] = {
            "run_number": run_number,
            "title": header.get("RunInfo/Run Title", ""),
            "sample": header.get("RunInfo/Sample Name", ""),
            "temperature": self._float_from_value(
                header.get("RunInfo/Sample Temperature"),
                default=0.0,
            ),
            "field": field,
            "orientation": header.get("RunInfo/Sample Orientation", ""),
            "setup": header.get("RunInfo/Setup", ""),
            "comment": header.get("RunInfo/Comment", ""),
            "started": header.get("RunInfo/Run Start Time", ""),
            "stopped": header.get("RunInfo/Run Stop Time", ""),
            "instrument": header.get("RunInfo/Instrument", ""),
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
        if not metadata["facility"] and metadata["instrument"] == "LEM":
            metadata["facility"] = "PSI"
        return metadata

    def _default_groups(
        self,
        labels: list[str],
    ) -> tuple[dict[int, list[int]], dict[int, str], int, int]:
        groups = {gid: [gid] for gid in range(1, len(labels) + 1)}
        group_names = self._unique_names(labels)
        forward_gid = self._first_group_matching(group_names, "forward")
        backward_gid = self._first_group_matching(group_names, "backward")
        if forward_gid is None:
            forward_gid = 1
        if backward_gid is None:
            backward_gid = 2 if len(groups) >= 2 and forward_gid != 2 else 1
        return groups, group_names, int(forward_gid), int(backward_gid)

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

    def _label_direction(self, label: str) -> str | None:
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if token.startswith(("forw", "fwd")) or "forward" in token or "left" in token:
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token or "right" in token:
            return "backward"
        return None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _time_resolution_us(self, header: dict[str, str]) -> float:
        value = header.get("RunInfo/Time Resolution")
        if not value:
            return 1.0
        number = self._float_from_value(value, default=1.0)
        unit = self._unit_from_value(value)
        if unit in {"ps", "picosec", "picosecond", "picoseconds"}:
            return number * 1.0e-6
        if unit in {"ns", "nanosec", "nanosecond", "nanoseconds"}:
            return number * 1.0e-3
        if unit in {"us", "microsec", "microsecond", "microseconds"}:
            return number
        return number

    def _field_gauss(self, value: str | None) -> float:
        if not value:
            return 0.0
        number = self._float_from_value(value, default=0.0)
        unit = self._unit_from_value(value)
        if unit in {"t", "tesla"}:
            return number * 1.0e4
        return number

    def _float_from_value(self, value: str | None, *, default: float = 0.0) -> float:
        if value is None:
            return default
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        return float(match.group(0)) if match else default

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

    def _unit_from_value(self, value: str | None) -> str:
        if value is None:
            return ""
        text = str(value)
        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*(?:\+-\s*[-+0-9.eE]+\s*)?([A-Za-z/]+)",
            text,
        )
        return match.group(1).lower() if match else ""

    def _parse_musrroot_string(self, text: str) -> tuple[str, str] | None:
        match = re.match(r"\s*\d+\s*-\s*(.*?)\s*(?:-@\d+)?\s*$", text)
        if match is None:
            return None
        body = match.group(1)
        if ":" not in body:
            return None
        key, value = body.split(":", 1)
        return key.strip(), self._parse_header_value(value)

    def _parse_header_value(self, value: Any) -> str:
        text = str(value).strip()
        text = re.sub(r"\s+-@\d+\s*$", "", text).strip()
        return text

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
