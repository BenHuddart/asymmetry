"""Load PSI BIN and MDU raw histogram data.

The binary layout follows the PSI reader used by musrfit, particularly its
``PRunDataHandler::ReadPsiBinFile`` path and the PSI BIN/MDU structures used
there. Mantid's ``LoadPSIMuonBin`` was also checked for PSI-BIN behavior. The
implementation here is pure Python and maps the raw histograms, detector
labels, per-detector ``t0`` values, and good-bin metadata into Asymmetry's
normal raw ``Run`` model.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.base import BaseLoader
from asymmetry.core.io.wim import _extract_field_from_comment
from asymmetry.core.transform import (
    apply_grouping_aligned,
    common_t0_for_groups,
    compute_asymmetry,
)

_MAX_PSI_HISTOGRAMS = 32
_BIN_HEADER_SIZE = 1024
_TEMPERATURE_FILE_EXT = ".mon"
_TEMPERATURE_FILE_MAX_SEARCH_DEPTH = 3
_PTA_TAG_TYPE_POSITRON = b"P"
_FE_HEADER = struct.Struct("<cc12s9s12s9sii41s63s20siiii200s50s50siiiii")
_SETTINGS_PREFIX = struct.Struct("<13i")
_PSI_MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


@dataclass
class _PsiTemperatureLogs:
    source_file: str
    start_time: str
    channels: list[str]
    time_series: dict[str, dict[str, Any]]


@dataclass
class _PsiRawRun:
    source_file: str
    psi_format: str
    run_number: int
    title: str
    sample: str
    temperature: float
    field: float
    orientation: str
    setup: str
    comment: str
    started: str
    stopped: str
    bin_width_us: float
    histogram_labels: list[str]
    counts: list[np.ndarray]
    t0_bins: list[int]
    first_good_bins: list[int]
    last_good_bins: list[int]
    instrument: str
    beamline: str
    muon_source: str
    temperature_logs: _PsiTemperatureLogs | None = None


@dataclass
class _MduHeader:
    fmt: str
    start_date: str
    start_time: str
    end_date: str
    end_time: str
    run_number: int
    run_title: str
    run_subtitle: str
    histo_resolution: int
    number_of_detectors: int
    detector_number_list: str
    mean_temp: str
    temp_dev: str
    bin_size: int
    num_bytes_header: int
    num_bytes_settings: int
    num_bytes_tag: int
    num_bytes_statistics: int


@dataclass
class _MduTag:
    label: str
    tag_type: bytes
    histominb: int
    histomaxb: int
    t0b: int
    tfb: int
    tlb: int

    @property
    def n_bins(self) -> int:
        return int(self.histomaxb) - int(self.histominb) + 1


class PsiLoader(BaseLoader):
    """Read PSI ``.bin`` and ``.mdu`` raw histogram files."""

    extensions = [".bin", ".mdu"]
    format_name = "PSI BIN/MDU (.bin, .mdu)"

    def load(self, filepath: str) -> MuonDataset:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with path.open("rb") as handle:
            fmt = handle.read(2)

        if fmt == b"1N" or (len(fmt) == 2 and fmt[:1] == b"1"):
            raw = self._read_bin(path)
        elif fmt in {b"M3", b"T4", b"T5"}:
            raw = self._read_mdu(path)
        else:
            raise ValueError(f"Unsupported PSI file signature {fmt!r}: {filepath}")
        return self._build_dataset(raw)

    # ------------------------------------------------------------------
    # PSI-BIN
    # ------------------------------------------------------------------

    def _read_bin(self, path: Path) -> _PsiRawRun:
        data = path.read_bytes()
        if len(data) < _BIN_HEADER_SIZE:
            raise ValueError("PSI-BIN file is too small to contain a header")

        header = data[:_BIN_HEADER_SIZE]
        fmt = header[:2]
        if fmt[:1] != b"1":
            raise ValueError("Unsupported PSI-BIN header")

        tdc_resolution = self._i16(header, 2)
        run_number = self._i16(header, 6)
        n_bins = self._i16(header, 28)
        n_hist = self._i16(header, 30)
        if n_hist <= 0 or n_hist > 16:
            raise ValueError(f"Unsupported PSI-BIN histogram count: {n_hist}")
        if n_bins <= 0:
            raise ValueError(f"Unsupported PSI-BIN histogram length: {n_bins}")

        n_records_file = self._i16(header, 128)
        record_len = self._i16(header, 130)
        records_per_hist = self._i16(header, 132)
        histograms_per_record = self._i16(header, 134)
        if histograms_per_record != 1:
            raise ValueError("PSI-BIN files with multiple histograms per record are unsupported")

        bin_width_us = self._f32(header, 1012)
        if bin_width_us == 0.0:
            bin_width_us = 0.125 * (625.0e-6) * (2.0 ** float(tdc_resolution))

        counts_block_size = int(n_records_file) * int(record_len) * 4
        counts_block = data[_BIN_HEADER_SIZE : _BIN_HEADER_SIZE + counts_block_size]
        if len(counts_block) < counts_block_size:
            raise ValueError("PSI-BIN file ended before all histogram records were read")

        labels = [self._text(header[948 + i * 4 : 952 + i * 4]) for i in range(n_hist)]
        counts: list[np.ndarray] = []
        for hist_idx in range(n_hist):
            start = hist_idx * int(records_per_hist) * int(record_len) * 4
            end = start + int(n_bins) * 4
            if end > len(counts_block):
                raise ValueError("PSI-BIN histogram data are shorter than the header declares")
            counts.append(np.frombuffer(counts_block[start:end], dtype="<i4").astype(np.float64))

        t0_bins = [self._i16(header, 458 + i * 2) for i in range(n_hist)]
        first_good = [self._i16(header, 490 + i * 2) for i in range(n_hist)]
        last_good = [self._i16(header, 522 + i * 2) for i in range(n_hist)]

        instrument, beamline, muon_source = self._guess_psi_instrument(path)
        temp_text = self._text(header[148:158])
        field_text = self._text(header[158:168])
        temperature_logs = self._read_temperature_logs(path, int(run_number))

        return _PsiRawRun(
            source_file=str(path),
            psi_format="psi-bin",
            run_number=int(run_number),
            title=self._text(header[860:922]),
            sample=self._text(header[138:148]),
            temperature=self._parse_temperature(temp_text),
            field=self._parse_field_gauss(field_text),
            orientation=self._text(header[168:178]),
            setup=self._text(header[178:188]),
            comment=self._text(header[860:922]),
            started=self._date_time(self._text(header[218:227]), self._text(header[236:244])),
            stopped=self._date_time(self._text(header[227:236]), self._text(header[244:252])),
            bin_width_us=float(bin_width_us),
            histogram_labels=labels,
            counts=counts,
            t0_bins=t0_bins,
            first_good_bins=first_good,
            last_good_bins=last_good,
            instrument=instrument,
            beamline=beamline,
            muon_source=muon_source,
            temperature_logs=temperature_logs,
        )

    # ------------------------------------------------------------------
    # PSI-MDU
    # ------------------------------------------------------------------

    def _read_mdu(self, path: Path) -> _PsiRawRun:
        with path.open("rb") as handle:
            header_data = handle.read(_FE_HEADER.size)
            if len(header_data) != _FE_HEADER.size:
                raise ValueError("PSI-MDU file is too small to contain a header")
            header = self._parse_mdu_header(header_data)
            if header.num_bytes_header != _FE_HEADER.size:
                raise ValueError(f"Unsupported PSI-MDU header size: {header.num_bytes_header}")
            if header.num_bytes_tag < 16 + 11 * 4:
                raise ValueError(f"Unsupported PSI-MDU tag size: {header.num_bytes_tag}")

            settings_data = handle.read(header.num_bytes_settings)
            if len(settings_data) != header.num_bytes_settings:
                raise ValueError("PSI-MDU file ended while reading settings")
            stats_data = handle.read(header.num_bytes_statistics)
            if len(stats_data) != header.num_bytes_statistics:
                raise ValueError("PSI-MDU file ended while reading statistics")

            settings_tags = self._parse_mdu_settings_tags(header, settings_data)
            bin_width_us, resolution_factor, total_tags = self._mdu_resolution(
                header,
                settings_data,
            )
            selected = self._selected_mdu_tags(header.detector_number_list)
            expected_length, expected_hists = self._mdu_histogram_shape(
                header,
                settings_tags,
                selected,
                total_tags,
            )
            if expected_hists <= 0 or expected_length <= 0:
                raise ValueError("PSI-MDU file does not contain usable positron histograms")

            labels: list[str] = []
            counts: list[np.ndarray] = []
            t0_bins: list[int] = []
            first_good: list[int] = []
            last_good: list[int] = []

            for tag_idx in range(total_tags):
                tag_data = handle.read(header.num_bytes_tag)
                if len(tag_data) != header.num_bytes_tag:
                    raise ValueError("PSI-MDU file ended while reading histogram tag records")
                tag = self._parse_mdu_tag(tag_data)
                if tag.tag_type != _PTA_TAG_TYPE_POSITRON or tag.n_bins <= 1:
                    continue

                hist_data = handle.read(tag.n_bins * 4)
                if len(hist_data) != tag.n_bins * 4:
                    raise ValueError("PSI-MDU file ended while reading histogram data")
                if header.fmt == "M3" and tag_idx not in selected:
                    continue

                arr = np.zeros(expected_length, dtype=np.float64)
                raw_counts = np.frombuffer(hist_data, dtype="<i4").astype(np.float64)
                start = max(0, int(tag.histominb))
                available = min(expected_length - start, len(raw_counts))
                if available > 0:
                    arr[start : start + available] = raw_counts[:available]
                counts.append(arr)
                labels.append(tag.label)
                t0_bins.append((int(tag.t0b) + 1) * int(resolution_factor) - 1)
                first_good.append((int(tag.tfb) + 1) * int(resolution_factor) - 1)
                last_good.append(int(tag.tlb) * int(resolution_factor))

        if len(counts) != expected_hists:
            raise ValueError(
                f"PSI-MDU histogram count mismatch: expected {expected_hists}, got {len(counts)}"
            )

        instrument, beamline, muon_source = self._guess_psi_instrument(path)
        temp_text = header.mean_temp
        if not temp_text.strip() or (
            self._parse_temperature(temp_text) == 0.0
            and self._parse_temperature(header.run_title[10:20]) != 0.0
        ):
            temp_text = header.run_title[10:20]
        field_text = header.run_title[20:30]

        return _PsiRawRun(
            source_file=str(path),
            psi_format="psi-mdu",
            run_number=header.run_number,
            title=header.run_subtitle,
            sample=header.run_title[:10].strip(),
            temperature=self._parse_temperature(temp_text),
            field=self._parse_field_gauss(field_text),
            orientation=header.run_title[30:40].strip(),
            setup="",
            comment=header.run_subtitle,
            started=self._date_time(header.start_date, header.start_time),
            stopped=self._date_time(header.end_date, header.end_time),
            bin_width_us=float(bin_width_us),
            histogram_labels=labels,
            counts=counts,
            t0_bins=t0_bins,
            first_good_bins=first_good,
            last_good_bins=last_good,
            instrument=instrument,
            beamline=beamline,
            muon_source=muon_source,
        )

    def _parse_mdu_header(self, data: bytes) -> _MduHeader:
        fields = _FE_HEADER.unpack(data)
        fmt = (fields[0] + fields[1]).decode("ascii", errors="ignore")
        if fmt not in {"M3", "T4", "T5"}:
            raise ValueError(f"Unsupported PSI-MDU format: {fmt!r}")
        return _MduHeader(
            fmt=fmt,
            start_date=self._text(fields[2]),
            start_time=self._text(fields[3]),
            end_date=self._text(fields[4]),
            end_time=self._text(fields[5]),
            run_number=int(fields[6]),
            run_title=self._text(fields[8]),
            run_subtitle=self._text(fields[9]),
            histo_resolution=int(fields[11]),
            number_of_detectors=int(fields[14]),
            detector_number_list=self._text(fields[15]),
            mean_temp=self._text(fields[16]),
            temp_dev=self._text(fields[17]),
            bin_size=int(fields[18]),
            num_bytes_header=int(fields[19]),
            num_bytes_settings=int(fields[20]),
            num_bytes_tag=int(fields[21]),
            num_bytes_statistics=int(fields[22]),
        )

    def _parse_mdu_settings_tags(self, header: _MduHeader, settings_data: bytes) -> list[_MduTag]:
        _SETTINGS_PREFIX.unpack_from(settings_data, 0)
        total_tags = 16 if header.fmt in {"M3", "T4"} else 32
        tags: list[_MduTag] = []
        offset = _SETTINGS_PREFIX.size
        for _ in range(total_tags):
            tags.append(self._parse_mdu_tag(settings_data[offset : offset + header.num_bytes_tag]))
            offset += header.num_bytes_tag
        return tags

    def _parse_mdu_tag(self, data: bytes) -> _MduTag:
        label = self._text(data[:12])
        tag_type = data[12:13]
        values = struct.unpack_from("<11i", data, 16)
        return _MduTag(
            label=label,
            tag_type=tag_type,
            histominb=int(values[6]),
            histomaxb=int(values[7]),
            t0b=int(values[8]),
            tfb=int(values[9]),
            tlb=int(values[10]),
        )

    def _mdu_resolution(self, header: _MduHeader, settings_data: bytes) -> tuple[float, int, int]:
        values = _SETTINGS_PREFIX.unpack_from(settings_data, 0)
        if header.fmt == "M3":
            timespan = int(values[9])
            table = {
                11: 0.000625,
                10: 0.0003125,
                9: 0.00015625,
                8: 0.000078125,
                7: 0.0000390625,
                6: 0.00001953125,
            }
            if timespan not in table:
                raise ValueError(f"Unsupported PSI-MDU pTA timespan: {timespan}")
            factor = 1
            for _ in range(max(0, int(header.histo_resolution) + 8 - timespan)):
                factor *= 2
            return table[timespan], factor, 16

        resolution_code = int(values[9])
        table = {
            25: 0.0000244140625,
            100: 0.00009765625,
            200: 0.0001953125,
            800: 0.00078125,
        }
        if resolution_code not in table:
            raise ValueError(f"Unsupported PSI-MDU TDC resolution code: {resolution_code}")
        total_tags = 16 if header.fmt == "T4" else 32
        return table[resolution_code], max(1, int(header.histo_resolution)), total_tags

    def _mdu_histogram_shape(
        self,
        header: _MduHeader,
        tags: list[_MduTag],
        selected: set[int],
        total_tags: int,
    ) -> tuple[int, int]:
        n_hist = 0
        length = 0
        for idx, tag in enumerate(tags[:total_tags]):
            if tag.tag_type != _PTA_TAG_TYPE_POSITRON or tag.n_bins <= 1:
                continue
            if header.fmt == "M3" and idx not in selected:
                continue
            candidate = int(tag.n_bins) + int(tag.histominb)
            if length == 0:
                length = candidate
            n_hist += 1
        if header.fmt == "M3" and length > 0:
            length -= 1
        return max(0, length), n_hist

    def _selected_mdu_tags(self, detector_list: str) -> set[int]:
        selected: set[int] = set()
        for token in detector_list.split():
            try:
                idx = int(token)
            except ValueError:
                continue
            if 0 <= idx < _MAX_PSI_HISTOGRAMS:
                selected.add(idx)
        return selected

    # ------------------------------------------------------------------
    # PSI-BIN temperature sidecars
    # ------------------------------------------------------------------

    def _read_temperature_logs(self, path: Path, run_number: int) -> _PsiTemperatureLogs | None:
        """Read the optional PSI ``.mon`` sidecar using Mantid-compatible rules."""
        log_path = self._find_temperature_log_file(path, run_number)
        if log_path is None:
            return None
        try:
            return self._parse_temperature_log_file(log_path)
        except (OSError, ValueError):
            return None

    def _find_temperature_log_file(self, path: Path, run_number: int) -> Path | None:
        """Find a PSI ``.mon`` file whose name contains the BIN run number."""
        run_token = str(int(run_number))
        queue: list[tuple[Path, int]] = [(path.parent, 0)]
        seen: set[Path] = set()
        while queue:
            directory, depth = queue.pop(0)
            try:
                resolved = directory.resolve()
            except OSError:
                resolved = directory
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                children = sorted(directory.iterdir(), key=lambda child: child.name.lower())
            except OSError:
                continue
            for child in children:
                if child.is_file():
                    if child.suffix.lower() == _TEMPERATURE_FILE_EXT and run_token in child.name:
                        return child
                elif child.is_dir() and depth < _TEMPERATURE_FILE_MAX_SEARCH_DEPTH:
                    queue.append((child, depth + 1))
        return None

    def _parse_temperature_log_file(self, path: Path) -> _PsiTemperatureLogs:
        contents = path.read_text(encoding="latin-1", errors="ignore").splitlines()
        titles: list[str] = []
        delimiter_is_backslash = False
        start_time = ""
        data_start = 0

        for line_no, line in enumerate(contents):
            if not line.startswith("!"):
                data_start = line_no
                break
            if line_no <= 6:
                continue
            if "Title" in line:
                titles, delimiter_is_backslash = self._parse_temperature_titles(line)
            elif self._looks_like_psi_temperature_date(line):
                start_time = self._parse_temperature_start_time(line)
        else:
            data_start = len(contents)

        if not titles:
            raise ValueError(f"PSI temperature log does not define channel titles: {path}")

        series_data: dict[str, dict[str, list[float]]] = {}
        for line in contents[data_start:]:
            if not line.strip() or line.startswith("!"):
                continue
            for channel, time_value, value in self._parse_temperature_data_line(
                line,
                titles,
                delimiter_is_backslash,
            ):
                bucket = series_data.setdefault(channel, {"time": [], "values": []})
                bucket["time"].append(time_value)
                bucket["values"].append(value)

        time_series: dict[str, dict[str, Any]] = {}
        for channel, values in sorted(series_data.items()):
            data_values = np.asarray(values["values"], dtype=np.float64)
            if data_values.size == 0:
                continue
            series_path = f"psi_temperature/{channel}"
            time_series[series_path] = {
                "path": series_path,
                "units": "K",
                "time": list(values["time"]),
                "values": list(values["values"]),
                "mean": float(np.mean(data_values)),
                "min": float(np.min(data_values)),
                "max": float(np.max(data_values)),
                "source_file": str(path),
                "source_format": "PSI .mon",
                "reader_provenance": "Mantid LoadPSIMuonBin-compatible",
            }

        if not time_series:
            raise ValueError(f"PSI temperature log does not contain numeric data: {path}")

        return _PsiTemperatureLogs(
            source_file=str(path),
            start_time=start_time,
            channels=[key.rsplit("/", 1)[-1] for key in sorted(time_series)],
            time_series=time_series,
        )

    def _parse_temperature_titles(self, line: str) -> tuple[list[str], bool]:
        _, _, title_text = line.partition(":")
        title_text = title_text.strip()
        if "\\" in title_text:
            return [title.strip() for title in title_text.split("\\") if title.strip()], True
        return [title.strip() for title in title_text.split() if title.strip()], False

    def _looks_like_psi_temperature_date(self, line: str) -> bool:
        if len(line) >= 8 and line[5:8].upper() in _PSI_MONTHS:
            return True
        return bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}", line))

    def _parse_temperature_start_time(self, line: str) -> str:
        psi_match = re.search(
            r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})\s+(\d{1,2}:\d{2}:\d{2})",
            line,
        )
        if psi_match:
            day, month, year, time_text = psi_match.groups()
            year_int = int(year)
            if year_int < 100:
                year_int += 2000
            month_number = _PSI_MONTHS.get(month.upper(), "01")
            return f"{year_int:04d}-{month_number}-{int(day):02d}T{time_text}"

        iso_match = re.search(
            r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}:\d{2}:\d{2})",
            line,
        )
        if iso_match:
            year, month, day, time_text = iso_match.groups()
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{time_text}"
        return ""

    def _parse_temperature_data_line(
        self,
        line: str,
        titles: list[str],
        delimiter_is_backslash: bool,
    ) -> list[tuple[str, float, float]]:
        segments = line.split("\\")
        if len(segments) != 5:
            raise ValueError(f"PSI temperature log data line is not backslash-delimited: {line}")

        time_value = self._seconds_from_clock_string(segments[0])
        num_values = int(segments[1])
        first_values = self._split_temperature_values(segments[2])
        second_values = self._split_temperature_values(segments[3])

        rows: list[tuple[str, float, float]] = []
        if delimiter_is_backslash:
            if len(titles) >= 1 and first_values:
                rows.append((f"Temp_{titles[0]}", time_value, first_values[0]))
            if len(titles) >= 2 and second_values:
                rows.append((f"Temp_{titles[1]}", time_value, second_values[0]))
            return rows

        limit = min(num_values, len(titles), len(first_values))
        for idx in range(limit):
            rows.append((f"Temp_{titles[idx]}", time_value, first_values[idx]))
        return rows

    def _split_temperature_values(self, text: str) -> list[float]:
        values: list[float] = []
        for token in text.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def _seconds_from_clock_string(self, text: str) -> float:
        match = re.match(r"\s*(\d{1,2}):(\d{2}):(\d{2})", text)
        if not match:
            raise ValueError(f"Invalid PSI temperature log time: {text!r}")
        hours, minutes, seconds = (int(part) for part in match.groups())
        return float(hours * 3600 + minutes * 60 + seconds)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(self, raw: _PsiRawRun) -> MuonDataset:
        n_hist = len(raw.counts)
        if n_hist == 0:
            raise ValueError("PSI file does not contain any histograms")

        n_bins = min(len(arr) for arr in raw.counts)
        max_bin = max(0, n_bins - 1)
        t0_bins = self._resize_ints(raw.t0_bins, n_hist, 0, max_value=max_bin)
        first_good_bins = self._resize_ints(
            raw.first_good_bins,
            n_hist,
            0,
            max_value=max_bin,
        )
        last_good_bins = self._resize_ints(
            raw.last_good_bins,
            n_hist,
            n_bins - 1,
            max_value=max_bin,
        )

        histograms = [
            Histogram(
                counts=np.asarray(raw.counts[i][:n_bins], dtype=np.float64),
                bin_width=float(raw.bin_width_us),
                t0_bin=int(t0_bins[i]),
                good_bin_start=int(first_good_bins[i]),
                good_bin_end=int(last_good_bins[i]),
            )
            for i in range(n_hist)
        ]

        groups, group_names, forward_gid, backward_gid = self._default_groups(
            raw.histogram_labels,
            n_hist,
        )
        forward_idx = [det - 1 for det in groups[forward_gid]]
        backward_idx = [det - 1 for det in groups[backward_gid]]
        common_t0 = common_t0_for_groups(histograms, forward_idx, backward_idx)

        forward = apply_grouping_aligned(histograms, forward_idx, common_t0_bin=common_t0)
        backward = apply_grouping_aligned(histograms, backward_idx, common_t0_bin=common_t0)
        n = min(len(forward), len(backward))
        forward = forward[:n]
        backward = backward[:n]
        alpha = 1.0
        asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)
        asymmetry = asymmetry * 100.0
        error = error * 100.0

        good_offsets = [max(0, int(first_good_bins[i]) - int(t0_bins[i])) for i in range(n_hist)]
        last_offsets = [max(0, int(last_good_bins[i]) - int(t0_bins[i])) for i in range(n_hist)]
        first_good = min(n - 1, int(common_t0) + max(good_offsets, default=0))
        last_good = min(n - 1, int(common_t0) + min(last_offsets, default=n - 1))
        if last_good < first_good:
            last_good = first_good

        time_axis = (np.arange(n, dtype=np.float64) - float(common_t0)) * float(raw.bin_width_us)
        time_axis = time_axis[first_good : last_good + 1]
        asymmetry = asymmetry[first_good : last_good + 1]
        error = error[first_good : last_good + 1]

        metadata = {
            "run_number": raw.run_number,
            "title": raw.title,
            "sample": raw.sample,
            "temperature": raw.temperature,
            "field": raw.field,
            "field_header": raw.field,
            "field_comment_candidate": _extract_field_from_comment(raw.comment),
            "orientation": raw.orientation,
            "setup": raw.setup,
            "comment": raw.comment,
            "started": raw.started,
            "stopped": raw.stopped,
            "instrument": raw.instrument,
            "beamline": raw.beamline,
            "facility": "PSI",
            "psi_format": raw.psi_format,
            "muon_source": raw.muon_source,
            "histogram_labels": list(raw.histogram_labels),
            "source_file": raw.source_file,
        }
        if raw.temperature_logs is not None:
            metadata["nexus_time_series"] = raw.temperature_logs.time_series
            metadata["psi_temperature_log"] = {
                "source_file": raw.temperature_logs.source_file,
                "source_format": "PSI .mon",
                "reader_provenance": "Mantid LoadPSIMuonBin-compatible",
                "start_time": raw.temperature_logs.start_time,
                "channels": list(raw.temperature_logs.channels),
            }
            metadata["psi_temperature_log_file"] = raw.temperature_logs.source_file
            metadata["psi_temperature_log_channels"] = list(raw.temperature_logs.channels)

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
            "source_bunching_factor": 1,
            "deadtime_correction": False,
            "detector_t0_bins": [int(v) for v in t0_bins],
            "detector_first_good_bins": [int(v) for v in first_good_bins],
            "detector_last_good_bins": [int(v) for v in last_good_bins],
            "histogram_labels": list(raw.histogram_labels),
            "instrument": raw.instrument,
        }

        run = Run(
            run_number=int(raw.run_number),
            histograms=histograms,
            metadata=metadata,
            grouping=grouping,
            source_file=raw.source_file,
        )
        return MuonDataset(
            time=np.asarray(time_axis, dtype=np.float64),
            asymmetry=np.asarray(asymmetry, dtype=np.float64),
            error=np.asarray(error, dtype=np.float64),
            metadata=metadata,
            run=run,
        )

    def _default_groups(
        self,
        labels: list[str],
        n_hist: int,
    ) -> tuple[dict[int, list[int]], dict[int, str], int, int]:
        label_groups = self._label_groups(labels, n_hist)
        if len(label_groups) >= 2:
            groups: dict[int, list[int]] = {}
            names: dict[int, str] = {}
            forward_gid: int | None = None
            backward_gid: int | None = None
            for gid, (name, detectors) in enumerate(label_groups, start=1):
                groups[gid] = detectors
                names[gid] = name
                direction = self._label_direction(name)
                if direction == "forward" and forward_gid is None:
                    forward_gid = gid
                elif direction == "backward" and backward_gid is None:
                    backward_gid = gid

            if forward_gid is None:
                forward_gid = 1
            if backward_gid is None:
                backward_gid = 2 if forward_gid != 2 and 2 in groups else 1
            return groups, names, int(forward_gid), int(backward_gid)

        back: list[int] = []
        forward: list[int] = []
        left: list[int] = []
        right: list[int] = []
        for i, label in enumerate(labels[:n_hist], start=1):
            direction = self._label_direction(label)
            if direction == "backward":
                back.append(i)
            elif direction == "forward":
                forward.append(i)
            elif direction == "left":
                left.append(i)
            elif direction == "right":
                right.append(i)

        if back and forward:
            groups = {1: back, 2: forward}
            names = {1: "Backward", 2: "Forward"}
            next_gid = 3
            if left:
                groups[next_gid] = left
                names[next_gid] = "Left"
                next_gid += 1
            if right:
                groups[next_gid] = right
                names[next_gid] = "Right"
            return groups, names, 2, 1

        split = max(1, n_hist // 2)
        groups = {1: list(range(1, split + 1)), 2: list(range(split + 1, n_hist + 1))}
        if not groups[2]:
            groups[2] = list(groups[1])
        return groups, {1: "Forward", 2: "Backward"}, 1, 2

    def _label_groups(self, labels: list[str], n_hist: int) -> list[tuple[str, list[int]]]:
        groups: list[tuple[str, list[int]]] = []
        seen: dict[str, int] = {}
        has_label = any(str(label).strip() for label in labels[:n_hist])
        if not has_label:
            return groups
        for detector_id in range(1, n_hist + 1):
            raw_label = labels[detector_id - 1] if detector_id - 1 < len(labels) else ""
            label = str(raw_label).strip()
            if not label:
                label = f"Detector {detector_id}"
            key = re.sub(r"[^a-z0-9]+", "", label.lower())
            if not key:
                key = f"detector{detector_id}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                label = f"{label} {seen[key]}"
            groups.append((label, [detector_id]))
        return groups

    def _label_direction(self, label: str) -> str | None:
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if (
            token.startswith("back")
            or token.startswith("bwd")
            or token == "b"
            or re.fullmatch(r"b\d+", token)
        ):
            return "backward"
        if (
            token.startswith("forw")
            or token.startswith("fwd")
            or token == "f"
            or re.fullmatch(r"f\d+", token)
        ):
            return "forward"
        if token.startswith("left") or token == "l" or re.fullmatch(r"l\d+", token):
            return "left"
        if (
            token.startswith("right")
            or token.startswith("rite")
            or token == "r"
            or re.fullmatch(r"r\d+", token)
        ):
            return "right"
        if token.startswith("up") or token.startswith("top") or re.fullmatch(r"u\d+", token):
            return "up"
        if token.startswith("down") or token.startswith("bottom") or re.fullmatch(r"d\d+", token):
            return "down"
        return None

    # ------------------------------------------------------------------
    # Small parsing helpers
    # ------------------------------------------------------------------

    def _i16(self, data: bytes, offset: int) -> int:
        return int(struct.unpack_from("<h", data, offset)[0])

    def _f32(self, data: bytes, offset: int) -> float:
        return float(struct.unpack_from("<f", data, offset)[0])

    def _text(self, data: bytes | str) -> str:
        if isinstance(data, str):
            text = data
        else:
            text = data.split(b"\x00", 1)[0].decode("latin-1", errors="ignore")
        return text.strip()

    def _date_time(self, date: str, time: str) -> str:
        date = self._text(date)
        time = self._text(time)
        if date and time:
            return f"{date} {time}"
        return date or time

    def _parse_temperature(self, text: str) -> float:
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", str(text))
        return float(match.group(1)) if match else 0.0

    def _parse_field_gauss(self, text: str) -> float:
        raw = str(text).strip()
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", raw)
        if not match:
            return 0.0
        value = float(match.group(1))
        token = raw.lower()
        if "t" in token and "g" not in token:
            return value * 1.0e4
        return value

    def _resize_ints(
        self,
        values: list[int],
        length: int,
        default: int,
        *,
        max_value: int,
    ) -> list[int]:
        out = list(values[:length])
        if len(out) < length:
            out.extend([default] * (length - len(out)))
        return [max(0, min(int(max_value), int(v))) for v in out]

    def _guess_psi_instrument(self, path: Path) -> tuple[str, str, str]:
        token = path.stem.lower()
        mapping = {
            "gps": ("GPS", "piM3.2", "continuous surface muon source"),
            "ltf": ("LTF", "piM3.3", "continuous surface muon source"),
            "gpd": ("GPD", "muE1", "continuous decay channel muon source"),
            "dolly": ("DOLLY", "piE1", "continuous surface muon source"),
            "alc": ("ALC", "piE3", "continuous surface muon source"),
            "hifi": ("HIFI", "piE3", "continuous surface muon source"),
        }
        for key, values in mapping.items():
            if f"_{key}_" in token or token.startswith(f"{key}_") or token.endswith(f"_{key}"):
                return values
        return "PSI", "n/a", "continuous muon source"
