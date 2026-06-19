"""Loader for ISIS muon NeXus files (legacy V1 and modern V2).

This module implements a pure-Python NeXus reader using ``h5py``. The
implementation is intentionally independent of Mantid code so that Asymmetry
can stay MIT-licensed while still supporting ISIS muon NeXus layouts.

Supported families
------------------
* Legacy V1 layout (``/run/...``)
* Modern V2 layout (typically ``/raw_data_1/...``)

Both single-period and multi-period files are supported. Two-period files are
loaded as a single dataset and retain both period histograms for red/green
mode selection in the grouping workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.base import BaseLoader, LoadResult
from asymmetry.core.io.hdf4 import is_hdf4, open_hdf4
from asymmetry.core.io.periods import combine_mapped_periods
from asymmetry.core.transform import apply_grouping, compute_asymmetry

try:  # optional dependency
    import h5py  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when h5py is not installed
    h5py = None


#: Unit tokens (lower-cased, with degree signs / spaces / dots stripped) that a
#: NeXus ``units`` attribute may use to name the Celsius scale. ``celcius`` is a
#: common misspelling seen in the wild.
_CELSIUS_UNIT_TOKENS = frozenset(
    {
        "c",
        "degc",
        "celsius",
        "celcius",
        "centigrade",
        "degreec",
        "degreecelsius",
        "degreescelsius",
        "degcelsius",
    }
)
_ABSOLUTE_ZERO_CELSIUS = 273.15

#: EMU's sample cryostat tops out near room temperature; a reading above this
#: ceiling implies a furnace/oven, whose NeXus header is the known unit-mislabel
#: risk (Celsius values stored under a ``Kelvin`` label). 320 K leaves a ~20 K
#: margin above room temperature so ordinary cold runs are never flagged. See
#: :meth:`NexusLoader._temperature_unit_suspect`.
_EMU_FURNACE_SUSPECT_CEILING_K = 320.0


def _is_celsius_unit(units: str | None) -> bool:
    """True when a NeXus ``units`` token names the Celsius scale.

    Normalisation strips degree signs, spaces, dots and underscores so
    ``°C`` / ``deg C`` / ``Celsius`` all match the tokens in
    :data:`_CELSIUS_UNIT_TOKENS`.
    """
    token = (
        str(units or "")
        .strip()
        .lower()
        .replace("°", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("_", "")
    )
    return token in _CELSIUS_UNIT_TOKENS


def _normalize_temperature_to_kelvin(value: float | None, units: str | None) -> float | None:
    """Convert a temperature to kelvin, honoring the NeXus ``units`` attribute.

    A Celsius unit (``degC`` / ``°C`` / ``Celsius``) shifts the value by
    ``+273.15``; Kelvin, an empty unit, or any unrecognized unit is passed
    through unchanged — we never *guess* a conversion the file did not declare.
    ``None`` propagates as ``None``.

    Note this honors the *declared* unit only. A file that stores Celsius values
    but mislabels the field ``Kelvin`` (seen on some EMU furnace runs) is left
    as-is; silently "correcting" it would corrupt genuinely-cold Kelvin runs.
    The suspected-mislabel case is *surfaced* (not converted) via
    :meth:`NexusLoader._temperature_unit_suspect`.
    """
    if value is None:
        return None
    if _is_celsius_unit(units):
        return float(value) + _ABSOLUTE_ZERO_CELSIUS
    return float(value)


def active_series_mean(entry: Any) -> float | None:
    """Mean of a logged NXlog series over its run-active (t >= 0) samples.

    The stored ``mean`` / ``min`` / ``max`` summarise the *whole* record,
    including the pre-run (t < 0) plateau — so the first run of a setpoint block
    reads the previous setpoint (Sn 91516 -> 4.62 K vs the correct 1.599 K).
    When the series carries a time axis, average only the t >= 0 samples;
    otherwise fall back to the precomputed full-record ``mean``.

    Pure (no Qt) so the loader (``sample_temperature_logged``) and the GUI Data
    Browser share one definition of the run-active mean and never disagree.
    """
    if not isinstance(entry, dict):
        return None
    times = entry.get("time")
    values = entry.get("values")
    if isinstance(times, (list, tuple)) and isinstance(values, (list, tuple)) and times and values:
        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        n = min(t.size, v.size)
        if n:
            t, v = t[:n], v[:n]
            active = v[(t >= 0.0) & np.isfinite(v)]
            if active.size:
                return float(np.mean(active))
    try:
        mean = float(entry.get("mean"))
    except (TypeError, ValueError):
        return None
    return mean if np.isfinite(mean) else None


@dataclass
class _GroupingSelection:
    """Resolved detector-group selection used for asymmetry reduction."""

    forward_indices: list[int]
    backward_indices: list[int]
    groups: dict[int, list[int]]
    forward_group_id: int
    backward_group_id: int


class NexusLoader(BaseLoader):
    """Read ISIS muon ``.nxs`` files and return one or more datasets."""

    extensions = [".nxs", ".nexus"]
    format_name = "ISIS NeXus (.nxs, .nexus)"

    def load(self, filepath: str) -> LoadResult:
        """Load a NeXus file and return reduced asymmetry dataset(s).

        Parameters
        ----------
        filepath
            Path to a NeXus file.

        Returns
        -------
        MuonDataset or list[MuonDataset]
            Single dataset for single-period and two-period files. Files with
            more than two periods return one dataset per period.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # ``.nxs``/``.nexus`` share an extension across two containers: modern
        # HDF5 (h5py) and legacy HDF4 (the v1 ``/run`` muonTD format WiMDA
        # reads). The file magic disambiguates; h5py cannot open HDF4 at all.
        if is_hdf4(str(path)):
            handle = open_hdf4(str(path))  # raises ImportError if pyhdf absent
            result = self._reduce_handle(handle, str(path))
        else:
            self._require_h5py()
            with h5py.File(path, "r") as handle:
                result = self._reduce_handle(handle, str(path))

        if len(result) == 1:
            return result[0]
        return result

    def _reduce_handle(self, handle: Any, source_file: str) -> list[MuonDataset]:
        """Detect the layout behind a handle and reduce to dataset(s).

        The handle is either an ``h5py.File`` or the HDF4 adapter from
        :func:`asymmetry.core.io.hdf4.open_hdf4`; both expose the same
        read-only surface, so ``_detect_layout`` / ``_load_v1`` / ``_load_v2``
        are container-agnostic.
        """
        version, entry = self._detect_layout(handle)
        if version == "v1":
            return self._load_v1(handle, entry, source_file)
        return self._load_v2(handle, entry, source_file)

    def _require_h5py(self) -> None:
        """Ensure ``h5py`` is available before trying to read HDF5 content."""
        if h5py is None:
            raise ImportError(
                "h5py is required for NeXus support. Install with "
                "'pip install h5py' or 'pip install asymmetry[hdf5]'."
            )

    def _detect_layout(self, handle: Any) -> tuple[str, str]:
        """Detect whether a file is V1 or V2 and return the selected entry name."""
        if "run" in handle:
            run = handle["run"]
            analysis = self._safe_str(self._read_optional(run, "analysis"))
            idf_v = self._safe_int(self._read_optional(run, "IDF_version"))
            if idf_v is None:
                idf_v = self._safe_int(self._read_optional(run, "idf_version"))
            if analysis in {"muonTD", "pulsedTD"} or idf_v == 1:
                return "v1", "run"

        for key in handle.keys():
            node = handle.get(key)
            if not hasattr(node, "keys"):
                continue
            definition = self._safe_str(self._read_optional(node, "definition"))
            idf_v = self._safe_int(self._read_optional(node, "IDF_version"))
            if idf_v is None:
                idf_v = self._safe_int(self._read_optional(node, "idf_version"))
            if definition in {"muonTD", "pulsedTD"} or idf_v == 2:
                return "v2", key

        raise ValueError("Unsupported NeXus file: could not detect ISIS muon V1/V2 layout")

    def _load_v1(self, handle: Any, entry_name: str, source_file: str) -> list[MuonDataset]:
        """Read legacy ``/run`` muon NeXus content and reduce to asymmetry."""
        entry = handle[entry_name]
        h_data = self._require_group(entry, "histogram_data_1")

        counts = np.asarray(self._require_dataset(h_data, "counts"), dtype=np.float64)
        # Legacy v1 files store multi-period histograms as a single flat
        # ``[n_periods * n_spectra, n_bins]`` block; ``switching_states`` gives
        # the period count (e.g. HiFi RF/ALC runs: 64 = 2 x 32). Reshape
        # period-major — identical to the v1->v2 converter — so periods split
        # the same way the modern v2 ``[n_periods, n_spectra, n_bins]`` layout
        # already does. Single-period files (absent or 1) are untouched.
        n_switching = self._safe_int(self._read_optional(entry, "switching_states"), default=1) or 1
        if counts.ndim == 2 and n_switching > 1 and counts.shape[0] % n_switching == 0:
            n_spectra = counts.shape[0] // n_switching
            counts = counts.reshape(n_switching, n_spectra, counts.shape[-1])
        counts_periods = self._split_period_counts(counts)

        corrected_time = np.asarray(
            self._read_optional(h_data, "corrected_time", default=[]),
            dtype=np.float64,
        )
        grouping_array = np.asarray(
            self._read_optional(h_data, "grouping", default=[]),
            dtype=np.int64,
        )
        good_frames_values = np.asarray(
            self._read_optional(entry, "good_frames", default=[]),
            dtype=np.float64,
        )
        if good_frames_values.size == 0:
            good_frames_values = np.asarray(
                self._read_optional(entry, "goodfrm", default=[]),
                dtype=np.float64,
            )
        if good_frames_values.size == 0:
            good_frames_values = np.asarray(
                self._read_optional(
                    self._read_optional(entry, "periods"), "good_frames", default=[]
                ),
                dtype=np.float64,
            )
        dead_time_values = np.asarray(
            self._read_optional(h_data, "dead_time", default=[]),
            dtype=np.float64,
        )
        if dead_time_values.size == 0:
            dead_time_values = np.asarray(
                self._read_optional(h_data, "deadtime", default=[]),
                dtype=np.float64,
            )
        # Legacy ISIS muon NeXus v1 files (and the HDF4 originals read directly)
        # store the dead-time table at ``instrument/detector/deadtimes`` (plural,
        # a different group) rather than under ``histogram_data_1``. The nxs4to5
        # converter maps that location to ``detector_1/dead_time`` (read by
        # _load_v2), so HDF5 twins are fine; a directly loaded v1 file needs this
        # fallback or it returns an all-zeros table. (deadtimes is the real key;
        # dead_time/deadtime are accepted defensively.)
        for dead_time_key in ("deadtimes", "dead_time", "deadtime"):
            if dead_time_values.size:
                break
            dead_time_values = np.asarray(
                self._read_optional(entry, f"instrument/detector/{dead_time_key}", default=[]),
                dtype=np.float64,
            )
        time_zero_values = np.asarray(
            self._read_optional(h_data, "time_zero", default=[]),
            dtype=np.float64,
        )

        run_number = self._safe_int(self._read_optional(entry, "number"), default=0)
        instrument_name = self._safe_str(
            self._read_optional(self._read_optional(entry, "instrument"), "name")
        )

        sample = self._read_optional(entry, "sample")
        temperature = self._read_temperature_kelvin(sample)
        temperature_units = self._sample_temperature_units(sample)
        magnetic_field = self._safe_float(
            self._read_optional(sample, "magnetic_field"), default=0.0
        )

        orientation_raw = self._safe_str(
            self._read_optional(self._read_optional(entry, "instrument"), "detector/orientation")
        )
        detector_orientation = self._normalise_orientation(orientation_raw)
        field_state = self._normalise_field_state(
            self._safe_str(self._read_optional(sample, "magnetic_field_state"))
        )
        field_direction = self._field_direction_from_state(field_state)

        metadata_base = {
            "run_number": run_number,
            "title": self._safe_str(self._read_optional(entry, "title")),
            "comment": self._safe_str(self._read_optional(entry, "notes")),
            "started": self._safe_str(self._read_optional(entry, "start_time")),
            "stopped": self._safe_str(self._read_optional(entry, "stop_time")),
            "temperature": temperature,
            "field": magnetic_field,
            "instrument": instrument_name,
            "field_direction": field_direction,
            "field_state": field_state,
            "detector_orientation": detector_orientation,
            "source_file": source_file,
            "nexus_version": "v1",
        }

        nexus_fields = self._extract_tree(entry)
        time_series = self._extract_time_series(entry)
        metadata_base["nexus_fields"] = nexus_fields
        metadata_base["nexus_time_series"] = time_series
        logged_temperature = self._logged_sample_temperature(time_series)
        if logged_temperature is not None:
            metadata_base["sample_temperature_logged"] = logged_temperature

        suspect, reason = self._temperature_unit_suspect(
            instrument_name, temperature, logged_temperature, temperature_units
        )
        if suspect:
            metadata_base["temperature_unit_suspect"] = True
            metadata_base["temperature_unit_suspect_reason"] = reason

        # Real ISIS v1 files carry the good-data window and t0 as attributes on
        # the ``counts`` SDS (1-based ``t0_bin`` / ``first_good_bin`` /
        # ``last_good_bin``), exactly as the v2 layout does — not as child
        # datasets of ``histogram_data_1``. Prefer those attributes; fall back
        # to child datasets / defaults for synthetic or attribute-less files.
        n_detectors, n_bins = counts_periods[0].shape
        counts_ds = h_data.get("counts")
        counts_attrs = getattr(counts_ds, "attrs", {}) if counts_ds is not None else {}

        t0_bin_values = self._t0_bin_values_from_attr(
            counts_attrs.get("t0_bin"), n_detectors=n_detectors
        )

        first_good_bin_raw = self._safe_int(counts_attrs.get("first_good_bin"), default=None)
        if first_good_bin_raw is None:
            first_good_bin_raw = self._safe_int(
                self._read_optional(h_data, "first_good_bin"), default=None
            )
        last_good_bin_raw = self._safe_int(counts_attrs.get("last_good_bin"), default=None)
        if last_good_bin_raw is None:
            last_good_bin_raw = self._safe_int(
                self._read_optional(h_data, "last_good_bin"), default=None
            )

        first_good_bin = 0 if first_good_bin_raw is None else int(first_good_bin_raw)
        last_good_bin = (n_bins - 1) if last_good_bin_raw is None else int(last_good_bin_raw)

        # Normalize 1-based bin metadata to 0-based using the corrected-time
        # axis, sharing the v2 inference so both layouts agree on the window.
        reference_axis, _ = self._build_time_axis(corrected_time, n_bins)
        index_offset = self._infer_v2_bin_index_offset(reference_axis, t0_bin_values)
        if index_offset:
            if t0_bin_values is not None:
                t0_bin_values = np.maximum(0, t0_bin_values - index_offset)
            first_good_bin = max(0, first_good_bin - index_offset)
            last_good_bin = max(0, last_good_bin - index_offset)

        return self._build_period_datasets(
            counts_periods=counts_periods,
            time_axis_source=corrected_time,
            axis_needs_time_zero_correction=False,
            grouping_array=grouping_array,
            good_frames_values=good_frames_values,
            dead_time_values=dead_time_values,
            time_zero_values=time_zero_values,
            time_zero_is_microseconds=False,
            t0_bin_values=t0_bin_values,
            metadata_base=metadata_base,
            run_number=run_number,
            first_good_bin=first_good_bin,
            last_good_bin=last_good_bin,
            bin_index_base=int(index_offset),
            source_file=source_file,
        )

    def _load_v2(self, handle: Any, entry_name: str, source_file: str) -> list[MuonDataset]:
        """Read modern V2 muon NeXus content and reduce to asymmetry.

        V2 files may contain both ``raw_time`` and ``corrected_time``.
        Mantid's loading flow uses ``raw_time`` and applies ``time_zero`` as a
        separate correction. To align user-visible behaviour with both Mantid
        and files that already provide corrected centres:

        * Prefer ``corrected_time`` when it has a shape compatible with counts.
        * Otherwise, build from ``raw_time`` and subtract ``time_zero``.
        * For per-histogram metadata, prefer ``counts.attrs['t0_bin']`` when
          present, otherwise derive ``t0_bin`` from ``time_zero / bin_width``.
        """
        entry = handle[entry_name]
        detector = self._require_group(self._require_group(entry, "instrument"), "detector_1")

        counts = np.asarray(self._require_dataset(detector, "counts"), dtype=np.float64)
        counts_periods = self._split_period_counts(counts)

        raw_time = np.asarray(
            self._read_optional(detector, "raw_time", default=[]), dtype=np.float64
        )
        if raw_time.size == 0:
            raw_time = np.asarray(
                self._read_optional(detector, "time_of_flight", default=[]), dtype=np.float64
            )
        corrected_time = np.asarray(
            self._read_optional(detector, "corrected_time", default=[]), dtype=np.float64
        )

        grouping_array = np.asarray(
            self._read_optional(detector, "grouping", default=[]), dtype=np.int64
        )
        good_frames_values = np.asarray(
            self._read_optional(entry, "good_frames", default=[]),
            dtype=np.float64,
        )
        if good_frames_values.size == 0:
            good_frames_values = np.asarray(
                self._read_optional(
                    self._read_optional(entry, "periods"), "good_frames", default=[]
                ),
                dtype=np.float64,
            )
        if good_frames_values.size == 0:
            good_frames_values = np.asarray(
                self._read_optional(entry, "goodfrm", default=[]),
                dtype=np.float64,
            )
        dead_time_values = np.asarray(
            self._read_optional(detector, "dead_time", default=[]),
            dtype=np.float64,
        )
        # The converter writes ``dead_time``; accept the ``deadtime``/``deadtimes``
        # spellings too so a hand-made or partially-converted v2 file still reads.
        for dead_time_key in ("deadtime", "deadtimes"):
            if dead_time_values.size:
                break
            dead_time_values = np.asarray(
                self._read_optional(detector, dead_time_key, default=[]),
                dtype=np.float64,
            )
        time_zero_values = np.asarray(
            self._read_optional(detector, "time_zero", default=[]), dtype=np.float64
        )

        run_number = self._safe_int(self._read_optional(entry, "run_number"), default=0)
        title = self._safe_str(self._read_optional(entry, "title"))
        started = self._safe_str(self._read_optional(entry, "start_time"))
        stopped = self._safe_str(self._read_optional(entry, "end_time"))
        instrument_name = self._safe_str(self._read_optional(entry, "name"))
        if not instrument_name:
            instrument_name = self._safe_str(
                self._read_optional(self._read_optional(entry, "instrument"), "name")
            )

        sample = self._read_optional(entry, "sample")
        temperature = self._read_temperature_kelvin(sample)
        temperature_units = self._sample_temperature_units(sample)
        magnetic_field = self._safe_float(
            self._read_optional(sample, "magnetic_field"), default=0.0
        )

        orientation_raw = self._safe_str(self._read_optional(detector, "orientation"))
        detector_orientation = self._normalise_orientation(orientation_raw)
        field_state = self._normalise_field_state(
            self._safe_str(self._read_optional(sample, "magnetic_field_state"))
        )
        field_direction = self._field_direction_from_state(field_state)

        counts_ds = detector.get("counts")
        n_bins = int(counts_periods[0].shape[-1])
        use_corrected_time = corrected_time.size in {n_bins, n_bins + 1}
        time_axis_source = corrected_time if use_corrected_time else raw_time
        axis_needs_time_zero_correction = not use_corrected_time

        t0_bin_values: np.ndarray | None = None
        if counts_ds is not None:
            t0_bin_values = self._t0_bin_values_from_attr(
                getattr(counts_ds, "attrs", {}).get("t0_bin"),
                n_detectors=counts_periods[0].shape[0],
            )

        first_good_bin_raw: int | None = None
        last_good_bin_raw: int | None = None
        if counts_ds is not None:
            attrs = getattr(counts_ds, "attrs", {})
            first_good_bin_raw = self._safe_int(attrs.get("first_good_bin"), default=None)
            last_good_bin_raw = self._safe_int(attrs.get("last_good_bin"), default=None)

        first_good_bin = 0 if first_good_bin_raw is None else int(first_good_bin_raw)
        last_good_bin_default = counts_periods[0].shape[-1] - 1
        last_good_bin = (
            last_good_bin_default if last_good_bin_raw is None else int(last_good_bin_raw)
        )
        first_good_time = self._safe_float(
            self._read_optional(detector, "first_good_time"),
            default=None,
        )
        last_good_time = self._safe_float(
            self._read_optional(detector, "last_good_time"),
            default=None,
        )

        metadata_base = {
            "run_number": run_number,
            "title": title,
            "comment": self._safe_str(self._read_optional(entry, "notes")),
            "started": started,
            "stopped": stopped,
            "temperature": temperature,
            "field": magnetic_field,
            "instrument": instrument_name,
            "field_direction": field_direction,
            "field_state": field_state,
            "detector_orientation": detector_orientation,
            "source_file": source_file,
            "nexus_version": "v2",
        }

        periods_group = self._read_optional(entry, "periods")
        if periods_group is not None:
            metadata_base["period_count"] = self._safe_int(
                self._read_optional(periods_group, "number"),
                default=len(counts_periods),
            )

        reference_axis, _ = self._build_time_axis(time_axis_source, n_bins)
        if axis_needs_time_zero_correction:
            reference_axis = reference_axis - self._global_time_zero_value(time_zero_values)

        index_offset = self._infer_v2_bin_index_offset(reference_axis, t0_bin_values)
        if index_offset:
            if t0_bin_values is not None:
                t0_bin_values = np.maximum(0, t0_bin_values - index_offset)
            first_good_bin = max(0, first_good_bin - index_offset)
            last_good_bin = max(0, last_good_bin - index_offset)

        # Keep integer bin metadata as the canonical source of the good-data
        # window. Floating-point good-time values are used only as a fallback
        # when the file does not provide usable integer bin attributes.
        use_first_good_time = first_good_bin_raw is None and first_good_time is not None
        use_last_good_time = last_good_bin_raw is None and last_good_time is not None

        nexus_fields = self._extract_tree(entry)
        time_series = self._extract_time_series(entry)
        metadata_base["nexus_fields"] = nexus_fields
        metadata_base["nexus_time_series"] = time_series
        logged_temperature = self._logged_sample_temperature(time_series)
        if logged_temperature is not None:
            metadata_base["sample_temperature_logged"] = logged_temperature

        suspect, reason = self._temperature_unit_suspect(
            instrument_name, temperature, logged_temperature, temperature_units
        )
        if suspect:
            metadata_base["temperature_unit_suspect"] = True
            metadata_base["temperature_unit_suspect_reason"] = reason

        return self._build_period_datasets(
            counts_periods=counts_periods,
            time_axis_source=time_axis_source,
            axis_needs_time_zero_correction=axis_needs_time_zero_correction,
            grouping_array=grouping_array,
            good_frames_values=good_frames_values,
            dead_time_values=dead_time_values,
            time_zero_values=time_zero_values,
            time_zero_is_microseconds=True,
            t0_bin_values=t0_bin_values,
            metadata_base=metadata_base,
            run_number=run_number,
            first_good_bin=first_good_bin,
            last_good_bin=last_good_bin,
            first_good_time=first_good_time,
            last_good_time=last_good_time,
            use_first_good_time=use_first_good_time,
            use_last_good_time=use_last_good_time,
            bin_index_base=int(index_offset),
            source_file=source_file,
        )

    def _build_period_datasets(
        self,
        *,
        counts_periods: list[np.ndarray],
        time_axis_source: np.ndarray,
        axis_needs_time_zero_correction: bool,
        grouping_array: np.ndarray,
        good_frames_values: np.ndarray,
        dead_time_values: np.ndarray,
        time_zero_values: np.ndarray,
        time_zero_is_microseconds: bool,
        t0_bin_values: np.ndarray | None,
        metadata_base: dict[str, Any],
        run_number: int,
        first_good_bin: int,
        last_good_bin: int,
        first_good_time: float | None = None,
        last_good_time: float | None = None,
        use_first_good_time: bool = False,
        use_last_good_time: bool = False,
        bin_index_base: int = 0,
        source_file: str,
    ) -> list[MuonDataset]:
        """Construct one :class:`MuonDataset` per period from detector counts.

        Parameters
        ----------
        time_axis_source
            Axis array used to build the dataset time values. For V1 this is
            ``corrected_time`` from file. For V2 this is either file
            ``corrected_time`` (when trustworthy) or ``raw_time``.
        axis_needs_time_zero_correction
            If ``True``, subtract the global ``time_zero`` value from the
            built axis so V2 raw-time paths align with Mantid behaviour.
        time_zero_is_microseconds
            Controls interpretation of ``time_zero_values`` when deriving
            histogram ``t0_bin`` values.
        t0_bin_values
            Optional explicit per-detector t0 bins (for example from
            ``counts.attrs['t0_bin']``). When provided these values take
            precedence over conversion from ``time_zero_values``.
        """
        datasets: list[MuonDataset] = []
        n_periods = len(counts_periods)

        dead_time_periods = self._split_period_vectors(
            dead_time_values,
            n_periods=n_periods,
            n_detectors=counts_periods[0].shape[0],
        )
        good_frames_periods = self._split_period_scalars(
            good_frames_values,
            n_periods=n_periods,
            default=1.0,
        )

        for period_idx, period_counts in enumerate(counts_periods, start=1):
            if period_counts.ndim != 2:
                raise ValueError("Detector counts must be 2D [n_detectors, n_bins] per period")

            n_detectors, n_bins = period_counts.shape
            time_axis, bin_width = self._build_time_axis(time_axis_source, n_bins)
            if axis_needs_time_zero_correction:
                time_zero_us = self._global_time_zero_value(time_zero_values)
                time_axis = time_axis - time_zero_us

            grouping = self._resolve_grouping(grouping_array, n_detectors)
            forward = apply_grouping(
                [
                    Histogram(counts=period_counts[i], bin_width=bin_width)
                    for i in range(n_detectors)
                ],
                grouping.forward_indices,
            )
            backward = apply_grouping(
                [
                    Histogram(counts=period_counts[i], bin_width=bin_width)
                    for i in range(n_detectors)
                ],
                grouping.backward_indices,
            )

            alpha = 1.0
            asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)

            # Asymmetry works in percent throughout Asymmetry/WiMDA-style UI.
            asymmetry = asymmetry * 100.0
            error = error * 100.0

            lo, hi = self._resolve_good_bin_range(
                time_axis,
                len(asymmetry),
                first_good_bin=first_good_bin,
                last_good_bin=last_good_bin,
                first_good_time=first_good_time,
                last_good_time=last_good_time,
                use_first_good_time=use_first_good_time,
                use_last_good_time=use_last_good_time,
            )
            if lo <= hi:
                time_axis = time_axis[lo : hi + 1]
                asymmetry = asymmetry[lo : hi + 1]
                error = error[lo : hi + 1]

            histograms = self._build_histograms(
                period_counts,
                bin_width,
                time_zero_values,
                time_zero_is_microseconds=time_zero_is_microseconds,
                t0_bin_values=t0_bin_values,
                first_good_bin=first_good_bin,
                last_good_bin=last_good_bin,
            )

            period_run_number = run_number
            run_label = str(run_number)
            if n_periods > 1:
                period_run_number = self._encode_period_run_number(run_number, period_idx)
                run_label = f"{run_number}/{period_idx}"

            run_meta = dict(metadata_base)
            run_meta["run_number"] = period_run_number
            run_meta["source_run_number"] = run_number
            run_meta["run_label"] = run_label
            run_meta["period_number"] = period_idx
            run_meta["period_count"] = n_periods

            run = Run(
                run_number=period_run_number,
                histograms=histograms,
                metadata=run_meta,
                grouping={
                    "groups": {
                        gid: [idx + 1 for idx in dets] for gid, dets in grouping.groups.items()
                    },
                    "forward_group": grouping.forward_group_id,
                    "backward_group": grouping.backward_group_id,
                    "alpha": alpha,
                    "first_good_bin": int(first_good_bin),
                    "last_good_bin": int(last_good_bin),
                    "t0_bin": int(histograms[0].t0_bin) if histograms else 0,
                    "t_good_offset": max(0, int(first_good_bin) - int(histograms[0].t0_bin))
                    if histograms
                    else 0,
                    "bin_index_base": 1 if int(bin_index_base) == 1 else 0,
                    "bunching_factor": 1,
                    "deadtime_correction": False,
                    "good_frames": float(good_frames_periods[period_idx - 1]),
                    "dead_time_us": [
                        float(v)
                        for v in np.asarray(
                            dead_time_periods[period_idx - 1], dtype=np.float64
                        ).tolist()
                    ],
                },
                source_file=source_file,
            )

            datasets.append(
                MuonDataset(
                    time=np.asarray(time_axis, dtype=np.float64),
                    asymmetry=np.asarray(asymmetry, dtype=np.float64),
                    error=np.asarray(error, dtype=np.float64),
                    metadata=run_meta,
                    run=run,
                )
            )

        if n_periods == 2 and len(datasets) == 2:
            return [
                self._combine_two_period_datasets(
                    datasets,
                    source_run_number=run_number,
                    source_file=source_file,
                )
            ]
        return datasets

    def _combine_two_period_datasets(
        self,
        period_datasets: list[MuonDataset],
        *,
        source_run_number: int,
        source_file: str,
    ) -> MuonDataset:
        """Merge two period datasets into one run with red/green period metadata.

        Delegates the count / good-frame / deadtime assembly to the shared
        :func:`asymmetry.core.io.periods.combine_mapped_periods` with the
        trivial ``{1: red, 2: green}`` mapping, so the loader's two-period
        combination and the N-period "Map periods…" reducer are one code path.
        It then caches each period's loader-default reduced arrays
        (``period_reduced``) so the scriptable period-selection API
        (:func:`asymmetry.core.io.periods.select_period`) can return exactly
        what the loader produced per period without redoing the good-bin
        windowing — the mapped reducer drops that cache because summed sets no
        longer correspond to a single per-period reduction.
        """
        combined = combine_mapped_periods(
            period_datasets,
            {1: "red", 2: "green"},
            source_run_number=int(source_run_number),
            source_file=source_file,
        )
        combined.run.grouping["period_reduced"] = [
            (
                np.asarray(ds.time, dtype=np.float64).copy(),
                np.asarray(ds.asymmetry, dtype=np.float64).copy(),
                np.asarray(ds.error, dtype=np.float64).copy(),
            )
            for ds in period_datasets
        ]
        return combined

    def _build_histograms(
        self,
        period_counts: np.ndarray,
        bin_width: float,
        time_zero_values: np.ndarray,
        *,
        time_zero_is_microseconds: bool,
        t0_bin_values: np.ndarray | None,
        first_good_bin: int,
        last_good_bin: int,
    ) -> list[Histogram]:
        """Create per-detector :class:`Histogram` objects for a period.

        ``time_zero`` may be stored either as a bin index (legacy V1) or as a
        time value in microseconds (V2). This method supports both forms and
        accepts an explicit ``t0_bin_values`` override from NeXus attributes
        when available.
        """
        histograms: list[Histogram] = []
        for i in range(period_counts.shape[0]):
            t0_bin = 0

            if t0_bin_values is not None and t0_bin_values.size == period_counts.shape[0]:
                t0_bin = int(t0_bin_values[i])
            else:
                t0_value = 0.0
                if time_zero_values.size == period_counts.shape[0]:
                    t0_value = float(time_zero_values[i])
                elif time_zero_values.size > 0:
                    t0_value = float(time_zero_values.flat[0])

                if np.isfinite(t0_value):
                    if time_zero_is_microseconds:
                        if np.isfinite(bin_width) and bin_width != 0.0:
                            t0_bin = int(round(t0_value / bin_width))
                    else:
                        t0_bin = int(round(t0_value))

            histograms.append(
                Histogram(
                    counts=np.asarray(period_counts[i], dtype=np.float64),
                    bin_width=float(bin_width),
                    t0_bin=t0_bin,
                    good_bin_start=int(first_good_bin),
                    good_bin_end=int(last_good_bin),
                )
            )
        return histograms

    def _build_time_axis(self, source_axis: np.ndarray, n_bins: int) -> tuple[np.ndarray, float]:
        """Build a usable time axis and bin width from NeXus time datasets."""
        if source_axis.size == n_bins + 1:
            axis = 0.5 * (source_axis[:-1] + source_axis[1:])
        elif source_axis.size >= n_bins:
            axis = source_axis[:n_bins]
        else:
            axis = np.arange(n_bins, dtype=np.float64)

        if axis.size >= 2:
            bin_width = float(np.nanmedian(np.diff(axis)))
        else:
            bin_width = 1.0
        if not np.isfinite(bin_width) or bin_width == 0.0:
            bin_width = 1.0
        return np.asarray(axis, dtype=np.float64), float(bin_width)

    def _global_time_zero_value(self, time_zero_values: np.ndarray) -> float:
        """Return a single global t0 value in microseconds.

        For grouped asymmetry a single axis is used, so this method selects the
        first available ``time_zero`` value when present and falls back to 0.0.
        """
        if time_zero_values.size == 0:
            return 0.0
        candidate = float(time_zero_values.flat[0])
        if not np.isfinite(candidate):
            return 0.0
        return candidate

    def _resolve_good_bin_range(
        self,
        time_axis: np.ndarray,
        n_bins: int,
        *,
        first_good_bin: int,
        last_good_bin: int,
        first_good_time: float | None,
        last_good_time: float | None,
        use_first_good_time: bool,
        use_last_good_time: bool,
    ) -> tuple[int, int]:
        """Resolve inclusive good-bin limits for a reduced dataset.

        Integer bin metadata is canonical. Floating-point good-time metadata is
        used only when the corresponding bin attribute is missing.
        """
        lo = max(0, int(first_good_bin))
        hi = min(n_bins - 1, int(last_good_bin))
        if time_axis.size != n_bins:
            return lo, hi

        tol = 1e-12
        if n_bins >= 2:
            step = float(np.nanmedian(np.diff(time_axis)))
            if np.isfinite(step) and step != 0.0:
                tol = max(1e-12, abs(step) * 1e-6)

        if use_first_good_time and first_good_time is not None and np.isfinite(first_good_time):
            lo = int(np.searchsorted(time_axis, float(first_good_time) - tol, side="left"))
            lo = max(0, min(lo, n_bins - 1))

        if use_last_good_time and last_good_time is not None and np.isfinite(last_good_time):
            hi = int(np.searchsorted(time_axis, float(last_good_time) + tol, side="right") - 1)
            hi = max(0, min(hi, n_bins - 1))

        return lo, hi

    def _t0_bin_values_from_attr(self, t0_bin_attr: Any, *, n_detectors: int) -> np.ndarray | None:
        """Build per-detector ``t0_bin`` values from a ``counts`` attribute.

        Accepts a scalar (broadcast across detectors) or a per-detector array;
        returns ``None`` when the attribute is missing or unusable. Shared by
        the v1 and v2 layouts, which both store ``t0_bin`` on the counts SDS.
        """
        if t0_bin_attr is None:
            return None
        t0_bin_array = np.asarray(t0_bin_attr, dtype=np.float64).ravel()
        if t0_bin_array.size == 1 and np.isfinite(t0_bin_array[0]):
            return np.full(n_detectors, int(round(float(t0_bin_array[0]))), dtype=np.int64)
        if t0_bin_array.size == n_detectors:
            return np.rint(t0_bin_array).astype(np.int64)
        return None

    def _infer_v2_bin_index_offset(
        self, time_axis: np.ndarray, t0_bin_values: np.ndarray | None
    ) -> int:
        """Infer whether V2 integer bin metadata is 1-based.

        Some files encode ``t0_bin``/``first_good_bin`` using 1-based center-bin
        numbering. When the explicit ``t0_bin`` points one sample past the value
        closest to ``t = 0``, normalize all integer bin metadata by one.
        """
        if t0_bin_values is None or time_axis.size == 0:
            return 0

        flat = np.asarray(t0_bin_values, dtype=np.int64).ravel()
        if flat.size == 0:
            return 0

        tol = 1e-12
        if time_axis.size >= 2:
            step = float(np.nanmedian(np.diff(time_axis)))
            if np.isfinite(step) and step != 0.0:
                tol = max(1e-12, abs(step) * 1e-6)

        votes_for_one_based = 0
        votes_considered = 0
        for raw_idx in flat[: min(8, flat.size)]:
            idx = int(raw_idx)
            if idx <= 0 or idx >= time_axis.size:
                continue

            current = abs(float(time_axis[idx]))
            shifted = abs(float(time_axis[idx - 1]))
            if shifted + tol < current:
                votes_for_one_based += 1
                votes_considered += 1
            elif current + tol < shifted:
                votes_considered += 1

        if votes_considered > 0 and votes_for_one_based == votes_considered:
            return 1
        return 0

    def _resolve_grouping(self, grouping_array: np.ndarray, n_detectors: int) -> _GroupingSelection:
        """Resolve forward/backward detector sets from file grouping or defaults."""
        groups: dict[int, list[int]] = {}
        if grouping_array.size >= n_detectors:
            vals = np.asarray(grouping_array[:n_detectors], dtype=np.int64)
            for i, gid in enumerate(vals):
                if gid <= 0:
                    continue
                groups.setdefault(int(gid), []).append(i)

        if len(groups) >= 2:
            group_ids = sorted(groups)
            forward_group_id = int(group_ids[0])
            backward_group_id = int(group_ids[1])
            forward_indices = groups[forward_group_id]
            backward_indices = groups[backward_group_id]
        else:
            split = max(1, n_detectors // 2)
            forward_indices = list(range(0, split))
            backward_indices = list(range(split, n_detectors))
            if not backward_indices:
                backward_indices = list(range(0, n_detectors))
            groups = {1: forward_indices, 2: backward_indices}
            forward_group_id = 1
            backward_group_id = 2

        return _GroupingSelection(
            forward_indices=forward_indices,
            backward_indices=backward_indices,
            groups=groups,
            forward_group_id=forward_group_id,
            backward_group_id=backward_group_id,
        )

    def _split_period_counts(self, counts: np.ndarray) -> list[np.ndarray]:
        """Normalise detector counts into a list of ``[detectors, bins]`` arrays."""
        if counts.ndim == 2:
            return [counts]
        if counts.ndim == 3:
            return [counts[i] for i in range(counts.shape[0])]
        raise ValueError(f"Unsupported counts array shape: {counts.shape}")

    def _split_period_vectors(
        self,
        values: np.ndarray,
        *,
        n_periods: int,
        n_detectors: int,
    ) -> list[np.ndarray]:
        """Normalise optional per-detector vectors into one vector per period."""
        if values.size == 0:
            return [np.zeros(n_detectors, dtype=np.float64) for _ in range(n_periods)]

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size == n_detectors:
                return [arr.copy() for _ in range(n_periods)]
            if arr.size == n_periods * n_detectors:
                return [
                    arr[i * n_detectors : (i + 1) * n_detectors].copy() for i in range(n_periods)
                ]
            return [
                np.resize(arr, n_detectors).astype(np.float64, copy=False) for _ in range(n_periods)
            ]

        if arr.ndim >= 2:
            if arr.shape[0] == n_periods:
                return [
                    np.resize(np.asarray(arr[i], dtype=np.float64), n_detectors)
                    for i in range(n_periods)
                ]
            if arr.shape[-1] == n_detectors:
                base = np.asarray(arr.reshape(-1, n_detectors)[0], dtype=np.float64)
                return [base.copy() for _ in range(n_periods)]

        flat = np.asarray(arr, dtype=np.float64).ravel()
        resized = np.resize(flat, n_detectors).astype(np.float64, copy=False)
        return [resized.copy() for _ in range(n_periods)]

    def _split_period_scalars(
        self,
        values: np.ndarray,
        *,
        n_periods: int,
        default: float,
    ) -> list[float]:
        """Normalise optional scalar metadata into one value per period."""
        if values.size == 0:
            return [float(default) for _ in range(n_periods)]

        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size >= n_periods:
            out = arr[:n_periods]
        elif arr.size == 1:
            out = np.full(n_periods, arr[0], dtype=np.float64)
        else:
            out = np.resize(arr, n_periods)

        result: list[float] = []
        for val in out:
            f = float(val)
            result.append(f if np.isfinite(f) and f > 0.0 else float(default))
        return result

    def _encode_period_run_number(self, run_number: int, period_idx: int) -> int:
        """Encode a stable unique run number for a specific period row.

        The data browser key is integer-based. For multi-period files we keep
        the user-facing label as ``run/period`` while encoding a unique integer
        key that avoids clashes with single-period runs.
        """
        return int(run_number) * 1000 + int(period_idx)

    def _extract_tree(self, node: Any) -> Any:
        """Recursively extract NeXus group/dataset content into plain Python types."""
        if node is None:
            return None

        if hasattr(node, "dtype") and hasattr(node, "shape"):
            return self._dataset_to_python(node)

        if hasattr(node, "keys"):
            out: dict[str, Any] = {}
            attrs = self._attrs_to_python(getattr(node, "attrs", {}))
            if attrs:
                out["@attrs"] = attrs
            for key in node.keys():
                out[str(key)] = self._extract_tree(node[key])
            return out

        return None

    def _extract_time_series(self, root: Any) -> dict[str, dict[str, Any]]:
        """Collect all ``time``/``value`` NXlog-like groups for advanced display."""
        series: dict[str, dict[str, Any]] = {}

        def _walk(node: Any, prefix: str) -> None:
            if not hasattr(node, "keys"):
                return

            keys = set(map(str, node.keys()))
            has_time = "time" in keys
            value_name = "value" if "value" in keys else ("values" if "values" in keys else "")
            if has_time and value_name:
                t = np.asarray(node["time"][()])
                v = np.asarray(node[value_name][()])
                t_num = self._to_numeric_array(t)
                v_num = self._to_numeric_array(v)
                units = self._safe_str(getattr(node[value_name], "attrs", {}).get("units", ""))
                # The NXlog's human sensor label lives in a sibling ``name``
                # dataset. Native HDF4 v1 files name the Vgroup generically and
                # carry the real sensor name (e.g. ``Temp_Cryostat``) only here,
                # so the path alone does not identify the sensor; the converted
                # HDF5 twin bakes that name into the selog path instead. Capture
                # it so path-based matching works identically across containers.
                name_label = ""
                if "name" in keys:
                    name_node = node["name"]
                    if hasattr(name_node, "dtype"):  # a name dataset, not a subgroup
                        name_label = self._safe_str(name_node[()])
                if v_num.size > 0:
                    # Guard the all-NaN case: np.nanmean/nanmin/nanmax emit
                    # "Mean of empty slice"/"All-NaN slice" RuntimeWarnings when
                    # no finite values are present (seen on some ARGUS files).
                    has_finite = bool(np.isfinite(v_num).any())
                    entry: dict[str, Any] = {
                        "path": prefix,
                        "units": units,
                        "time": t_num.tolist(),
                        "values": v_num.tolist(),
                        "mean": float(np.nanmean(v_num)) if has_finite else None,
                        "min": float(np.nanmin(v_num)) if has_finite else None,
                        "max": float(np.nanmax(v_num)) if has_finite else None,
                    }
                    if name_label:
                        entry["name"] = name_label
                    series[prefix] = entry

            for child in node.keys():
                child_name = str(child)
                child_node = node[child]
                child_path = f"{prefix}/{child_name}" if prefix else child_name
                _walk(child_node, child_path)

        _walk(root, "")
        return series

    def _logged_sample_temperature(self, time_series: dict[str, dict[str, Any]]) -> float | None:
        """Return a representative *logged* sample temperature, if available.

        Unlike ``metadata['temperature']`` (the ``sample/temperature``
        setpoint), this is derived from a sample-thermometer NXlog — the actual
        recorded sample temperature, which can differ from the parked setpoint
        (e.g. CdS parks at 1 K while the sample sits near 5 K). The series mean
        over the run is used as the representative value. Returns ``None`` when
        no usable logged series is present.

        Block matching is deliberately conservative: a candidate path must name
        a *sample* thermometer (a segment containing both "sample" and "temp"),
        which catches ``Temp_Sample`` at any depth — flat as
        ``sample/Temp_Sample``, or nested on ISIS selog files as
        ``selog/Temp_Sample/value_log``. Controller / cryostat / furnace
        readbacks (``Temp_RBV``, ``Temp_Cryostat``, ``Temp_Set`` …) are **not**
        matched: an EMU furnace run that logs only those has no sample
        thermometer, so ``None`` is the honest answer rather than a guess.

        Two robustness rules:

        * The value is normalized to kelvin via the logged series' ``units``
          attribute (a Celsius log → +273.15), mirroring the setpoint path.
        * A logged sample temperature is a physical reading > 0 K. An all-zero
          series (mean 0.0 K — a disconnected/unlogged sensor, seen on some EMU
          runs) is skipped rather than reported as a misleading ``0.0``.
        """
        for path, entry in time_series.items():
            if not self._is_sample_temperature_path(path, entry):
                continue
            # Gate to run-active (t >= 0) samples so a parked pre-run plateau
            # does not contaminate the representative value (shared with the GUI).
            mean = active_series_mean(entry)
            if mean is None or not np.isfinite(mean):
                continue
            kelvin = _normalize_temperature_to_kelvin(float(mean), entry.get("units", ""))
            if kelvin is None or not np.isfinite(kelvin) or kelvin <= 0.0:
                continue
            return float(kelvin)
        return None

    @staticmethod
    def _is_sample_temperature_path(path: str, entry: Any = None) -> bool:
        """True when a log series names a sample thermometer block.

        Requires a single path segment *or the NXlog ``name`` label* to contain
        both "sample" and "temp" (case-insensitive), so ``Temp_Sample`` /
        ``sample_temperature`` match while controller readbacks like
        ``Temp_RBV`` or ``Temp_Cryostat`` do not (they lack "sample"). The
        ``name`` label is checked as an extra segment so a native HDF4 v1 log,
        whose Vgroup is generically named but whose ``name`` child is
        ``Temp_Sample``, matches just as its converted HDF5 twin (which carries
        the sensor name in the path) already does.
        """
        segments = list(str(path).split("/"))
        if isinstance(entry, dict):
            name = str(entry.get("name", "") or "")
            if name:
                segments.append(name)
        for segment in segments:
            seg = segment.lower()
            if "sample" in seg and "temp" in seg:
                return True
        return False

    def _dataset_to_python(self, dataset: Any) -> Any:
        """Convert a dataset payload into JSON-safe Python data.

        Large arrays are summarised to avoid bloating metadata while still
        exposing useful diagnostics for the advanced information view.
        """
        data = dataset[()]
        attrs = self._attrs_to_python(getattr(dataset, "attrs", {}))

        value = self._value_to_python(data)
        if attrs:
            return {"value": value, "@attrs": attrs}
        return value

    def _value_to_python(self, value: Any) -> Any:
        """Convert HDF5 scalar/array values into plain Python objects."""
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._value_to_python(value.item())

            if value.dtype.kind in {"S", "O", "U"}:
                flat = [self._safe_str(v) for v in value.ravel().tolist()]
                if len(flat) <= 64:
                    return flat
                return {
                    "kind": "array",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "preview": flat[:16],
                }

            numeric = np.asarray(value, dtype=np.float64)
            if numeric.size <= 64:
                return numeric.tolist()
            return {
                "kind": "array",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "min": float(np.nanmin(numeric)),
                "max": float(np.nanmax(numeric)),
                "mean": float(np.nanmean(numeric)),
            }

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (bytes, np.bytes_)):
            return self._safe_str(value)
        return value

    def _attrs_to_python(self, attrs: Any) -> dict[str, Any]:
        """Convert attribute mappings to JSON-safe primitives."""
        out: dict[str, Any] = {}
        if attrs is None:
            return out
        for key in attrs.keys():
            out[str(key)] = self._value_to_python(attrs[key])
        return out

    def _to_numeric_array(self, values: np.ndarray) -> np.ndarray:
        """Convert mixed/string arrays to numeric values where possible."""
        if values.size == 0:
            return np.asarray([], dtype=np.float64)
        if values.dtype.kind in {"i", "u", "f"}:
            return np.asarray(values, dtype=np.float64).ravel()

        out: list[float] = []
        for item in values.ravel().tolist():
            s = self._safe_str(item)
            try:
                out.append(float(s))
            except ValueError:
                continue
        return np.asarray(out, dtype=np.float64)

    def _normalise_orientation(self, raw: str) -> str:
        """Map short detector-bank orientation labels to user-facing text.

        This describes where the detector banks physically sit (an
        instrument-build property), not the applied-field geometry of the run.
        It is surfaced as ``detector_orientation`` and must not be conflated
        with ``field_direction`` (see docs/porting/field-geometry/).
        """
        text = (raw or "").strip().upper()
        if text.startswith("L"):
            return "Longitudinal"
        if text.startswith("T"):
            return "Transverse"
        return raw or ""

    def _normalise_field_state(self, raw: str) -> str:
        """Normalise ``sample/magnetic_field_state`` to a ``TF``/``LF``/``ZF`` code.

        Returns an empty string for blank, ``"n/a"``, or unrecognised values so
        callers treat the field geometry as unknown rather than guessing.
        """
        text = (raw or "").strip().upper()
        return text if text in {"TF", "LF", "ZF"} else ""

    def _field_direction_from_state(self, state: str) -> str:
        """Map a field-state code to a user-facing geometry, ``""`` when unknown.

        The applied-field geometry is taken solely from
        ``sample/magnetic_field_state``. Detector orientation is deliberately
        NOT used as a fallback: the banks read ``"L"`` regardless of the applied
        field, so deriving a direction from orientation would be a misleading
        guess. When the field state is absent the geometry is reported as
        unknown (empty). See docs/porting/field-geometry/ for the rationale.
        """
        return {
            "TF": "Transverse",
            "LF": "Longitudinal",
            "ZF": "Zero field",
        }.get(state, "")

    def _read_temperature_kelvin(
        self, sample: Any, name: str = "temperature", default: float = 0.0
    ) -> float:
        """Read ``sample/<name>`` as a temperature in kelvin.

        Reads both the value and its NeXus ``units`` attribute so a Celsius
        field (``degC`` / ``°C``) is normalized to kelvin via
        :func:`_normalize_temperature_to_kelvin`. A Kelvin, missing, or
        unrecognized unit passes the value through unchanged.
        """
        if sample is None or not hasattr(sample, "get"):
            return float(default)
        # Cannot use _read_optional here: it returns the unwrapped value and
        # drops the node, but we need node.attrs['units'] to decide the scale.
        node = sample.get(name)
        if node is None:
            return float(default)
        raw = node[()] if hasattr(node, "dtype") else node
        value = self._safe_float(raw, default=default)
        units = ""
        if hasattr(node, "attrs"):
            units = self._safe_str(node.attrs.get("units", ""))
        normalized = _normalize_temperature_to_kelvin(value, units)
        return float(default) if normalized is None else float(normalized)

    def _sample_temperature_units(self, sample: Any, name: str = "temperature") -> str:
        """Return the declared ``units`` attribute of ``sample/<name>`` (``""`` if none)."""
        if sample is None or not hasattr(sample, "get"):
            return ""
        node = sample.get(name)
        if node is None or not hasattr(node, "attrs"):
            return ""
        return self._safe_str(node.attrs.get("units", ""))

    def _temperature_unit_suspect(
        self,
        instrument: str | None,
        temperature: float | None,
        logged_temperature: float | None,
        units: str | None,
    ) -> tuple[bool, str]:
        """Heuristically flag an EMU furnace run whose temperature unit looks mislabelled.

        Returns ``(suspect, reason)``. A true result is only a *hint* for the GUI
        to surface — the temperature value itself is **never** changed here. We
        deliberately do not convert: see :func:`_normalize_temperature_to_kelvin`
        — silently adding 273 would corrupt a genuinely-cold Kelvin run.

        The detection is intentionally conservative; every condition must hold:

        * the instrument is **EMU** — the furnace whose NeXus header is known to
          store Celsius values under a ``Kelvin`` label;
        * the file did **not** declare a Celsius unit — a declared ``°C`` is
          already converted and trustworthy, so only a Kelvin / blank / unknown
          declaration is at risk of being a disguised Celsius reading;
        * there is **no logged sample thermometer** to corroborate the setpoint
          (``sample_temperature_logged`` is ``None``);
        * the value exceeds EMU's plausible-cryostat ceiling
          (:data:`_EMU_FURNACE_SUSPECT_CEILING_K`), i.e. it sits in furnace
          territory where the mislabel occurs.

        False positives cost nothing (a furnace run is flagged for the user to
        sanity-check its unit); false negatives just leave the existing silent
        pass-through. Below the ceiling the value is genuinely ambiguous (300 K
        room temperature vs 300 °C furnace), so we refuse to guess.
        """
        if temperature is None or not np.isfinite(temperature):
            return False, ""
        if str(instrument or "").strip().upper() != "EMU":
            return False, ""
        if _is_celsius_unit(units):
            return False, ""
        if logged_temperature is not None:
            return False, ""
        if float(temperature) <= _EMU_FURNACE_SUSPECT_CEILING_K:
            return False, ""
        reason = (
            f"EMU temperature {float(temperature):.1f} exceeds the "
            f"{_EMU_FURNACE_SUSPECT_CEILING_K:.0f} K cryostat ceiling with no logged "
            "sample thermometer; EMU furnace NeXus files are known to store °C under a "
            "'Kelvin' label, so this may be a Celsius value. Value left unchanged — verify the unit."
        )
        return True, reason

    def _read_optional(self, node: Any, name: str, default: Any = None) -> Any:
        """Read a dataset or nested path from a group-like object if present."""
        if node is None:
            return default
        if "/" in name:
            current = node
            for part in name.split("/"):
                if current is None or not hasattr(current, "get"):
                    return default
                current = current.get(part)
            if current is None:
                return default
            if hasattr(current, "dtype"):
                return current[()]
            return current

        if not hasattr(node, "get"):
            return default
        child = node.get(name)
        if child is None:
            return default
        if hasattr(child, "dtype"):
            return child[()]
        return child

    def _require_group(self, node: Any, name: str) -> Any:
        """Return required child group or raise a descriptive error."""
        group = self._read_optional(node, name)
        if group is None or not hasattr(group, "keys"):
            raise ValueError(f"NeXus file missing required group: {name}")
        return group

    def _require_dataset(self, node: Any, name: str) -> Any:
        """Return required child dataset payload or raise a descriptive error."""
        value = self._read_optional(node, name)
        if value is None:
            raise ValueError(f"NeXus file missing required dataset: {name}")
        return value

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Convert scalar HDF5 values to ``str`` while handling bytes arrays."""
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return self._safe_str(value.flat[0], default=default)
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    def _safe_int(self, value: Any, default: int | None = None) -> int | None:
        """Best-effort integer conversion for HDF5 scalar values."""
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.flat[0]
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _safe_float(self, value: Any, default: float | None = 0.0) -> float | None:
        """Best-effort float conversion for HDF5 scalar values."""
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.flat[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_attr_int(self, dataset: Any, attr_name: str, default: int) -> int:
        """Read an integer attribute from a dataset with robust conversion."""
        if dataset is None:
            return int(default)
        attrs = getattr(dataset, "attrs", {})
        raw = attrs.get(attr_name)
        converted = self._safe_int(raw)
        if converted is None:
            return int(default)
        return int(converted)
