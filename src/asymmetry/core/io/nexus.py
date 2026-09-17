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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.base import BaseLoader, LoadResult
from asymmetry.core.io.hdf4 import is_hdf4, open_hdf4
from asymmetry.core.io.icp_log import parse_icp_log_file, sibling_icp_log_path
from asymmetry.core.io.periods import combine_mapped_periods, encode_period_run_number
from asymmetry.core.transform import compute_asymmetry
from asymmetry.core.transform.grouping import apply_grouping_aligned, common_t0_for_groups
from asymmetry.core.transform.t0 import good_window_for_groups, run_t0_time_us

try:  # optional dependency
    import h5py  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when h5py is not installed
    h5py = None

logger = logging.getLogger(__name__)

#: Tolerance, in bins, for comparing ``time_zero / resolution`` against the
#: ``t0_bin`` attribute. ISIS writes both in float32, so an exact bin edge can
#: read back as 9.99999978 or 25.00001; a whole-bin disagreement is a genuine
#: header conflict (docs/porting/t0-determination/isis-header-index-base.md).
_T0_BIN_TOLERANCE = 1.0e-6


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


#: Unit tokens (lower-cased, with spaces / dots / underscores stripped) that a
#: NeXus ``sample/magnetic_field`` ``units`` attribute may use to name the tesla
#: and millitesla scales. The rest of the application treats
#: ``metadata['field']`` as gauss (the ROOT/PSI/ICP-log loaders all normalise to
#: gauss at the I/O boundary), so a declared tesla / millitesla unit is converted
#: here; an absent, blank, gauss, or unrecognised unit is left as-is.
_TESLA_UNIT_TOKENS = frozenset({"t", "tesla", "teslas"})
_MILLITESLA_UNIT_TOKENS = frozenset({"mt", "millitesla", "milliteslas"})
_TESLA_TO_GAUSS = 1.0e4
_MILLITESLA_TO_GAUSS = 10.0


def _normalize_field_to_gauss(value: float | None, units: str | None) -> float | None:
    """Convert a magnetic field to gauss, honoring the NeXus ``units`` attribute.

    A tesla unit (``T`` / ``tesla``) scales the value by ``1e4`` and a
    millitesla unit (``mT`` / ``millitesla``) by ``10``; gauss, an empty unit, or
    any unrecognized unit is passed through unchanged — we never *guess* a
    conversion the file did not declare, preserving the historical
    gauss-pass-through behaviour. ``None`` propagates as ``None``.
    """
    if value is None:
        return None
    token = str(units or "").strip().lower().replace(" ", "").replace(".", "").replace("_", "")
    if token in _TESLA_UNIT_TOKENS:
        return float(value) * _TESLA_TO_GAUSS
    if token in _MILLITESLA_UNIT_TOKENS:
        return float(value) * _MILLITESLA_TO_GAUSS
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


@dataclass
class _T0Decode:
    """Per-detector time zero decoded from an ISIS file's header (D5–D7)."""

    #: 0-based index of the bin containing t0, one per detector.
    t0_bins: np.ndarray
    #: Exact t0 in µs from acquisition start, one per detector (``None`` where
    #: the file provides no usable sub-bin value).
    t0_time_us: list[float | None]
    #: ``"file"``, ``"conflict"`` (fields disagree — the attribute wins) or
    #: ``"missing"`` (no t0 in the file at all).
    source: str


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
        if good_frames_values.size == 0:
            good_frames_values = self._good_frames_from_beam(entry)
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
        magnetic_field_node = self._read_optional(sample, "magnetic_field")
        magnetic_field = self._read_field_gauss(sample)

        orientation_raw = self._safe_str(
            self._read_optional(self._read_optional(entry, "instrument"), "detector/orientation")
        )
        detector_orientation = self._normalise_orientation(orientation_raw)
        field_state = self._normalise_field_state(
            self._safe_str(self._read_optional(sample, "magnetic_field_state"))
        )
        field_direction = self._field_direction_from_state(field_state)
        field_vector = self._read_field_vector(sample)

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
        if field_vector is not None:
            metadata_base["field_vector"] = field_vector
        self._fill_field_from_sidecar_log(
            metadata_base,
            source_file=source_file,
            field_present=magnetic_field_node is not None,
        )

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

        first_good_bin, last_good_bin = self._good_bin_window(counts_attrs, h_data, n_bins)
        _, bin_width = self._build_time_axis(corrected_time, n_bins)
        decode = self._decode_t0(
            t0_bin_attr=counts_attrs.get("t0_bin"),
            time_zero_values=time_zero_values,
            n_detectors=n_detectors,
            resolution_us=bin_width,
            corrected_time=corrected_time,
            n_bins=n_bins,
            source_file=source_file,
        )
        if corrected_time.size:
            metadata_base["nexus_corrected_time"] = [float(v) for v in corrected_time]

        return self._build_period_datasets(
            counts_periods=counts_periods,
            bin_width=bin_width,
            decode=decode,
            grouping_array=grouping_array,
            good_frames_values=good_frames_values,
            dead_time_values=dead_time_values,
            metadata_base=metadata_base,
            run_number=run_number,
            first_good_bin=first_good_bin,
            last_good_bin=last_good_bin,
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
        if good_frames_values.size == 0:
            good_frames_values = self._good_frames_from_beam(entry)
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
        magnetic_field_node = self._read_optional(sample, "magnetic_field")
        magnetic_field = self._read_field_gauss(sample)

        orientation_raw = self._safe_str(self._read_optional(detector, "orientation"))
        detector_orientation = self._normalise_orientation(orientation_raw)
        field_state = self._normalise_field_state(
            self._safe_str(self._read_optional(sample, "magnetic_field_state"))
        )
        field_direction = self._field_direction_from_state(field_state)
        field_vector = self._read_field_vector(sample)

        counts_attrs = getattr(detector.get("counts"), "attrs", {})
        n_bins = int(counts_periods[0].shape[-1])
        use_corrected_time = corrected_time.size in {n_bins, n_bins + 1}
        time_axis_source = corrected_time if use_corrected_time else raw_time

        first_good_bin_raw = self._safe_int(counts_attrs.get("first_good_bin"), default=None)
        last_good_bin_raw = self._safe_int(counts_attrs.get("last_good_bin"), default=None)
        first_good_bin, last_good_bin = self._good_bin_window(counts_attrs, None, n_bins)
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
        if field_vector is not None:
            metadata_base["field_vector"] = field_vector
        self._fill_field_from_sidecar_log(
            metadata_base,
            source_file=source_file,
            field_present=magnetic_field_node is not None,
        )

        periods_group = self._read_optional(entry, "periods")
        if periods_group is not None:
            metadata_base["period_count"] = self._safe_int(
                self._read_optional(periods_group, "number"),
                default=len(counts_periods),
            )

        _, bin_width = self._build_time_axis(time_axis_source, n_bins)
        decode = self._decode_t0(
            t0_bin_attr=counts_attrs.get("t0_bin"),
            time_zero_values=time_zero_values,
            n_detectors=counts_periods[0].shape[0],
            resolution_us=bin_width,
            corrected_time=corrected_time if use_corrected_time else np.asarray([]),
            n_bins=n_bins,
            source_file=source_file,
        )
        if use_corrected_time:
            metadata_base["nexus_corrected_time"] = [float(v) for v in corrected_time]

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
            bin_width=bin_width,
            decode=decode,
            grouping_array=grouping_array,
            good_frames_values=good_frames_values,
            dead_time_values=dead_time_values,
            metadata_base=metadata_base,
            run_number=run_number,
            first_good_bin=first_good_bin,
            last_good_bin=last_good_bin,
            first_good_time=first_good_time,
            last_good_time=last_good_time,
            use_first_good_time=use_first_good_time,
            use_last_good_time=use_last_good_time,
            source_file=source_file,
        )

    def _good_frames_from_beam(self, entry: Any) -> np.ndarray:
        """Good-frame count(s) from ``instrument/beam``.

        Legacy ISIS muon NeXus v1 files (and the HDF4 originals read directly)
        do not carry a top-level ``good_frames``/``goodfrm``; the authoritative
        good-frame count lives under ``instrument/beam`` instead. Prefer the
        per-period ``frames_period`` array so multi-period normalisation stays
        correct, then fall back to the run totals (``frames_good`` / ``frames``).

        Without this, the deadtime path defaults ``good_frames`` to ``1.0`` and
        ``prepare_histograms_with_deadtime`` over-corrects HDF4 counts by ~5
        orders of magnitude.
        """
        beam = self._read_optional(self._read_optional(entry, "instrument"), "beam")
        if beam is None:
            return np.asarray([], dtype=np.float64)
        for key in ("frames_period", "frames_good", "frames"):
            values = np.asarray(self._read_optional(beam, key, default=[]), dtype=np.float64)
            if values.size:
                return values
        return np.asarray([], dtype=np.float64)

    def _build_period_datasets(
        self,
        *,
        counts_periods: list[np.ndarray],
        bin_width: float,
        decode: _T0Decode,
        grouping_array: np.ndarray,
        good_frames_values: np.ndarray,
        dead_time_values: np.ndarray,
        metadata_base: dict[str, Any],
        run_number: int,
        first_good_bin: int,
        last_good_bin: int,
        first_good_time: float | None = None,
        last_good_time: float | None = None,
        use_first_good_time: bool = False,
        use_last_good_time: bool = False,
        source_file: str,
    ) -> list[MuonDataset]:
        """Construct one :class:`MuonDataset` per period from detector counts.

        The dataset time axis is stamped from the run's own t0 rather than the
        file's ``corrected_time``: ``(k + 0.5)·w − t0_time_us``, the bin-centre
        convention reduction uses. With no exact t0 in the file this is
        identical to ``(k − t0_bin)·w``; with one it places the axis on the
        sub-bin position the DAQ recorded. The file's ``corrected_time`` is kept
        in ``metadata["nexus_corrected_time"]`` for diagnostics.
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

            n_detectors = period_counts.shape[0]
            histograms = self._build_histograms(
                period_counts,
                bin_width,
                decode=decode,
                first_good_bin=first_good_bin,
                last_good_bin=last_good_bin,
            )

            grouping = self._resolve_grouping(grouping_array, n_detectors)
            # ISIS files normally carry one t0 for every detector, in which case
            # the aligned sum is bit-identical to the unaligned one; a
            # per-detector ``t0_bin``/``time_zero`` array is aligned like PSI's.
            common_t0 = common_t0_for_groups(
                histograms, grouping.forward_indices, grouping.backward_indices
            )
            forward = apply_grouping_aligned(
                histograms, grouping.forward_indices, common_t0_bin=common_t0
            )
            backward = apply_grouping_aligned(
                histograms, grouping.backward_indices, common_t0_bin=common_t0
            )
            n_grouped = min(len(forward), len(backward))

            alpha = 1.0
            asymmetry, error = compute_asymmetry(
                forward[:n_grouped], backward[:n_grouped], alpha=alpha
            )

            # Asymmetry works in percent throughout Asymmetry/WiMDA-style UI.
            asymmetry = asymmetry * 100.0
            error = error * 100.0

            t0_time_us = run_t0_time_us(histograms, common_t0)
            t0_stamp = (
                (float(common_t0) + 0.5) * bin_width if t0_time_us is None else float(t0_time_us)
            )
            time_axis = (np.arange(n_grouped, dtype=np.float64) + 0.5) * bin_width - t0_stamp

            group_first_good, group_last_good = self._group_good_window(
                histograms,
                sorted(set(grouping.forward_indices) | set(grouping.backward_indices)),
                common_t0=common_t0,
                n_grouped=n_grouped,
            )
            lo, hi = self._resolve_good_bin_range(
                time_axis,
                len(asymmetry),
                first_good_bin=group_first_good,
                last_good_bin=group_last_good,
                first_good_time=first_good_time,
                last_good_time=last_good_time,
                use_first_good_time=use_first_good_time,
                use_last_good_time=use_last_good_time,
            )
            if lo <= hi:
                time_axis = time_axis[lo : hi + 1]
                asymmetry = asymmetry[lo : hi + 1]
                error = error[lo : hi + 1]

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

            run_grouping = {
                "groups": {gid: [idx + 1 for idx in dets] for gid, dets in grouping.groups.items()},
                "forward_group": grouping.forward_group_id,
                "backward_group": grouping.backward_group_id,
                "alpha": alpha,
                "first_good_bin": int(group_first_good),
                "last_good_bin": int(group_last_good),
                "t0_bin": int(common_t0),
                "t_good_offset": max(0, int(group_first_good) - int(common_t0)),
                # ISIS header bins are 1-based; the payload keeps 0-based values
                # and records the file's convention for display only.
                "bin_index_base": 1,
                "bunching_factor": 1,
                "deadtime_correction": False,
                "detector_t0_bins": [int(hist.t0_bin) for hist in histograms],
                "detector_first_good_bins": [int(hist.good_bin_start) for hist in histograms],
                "detector_last_good_bins": [int(hist.good_bin_end) for hist in histograms],
                "t0_source": decode.source,
                "good_frames": float(good_frames_periods[period_idx - 1]),
                "dead_time_us": [
                    float(v)
                    for v in np.asarray(
                        dead_time_periods[period_idx - 1], dtype=np.float64
                    ).tolist()
                ],
            }
            if t0_time_us is not None:
                run_grouping["t0_time_us"] = float(t0_time_us)

            run = Run(
                run_number=period_run_number,
                histograms=histograms,
                metadata=run_meta,
                grouping=run_grouping,
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
        *,
        decode: _T0Decode,
        first_good_bin: int,
        last_good_bin: int,
    ) -> list[Histogram]:
        """Create per-detector :class:`Histogram` objects for a period.

        The good-data window is a single run-level pair in ISIS files, so every
        detector carries the same one; t0 comes from :meth:`_decode_t0`.
        """
        return [
            Histogram(
                counts=np.asarray(period_counts[i], dtype=np.float64),
                bin_width=float(bin_width),
                t0_bin=int(decode.t0_bins[i]),
                good_bin_start=int(first_good_bin),
                good_bin_end=int(last_good_bin),
                t0_time_us=decode.t0_time_us[i],
            )
            for i in range(period_counts.shape[0])
        ]

    def _group_good_window(
        self,
        histograms: list[Histogram],
        group_indices: list[int],
        *,
        common_t0: int,
        n_grouped: int,
    ) -> tuple[int, int]:
        """The good window of the *aligned* group sums, in grouped-bin indices.

        Thin adapter over :func:`good_window_for_groups`, which owns the rule for
        every loader and for profile resolution; ISIS keeps the per-detector
        tables on the histograms rather than in parallel lists.
        """
        return good_window_for_groups(
            group_indices,
            [int(hist.t0_bin) for hist in histograms],
            [int(hist.good_bin_start) for hist in histograms],
            [int(hist.good_bin_end) for hist in histograms],
            common_t0_bin=common_t0,
            n_bins=n_grouped,
        )

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

    def _attr_bin_values(self, attr: Any, *, n_detectors: int) -> np.ndarray | None:
        """Per-detector values from an integer ``counts`` attribute, as written.

        Accepts a scalar (broadcast across detectors) or a per-detector array;
        returns ``None`` when the attribute is missing or unusable. Shared by
        the v1 and v2 layouts, which both store ``t0_bin`` on the counts SDS.
        The values are the file's own 1-based indices — :meth:`_decode_t0`
        converts them.
        """
        if attr is None:
            return None
        values = np.asarray(attr, dtype=np.float64).ravel()
        if values.size == 1 and np.isfinite(values[0]):
            return np.full(n_detectors, int(round(float(values[0]))), dtype=np.int64)
        if values.size == n_detectors and np.all(np.isfinite(values)):
            return np.rint(values).astype(np.int64)
        return None

    def _good_bin_window(self, counts_attrs: Any, h_data: Any, n_bins: int) -> tuple[int, int]:
        """The 0-based, inclusive good-data window from 1-based ISIS headers.

        ``first_good_bin`` / ``last_good_bin`` are 1-based and inclusive in
        every ISIS file surveyed (``last_good_bin == n_bins`` in all 1,245 —
        docs/porting/t0-determination/isis-header-index-base.md), so both simply
        lose one. ``h_data`` supplies the legacy child-dataset spelling for v1
        files that carry no attributes; an absent field means "the whole
        histogram".
        """
        first_raw = self._safe_int(counts_attrs.get("first_good_bin"), default=None)
        last_raw = self._safe_int(counts_attrs.get("last_good_bin"), default=None)
        if h_data is not None:
            if first_raw is None:
                first_raw = self._safe_int(
                    self._read_optional(h_data, "first_good_bin"), default=None
                )
            if last_raw is None:
                last_raw = self._safe_int(
                    self._read_optional(h_data, "last_good_bin"), default=None
                )
        first_good = 0 if first_raw is None else max(0, int(first_raw) - 1)
        last_good = (n_bins - 1) if last_raw is None else max(0, int(last_raw) - 1)
        return first_good, last_good

    def _decode_t0(
        self,
        *,
        t0_bin_attr: Any,
        time_zero_values: np.ndarray,
        n_detectors: int,
        resolution_us: float,
        corrected_time: np.ndarray,
        n_bins: int,
        source_file: str,
    ) -> _T0Decode:
        """Decode per-detector t0 from the ISIS header (decisions D5, D6, D7).

        ``t0_bin`` is the 1-based index of the bin *containing* the file's own
        ``time_zero`` (µs), so the 0-based bin is ``t0_bin − 1`` — deterministic,
        never inferred. ``time_zero`` additionally places t0 *within* that bin;
        it is kept when the two agree (``floor(time_zero / resolution) + 1 ==
        t0_bin``) and dropped when they do not, in which case the attribute wins
        and ``t0_source`` is ``"conflict"`` (D6). With only ``time_zero`` the
        containing bin is derived from it; with neither field the run has no
        usable t0 (``"missing"``, D7 — resolution searches for one).
        """
        attr_bins = self._attr_bin_values(t0_bin_attr, n_detectors=n_detectors)
        time_zeros = self._time_zero_per_detector(time_zero_values, n_detectors)

        if attr_bins is None and time_zeros is None:
            return _T0Decode(np.zeros(n_detectors, dtype=np.int64), [None] * n_detectors, "missing")

        if attr_bins is None:
            # ``time_zero`` alone: the bin containing it, with the same edge
            # tolerance the agreement test uses (a quotient one ulp below an
            # integer is an exact edge).
            bins = np.maximum(
                0,
                np.asarray(
                    [int(np.floor(tz / resolution_us + _T0_BIN_TOLERANCE)) for tz in time_zeros],
                    dtype=np.int64,
                ),
            )
            exact: list[float | None] = list(time_zeros)
        elif time_zeros is None:
            bins = np.maximum(0, attr_bins - 1)
            exact = [None] * n_detectors
        else:
            bins = np.maximum(0, attr_bins - 1)
            disagree = [
                i
                for i, tz in enumerate(time_zeros)
                if not self._t0_fields_agree(tz, resolution_us, int(attr_bins[i]))
            ]
            if disagree:
                logger.warning(
                    "%s: NeXus t0_bin attribute %d disagrees with time_zero %.6f us at %.6f us "
                    "binning (detector %d); keeping the attribute and dropping the exact t0.",
                    source_file,
                    int(attr_bins[disagree[0]]),
                    float(time_zeros[disagree[0]]),
                    float(resolution_us),
                    disagree[0] + 1,
                )
                return _T0Decode(bins, [None] * n_detectors, "conflict")
            exact = list(time_zeros)

        return self._cross_check_corrected_time(
            _T0Decode(bins, exact, "file"),
            corrected_time=corrected_time,
            n_bins=n_bins,
            source_file=source_file,
        )

    def _time_zero_per_detector(
        self, time_zero_values: np.ndarray, n_detectors: int
    ) -> list[float] | None:
        """``time_zero`` in µs per detector, or ``None`` when the file has none.

        A scalar broadcasts across detectors; v2 files may store one value per
        detector. Non-finite content counts as absent.
        """
        values = np.asarray(time_zero_values, dtype=np.float64).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        if values.size == n_detectors:
            return [float(v) for v in values]
        return [float(values[0])] * n_detectors

    def _t0_fields_agree(self, time_zero_us: float, resolution_us: float, attr_bin: int) -> bool:
        """True when ``time_zero`` falls inside the bin ``t0_bin`` names.

        That is ``attr_bin − 1 <= time_zero / resolution < attr_bin``, with a
        tolerance of :data:`_T0_BIN_TOLERANCE` bins because both fields are
        written in float32 (an exact edge reads back as 9.99999978).
        """
        q = float(time_zero_us) / float(resolution_us)
        return (attr_bin - 1) - _T0_BIN_TOLERANCE <= q < attr_bin + _T0_BIN_TOLERANCE

    def _cross_check_corrected_time(
        self,
        decode: _T0Decode,
        *,
        corrected_time: np.ndarray,
        n_bins: int,
        source_file: str,
    ) -> _T0Decode:
        """Flag a ``corrected_time`` axis that disagrees with the decoded t0.

        The file's own axis is zero inside the t0 bin, so ``|ct|`` must not be
        *smaller* one bin later. When it is, the axis was built from a different
        t0 than the attribute declares (the 2003-era stale-header case) — the
        attribute still wins, but the run is marked ``"conflict"``.
        """
        axis, _ = self._build_time_axis(corrected_time, n_bins)
        if corrected_time.size == 0 or axis.size == 0:
            return decode
        k = int(np.max(decode.t0_bins))
        if k + 1 >= axis.size:
            return decode
        if abs(float(axis[k + 1])) < abs(float(axis[k])):
            logger.warning(
                "%s: NeXus corrected_time is closer to zero at bin %d than at the "
                "decoded t0 bin %d; keeping the t0_bin attribute.",
                source_file,
                k + 1,
                k,
            )
            return _T0Decode(decode.t0_bins, [None] * len(decode.t0_time_us), "conflict")
        return decode

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

        Delegates to :func:`asymmetry.core.io.periods.encode_period_run_number`
        so the loader's list path and the scriptable ``select_period`` reducer
        share one encoding scheme.
        """
        return encode_period_run_number(run_number, period_idx)

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

    def _fill_field_from_sidecar_log(
        self, metadata: dict[str, Any], *, source_file: str, field_present: bool
    ) -> None:
        """Fill missing ``field``/``field_direction``/``field_state`` from a ``.log``.

        Many ISIS NeXus files carry no ``sample/magnetic_field_state`` at all
        (and, more rarely, no readable ``magnetic_field`` either), which sends
        the fit wizard's Auto scope to the metadata-poor "screen everything"
        fallback (see :mod:`asymmetry.core.fitting.wizard_scope`). The ICP
        ``.log`` sidecar that ISIS writes beside every run
        (:mod:`asymmetry.core.io.icp_log`) often has the same information from
        the instrument control system's own record of the selected magnet.

        This only ever *fills gaps* — a ``field_direction``/``field_state``
        already read from the NeXus file is never overwritten, and neither is
        a genuinely-present (even if zero) ``magnetic_field`` value. Best
        effort throughout: a missing/malformed log leaves the metadata
        unchanged (mutates *metadata* in place).
        """
        if field_present and metadata.get("field_direction"):
            return  # nothing to fill

        log_path = sibling_icp_log_path(source_file)
        if not log_path.is_file():
            return

        reading = parse_icp_log_file(log_path)
        if reading is None:
            return

        if not field_present and reading.field_gauss is not None:
            metadata["field"] = reading.field_gauss
            metadata["field_source"] = "icp_log"

        if not metadata.get("field_direction") and reading.field_direction:
            metadata["field_direction"] = reading.field_direction
            metadata["field_direction_source"] = "icp_log"
            if not metadata.get("field_state"):
                metadata["field_state"] = "ZF"

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

    def _read_field_gauss(self, sample: Any, default: float = 0.0) -> float:
        """Read ``sample/magnetic_field`` as a field in gauss.

        Reads both the value and its NeXus ``units`` attribute so a field
        declared in tesla (``T`` / ``tesla``) or millitesla (``mT`` /
        ``millitesla``) is normalized to gauss via
        :func:`_normalize_field_to_gauss`. A gauss, missing, or unrecognized
        unit passes the value through unchanged, matching the historical
        behaviour and the gauss convention the rest of the app assumes.
        """
        if sample is None or not hasattr(sample, "get"):
            return float(default)
        # Cannot use _read_optional here: it returns the unwrapped value and
        # drops the node, but we need node.attrs['units'] to decide the scale.
        node = sample.get("magnetic_field")
        if node is None:
            return float(default)
        raw = node[()] if hasattr(node, "dtype") else node
        value = self._safe_float(raw, default=default)
        units = ""
        if hasattr(node, "attrs"):
            units = self._safe_str(node.attrs.get("units", ""))
        normalized = _normalize_field_to_gauss(value, units)
        return float(default) if normalized is None else float(normalized)

    def _read_field_vector(self, sample: Any) -> list[float] | None:
        """Read ``sample/magnetic_field_vector`` when the file marks it usable.

        ISIS NeXus files carry a ``magnetic_field_vector`` dataset with an
        ``available`` attribute (0/1). In this loader's corpus survey every
        unavailable vector held the same uninformative placeholder
        (``[1, 1, 1]``, ``available=0``); a few real TF calibration runs do set
        ``available=1``, but even then the vector only encodes a unit *axis*
        (e.g. ``[0, 0, 1]``) rather than a genuinely per-run direction reading,
        and combining it with magnitude would not add information beyond
        ``sample/magnetic_field``/``magnetic_field_state`` already used above.

        This is exposed as raw provenance (``metadata["field_vector"]``) for
        advanced/debug display only. It is deliberately **not** used to infer
        TF/LF geometry anywhere in this loader or in
        :mod:`asymmetry.core.fitting.wizard_scope` — deriving a field
        direction from a hardware axis vector is the same forbidden
        orientation-based guess that ``docs/porting/field-geometry/`` rejected
        for detector orientation. Returns ``None`` when the node is absent,
        unreadable, or marked unavailable.
        """
        if sample is None or not hasattr(sample, "get"):
            return None
        node = sample.get("magnetic_field_vector")
        if node is None:
            return None
        attrs = getattr(node, "attrs", {}) or {}
        available = self._safe_int(attrs.get("available"), default=1)
        if available == 0:
            return None
        try:
            data = np.asarray(node[()], dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        if data.size != 3 or not np.all(np.isfinite(data)):
            return None
        return [float(v) for v in data]

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
