"""Recognise instruments from names, filenames, and detector labels."""

import pytest

from asymmetry.core.instrument import detect_instrument
from tests.core.test_instrument import GPS_ROOT15_LABELS


class TestDetectInstrument:
    @pytest.mark.parametrize("n_histograms", [64, 96])
    @pytest.mark.parametrize("metadata", [None, {"instrument": "Unknown"}])
    def test_detector_count_does_not_identify_instrument(self, n_histograms, metadata):
        assert (
            detect_instrument(n_histograms, metadata=metadata, source_file="run000123.nxs") is None
        )

    def test_metadata_identifies_instrument(self):
        assert detect_instrument(64, metadata={"instrument": "MUSR"}) == "MuSR"

    def test_source_file_prefix_can_hint_instrument(self):
        assert detect_instrument(64, source_file="/tmp/emu000123.nxs") == "EMU"

    def test_flame_can_be_detected_from_psi_metadata_or_filename(self):
        assert detect_instrument(8, metadata={"facility": "PSI", "instrument": "FLAME"}) == "FLAME"
        assert (
            detect_instrument(
                8,
                metadata={"facility": "PSI", "instrument": ""},
                source_file="/tmp/run_flame_001.bin",
            )
            == "FLAME"
        )

    def test_flame_can_be_detected_from_psi_detector_labels(self):
        labels = ["Forw", "Back", "Righ", "Left", "R_F", "R_B", "L_F", "L_B"]
        assert (
            detect_instrument(
                8,
                metadata={"facility": "PSI", "instrument": "PSI", "histogram_labels": labels},
            )
            == "FLAME"
        )

    def test_psi_metadata_does_not_select_isis_layout(self):
        # PSI "HIFI" is the HAL-9500 high-field instrument, not the ISIS HiFi.
        assert detect_instrument(64, metadata={"facility": "PSI", "instrument": "HIFI"}) == "HAL"
        # A PSI file claiming an ISIS-only instrument name still must not map to
        # that ISIS layout.
        assert (
            detect_instrument(64, metadata={"psi_format": "psi-bin", "instrument": "MuSR"}) is None
        )

    def test_psi_hal_detected_from_metadata_and_filename(self):
        md = {"facility": "PSI", "psi_format": "psi-mdu", "instrument": "HIFI"}
        assert detect_instrument(9, metadata=md) == "HAL"
        assert detect_instrument(17, metadata=md) == "HAL"
        assert (
            detect_instrument(
                17,
                metadata={"facility": "PSI", "psi_format": "psi-mdu", "instrument": ""},
                source_file="/data/tdc_hifi_2025_01425.mdu",
            )
            == "HAL"
        )

    def test_psi_gps_detected_from_metadata_and_filename(self):
        md = {"facility": "PSI", "psi_format": "psi-bin", "instrument": "GPS"}
        assert detect_instrument(6, metadata=md) == "GPS"
        assert (
            detect_instrument(
                6,
                metadata={"facility": "PSI", "psi_format": "psi-bin", "instrument": ""},
                source_file="/data/deltat_tdc_gps_5848.bin",
            )
            == "GPS"
        )

    def test_psi_gps_detected_from_detector_labels(self):
        # The fixed six-counter F/B/U/D/R/L label set identifies GPS even when
        # the instrument/filename token is absent.
        assert (
            detect_instrument(
                6,
                metadata={
                    "facility": "PSI",
                    "psi_format": "psi-bin",
                    "histogram_labels": ["Forw", "Back", "Up", "Down", "Righ", "Left"],
                },
            )
            == "GPS"
        )

    def test_psi_classic_gps_bin_five_counters_detected_as_gps(self):
        # Classic GPS .bin files omit the Left counter: only five histograms
        # (Forw, Back, Up, Down, Righ). They still resolve to GPS, never FLAME,
        # even when the instrument token is the generic "PSI".
        labels = ["Forw", "Back", "Up", "Down", "Righ"]
        assert (
            detect_instrument(
                5,
                metadata={
                    "facility": "PSI",
                    "psi_format": "psi-bin",
                    "instrument": "PSI",
                    "histogram_labels": labels,
                },
            )
            == "GPS"
        )

    def test_psi_classic_gps_bin_detected_without_any_instrument_token(self):
        # A GPS .bin whose filename carries no instrument token (e.g. run1234.bin)
        # and whose stored instrument fell through to the generic "PSI" is still
        # positively identified as GPS from its five-counter label set — it must
        # not fall through to None (which the layout editor would turn into a
        # wrong HiFi/ISIS default) or to FLAME.
        assert (
            detect_instrument(
                5,
                metadata={
                    "facility": "PSI",
                    "psi_format": "psi-bin",
                    "instrument": "PSI",
                    "histogram_labels": ["Forw", "Back", "Up", "Down", "Righ"],
                },
                source_file="/data/run1234.bin",
            )
            == "GPS"
        )

    def test_psi_gps_digit_adjacent_filename_detected_as_gps(self):
        # gps2923.bin: the instrument token abuts the run number with no
        # separator. It must resolve to GPS (never FLAME).
        assert (
            detect_instrument(
                5,
                metadata={"facility": "PSI", "psi_format": "psi-bin", "instrument": "GPS"},
                source_file="/data/gps2923.bin",
                # instrument already "GPS" here because the loader's
                # _guess_psi_instrument now matches the digit-adjacent token;
                # detection must agree.
            )
            == "GPS"
        )

    def test_psi_gpd_is_not_detected_as_gps(self):
        # GPD (decay-channel instrument) carries a "gpd" token, not "gps".
        assert (
            detect_instrument(
                6,
                metadata={"facility": "PSI", "psi_format": "psi-bin", "instrument": "GPD"},
                source_file="/data/deltat_tdc_gpd_0001.bin",
            )
            is None
        )

    def test_psi_gpd_label_set_is_not_detected_as_gps(self):
        # GPD's four-counter Back/Forw/Up/Down set has no transverse (right/left)
        # counter, so it must never be mistaken for the GPS classic label set.
        for labels in (["B", "F", "U", "D"], ["Back", "Forw", "Up", "Down"]):
            assert (
                detect_instrument(
                    4,
                    metadata={
                        "facility": "PSI",
                        "psi_format": "psi-bin",
                        "instrument": "PSI",
                        "histogram_labels": labels,
                    },
                )
                is None
            )

    def test_psi_dolly_label_set_is_not_detected_as_gps(self):
        # DOLLY's four-counter Forw/Back/Left/Right set has no up/down axis, so
        # it must never be mistaken for the GPS classic label set (nor FLAME).
        assert (
            detect_instrument(
                4,
                metadata={
                    "facility": "PSI",
                    "psi_format": "psi-bin",
                    "instrument": "PSI",
                    "histogram_labels": ["Forw", "Back", "Left", "Rite"],
                },
            )
            is None
        )

    def test_flame_requires_split_plate_corners_not_just_main_axes(self):
        # An eight-histogram PSI file with the FLAME main axes but WITHOUT the
        # four split-plate corners (R_F/R_B/L_F/L_B) is not FLAME — the corners
        # are the FLAME-specific positive evidence.
        assert (
            detect_instrument(
                8,
                metadata={
                    "facility": "PSI",
                    "instrument": "PSI",
                    "histogram_labels": [
                        "Forw",
                        "Back",
                        "Left",
                        "Righ",
                        "Up",
                        "Down",
                        "Mon1",
                        "Mon2",
                    ],
                },
            )
            != "FLAME"
        )

    def test_psi_gps_root_subdetectors_detected_as_gps_rd(self):
        # ROOT export has 11 sub-detector histograms -> the GPS-RD variant.
        md = {"facility": "PSI", "instrument": "LMU_BULKMUSR_GPS"}
        assert detect_instrument(11, metadata=md) == "GPS-RD"

    def test_psi_gps_root_detected_from_subdetector_labels(self):
        # The 11-counter sub-detector label set identifies GPS-RD without a token.
        assert (
            detect_instrument(
                11,
                metadata={
                    "facility": "PSI",
                    "histogram_labels": [
                        "Forw",
                        "Back",
                        "Up_B",
                        "Up_F",
                        "Down_B",
                        "Down_F",
                        "Right_B",
                        "Right_F",
                        "Left_B",
                        "Left_F",
                        "Mob-RL",
                    ],
                },
            )
            == "GPS-RD"
        )

    def test_gps_bin_and_root_select_different_variants(self):
        # Same instrument token, different histogram count -> different variant.
        md = {"facility": "PSI", "instrument": "GPS"}
        assert detect_instrument(6, metadata=md) == "GPS"
        assert detect_instrument(11, metadata=md) == "GPS-RD"

    def test_psi_gps_root15_combined_plus_subdetectors_detected_as_gps_rd15(self):
        # The 15-histogram ROOT export ships the six combined counters *and* the
        # half-counters, so its detector ids differ from the 11-histogram export's
        # -- it needs its own variant, not GPS-RD.
        md = {
            "facility": "PSI",
            "instrument": "LMU_BULKMUSR_GPS",
            "histogram_labels": list(GPS_ROOT15_LABELS),
        }
        assert detect_instrument(15, metadata=md) == "GPS-RD15"

    def test_psi_gps_root15_detected_from_labels_alone(self):
        # No instrument token: the combined-plus-half label set identifies the
        # 15-histogram export on its own.
        assert (
            detect_instrument(
                15,
                metadata={"facility": "PSI", "histogram_labels": list(GPS_ROOT15_LABELS)},
            )
            == "GPS-RD15"
        )

    def test_psi_gps_root15_labels_not_mistaken_for_subdetector_export(self):
        # Regression for the silent-misgrouping bug: the 15-histogram label set
        # contains every 11-histogram sub-detector label, so a matcher keyed on
        # the half-counter labels alone resolved these runs to GPS-RD, whose ids
        # 3-6 mean Up_B/Up_F/Down_B/Down_F rather than Up/Down/Right/Left.
        md = {
            "facility": "PSI",
            "instrument": "LMU_BULKMUSR_GPS",
            "histogram_labels": list(GPS_ROOT15_LABELS),
        }
        # Even with the histogram count reported unexpectedly, the label set alone
        # must resolve to the 15-histogram variant.
        assert detect_instrument(99, metadata=md) == "GPS-RD15"

    def test_gps_bin_root_and_root15_select_different_variants(self):
        # Same instrument token, different histogram count -> different variant.
        md = {"facility": "PSI", "instrument": "GPS"}
        assert detect_instrument(6, metadata=md) == "GPS"
        assert detect_instrument(11, metadata=md) == "GPS-RD"
        assert detect_instrument(15, metadata=md) == "GPS-RD15"

    def test_gps_root_variant_falls_back_to_subdetector_labels(self):
        # The sub-detector label set selects GPS-RD even when the histogram count
        # is reported unexpectedly (the label match is not gated on count).
        md = {
            "facility": "PSI",
            "instrument": "GPS",
            "histogram_labels": [
                "Forw",
                "Back",
                "Up_B",
                "Up_F",
                "Down_B",
                "Down_F",
                "Right_B",
                "Right_F",
                "Left_B",
                "Left_F",
                "Mob-RL",
            ],
        }
        assert detect_instrument(99, metadata=md) == "GPS-RD"

    def test_isis_hifi_still_resolves_without_psi_flag(self):
        # Without a PSI marker, the bare "HiFi" name remains the ISIS layout.
        assert detect_instrument(64, metadata={"instrument": "HiFi"}) == "HiFi"

    def test_metadata_identifies_emu(self):
        assert detect_instrument(96, metadata={"instrument": "EMU"}) == "EMU"

    def test_unknown_returns_none(self):
        assert detect_instrument(0) is None
        assert detect_instrument(32) is None
        assert detect_instrument(128) is None

    def test_95_returns_none(self):
        assert detect_instrument(95) is None

    def test_65_returns_none(self):
        assert detect_instrument(65) is None


