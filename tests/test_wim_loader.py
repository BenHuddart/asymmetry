"""Tests for the WiMDA .wim file loader."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io.wim import WimLoader


@pytest.fixture()
def sample_wim(tmp_path: Path) -> Path:
    """Write a minimal .wim file for testing."""
    content = textwrap.dedent("""\
        ! START OF RUN INFORMATION
        ! Run number : 12345
        ! Title : ZnCu3(OH)6Cl2 ZF
        ! Temperature : 1.500 K (label 1.5 K)
        ! Field : 0.00 G
        ! Comment : Test sample
        ! Started : 2025-01-15 10:00:00
        ! Stopped : 2025-01-15 11:00:00
        ! Histograms : 8 (2000 bins of 16000.000 ps = 32.00 µs)
        ! Events : 5.00 MEv grouped in range (raw = 10.00)
        ! END OF RUN INFORMATION
        ! START OF GROUPING INFORMATION
        ! Group#01  Hist(t0): 01(100) 02(100)
        ! Group#02  Hist(t0): 03(100) 04(100)
        ! Forward Group = 1, Backward Group = 2, Alpha = 1.000
        ! Offset to first good bin = 5, Last good bin = 1990
        ! Fixed binning, bunching factor = 1
        ! Deadtime correction on
        ! END OF GROUPING INFORMATION
        ! START OF DATA SET INFORMATION
        ! Datarow : x y e
        ! Title : Asymmetry
        ! XLabel : Time (us)
        ! YLabel : Asymmetry (%)
        ! END OF DATA SET INFORMATION
        0.000000 25.00 0.50
        0.016000 24.50 0.50
        0.032000 23.80 0.51
        0.048000 22.90 0.52
        0.064000 21.80 0.53
    """)
    path = tmp_path / "test.wim"
    path.write_text(content, encoding="iso-8859-1")
    return path


class TestWimLoader:
    def test_load_metadata(self, sample_wim):
        loader = WimLoader()
        ds = loader.load(str(sample_wim))

        assert ds.run_number == 12345
        assert ds.metadata["title"] == "ZnCu3(OH)6Cl2 ZF"
        assert ds.metadata["temperature"] == pytest.approx(1.5)
        assert ds.metadata["field"] == pytest.approx(0.0)
        assert ds.metadata["comment"] == "Test sample"

    def test_load_data(self, sample_wim):
        loader = WimLoader()
        ds = loader.load(str(sample_wim))

        assert ds.n_points == 5
        assert ds.time[0] == pytest.approx(0.0)
        assert ds.asymmetry[0] == pytest.approx(25.0)
        assert ds.error[0] == pytest.approx(0.5)

    def test_load_grouping(self, sample_wim):
        loader = WimLoader()
        ds = loader.load(str(sample_wim))

        assert ds.run.grouping["forward_group"] == 1
        assert ds.run.grouping["backward_group"] == 2
        assert ds.run.grouping["alpha"] == pytest.approx(1.0)
        assert ds.run.grouping["first_good_bin"] == 5

    def test_file_not_found(self):
        loader = WimLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.wim")

    def test_temperature_parsing_with_label(self, sample_wim):
        """Test that temperature label is parsed correctly."""
        loader = WimLoader()
        ds = loader.load(str(sample_wim))
        assert "temperature_label" in ds.metadata

    def test_metadata_fields(self, sample_wim):
        """Test all expected metadata fields are present."""
        loader = WimLoader()
        ds = loader.load(str(sample_wim))
        assert "started" in ds.metadata
        assert "stopped" in ds.metadata
        assert "source_file" in ds.metadata

    def test_run_object_attached(self, sample_wim):
        """Test that run object is properly attached to dataset."""
        loader = WimLoader()
        ds = loader.load(str(sample_wim))
        assert ds.run is not None
        assert ds.run.run_number == 12345
        assert ds.run.grouping is not None
