"""Tests for the core data model."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run


class TestHistogram:
    def test_time_axis(self):
        h = Histogram(counts=np.ones(100), bin_width=0.01, t0_bin=10)
        t = h.time_axis
        assert len(t) == 100
        assert t[10] == pytest.approx(0.0)
        assert t[11] == pytest.approx(0.01)

    def test_n_bins(self):
        h = Histogram(counts=np.zeros(256), bin_width=0.016)
        assert h.n_bins == 256


class TestRun:
    def test_summary(self):
        r = Run(run_number=12345, metadata={"title": "Test", "temperature": 10.0, "field": 100.0})
        s = r.summary()
        assert "12345" in s
        assert "Test" in s

    def test_summary_with_histograms(self):
        """Test summary includes histogram info when histograms are present."""
        h = Histogram(counts=np.ones(100), bin_width=0.016, t0_bin=10)
        r = Run(
            run_number=12345,
            histograms=[h],
            metadata={
                "title": "Test",
                "temperature": 10.0,
                "field": 100.0,
                "comment": "Sample comment",
                "started": "2024-01-01",
                "stopped": "2024-01-02",
            },
        )
        s = r.summary()
        assert "12345" in s
        assert "100" in s  # histogram count
        assert "0.016" in s  # bin width
        assert "comment" in s.lower()
        assert "started" in s.lower()
        assert "stopped" in s.lower()

    def test_repr(self):
        """Test __repr__ method."""
        r = Run(run_number=999, metadata={"title": "T", "temperature": 5.0, "field": 20.0})
        repr_str = repr(r)
        assert "999" in repr_str
        assert "5.0" in repr_str
        assert "20.0" in repr_str

    def test_properties(self):
        r = Run(metadata={"title": "T", "temperature": 5.0, "field": 20.0})
        assert r.title == "T"
        assert r.temperature == 5.0
        assert r.field == 20.0


class TestMuonDataset:
    def _make_dataset(self, n=50):
        t = np.linspace(0, 10, n)
        a = 0.2 * np.exp(-t)
        e = np.full(n, 0.01)
        return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 1})

    def test_n_points(self):
        ds = self._make_dataset()
        assert ds.n_points == 50

    def test_run_number_from_metadata(self):
        """Test run_number property when no run object is attached."""
        ds = self._make_dataset()
        assert ds.run_number == 1

    def test_run_number_from_run_object(self):
        """Test run_number property when a run object is attached."""
        ds = self._make_dataset()
        ds.run = Run(run_number=999, metadata={})
        assert ds.run_number == 999

    def test_time_range(self):
        ds = self._make_dataset(100)
        sub = ds.time_range(t_min=2.0, t_max=5.0)
        assert sub.time[0] >= 2.0
        assert sub.time[-1] <= 5.0
        assert sub.n_points < ds.n_points

    def test_summary(self):
        """Test summary method output."""
        ds = self._make_dataset(100)
        s = ds.summary()
        assert "100 points" in s
        assert "Time range" in s
        assert "Asymmetry" in s

    def test_summary_with_run(self):
        """Test summary includes run info when available."""
        ds = self._make_dataset(50)
        ds.run = Run(
            run_number=12345, metadata={"title": "Test", "temperature": 10.0, "field": 100.0}
        )
        s = ds.summary()
        assert "50 points" in s
        assert "12345" in s

    def test_repr(self):
        """Test __repr__ method."""
        ds = self._make_dataset(75)
        repr_str = repr(ds)
        assert "75" in repr_str
        assert "1" in repr_str  # run number
