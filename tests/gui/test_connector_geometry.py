"""Independent source-index checks for dense and sparse connector geometry."""

import numpy as np
import pytest

from asymmetry.gui.utils.connector_geometry import ConnectorGeometry


def test_m4_keeps_each_bucket_first_last_and_extrema():
    rng = np.random.default_rng(690)
    x = np.linspace(0, 10, 10001)
    y = rng.normal(size=x.size)
    geometry = ConnectorGeometry(x, y, np.ones_like(x), np.ones_like(x, dtype=bool))
    selected = geometry.select((0, 10), 40)
    assert np.all(np.diff(selected) > 0)
    assert selected.size <= 4 * 41
    buckets = np.floor(x * 4).astype(int)
    for bucket in np.unique(buckets):
        members = np.flatnonzero(buckets == bucket)
        assert set(
            [
                members[0],
                members[-1],
                members[np.argmin(y[members])],
                members[np.argmax(y[members])],
            ]
        ) <= set(selected)


def test_sparse_zoom_recovers_source_and_clipping_neighbours():
    x = np.arange(10000.0)
    geometry = ConnectorGeometry(x, np.sin(x), np.ones_like(x), np.ones_like(x, dtype=bool))
    np.testing.assert_array_equal(geometry.select((20, 30), 100), np.arange(19, 32))
    np.testing.assert_array_equal(geometry.select((20.1, 20.9), 100), [20, 21])


def test_gaps_and_nonfinite_errors_remain_separate_even_after_decimation():
    x = np.arange(1000.0)
    y = np.sin(x)
    err = np.ones_like(x)
    err[500] = np.nan
    mask = np.ones_like(x, dtype=bool)
    mask[600:650] = False
    x[700:] += 100
    geometry = ConnectorGeometry(x, y, err, mask)
    vertices = geometry.vertices((0, 1200), 10)
    assert np.isnan(vertices[:, 0]).sum() == 3
    for segment in np.split(vertices, np.flatnonzero(np.isnan(vertices[:, 0]))):
        segment = segment[np.isfinite(segment[:, 0])]
        if segment.size:
            assert not (segment[0, 0] < 500 < segment[-1, 0])
            assert not (segment[0, 0] < 600 < segment[-1, 0])
            assert not (segment[0, 0] < 750 < segment[-1, 0])


def test_alias_counterexample_retains_the_full_band_extent():
    x = np.arange(100000) * 0.001
    y = np.cos(2 * np.pi * 40.2 * x)
    np.testing.assert_allclose(y[::25], np.cos(2 * np.pi * 0.2 * x[::25]), atol=1e-9)
    geometry = ConnectorGeometry(x, y, np.ones_like(x), np.ones_like(x, dtype=bool))
    selected = geometry.select((0, 100), 200)
    for left in np.arange(199) / 2:
        values = y[selected[(x[selected] >= left) & (x[selected] < left + 0.5)]]
        assert values.min() < -0.99 and values.max() > 0.99


def test_time_resets_do_not_connect_independent_acquisitions():
    x = np.tile(np.arange(10.0), 2)
    geometry = ConnectorGeometry(x, x, np.ones_like(x), np.ones_like(x, dtype=bool))
    assert np.isnan(geometry.vertices((2, 4), 5)[:, 0]).sum() == 1


@pytest.mark.parametrize("case", ["resolved", "dense", "noisy", "chirp"])
def test_selected_solid_geometry_stays_within_one_pixel_of_full_observations(case):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.ndimage import distance_transform_edt

    x = np.linspace(0, 32, 100001)
    phase = 2 * np.pi * (1.5 * x if case == "resolved" else 18 * x)
    if case == "chirp":
        phase = 2 * np.pi * (1.5 * x + 0.5 * x**2)
    y = np.cos(phase)
    if case == "noisy":
        y += np.random.default_rng(690).normal(0, 0.15, x.size)
    geometry = ConnectorGeometry(x, y, np.ones_like(x), np.ones_like(x, dtype=bool))

    def ink(vertices):
        figure = Figure(figsize=(7.2, 3.6), dpi=100)
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 32), ylim=(-1.8, 1.8))
        ax.set_axis_off()
        line = ax.plot(
            vertices[:, 0], vertices[:, 1], color="black", linewidth=0.65, antialiased=False
        )[0]
        line.get_path().should_simplify = False
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, 0] < 128

    reference = ink(np.column_stack([x, y]))
    selected = ink(geometry.vertices((0, 32), 1440))
    assert distance_transform_edt(~reference)[selected].max() <= 1
    assert distance_transform_edt(~selected)[reference].max() <= 1
