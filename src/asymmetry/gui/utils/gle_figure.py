"""Independent gleplot figures without its module-level current-figure state."""

from copy import deepcopy


def gle_figure(glp, **kwargs):
    """Capture independent configurations for each worker's figure."""
    for key in ("style", "graph", "marker"):
        if kwargs.get(key) is None:
            kwargs[key] = deepcopy(getattr(glp.GlobalConfig, f"get_{key}")())
    return glp.Figure(**kwargs)


def gle_subplots(glp, *, nrows, ncols, figsize, sharex=False):
    """Build a rectangular set of axes on one independent figure."""
    figure = gle_figure(glp, figsize=figsize, sharex=sharex)
    axes = []
    for index in range(nrows * ncols):
        axes.append(figure.add_subplot(nrows, ncols, index + 1))
    return figure, axes
