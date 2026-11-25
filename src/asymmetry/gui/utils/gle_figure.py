"""Independent gleplot figures without its module-level current-figure state."""

from copy import deepcopy


def gle_figure(glp, **kwargs):
    """Capture independent configurations for each worker's figure."""
    for key in ("style", "graph", "marker"):
        if kwargs.get(key) is None:
            kwargs[key] = deepcopy(getattr(glp.GlobalConfig, f"get_{key}")())
    # Joining samples with straight segments introduces no spline extrema.
    kwargs["graph"].smooth_curves = False
    return glp.Figure(**kwargs)


def save_gle_figure(figure, path, *, output_format="pdf", preserve_alpha=False):
    """Use a white native backdrop when independent fills cannot overlap."""
    figure.style.opaque_backdrop = None if output_format == "svg" or preserve_alpha else "white"
    cairo = output_format == "svg" or figure.requires_cairo()
    if cairo:
        # Cairo rejects GLE's PostScript default; this face preserves ASCII glyphs.
        figure.style.font = "texcmtt"
    figure.savefig(str(path))
    return cairo


def gle_subplots(glp, *, nrows, ncols, figsize, sharex=False):
    """Build a rectangular set of axes on one independent figure."""
    figure = gle_figure(glp, figsize=figsize, sharex=sharex)
    axes = []
    for index in range(nrows * ncols):
        axes.append(figure.add_subplot(nrows, ncols, index + 1))
    return figure, axes
