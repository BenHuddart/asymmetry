"""Generate instrument review prose and diagrams from resolved profile data."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from asymmetry.core import instruments
from asymmetry.core.instruments.isis_emu import PROFILE as EMU
from asymmetry.core.instruments.isis_hifi import PROFILE as HIFI
from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.core.instruments.schema import InstrumentProfile

PROFILES = {profile.identity.key: profile for profile in (EMU, HIFI, MUSR)}


def _table(headers, rows):
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + "\n     - ".join(headers)]
    lines.extend(
        "   * - " + "\n     - ".join(str(value).replace("\n", " ") for value in row) for row in rows
    )
    return "\n".join(lines) + "\n\n"


def _channel_ranges(keys, ordinals):
    numbers = sorted(ordinals[key] for key in keys)
    runs = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}" for run in runs)


def _review_figure(profile, config, uri):
    text = f".. figure:: {uri}\n   :width: 100%\n   :alt: {profile.identity.name} labelled {config.label} schematic\n\n   {config.label}; numbers identify the recorded histograms.\n\n"
    text += f":download:`Full-size labelled diagram (SVG) <{uri}>`\n\n"
    conventions = dict.fromkeys(view.convention for view in config.views)
    text += "\n\n".join(conventions) + "\n\n"
    return text


def _detail_table(title, headers, rows):
    return f".. dropdown:: {title}\n\n" + textwrap.indent(_table(headers, rows), "   ") + "\n"


def profile_rst(profile: InstrumentProfile, diagrams: dict[str, str]) -> str:
    """Return deterministic reviewer-facing prose without publishing private names."""
    identity, review = profile.identity, profile.review
    title = f"{identity.name} ({identity.facility})"
    status = "Review recorded" if review.reviewed else "No instrument-scientist review recorded"
    text = f"{title}\n{'=' * len(title)}\n\n"
    text += f"Profile ``{identity.key}``, revision ``{identity.revision}``. **{status}.**\n\n{review.scope}\n\n"
    text += f"Beamline: **{identity.beamline or 'Unknown (unverified)'}**.\n\n"
    if review.reviewed:
        text += f"Reviewer: {review.public_name or 'name withheld'}.\n\n"
    leading = next(iter(profile.detectors.acquisitions), None)
    if leading is not None and leading.key in diagrams:
        text += _review_figure(profile, leading, diagrams[leading.key])
    assumptions = [
        evidence
        for evidence in profile.sources.evidence
        if evidence.fact.startswith(("acquisition.", "preset."))
        and evidence.status.value in {"Unverified", "Asymmetry-defined"}
    ]
    if assumptions:
        text += (
            "Key assumptions\n---------------\n\n"
            + "\n".join(f"* **{item.status.value}:** {item.detail}" for item in assumptions)
            + "\n\n"
        )
    text += f"Source\n------\n\nDeclared family: **{profile.source.kind}**. {profile.source.description}\n\n"
    text += "Source declarations do not replace recorded timing, grouping, correction or calibration settings.\n\n"
    text += "Physical structure\n------------------\n\n"
    text += (
        _table(
            ("Bank key", "Name", "Structure"),
            ((bank.key, bank.label, bank.description) for bank in profile.detectors.banks),
        )
        if profile.detectors.banks
        else "Physical bank structure is unknown.\n\n"
    )
    if profile.detectors.elements:
        text += _detail_table(
            "Detailed physical elements and affiliations",
            ("Physical key", "Label", "Bank", "Layer"),
            (
                (element.key, element.label, element.bank or "Unknown", element.layer or "Unknown")
                for element in profile.detectors.elements
            ),
        )
    dimensions = [
        (bank.key, dimension.name, dimension.value, dimension.units)
        for bank in profile.detectors.banks
        for dimension in bank.dimensions
    ]
    text += (
        _table(("Bank", "Physical annotation", "Value", "Units"), dimensions)
        if dimensions
        else "No numerical physical dimensions are declared. Schematic distances are not detector dimensions or acceptance weights.\n\n"
    )
    if not profile.detectors.acquisitions:
        text += "Acquisition and geometry\n------------------------\n\nNo acquisition map or geometry is declared. A preset-free generic schematic requires the actual acquired histogram count.\n\n"
    for config in profile.detectors.acquisitions:
        heading = f"Acquisition: {config.label}"
        text += f"{heading}\n{'-' * len(heading)}\n\nConfiguration key: ``{config.key}``. Histogram numbers identify the recorded channels; geometry does not redefine them. Shared hardware affiliations do not establish an exact count equation or statistical independence.\n\n"
        if config.key in diagrams and config is not leading:
            text += _review_figure(profile, config, diagrams[config.key])
        text += _table(
            ("View", "Coordinate status", "Units", "Viewing convention"),
            ((view.label, view.scale, view.units, view.convention) for view in config.views),
        )
        entry = next(entry for entry in profile.presets if entry.acquisition == config.key)
        ordinals = {channel.key: channel.ordinal for channel in config.channels}
        if not entry.presets:
            text += "No scientific grouping preset is declared for this configuration.\n\n"
        for index, preset in enumerate(entry.presets):
            text += f"**{preset.name}{' (default)' if index == 0 else ''}**\n\n"
            text += _table(
                ("Group", "Name", "Histograms"),
                (
                    (group.id, group.name, _channel_ranges(group.channels, ordinals))
                    for group in preset.groups
                ),
            )
            text += f"Positive/forward leg: group {preset.forward_group}; negative/backward leg: group {preset.backward_group}. The scalar balance remains experiment-specific.\n\n"
            if preset.projections:
                text += _table(
                    (
                        "Projection",
                        "Positive/forward leg",
                        "Negative/backward leg",
                        "Default alpha",
                    ),
                    (
                        (
                            projection.label,
                            projection.forward_group,
                            projection.backward_group,
                            projection.alpha,
                        )
                        for projection in preset.projections
                    ),
                )
                text += "Projection labels and initial balances do not certify calibrated physical vector components. Overlap across separate projections is intentional.\n\n"
        text += _detail_table(
            "Detailed histogram map",
            (
                "Channel key",
                "Histogram number",
                "Physical affiliation",
                "Expected file labels",
                "Relationship limits",
            ),
            (
                (
                    channel.key,
                    channel.ordinal,
                    ", ".join(channel.physical) or "Unknown",
                    ", ".join(channel.file_labels) or "Unverified",
                    channel.relationship or "No additional relationship asserted",
                )
                for channel in config.channels
            ),
        )
    text += "File conventions and defaults\n-----------------------------\n\n"
    text += (
        "\n".join(f"* {quirk}" for quirk in profile.quirks) + "\n\n"
        if profile.quirks
        else "No instrument-specific file quirk is declared.\n\n"
    )
    text += f"Automatic default-preset application on fresh load: **{'yes' if profile.defaults.apply_default_preset_on_load else 'no'}**. This definition supplies display and explicit preset data; it does not apply revised defaults to recorded analyses.\n\n"
    text += "Evidence and unresolved assumptions\n-----------------------------------\n\n"
    text += "Status applies to the named fact or coherent table, separately from expert review. Unknown affiliations stay unknown even when other rows have source support.\n\n"
    text += _table(
        ("Fact or table", "Evidence status", "Source and locator", "Scope and limits"),
        (
            (
                evidence.fact,
                evidence.status.value,
                f"{evidence.reference or 'No independent source'} {evidence.locator}".strip(),
                evidence.detail,
            )
            for evidence in profile.sources.evidence
        ),
    )
    text += "References\n----------\n\n"
    text += (
        "\n".join(
            f"* ``{reference.key}``: `{reference.title} <{reference.url}>`_."
            for reference in profile.sources.references
        )
        or "No independent primary reference is recorded."
    )
    return text + "\n"


def render_profile_diagram(
    profile: InstrumentProfile,
    acquisition: str | None,
    output: Path,
    *,
    n_histograms: int | None = None,
) -> None:
    """Render the same resolved primitive coordinates as the detector editor."""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle, Wedge

    layout = layout_from_profile(profile, acquisition, n_histograms=n_histograms)
    figure = Figure(figsize=(max(6, 5 * len(layout.banks)), 5.3))
    views = (
        next(config.views for config in profile.detectors.acquisitions if config.key == acquisition)
        if acquisition is not None
        else ()
    )
    preset = next(iter(layout.presets.values()), None)
    memberships = (
        {
            ordinal: group_id
            for group_id, group in preset.groups.items()
            for ordinal in group.detector_ids
        }
        if preset
        else {}
    )
    for index, bank in enumerate(layout.banks):
        axes = figure.add_subplot(1, len(layout.banks), index + 1)
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(bank.name)
        units = (
            f"{views[index].scale}; {views[index].units}" if views else "dimensionless schematic"
        )
        axes.text(
            0.5,
            -0.06,
            units,
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        for segment in bank.segments:
            color = (
                "0.93"
                if segment.read_only or segment.detector_id not in memberships
                else f"C{(memberships[segment.detector_id] - 1) % 10}"
            )
            if segment.shape == "wedge":
                patch = Wedge(
                    (0, 0),
                    segment.r_outer,
                    segment.angle_start_deg,
                    segment.angle_end_deg,
                    width=segment.r_outer - segment.r_inner,
                )
                import math

                radius = (segment.r_inner + segment.r_outer) / 2
                x, y = (
                    radius * math.cos(math.radians(segment.angle_center_deg)),
                    radius * math.sin(math.radians(segment.angle_center_deg)),
                )
            elif segment.shape == "rectangle":
                patch = Rectangle(
                    (segment.x_center - segment.width / 2, segment.y_center - segment.height / 2),
                    segment.width,
                    segment.height,
                    angle=segment.rotation_deg,
                    rotation_point="center",
                )
                x, y = segment.x_center, segment.y_center
            else:
                radius = segment.width / 2
                patch = Circle((segment.x_center, segment.y_center), radius, fill=False)
                axes.text(
                    segment.x_center,
                    segment.y_center,
                    "·" if segment.shape == "endon_out" else "×",
                    ha="center",
                    va="center",
                )
                x, y = segment.x_center, segment.y_center + radius + 0.1
            patch.set_facecolor(
                mpl.colors.to_rgba(color, 0.4)
                if segment.shape in {"wedge", "rectangle"}
                else "none"
            )
            patch.set_edgecolor("0.35")
            patch.set_linewidth(0.7)
            if segment.read_only:
                patch.set_linestyle("--")
            axes.add_patch(patch)
            label = segment.label or (
                str(segment.detector_id) if segment.detector_id is not None else ""
            )
            axes.text(x, y, label, ha="center", va="center", fontsize=11)
        axes.autoscale_view()
        axes.margins(0.12)
    figure.suptitle(
        f"{profile.identity.name}: {next(config.label for config in profile.detectors.acquisitions if config.key == acquisition) if acquisition is not None else 'Generic acquired channels'}"
    )
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.12, top=0.88, wspace=0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.hashsalt": "asymmetry-instruments", "svg.fonttype": "none"}):
        figure.savefig(output, format="svg", metadata={"Date": None, "Creator": "Asymmetry"})


def generate_profile(
    profile: InstrumentProfile, directory: Path, *, image_prefix: str = ""
) -> Path:
    """Write a deterministic reviewer artifact for every declared acquisition."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = profile.identity.key.replace(".", "_")
    diagrams = {}
    for config in profile.detectors.acquisitions:
        name = f"{stem}-{config.key}.svg"
        render_profile_diagram(profile, config.key, directory / name)
        diagrams[config.key] = f"{image_prefix}/{name}" if image_prefix else name
    page = directory / f"{stem}.rst"
    page.write_text(profile_rst(profile, diagrams), encoding="utf-8")
    return page


def setup(app):
    from docutils import nodes
    from docutils.parsers.rst import Directive
    from docutils.statemachine import StringList
    from sphinx.util.nodes import nested_parse_with_titles

    class InstrumentProfileDirective(Directive):
        required_arguments = 1

        def run(self):
            env = self.state.document.settings.env
            for dependency in (Path(__file__), *Path(instruments.__file__).parent.glob("*.py")):
                env.note_dependency(str(dependency.resolve()))
            profile = PROFILES[self.arguments[0]]
            directory = Path(env.srcdir) / "_generated" / "instruments"
            page = generate_profile(profile, directory, image_prefix="/_generated/instruments")
            container = nodes.container()
            nested_parse_with_titles(
                self.state, StringList(page.read_text().splitlines(), source=str(page)), container
            )
            return container.children

    app.add_directive("instrument-profile", InstrumentProfileDirective)
    return {"parallel_read_safe": False, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_profile(EMU, args.output))
