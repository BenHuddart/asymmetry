"""Generate instrument review prose and diagrams from resolved profile data."""

from __future__ import annotations

import argparse
import csv
import textwrap
from functools import partial
from pathlib import Path

from asymmetry.core import instruments
from asymmetry.core.instruments.catalog import PROFILES as BUNDLED_PROFILES
from asymmetry.core.instruments.isis_emu import PROFILE as EMU
from asymmetry.core.instruments.numbering import numbering_template
from asymmetry.core.instruments.readouts import layout_for_readout_set
from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.core.instruments.schema import InstrumentProfile
from asymmetry.gui.utils import detector_drawing

PROFILES = {profile.identity.key: profile for profile in BUNDLED_PROFILES}
GENERATED_REVIEW_MARKER = ".. Generated instrument review page; do not edit."


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
    if config.readout_sets:
        return "".join(
            f"**{readouts.label}**\n\n"
            + _readout_figure(
                profile,
                config,
                uri if index == 0 else uri.removesuffix(".svg") + f"-{readouts.key}.svg",
            )
            for index, readouts in enumerate(config.readout_sets)
        )
    return _readout_figure(profile, config, uri)


def _readout_figure(profile, config, uri):
    text = f".. figure:: {uri}\n   :width: 100%\n   :alt: {profile.identity.name} labelled {config.label} schematic\n\n   {config.label}; numbers identify the recorded histograms.\n\n"
    text += f":download:`Full-size labelled diagram (SVG) <{uri}>`\n\n"
    stem = uri.removesuffix(".svg")
    text += f":download:`Blank numbering diagram (SVG) <{stem}-numbering.svg>` · :download:`Editable numbering table (CSV) <{stem}-numbering.csv>`\n\n"
    text += "Template slot IDs are acquisition/channel identities, independent of view order. Fill in readout labels and histogram numbers; optional physical associations can be unknown or multiple.\n\n"
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
    assumptions = tuple(
        dict.fromkeys(
            (evidence.status.value, evidence.detail)
            for evidence in profile.sources.evidence
            if evidence.fact.startswith(("acquisition.", "preset."))
            and evidence.status.value in {"Unverified", "Asymmetry-defined"}
        )
    )
    if assumptions:
        text += (
            "Key assumptions\n---------------\n\n"
            + "\n".join(f"* **{status}:** {detail}" for status, detail in assumptions)
            + "\n\n"
        )
    text += f"Source\n------\n\nDeclared family: **{profile.source.kind}**. {profile.source.description}\n\n"
    text += "Source declarations do not replace recorded timing, grouping, correction or calibration settings.\n\n"
    text += "Recognition\n-----------\n\n"
    text += _table(
        ("Evidence kind", "Declared names", "Matching"),
        (
            ("Metadata", ", ".join(profile.recognition.aliases) or "None", "Compact substring"),
            (
                "Filename",
                ", ".join(profile.recognition.filename_tokens) or "None",
                "Basename token boundaries; run digits may follow",
            ),
        ),
    )
    text += "Facility precedence and acquired-channel checks remain authoritative; an ambiguous name alone does not certify an instrument.\n\n"
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
        layout = layout_from_profile(profile, config.key)
        text += f"Legacy layout key: ``{layout.name}``; displayed instrument: **{layout.display_name or layout.name}**.\n\n"
        if config.readout_sets:
            text += _table(
                ("Readout set", "Acquired channels", "Default preset"),
                ((r.label, ", ".join(r.channels), r.default_preset) for r in config.readout_sets),
            )
            text += "These are alternative analysis selections, not count equations or independent copies of shared events.\n\n"
        if config.combined_readouts:
            text += _table(
                ("Combined channel", "Known constituents"),
                ((r.channel, ", ".join(r.constituents)) for r in config.combined_readouts),
            )
        if config.conditional_contributions:
            text += _table(
                ("Conditional contributor", "Possible destinations"),
                ((r.channel, ", ".join(r.destinations)) for r in config.conditional_contributions),
            )
            text += "An unresolved contributor conflicts with either possible combined destination when selected together; no port is inferred.\n\n"
        if config.reference_arrows:
            text += _table(
                ("Schematic direction cue", "Start", "End"),
                ((arrow.label, arrow.start, arrow.end) for arrow in config.reference_arrows),
            )
            text += "Direction cues are schematic declarations, not measured run-specific field or spin settings.\n\n"
        if config.key in diagrams and config is not leading:
            text += _review_figure(profile, config, diagrams[config.key])
        text += _table(
            ("View", "Coordinate status", "Units", "Viewing convention"),
            ((view.label, view.scale, view.units, view.convention) for view in config.views),
        )
        if config.beam is not None:
            text += f"Beam direction in the common arrangement frame: ``{config.beam.vector}``.\n\n"
            beam_rows = []
            for view, bank in zip(config.views, layout.banks, strict=True):
                orientation, beam = view.orientation, bank.beam
                beam_rows.append(
                    (
                        view.label,
                        orientation.right if orientation else "Unknown",
                        orientation.up if orientation else "Unknown",
                        orientation.toward
                        if orientation and orientation.toward is not None
                        else "Unknown",
                        beam.screen if beam else "Unknown",
                        beam.toward if beam and beam.toward is not None else "Unknown",
                    )
                )
            text += _table(
                (
                    "View",
                    "Screen right",
                    "Screen up",
                    "Toward viewer",
                    "Beam screen components",
                    "Beam toward component",
                ),
                beam_rows,
            )
            text += "Toward components are positive out of the screen and negative into it. Unknown viewing sense remains unknown. Basis vectors describe the declared arrangement frame, independently of view-local detector placements.\n\n"
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
    rule = profile.defaults.absent_file_grouping
    if rule is not None:
        text += (
            f"An absent file grouping table on fresh **{rule.acquisition}** data with "
            f"explicit **{rule.detector_orientation}** detector orientation selects "
            f"the **{rule.preset}** preset. Present file tables and saved requests "
            "retain precedence.\n\n"
        )
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
    blank: bool = False,
    readout_set: str | None = None,
) -> None:
    """Render the same resolved primitive coordinates as the detector editor."""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle, Wedge

    layout = layout_from_profile(profile, acquisition, n_histograms=n_histograms)
    if layout.readout_sets:
        layout = layout_for_readout_set(layout, readout_set or layout.readout_sets[0].key)
    figure = Figure(figsize=(max(6, 5 * len(layout.banks)), 6.5 if layout.auxiliary else 5.3))
    bank_axes, auxiliary_axes = detector_drawing.schematic_axes(figure, layout)
    slots = (
        {slot.ordinal: slot for slot in numbering_template(profile, acquisition)}
        if acquisition
        else {}
    )
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
    panels = [
        (axes, bank.name, bank.segments, bank.beam)
        for axes, bank in zip(bank_axes, layout.banks, strict=True)
    ]
    if auxiliary_axes is not None:
        panels.append((auxiliary_axes, "Auxiliary — position unspecified", layout.auxiliary, None))
    for index, (axes, name, segments, beam) in enumerate(panels):
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(name)
        for segment in segments:
            color = (
                "0.93"
                if blank or segment.read_only or segment.detector_id not in memberships
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
                x, y = segment.x_center, segment.y_center - radius - 0.1
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
            label = (
                str(segment.detector_id) if segment.detector_id is not None else segment.label or ""
            )
            if blank and segment.detector_id in slots:
                slot = slots[segment.detector_id]
                label = slot.channel if segment.read_only else f"{slot.channel}\n___\n# ___"
            elif segment.shape == "rectangle" and segment.detector_id in slots:
                label = f"{segment.detector_id}\n{slots[segment.detector_id].channel}"
            axes.text(
                x,
                y,
                label,
                ha="center",
                va="top" if segment.shape.startswith("endon_") else "center",
                fontsize=6.5 if blank else 9,
                color="0.5" if segment.read_only else "0.15",
            )
        detector_drawing.draw_beam(axes, beam)
        detector_drawing.draw_readout_rows(axes, segments)
        for arrow in layout.reference_arrows:
            axes.annotate(
                "",
                xy=arrow.end,
                xytext=arrow.start,
                color=arrow.color,
                fontsize=10,
                ha="center",
                va="bottom",
                arrowprops={"arrowstyle": "->", "color": arrow.color},
            )
            axes.text(
                (arrow.start[0] + arrow.end[0]) / 2,
                (arrow.start[1] + arrow.end[1]) / 2 + 0.1,
                arrow.label,
                color=arrow.color,
                ha="center",
                va="bottom",
                fontsize=10,
            )
            axes.update_datalim((arrow.start, arrow.end))
        axes.autoscale_view()
        axes.margins(0.12)
    figure.suptitle(
        f"{profile.identity.name}: {next(config.label for config in profile.detectors.acquisitions if config.key == acquisition) if acquisition is not None else 'Generic acquired channels'}"
    )
    figure.text(
        0.5,
        0.055,
        " · ".join(dict.fromkeys(f"{view.scale}; {view.units}" for view in views))
        or "Dimensionless schematic",
        ha="center",
        fontsize=8,
    )
    if blank:
        figure.text(
            0.5,
            0.02,
            f"Slot prefix: {profile.identity.key}/{acquisition}/ · Blanks: readout label / histogram number; grey: repeated context",
            ha="center",
            fontsize=8,
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
        for index, readouts in enumerate(config.readout_sets or (None,)):
            suffix = f"-{readouts.key}" if index else ""
            name = f"{stem}-{config.key}{suffix}.svg"
            key = readouts.key if readouts else None
            slots = numbering_template(profile, config.key)
            if readouts is not None:
                slots = tuple(slot for slot in slots if slot.channel in readouts.channels)
            for blank in (False, True):
                output = name.replace(".svg", "-numbering.svg") if blank else name
                render_profile_diagram(
                    profile, config.key, directory / output, blank=blank, readout_set=key
                )
            with (directory / name.replace(".svg", "-numbering.csv")).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "Slot ID",
                        "Readout label",
                        "Histogram ordinal",
                        "Physical association / detector label(s), if known",
                    )
                )
                writer.writerows((slot.id, "", "", "") for slot in slots)
            if index == 0:
                diagrams[config.key] = f"{image_prefix}/{name}" if image_prefix else name
    page = directory / f"{stem}.rst"
    page.write_text(profile_rst(profile, diagrams), encoding="utf-8")
    return page


def remove_retired_review_html(app, *, docnames):
    """Remove only HTML corresponding to retired marker-owned source stubs."""
    if app.builder.name != "html":
        return
    for docname in docnames:
        output = Path(app.builder.get_outfilename(docname))
        if not output.resolve().is_relative_to(Path(app.outdir).resolve()):
            raise ValueError(
                "Generated instrument review output must stay inside the build directory"
            )
        output.unlink(missing_ok=True)


def generate_review_pages(app, config):
    """Materialise catalog-owned source stubs before Sphinx discovers documents."""
    if config.instrument_review_directory is None:
        return
    source = Path(app.srcdir).resolve()
    directory = (source / config.instrument_review_directory).resolve()
    if directory == source or not directory.is_relative_to(source):
        raise ValueError(
            "Generated instrument review directory must be inside the source directory"
        )
    wanted = {
        key.replace(".", "_")
        + ".rst": f"{GENERATED_REVIEW_MARKER}\n\n.. instrument-profile:: {key}\n"
        for key in sorted(PROFILES)
    }
    if any(Path(name).name != name or "\\" in name for name in wanted):
        raise ValueError("Generated instrument review pages require local filenames")
    directory.mkdir(parents=True, exist_ok=True)
    if len(wanted) != len(PROFILES):
        raise ValueError("Instrument review profiles require distinct public page names")
    retired = []
    for page in directory.glob("*.rst"):
        if page.is_symlink() or not page.resolve().is_relative_to(directory):
            raise ValueError(
                "Generated instrument review sources must be ordinary files inside their directory"
            )
        owned = page.read_text(encoding="utf-8").startswith(GENERATED_REVIEW_MARKER + "\n")
        if page.name in wanted and not owned:
            raise ValueError(f"Cannot replace authored instrument review source: {page}")
        if owned and page.name not in wanted:
            retired.append(page)
    for page in retired:
        page.unlink()
    for name, text in wanted.items():
        page = directory / name
        if not page.exists() or page.read_text(encoding="utf-8") != text:
            page.write_text(text, encoding="utf-8")
    if retired:
        app.connect(
            "builder-inited",
            partial(
                remove_retired_review_html,
                docnames=tuple(
                    page.relative_to(source).with_suffix("").as_posix() for page in retired
                ),
            ),
        )


def setup(app):
    from docutils import nodes
    from docutils.parsers.rst import Directive
    from docutils.statemachine import StringList
    from sphinx.util.nodes import nested_parse_with_titles

    class InstrumentProfileDirective(Directive):
        required_arguments = 1

        def run(self):
            env = self.state.document.settings.env
            for dependency in (
                Path(__file__),
                Path(detector_drawing.__file__),
                *Path(instruments.__file__).parent.glob("*.py"),
            ):
                env.note_dependency(str(dependency.resolve()))
            profile = PROFILES[self.arguments[0]]
            directory = Path(env.srcdir) / "_generated" / "instruments"
            page = generate_profile(profile, directory, image_prefix="/_generated/instruments")
            container = nodes.container()
            nested_parse_with_titles(
                self.state, StringList(page.read_text().splitlines(), source=str(page)), container
            )
            return container.children

    app.add_config_value("instrument_review_directory", None, "env", types=(str, type(None)))
    app.connect("config-inited", generate_review_pages)
    app.add_directive("instrument-profile", InstrumentProfileDirective)
    return {"parallel_read_safe": False, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_profile(EMU, args.output))
