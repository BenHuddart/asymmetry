"""ISIS EMU: axial hardware layers represented by concentric schematic bands."""

from asymmetry.core.instruments.geometry import Circular, Occurrence, View, Wedge
from asymmetry.core.instruments.layout import PROJECTION_TINTS, AsymmetryProjection
from asymmetry.core.instruments.schema import (
    Acquisition,
    Channel,
    ChannelGroup,
    ConfigurationPresets,
    Defaults,
    Detectors,
    Evidence,
    EvidenceStatus,
    Identity,
    InstrumentProfile,
    PhysicalBank,
    PhysicalElement,
    Preset,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

FORWARD_TRIPLETS = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
    (16, 17, 18),
    (19, 20, 21),
    (22, 23, 24),
    (25, 26, 27),
    (28, 29, 30),
    (31, 32, 33),
    (34, 35, 36),
    (37, 38, 39),
    (40, 41, 42),
    (43, 44, 45),
    (46, 47, 48),
)
BACKWARD_TRIPLETS = (
    (49, 50, 51),
    (52, 53, 54),
    (55, 56, 57),
    (58, 59, 60),
    (61, 62, 63),
    (64, 65, 66),
    (67, 68, 69),
    (70, 71, 72),
    (73, 74, 75),
    (76, 77, 78),
    (79, 80, 81),
    (82, 83, 84),
    (85, 86, 87),
    (88, 89, 90),
    (91, 92, 93),
    (94, 95, 96),
)

FORWARD_CHANNELS = tuple(f"c{number}" for number in range(1, 49))
BACKWARD_CHANNELS = tuple(f"c{number}" for number in range(49, 97))

PROFILE = InstrumentProfile(
    identity=Identity("isis.emu", "ISIS", "EMU", "1", "EMU", "EMU"),
    source=Source(
        "pulsed",
        "First source bunch split toward EMU and HiFi; no pulse-response default is inferred.",
    ),
    recognition=Recognition(("EMU",), ("emu",)),
    detectors=Detectors(
        banks=(
            PhysicalBank(
                "forward", "Forward", "Three axial layers, each with 16 azimuthal elements."
            ),
            PhysicalBank(
                "backward", "Backward", "Three axial layers, each with 16 azimuthal elements."
            ),
        ),
        elements=tuple(
            PhysicalElement(f"d{number}", str(number), bank, ("inner", "middle", "outer")[ring])
            for bank, rows in (("forward", FORWARD_TRIPLETS), ("backward", BACKWARD_TRIPLETS))
            for row in rows
            for ring, number in enumerate(row)
        ),
        acquisitions=(
            Acquisition(
                key="nexus96",
                label="96-channel positional export",
                channels=tuple(
                    Channel(f"c{number}", number, (f"d{number}",))
                    for rows in (FORWARD_TRIPLETS, BACKWARD_TRIPLETS)
                    for row in rows
                    for number in row
                ),
                views=tuple(
                    View(
                        key=bank,
                        label=label,
                        mode="radial",
                        convention="Viewed from downstream looking upstream; clockwise numbering from 12 o'clock. Concentric bands depict axial layers; radii are not physical measurements.",
                        arrangement=Circular(
                            rows=tuple(
                                tuple(Occurrence(channel=f"c{number}") for number in row)
                                for row in rows
                            ),
                            shapes=(
                                Wedge(0.28, 0.52, 11.25),
                                Wedge(0.52, 0.76, 11.25),
                                Wedge(0.76, 1.0, 11.25),
                            ),
                            radii=(0.0, 0.0, 0.0),
                            start_angle=90.0,
                            step_angle=-22.5,
                        ),
                    )
                    for bank, label, rows in (
                        ("forward", "Forward", FORWARD_TRIPLETS),
                        ("backward", "Backward", BACKWARD_TRIPLETS),
                    )
                ),
            ),
        ),
    ),
    presets=(
        ConfigurationPresets(
            "nexus96",
            (
                Preset(
                    name="Longitudinal",
                    groups=(
                        ChannelGroup(
                            1,
                            "Forward",
                            FORWARD_CHANNELS,
                        ),
                        ChannelGroup(
                            2,
                            "Backward",
                            BACKWARD_CHANNELS,
                        ),
                    ),
                    forward_group=1,
                    backward_group=2,
                ),
                Preset(
                    name="Vector Polarization",
                    groups=(
                        ChannelGroup(
                            1,
                            "Pz Forward",
                            FORWARD_CHANNELS,
                        ),
                        ChannelGroup(
                            2,
                            "Pz Backward",
                            BACKWARD_CHANNELS,
                        ),
                        ChannelGroup(
                            3,
                            "Py Top",
                            tuple(
                                f"c{number}"
                                for number in (*range(1, 13), *range(37, 61), *range(85, 97))
                            ),
                        ),
                        ChannelGroup(
                            4,
                            "Py Bottom",
                            tuple(f"c{number}" for number in (*range(13, 37), *range(61, 85))),
                        ),
                        ChannelGroup(
                            5,
                            "Px Left",
                            tuple(f"c{number}" for number in (*range(25, 49), *range(73, 97))),
                        ),
                        ChannelGroup(
                            6,
                            "Px Right",
                            tuple(f"c{number}" for number in (*range(1, 25), *range(49, 73))),
                        ),
                    ),
                    forward_group=1,
                    backward_group=2,
                    projections=(
                        AsymmetryProjection("P_x", 5, 6, alpha=1.0, tint=PROJECTION_TINTS["P_x"]),
                        AsymmetryProjection("P_y", 3, 4, alpha=1.0, tint=PROJECTION_TINTS["P_y"]),
                        AsymmetryProjection("P_z", 1, 2, alpha=1.0, tint=PROJECTION_TINTS["P_z"]),
                    ),
                ),
            ),
        ),
    ),
    quirks=(
        "Temperature-unit suspicion is conditional on header/log evidence; values are not automatically converted.",
    ),
    defaults=Defaults(apply_default_preset_on_load=False),
    sources=Sources(
        references=(
            Reference(
                "emu-guide",
                "EMU User Guide, updated March 2021",
                "https://www.isis.stfc.ac.uk/wp-content/uploads/2025/04/EMU_Manual.pdf",
            ),
            Reference(
                "musr-guide",
                "MuSR Manual, Baker et al., 29 June 2011",
                "https://www.isis.stfc.ac.uk/wp-content/uploads/2025/04/MuSR_manual.pdf",
            ),
        ),
        evidence=(
            Evidence(
                "identity",
                EvidenceStatus.SOURCE,
                "Instrument identity; this is not a dated hardware-configuration certification.",
                "emu-guide",
                "Introduction",
            ),
            Evidence(
                "source",
                EvidenceStatus.SOURCE,
                "Pulsed source/bunch routing only; effective analysis pulse widths remain separately authored settings.",
                "musr-guide",
                "Section 2.4, p.7",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Preserved metadata/filename token convention; no independently checked file-header or wiring example.",
            ),
            Evidence(
                "physical.banks",
                EvidenceStatus.SOURCE,
                "Two banks with three axial rings each and 16 numbered elements per ring. No measured radial coordinates are asserted.",
                "emu-guide",
                "Section 8.1, pp.33–34, Figs.15–16",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.SOURCE,
                "Numbered elements and axial-ring affiliations follow the manual; acquired-channel mapping is separately unverified.",
                "emu-guide",
                "Section 8.1, p.33",
            ),
            Evidence(
                "acquisition.nexus96",
                EvidenceStatus.UNVERIFIED,
                "Preserved identity mapping of acquired ordinal to numbered detector; exact export wiring not independently established.",
            ),
            Evidence(
                "view.nexus96.forward",
                EvidenceStatus.SCHEMATIC,
                "Azimuthal numbering follows the guide; normalized concentric radii are drawing choices representing axial layers.",
                "emu-guide",
                "Section 8.1, pp.33–34",
            ),
            Evidence(
                "view.nexus96.backward",
                EvidenceStatus.SCHEMATIC,
                "Azimuthal numbering follows the guide; normalized concentric radii are drawing choices representing axial layers.",
                "emu-guide",
                "Section 8.1, pp.33–34",
            ),
            Evidence(
                "preset.nexus96.Longitudinal",
                EvidenceStatus.SOURCE,
                "The ordinary longitudinal grouping sums each bank separately; calibration balance is experiment-specific.",
                "emu-guide",
                "Section 8.1.1, p.34",
            ),
            Evidence(
                "preset.nexus96.Vector Polarization",
                EvidenceStatus.AUTHORED,
                "Preserved Asymmetry-defined overlapping P_x/P_y/P_z sums. This is not a facility-validated quantitative vector measurement.",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "Existing conditional temperature warning; not a universal Celsius conversion or a newly verified file-format fact.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.AUTHORED,
                "Preserve current preset order, defaults and no automatic preset application on load.",
            ),
        ),
    ),
    review=Review("1", "Profile definition and schematic."),
)
