"""ISIS MuSR: explicit manual group signs and a dimensionless two-array schematic."""

from asymmetry.core.instruments.geometry import Circular, Occurrence, View, Wedge
from asymmetry.core.instruments.layout import TRANSVERSE_PROJECTION_TINTS, AsymmetryProjection
from asymmetry.core.instruments.schema import (
    AbsentGroupingDefault,
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

BACKWARD = tuple(f"c{number}" for number in range(1, 33))
FORWARD = tuple(f"c{number}" for number in range(33, 65))

PROFILE = InstrumentProfile(
    identity=Identity("isis.musr", "ISIS", "MuSR", "2", "MuSR", "MuSR"),
    source=Source(
        "pulsed", "Second source bunch routed toward MuSR; no pulse-response default is inferred."
    ),
    recognition=Recognition(("MuSR",), ("musr",)),
    detectors=Detectors(
        banks=(
            PhysicalBank(
                "backward", "Backward", "Axially separated array with 32 numbered elements."
            ),
            PhysicalBank(
                "forward", "Forward", "Axially separated array with 32 numbered elements."
            ),
        ),
        elements=tuple(
            PhysicalElement(f"d{number}", str(number), bank)
            for bank, numbers in (("backward", range(1, 33)), ("forward", range(33, 65)))
            for number in numbers
        ),
        acquisitions=(
            Acquisition(
                "nexus64",
                "64-channel positional export",
                tuple(Channel(f"c{number}", number, (f"d{number}",)) for number in range(1, 65)),
                tuple(
                    View(
                        bank,
                        label,
                        "radial",
                        "Two arrays shown separately using dimensionless annular bands. Backward numbering is counterclockwise and forward numbering clockwise in this schematic; exact angular centers and radii are not measured coordinates. Instrument rotation and applied-field orientation remain run-specific.",
                        Circular(
                            rows=tuple((Occurrence(channel=channel),) for channel in channels),
                            shapes=(Wedge(0.28, 1.0, 5.625),),
                            radii=(0.0,),
                            start_angle=start,
                            step_angle=step,
                        ),
                    )
                    for bank, label, channels, start, step in (
                        ("backward", "Backward", BACKWARD, 230.625, 11.25),
                        ("forward", "Forward", FORWARD, 309.375, -11.25),
                    )
                ),
            ),
        ),
    ),
    presets=(
        ConfigurationPresets(
            "nexus64",
            (
                Preset(
                    "Longitudinal",
                    (ChannelGroup(1, "Backward", BACKWARD), ChannelGroup(2, "Forward", FORWARD)),
                    2,
                    1,
                ),
                Preset(
                    "Transverse (Vector)",
                    (
                        ChannelGroup(
                            1,
                            "Top-Bottom Top",
                            tuple(f"c{number}" for number in (*range(17, 25), *range(49, 57))),
                        ),
                        ChannelGroup(
                            2,
                            "Top-Bottom Bottom",
                            tuple(f"c{number}" for number in (*range(1, 9), *range(33, 41))),
                        ),
                        ChannelGroup(
                            3,
                            "Fwd-Back Forward",
                            tuple(f"c{number}" for number in (*range(9, 17), *range(57, 65))),
                        ),
                        ChannelGroup(
                            4,
                            "Fwd-Back Backward",
                            tuple(f"c{number}" for number in (*range(25, 33), *range(41, 49))),
                        ),
                    ),
                    1,
                    2,
                    (
                        AsymmetryProjection(
                            "Top-Bottom", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]
                        ),
                        AsymmetryProjection(
                            "Fwd-Back", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Fwd-Back"]
                        ),
                    ),
                ),
            ),
        ),
    ),
    quirks=(),
    defaults=Defaults(
        absent_file_grouping=AbsentGroupingDefault("nexus64", "Longitudinal", "Longitudinal")
    ),
    sources=Sources(
        references=(
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
                "Instrument identity only; beamline is unknown and no dated hardware certification is asserted.",
                "musr-guide",
                "Section 2.6.1, pp.12–13",
            ),
            Evidence(
                "source",
                EvidenceStatus.SOURCE,
                "Second source-bunch routing only; no effective analysis pulse width is inferred.",
                "musr-guide",
                "Section 2.4, p.7",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Metadata and filename tokens are application conventions, not an independently checked export-header contract. Detector count alone never establishes identity.",
            ),
            Evidence(
                "physical.banks",
                EvidenceStatus.SOURCE,
                "Two axially separated 32-element arrays; no numerical radii, separation or acceptance model is declared.",
                "musr-guide",
                "Section 2.6.1, p.12; Fig.9, p.13",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.SOURCE,
                "Numbered elements and bank affiliations follow the guide; acquired-channel wiring is separately unverified.",
                "musr-guide",
                "Section 2.6.1, pp.12–13",
            ),
            Evidence(
                "acquisition.nexus64",
                EvidenceStatus.UNVERIFIED,
                "Identity mapping from acquired ordinal to numbered element; exact export wiring has not been independently established.",
            ),
            Evidence(
                "view.nexus64.backward",
                EvidenceStatus.SCHEMATIC,
                "Numbering follows the detector diagram; angular centers and normalized annular bounds are schematic, not independently measured coordinates.",
                "musr-guide",
                "Fig.9, p.13",
            ),
            Evidence(
                "view.nexus64.forward",
                EvidenceStatus.SCHEMATIC,
                "The opposing numbering direction follows the detector diagram; angular centers and normalized radii remain schematic.",
                "musr-guide",
                "Fig.9, p.13",
            ),
            Evidence(
                "preset.nexus64.Longitudinal",
                EvidenceStatus.SOURCE,
                "Forward=33–64 and backward=1–32 in the manual's longitudinal mode; the balance remains experiment-specific. Instrument identity alone does not establish applied-field orientation.",
                "musr-guide",
                "Section 2.6.1, p.12",
            ),
            Evidence(
                "preset.nexus64.Transverse (Vector)",
                EvidenceStatus.SOURCE,
                "Only manual group memberships are source-backed. The combined two-projection presentation and initial alpha=1 are Asymmetry conventions, not a quantitative vector-sensitivity calibration.",
                "musr-guide",
                "Section 2.6.1, p.12",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "No instrument-specific file-format quirk is declared.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.AUTHORED,
                "Absent-table eligibility is an Asymmetry policy: only fresh, unconfigured nexus64 acquisitions with explicit longitudinal detector orientation use the existing Longitudinal preset. Its forward/backward membership follows the manual. Present file tables and recorded scientific requests retain precedence.",
                "musr-guide",
                "Section 2.6.1, p.12",
            ),
        ),
    ),
    review=Review("2", "Profile definition and schematic."),
)
