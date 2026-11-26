"""ISIS HiFi: manual detector memberships with a dimensionless two-bank schematic."""

from asymmetry.core.instruments.geometry import Circular, Occurrence, View, Wedge
from asymmetry.core.instruments.layout import TRANSVERSE_PROJECTION_TINTS, AsymmetryProjection
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
    OrientationGroupingDefault,
    PhysicalBank,
    PhysicalElement,
    Preset,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

FORWARD = tuple(f"c{number}" for number in range(1, 33))
BACKWARD = tuple(f"c{number}" for number in range(33, 65))

PROFILE = InstrumentProfile(
    identity=Identity("isis.hifi", "ISIS", "HiFi", "2", "HiFi", "HiFi"),
    source=Source(
        "pulsed",
        "First source bunch split toward HiFi and EMU; no pulse-response default is inferred.",
    ),
    recognition=Recognition(("HiFi",), ("hifi",)),
    detectors=Detectors(
        banks=(
            PhysicalBank("forward", "Forward", "Disc bank with 32 numbered elements."),
            PhysicalBank("backward", "Backward", "Disc bank with 32 numbered elements."),
        ),
        elements=tuple(
            PhysicalElement(f"d{number}", str(number), bank)
            for bank, numbers in (("forward", range(1, 33)), ("backward", range(33, 65)))
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
                        "Viewed from downstream looking upstream. Forward numbering is counterclockwise, backward numbering clockwise; the backward half-sector offset is inferred from the manual diagram. Radii are dimensionless drawing choices.",
                        Circular(
                            rows=tuple((Occurrence(channel=channel),) for channel in channels),
                            shapes=(Wedge(0.28, 1.0, 5.625),),
                            radii=(0.0,),
                            start_angle=start,
                            step_angle=step,
                        ),
                    )
                    for bank, label, channels, start, step in (
                        ("forward", "Forward", FORWARD, 270.0, 11.25),
                        ("backward", "Backward", BACKWARD, 264.375, -11.25),
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
                    (ChannelGroup(1, "Forward", FORWARD), ChannelGroup(2, "Backward", BACKWARD)),
                    1,
                    2,
                ),
                Preset(
                    "Transverse (Vector)",
                    (
                        ChannelGroup(
                            1,
                            "Left-Right Left",
                            tuple(f"c{number}" for number in (*range(5, 14), *range(52, 61))),
                        ),
                        ChannelGroup(
                            2,
                            "Left-Right Right",
                            tuple(f"c{number}" for number in (*range(21, 30), *range(36, 45))),
                        ),
                        ChannelGroup(
                            3,
                            "Top-Bottom Top",
                            tuple(f"c{number}" for number in (*range(13, 22), *range(44, 53))),
                        ),
                        ChannelGroup(
                            4,
                            "Top-Bottom Bottom",
                            tuple(
                                f"c{number}"
                                for number in (*range(1, 6), *range(29, 37), *range(60, 65))
                            ),
                        ),
                    ),
                    1,
                    2,
                    (
                        AsymmetryProjection(
                            "Left-Right", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Left-Right"]
                        ),
                        AsymmetryProjection(
                            "Top-Bottom", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]
                        ),
                    ),
                ),
            ),
        ),
    ),
    quirks=(),
    defaults=Defaults(
        legacy_v1_zero_grouping=OrientationGroupingDefault(
            "nexus64", "Longitudinal", "Longitudinal"
        )
    ),
    sources=Sources(
        references=(
            Reference(
                "hifi-guide",
                "HiFi User Manual",
                "https://isis.stfc.ac.uk/wp-content/uploads/2025/04/HiFi-User-Manual.pdf",
            ),
            Reference(
                "rigi-school-data",
                "2026 ISIS/PSI Muon School at Rigi Kaltbad data",
                "https://shadow.nd.rl.ac.uk/wimda/",
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
                "Instrument identity only; beamline is unknown and no dated hardware certification is asserted.",
                "hifi-guide",
                "Section 7.1, pp.30–31",
            ),
            Evidence(
                "source",
                EvidenceStatus.SOURCE,
                "Pulsed source/bunch routing only; no effective analysis pulse width is inferred.",
                "musr-guide",
                "Section 2.4, p.7",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Metadata and filename tokens are application conventions, not an independently checked export-header contract. PSI HIFI remains a distinct facility-qualified identity.",
            ),
            Evidence(
                "physical.banks",
                EvidenceStatus.SOURCE,
                "Two disc banks, each with 32 elements; no numerical dimensions or acceptance weights are declared.",
                "hifi-guide",
                "Section 7.1, p.30",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.SOURCE,
                "Numbered elements and bank affiliations follow the guide; acquired-channel wiring is separately unverified.",
                "hifi-guide",
                "Section 7.1, p.30, detector diagrams",
            ),
            Evidence(
                "acquisition.nexus64",
                EvidenceStatus.UNVERIFIED,
                "Identity mapping from acquired ordinal to numbered element; exact export wiring has not been independently established.",
            ),
            Evidence(
                "view.nexus64.forward",
                EvidenceStatus.SCHEMATIC,
                "Numbering follows the guide's upstream-looking diagram; normalized hole/outer radii are drawing choices, not measured dimensions.",
                "hifi-guide",
                "Section 7.1, p.30, detector diagrams",
            ),
            Evidence(
                "view.nexus64.backward",
                EvidenceStatus.INFERRED,
                "Half-sector angular offset is inferred from the detector diagram, not angular metrology. Normalized radii remain schematic.",
                "hifi-guide",
                "Section 7.1, p.30, detector diagrams",
            ),
            Evidence(
                "preset.nexus64.Longitudinal",
                EvidenceStatus.SOURCE,
                "Manual forward/backward bank sums; the balance coefficient remains experiment-specific.",
                "hifi-guide",
                "Section 7.1, pp.30–31",
            ),
            Evidence(
                "preset.nexus64.Transverse (Vector)",
                EvidenceStatus.SOURCE,
                "Only RF group memberships and their intentional shared boundary elements are source-backed. The combined two-projection presentation and initial alpha=1 are Asymmetry conventions, not timing or vector-sensitivity calibration.",
                "hifi-guide",
                "Section 7.1, pp.30–31",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "No instrument-specific file quirk is declared; this does not certify all export formats.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.SOURCE,
                "The verified single-period legacy declaration and acquisition support the Longitudinal grouping.",
                "rigi-school-data",
                "HiFi run 9023",
            ),
        ),
    ),
    review=Review("2", "Profile definition, schematic, and legacy NeXus grouping."),
)
