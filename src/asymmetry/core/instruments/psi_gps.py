"""PSI GPS: configuration-specific readouts of shared detector hardware."""

from asymmetry.core.instruments.geometry import (
    BankRow,
    BeamDirection,
    Composite,
    Directional,
    Occurrence,
    View,
    ViewOrientation,
)
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
    PhysicalBank,
    PhysicalElement,
    Preset,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

CHANNEL_ORDERS = (
    (
        "bin_combined",
        ("F", "B", "U", "D", "R", "L"),
        ("Forward", "Backward", "Up", "Down", "Right", "Left"),
        None,
    ),
    (
        "root_split",
        ("F", "B", "UB", "UF", "DB", "DF", "RB", "RF", "LB", "LF", "M"),
        (
            "Forward",
            "Backward",
            "Up_B",
            "Up_F",
            "Down_B",
            "Down_F",
            "Right_B",
            "Right_F",
            "Left_B",
            "Left_F",
            "Mob-RL",
        ),
        "GPS-RD",
    ),
    (
        "root_combined_split",
        ("F", "B", "U", "D", "R", "L", "UB", "UF", "DB", "DF", "RB", "RF", "LB", "LF", "M"),
        (
            "Forward",
            "Backward",
            "Up",
            "Down",
            "Right",
            "Left",
            "Up_B",
            "Up_F",
            "Down_B",
            "Down_F",
            "Right_B",
            "Right_F",
            "Left_B",
            "Left_F",
            "Mob-RL",
        ),
        "GPS-RD15",
    ),
)

# Beam, Left and Up define this schematic frame; positions remain view-local.
VIEWS = {
    key: tuple(
        View(
            name,
            label,
            "plan",
            "Dimensionless readout schematic: beam points right, top view upward is Left and side view upward is Up. Combined and split rows are distinct acquired readouts, not a sum equation. Grey Forward/Backward rectangles in the side view are read-only context, not extra channels.",
            Composite(
                (
                    *(
                        Directional(
                            direction,
                            (BankRow((Occurrence(channel=channel, editable=editable),), (1.7,)),),
                            2.2,
                        )
                        for direction, channel in (("right", "F"), ("left", "B"))
                    ),
                    *(
                        Directional(
                            direction,
                            tuple(
                                BankRow(
                                    tuple(Occurrence(channel=channel) for channel in row),
                                    (2.05,) if len(row) == 1 else (1.05, 1.05),
                                    label=kind if key == "root_combined_split" else "",
                                )
                                for kind, row in (
                                    ("combined", (bank,)),
                                    ("split", (bank + "B", bank + "F")),
                                )
                                if all(channel in order for channel in row)
                            ),
                        )
                        for direction, bank in (("up", upper), ("down", lower))
                    ),
                )
            ),
            orientation=orientation,
        )
        for name, label, upper, lower, editable, orientation in (
            (
                "top_view",
                "Top view",
                "L",
                "R",
                True,
                ViewOrientation((1, 0, 0), (0, 1, 0)),
            ),
            (
                "side_view",
                "Side view",
                "U",
                "D",
                False,
                ViewOrientation((1, 0, 0), (0, 0, 1)),
            ),
        )
    )
    for key, order, labels, legacy_key in CHANNEL_ORDERS
}

ACQUISITIONS = tuple(
    Acquisition(
        key,
        f"{len(order)}-channel {key.replace('_', ' ')} readout",
        tuple(
            Channel(
                channel,
                index + 1,
                (channel + "B", channel + "F")
                if channel in {"U", "D", "R", "L"}
                else (channel,)
                if channel not in {"F", "M"}
                else (),
                label=label,
                file_labels=(label,),
                relationship="Hardware association is descriptive: separate readout roads need not sum exactly or be independent. Fcenter and Mobile contributions depend on unrecorded configuration; no port is inferred.",
            )
            for index, (channel, label) in enumerate(zip(order, labels, strict=True))
        ),
        VIEWS[key],
        layout_key=legacy_key,
        beam=BeamDirection((1, 0, 0)),
        auxiliary=(Occurrence(channel="M"),) if "M" in order else (),
    )
    for key, order, labels, legacy_key in CHANNEL_ORDERS
)

PRESETS = tuple(
    ConfigurationPresets(
        key,
        (
            Preset(
                "Longitudinal",
                (ChannelGroup(1, "Forward", ("F",)), ChannelGroup(2, "Backward", ("B",))),
                2,
                1,
            ),
            Preset(
                "Transverse (Vector)",
                (
                    ChannelGroup(1, "Up-Down Up", up),
                    ChannelGroup(2, "Up-Down Down", down),
                    ChannelGroup(3, "Left-Right Right", right),
                    ChannelGroup(4, "Left-Right Left", left),
                ),
                1,
                2,
                (
                    AsymmetryProjection(
                        "Up-Down", 1, 2, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]
                    ),
                    AsymmetryProjection(
                        "Left-Right", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Left-Right"]
                    ),
                ),
            ),
            Preset(
                "Spin-rotated (B+U/F+D)",
                (ChannelGroup(1, "B+U", ("B", *up)), ChannelGroup(2, "F+D", ("F", *down))),
                1,
                2,
            ),
            Preset(
                "WEP (spin-rotated)",
                (
                    ChannelGroup(1, "F", ("F",)),
                    ChannelGroup(2, "B", ("B",)),
                    ChannelGroup(3, "U", up),
                    ChannelGroup(4, "D", down),
                ),
                2,
                1,
                (
                    AsymmetryProjection(
                        "FB", 2, 1, alpha=0.75, tint=TRANSVERSE_PROJECTION_TINTS["Fwd-Back"]
                    ),
                    AsymmetryProjection("UD", 3, 4, tint=TRANSVERSE_PROJECTION_TINTS["Top-Bottom"]),
                ),
            ),
        ),
    )
    for key, up, down, right, left in (
        ("bin_combined", ("U",), ("D",), ("R",), ("L",)),
        ("root_split", ("UB", "UF"), ("DB", "DF"), ("RB", "RF"), ("LB", "LF")),
        ("root_combined_split", ("UB", "UF"), ("DB", "DF"), ("RB", "RF"), ("LB", "LF")),
    )
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.gps", "PSI", "GPS", "1", "GPS", "GPS", "piM3.2"),
    source=Source(
        "continuous",
        "Continuous surface-muon source; beam preparation and recorded timing remain run-specific.",
    ),
    recognition=Recognition(("GPS", "LMU_BULKMUSR_GPS"), ("gps",)),
    detectors=Detectors(
        (
            PhysicalBank(
                "assembly",
                "Sample detector assembly",
                "Split U/D/R/L counters, backward detector, configuration-dependent forward/veto and Mobile elements.",
            ),
        ),
        tuple(
            PhysicalElement(key, label, "assembly")
            for key, label in (
                ("Fouter", "Forward outer"),
                ("Fcenter", "Forward centre"),
                ("B", "Backward"),
                ("UB", "Up backward half"),
                ("UF", "Up forward half"),
                ("DB", "Down backward half"),
                ("DF", "Down forward half"),
                ("RB", "Right backward half"),
                ("RF", "Right forward half"),
                ("LB", "Left backward half"),
                ("LF", "Left forward half"),
                ("M", "Mobile"),
            )
        ),
        ACQUISITIONS,
    ),
    presets=PRESETS,
    quirks=(
        "Six-channel BIN, eleven-channel split ROOT and fifteen-channel combined-plus-split ROOT maps retain separate legacy layout identities GPS, GPS-RD and GPS-RD15.",
        "Partial classic BIN files may omit Left; unavailable preset channels are not invented.",
        "Mobile and, in the fifteen-channel map, combined transverse channels remain selectable but absent from stock presets.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (
            Reference(
                "gps-paper",
                "Amato et al., Rev. Sci. Instrum. 88, 093301 (2017)",
                "https://arxiv.org/abs/1705.10687",
            ),
            Reference("gps-page", "PSI GPS instrument", "https://www.psi.ch/en/smus/gps"),
        ),
        (
            Evidence(
                "identity",
                EvidenceStatus.SOURCE,
                "GPS on piM3.2.",
                "gps-page",
                "Instrument description",
            ),
            Evidence(
                "source",
                EvidenceStatus.SOURCE,
                "Continuous surface-muon beam; no automatic run timing is inferred.",
                "gps-paper",
                "Section II",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Existing GPS metadata and filename tokens; exact historical export applicability needs review.",
            ),
            Evidence(
                "physical.banks",
                EvidenceStatus.SOURCE,
                "One detector assembly; view panels are not physical banks.",
                "gps-paper",
                "Section II A, Figure 2",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.SOURCE,
                "Split lateral counters, configuration-dependent Fcenter and Mobile role; this is not a complete instrumentation inventory.",
                "gps-paper",
                "Section II A, Figure 2 and Table I",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "Established export maps and partial-file handling; no private acquisition file is used as published evidence.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.AUTHORED,
                "Presets require explicit application; recorded settings retain their own provenance.",
            ),
            *(
                Evidence(
                    f"acquisition.{config.key}",
                    EvidenceStatus.UNVERIFIED,
                    "Preserved ordinal/label map; combined/split roads are descriptive, not an exact count equation or independence claim.",
                )
                for config in ACQUISITIONS
            ),
            *(
                Evidence(
                    f"view.{config.key}.{view.key}",
                    EvidenceStatus.SCHEMATIC,
                    "Dimensionless readout drawing; repeated contexts do not duplicate hardware. Rows identify readouts, not additional detector plates. Mobile is auxiliary with unspecified position.",
                )
                for config in ACQUISITIONS
                for view in config.views
            ),
            *(
                Evidence(
                    f"preset.{entry.acquisition}.{preset.name}",
                    EvidenceStatus.AUTHORED,
                    "Explicit compatibility memberships and leg order. Longitudinal uses B minus alpha F, consistent with the GPS paper Eq. (2). WEP's 0.75/1.0 are initial projection balances, not universal measured calibrations; vector labels do not certify calibrated components.",
                )
                for entry in PRESETS
                for preset in entry.presets
            ),
        ),
    ),
    review=Review(
        "1",
        "Physical affiliations, export ordinals, presets and schematic choices are separately qualified.",
    ),
)
