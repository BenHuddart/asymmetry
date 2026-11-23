"""PSI FLAME: explicit acquired channels and qualified schematic conventions."""

from asymmetry.core.instruments.geometry import (
    BankRow,
    BeamDirection,
    Composite,
    Directional,
    Occurrence,
    View,
    ViewOrientation,
)
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
    Preset,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

CHANNELS = (
    ("F", "Forward"),
    ("B", "Backward"),
    ("R", "Right"),
    ("L", "Left"),
    ("RF", "R_F"),
    ("RB", "R_B"),
    ("LF", "L_F"),
    ("LB", "L_B"),
)

ACQUISITION = Acquisition(
    "eight_channel",
    "Eight-channel readout",
    tuple(
        Channel(key, index + 1, label=label, file_labels=(label,))
        for index, (key, label) in enumerate(CHANNELS)
    ),
    (
        View(
            "flame_top_view",
            "FLAME top view",
            "plan",
            "Readout schematic; positions and sizes are dimensionless drawing choices. Viewed from above; beam points right.",
            Composite(
                tuple(
                    Directional(
                        direction,
                        (BankRow(tuple(Occurrence(channel=key) for key in keys), widths),),
                        distance,
                    )
                    for direction, keys, widths, distance in (
                        ("right", ("F",), (1.7,), 3.55),
                        ("left", ("B",), (1.7,), 3.55),
                        ("up", ("LB", "L", "LF"), (0.82, 2.18, 0.82), 2.18),
                        ("down", ("RB", "R", "RF"), (0.82, 2.18, 0.82), 2.18),
                    )
                )
            ),
            orientation=ViewOrientation((1, 0, 0), (0, 1, 0)),
        ),
    ),
    beam=BeamDirection((1, 0, 0)),
)

PRESETS = (
    Preset(
        "Longitudinal",
        (
            ChannelGroup(1, "Forward", ("F",)),
            ChannelGroup(2, "Backward", ("B",)),
        ),
        2,
        1,
    ),
    Preset(
        "Transverse",
        (
            ChannelGroup(1, "Right", ("R", "RB", "RF")),
            ChannelGroup(2, "Left", ("L", "LB", "LF")),
        ),
        1,
        2,
    ),
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.flame", "PSI", "FLAME", "1", "FLAME", "FLAME", "piM3.3"),
    source=Source(
        "continuous", "Continuous muon source; spin rotation, field and timing remain run-specific."
    ),
    recognition=Recognition(("FLAME",), ("flame",)),
    detectors=Detectors(
        (
            PhysicalBank(
                "assembly",
                "Detector assembly",
                "Eight positron detectors; channel-to-hardware map is unverified.",
            ),
        ),
        (),
        (ACQUISITION,),
    ),
    presets=(ConfigurationPresets("eight_channel", PRESETS),),
    quirks=(
        "Eight-channel positional map; beam and detector positions are schematic conventions.",
    ),
    defaults=Defaults(apply_default_preset_on_load=False),
    sources=Sources(
        (
            Reference(
                "instrument-page",
                "PSI FLAME instrument",
                "https://www.psi.ch/en/smus/flexible-advanced-musr-environment-flame",
            ),
        ),
        (
            Evidence(
                "identity",
                EvidenceStatus.SOURCE,
                "FLAME on piM3.3.",
                "instrument-page",
                "Instrument description",
            ),
            Evidence(
                "source",
                EvidenceStatus.SOURCE,
                "Continuous muon beam; recorded run settings take precedence.",
                "instrument-page",
                "Beam and instrument description",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Preserved metadata/filename aliases; facility precedence is retained.",
            ),
            Evidence(
                "physical.banks",
                EvidenceStatus.SOURCE,
                "Eight positron detectors.",
                "instrument-page",
                "Detector description",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.UNVERIFIED,
                "Individual hardware identities are not declared without a sourced map.",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "Preserved file conventions; schematic beam and readout labels do not certify run-specific instrumentation.",
            ),
            Evidence("defaults", EvidenceStatus.AUTHORED, "Explicit preset application."),
            Evidence(
                "acquisition.eight_channel",
                EvidenceStatus.UNVERIFIED,
                "Preserved ordinal/label map and explicit groups; no new hardware or statistical relationship is inferred.",
            ),
            *(
                Evidence(
                    f"view.eight_channel.{view.key}",
                    EvidenceStatus.SCHEMATIC,
                    "Dimensionless schematic; coordinates, rotations and any direction arrows are drawing conventions, not measured detector dimensions or run settings.",
                )
                for view in ACQUISITION.views
            ),
            *(
                Evidence(
                    f"preset.eight_channel.{preset.name}",
                    EvidenceStatus.AUTHORED,
                    "Compatibility grouping and ordered asymmetry legs; initial balances require experiment-specific calibration.",
                )
                for preset in PRESETS
            ),
        ),
    ),
    review=Review(
        "1",
        "Instrument identity and published detector count are separate from unverified export mapping and authored presets.",
    ),
)
