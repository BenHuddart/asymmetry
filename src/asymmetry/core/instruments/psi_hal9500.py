"""PSI HAL: explicit acquired channels and qualified schematic conventions."""

import math

from asymmetry.core.instruments.geometry import (
    Explicit,
    Occurrence,
    Placement,
    Rectangle,
    View,
    Wedge,
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
    PhysicalElement,
    Preset,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

CHANNELS = (
    ("MV", "MV"),
    ("F1", "F1"),
    ("F2", "F2"),
    ("F3", "F3"),
    ("F4", "F4"),
    ("F5", "F5"),
    ("F6", "F6"),
    ("F7", "F7"),
    ("F8", "F8"),
    ("B1", "B1"),
    ("B2", "B2"),
    ("B3", "B3"),
    ("B4", "B4"),
    ("B5", "B5"),
    ("B6", "B6"),
    ("B7", "B7"),
    ("B8", "B8"),
)

ACQUISITION = Acquisition(
    "full17",
    "Seventeen-channel readout",
    tuple(
        Channel(key, index + 1, label=label, file_labels=(label,))
        for index, (key, label) in enumerate(CHANNELS)
    ),
    tuple(
        View(
            bank.lower(),
            bank,
            "radial",
            "Viewed along the beam axis; clockwise channel numbering from the top is an unverified export convention. The octagon is a dimensionless schematic.",
            Explicit(
                (
                    *(
                        (Placement(Occurrence(channel="MV"), Wedge(0.0, 0.26, 180.0), angle=90.0),)
                        if prefix == "F"
                        else ()
                    ),
                    *(
                        Placement(
                            Occurrence(channel=f"{prefix}{index + 1}"),
                            Rectangle(0.54, 0.26, -90.0),
                            (math.cos(math.radians(22.5)) - 0.13) * math.cos(math.radians(angle)),
                            (math.cos(math.radians(22.5)) - 0.13) * math.sin(math.radians(angle)),
                            angle=angle,
                            sector=index,
                        )
                        for index in range(8)
                        for angle in ((90.0 - index * 45.0) % 360.0,)
                    ),
                )
            ),
        )
        for bank, prefix in (("Forward", "F"), ("Backward", "B"))
    ),
)

PRESETS = (
    Preset(
        "Per-octant",
        (
            ChannelGroup(1, "Octant 1", ("F1", "B1")),
            ChannelGroup(2, "Octant 2", ("F2", "B2")),
            ChannelGroup(3, "Octant 3", ("F3", "B3")),
            ChannelGroup(4, "Octant 4", ("F4", "B4")),
            ChannelGroup(5, "Octant 5", ("F5", "B5")),
            ChannelGroup(6, "Octant 6", ("F6", "B6")),
            ChannelGroup(7, "Octant 7", ("F7", "B7")),
            ChannelGroup(8, "Octant 8", ("F8", "B8")),
        ),
        1,
        5,
    ),
    Preset(
        "Longitudinal",
        (
            ChannelGroup(1, "Forward", ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")),
            ChannelGroup(2, "Backward", ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")),
        ),
        2,
        1,
    ),
    Preset(
        "Transverse (opposed pairs)",
        (
            ChannelGroup(1, "F1", ("F1",)),
            ChannelGroup(2, "F2", ("F2",)),
            ChannelGroup(3, "F3", ("F3",)),
            ChannelGroup(4, "F4", ("F4",)),
            ChannelGroup(5, "F5", ("F5",)),
            ChannelGroup(6, "F6", ("F6",)),
            ChannelGroup(7, "F7", ("F7",)),
            ChannelGroup(8, "F8", ("F8",)),
        ),
        1,
        5,
    ),
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.hal9500", "PSI", "HAL-9500", "1", "HAL", "HAL", "piE3"),
    source=Source(
        "continuous", "Continuous muon source; spin rotation, field and timing remain run-specific."
    ),
    recognition=Recognition(("HIFI", "HAL9500"), ("hifi", "hal9500")),
    detectors=Detectors(
        (
            PhysicalBank("forward", "Forward", "Ring of eight positron detectors."),
            PhysicalBank("backward", "Backward", "Ring of eight positron detectors."),
        ),
        tuple(
            PhysicalElement(f"{prefix}{index}", f"{prefix}{index}", bank)
            for prefix, bank in (("F", "forward"), ("B", "backward"))
            for index in range(1, 9)
        ),
        (ACQUISITION,),
    ),
    presets=(ConfigurationPresets("full17", PRESETS),),
    quirks=(
        "PSI-scoped HIFI/tdc_hifi aliases identify HAL-9500, not ISIS HiFi. MV is an acquired channel with unverified hardware affiliation. Partial forward-only files remain supported by their loader.",
    ),
    defaults=Defaults(apply_default_preset_on_load=True),
    sources=Sources(
        (
            Reference(
                "instrument-page", "PSI HAL-9500 instrument", "https://www.psi.ch/en/smus/hal-9500"
            ),
        ),
        (
            Evidence(
                "identity",
                EvidenceStatus.SOURCE,
                "HAL-9500 on piE3.",
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
                "Two rings of eight positron detectors; drawn views are not physical dimensions.",
                "instrument-page",
                "Detector description",
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.UNVERIFIED,
                "F/B element keys describe the rings, but exact hardware-to-channel and angular numbering need review.",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "Preserved file conventions; schematic arrows and MV labels do not certify run-specific instrumentation.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.AUTHORED,
                "Per-octant remains the first preset and the existing fresh-load flag is retained; no loader path is newly enabled.",
            ),
            Evidence(
                "acquisition.full17",
                EvidenceStatus.UNVERIFIED,
                "Preserved ordinal/label map and explicit groups; no new hardware or statistical relationship is inferred.",
            ),
            *(
                Evidence(
                    f"view.full17.{view.key}",
                    EvidenceStatus.SCHEMATIC,
                    "Dimensionless schematic; coordinates, rotations and any direction arrows are drawing conventions, not measured detector dimensions or run settings.",
                )
                for view in ACQUISITION.views
            ),
            *(
                Evidence(
                    f"preset.full17.{preset.name}",
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
