"""PSI HAL: explicit acquired channels and qualified schematic conventions."""

import math
from dataclasses import replace

from asymmetry.core.instruments.geometry import (
    BeamDirection,
    Circular,
    Occurrence,
    Rectangle,
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
            "Viewed along the beam axis; clockwise channel numbering from the top is an unverified export convention. The octagon is a dimensionless schematic. Axial viewing sense is unknown; MV has no declared physical position.",
            Circular(
                tuple((Occurrence(channel=f"{prefix}{index}"),) for index in range(1, 9)),
                (Rectangle(0.54, 0.26, -90.0),),
                (math.cos(math.radians(22.5)) - 0.13,),
                90.0,
                -45.0,
                radial_rotation=True,
            ),
            orientation=ViewOrientation((1, 0, 0), (0, 1, 0)),
        )
        for bank, prefix in (("Forward", "F"), ("Backward", "B"))
    ),
    beam=BeamDirection((0, 0, 1)),
    auxiliary=(Occurrence(channel="MV"),),
)

FORWARD_RING = replace(
    ACQUISITION,
    key="forward_ring",
    label="Nine-channel forward-ring readout",
    channels=ACQUISITION.channels[:9],
    views=ACQUISITION.views[:1],
    layout_key="HAL-FORWARD",
    require_file_labels=True,
)

TRANSVERSE = Preset(
    "Transverse (opposed pairs)",
    tuple(ChannelGroup(i, f"F{i}", (f"F{i}",)) for i in range(1, 9)),
    1,
    5,
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
    TRANSVERSE,
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.hal9500", "PSI", "HAL-9500", "2", "HAL", "HAL", "piE3"),
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
        (ACQUISITION, FORWARD_RING),
    ),
    presets=(
        ConfigurationPresets(ACQUISITION.key, PRESETS),
        ConfigurationPresets(FORWARD_RING.key, (TRANSVERSE,)),
    ),
    quirks=(
        "PSI-scoped HIFI/tdc_hifi aliases identify HAL-9500, not ISIS HiFi. MV is an acquired channel with unverified hardware affiliation. The nine-channel map requires matching file labels in histogram order.",
    ),
    defaults=Defaults(apply_default_preset_on_load=True),
    sources=Sources(
        (
            Reference(
                "instrument-page", "PSI HAL-9500 instrument", "https://www.psi.ch/en/smus/hal-9500"
            ),
            Reference(
                "rigi-school-data",
                "2026 ISIS/PSI Muon School at Rigi Kaltbad data",
                "https://shadow.nd.rl.ac.uk/wimda/",
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
                "Fresh files use Per-octant for the seventeen-channel readout and Transverse (opposed pairs) for the matched forward-ring readout.",
            ),
            Evidence(
                "acquisition.full17",
                EvidenceStatus.UNVERIFIED,
                "Preserved ordinal/label map and explicit groups; no new hardware or statistical relationship is inferred.",
            ),
            Evidence(
                "acquisition.forward_ring",
                EvidenceStatus.SOURCE,
                "The MDU headers export MV followed by F1 through F8 in histogram order.",
                "rigi-school-data",
                "Wimda Workshop: HAL-9500 runs 686 and 693",
            ),
            *(
                Evidence(
                    f"view.{config.key}.{view.key}",
                    EvidenceStatus.SCHEMATIC,
                    "Dimensionless schematic; coordinates, rotations and any direction arrows are drawing conventions, not measured detector dimensions or run settings.",
                )
                for config in (ACQUISITION, FORWARD_RING)
                for view in config.views
            ),
            *(
                Evidence(
                    f"preset.full17.{preset.name}",
                    EvidenceStatus.AUTHORED,
                    "Compatibility grouping and ordered asymmetry legs; initial balances require experiment-specific calibration.",
                )
                for preset in PRESETS
            ),
            Evidence(
                f"preset.{FORWARD_RING.key}.{TRANSVERSE.name}",
                EvidenceStatus.AUTHORED,
                "F1 and F5 are the ordered analysis legs. The pairing and angular convention are unverified pending instrument review.",
            ),
        ),
    ),
    review=Review(
        "2",
        "File headers establish the forward-ring export order. The authored F1/F5 pairing and angular convention await instrument review.",
    ),
)
