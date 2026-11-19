"""PSI ALC identity without an invented acquisition map or geometry."""

from asymmetry.core.instruments.schema import (
    Defaults,
    Detectors,
    Evidence,
    EvidenceStatus,
    Identity,
    InstrumentProfile,
    Recognition,
    Review,
    Source,
    Sources,
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.alc", "PSI", "ALC", "1", "ALC", "ALC", "piE3"),
    source=Source(
        "continuous",
        "Legacy ALC filename identity; exact historical hardware and beamline applicability need review.",
    ),
    recognition=Recognition(("ALC",), ("alc",)),
    detectors=Detectors((), (), ()),
    presets=(),
    quirks=(
        "Legacy ALC filename identity; exact historical hardware and beamline applicability need review.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (),
        (
            Evidence(
                "identity",
                EvidenceStatus.UNVERIFIED,
                "Preserved instrument/beamline metadata identity; exact historical export applicability is not independently established here.",
            ),
            Evidence(
                "source",
                EvidenceStatus.UNVERIFIED,
                "Existing continuous-source metadata convention; no source/timing fallback is applied by this declaration.",
            ),
            Evidence(
                "recognition",
                EvidenceStatus.UNVERIFIED,
                "Preserved PSI metadata/filename identity; generic drawing requires the actual acquired count.",
            ),
            Evidence(
                "physical.banks", EvidenceStatus.UNVERIFIED, "No physical bank map is declared."
            ),
            Evidence(
                "physical.elements",
                EvidenceStatus.UNVERIFIED,
                "No physical element map is declared.",
            ),
            Evidence(
                "quirks",
                EvidenceStatus.UNVERIFIED,
                "Legacy ALC filename identity; exact historical hardware and beamline applicability need review.",
            ),
            Evidence(
                "defaults",
                EvidenceStatus.AUTHORED,
                "No geometry or scientific grouping preset is invented.",
            ),
        ),
    ),
    review=Review("1", "Metadata-only identity."),
)
