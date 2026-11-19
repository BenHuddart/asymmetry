"""PSI LTF identity without an invented acquisition map or geometry."""

from asymmetry.core.instruments.schema import (
    Defaults,
    Detectors,
    Evidence,
    EvidenceStatus,
    Identity,
    InstrumentProfile,
    Recognition,
    Reference,
    Review,
    Source,
    Sources,
)

PROFILE = InstrumentProfile(
    identity=Identity("psi.ltf", "PSI", "LTF", "1", "LTF", "LTF", "piM3.3"),
    source=Source(
        "continuous",
        "Legacy LTF identity; FLAME replacing the instrument does not make their hardware interchangeable.",
    ),
    recognition=Recognition(("LTF",), ("ltf",)),
    detectors=Detectors((), (), ()),
    presets=(),
    quirks=(
        "Legacy LTF identity; FLAME replacing the instrument does not make their hardware interchangeable.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (
            Reference(
                "instrument-page", "PSI LTF context", "https://www.psi.ch/en/smus/flame-project"
            ),
        ),
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
                "Legacy LTF identity; FLAME replacing the instrument does not make their hardware interchangeable.",
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
