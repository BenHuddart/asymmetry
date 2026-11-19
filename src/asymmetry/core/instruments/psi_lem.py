"""PSI LEM identity without an invented acquisition map or geometry."""

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
    identity=Identity("psi.lem", "PSI", "LEM", "1", "LEM", "LEM", "muE4"),
    source=Source(
        "continuous",
        "Low-energy muon instrument; source family does not specify run preparation or timing.",
    ),
    recognition=Recognition(("LEM",), ("lem",)),
    detectors=Detectors((), (), ()),
    presets=(),
    quirks=(
        "Low-energy muon instrument; source family does not specify run preparation or timing.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (Reference("instrument-page", "PSI LEM context", "https://www.psi.ch/de/smus/lem"),),
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
                "Low-energy muon instrument; source family does not specify run preparation or timing.",
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
