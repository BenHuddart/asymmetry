"""PSI GPD identity without an invented acquisition map or geometry."""

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
    identity=Identity("psi.gpd", "PSI", "GPD", "1", "GPD", "GPD", "muE1"),
    source=Source(
        "continuous",
        "Decay-channel instrument accepting positive or negative muons; no universal surface-muon charge assumption.",
    ),
    recognition=Recognition(("GPD",), ("gpd",)),
    detectors=Detectors((), (), ()),
    presets=(),
    quirks=(
        "Decay-channel instrument accepting positive or negative muons; no universal surface-muon charge assumption.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (Reference("instrument-page", "PSI GPD context", "https://www.psi.ch/en/smus/gpd"),),
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
                "Decay-channel instrument accepting positive or negative muons; no universal surface-muon charge assumption.",
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
