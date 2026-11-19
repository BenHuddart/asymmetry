"""PSI DOLLY identity without an invented acquisition map or geometry."""

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
    identity=Identity("psi.dolly", "PSI", "DOLLY", "1", "DOLLY", "DOLLY", "piE1"),
    source=Source(
        "continuous",
        "Legacy Dolly identity; the VMS upgrade does not identify every historical acquisition with current hardware.",
    ),
    recognition=Recognition(("DOLLY",), ("dolly",)),
    detectors=Detectors((), (), ()),
    presets=(),
    quirks=(
        "Legacy Dolly identity; the VMS upgrade does not identify every historical acquisition with current hardware.",
    ),
    defaults=Defaults(),
    sources=Sources(
        (Reference("instrument-page", "PSI DOLLY context", "https://www.psi.ch/en/smus/vms"),),
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
                "Legacy Dolly identity; the VMS upgrade does not identify every historical acquisition with current hardware.",
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
