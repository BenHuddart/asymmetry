"""Small synthetic declarations, not additional public instrument definitions."""

from asymmetry.core.instruments.geometry import (
    Circular,
    EndOn,
    Explicit,
    Grid,
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
    Review,
    Source,
    Sources,
)


def synthetic_profile(acquisitions=(), *, elements=()):
    presets = tuple(
        ConfigurationPresets(
            config.key,
            (
                Preset(
                    "Pair",
                    (
                        ChannelGroup(1, "Positive", (config.channels[0].key,)),
                        ChannelGroup(
                            2, "Negative", tuple(channel.key for channel in config.channels[1:])
                        ),
                    ),
                    1,
                    2,
                ),
            ),
        )
        for config in acquisitions
    )
    facts = (
        "identity",
        "source",
        "recognition",
        "physical.banks",
        "physical.elements",
        "quirks",
        "defaults",
        *(f"acquisition.{config.key}" for config in acquisitions),
        *(f"view.{config.key}.{view.key}" for config in acquisitions for view in config.views),
        *(f"readout_set.{c.key}.{r.key}" for c in acquisitions for r in c.readout_sets),
        *(
            f"combined_readout.{c.key}.{r.channel}"
            for c in acquisitions
            for r in c.combined_readouts
        ),
        *(
            f"conditional_contribution.{c.key}.{r.channel}"
            for c in acquisitions
            for r in c.conditional_contributions
        ),
        *(f"preset.{config.key}.Pair" for config in acquisitions),
    )
    return InstrumentProfile(
        Identity("test.fixture", "Test", "Fixture", "1", "Fixture", "Fixture"),
        Source("unknown", "Synthetic fixture; source is not inferred."),
        Recognition(("Fixture",)),
        Detectors(
            (PhysicalBank("assembly", "Assembly", "Synthetic assembly"),) if elements else (),
            elements,
            tuple(acquisitions),
        ),
        presets,
        (),
        Defaults(),
        Sources(
            (),
            tuple(
                Evidence(
                    fact,
                    EvidenceStatus.UNVERIFIED,
                    "Synthetic schema example; not a facility definition or measurement.",
                )
                for fact in facts
            ),
        ),
        Review("1", "Synthetic construction coverage"),
    )


def synthetic_profiles():
    channels = tuple(
        Channel(key, index + 1, (f"h{index + 1}",))
        for index, key in enumerate(("a", "b", "c", "d"))
    )
    hardware = tuple(PhysicalElement(f"h{index}", f"Hardware {index}") for index in range(1, 5))
    occurrences = tuple(Occurrence(channel=key) for key in ("a", "b", "c", "d"))
    a, b, c, d = occurrences
    result = {}
    arrangements = (
        (
            "concentric",
            "radial",
            Circular(
                ((a, b), (c, d)),
                (Wedge(0.1, 0.4, 45), Wedge(0.4, 0.7, 45)),
                (0.0, 0.0),
                90.0,
                -90.0,
            ),
        ),
        ("grid", "plan", Grid(((b, a), (d, c)), Rectangle(0.8, 0.4), (-1.0, 2.0), (2.0, -3.0))),
        (
            "ring_rectangles",
            "radial",
            Circular(
                ((a,), (b,), (c,), (d,)), (Rectangle(0.2, 0.6, 90.0),), (0.8,), 0.0, 90.0, True
            ),
        ),
    )
    for key, mode, arrangement in arrangements:
        view = View(
            "view", key, mode, "Synthetic rightward x, upward y; schematic dimensions.", arrangement
        )
        result[key] = synthetic_profile(
            (Acquisition("test", key, channels, (view,)),), elements=hardware
        )
    explicit = View(
        "view",
        "Explicit",
        "plan",
        "Synthetic Cartesian schematic",
        Explicit(
            (
                Placement(a, Rectangle(0.8, 0.4), -1.0, 1.0, 30.0),
                Placement(b, Rectangle(0.6, 0.5), 2.0, -1.0, -45.0),
                Placement(Occurrence(hardware="h1", editable=False), EndOn(radius=0.2), 0.0, 2.0),
            )
        ),
    )
    result["explicit"] = synthetic_profile(
        (Acquisition("test", "Explicit", channels[:2], (explicit,)),), elements=hardware[:2]
    )
    top = View(
        "top",
        "Top",
        "plan",
        "Synthetic top view",
        Explicit(
            (
                Placement(a, Rectangle(0.8, 0.4), -1.0, 0.0),
                Placement(Occurrence(channel="b", editable=False), EndOn(), 1.0, 0.0),
                Placement(Occurrence(hardware="h1", editable=False), Rectangle(0.5, 0.5), 0.0, 2.0),
            )
        ),
    )
    side = View(
        "side",
        "Side",
        "plan",
        "Synthetic side view",
        Explicit(
            (
                Placement(b, Rectangle(0.8, 0.4), 1.0, 0.0),
                Placement(Occurrence(channel="a", editable=False), EndOn(toward=False), -1.0, 0.0),
            )
        ),
    )
    result["multiview"] = synthetic_profile(
        (Acquisition("test", "Multiple views", channels[:2], (top, side)),), elements=hardware[:2]
    )
    keys = (
        ("F", "B", "U", "D", "R", "L"),
        ("F", "B", "UB", "UF", "DB", "DF", "RB", "RF", "LB", "LF", "M"),
        ("F", "B", "U", "D", "R", "L", "UB", "UF", "DB", "DF", "RB", "RF", "LB", "LF", "M"),
    )
    physical = tuple(
        PhysicalElement(key, key)
        for key in ("Fouter", "Fcenter", "B", "UB", "UF", "DB", "DF", "RB", "RF", "LB", "LF", "M")
    )
    acquisitions = []
    for order in keys:
        configured = tuple(
            Channel(
                key,
                index + 1,
                (key + "B", key + "F")
                if key in {"U", "D", "R", "L"}
                else (key,)
                if key not in {"F", "M"}
                else (),
                relationship="Descriptive hardware association only; F/M affiliation is unknown. No count equation or independence is asserted.",
            )
            for index, key in enumerate(order)
        )
        view = View(
            "readout",
            "Synthetic readout",
            "plan",
            "Synthetic channel order, not a physical layout",
            Grid((tuple(Occurrence(channel=key) for key in order),), Rectangle(0.8, 0.6)),
        )
        acquisitions.append(
            Acquisition(
                f"channels{len(order)}",
                f"{len(order)} channel illustration",
                configured,
                (view,),
                layout_key=f"Fixture-{len(order)}",
            )
        )
    result["combined_split"] = synthetic_profile(tuple(acquisitions), elements=physical)
    return result
