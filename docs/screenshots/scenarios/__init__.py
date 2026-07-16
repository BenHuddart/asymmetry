"""Screenshot scenarios for the Asymmetry GUI documentation.

Each module defines a single :class:`Scenario` subclass and registers it via
:func:`register`. New scenarios should be imported from
:mod:`docs.screenshots.capture` so the CLI sees them.
"""

from ._base import Scenario, register, registered_scenarios  # noqa: F401
