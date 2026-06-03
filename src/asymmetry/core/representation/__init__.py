"""First-class analysis representations (Domain → Representation model).

Each dataset owns up to four representations — F-B asymmetry and individual
groups (time domain), FFT and MaxEnt (frequency domain).  Representations carry
a recipe, one stored fit, and trend state; computed arrays are transient.
Batches own ordered series that drive batch/global fits and trending.
"""

from __future__ import annotations

from asymmetry.core.representation.base import (
    DOMAIN_OF,
    FIT_PROVENANCE,
    FitSlot,
    Representation,
    RepresentationType,
)
from asymmetry.core.representation.batch import (
    ORDER_KEYS,
    PARAM_ROLES,
    Batch,
    canonical_model_matches,
)
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.factory import (
    REPRESENTATION_REGISTRY,
    make_representation,
    representation_from_dict,
)
from asymmetry.core.representation.frequency import FrequencyFFT, FrequencyMaxEnt
from asymmetry.core.representation.time import TimeFBAsymmetry, TimeGroups

__all__ = [
    "DOMAIN_OF",
    "FIT_PROVENANCE",
    "ORDER_KEYS",
    "PARAM_ROLES",
    "REPRESENTATION_REGISTRY",
    "Batch",
    "DatasetRepresentations",
    "FitSlot",
    "FrequencyFFT",
    "FrequencyMaxEnt",
    "Representation",
    "RepresentationType",
    "TimeFBAsymmetry",
    "TimeGroups",
    "canonical_model_matches",
    "make_representation",
    "representation_from_dict",
]
