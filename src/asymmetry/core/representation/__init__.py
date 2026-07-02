"""First-class analysis representations (Domain → Representation model).

Each dataset owns a small set of representations — F-B asymmetry, individual
groups, and the MaxEnt time-domain reconstruction (time domain), plus FFT and
MaxEnt spectra (frequency domain).  Representations carry
a recipe, one stored fit, and trend state; computed arrays are transient.
Fit series own ordered member collections that drive batch/global fits and
trending.
"""

from __future__ import annotations

from asymmetry.core.representation.base import (
    DOMAIN_OF,
    FIT_PROVENANCE,
    FitSlot,
    Representation,
    RepresentationType,
)
from asymmetry.core.representation.container import DatasetRepresentations
from asymmetry.core.representation.factory import (
    REPRESENTATION_REGISTRY,
    make_representation,
    representation_from_dict,
)
from asymmetry.core.representation.frequency import FrequencyFFT, FrequencyMaxEnt
from asymmetry.core.representation.group import DataGroup
from asymmetry.core.representation.naming import (
    composite_model_label,
    default_series_label,
    member_range,
)
from asymmetry.core.representation.series import (
    MEMBER_KINDS,
    ORDER_KEYS,
    PARAM_ROLES,
    FitSeries,
    canonical_model_matches,
)
from asymmetry.core.representation.time import (
    TimeFBAsymmetry,
    TimeGroups,
    TimeMaxEntReconstruction,
    build_maxent_reconstruction_datasets,
)
from asymmetry.core.representation.trend_state import TrendState

__all__ = [
    "DOMAIN_OF",
    "FIT_PROVENANCE",
    "MEMBER_KINDS",
    "ORDER_KEYS",
    "PARAM_ROLES",
    "REPRESENTATION_REGISTRY",
    "DataGroup",
    "DatasetRepresentations",
    "FitSeries",
    "FitSlot",
    "FrequencyFFT",
    "FrequencyMaxEnt",
    "Representation",
    "RepresentationType",
    "TrendState",
    "TimeFBAsymmetry",
    "TimeGroups",
    "TimeMaxEntReconstruction",
    "build_maxent_reconstruction_datasets",
    "canonical_model_matches",
    "composite_model_label",
    "default_series_label",
    "make_representation",
    "member_range",
    "representation_from_dict",
]
