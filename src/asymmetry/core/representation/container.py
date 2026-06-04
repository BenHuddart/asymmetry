"""Per-dataset owner of up to four representations (one per type)."""

from __future__ import annotations

from asymmetry.core.representation.base import Representation, RepresentationType
from asymmetry.core.representation.factory import make_representation, representation_from_dict


class DatasetRepresentations:
    """The up-to-four representations belonging to a single run.

    Representations are created lazily: :meth:`ensure` builds an empty
    representation of a given type on first access.
    """

    def __init__(
        self,
        run_number: int,
        by_type: dict[RepresentationType, Representation] | None = None,
    ) -> None:
        self.run_number = int(run_number)
        self.by_type: dict[RepresentationType, Representation] = dict(by_type or {})

    def get(self, rep_type: RepresentationType) -> Representation | None:
        """Return the representation of *rep_type*, or ``None`` if absent."""
        return self.by_type.get(rep_type)

    def ensure(self, rep_type: RepresentationType) -> Representation:
        """Return the representation of *rep_type*, creating an empty one if needed."""
        existing = self.by_type.get(rep_type)
        if existing is None:
            existing = make_representation(rep_type)
            self.by_type[rep_type] = existing
        return existing

    def __contains__(self, rep_type: object) -> bool:
        return rep_type in self.by_type

    def __iter__(self):
        return iter(self.by_type.values())

    def to_dict(self) -> dict:
        """Serialise as ``{run_number, representations: {type_value: rep_dict}}``."""
        return {
            "run_number": self.run_number,
            "representations": {
                rep_type.value: rep.to_dict() for rep_type, rep in self.by_type.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> DatasetRepresentations:
        """Reconstruct from :meth:`to_dict` output."""
        run_number = int(data.get("run_number", 0))
        by_type: dict[RepresentationType, Representation] = {}
        raw = data.get("representations")
        if isinstance(raw, dict):
            for key, rep_data in raw.items():
                if not isinstance(rep_data, dict):
                    continue
                payload = dict(rep_data)
                payload.setdefault("rep_type", key)
                rep = representation_from_dict(payload)
                by_type[rep.rep_type] = rep
        return cls(run_number, by_type)
