"""Scientific execution compatibility, independent of storage and release versions."""

from dataclasses import fields, is_dataclass, replace

# Bump for numerical algorithms or interpretation changes; see ARCHITECTURE.md.
COMPUTATION_REVISION = "1"


def is_current(revision: str | None) -> bool:
    """Only a known matching execution can qualify for automatic current reuse."""
    return revision == COMPUTATION_REVISION


def without_revision(value):
    """Keep execution attribution out of recipe and observation identities.

    Observation hashing retains pickle aliases; unchanged containers keep their
    identity, and rewritten shared containers remain shared through the memo.
    """
    memo = {}

    def identity_value(item):
        if not isinstance(item, (dict, list, tuple)) and not is_dataclass(item):
            return item
        if id(item) in memo:
            return memo[id(item)]
        if is_dataclass(item):
            cleaned_fields = {f.name: identity_value(getattr(item, f.name)) for f in fields(item)}
            unchanged = all(cleaned_fields[f.name] is getattr(item, f.name) for f in fields(item))
            cleaned = item if unchanged else replace(item, **cleaned_fields)
        elif isinstance(item, dict):
            cleaned = {
                key: identity_value(v) for key, v in item.items() if key != "computation_revision"
            }
            unchanged = len(cleaned) == len(item) and all(cleaned[k] is v for k, v in item.items())
        else:
            cleaned = [identity_value(v) for v in item]
            unchanged = all(a is b for a, b in zip(cleaned, item, strict=True))
            if isinstance(item, tuple):
                cleaned = tuple(cleaned)
        result = item if unchanged else cleaned
        memo[id(item)] = result
        return result

    return identity_value(value)
