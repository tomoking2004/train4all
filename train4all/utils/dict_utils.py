from collections.abc import Mapping

__all__ = ["MetricTable", "replace_dict_keys"]

type MetricTable = dict[str, dict[str, list[float]]]
"""Mapping of ``metric_name → phase_name → list[value]`` recorded over training."""


def replace_dict_keys(
    obj: object,
    name_map: Mapping[str, str],
) -> object:
    """
    Recursively replace substrings in dictionary keys.

    Nested mappings, lists, and tuples are traversed; all other objects are
    returned unchanged.

    Args:
        obj: Input object (mapping, list, tuple, or other).
        name_map: Mapping of old substring to new substring.

    Returns:
        New object with transformed keys.
    """
    if not name_map:
        return obj

    if isinstance(obj, Mapping):
        result: dict[object, object] = {}
        for key, value in obj.items():
            new_key = key
            if isinstance(key, str):
                for old, new in name_map.items():
                    new_key = new_key.replace(old, new)
            result[new_key] = replace_dict_keys(value, name_map)
        return result

    # Lists and tuples by name, not `Sequence`: the rebuild calls `type(obj)(items)`,
    # which only some sequences accept — `range` is a Sequence and `range(items)` is a
    # TypeError. Anything else falls through to being returned unchanged, which is
    # what the docstring already promises.
    if isinstance(obj, (list, tuple)):
        return type(obj)(replace_dict_keys(item, name_map) for item in obj)

    return obj
