from collections.abc import Mapping, Sequence

type MetricTable = dict[str, dict[str, list[float]]]
"""Mapping of ``metric_name → phase → list[value]`` recorded over training."""


def replace_dict_keys(
    obj: object,
    name_map: Mapping[str, str],
) -> object:
    """
    Recursively replace substrings in dictionary keys.

    Nested mappings, lists, and tuples are traversed; all other objects are
    returned unchanged.

    Args:
        obj: Input object (mapping, sequence, or other).
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

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(replace_dict_keys(item, name_map) for item in obj)

    return obj
