from collections.abc import Mapping, Sequence
from typing import Any


def exclude_none(
    obj: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return a new dictionary excluding keys whose value is None.
    """
    return {k: v for k, v in obj.items() if v is not None}


def replace_dict_keys(
    obj: Any,
    name_map: Mapping[str, str],
) -> Any:
    """
    Recursively replace substrings in dictionary keys.

    Supports nested mappings, lists, and tuples.
    Other objects are returned unchanged.

    Args:
        obj: Input object (mapping, sequence, or other).
        name_map: Mapping of substring replacements (old → new).

    Returns:
        New object with transformed keys.
    """
    if not name_map:
        return obj

    # Mapping case
    if isinstance(obj, Mapping):
        new_obj: dict[Any, Any] = {}

        for key, value in obj.items():
            new_key = key

            if isinstance(key, str):
                for old, new in name_map.items():
                    new_key = new_key.replace(old, new)

            new_obj[new_key] = replace_dict_keys(value, name_map)

        return new_obj

    # Sequence case (exclude str/bytes)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(
            replace_dict_keys(item, name_map)
            for item in obj
        )

    return obj
