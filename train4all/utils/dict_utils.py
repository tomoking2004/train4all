from typing import Any, Dict, Iterable, List, Set, TypeVar
from dataclasses import is_dataclass, fields

K = TypeVar("K")
V = TypeVar("V")


def exclude_none(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def replace_dict_keys(obj: Any, name_map: Dict[str, str]) -> Any:
    """
    Recursively replace keys in a dictionary according to name_map.

    Works for nested dictionaries, including state_dicts.

    Args:
        obj: Original dictionary or any object.
        name_map: Dictionary mapping old substrings to new substrings.

    Returns:
        New object with keys replaced where applicable.
    """
    if not name_map:
        return obj

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_k = k
            for old, new in name_map.items():
                new_k = new_k.replace(old, new)
            new_obj[new_k] = replace_dict_keys(v, name_map)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(replace_dict_keys(v, name_map) for v in obj)
    else:
        return obj


def deep_update(base: Dict[str, Any], updates: Dict[str, Any], validate: bool = False) -> Dict[str, Any]:
    if validate:
        invalid_keys = set(updates) - set(base)
        if invalid_keys:
            raise ValueError(f"Invalid keys in updates: {invalid_keys}")

    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value, validate=validate)
        else:
            base[key] = value
    return base


def select_keys(data: Dict[K, V], keys: Iterable[K]) -> Dict[K, V]:
    """Return a new dictionary containing only the specified keys."""
    return {key: data[key] for key in keys if key in data}


def dataclass_to_dict(obj: Any, exclude_keys: List[str] | Set[str] | None = None) -> Dict[str, Any]:
    """
    Recursively convert a dataclass (possibly nested) to a dict,
    excluding specific keys at any nesting level.
    """
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)}")

    exclude_keys = set(exclude_keys or [])

    result = {}
    for f in fields(obj):
        if f.name in exclude_keys:
            continue

        value = getattr(obj, f.name)
        if is_dataclass(value):
            result[f.name] = dataclass_to_dict(value, exclude_keys)
        elif isinstance(value, (list, tuple)):
            # Handle list/tuple of dataclasses
            result[f.name] = [
                dataclass_to_dict(v, exclude_keys) if is_dataclass(v) else v
                for v in value
            ]
        elif isinstance(value, dict):
            # Handle dict values that may be dataclasses
            result[f.name] = {
                k: dataclass_to_dict(v, exclude_keys) if is_dataclass(v) else v
                for k, v in value.items()
            }
        else:
            result[f.name] = value

    return result
