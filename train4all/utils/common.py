from datetime import datetime


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_run_id(
    timestamp: str | None = None,
    name: str | None = None,
    debug: bool = False
) -> str:
    parts = [timestamp or get_timestamp()]
    if name:
        parts.append(name)
    if debug:
        parts.append("debug")

    return "__".join(parts)
