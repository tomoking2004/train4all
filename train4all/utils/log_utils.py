import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, TypeAlias

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


LogLevel: TypeAlias = Literal["info", "debug", "warn"]

_SEPARATOR_PAD = 48  # separator rule width = key_width + this pad


def _render_tree(
    tree: Mapping[str, object],
    key_width: int,
    float_fmt: int,
    max_depth: int | None,
    depth: int,
    prefix: str,
) -> list[str]:
    """Recursively render a mapping as ├─/└─ tree lines."""
    items = list(tree.items())
    lines: list[str] = []
    effective_key_width = max(key_width - len(prefix), 0)
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        conn = "└─" if is_last else "├─"
        child_prefix = prefix + ("   " if is_last else "│  ")
        can_expand = isinstance(value, Mapping) and (max_depth is None or depth < max_depth)
        if can_expand:
            lines.append(f" {prefix}{conn} {key}")
            lines.extend(
                _render_tree(value, key_width, float_fmt, max_depth, depth + 1, child_prefix)
            )
        else:
            val = f"{value:.{float_fmt}f}" if isinstance(value, float) else str(value)
            lines.append(f" {prefix}{conn} {key:<{effective_key_width}}: {val}")
    return lines


def _print_header(
    printer: Callable[[str], object],
    header: str | None,
    key_width: int,
) -> None:
    if header is not None:
        printer(header)
        printer(f" {'─' * (key_width + _SEPARATOR_PAD)}")


def print_flat_dict_tree(
    data: Mapping[str, object],
    header: str | None = None,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Callable[[str], object] | None = None,
) -> None:
    """
    Pretty-print a flat (single-level) mapping in a tree-like format.

    Args:
        data: Flat mapping to display.
        header: Title printed above the tree. Omit to skip header and separator.
        key_width: Column width for keys.
        float_fmt: Decimal places used when formatting float values.
        trailing_newline: Print an empty line after the tree.
        print_fn: Output function. Defaults to built-in ``print``.
    """
    printer: Callable[[str], object] = print_fn or print
    _print_header(printer, header, key_width)
    for line in _render_tree(data, key_width, float_fmt, 0, 0, ""):
        printer(line)
    if trailing_newline:
        printer("")


def print_dict_tree(
    tree: Mapping[str, object],
    max_depth: int | None = None,
    header: str | None = None,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Callable[[str], object] | None = None,
) -> None:
    """
    Pretty-print a nested mapping in a recursive tree-like format.

    Args:
        tree: Mapping to display (may contain nested mappings).
        max_depth: Maximum nesting depth to expand. ``None`` is unlimited.
        header: Title printed at the root level. Omit to skip header and separator.
        key_width: Column width for leaf keys.
        float_fmt: Decimal places used when formatting float values.
        trailing_newline: Print an empty line after the tree.
        print_fn: Output function. Defaults to built-in ``print``.
    """
    printer: Callable[[str], object] = print_fn or print
    _print_header(printer, header, key_width)
    for line in _render_tree(tree, key_width, float_fmt, max_depth, 0, ""):
        printer(line)
    if trailing_newline:
        printer("")


class UnifiedLogger:
    """
    Logger that writes to the console and optionally to a file.

    Args:
        name: Unique name for the underlying ``logging.Logger`` instance.
        log_path: Output file path. File logging is disabled when ``None``.
        verbose: Echo log messages to stdout.
        debug_mode: Enable debug-level output; ``"debug"``-level calls are
            silently ignored when ``False``.
        file_mode: File open mode — ``"a"`` to append, ``"w"`` to overwrite.
    """

    _LEVEL_MAP: dict[LogLevel, tuple[int, str]] = {
        "debug": (logging.DEBUG, "[DEBUG] "),
        "info":  (logging.INFO,  ""),
        "warn":  (logging.WARNING, "⚠️  "),
    }

    def __init__(
        self,
        name: str,
        log_path: Path | str | None = None,
        *,
        verbose: bool = True,
        debug_mode: bool = False,
        file_mode: str = "a",
    ) -> None:
        self.name = name
        self.log_path = Path(log_path) if log_path else None
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.file_mode = file_mode
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.log_path, mode=self.file_mode, encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(handler)

        return logger

    def log(
        self,
        msg: str | None = None,
        level: LogLevel = "info",
        *,
        indent: int = 0,
    ) -> None:
        """
        Emit a message to the console and/or log file.

        Args:
            msg: Message text. ``None`` emits an empty line.
            level: Severity level — ``"info"``, ``"debug"``, or ``"warn"``.
            indent: Number of leading spaces added to non-empty lines.
        """
        log_level, prefix = self._LEVEL_MAP[level]

        if log_level == logging.DEBUG and not self.debug_mode:
            return

        text = msg or ""
        for line in text.split("\n"):
            console_msg = f"{' ' * indent}{prefix}{line}" if line.strip() else ""
            if self.verbose:
                print(console_msg)
            if self.log_path:
                self.logger.log(log_level, line)
