import contextlib
import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Literal, Protocol, runtime_checkable

with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")


type LogLevel = Literal["info", "debug", "warn"]
type Printer = Callable[[str], object]

_SEPARATOR_PAD = 48  # separator rule width = key_width + this pad


def _format_value(value: object, float_fmt: int) -> str:
    """Render a leaf value as text.

    Floats use ``float_fmt`` decimal places, falling back to scientific notation
    when fixed-point would round a nonzero value down to all zeros (e.g. ``1e-5``
    at 4 dp). Redundant zeros are stripped from the mantissa and exponent, so
    ``1.0000e-05`` renders as ``1e-5``. Every other type is rendered with ``str``.
    """
    if not isinstance(value, float):
        return str(value)
    fixed = f"{value:.{float_fmt}f}"
    if value and float(fixed) == 0:
        mantissa, exponent = f"{value:.{float_fmt}e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return fixed


def _render_tree(
    tree: Mapping[str, object],
    key_width: int,
    float_fmt: int,
    max_depth: int | None,
    depth: int = 0,
    prefix: str = "",
) -> list[str]:
    """Render a mapping as ``├─``/``└─`` tree lines, recursing into nested
    mappings up to ``max_depth`` (``None`` is unlimited)."""
    lines: list[str] = []
    leaf_width = max(key_width - len(prefix), 0)
    items = list(tree.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└─" if is_last else "├─"
        if isinstance(value, Mapping) and (max_depth is None or depth < max_depth):
            lines.append(f" {prefix}{connector} {key}")
            child_prefix = prefix + ("   " if is_last else "│  ")
            lines += _render_tree(value, key_width, float_fmt, max_depth, depth + 1, child_prefix)
        else:
            val = _format_value(value, float_fmt)
            lines.append(f" {prefix}{connector} {key:<{leaf_width}}: {val}")
    return lines


def print_dict_tree(
    tree: Mapping[str, object],
    *,
    max_depth: int | None = None,
    header: str | None = None,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Printer | None = None,
) -> None:
    """Pretty-print a (possibly nested) mapping as a tree.

    Args:
        tree: Mapping to display; nested mappings are expanded recursively.
        max_depth: Deepest nesting level to expand. ``0`` keeps the output flat;
            ``None`` expands without limit.
        header: Title printed above the tree. Omit to skip header and separator.
        key_width: Column width for leaf keys.
        float_fmt: Decimal places used when formatting float values.
        trailing_newline: Print an empty line after the tree.
        print_fn: Output function. Defaults to the built-in ``print``.
    """
    lines: list[str] = []
    if header is not None:
        lines.append(header)
        lines.append(f" {'─' * (key_width + _SEPARATOR_PAD)}")
    lines += _render_tree(tree, key_width, float_fmt, max_depth)
    if trailing_newline:
        lines.append("")

    printer = print_fn or print
    for line in lines:
        printer(line)


@runtime_checkable
class TrainerLogger(Protocol):
    """Minimal logging interface the trainer depends on.

    Any object implementing ``log`` with this signature can be injected in
    place of the default :class:`UnifiedLogger`, letting callers plug in their
    own logging backend.
    """

    def log(self, msg: str | None = None, level: LogLevel = "info", *, indent: int = 0) -> None:
        """Emit *msg* at *level*, optionally indented by *indent* spaces."""
        ...


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

    _LEVEL_MAP: ClassVar[dict[LogLevel, tuple[int, str]]] = {
        "debug": (logging.DEBUG,   "[DEBUG] "),
        "info":  (logging.INFO,    ""),
        "warn":  (logging.WARNING, "⚠️  "),
    }

    def __init__(
        self,
        name: str,
        log_path: Path | str | None = None,
        *,
        verbose: bool = True,
        debug_mode: bool = False,
        file_mode: Literal["a", "w"] = "a",
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

        for line in (msg or "").split("\n"):
            if self.verbose:
                print(f"{' ' * indent}{prefix}{line}" if line.strip() else "")
            if self.log_path:
                self.logger.log(log_level, line)
