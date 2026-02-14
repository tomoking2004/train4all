import logging
from pathlib import Path
from typing import Any, Callable, Literal
from collections.abc import Mapping


LogLevel = Literal["info", "debug", "warn"]


def print_flat_dict_tree(
    data: Mapping[str, Any],
    header: str,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Callable[[str], object] | None = None,
) -> None:
    """
    Pretty-print a flat (single-level) mapping in a tree-like format.
    """
    printer: Callable[[str], object] = print_fn or print

    printer(header)

    lines: list[str] = []

    for key, value in data.items():
        value_str = (
            f"{value:.{float_fmt}f}"
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f" ├─ {key:<{key_width}}: {value_str}")

    if lines:
        lines[-1] = lines[-1].replace("├", "└", 1)
        printer("\n".join(lines))

    if trailing_newline:
        printer("")


def print_dict_tree(
    tree: Mapping[str, Any],
    max_depth: int | None = None,
    header: str | None = None,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Callable[[str], object] | None = None,
    *,
    indent: int = 0,
) -> None:
    """
    Pretty-print nested mappings in a tree-like structure.
    """
    printer: Callable[[str], object] = print_fn or print

    if indent == 0 and header:
        printer(header)

    base_indent = " " * (indent * 2)

    for key, value in tree.items():
        is_mapping = isinstance(value, Mapping)
        can_expand = is_mapping and (
            max_depth is None or indent < max_depth
        )

        if can_expand:
            printer(f"{base_indent}  - {key}")
            print_dict_tree(
                value,
                max_depth=max_depth,
                header=None,
                key_width=key_width,
                float_fmt=float_fmt,
                trailing_newline=False,
                print_fn=printer,
                indent=indent + 1,
            )
        else:
            value_str = (
                f"{value:.{float_fmt}f}"
                if isinstance(value, float)
                else str(value)
            )

            adjusted_width = max(key_width - indent * 2, 0)

            printer(
                f"{base_indent}  - {key:<{adjusted_width}}: {value_str}"
            )

    if indent == 0 and trailing_newline:
        printer("")


class UnifiedLogger:
    """
    Unified logger for console and optional file output (Python 3.12+).

    Behavior:
    - Uses split("\\n") for line splitting (intentionally)
    - Preserves legacy newline semantics
    - Uses print() default newline behavior
    """

    _LEVEL_MAP: dict[LogLevel, tuple[int, str]] = {
        "debug": (logging.DEBUG, "[DEBUG] "),
        "info": (logging.INFO, ""),
        "warn": (logging.WARNING, "⚠️  "),
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

            handler = logging.FileHandler(
                self.log_path,
                mode=self.file_mode,
                encoding="utf-8",
            )

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            handler.setFormatter(formatter)
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
        Log a message to console and/or file.

        Note:
        - Splits using str.split("\\n") intentionally.
        - Relies on print() default newline behavior.
        """
        log_level, prefix = self._LEVEL_MAP[level]

        if log_level == logging.DEBUG and not self.debug_mode:
            return

        text = msg or ""

        # Legacy behavior: split only on '\n'
        lines = text.split("\n") or [""]

        for line in lines:
            console_msg = (
                f"{' ' * indent}{prefix}{line}"
                if line.strip()
                else ""
            )

            if self.verbose:
                print(console_msg)

            if self.log_path:
                self.logger.log(log_level, line)
