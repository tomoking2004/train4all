import logging
from pathlib import Path
from typing import Any, Callable, Dict


def print_flat_dict_tree(
    data: Dict[str, Any],
    header: str,
    key_width: int = 32,
    float_fmt: int = 4,
    trailing_newline: bool = True,
    print_fn: Callable[[str], None] | None = None,
) -> None:
    """
    Pretty-print a flat (1-level) dictionary in a tree-like format.

    Each key-value pair is displayed in a neat "├─ key: value" style,
    with the last item using "└─" for a visually pleasing finish.

    Args:
        data (dict[str, Any]): Dictionary to print. Values can be int, float, str, etc.
        header (str): Header text displayed above the dictionary.
        key_width (int): Width used to align keys.
        float_fmt (int): Number of decimal places for float values.
        trailing_newline (bool): Add a blank line after the output.
        print_fn (callable | None): Optional custom print function.
    """
    if print_fn is None:
        print_fn = print

    print_fn(header)

    lines = []
    for k, v in data.items():
        if isinstance(v, float):
            v_str = f"{v:.{float_fmt}f}"
        else:
            v_str = str(v)
        lines.append(f" ├─ {k:<{key_width}}: {v_str}")

    if lines:
        lines[-1] = lines[-1].replace("├", "└")

    print_fn("\n".join(lines))

    if trailing_newline:
        print_fn()


def print_dict_tree(
    tree: Dict[str, Any],
    max_depth: int | None = None,
    header: str | None = None,
    key_width: int = 32,
    trailing_newline: bool = True,
    print_fn: Callable[[str], None] | None = None,
    # internal
    indent: int = 0,
) -> None:
    """
    Pretty-print nested dictionaries in a tree-like format.

    Args:
        tree (dict): Dictionary to print.
        max_depth (int | None): Maximum depth to expand. `None` means unlimited.
        header (str | None): Header shown at root.
        key_width (int): Width for key alignment.
        trailing_newline (bool): Add blank line at the end.
        print_fn (callable | None): Print function to use.
        indent (int): Current indentation level (internal).
    """
    if print_fn is None:
        print_fn = print

    if indent == 0 and header:
        print_fn(header)

    base_indent = " " * (indent * 2)
    next_indent = indent + 1

    for key, value in tree.items():
        is_dict = isinstance(value, dict)

        can_expand = (
            is_dict
            and (max_depth is None or indent < max_depth)
        )

        if can_expand:
            print_fn(f"{base_indent}  - {key}")
            print_dict_tree(
                value,
                print_fn=print_fn,
                key_width=key_width,
                indent=next_indent,
                max_depth=max_depth,
                header=None,
                trailing_newline=False,
            )
        else:
            print_fn(f"{base_indent}  - {key:<{key_width - indent * 2}}: {value}")

    if indent == 0 and trailing_newline:
        print_fn()


class UnifiedLogger:
    """
    Unified logger for console and optional file output.

    - Console printing can be enabled/disabled via `verbose`.
    - File logging occurs if `log_path` is provided.
    - Debug messages are shown only if `debug_mode` is True.
    - Supports multi-line messages, blank lines, indentation, and level-specific prefixes.

    Args:
        name (str): Logger name.
        log_path (str | Path | None): Path to log file. If None, file logging is disabled.
        verbose (bool, default=True): Enable/disable console output.
        debug_mode (bool, default=False): Enable debug-level messages.
        file_mode (str, default='a'): File mode ('a' for append, 'w' for overwrite).
    """

    _LEVEL_MAP = {
        "debug": (logging.DEBUG, "[DEBUG] "),
        "info": (logging.INFO, ""),
        "warn": (logging.WARNING, "⚠️  "),
    }

    def __init__(
        self,
        name: str,
        log_path: str | Path | None = None,
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
        logger.handlers.clear()

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                self.log_path, mode=self.file_mode, encoding="utf-8"
            )
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log(self, msg: str | None = None, level: str = "info", indent: int = 0) -> None:
        """
        Log a message to console and/or file.

        Console output respects `verbose`. File logging occurs only if `log_path` is set.
        Debug messages are logged only if `debug_mode` is True.

        Args:
            msg (str | None): Message to log. None or empty string prints a blank line.
            level (str, default="info"): Log level ("info", "debug", "warn").
            indent (int, default=0): Number of spaces for console indentation.
        """
        log_level, prefix = self._LEVEL_MAP.get(level, (logging.INFO, ""))
        
        # Skip debug messages if debug_mode is False
        if log_level == logging.DEBUG and not self.debug_mode:
            return

        lines = (msg or "").split("\n")

        for line in lines:
            stripped = line.strip()
            console_msg = f"{' ' * indent}{prefix}{line}" if stripped else ""

            if self.verbose:
                print(console_msg)

            if self.log_path:
                self.logger.log(log_level, console_msg)
