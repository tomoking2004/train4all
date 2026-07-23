"""Machine introspection: what host is this run on, and what is its GPU doing?

Nothing here knows about training. It reads the Windows registry for a CPU model,
parses ``/proc/cpuinfo`` on Linux, initializes NVML, and shells out to
``nvidia-smi`` — all of which is about the machine, not the loop.
:class:`~train4all.BaseTrainer` keeps its public methods and delegates here, the
same way it delegates the on-disk format to :class:`~train4all.Checkpoint`, so the
trainer's own file stays about training.
"""

import contextlib
import gc
import importlib.metadata
import multiprocessing
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
import torch

__all__ = [
    "GpuProbe",
    "cpu_name",
    "cuda_index",
    "empty_cuda_cache",
    "env_summary",
    "gpu_temperature",
    "os_name",
]


# ── Host ──────────────────────────────────────────────────────────────────────

def os_name() -> str:
    """Human-readable OS name and version.

    The distro on Linux (e.g. ``Ubuntu 24.04``) and ``macOS <ver>`` on Darwin —
    not the kernel release, which is what ``platform.release()`` would give.
    """
    system = platform.system()
    if system == "Linux":
        try:
            return platform.freedesktop_os_release().get("PRETTY_NAME") or "Linux"
        except OSError:
            return f"Linux {platform.release()}"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}".rstrip()
    return f"{system} {platform.release()}"


def cpu_name() -> str:
    """Best-effort CPU model name.

    ``platform.processor()`` yields only the architecture (e.g. ``x86_64``) off
    Windows, so query the OS directly and fall back to the architecture only when
    the model is genuinely unavailable.
    """
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def env_summary(disk_path: Path | str, gpu_index: int | None = None) -> dict[str, Any]:
    """The system and runtime summary printed as a run's reproducibility banner.

    Args:
        disk_path: Path whose filesystem the free/total disk figures describe.
        gpu_index: CUDA device to report, or ``None`` when no GPU is in play.

    Returns:
        Ordered mapping of label to value, ready for a tree print.
    """
    disk = shutil.disk_usage(disk_path)
    result: dict[str, Any] = {
        "OS":        os_name(),
        "CPU":       cpu_name(),
        "CPU cores": multiprocessing.cpu_count(),
        "RAM":       f"{psutil.virtual_memory().total / 1e9:.2f} GB",
        "Disk":      f"{disk.free / 1e9:.2f} / {disk.total / 1e9:.2f} GB free",
    }
    if gpu_index is not None:
        props = torch.cuda.get_device_properties(gpu_index)
        result["GPU"]   = f"cuda:{gpu_index} {torch.cuda.get_device_name(gpu_index)}"
        result["VRAM"]  = f"{props.total_memory / 1e9:.2f} GB"
        result["CUDA"]  = torch.version.cuda
        result["cuDNN"] = str(torch.backends.cudnn.version())
    else:
        result |= {"GPU": "Not available", "VRAM": "-", "CUDA": "-", "cuDNN": "-"}
    result["Python"]  = platform.python_version()
    result["PyTorch"] = torch.__version__
    for pkg in ("torchvision", "torchaudio"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result[pkg] = importlib.metadata.version(pkg)
    return result


# ── GPU ───────────────────────────────────────────────────────────────────────

def cuda_index(device: torch.device) -> int:
    """Index of the CUDA device to report on and probe."""
    if device.type == "cuda" and device.index is not None:
        return device.index
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def gpu_temperature(index: int) -> int | None:
    """Current GPU temperature in °C via ``nvidia-smi``.

    Returns ``None`` when the tool answers with something that is not a number.
    The caller decides how to report a failure, so the errors propagate.

    Raises:
        FileNotFoundError: ``nvidia-smi`` is not installed.
        subprocess.CalledProcessError: it ran but exited non-zero.
    """
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    reading = result.stdout.strip()
    return int(reading) if reading.isdigit() else None


def empty_cuda_cache() -> None:
    """Free Python-held tensor references and clear the CUDA memory cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


class GpuProbe:
    """Cached GPU-memory readings for one CUDA device.

    NVML is initialized once and its device handle reused on every later call, so
    querying memory inside a per-step progress bar costs a single cheap lookup
    rather than an init/shutdown cycle. Where NVML is unavailable, it falls back to
    an ``nvidia-smi`` query whose result is cached for ``ttl_s`` seconds, so no
    subprocess is spawned per step either.

    Args:
        index: CUDA device index to probe.
        ttl_s: Seconds an ``nvidia-smi`` reading stays valid.
    """

    def __init__(self, index: int, *, ttl_s: float = 2.0) -> None:
        self.index = index
        self.ttl_s = ttl_s
        self._pynvml: Any = None
        self._handle: Any = None
        self._nvml_failed = False
        self._cache: tuple[int, int, int] = (0, 0, 0)
        self._cache_t: float = 0.0

    def memory_mib(self) -> tuple[int, int, int]:
        """Return ``(used, total, free)`` GPU memory in MiB. ``(0, 0, 0)`` if unreadable."""
        # Reuse the live NVML handle; drop it on error so the init path retries.
        if self._handle is not None:
            try:
                return self._nvml_mib(self._handle)
            except Exception:
                self._handle = None

        if not self._nvml_failed:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                return self._nvml_mib(self._handle)
            except Exception:
                self._nvml_failed = True   # NVML unavailable — use the smi fallback

        now = time.time()
        if now - self._cache_t < self.ttl_s:
            return self._cache
        self._cache_t = now
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i", str(self.index),
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            used, total = (int(x) for x in output.split(","))  # int() tolerates whitespace
            self._cache = (used, total, total - used)
        except Exception:
            self._cache = (0, 0, 0)
        return self._cache

    def memory_gb(self) -> tuple[float, float] | None:
        """Return ``(used_gb, total_gb)``, or ``None`` when no reading is available.

        Decimal GB, to match the VRAM total reported by :func:`env_summary`.
        """
        used_mib, total_mib, _ = self.memory_mib()
        if total_mib <= 0:
            return None
        mib_to_gb = (1 << 20) / 1e9
        return (used_mib * mib_to_gb, total_mib * mib_to_gb)

    def _nvml_mib(self, handle: Any) -> tuple[int, int, int]:
        """Query an NVML device *handle*, returning ``(used, total, free)`` in MiB."""
        mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.used >> 20, mem.total >> 20, mem.free >> 20
