"""Machine introspection, on a machine that is not the one running the tests.

`system.py` is almost entirely branches no single runner can enter: the Windows
registry read, the `sysctl` call, `/proc/cpuinfo`, NVML, `nvidia-smi`. An OS matrix
does not help — each runner covers its own branch and misses the other three, and
the union never closes. So every window the module opens onto the outside world is
substituted here: `platform` and `sys.platform` for the host it thinks it is on,
`Path` and `subprocess` for what it reads, and the `winreg` and `pynvml` imports for
the two modules it reaches for only where they exist. That lets every branch run
everywhere. A handful of tests below still read the real machine, and say so.
"""

import contextlib
import sys
from types import SimpleNamespace

import torch

from train4all.utils import (
    GpuProbe,
    cpu_name,
    cuda_index,
    empty_cuda_cache,
    env_summary,
    gpu_temperature,
    os_name,
    package_versions,
    system,
)


def stub_platform(monkeypatch, **attrs):
    """Replace `system`'s view of the `platform` module with only what a call reads."""
    monkeypatch.setattr(system, "platform", SimpleNamespace(**attrs))


def stub_host(monkeypatch, name):
    """Replace `system`'s view of `sys`, whose `platform` selects the `cpu_name` branch."""
    monkeypatch.setattr(system, "sys", SimpleNamespace(platform=name))


def fake_pynvml(used_mib, total_mib, free_mib):
    """The three NVML calls `GpuProbe` makes, reporting the given MiB figures."""
    mib = 1 << 20
    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: f"handle-{index}",
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(
            used=used_mib * mib, total=total_mib * mib, free=free_mib * mib
        ),
    )


# ── os_name ───────────────────────────────────────────────────────────────────


def test_os_name_names_the_real_host():
    """Read from the real machine: the stubs below agree with each other by construction,
    so something has to check that the unstubbed call answers at all."""
    name = os_name()
    assert name and name != "Unknown"


def test_os_name_reads_the_distro_from_the_os_release_file(monkeypatch):
    stub_platform(
        monkeypatch,
        system=lambda: "Linux",
        freedesktop_os_release=lambda: {"PRETTY_NAME": "Ubuntu 24.04.1 LTS"},
        release=lambda: "6.8.0-41-generic",
    )
    assert os_name() == "Ubuntu 24.04.1 LTS"


def test_os_name_falls_back_to_the_kernel_when_there_is_no_os_release_file(monkeypatch):
    def unreadable():
        raise OSError("no /etc/os-release on this image")

    stub_platform(
        monkeypatch,
        system=lambda: "Linux",
        freedesktop_os_release=unreadable,
        release=lambda: "6.8.0-41-generic",
    )
    assert os_name() == "Linux 6.8.0-41-generic"


def test_os_name_falls_back_when_the_os_release_file_names_nothing(monkeypatch):
    """The file can parse and still carry no `PRETTY_NAME`."""
    stub_platform(
        monkeypatch,
        system=lambda: "Linux",
        freedesktop_os_release=lambda: {"ID": "linux"},
        release=lambda: "6.8.0-41-generic",
    )
    assert os_name() == "Linux"


def test_os_name_reports_the_macos_version_not_the_darwin_release(monkeypatch):
    stub_platform(
        monkeypatch,
        system=lambda: "Darwin",
        mac_ver=lambda: ("15.1", ("", "", ""), "arm64"),
        release=lambda: "24.1.0",
    )
    assert os_name() == "macOS 15.1"


def test_os_name_on_darwin_leaves_no_trailing_space_without_a_version(monkeypatch):
    stub_platform(monkeypatch, system=lambda: "Darwin", mac_ver=lambda: ("", ("", "", ""), ""))
    assert os_name() == "macOS"


def test_os_name_elsewhere_pairs_the_system_with_its_release(monkeypatch):
    stub_platform(monkeypatch, system=lambda: "Windows", release=lambda: "11")
    assert os_name() == "Windows 11"


# ── cpu_name ──────────────────────────────────────────────────────────────────


def test_cpu_name_is_a_model_not_just_the_architecture():
    name = cpu_name()
    assert name and name != "Unknown"


def test_cpu_name_reads_the_registry_on_windows(monkeypatch):
    stub_host(monkeypatch, "win32")
    monkeypatch.setitem(sys.modules, "winreg", SimpleNamespace(
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=lambda *_args: contextlib.nullcontext("key"),
        QueryValueEx=lambda *_args: ("Intel(R) Core(TM) i9-13900K  ", 1),
    ))
    assert cpu_name() == "Intel(R) Core(TM) i9-13900K"


def test_cpu_name_reads_sysctl_on_macos(monkeypatch):
    stub_host(monkeypatch, "darwin")
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(
        check_output=lambda *_args, **_kwargs: "Apple M3 Pro\n",
    ))
    assert cpu_name() == "Apple M3 Pro"


def test_cpu_name_reads_proc_cpuinfo_on_linux(monkeypatch):
    stub_host(monkeypatch, "linux")
    cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core\ncache size\t: 1024 KB\n"
    monkeypatch.setattr(system, "Path", lambda *_args: SimpleNamespace(read_text=lambda: cpuinfo))
    assert cpu_name() == "AMD Ryzen 9 7950X 16-Core"


def test_cpu_name_falls_back_to_the_architecture_when_the_model_is_unreadable(monkeypatch):
    def unreadable(*_args):
        raise OSError("/proc is not mounted")

    stub_host(monkeypatch, "linux")
    monkeypatch.setattr(system, "Path", unreadable)
    stub_platform(monkeypatch, processor=lambda: "", machine=lambda: "aarch64")
    assert cpu_name() == "aarch64"


def test_cpu_name_is_unknown_when_even_the_architecture_is_blank(monkeypatch):
    stub_host(monkeypatch, "linux")
    monkeypatch.setattr(system, "Path", lambda *_args: SimpleNamespace(read_text=lambda: ""))
    stub_platform(monkeypatch, processor=lambda: "", machine=lambda: "")
    assert cpu_name() == "Unknown"


# ── package_versions, env_summary ─────────────────────────────────────────────


def test_package_versions_reports_what_it_is_asked_for_in_order():
    versions = package_versions("pytest", "psutil")
    assert list(versions) == ["pytest", "psutil"]
    assert all(v[0].isdigit() for v in versions.values())


def test_package_versions_leaves_out_what_is_not_installed():
    assert list(package_versions("psutil", "no-such-distribution")) == ["psutil"]


def test_env_summary_reports_the_host(tmp_path):
    summary = env_summary(tmp_path)
    assert {"OS", "CPU", "CPU cores", "RAM", "Disk", "GPU", "Python", "PyTorch"} <= set(summary)
    assert summary["PyTorch"] == torch.__version__
    assert "GB" in summary["RAM"]


def test_env_summary_says_so_when_there_is_no_gpu(tmp_path):
    summary = env_summary(tmp_path, gpu_index=None)
    assert summary["GPU"] == "Not available"
    assert summary["VRAM"] == "-"


def test_env_summary_names_the_gpu_it_is_given(tmp_path, monkeypatch):
    """The GPU rows are the half of the banner a CPU-only runner never writes."""
    monkeypatch.setattr(
        system.torch.cuda, "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=24 * 1e9),
    )
    monkeypatch.setattr(system.torch.cuda, "get_device_name", lambda _index: "NVIDIA RTX 4090")

    summary = env_summary(tmp_path, gpu_index=0)

    assert summary["GPU"] == "cuda:0 NVIDIA RTX 4090"
    assert summary["VRAM"] == "24.00 GB"
    assert {"CUDA", "cuDNN"} <= set(summary)


def test_a_summary_grows_by_merging_package_versions(tmp_path):
    summary = env_summary(tmp_path) | package_versions("pytest")
    assert list(summary)[-1] == "pytest"   # merged rows close the banner
    assert summary["PyTorch"] == torch.__version__


# ── cuda_index, gpu_temperature, empty_cuda_cache ─────────────────────────────


def test_cuda_index_of_a_cpu_device_is_zero():
    assert cuda_index(torch.device("cpu")) == 0


def test_cuda_index_honours_an_explicit_gpu():
    assert cuda_index(torch.device("cuda:3")) == 3


def test_gpu_temperature_reads_the_number_nvidia_smi_prints(monkeypatch):
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(stdout=" 65 \n"),
    ))
    assert gpu_temperature(0) == 65


def test_gpu_temperature_is_none_when_the_reading_is_not_a_number(monkeypatch):
    """`nvidia-smi` answers `N/A` for a GPU that reports no temperature sensor."""
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(stdout="N/A\n"),
    ))
    assert gpu_temperature(0) is None


def test_empty_cuda_cache_is_safe_without_cuda():
    empty_cuda_cache()          # a no-op off CUDA; must not raise


def test_empty_cuda_cache_clears_the_cache_when_cuda_is_present(monkeypatch):
    cleared: list[bool] = []
    monkeypatch.setattr(system.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(system.torch.cuda, "empty_cache", lambda: cleared.append(True))

    empty_cuda_cache()

    assert cleared == [True], "the CUDA cache was never cleared"


# ── GpuProbe ──────────────────────────────────────────────────────────────────


def test_the_probe_returns_zeros_rather_than_raising_without_a_gpu():
    """On a machine with no NVML and no nvidia-smi, the probe must stay quiet."""
    probe = GpuProbe(index=0, ttl_s=0.0)
    used, total, free = probe.memory_mib()
    assert (used, total, free) == (0, 0, 0) or total > 0     # either unreadable, or a real GPU
    gb = probe.memory_gb()
    assert gb is None or len(gb) == 2


def test_the_probe_reads_nvml_when_it_is_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(2048, 8192, 6144))
    assert GpuProbe(index=0).memory_mib() == (2048, 8192, 6144)


def test_the_probe_initializes_nvml_once_and_reuses_the_handle(monkeypatch):
    """The reason the probe is affordable inside a per-step progress bar."""
    nvml = fake_pynvml(1024, 4096, 3072)
    inits: list[str] = []
    nvml.nvmlInit = lambda: inits.append("init")
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    probe = GpuProbe(index=0)

    assert probe.memory_mib() == (1024, 4096, 3072)
    assert probe.memory_mib() == (1024, 4096, 3072)

    assert inits == ["init"], "NVML was re-initialized instead of reusing its handle"


def test_a_handle_that_stops_reading_is_retired_rather_than_raising(monkeypatch):
    """A driver reset invalidates the handle; the next call must re-establish one."""
    nvml = fake_pynvml(1024, 4096, 3072)
    healthy = nvml.nvmlDeviceGetMemoryInfo
    stale = {"now": False}

    def read(handle):
        if stale["now"]:
            stale["now"] = False        # the reset is over by the time NVML is re-entered
            raise RuntimeError("invalid handle")
        return healthy(handle)

    nvml.nvmlDeviceGetMemoryInfo = read
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    probe = GpuProbe(index=0)
    assert probe.memory_mib() == (1024, 4096, 3072)

    stale["now"] = True

    assert probe.memory_mib() == (1024, 4096, 3072)


def test_the_probe_falls_back_to_nvidia_smi_when_nvml_is_absent(monkeypatch):
    # A module without `nvmlInit`: the import succeeds and the first call does not,
    # which is what an NVML too old for this binding looks like.
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace())
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(
        check_output=lambda *_args, **_kwargs: " 2048 , 8192 \n",
    ))
    assert GpuProbe(index=0, ttl_s=60.0).memory_mib() == (2048, 8192, 6144)


def test_an_nvidia_smi_reading_is_cached_for_its_ttl(monkeypatch):
    """The fallback costs a subprocess, so a per-step caller must not pay it per step."""
    spawns: list[tuple] = []

    def check_output(*args, **_kwargs):
        spawns.append(args)
        return "1024, 4096\n"

    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace())
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(check_output=check_output))
    probe = GpuProbe(index=0, ttl_s=60.0)

    assert probe.memory_mib() == (1024, 4096, 3072)
    assert probe.memory_mib() == (1024, 4096, 3072)

    assert len(spawns) == 1, "the cached reading spawned a second nvidia-smi"


def test_memory_gb_converts_the_mib_reading_to_decimal_gb(monkeypatch):
    """Decimal GB, to match the VRAM total `env_summary` reports."""
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(1024, 4096, 3072))

    gb = GpuProbe(index=0).memory_gb()

    assert gb is not None
    used_gb, total_gb = gb
    assert round(used_gb, 3) == 1.074
    assert round(total_gb, 3) == 4.295


def test_memory_gb_is_none_when_nothing_can_be_read(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace())
    monkeypatch.setattr(system, "subprocess", SimpleNamespace(check_output=unavailable))
    probe = GpuProbe(index=0, ttl_s=0.0)

    assert probe.memory_mib() == (0, 0, 0)
    assert probe.memory_gb() is None
