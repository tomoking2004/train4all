"""The helpers the trainer is built from — the modules coverage found untested."""

import torch

from train4all.utils import (
    DEFAULT_KEY_WIDTH,
    GpuProbe,
    UnifiedLogger,
    cpu_name,
    cuda_index,
    empty_cuda_cache,
    env_summary,
    get_metric_plot_filename,
    get_metric_plot_title,
    os_name,
    package_versions,
    print_dict_tree,
    remove_dir,
    replace_dict_keys,
    save_curves_plot,
    separator_rule,
)

# ── system: machine introspection ─────────────────────────────────────────────


def test_os_name_is_not_the_bare_kernel_release():
    name = os_name()
    assert name and name != "Unknown"


def test_cpu_name_is_a_model_not_just_the_architecture():
    name = cpu_name()
    assert name and name != "Unknown"


def test_env_summary_reports_the_host(tmp_path):
    summary = env_summary(tmp_path)
    assert {"OS", "CPU", "CPU cores", "RAM", "Disk", "GPU", "Python", "PyTorch"} <= set(summary)
    assert summary["PyTorch"] == torch.__version__
    assert "GB" in summary["RAM"]


def test_env_summary_says_so_when_there_is_no_gpu(tmp_path):
    summary = env_summary(tmp_path, gpu_index=None)
    assert summary["GPU"] == "Not available"
    assert summary["VRAM"] == "-"


def test_package_versions_reports_what_it_is_asked_for_in_order():
    versions = package_versions("pytest", "psutil")
    assert list(versions) == ["pytest", "psutil"]
    assert all(v[0].isdigit() for v in versions.values())


def test_package_versions_leaves_out_what_is_not_installed():
    assert list(package_versions("psutil", "no-such-distribution")) == ["psutil"]


def test_a_summary_grows_by_merging_package_versions(tmp_path):
    summary = env_summary(tmp_path) | package_versions("pytest")
    assert list(summary)[-1] == "pytest"   # merged rows close the banner
    assert summary["PyTorch"] == torch.__version__


def test_cuda_index_of_a_cpu_device_is_zero():
    assert cuda_index(torch.device("cpu")) == 0


def test_cuda_index_honours_an_explicit_gpu():
    assert cuda_index(torch.device("cuda:3")) == 3


def test_gpu_probe_returns_zeros_rather_than_raising_without_a_gpu():
    """On a machine with no NVML and no nvidia-smi, the probe must stay quiet."""
    probe = GpuProbe(index=0, ttl_s=0.0)
    used, total, free = probe.memory_mib()
    assert (used, total, free) == (0, 0, 0) or total > 0     # either unreadable, or a real GPU
    assert probe.memory_gb() is None or len(probe.memory_gb()) == 2


def test_empty_cuda_cache_is_safe_without_cuda():
    empty_cuda_cache()          # a no-op off CUDA; must not raise


# ── dict_utils: the engine behind load_checkpoint(key_map=...) ────────────────


def test_replace_dict_keys_rewrites_nested_keys():
    obj = {"old.layer.weight": 1, "keep": {"old.inner": 2}}
    out = replace_dict_keys(obj, {"old.": "new."})
    assert out == {"new.layer.weight": 1, "keep": {"new.inner": 2}}


def test_replace_dict_keys_walks_lists_and_tuples():
    out = replace_dict_keys([{"old.a": 1}, ({"old.b": 2},)], {"old.": "new."})
    assert out == [{"new.a": 1}, ({"new.b": 2},)]


def test_a_sequence_that_cannot_be_rebuilt_passes_through():
    """`range` is a Sequence, and `range(items)` is a TypeError — so it is left alone."""
    obj = range(3)
    assert replace_dict_keys(obj, {"old.": "new."}) is obj


def test_replace_dict_keys_leaves_non_string_keys_alone():
    assert replace_dict_keys({1: "x", "old.k": "y"}, {"old.": "new."}) == {1: "x", "new.k": "y"}


def test_an_empty_map_is_the_identity():
    obj = {"old.a": 1}
    assert replace_dict_keys(obj, {}) is obj


def test_scalars_pass_straight_through():
    assert replace_dict_keys(7, {"a": "b"}) == 7
    assert replace_dict_keys("old.a", {"old.": "new."}) == "old.a"   # a value, not a key


# ── log_utils ─────────────────────────────────────────────────────────────────


def test_print_dict_tree_nests(capsys):
    print_dict_tree({"top": {"inner": 1}, "leaf": 2}, header="H")
    out = capsys.readouterr().out
    assert "H" in out and "top" in out and "inner" in out and "leaf" in out
    assert "└─" in out and "├─" in out


def test_max_depth_zero_keeps_the_output_flat(capsys):
    """A nested dict opens no branch — it is rendered as a value on its own line."""
    print_dict_tree({"top": {"inner": 1}, "leaf": 2}, max_depth=0)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2, "max_depth=0 expanded a branch it should have flattened"
    assert "{'inner': 1}" in lines[0]


def test_a_tiny_float_falls_back_to_scientific_notation(capsys):
    """1e-5 at 4 decimal places would print as 0.0000 — a nonzero value shown as zero."""
    print_dict_tree({"lr": 1e-5}, float_fmt=4)
    out = capsys.readouterr().out
    assert "0.0000" not in out
    assert "1e-5" in out


def test_separator_rule_tracks_the_key_width():
    assert len(separator_rule(10)) < len(separator_rule(40))
    assert separator_rule(DEFAULT_KEY_WIDTH).strip("─ ") == ""


def test_the_logger_writes_to_console_and_file(tmp_path, capsys):
    path = tmp_path / "log.txt"
    logger = UnifiedLogger("t4a-test", log_path=path, verbose=True)
    logger.log("hello")
    logger.log("shh", level="debug")             # debug_mode is off — dropped
    logger.log("careful", level="warn")

    out = capsys.readouterr().out
    assert "hello" in out and "careful" in out and "shh" not in out
    written = path.read_text(encoding="utf-8")
    assert "hello" in written and "shh" not in written


# ── plot_utils ────────────────────────────────────────────────────────────────


def test_plot_names_are_built_from_their_parts():
    assert get_metric_plot_filename("loss") == "loss.png"
    assert get_metric_plot_filename("loss", phase_name="train", prefix="step") == "step_loss_train.png"
    assert get_metric_plot_title("loss", phase_name="train", prefix="step-level") == "Step-level loss (train)"
    assert get_metric_plot_title("accuracy") == "Accuracy"


def test_save_curves_plot_writes_a_png(tmp_path):
    out = save_curves_plot({"train": [1.0, 0.5], "val": [1.1, 0.6]}, tmp_path / "p" / "loss.png")
    assert out.exists() and out.stat().st_size > 0


def test_save_curves_plot_skips_empty_series(tmp_path):
    out = save_curves_plot({"train": [1.0], "val": []}, tmp_path / "loss.png")
    assert out.exists()


# ── file_utils ────────────────────────────────────────────────────────────────


def test_remove_dir_is_a_no_op_when_absent(tmp_path):
    remove_dir(tmp_path / "never-existed")       # must not raise


def test_remove_dir_deletes_recursively(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "f.txt").write_text("x", encoding="utf-8")
    remove_dir(tmp_path / "a")
    assert not (tmp_path / "a").exists()
