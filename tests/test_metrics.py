"""Epoch metrics are sample-weighted averages, so an uneven final batch is not over-counted."""

from typing import Any

import torch
from conftest import TinyTrainer, make_loader
from torch.utils.data import DataLoader, TensorDataset

from train4all import Phase


class ConstantMetric(TinyTrainer):
    """Reports 1.0 for the first batch and 0.0 for every later one.

    With batches of 5 and 3, an unweighted mean would give 0.5; the correct
    sample-weighted mean is (5·1 + 3·0) / 8 = 0.625.
    """

    def setup(self) -> None:
        super().setup()
        self.calls = 0

    def compute_metrics(self, batch: Any) -> dict[str, float]:
        # The probe does not look at the data — it only marks which batch it is,
        # so the weighting can be checked against a number known in advance.
        value = 1.0 if self.calls == 0 else 0.0
        self.calls += 1
        return {"probe": value}


def test_epoch_metrics_are_sample_weighted(tmp_path):
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)   # -> 5, then 3

    trainer = ConstantMetric(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path, use_progress_bar=False,
    )
    trainer.train(Phase("train", loader, training=True))

    probe = trainer.get_epoch_metrics()["probe"]["train"][0]
    assert probe == 0.625, "an unweighted mean would have given 0.5"


class TokenWeighted(ConstantMetric):
    def get_batch_weight(self, batch: Any) -> int:
        x, _ = batch
        return int(x.size(0)) * 10        # any consistent weight; scale must cancel


def test_get_batch_weight_drives_the_average(tmp_path):
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)

    trainer = TokenWeighted(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path, use_progress_bar=False,
    )
    trainer.train(Phase("train", loader, training=True))
    # A constant scale factor cancels in Σ(m·w)/Σw, so the answer is unchanged.
    assert trainer.get_epoch_metrics()["probe"]["train"][0] == 0.625


def test_step_metrics_follow_the_phase_and_the_master_switch(tmp_path):
    trainer = TinyTrainer(
        num_epochs=1, learning_rate=0.1, run_dir=tmp_path,
        record_step_metrics=True, use_progress_bar=False,
    )
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),                    # record_steps defaults to training
    )
    step_phases = {p for series in trainer.get_step_metrics().values() for p in series}
    assert step_phases == {"train"}, "evaluation phases should not record steps by default"


def test_step_metrics_are_off_unless_the_master_switch_is_on(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    assert trainer.get_step_metrics() == {}


def test_artifacts_are_written_for_every_recorded_phase(trainer):
    trainer.train(
        Phase("train", make_loader(8), training=True),
        Phase("val", make_loader(8)),
    )
    assert trainer.get_epoch_metrics_path().exists()
    assert (trainer.run_dir / "plots" / "loss.png").exists()
    assert (trainer.run_dir / "plots" / "accuracy.png").exists()


def test_clear_metrics_empties_both_tables(trainer):
    trainer.train(Phase("train", make_loader(8), training=True))
    trainer.clear_metrics()
    assert trainer.get_epoch_metrics() == {}
    assert trainer.get_step_metrics() == {}
