import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train4all.trainer import BaseTrainer


def build_loader(train: bool, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


class Trainer(BaseTrainer):
    def setup(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        self.set_models({"model": self.model})
        self.set_optimizer(
            torch.optim.Adam(self.get_trainable_params(), lr=self.learning_rate)
        )

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)
        self.set_cache("logits", logits.detach())
        return F.cross_entropy(logits, y)

    def compute_metrics(self, batch):
        logits = self.get_cache("logits")
        _, y = batch
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {"accuracy": acc.item()}


def main():
    train_loader = build_loader(train=True)
    test_loader = build_loader(train=False)

    trainer = Trainer(
        num_epochs=5,
        learning_rate=1e-3,
        run_dir="mnist_run",
    )

    trainer.train(train_loader)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
