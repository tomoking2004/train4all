# Examples

This directory contains minimal runnable examples demonstrating how to use `train4all`.

Each example:

- subclasses `train4all.trainer.BaseTrainer`
- implements `setup`
- implements `compute_loss`
- implements `compute_metrics`
- runs training and evaluation

## mnist

A minimal CNN training example demonstrating:

- model registration
- optimizer registration
- loss and metric definition
- the training and testing workflow

Run:

```bash
cd mnist
python train.py
```
