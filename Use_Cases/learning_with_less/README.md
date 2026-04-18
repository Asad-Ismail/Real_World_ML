# Learning With Less Data

This folder is now a runnable study example for low-label learning, not just a research note.

It demonstrates three practical settings:

1. `supervised`: train only on the small labeled subset
2. `self_supervised`: learn an encoder from unlabeled images with a SimCLR-style contrastive objective
3. `semi_supervised`: combine labeled and unlabeled images with MixMatch or C-MixMatch style regression training

## Dataset Behavior

The original use case was age prediction on UTK-Face.

For this repository to stay runnable offline, the code now supports:

- `--dataset_source utkface`: use the original Hugging Face UTK-Face dataset
- `--dataset_source digits`: use the built-in scikit-learn digits dataset
- `--dataset_source auto`: try UTK-Face first, then fall back to digits if the dataset is unavailable

The digits fallback treats digit value as a regression target. It is not a face-age benchmark, but it is a good local teaching example for limited-data supervised, self-supervised, and semi-supervised learning.

## Install

```bash
pip install -r Use_Cases/learning_with_less/requirements.txt
```

## Quick Smoke Test

Run a short supervised experiment:

```bash
python Use_Cases/learning_with_less/run.py --dataset_source digits --mode supervised --epochs 2
```

Try self-supervised pretraining:

```bash
python Use_Cases/learning_with_less/run.py --dataset_source digits --mode self_supervised --epochs 2
```

Try semi-supervised learning:

```bash
python Use_Cases/learning_with_less/run.py --dataset_source digits --mode semi_supervised --epochs 2
```

Try sequential training:

```bash
python Use_Cases/learning_with_less/run.py --dataset_source digits --mode semi_supervised --sequential_training --self_supervised_epochs 2 --epochs 2
```

## Outputs

Training artifacts are written to:

- `Use_Cases/learning_with_less/results/*.pth`
- `Use_Cases/learning_with_less/results/*_history.json`
- `Use_Cases/learning_with_less/results/*_loss.png`

## Study Notes

- Start with `supervised` to understand the small-label baseline.
- Move to `self_supervised` to see how unlabeled images can still train a useful encoder.
- Move to `semi_supervised` to see how pseudo-targets and consistency-style mixing use both labeled and unlabeled data.
- Use `--train_pct` and `--val_pct` to see how performance changes as labeled data becomes scarce.

This is intentionally a compact teaching implementation. It is designed to be read and modified by someone learning the ideas, not to be a production training stack.
