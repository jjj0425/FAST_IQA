# FAST-IQA

A lightweight training and inference toolkit for image quality assessment
experiments.  The project now exposes a single reusable Python package
(`fast_iqa`) with an ergonomic command line interface for training and
analysing classifiers.

## Installation

```bash
pip install -r requirements.txt  # make sure PyTorch + torchvision are available
```

## Command line usage

### Training

```bash
python -m fast_iqa.cli train \
  --data-root /path/to/dataset \
  --model resnet50 \
  --epochs 50 \
  --output-dir runs/resnet50
```

### Inference and analysis

```bash
python -m fast_iqa.cli infer \
  --data-root /path/to/dataset \
  --model resnet50 \
  --checkpoint runs/resnet50/model.pth \
  --output-dir runs/inference
```

Both commands accept `--help` to inspect all available flags.
