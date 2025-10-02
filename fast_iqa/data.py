"""Data loading utilities for FAST-IQA.

The previous code base stored the dataset logic in a couple of loosely
connected scripts.  This module consolidates the behaviour into a single
set of reusable helpers with an explicit configuration object.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import random

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing how to load and split an image dataset."""

    root: Path
    image_size: int = 224
    batch_size: int = 64
    train_ratio: float = 0.8
    num_workers: int = 4
    seed: int = 42
    normalize_mean: Sequence[float] = DEFAULT_MEAN
    normalize_std: Sequence[float] = DEFAULT_STD
    augment: bool = True
    split_info_path: Path | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


def _build_train_transform(config: DatasetConfig) -> transforms.Compose:
    augmentations = [
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
    ]
    if not config.augment:
        augmentations = [transforms.Resize((config.image_size, config.image_size))]

    return transforms.Compose(
        augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )


def _build_eval_transform(config: DatasetConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )


def _split_indices(num_items: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    split_idx = int(train_ratio * num_items)
    return indices[:split_idx], indices[split_idx:]


def _write_split_file(config: DatasetConfig, dataset: datasets.ImageFolder,
                      train_indices: Iterable[int], val_indices: Iterable[int]) -> None:
    if config.split_info_path is None:
        return

    split_path = Path(config.split_info_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    train_filenames = [dataset.samples[i][0] for i in train_indices]
    val_filenames = [dataset.samples[i][0] for i in val_indices]

    payload = {
        "train": train_filenames,
        "val": val_filenames,
    }
    split_path.write_text(json.dumps(payload, indent=2))


def build_dataloaders(config: DatasetConfig) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """Create PyTorch dataloaders for training and validation phases."""

    root = Path(config.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist")

    base_dataset = datasets.ImageFolder(root)
    train_indices, val_indices = _split_indices(len(base_dataset), config.train_ratio, config.seed)

    datasets_by_phase = {
        "train": datasets.ImageFolder(root, transform=_build_train_transform(config)),
        "val": datasets.ImageFolder(root, transform=_build_eval_transform(config)),
    }

    samplers = {
        "train": SubsetRandomSampler(train_indices),
        "val": SubsetRandomSampler(val_indices),
    }

    dataloaders = {
        phase: DataLoader(
            datasets_by_phase[phase],
            batch_size=config.batch_size,
            sampler=samplers[phase],
            num_workers=config.num_workers,
        )
        for phase in ["train", "val"]
    }

    _write_split_file(config, base_dataset, train_indices, val_indices)

    return dataloaders, datasets_by_phase


__all__ = ["DatasetConfig", "build_dataloaders"]
