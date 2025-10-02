"""Training utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping
import json
import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data import DatasetConfig, build_dataloaders
from .models import create_model


@dataclass
class TrainingConfig:
    model_name: str
    num_classes: int
    dataset: DatasetConfig
    epochs: int = 100
    learning_rate: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    device: str = "auto"
    output_dir: Path = Path("runs/train")
    pretrained: bool = True
    freeze_backbone: bool = False
    save_checkpoint: bool = True
    history_filename: str = "history.json"

    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }


class Trainer:
    def __init__(self, config: TrainingConfig, dataloaders: Mapping[str, DataLoader] | None = None) -> None:
        self.config = config
        self.output_dir = config.resolved_output_dir()
        self.logger = self._setup_logger()
        self.device = self._select_device(config.device)
        self.dataloaders, self.datasets = self._prepare_dataloaders(dataloaders)
        self.model = create_model(
            config.model_name,
            config.num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        ).to(self.device)
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("fast_iqa.trainer")
        logger.setLevel(logging.INFO)
        log_path = self.output_dir / "training.log"
        if not logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _select_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
        if requested == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_dataloaders(self, dataloaders: Mapping[str, DataLoader] | None) -> tuple[Dict[str, DataLoader], Dict[str, Iterable]]:
        if dataloaders is not None:
            return dict(dataloaders), {}
        loaders, datasets = build_dataloaders(self.config.dataset)
        return loaders, datasets

    def train(self) -> TrainingHistory:
        history = TrainingHistory()

        for epoch in range(self.config.epochs):
            self.logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)
            for phase in ["train", "val"]:
                self.model.train() if phase == "train" else self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                dataset_size = len(self.dataloaders[phase].sampler) if hasattr(self.dataloaders[phase], "sampler") else len(self.dataloaders[phase].dataset)

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if phase == "train":
                        self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects / dataset_size

                if phase == "train":
                    history.train_loss.append(epoch_loss)
                    history.train_acc.append(epoch_acc)
                else:
                    history.val_loss.append(epoch_loss)
                    history.val_acc.append(epoch_acc)

                self.logger.info("%s Loss: %.4f Acc: %.4f", phase.capitalize(), epoch_loss, epoch_acc)

        if self.config.save_checkpoint:
            checkpoint_path = self.output_dir / "model.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            self.logger.info("Saved checkpoint to %s", checkpoint_path)

        history_path = self.output_dir / self.config.history_filename
        history_path.write_text(json.dumps(history.to_dict(), indent=2))
        self.logger.info("Saved training history to %s", history_path)

        return history


__all__ = ["TrainingConfig", "TrainingHistory", "Trainer"]
