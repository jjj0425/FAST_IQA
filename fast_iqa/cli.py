"""Command line interface for FAST-IQA."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from torchvision import datasets

from .data import DatasetConfig
from .inference import InferenceConfig, run_inference
from .training import Trainer, TrainingConfig


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the dataset root folder")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for resizing/augmentation")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation for the training split")
    parser.add_argument("--split-info", type=Path, default=None, help="Optional path to write the train/val split information")


def _configure_training_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train_parser = subparsers.add_parser("train", help="Train an image quality classifier")
    _add_dataset_arguments(train_parser)
    train_parser.add_argument("--model", choices=["vgg16", "resnet50", "mobilenetv2", "googlenet"], required=True)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    train_parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all backbone weights")
    train_parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights")


def _configure_inference_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    infer_parser = subparsers.add_parser("infer", help="Run inference with a trained classifier")
    infer_parser.add_argument("--data-root", type=Path, required=True)
    infer_parser.add_argument("--model", choices=["vgg16", "resnet50", "mobilenetv2", "googlenet"], required=True)
    infer_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint")
    infer_parser.add_argument("--num-classes", type=int, default=None, help="Number of classes for the classifier")
    infer_parser.add_argument("--batch-size", type=int, default=32)
    infer_parser.add_argument("--num-workers", type=int, default=4)
    infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    infer_parser.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    infer_parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding analysis computations")
    infer_parser.add_argument("--random-state", type=int, default=0, help="Random seed for dimensionality reduction")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAST-IQA utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _configure_training_parser(subparsers)
    _configure_inference_parser(subparsers)
    return parser


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.no_augment,
        split_info_path=args.split_info,
    )


def _infer_num_classes(dataset_root: Path) -> int:
    dataset = datasets.ImageFolder(dataset_root)
    return len(dataset.classes)


def handle_train(args: argparse.Namespace) -> None:
    dataset_config = _build_dataset_config(args)
    num_classes = _infer_num_classes(args.data_root)
    training_config = TrainingConfig(
        model_name=args.model,
        num_classes=num_classes,
        dataset=dataset_config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
    )
    trainer = Trainer(training_config)
    history = trainer.train()
    print(f"Training complete. Final val accuracy: {history.val_acc[-1]:.4f}" if history.val_acc else "Training complete.")


def handle_infer(args: argparse.Namespace) -> None:
    config = InferenceConfig(
        data_root=args.data_root,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        compute_embeddings=not args.no_embeddings,
        random_state=args.random_state,
    )
    metrics = run_inference(config)
    print(f"Inference accuracy: {metrics.accuracy:.4f}, F1-score: {metrics.f1_score:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        handle_train(args)
    elif args.command == "infer":
        handle_infer(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
