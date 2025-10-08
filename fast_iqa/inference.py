"""Inference pipeline for FAST-IQA models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .data import DEFAULT_MEAN, DEFAULT_STD
from .analysis import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_dimensionality_reduction,
    compute_distance_metrics,
    compute_silhouette_scores,
    save_confusion_matrix,
    save_metrics_tables,
    save_scatter,
)
from .models import build_feature_extractor, create_model


@dataclass
class InferenceConfig:
    data_root: Path
    model_name: str
    checkpoint_path: Path
    num_classes: int | None = None
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"
    output_dir: Path = Path("runs/inference")
    compute_embeddings: bool = True
    random_state: int = 0
    normalize_mean: Sequence[float] = DEFAULT_MEAN
    normalize_std: Sequence[float] = DEFAULT_STD

    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _build_dataloader(config: InferenceConfig) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )
    dataset = datasets.ImageFolder(config.data_root, transform=transform)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


def _load_model(model_name: str, num_classes: int, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_model(model_name, num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _gather_results(dataloader: DataLoader, model: torch.nn.Module, device: torch.device,
                    config: InferenceConfig) -> Dict[str, Union[List[int], List[str], List[float], np.ndarray]]:
    all_preds: List[int] = []
    all_labels: List[int] = []
    image_filenames: List[str] = []
    inference_times: List[float] = []
    features: List[np.ndarray] = []

    feature_extractor = None
    if config.compute_embeddings:
        feature_extractor = build_feature_extractor(model, config.model_name).to(device)
        feature_extractor.eval()

    sample_offset = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.perf_counter()
            outputs = model(inputs)
            elapsed = time.perf_counter() - start_time

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            inference_times.extend([elapsed / len(inputs)] * len(inputs))

            batch_paths = [dataloader.dataset.samples[i + sample_offset][0] for i in range(len(inputs))]
            image_filenames.extend([Path(path).name for path in batch_paths])
            sample_offset += len(inputs)

            if feature_extractor is not None:
                embeddings = feature_extractor(inputs)
                embeddings = torch.flatten(embeddings, 1)
                features.append(embeddings.cpu().numpy())

    gathered: Dict[str, Union[List[int], List[str], List[float], np.ndarray]] = {
        "predictions": all_preds,
        "labels": all_labels,
        "filenames": image_filenames,
        "times": inference_times,
    }
    if features:
        gathered["features"] = np.concatenate(features, axis=0)
    return gathered


def _prepare_feature_matrices(gathered: Dict[str, Union[List[int], List[str], List[float], np.ndarray]]) -> np.ndarray | None:
    feature_matrix = gathered.get("features")
    if feature_matrix is None:
        return None
    if isinstance(feature_matrix, np.ndarray):
        return feature_matrix
    raise ValueError("Invalid feature storage format")


def _binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    binary_labels = [0 if label in (0, 1) else 1 for label in labels]
    binary_preds = [0 if pred in (0, 1) else 1 for pred in predictions]
    accuracy = float(np.mean(np.array(binary_labels) == np.array(binary_preds)))
    # Avoid importing sklearn for a single binary F1 calculation
    true_positive = sum(bl == bp == 1 for bl, bp in zip(binary_labels, binary_preds))
    precision = true_positive / sum(binary_preds) if any(binary_preds) else 0.0
    recall = true_positive / sum(binary_labels) if any(binary_labels) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"binary_accuracy": accuracy, "binary_f1": f1}


def run_inference(config: InferenceConfig) -> ClassificationMetrics:
    device = _select_device(config.device)
    output_dir = config.resolved_output_dir()

    dataloader = _build_dataloader(config)
    num_classes = config.num_classes or len(dataloader.dataset.classes)
    model = _load_model(config.model_name, num_classes, config.checkpoint_path, device)
    gathered = _gather_results(dataloader, model, device, config)

    metrics = compute_classification_metrics(gathered["labels"], gathered["predictions"])

    results_path = output_dir / "inference_results.json"
    payload = {
        "filenames": gathered["filenames"],
        "predicted_label": gathered["predictions"],
        "true_label": gathered["labels"],
        "average_time": float(np.mean(gathered["times"])) if gathered["times"] else 0.0,
    }
    binary = _binary_metrics(gathered["labels"], gathered["predictions"])
    payload.update(binary)
    results_path.write_text(json.dumps(payload, indent=2))

    class_names = dataloader.dataset.classes
    save_confusion_matrix(metrics.confusion, class_names, output_dir / "confusion_matrix.png")

    features = _prepare_feature_matrices(gathered)
    if features is not None:
        reductions = compute_dimensionality_reduction(features, random_state=config.random_state)
        silhouette = compute_silhouette_scores(
            {
                "original": features,
                "tsne": reductions.tsne,
                "pca": reductions.pca,
                "umap": reductions.umap,
            },
            gathered["labels"],
            num_classes,
        )
        intra, inter = compute_distance_metrics(
            {
                "original": features,
                "tsne": reductions.tsne,
                "pca": reductions.pca,
                "umap": reductions.umap,
            },
            gathered["labels"],
            num_classes,
        )

        save_scatter(reductions.tsne, gathered["labels"], class_names, output_dir / "tsne.png", "t-SNE Embeddings")
        save_scatter(reductions.pca, gathered["labels"], class_names, output_dir / "pca.png", "PCA Embeddings")
        save_scatter(reductions.umap, gathered["labels"], class_names, output_dir / "umap.png", "UMAP Embeddings")
        save_metrics_tables(output_dir, silhouette, intra, inter)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "accuracy": metrics.accuracy,
        "f1_score": metrics.f1_score,
        **binary,
    }, indent=2))

    return metrics


__all__ = ["InferenceConfig", "run_inference"]
