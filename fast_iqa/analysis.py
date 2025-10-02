"""Analytical helpers for inspecting model predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score, f1_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_score: float
    confusion: np.ndarray


@dataclass
class DimensionalityReduction:
    tsne: np.ndarray
    pca: np.ndarray
    umap: np.ndarray


@dataclass
class SilhouetteScores:
    original: np.ndarray
    tsne: np.ndarray
    pca: np.ndarray
    umap: np.ndarray
    average: Dict[str, float]
    per_method_per_class: Dict[str, Dict[int, float]]


def compute_classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> ClassificationMetrics:
    labels_array = np.asarray(labels)
    preds_array = np.asarray(predictions)
    accuracy = float((labels_array == preds_array).mean())
    confusion = confusion_matrix(labels_array, preds_array)
    f1 = float(f1_score(labels_array, preds_array, average="weighted"))
    return ClassificationMetrics(accuracy=accuracy, f1_score=f1, confusion=confusion)


def compute_dimensionality_reduction(features: np.ndarray, random_state: int = 0) -> DimensionalityReduction:
    tsne = TSNE(n_components=2, random_state=random_state, init="pca")
    tsne_features = tsne.fit_transform(features)
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_features = umap_reducer.fit_transform(features)
    return DimensionalityReduction(tsne=tsne_features, pca=pca_features, umap=umap_features)


def _silhouette_for_method(features: np.ndarray, labels: Sequence[int], num_classes: int) -> Tuple[np.ndarray, Dict[int, float]]:
    values = silhouette_samples(features, labels)
    per_class: Dict[int, float] = {}
    labels_array = np.asarray(labels)
    for cls in range(num_classes):
        mask = labels_array == cls
        per_class[cls] = float(values[mask].mean()) if mask.any() else float("nan")
    return values, per_class


def compute_silhouette_scores(features: Mapping[str, np.ndarray], labels: Sequence[int], num_classes: int) -> SilhouetteScores:
    averages: Dict[str, float] = {}
    per_method_per_class: Dict[str, Dict[int, float]] = {}
    values_by_method: Dict[str, np.ndarray] = {}

    for method, feats in features.items():
        values, per_class = _silhouette_for_method(feats, labels, num_classes)
        values_by_method[method] = values
        per_method_per_class[method] = per_class
        averages[method] = float(silhouette_score(feats, labels))

    return SilhouetteScores(
        original=values_by_method.get("original"),
        tsne=values_by_method.get("tsne"),
        pca=values_by_method.get("pca"),
        umap=values_by_method.get("umap"),
        average=averages,
        per_method_per_class=per_method_per_class,
    )


def compute_distance_metrics(features: Mapping[str, np.ndarray], labels: Sequence[int], num_classes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels_array = np.asarray(labels)

    def intra_class(feats: np.ndarray) -> Dict[int, float]:
        distances: Dict[int, float] = {}
        for cls in range(num_classes):
            cls_features = feats[labels_array == cls]
            if len(cls_features) > 1:
                distances[cls] = float(cdist(cls_features, cls_features, "euclidean").mean())
            else:
                distances[cls] = float("nan")
        return distances

    def inter_class(feats: np.ndarray) -> List[Tuple[int, int, float]]:
        values: List[Tuple[int, int, float]] = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                first = feats[labels_array == i]
                second = feats[labels_array == j]
                if len(first) and len(second):
                    distance = float(cdist(first, second, "euclidean").mean())
                    values.append((i, j, distance))
        return values

    intra_frames: Dict[str, Dict[int, float]] = {method: intra_class(feats) for method, feats in features.items()}
    inter_frames: Dict[str, List[Tuple[int, int, float]]] = {method: inter_class(feats) for method, feats in features.items()}

    intra_df = pd.DataFrame(intra_frames).rename_axis("class").reset_index()
    inter_map: Dict[Tuple[int, int], Dict[str, float]] = {}
    for method, rows in inter_frames.items():
        for i, j, value in rows:
            entry = inter_map.setdefault((i, j), {})
            entry[method] = value
    inter_df = pd.DataFrame([{"class_1": i, "class_2": j, **values} for (i, j), values in inter_map.items()])
    return intra_df, inter_df


def save_confusion_matrix(confusion: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_scatter(points: np.ndarray, labels: Sequence[int], class_names: Sequence[str], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["label"] = [class_names[label] for label in labels]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics_tables(output_dir: Path, silhouette: SilhouetteScores, intra: pd.DataFrame, inter: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    silhouette_table = pd.DataFrame({"method": list(silhouette.average.keys()), "silhouette": list(silhouette.average.values())})
    silhouette_table.to_csv(output_dir / "silhouette_scores.csv", index=False)
    intra.to_csv(output_dir / "intra_class_distances.csv", index=False)
    inter.to_csv(output_dir / "inter_class_distances.csv", index=False)


__all__ = [
    "ClassificationMetrics",
    "DimensionalityReduction",
    "SilhouetteScores",
    "compute_classification_metrics",
    "compute_dimensionality_reduction",
    "compute_silhouette_scores",
    "compute_distance_metrics",
    "save_confusion_matrix",
    "save_scatter",
    "save_metrics_tables",
]
