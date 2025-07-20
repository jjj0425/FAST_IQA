import os
import argparse
import time
import torch
import umap
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils.fixes import threadpool_limits
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
import seaborn as sns
from model import get_model
from tqdm import tqdm

def compute_silhouette_per_class(silhouette_values, labels, num_classes):
    """
    Compute the mean silhouette score for each class.
    """
    silhouette_per_class = {}
    for i in range(num_classes):
        class_silhouette_values = silhouette_values[np.array(labels) == i]
        class_sample_count = len(class_silhouette_values)
        if class_sample_count > 0:
            silhouette_per_class[i] = np.mean(class_silhouette_values)
        else:
            silhouette_per_class[i] = np.nan
    return silhouette_per_class

def compute_intra_inter_class_distances(features, labels, num_classes):
    """
    Compute intra-class and inter-class distances for feature vectors.
    """
    distances_within_classes = {}
    distances_between_classes = []

    for i in range(num_classes):
        class_features = features[np.array(labels) == i]
        if len(class_features) > 1:
            intra_class_distance = np.mean(cdist(class_features, class_features, 'euclidean'))
            distances_within_classes[i] = intra_class_distance

    all_class_features = [features[np.array(labels) == i] for i in range(num_classes)]
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if all_class_features[i].size > 0 and all_class_features[j].size > 0:
                inter_class_distance = np.mean(cdist(all_class_features[i], all_class_features[j], 'euclidean'))
                distances_between_classes.append((i, j, inter_class_distance))

    return distances_within_classes, distances_between_classes

def inference(data_root, model_name, model_path, output_dir, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_root, data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = get_model(model_name, num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    class_names = dataset.classes

    all_preds = []
    all_labels = []
    inference_times = []
    image_filenames = []
    feature_vectors = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            _, preds = torch.max(outputs, 1)

            inference_time = end_time - start_time

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            inference_times.append(inference_time)
            image_filenames.extend([os.path.basename(dataset.samples[i + len(image_filenames)][0]) for i in range(len(inputs))])

            # Feature extraction using model trunk (removing classifier head)
            if model_name == 'vgg16':
                n_model = torch.nn.Sequential(*list(model.children())[:-1])
            elif model_name == 'resnet50':
                n_model = torch.nn.Sequential(*list(model.children())[:-1])
            elif model_name == 'mobilenetv2':
                n_model = torch.nn.Sequential(*list(model.children())[:-1])
            elif model_name == 'googlenet':
                n_model = torch.nn.Sequential(*list(model.children())[:-2])
            features = n_model(inputs)
            features = features.view(features.size(0), -1)
            feature_vectors.extend(features.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Binary classification: merge classes 0,1 as 0; others as 1
    binary_labels = [0 if l in [0, 1] else 1 for l in all_labels]
    binary_preds = [0 if p in [0, 1] else 1 for p in all_preds]
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds)

    avg_inference_time = np.mean(inference_times)

    results = {
        'image_filename': image_filenames,
        'predicted_label': [class_names[p] for p in all_preds],
        'true_label': [class_names[l] for l in all_labels],
        'binary_predicted_label': binary_preds,
        'binary_true_label': binary_labels,
        'inference_time': [avg_inference_time] * len(image_filenames)
    }

    os.makedirs(output_dir, exist_ok=True)

    sns.set(font_scale=2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    # t-SNE analysis
    feature_vectors = np.array(feature_vectors)
    with threadpool_limits(limits=1, user_api='blas'):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_features = tsne.fit_transform(feature_vectors)

    tsne_df = pd.DataFrame(tsne_features, columns=['tsne_x', 'tsne_y'])
    tsne_df['label'] = [class_names[l] for l in all_labels]
    results['tsne_x'] = tsne_df['tsne_x']
    results['tsne_y'] = tsne_df['tsne_y']

    sns.set(font_scale=1)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='tsne_x', y='tsne_y', hue='label', palette=sns.color_palette("hsv", len(class_names)), data=tsne_df)
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))

    # PCA analysis
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(feature_vectors)
    pca_df = pd.DataFrame(pca_features, columns=['pca_x', 'pca_y'])
    pca_df['label'] = [class_names[l] for l in all_labels]
    results['pca_x'] = pca_df['pca_x']
    results['pca_y'] = pca_df['pca_y']

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='pca_x', y='pca_y', hue='label', palette=sns.color_palette("hsv", len(class_names)), data=pca_df)
    plt.title('PCA Visualization of Feature Vectors')
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))

    # UMAP analysis
    umap_reducer = umap.UMAP(n_components=2)
    umap_features = umap_reducer.fit_transform(feature_vectors)
    umap_df = pd.DataFrame(umap_features, columns=['umap_x', 'umap_y'])
    umap_df['label'] = [class_names[l] for l in all_labels]
    results['umap_x'] = umap_df['umap_x']
    results['umap_y'] = umap_df['umap_y']

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='umap_x', y='umap_y', hue='label', palette=sns.color_palette("hsv", len(class_names)), data=umap_df)
    plt.title('UMAP Visualization of Feature Vectors')
    plt.savefig(os.path.join(output_dir, 'umap_visualization.png'))

    # Silhouette scores
    silhouette_values_original = silhouette_samples(feature_vectors, all_labels)
    silhouette_original = silhouette_score(feature_vectors, all_labels)
    silhouette_per_class_original = compute_silhouette_per_class(silhouette_values_original, all_labels, num_classes=5)

    silhouette_values_tsne = silhouette_samples(tsne_features, all_labels)
    silhouette_tsne = silhouette_score(tsne_features, all_labels)
    silhouette_per_class_tsne = compute_silhouette_per_class(silhouette_values_tsne, all_labels, num_classes=5)

    silhouette_values_pca = silhouette_samples(pca_features, all_labels)
    silhouette_pca = silhouette_score(pca_features, all_labels)
    silhouette_per_class_pca = compute_silhouette_per_class(silhouette_values_pca, all_labels, num_classes=5)

    silhouette_values_umap = silhouette_samples(umap_features, all_labels)
    silhouette_umap = silhouette_score(umap_features, all_labels)
    silhouette_per_class_umap = compute_silhouette_per_class(silhouette_values_umap, all_labels, num_classes=5)

    results['silhouette_original'] = silhouette_values_original
    results['silhouette_tsne'] = silhouette_values_tsne
    results['silhouette_pca'] = silhouette_values_pca
    results['silhouette_umap'] = silhouette_values_umap

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'inference_results.csv'), index=False)

    print(f'Original Silhouette Score: {silhouette_original:.4f}')
    print(f'Per Class Original Silhouette Scores: {silhouette_per_class_original}')
    print(f't-SNE Silhouette Score: {silhouette_tsne:.4f}')
    print(f'Per Class t-SNE Silhouette Scores: {silhouette_per_class_tsne}')
    print(f'PCA Silhouette Score: {silhouette_pca:.4f}')
    print(f'Per Class PCA Silhouette Scores: {silhouette_per_class_pca}')
    print(f'UMAP Silhouette Score: {silhouette_umap:.4f}')
    print(f'Per Class UMAP Silhouette Scores: {silhouette_per_class_umap}')

    # Save silhouette scores and distances to CSV files
    silhouette_scores = {
        'Method': ['Original', 't-SNE', 'PCA', 'UMAP'],
        'Silhouette Score': [silhouette_original, silhouette_tsne, silhouette_pca, silhouette_umap]
    }
    df_silhouette_scores = pd.DataFrame(silhouette_scores)
    df_silhouette_scores.to_csv(os.path.join(output_dir, 'silhouette_scores.csv'), index=False)

    # Compute intra/inter class distances
    distances_within_original, distances_between_original = compute_intra_inter_class_distances(feature_vectors, all_labels, num_classes=5)
    distances_within_tsne, distances_between_tsne = compute_intra_inter_class_distances(tsne_features, all_labels, num_classes=5)
    distances_within_pca, distances_between_pca = compute_intra_inter_class_distances(pca_features, all_labels, num_classes=5)
    distances_within_umap, distances_between_umap = compute_intra_inter_class_distances(umap_features, all_labels, num_classes=5)

    # Save intra-class distances
    df_distances_within = pd.DataFrame(distances_within_original.items(), columns=['Class', 'Intra-Class Distance (Original)'])
    df_distances_within['Intra-Class Distance (t-SNE)'] = [distances_within_tsne.get(i, np.nan) for i in df_distances_within['Class']]
    df_distances_within['Intra-Class Distance (PCA)'] = [distances_within_pca.get(i, np.nan) for i in df_distances_within['Class']]
    df_distances_within['Intra-Class Distance (UMAP)'] = [distances_within_umap.get(i, np.nan) for i in df_distances_within['Class']]
    df_distances_within.loc['Average'] = df_distances_within.mean(numeric_only=True)
    df_distances_within.to_csv(os.path.join(output_dir, 'intra_class_distances.csv'), index=False)

    # Save inter-class distances
    df_distances_between = pd.DataFrame(distances_between_original, columns=['Class1', 'Class2', 'Inter-Class Distance (Original)'])
    df_distances_between['Inter-Class Distance (t-SNE)'] = [d[2] for d in distances_between_tsne]
    df_distances_between['Inter-Class Distance (PCA)'] = [d[2] for d in distances_between_pca]
    df_distances_between['Inter-Class Distance (UMAP)'] = [d[2] for d in distances_between_umap]
    df_distances_between.loc['Average'] = df_distances_between.mean(numeric_only=True)
    df_distances_between.to_csv(os.path.join(output_dir, 'inter_class_distances.csv'), index=False)

    print(f'Binary Accuracy: {binary_accuracy:.4f}')
    print(f'Binary F1-Score: {binary_f1:.4f}')
    print(f'Average Inference Time: {avg_inference_time:.4f}s')

    # Save metrics to text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Binary Accuracy: {binary_accuracy:.4f}\n')
        f.write(f'F1-Score: {f1:.4f}\n')
        f.write(f'Average Inference Time: {avg_inference_time:.4f}s\n')

    print(f'Inference Accuracy: {accuracy:.4f}')
    print(f'Results saved to {output_dir}')

def main():
    parser = argparse.ArgumentParser(description="Inference on a dataset using a trained model")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet50', 'mobilenetv2', 'googlenet'], help='Model name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')

    args = parser.parse_args()

    inference(args.data_root, args.model, args.model_path, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()
