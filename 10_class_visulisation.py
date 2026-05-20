"""Perceptual hash visualization for STL-10.

This script supports multiple hash functions and distance metrics.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import transforms  # pyright: ignore[reportMissingImports]
from torchvision.datasets import STL10  # pyright: ignore[reportMissingImports]

import compute_photoDNA
import pdqhash  # pyright: ignore[reportMissingImports]


CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize STL-10 perceptual hashes")
    parser.add_argument("--hash-method", type=str, default="pdq", choices=["pdq", "photodna"], help="Hash function to use")
    parser.add_argument("--distance-metric", type=str, default="hamming", choices=["hamming", "l1", "euclidean", "cosine"], help="Distance function to use")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of STL-10 samples to process")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images before hashing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-tsne", dest="use_tsne", action="store_true", help="Use t-SNE for 2D embedding")
    parser.add_argument("--no-tsne", dest="use_tsne", action="store_false", help="Use PCA instead of t-SNE")
    parser.set_defaults(use_tsne=True)
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components before t-SNE")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--max-distance-samples", type=int, default=1000, help="Max samples used for pairwise distance analysis")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of neighbors for kNN")
    return parser.parse_args()


def load_dataset(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])

    dataset = STL10(
        root="./data",
        split="train",
        download=True,
        transform=transform,
    )
    return dataset


def compute_hash_vector(img_np, hash_method):
    if hash_method == "pdq":
        hash_vec, _ = pdqhash.compute(img_np)
    elif hash_method == "photodna":
        hash_vec, _ = compute_photoDNA.compute(img_np)
    else:
        raise ValueError(f"Unsupported hash method: {hash_method}")

    return np.asarray(hash_vec, dtype=np.float32)


def compute_hashes(dataset, indices, hash_method):
    hash_vectors = []
    labels = []

    print(f"Computing {hash_method.upper()} hashes...")

    for idx in tqdm(indices):
        img, label = dataset[idx]
        img_np = np.asarray(img, dtype=np.uint8)
        hash_vectors.append(compute_hash_vector(img_np, hash_method))
        labels.append(label)

    return np.asarray(hash_vectors), np.asarray(labels)


def prepare_for_hamming(hash_vectors):
    unique_values = np.unique(hash_vectors)
    if np.all(np.isin(unique_values, [0.0, 1.0])):
        return hash_vectors.astype(np.uint8)

    return (hash_vectors > np.median(hash_vectors, axis=1, keepdims=True)).astype(np.uint8)


def pairwise_distance(a, b, distance_metric):
    if distance_metric == "hamming":
        return float(np.mean(a != b))
    if distance_metric == "l1":
        return float(np.sum(np.abs(a - b)))
    if distance_metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if distance_metric == "cosine":
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(1.0 - np.dot(a, b) / (a_norm * b_norm))

    raise ValueError(f"Unsupported distance metric: {distance_metric}")


def sklearn_knn_metric(distance_metric):
    if distance_metric == "l1":
        return "manhattan"
    return distance_metric


def reduce_to_2d(hash_vectors, use_tsne, random_seed, pca_components, perplexity):
    scaler = StandardScaler()
    hash_vectors_scaled = scaler.fit_transform(hash_vectors)

    print("Running PCA...")
    pca = PCA(n_components=min(pca_components, hash_vectors_scaled.shape[1], hash_vectors_scaled.shape[0]))
    hash_pca = pca.fit_transform(hash_vectors_scaled)
    print("PCA output shape:", hash_pca.shape)

    if use_tsne:
        print("Running t-SNE (this may take a while)...")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_seed,
            verbose=1,
        )
        return tsne.fit_transform(hash_pca)

    return PCA(n_components=2).fit_transform(hash_pca)


def plot_embedding(embeddings_2d, labels, hash_method, use_tsne):
    print("Plotting...")
    plt.figure(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for class_id in range(10):
        mask = labels == class_id
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=15,
            alpha=0.7,
            color=colors[class_id],
            label=CLASS_NAMES[class_id],
        )

    title = f"{'t-SNE' if use_tsne else 'PCA'} Visualization of {hash_method.upper()} Hash Embeddings (STL-10)"
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()


def compute_distance_analysis(hash_vectors, labels, distance_metric, hash_method, max_samples):
    print("\nComputing intra/inter-class distances...")

    analysis_vectors = hash_vectors
    if distance_metric == "hamming":
        analysis_vectors = prepare_for_hamming(hash_vectors)

    intra_distances = []
    inter_distances = []

    n = len(analysis_vectors)
    limit = min(max_samples, n)

    for i in range(limit):
        for j in range(i + 1, limit):
            distance = pairwise_distance(analysis_vectors[i], analysis_vectors[j], distance_metric)
            if labels[i] == labels[j]:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)

    print(f"Mean intra-class distance: {np.mean(intra_distances):.4f}")
    print(f"Mean inter-class distance: {np.mean(inter_distances):.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(intra_distances, bins=50, alpha=0.6, label="Intra-class")
    plt.hist(inter_distances, bins=50, alpha=0.6, label="Inter-class")
    plt.title(f"{hash_method.upper()} Hash Distance Distribution ({distance_metric})")
    plt.xlabel(distance_metric.upper())
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{hash_method}_{distance_metric}_distance_histogram.png", dpi=300)
    plt.show()


def run_knn(hash_vectors, labels, distance_metric, random_seed, knn_k):
    print("\nRunning kNN classification on hash vectors...")

    if distance_metric == "hamming":
        features = prepare_for_hamming(hash_vectors)
    else:
        features = hash_vectors

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels,
    )

    knn = KNeighborsClassifier(n_neighbors=knn_k, metric=sklearn_knn_metric(distance_metric))
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"kNN accuracy with {distance_metric} distance: {acc:.4f}")


def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    dataset = load_dataset(args.image_size)
    print(f"Dataset size: {len(dataset)}")

    sample_count = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_count, replace=False)

    hash_vectors, labels = compute_hashes(dataset, indices, args.hash_method)
    print("Hash vectors shape:", hash_vectors.shape)

    embeddings_2d = reduce_to_2d(
        hash_vectors,
        use_tsne=args.use_tsne,
        random_seed=args.random_seed,
        pca_components=args.pca_components,
        perplexity=args.perplexity,
    )

    plot_embedding(embeddings_2d, labels, args.hash_method, args.use_tsne)
    plt.savefig(f"results/{args.hash_method}_{args.distance_metric}_{'tsne' if args.use_tsne else 'pca'}_visualization.png", dpi=300)
    plt.show()

    compute_distance_analysis(
        hash_vectors,
        labels,
        distance_metric=args.distance_metric,
        hash_method=args.hash_method,
        max_samples=args.max_distance_samples,
    )

    run_knn(hash_vectors, labels, args.distance_metric, args.random_seed, args.knn_k)


if __name__ == "__main__":
    main()
