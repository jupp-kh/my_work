# ============================================================
# Perceptual Hash Visualization with STL-10 + PDQ + t-SNE
# ============================================================
#
# WHAT THIS SCRIPT DOES:
# ----------------------
# 1. Loads STL-10 dataset
# 2. Computes PDQ perceptual hashes
# 3. Converts hashes to vectors
# 4. Computes t-SNE embeddings
# 5. Visualizes clustering by semantic class
#
# INSTALL:
# --------
# pip install torch torchvision pillow matplotlib scikit-learn pdqhash tqdm
#
# RUN:
# ----
# python pdq_tsne_visualization.py
#
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
#
import compute_photoDNA

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torchvision.datasets import STL10
from torchvision import transforms

import pdqhash


# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 2000          # increase later if needed
IMAGE_SIZE = 224            # upscale for PDQ
USE_TSNE = True             # set False for PCA only
RANDOM_SEED = 42
hash_method = 'pdq' # 'pdq' or 'photoDNA'
np.random.seed(RANDOM_SEED)

# STL-10 classes
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
    "truck"
]


# ============================================================
# LOAD DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])

dataset = STL10(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

print(f"Dataset size: {len(dataset)}")

# Random subset
indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)


# ============================================================
# COMPUTE {hash_method.upper()} HASHES
# ============================================================

hash_vectors = []
labels = []

print(f"Computing {hash_method.upper()} hashes...")

for idx in tqdm(indices):

    img, label = dataset[idx]

    # Convert PIL image -> numpy uint8
    img_np = np.array(img).astype(np.uint8)

    # Compute PDQ hash
    if hash_method == 'pdq':
        hash_vec, quality = pdqhash.compute(img_np)
    else:
        hash_vec, quality = compute_photoDNA.compute(img_np)
  
    # Convert boolean/binary vector -> int vector
    hash_vec = np.array(hash_vec).astype(np.float32)

    hash_vectors.append(hash_vec)
    labels.append(label)

hash_vectors = np.array(hash_vectors)
labels = np.array(labels)

print("Hash vectors shape:", hash_vectors.shape)


# ============================================================
# OPTIONAL STANDARDIZATION
# ============================================================

scaler = StandardScaler()
hash_vectors_scaled = scaler.fit_transform(hash_vectors)


# ============================================================
# PCA REDUCTION (OPTIONAL PREPROCESSING FOR t-SNE)
# ============================================================

print("Running PCA...")

pca = PCA(n_components=50)
hash_pca = pca.fit_transform(hash_vectors_scaled)

print("PCA output shape:", hash_pca.shape)


# ============================================================
# t-SNE
# ============================================================

if USE_TSNE:

    print("Running t-SNE (this may take a while)...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=RANDOM_SEED,
        verbose=1
    )

    embeddings_2d = tsne.fit_transform(hash_pca)

else:

    embeddings_2d = PCA(n_components=2).fit_transform(hash_pca)


# ============================================================
# VISUALIZATION
# ============================================================

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
        label=CLASS_NAMES[class_id]
    )

plt.title(f"t-SNE Visualization of {hash_method.upper()} Hash Embeddings (STL-10)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(markerscale=2)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{hash_method}_tsne_visualization.png", dpi=300)
plt.show()


# ============================================================
# OPTIONAL: DISTANCE ANALYSIS
# ============================================================

print("\nComputing intra/inter-class distances...")

from scipy.spatial.distance import hamming

intra_distances = []
inter_distances = []

N = len(hash_vectors)
print(hash_vectors.shape)
print(hash_vectors.dtype)
print(np.unique(hash_vectors))

for i in range(min(1000, N)):
    for j in range(i + 1, min(1000, N)):

        d = hamming(hash_vectors[i], hash_vectors[j])

        if labels[i] == labels[j]:
            intra_distances.append(d)
        else:
            inter_distances.append(d)

print(f"Mean intra-class distance: {np.mean(intra_distances):.4f}")
print(f"Mean inter-class distance: {np.mean(inter_distances):.4f}")


# ============================================================
# HISTOGRAM VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    intra_distances,
    bins=50,
    alpha=0.6,
    label="Intra-class"
)

plt.hist(
    inter_distances,
    bins=50,
    alpha=0.6,
    label="Inter-class"
)

plt.title(f"{hash_method.upper()} Hash Distance Distribution")
plt.xlabel("Hamming Distance")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(f"{hash_method}_distance_histogram.png", dpi=300)
plt.show()


# ============================================================
# OPTIONAL: kNN CLASSIFICATION
# ============================================================

print("\nRunning kNN classification on hash vectors...")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    hash_vectors,
    labels,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=labels
)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

preds = knn.predict(X_test)

acc = accuracy_score(y_test, preds)

print(f"kNN accuracy on {hash_method.upper()} hashes: {acc:.4f}")
