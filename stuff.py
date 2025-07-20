import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# Generate synthetic data
X, y = make_blobs(n_samples=10, centers=3, cluster_std=1.5, random_state=42)
dist_matrix = pairwise_distances(X)

# Classical MDS
mds_classical = manifold.MDS(n_components=2, dissimilarity='precomputed', metric=True, random_state=42)
X_classical = mds_classical.fit_transform(dist_matrix)

# Non-metric MDS (Kruskal)
mds_nonmetric = manifold.MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42)
X_nonmetric = mds_nonmetric.fit_transform(dist_matrix)

# Simulated Sammon Mapping (scaling classical MDS for local distances)
X_sammon = X_classical * (1 / (1 + np.linalg.norm(X_classical, axis=1, keepdims=True)))

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Classical MDS (Metric)', 'Non-metric MDS (Kruskal)', 'Simulated Sammon Mapping']
embeddings = [X_classical, X_nonmetric, X_sammon]

for ax, title, embedding in zip(axs, titles, embeddings):
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10', s=100)
    for i, (x0, y0) in enumerate(embedding):
        ax.text(x0 + 0.02, y0 + 0.02, str(i), fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

plt.tight_layout()
plt.show()
