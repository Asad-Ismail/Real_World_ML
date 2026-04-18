from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        centroid_indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[centroid_indices].copy()

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = self.centroids.copy()
            for cluster_index in range(self.n_clusters):
                cluster_points = X[labels == cluster_index]
                if cluster_points.size == 0:
                    new_centroids[cluster_index] = X[rng.integers(0, X.shape[0])]
                else:
                    new_centroids[cluster_index] = np.mean(cluster_points, axis=0)

            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if centroid_shift <= self.tol:
                break

        self.labels_ = self.predict(X)
        self.inertia_ = float(np.sum((X - self.centroids[self.labels_]) ** 2))
        return self

    def predict(self, X):
        if not hasattr(self, "centroids"):
            raise ValueError("You must call `fit` before `predict`.")

        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


def get_data(random_state=42):
    rng = np.random.default_rng(random_state)
    n_points = 100
    cluster1 = rng.normal(size=(n_points, 2)) + np.array([-2, 2])
    cluster2 = rng.normal(size=(n_points, 2)) + np.array([2, 2])
    cluster3 = rng.normal(size=(n_points, 2)) + np.array([2, -2])
    cluster4 = rng.normal(size=(n_points, 2)) + np.array([-2, -2])
    return np.vstack([cluster1, cluster2, cluster3, cluster4])


def get_scipy_data():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=1000, centers=8, n_features=2, random_state=42)
    return X


def pairwise_distances(X):
    n = X.shape[0]
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def silhouette_score(X, labels):
    labels = np.asarray(labels)
    n = X.shape[0]
    distance_matrix = pairwise_distances(X)
    s = np.zeros(n, dtype=float)
    eps = 1e-8

    for i in range(n):
        cluster_indices = np.where(labels == labels[i])[0]
        cohesion = np.sum(distance_matrix[i, cluster_indices]) / max(len(cluster_indices) - 1, 1)

        separation = np.inf
        for other_label in np.unique(labels):
            if other_label == labels[i]:
                continue
            other_cluster_indices = np.where(labels == other_label)[0]
            d = np.sum(distance_matrix[i, other_cluster_indices]) / (len(other_cluster_indices) + eps)
            separation = min(separation, d)

        if separation == np.inf:
            s[i] = 0.0
        else:
            s[i] = (separation - cohesion) / (max(separation, cohesion) + eps)

    return float(np.mean(s))


def get_elbow_k(X, max_k=10, save_path=None):
    save_path = Path(save_path or RESULTS_DIR / "elbow.png")
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    if len(inertias) >= 3:
        curvature = np.abs(np.diff(inertias, n=2))
        optimal_k = int(np.argmax(curvature) + 2)
    else:
        optimal_k = 1

    fig, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), inertias, marker="o")
    ax.plot(optimal_k, inertias[optimal_k - 1], marker="o", markersize=10, color="red")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Approximate elbow K: {optimal_k}")
    return optimal_k, inertias


def run_silhouette_method(X, max_k=10, save_path=None):
    save_path = Path(save_path or RESULTS_DIR / "silhouette.png")
    candidate_ks = list(range(2, max_k + 1))
    scores = []

    for k in candidate_ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        scores.append(silhouette_score(X, labels))

    optimal_k = candidate_ks[int(np.argmax(scores))]

    fig, ax = plt.subplots()
    ax.plot(candidate_ks, scores, marker="o")
    ax.plot(optimal_k, max(scores), marker="o", markersize=10, color="red")
    ax.set_title("Silhouette Method")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Silhouette scores: {scores}")
    print(f"Best silhouette K: {optimal_k}")
    return optimal_k, scores


def run_k_means(X, numberclusters=4, save_path=None):
    save_path = Path(save_path or RESULTS_DIR / "cluster.png")
    kmeans = KMeans(n_clusters=numberclusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.8)
    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker="x", color="black", s=120)
    ax.set_title(f"K-Means with K={numberclusters}")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"K-Means inertia for K={numberclusters}: {kmeans.inertia_:.3f}")
    return kmeans


if __name__ == "__main__":
    ensure_results_dir()
    X = get_data()

    run_k_means(X, numberclusters=4)
    get_elbow_k(X)
    run_silhouette_method(X)
