# Unsupervised Learning Study Path

This section is for methods that discover structure without target labels. The best first-principles order in this repository is:

1. `kmeans/`
2. `pca/`
3. `autoencoder/`
4. `t-SNE/`

## Recommended Reading Order

- `kmeans/`: Learn clustering through centroid updates and distance-based assignment.
- `pca/`: Learn linear dimensionality reduction through covariance and eigenvectors.
- `autoencoder/`: Compare neural compression against PCA.
- `t-SNE/`: Treat this as a visualization tool, not a general-purpose feature extractor.

## Verified Starter Commands

- `python learn/Unsupervised/kmeans/kmeans.py`
- `python learn/Unsupervised/pca/pca.py`
- `python learn/Unsupervised/autoencoder/autoencoder.py`
- `python learn/Unsupervised/t-SNE/tsne.py`

## How To Study These

1. Identify what is being optimized or preserved.
2. Separate the representation step from the visualization step.
3. Check what assumptions the method makes about geometry or variance.
4. Inspect the saved figures in each folder's `results/` directory.
