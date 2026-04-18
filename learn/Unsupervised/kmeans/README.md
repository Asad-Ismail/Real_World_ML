# K-Means From First Principles

K-Means is one of the simplest clustering algorithms in machine learning.

## Core Loop

1. Pick `k` initial centroids
2. Assign each point to its nearest centroid
3. Recompute each centroid as the mean of its assigned points
4. Repeat until assignments stop changing or the centroids move very little

## What This Implementation Teaches

- Distance-based assignment
- Centroid updates as averages
- Why initialization matters
- How inertia and silhouette scores help choose `k`

## Run It

```bash
python learn/Unsupervised/kmeans/kmeans.py
```

It will save:

- `learn/Unsupervised/kmeans/results/cluster.png`
- `learn/Unsupervised/kmeans/results/elbow.png`
- `learn/Unsupervised/kmeans/results/silhouette.png`

## Questions To Answer Yourself

- Why does K-Means fail on non-spherical clusters?
- Why can different initial centroids lead to different answers?
- What is the difference between inertia and silhouette score?
