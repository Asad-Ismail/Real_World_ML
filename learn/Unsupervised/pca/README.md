# PCA From First Principles

PCA reduces dimensionality by finding orthogonal directions that capture the most variance in the data.

## Core Steps

1. Center the data:
   `X_centered = X - mean(X)`
2. Compute the covariance matrix:
   `C = cov(X_centered)`
3. Find eigenvalues and eigenvectors of `C`
4. Keep the top `k` eigenvectors
5. Project the data onto those directions:
   `Z = X_centered @ V_k`
6. Reconstruct approximately:
   `X_reconstructed = Z @ V_k.T + mean(X)`

## What This Script Teaches

- Why centering matters
- How covariance captures correlated variation
- Why eigenvectors define principal directions
- How reconstruction quality changes as you keep more components

## Run It

```bash
python learn/Unsupervised/pca/pca.py
```

The script uses the built-in scikit-learn digits dataset, so it does not require an extra `mnist` package.

It will save:

- `learn/Unsupervised/pca/results/recons.png`
- `learn/Unsupervised/pca/results/variance.png`

## Questions To Answer Yourself

- Why does PCA need centered data?
- Why are the principal directions orthogonal?
- What information is lost when reconstruction is imperfect?
- When would PCA fail to capture useful structure?
