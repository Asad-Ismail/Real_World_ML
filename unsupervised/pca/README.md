## PCA (Princial Componenet Analysis)

PCA (Principal Component Analysis) is a technique used to reduce the dimensionality of data while retaining as much information as possible. The forward and reverse steps of PCA can be expressed mathematically as follows:

Forward Step:
Data centering: Subtract the mean of each variable from the corresponding values to obtain centered data.

X_centered = X - X̄

Compute the covariance matrix: Calculate the covariance matrix of the centered data. The covariance matrix measures how much two variables vary together.

C = (1 / (n-1)) X_centered.T @ X_centered

Compute eigenvectors and eigenvalues of the covariance matrix: Find the eigenvectors and eigenvalues of the covariance matrix. Eigenvectors represent the directions in which the data varies the most, and eigenvalues represent the amount of variance explained by each eigenvector.

C @ v_i = λ_i v_i

Select the principal components: Choose the top k eigenvectors with the largest eigenvalues. These eigenvectors represent the principal components of the data.

V_k = [v_1, v_2, ..., v_k]

Transform the data: Project the centered data onto the k principal components to obtain the transformed data.

Y = X_centered @ V_k

Reverse Step:
Compute the inverse transformation: Multiply the transformed data by the transpose of the principal component matrix to obtain the reconstructed data.

X_reconstructed = Y @ V_k.T

Add back the mean: Add the mean of each variable back to the reconstructed data to obtain the final reconstructed data.

X_final = X_reconstructed + X̄