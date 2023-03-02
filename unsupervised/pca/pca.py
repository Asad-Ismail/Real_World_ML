import numpy as np

def pca(X, n_components):
    # standardize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # compute the covariance matrix
    cov_matrix = np.cov(X.T)
    
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:,sorted_indices]
    
    # select the top n components
    top_n_eigenvectors = sorted_eigenvectors[:,:n_components]
    
    # transform the data into the new coordinate system
    transformed_data = np.dot(X, top_n_eigenvectors)
    
    return transformed_data



def pca_inverse_transform(transformed_data, pca_components, mean_vector):
    """Perform PCA inverse transform on transformed data."""
    
    # Multiply transformed data by transpose of PCA component matrix
    inverse_transformed_data = np.dot(transformed_data, pca_components.T)
    
    # Add mean vector to shift back to original scale
    original_data = inverse_transformed_data + mean_vector
    
    return original_data
