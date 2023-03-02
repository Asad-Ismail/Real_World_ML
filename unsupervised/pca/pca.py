import numpy as np


class PCA:
    def __init__(self,compenents):
        self.compenents=components

    @staticmethod
    def standardize(X):
        X = (X - self.mean) / self.std
        return X


    def train(X):
        # calculate mean and variance
        self.mean=np.mean(X, axis=0)
        self.std=np.std(X, axis=0)
        # standardize the data
        X = standardize(X)
        # compute the covariance matrix
        cov_matrix = np.cov(X.T)
        
        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:,sorted_indices]
        
        # select the top n components
        top_n_eigenvectors = sorted_eigenvectors[:,:self.compenents]
        
        self.top_n_eigenvectors=top_n_eigenvectors

    def pred(X):
        # Standardize data
        X=standardize(X)
        # transform the data into the new coordinate system
        transformed_data = np.dot(X, slef.top_n_eigenvectors)
        return transformed_data


    def pca_inverse_transform(self,transformed_data):
        """Perform PCA inverse transform on transformed data."""
        
        # Multiply transformed data by transpose of PCA component matrix
        inverse_transformed_data = np.dot(transformed_data, self.slef.top_n_eigenvectors.T)
        
        # Add mean vector to shift back to original scale
        original_data = inverse_transformed_data + self.mean
        
        return original_data


if __name__ =="__main__":
    


