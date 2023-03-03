import numpy as np
import matplotlib.pyplot as plt

# only to downalod dataset
#pip install mnist

def get_data():
    # only to get sample dataset
    import mnist   
    train_imgs=mnist.train_images()
    test_imgs= mnist.test_images()
    print(f"Train and test shape are {train_imgs.shape},{test_imgs.shape}")
    return train_imgs,test_imgs

class PCA:
    def __init__(self,components):
        self.components=components

    def standardize(self,X,eps=1e-5):
        X = (X - self.mean) 
        #X = (X - self.mean) / (self.std+eps)
        return X


    def fit(self,X):
        # calculate mean and variance
        self.mean=np.mean(X, axis=0)
        self.std=np.std(X, axis=0)
        print(f"Mean and std of data is {X.shape},{self.mean.shape}, {self.std.shape}")
        # standardize the data
        X = self.standardize(X)
        # compute the covariance matrix
        cov_matrix = np.cov(X.T)
        
        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        print(f"Eigen values and vectors shape is {eigenvalues.shape}, {eigenvectors.shape}")
        
        # sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:,sorted_indices]
        
        # select the top n components
        top_n_eigenvectors = sorted_eigenvectors[:,:self.components]
        print(f"Top N Eigen vector shape is {top_n_eigenvectors.shape}")
        self.top_n_eigenvectors=top_n_eigenvectors

    def pred(self,X):
        # Standardize data
        X=self.standardize(X)
        # transform the data into the new coordinate system
        transformed_data = np.dot(X, self.top_n_eigenvectors)
        return transformed_data


    def pca_inverse_transform(self,transformed_data):
        """Perform PCA inverse transform on transformed data."""
        
        # Multiply transformed data by transpose of PCA component matrix
        inverse_transformed_data = np.dot(transformed_data, self.top_n_eigenvectors.T)
        
        # Add mean vector to shift back to original scale
        original_data = inverse_transformed_data + self.mean
        
        return original_data


if __name__ =="__main__":
    train_imgs,test_imgs=get_data()
    plt.imshow(train_imgs[np.random.randint(0,len(train_imgs))])
    img_dim=np.prod(np.array(train_imgs.shape[1:]))
    #Flatten train and test images
    train_imgs=train_imgs.reshape(-1,img_dim)
    test_imgs=train_imgs.reshape(-1,img_dim) 
    # keep 50 pct of componeents of image
    #pct=50
    #n_components=int((pct/100)*(img_dim))
    n_components=361
    print(f"Reduction in Images dimensions are {(n_components/img_dim)*100} %")
    model=PCA(n_components)
    model.fit(train_imgs)
    transformed=model.pred(test_imgs)
    reconstructed=model.pca_inverse_transform(transformed)
    out_dim=int(np.sqrt(img_dim))
    reconstructed=reconstructed.reshape(-1,out_dim,out_dim)
    testidx=np.random.randint(0,len(test_imgs))
    print(f"Test index is {testidx}")
    orgimg=test_imgs[testidx,...].reshape(out_dim,out_dim)
    recons=reconstructed[testidx,...]
    print(f"Test Images and Prediction shape is {test_imgs.shape}, {reconstructed.shape}")
    plt.imshow(np.concatenate((orgimg,recons),axis=1))
    plt.savefig("results/recons.png")



