import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=2, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        self.centroids = {}
        
        # Initialize centroids randomly
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.n_clusters):
            self.centroids[i] = centroids[i]

        # Iterate until convergence or max_iter
        for i in range(self.max_iter):
            self.classes = {}
            for j in range(self.n_clusters):
                self.classes[j] = []
                
            # Assign each data point to its nearest centroid
            for x in X:
                distances = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(x)

            # Update the centroids to be the average of the points in each cluster
            prev_centroids = dict(self.centroids)
            for c in self.classes:
                self.centroids[c] = np.average(self.classes[c], axis=0)
            
            # If the centroids don't change, we've converged
            converged = True
            for c in self.centroids:
                if not np.array_equal(self.centroids[c], prev_centroids[c]):
                    converged = False
                    break
            if converged:
                break
        self.wccs=0
        for k in self.classes.keys():
            self.wccs+=np.sqrt(((self.classes[k]-self.centroids[k])**2).sum())
        
                
    def predict(self, X):
        y_pred = []
        for x in X:
            distances = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
            classification = distances.index(min(distances))
            y_pred.append(classification)
        return y_pred

def get_Data():
    # Generate some random data
    # set random seed for reproducibility
    np.random.seed(42)
    # define number of data points for each cluster
    n_points = 100
    # generate random data for each cluster
    cluster1 = np.random.randn(n_points, 2) + np.array([-2, 2])
    cluster2 = np.random.randn(n_points, 2) + np.array([2, 2])
    cluster3 = np.random.randn(n_points, 2) + np.array([2, -2])
    cluster4 = np.random.randn(n_points, 2) + np.array([-2, -2])

    # combine data into one array
    data = np.vstack([cluster1, cluster2, cluster3, cluster4])
    return data

def get_scipy_data():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=8, n_features=2, random_state=42)
    return X

if __name__=="__main__":

    # Generate some random data
    X = get_Data()

    # Cluster the data
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    # Plot the clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color=colors[y_pred[i]])
    plt.scatter([kmeans.centroids[c][0] for c in kmeans.centroids], [kmeans.centroids[c][1] for c in kmeans.centroids], marker='x', color='black')
    #plt.show()
    plt.savefig('results/cluster.png')

    ## check Elbow method to check wccs
