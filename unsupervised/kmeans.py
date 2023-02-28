import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=2, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        self.centroids = {}
        
        # Initialize centroids randomly
        for i in range(self.n_clusters):
            self.centroids[i] = X[np.random.choice(X.shape[0], 1)][0]
        
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
            if converged:
                break
                
    def predict(self, X):
        y_pred = []
        for x in X:
            distances = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
            classification = distances.index(min(distances))
            y_pred.append(classification)
        return y_pred


if __name__=="__main__":
    ## Test on 2D dummy dataset
    import matplotlib.pyplot as plt

    # Generate some random data
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    # Cluster the data
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    # Plot the clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color=colors[y_pred[i]])
    plt.scatter([kmeans.centroids[c][0] for c in kmeans.centroids], [kmeans.centroids[c][1] for c in kmeans.centroids], marker='x', color='black')
    plt.show()
