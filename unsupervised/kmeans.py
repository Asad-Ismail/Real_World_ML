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

def get_elbow_k(X):
    # Calculate the within-cluster sum of squares (WCSS) for different values of K
    wcss = []
    maxk=10
    for k in range(1, maxk):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.wccs)
    # Find the "elbow" point in the WCSS plot
    distances = []
    for i in range(1, len(wcss)):
        distance = wcss[i-1] - wcss[i]
        distances.append(distance)
    elbow_point = np.argmax(distances) + 1
    optimal_k = elbow_point + 1
    print(f"Optimal K are {optimal_k}")
    plt.figure()
    plt.plot(range(1, maxk), wcss)
    plt.plot(optimal_k, wcss[optimal_k-1], marker='o', markersize=10, color="red")
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    #plt.show()
    plt.savefig('results/elbow.png')


def pairwise_distances(X):
    """
    Calculate the pairwise distance matrix of a matrix X.
    """
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def silhouette_score(X, labels):
    """
    Calculate the silhouette score of a clustering given the data and the labels.
    X: the data matrix, where each row is a data point and each column is a feature
    labels: a vector of cluster assignments, where labels[i] is the cluster index assigned to the ith data point
    The function returns the silhouette score as a float value between -1 and 1. 
    A score of 1 indicates that the clustering is very good, with well-separated clusters, while a score of -1 
    indicates that the clustering is very bad, with poorly separated clusters. A score of 0 indicates that the clusters are overlapping.
    """
    def pairwise_distances(X):
        """
        Calculate the pairwise distance matrix of a matrix X.
        """
        n = X.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

    n = X.shape[0]
    distance_matrix = pairwise_distances(X)
    s = np.zeros(n)
    # To avoid zero division
    eps=1e-5
    for i in range(n):
        # Calculate the cohesion of the ith data point
        cluster_indices = np.where(np.array(labels) == labels[i])[0]
        cohesion = np.sum(distance_matrix[i, cluster_indices]) / (len(cluster_indices) - 1+eps)
        
        # Calculate the separation of the ith data point
        separation = np.inf
        for j in range(n):
            if labels[j] != labels[i]:
                other_cluster_indices = np.where(labels == labels[j])[0]
                d = np.sum(distance_matrix[i, other_cluster_indices]) / (len(other_cluster_indices)+eps)
                separation = min(separation, d)
        #Edge case of one cluster in the dataset
        if separation==np.inf:
            s[i]=0
        else:
            # Calculate the silhouette coefficient of the ith data point
            s[i] = (separation - cohesion) / (max(separation, cohesion)+eps)
    # Calculate the average silhouette score of all data points
    score = np.mean(s)
    return score


def run_silhoutte_method(X):
    # Cluster the data
    silh = []
    maxk=11
    for k in range(1, maxk):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        y_pred = kmeans.predict(X)
        y_pred=np.array(y_pred)
        silh.append(silhouette_score(X,y_pred))
    # Find the "elbow" point in the WCSS plot
    print(f"Sil scores are {silh}")
    optimal_k = silh.index(max(silh))+1
    print(f"Optimal silohutee k is {optimal_k}")


def run_k_means(X,numberclusters=4):
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


if __name__=="__main__":

    # Generate some random data
    X = get_Data()

    # Run basic K means
    run_k_means(X,numberclusters=4)

    ## check Elbow method to check wccs
    get_elbow_k(X)

    run_silhoutte_method(X)
