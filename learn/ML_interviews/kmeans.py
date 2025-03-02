import numpy as np
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:

    # Two steps
    # One is assignment
    # Second is update of centroids
        
    points= np.array(points) # mx2
    centroids = np.array(initial_centroids) # k x2

    for it in range(max_iterations):
        # Assginemnt
        distance = np.sum((points[:,None,:] - centroids[None,:,:])**2,axis=-1) # m x k 
        indices= np.argmin(distance,axis=-1)

        for i in range(k):
            new_centroid_k=np.mean(points[indices==i],axis=0)
            centroids[i]=new_centroid_k
        
    return centroids

points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
k = 2
initial_centroids = [(1, 1), (10, 1)]
max_iterations = 10

out=k_means_clustering(points, k, initial_centroids,max_iterations)
print(out)