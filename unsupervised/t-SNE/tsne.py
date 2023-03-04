import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_data():
    # only to get sample dataset
    import mnist   
    train_imgs=mnist.train_images()
    test_imgs= mnist.test_images()
    print(f"Train and test shape are {train_imgs.shape},{test_imgs.shape}")
    return train_imgs,test_imgs

def pairwise_distances(X):
    """
    Compute pairwise distances between all points in X.
    """
    return squareform(pdist(X, metric='euclidean'))


def gaussian_similarity_matrix(distances, sigma):
    """
    Compute the Gaussian similarity matrix from pairwise distances.
    """
    return np.exp(-(distances ** 2) / (2 * (sigma ** 2)))


def compute_joint_probabilities(distances, perplexity):
    """
    Compute joint probabilities between all points in X.
    """
    # Compute Gaussian similarity matrix
    similarities = gaussian_similarity_matrix(distances, sigma=1.0)

    # Compute joint probabilities
    P = similarities / np.sum(similarities)
    P = P + np.transpose(P)
    P = P / (2 * len(distances))

    # Symmetrize P
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)

    return P

def perplexity(P):
    """
    Compute the perplexity of the joint probabilities.
    """
    H = np.log2(P)
    H[H == -np.inf] = 0
    H = -np.sum(P * H)
    return 2 ** H

def compute_gradient(Y, P, Q):
    """
    Compute the gradient of the cost function.
    """
    # Compute pairwise distances in the low-dimensional space
    distances = pairwise_distances(Y)

    # Compute the difference between joint and low-dimensional probabilities
    PQ_diff = P - Q

    # Compute the sum of the difference multiplied by the Gaussian kernel
    summands = np.zeros((Y.shape[0], Y.shape[1]))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            if i != j:
                summands[i, :] += (PQ_diff[i, j] * (Y[i, :] - Y[j, :])) / (1 + distances[i, j] ** 2)
    # Multiply by the perplexity and sum over all pairs
    summands = 4 * summands * perplexity(P)
    gradient = np.sum(summands, axis=0)

    return gradient


def t_SNE(X, n_components=2, perplexity=30, n_iter=1000, learning_rate=200, momentum=0.8):
    """
    Perform t-SNE on the high-dimensional dataset X.
    """
    # Compute pairwise distances between all points in X
    distances = pairwise_distances(X)

    # Compute joint probabilities
    P = compute_joint_probabilities(distances, perplexity)

    # Initialize low-dimensional embeddings randomly
    Y = np.random.randn(X.shape[0], n_components)

    # Initialize momentum
    prev_gradient = 0

    # Perform t-SNE
    for i in range(n_iter):
        # Compute low-dimensional similarities
        Q = gaussian_similarity_matrix(pairwise_distances(Y), sigma=1.0)

        # Compute gradient of cost function
        gradient = compute_gradient(Y, P, Q)

        # Update momentum
        if i == 0:
            velocity = momentum * gradient
        else:
            velocity = momentum * prev_gradient - learning_rate * gradient

        # Update low-dimensional embeddings
        Y += velocity

        # Normalize embeddings
        Y = Y - np.mean(Y, axis=0)

        # Print progress
        if (i + 1) % 50 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {i + 1}/{n_iter}: cost={cost}")

        # Save gradient for next iteration
        prev_gradient = gradient

    return Y

if __name__=="__main__":
    train_imgs,test_imgs=get_data()
    train_imgs=train_imgs.reshape(-1,784)
    t_SNE(train_imgs)
    


