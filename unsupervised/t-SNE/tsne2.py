import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

EPSILON = 1e-12

def get_data():
    # only to get sample dataset
    import mnist   
    train_imgs=mnist.train_images()
    test_imgs= mnist.test_images()
    train_lbls=mnist.train_labels()
    test_lbls= mnist.test_labels()
    #print(f"Train and test shape are {train_imgs.shape},{test_imgs.shape}")
    return train_imgs,train_lbls,test_imgs,test_lbls


def squared_dist_mat(X):
    """calculates the squared eucledian distance matrix
    function source: https://lvdmaaten.github.io/tsne/
    Parameters:
    X : ndarray of shape (n_samples, n_features)
    Returns:
    D: Squared eucledian distance matrix of shape (n_samples, n_samples)
    """
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return D


def pairwise_affinities(data, sigmas, dist_mat):
    """calculates the pairwise affinities p_{j|i} using the given values of sigma
    Parameters:
    data : ndarray of shape (n_samples, n_features)
    sigmas : column array of shape (n_samples, 1)
    dist_mat : data distance matrix; ndarray of shape (n_samples, n_samples)
    Returns:
    P: pairwise affinity matrix of size (n_samples, n_samples)
    """
    assert sigmas.shape == (data.shape[0], 1)
    inner = (-dist_mat) / (2 * (sigmas ** 2))
    numers = np.exp(inner)
    denoms = np.sum(numers, axis=1) - np.diag(numers)
    denoms = denoms.reshape(-1, 1)
    denoms += EPSILON  # Avoid div/0
    P = numers / denoms
    np.fill_diagonal(P, 0.0)
    return P


def get_entropies(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P
    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)
    Returns:
    array-like row-wise Shannon entropy of shape (n_samples,)
    """
    asym_affinities = np.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    return -np.sum(asym_affinities * np.log2(asym_affinities), axis=1)


def get_perplexities(asym_affinities):
    """
    compute perplexities of pairwise affinity matrix P
    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)
    Returns:
    array-like row-wise perplexities of shape (n_samples,)
    """
    return 2 ** get_entropies(asym_affinities)


def all_sym_affinities(data, perp, tol, attempts=100):
    """
    finds the data specific sigma values and calculates the symmetric affinities matrix P
    Parameters:
    data : ndarray of shape (n_samples, n_features)
    perp : float, cost function parameter
    tol : float, tolerance of how close the current perplexity is to the target perplexity
    attempts : int, a maximum limit to the binary search attempts
    Returns:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)
    """
    dist_mat = squared_dist_mat(data)  # mxm

    sigma_maxs = np.full(data.shape[0], 1e12)

    # zero here causes div/0, /2sigma**2 in P calc
    sigma_mins = np.full(data.shape[0], 1e-12)

    current_perps = np.full(data.shape[0], np.inf)

    while (not np.allclose(current_perps, perp, atol=tol)) and attempts > 0:
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat)
        current_perps = get_perplexities(P)
        attempts -= 1
        for i in range(len(current_perps)):
            current_perp = current_perps[i]
            if current_perp > perp:
                sigma_maxs[i] = sigmas[i]
            elif current_perp < perp:
                sigma_mins[i] = sigmas[i]

    if attempts == 0:
        print(
            "Warning: Ran out attempts before converging, try a different perplexity?"
        )
    P = (P + P.T) / (2 * data.shape[0])
    return P


def low_dim_affinities(Y, Y_dist_mat):
    """
    computes the low dimensional affinities matrix Q
    Parameters:
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)
    Returns:
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)
    """
    numers = (1 + Y_dist_mat) ** (-1)
    denom = np.sum(numers) - np.sum(np.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    np.fill_diagonal(Q, 0.0)
    return Q


def compute_grad(P, Q, Y, Y_dist_mat):
    """
    computes the gradient vector needed to update the Y values
    Parameters:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)
    Returns:
    grad: the gradient vector, shape (n_samples, n_components)
    """
    Ydiff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    pq_factor = (P - Q)[:, :, np.newaxis]
    dist_factor = ((1 + Y_dist_mat) ** (-1))[:, :, np.newaxis]
    return np.sum(4 * pq_factor * Ydiff * dist_factor, axis=1)


def momentum_func(t):
    """returns optimization parameter
    Parameters:
    t: integer, iteration number
    Returns:
    float representing the momentum term added to the gradient
    """
    if t < 250:
        return 0.5
    else:
        return 0.8


def tsne(
    data,
    n_components,
    perp,
    n_iter,
    lr,
    momentum_fn,
    perp_tol=1e-8,
    early_exaggeration=4.0,
    pbar=False,
    random_state=None,
):
    """calculates the pairwise affinities p_{j|i} using the given values of sigma
    Parameters:
    data : ndarray of shape (n_samples, n_features)
    n_components : int, target number of dimensions
    perp : float, cost function parameter
    n_iter : number of iterations to run, recommended to be no less than 250
    lr : learning rate
    momentum_fn : function that controls the momentum term
    perp_tol : float, tolerance of how close the current perplexity is to the target perplexity
    early_exaggeration : optimization parameter
    pbar : flag to show tqdm progress bar during iterations
    random_state : determines the random number generator, set for reproducible results
    Returns:
    Y: low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    """
    rand = np.random.RandomState(random_state)
    P = all_sym_affinities(data, perp, perp_tol) * early_exaggeration
    P = np.clip(P, EPSILON, None)

    init_mean = np.zeros(n_components)
    init_cov = np.identity(n_components) * 1e-4

    Y = rand.multivariate_normal(mean=init_mean, cov=init_cov, size=data.shape[0])

    Y_old = np.zeros_like(Y)
    iter_range = range(n_iter)
    if pbar:
        iter_range = tqdm(iter_range, "Iterations")
    for t in iter_range:
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities(Y, Y_dist_mat)
        Q = np.clip(Q, EPSILON, None)
        grad = compute_grad(P, Q, Y, Y_dist_mat)
        Y = Y - lr * grad + momentum_fn(t) * (Y - Y_old)
        Y_old = Y.copy()
        Y_old = Y.copy()
        if t == 100:
            P = P / early_exaggeration
            pass
        pass

    return Y

if __name__=="__main__":
    #train_imgs,train_lbls,test_imgs,test_lbls=get_data()
    #print(train_imgs.shape)
    #get first n 
    #train_imgs=train_imgs[:2000,...].reshape(-1,784)
    #train_lbls=train_lbls[:2000,...]
    #print(f"Data min and max before normalization is {train_imgs.min()},{train_imgs.max()}")
    ##train_imgs=normalize_data(train_imgs)
    #print(f"Data min and max after normalization is {train_imgs.min()},{train_imgs.max()}")
    from sklearn.datasets import load_digits
    train_imgs, train_lbls = load_digits(return_X_y=True)
    unique_values, unique_counts = np.unique(train_lbls, return_counts=True)
    for value, count in zip(unique_values, unique_counts):
        print(f"Label {value} has count: {count}")
    y=tsne(train_imgs, 2, 30, 1000, lr=100, momentum_fn=momentum_func,pbar=True,random_state=42)
    np.save("results/tsne.np",y)
    plt.scatter(y[:,0], y[:,1],c=train_lbls)
    plt.savefig("results/clusters.png")