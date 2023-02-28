import numpy as np
import matplotlib.pyplot as plt



def get_data():
    np.random.seed(42)
    num_observations = 1000

    # Class 1
    x1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.75], [0.75, 1]], size=num_observations)
    y1 = np.zeros(num_observations)

    # Class 2
    x2 = np.random.multivariate_normal(mean=[1, 4], cov=[[1, 0.75], [0.75, 1]], size=num_observations)
    y2 = np.ones(num_observations)

    # Concatenate the data
    X = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    return x,y


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()