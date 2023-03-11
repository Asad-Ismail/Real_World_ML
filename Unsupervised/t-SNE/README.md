# t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that is commonly used for visualizing high-dimensional data.

t-SNE works by first calculating a probability distribution over pairs of high-dimensional objects in such a way that similar objects have a higher probability of being chosen than dissimilar objects. Then, it constructs a similar probability distribution over pairs of low-dimensional objects, and it minimizes the Kullback-Leibler divergence between the two distributions. This process results in a low-dimensional mapping of the high-dimensional data points that preserves the underlying structure of the data.

One of the main benefits of t-SNE is its ability to capture complex non-linear structures in high-dimensional data, making it useful in a variety of applications such as image analysis, natural language processing, and bioinformatics. However, it is important to note that t-SNE is a computationally intensive algorithm and can be sensitive to its hyperparameters.
