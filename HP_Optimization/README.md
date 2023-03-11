## From scratch implementations of Hyper Parameter optimizatins

We Implemnt three hyper paramter optimizations

1. Random Search
2. Grid Search
3. Baysian Optimization



# Baysian Optimization

Bayesian optimization is a powerful method for optimizing complex functions that are expensive to evaluate. It involves using a probabilistic model, such as a Gaussian process, to guide the search for the optimum.

Here's a simple implementation of Bayesian optimization using a Gaussian process:

Define the search space: Define the range of values for each parameter that you want to optimize.

Choose the acquisition function: The acquisition function is used to decide which point to sample next. Common choices include Upper Confidence Bound (UCB), Expected Improvement (EI), and Probability of Improvement (PI).

Initialize the Gaussian process: Set the mean and covariance functions for the Gaussian process, and choose an appropriate kernel function.

Sample the initial points: Sample a few points from the search space to initialize the Gaussian process.

Loop until convergence: Repeat the following steps until convergence:

a. Fit the Gaussian process to the current data.

b. Use the acquisition function to choose the next point to sample.

c. Sample the function at the chosen point and add it to the data.

Return the best point found: After convergence, return the point with the highest observed value.

Where does the Baysian come into place here

Bayesian optimization is a probabilistic approach to optimization, which means that it uses probability distributions to model the uncertainty in the objective function. The Gaussian process is one such probabilistic model that is commonly used in Bayesian optimization.

The Gaussian process is a type of Bayesian regression model that models the objective function as a random function with a prior distribution over functions. The prior distribution over functions is typically assumed to be a multivariate Gaussian distribution, which is characterized by a mean function and a covariance function.

After observing some data, the prior distribution is updated using Bayes' rule to obtain the posterior distribution over functions. The posterior distribution is also a multivariate Gaussian distribution, with a mean and covariance that depend on the observed data and the prior distribution.

In the context of Bayesian optimization, the observed data consists of the function values at the points that have been evaluated so far. The posterior distribution over functions is used to guide the search for the next point to evaluate, typically by selecting the point that maximizes some acquisition function that trades off exploration and exploitation.



To Read about Baysian optimization please check out
https://distill.pub/2020/bayesian-optimization/#FurtherReading