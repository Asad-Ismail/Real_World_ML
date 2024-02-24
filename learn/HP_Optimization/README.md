## From scratch implementations of Hyper Parameter optimizatins

We Implemnt three hyper paramter optimizations

1. Random Search
2. Grid Search
3. Baysian Optimization



# Bayesian Optimization with Gaussian Processes

Bayesian optimization is a robust strategy for finding the optimum of complex functions that are costly to evaluate. This method leverages a probabilistic model, often a Gaussian process, to navigate the search for the best solution.

## Simple Implementation of Bayesian Optimization

Here's a straightforward approach to implementing Bayesian optimization with a Gaussian process:

1. **Define the Search Space**: Specify the range of possible values for each parameter you're looking to optimize.

2. **Choose an Acquisition Function**: The acquisition function determines the next point to evaluate. Popular options include Upper Confidence Bound (UCB), Expected Improvement (EI), and Probability of Improvement (PI).

3. **Initialize the Gaussian Process**: Establish the mean and covariance functions for the Gaussian process. Select a kernel function that suits your problem.

4. **Sample Initial Points**: Begin with sampling a handful of points from the search space to provide initial data for the Gaussian process.

5. **Iterate Until Convergence**:
    - Fit the Gaussian process to the existing data points.
    - Utilize the acquisition function to select the next sampling point.
    - Evaluate the function at the new point and incorporate the results into your dataset.

6. **Identify the Optimum**: Once the process converges, return the point with the highest observed function value.

## Bayesian Principles in Optimization

Bayesian optimization adopts a probabilistic view, modeling uncertainty in the objective function through probability distributions. A Gaussian process serves as a Bayesian regression model, treating the objective function as a random function with a prior distribution over possible functions.

The Gaussian process assumes this prior to be a multivariate Gaussian distribution, defined by a mean function and a covariance function. With new data, the prior is updated via Bayes' rule to a posterior distribution, which informs the selection of the next evaluation point by maximizing an acquisition function that balances exploration with exploitation.

To Read about Baysian optimization please check out
https://distill.pub/2020/bayesian-optimization/#FurtherReading