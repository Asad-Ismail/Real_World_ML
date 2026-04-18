# Linear Regression From First Principles

This is the cleanest place to begin if you are brand new to ML. The implementation fits a linear model with a closed-form solution, so you can focus on the relationship between parameters, predictions, and error before moving to iterative optimization.

## Core Idea

Linear regression predicts a continuous value using:

`y_hat = Xw + b`

This implementation uses a ridge-style closed-form solution, which means it solves for the best parameters directly with linear algebra instead of gradient descent.

## What The Code Is Teaching

- How to represent the intercept with a column of ones
- How weights are solved from the data matrix
- Why regularization changes the fitted coefficients
- How prediction is just matrix multiplication after fitting

## Run It

```bash
python learn/Supervised/LinearRegression/linearregression.py
```

This will save:

- `learn/Supervised/LinearRegression/results/data.png`
- `learn/Supervised/LinearRegression/results/linear_regression.png`

## Questions To Answer Yourself

- Why do we prepend a column of ones to `X`?
- Why is the intercept not regularized?
- What happens if you increase `alpha` a lot?
- How is this different from logistic regression?

## Good Next Steps

- Add mean squared error explicitly as a function.
- Compare `alpha=0` against a large regularization value.
- Rewrite the same model with gradient descent as an exercise.
