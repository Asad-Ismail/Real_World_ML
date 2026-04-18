In simple linear regression, we can derive a closed-form solution for the coefficients (slope and intercept) using ordinary least squares (OLS) method. However, logistic regression uses a non-linear sigmoid function to model the probability of the outcome variable. The maximum likelihood estimation method is commonly used to estimate the coefficients in logistic regression. This method does not have a closed-form solution, and instead, numerical optimization techniques like gradient descent are used to find the optimal coefficients.
# Logistic Regression From First Principles

This implementation is a compact binary classifier built with NumPy. It is a good first example of gradient-based learning because the full training loop is small enough to understand in one sitting.

## Core Idea

Logistic regression does not predict a class directly. It predicts a score:

`z = Xw + b`

That score is converted into a probability with the sigmoid function:

`sigma(z) = 1 / (1 + exp(-z))`

If the probability is above `0.5`, we predict class `1`. Otherwise we predict class `0`.

## What The Code Is Teaching

- How model parameters are stored in `weights` and `bias`
- How forward propagation computes probabilities
- How gradients are computed for each parameter
- How gradient descent updates the parameters repeatedly

## Read The Code In This Order

1. `sigmoid()`
2. `fit()`
3. `predict()`
4. The `__main__` block that generates data and saves the plots

## Run It

```bash
python learn/Supervised/LogisticRegression/logisticregression.py
```

This will save:

- `learn/Supervised/LogisticRegression/results/data.png`
- `learn/Supervised/LogisticRegression/results/logistic_regression.png`

## Questions To Answer Yourself

- Why do we need a sigmoid instead of using `Xw + b` directly?
- Why does the model need multiple iterations?
- What happens if the learning rate is too large?
- Why is the decision boundary linear?

## Good Next Steps

- Print the loss every 100 iterations.
- Add a train/test split.
- Compare the classifier against Naive Bayes on the same dataset.
