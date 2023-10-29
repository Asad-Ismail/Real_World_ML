## Important Probability Concepts

# Bayesian Statistics: 📊

Bayesian statistics is a branch of statistics based on the Bayesian probability theorem. It provides a framework to update the probabilities of hypotheses as new evidence emerges.

## 📜 Table of Contents
- [Key Components of Bayesian Statistics](#key-components)
- [Why Do We Need the Posterior?](#why-the-posterior)
- [Bayesian vs. Frequentist Approaches](#bayesian-vs-frequentist)
- [Python Code Example: Bayesian Update](#python-example)

<a name="key-components"></a>
## Key Components of Bayesian Statistics

1. **Prior (𝑃(𝜃))**: 
   - Your beliefs about the parameter before seeing the data.
2. **Likelihood (𝑃(𝐷|𝜃))**: 
   - How likely the observed data is under various assumptions.
3. **Posterior (𝑃(𝜃|𝐷))**: 
   - Updated beliefs after seeing the data. Calculated using Bayes' theorem.

<a name="why-the-posterior"></a>
## Why Do We Need the Posterior?

The posterior gives us a revised view of the probability distribution of a parameter after considering new evidence (data). By using the prior and the likelihood of the observed data, the posterior provides a complete updated belief about the parameter.

<a name="bayesian-vs-frequentist"></a>
## Bayesian vs. Frequentist Approaches

- **Bayesian**: Incorporates prior knowledge or beliefs. Updates beliefs with new data.
- **Frequentist**: Does not use prior. Relies on long-term frequencies.

<a name="python-example"></a>
## Python Code Example: Bayesian Update

```python
def bayesian_update(prior, likelihood):
    posterior = prior * likelihood
    normalization = sum(posterior)
    return [p / normalization for p in posterior]
```




Confidence Intervals: A common misconception is that a 95% confidence interval means there's a 95% probability that the true parameter lies within the interval. However, this interpretation is incorrect. Instead, if we repeatedly sampled data and computed the 95% CI for each dataset, about 95% of such intervals would contain the true parameter.
Mnemonics/Cues:

Hessian Matrix: Think "2nd order" since it deals with second-order partial derivatives.
Fisher Information Matrix: Remember "how much information about θ".
Logistic Regression: Link "logistic" with "binary outcomes" (0 or 1).
Exponential Family: Many common distributions are part of this family.
Confidence Intervals: It's about "repeated sampling", not the probability of a single interval.
