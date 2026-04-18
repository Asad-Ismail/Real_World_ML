# Supervised Learning Study Path

This folder contains the most approachable from-scratch ML code in the repository. If your goal is first-principles understanding, start here before moving into advanced notebooks or production use cases.

## Recommended Order

1. `LinearRegression/`
2. `LogisticRegression/`
3. `NaiveBayes/`
4. `knn/`
5. `DecisionTree/`
6. `svm/`
7. `ensemble/`

## What Each Folder Teaches

- `LinearRegression/`: Linear models, parameters, and closed-form fitting.
- `LogisticRegression/`: Probabilistic binary classification with gradient descent.
- `NaiveBayes/`: Bayes rule, likelihoods, priors, and conditional independence assumptions.
- `knn/`: Distance-based classification without a real training phase.
- `DecisionTree/`: Recursive splitting with entropy and information gain.
- `svm/`: Margins, hinge loss, and regularization.
- `ensemble/`: How multiple trees are combined in random forests and boosting.

## Suggested Workflow

1. Read the matching README for the algorithm.
2. Skim `fit()` and `predict()` before running anything.
3. Run the script from the repository root.
4. Inspect the figure saved under that algorithm's `results/` folder.
5. Change one hyperparameter and explain the effect in words.

## Verified Starter Commands

- `python learn/Supervised/LinearRegression/linearregression.py`
- `python learn/Supervised/LogisticRegression/logisticregression.py`
- `python learn/Supervised/NaiveBayes/naive_bayes.py`
- `python learn/Supervised/knn/knn.py`
- `python learn/Supervised/DecisionTree/decisiontree.py`
- `python learn/Supervised/svm/svm.py`
- `python learn/Supervised/ensemble/test_gbm.py`

For a repository-wide beginner roadmap, go back to [FIRST_PRINCIPLES_STUDY_GUIDE.md](../../FIRST_PRINCIPLES_STUDY_GUIDE.md).
