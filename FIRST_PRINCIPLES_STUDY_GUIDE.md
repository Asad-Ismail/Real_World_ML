# First Principles Study Guide

This guide is for learners who want to understand machine learning from the ground up before diving into large frameworks, production systems, or advanced notebooks.

## How To Study This Repository

For each algorithm:

1. Write down the learning objective in plain language.
2. Identify the model parameters and what they mean.
3. Trace the `fit()` method line by line.
4. Trace the `predict()` method line by line.
5. Run the script and inspect the saved plot in the local `results/` folder.
6. Change one hyperparameter and explain what changed.

Do not try to cover the entire repository in one pass. Start with the simple supervised implementations first.

## Recommended Order

### 1. Linear Regression
- Goal: Understand parameters, prediction as `Xw + b`, and fitting with a closed-form solution.
- Files:
  - `learn/Supervised/LinearRegression/linearregression.py`
  - `learn/Supervised/LinearRegression/README.md`
- Run:
  - `python learn/Supervised/LinearRegression/linearregression.py`

### 2. Logistic Regression
- Goal: Move from regression to classification using the sigmoid function and gradient descent.
- Files:
  - `learn/Supervised/LogisticRegression/logisticregression.py`
  - `learn/Supervised/LogisticRegression/README.md`
- Run:
  - `python learn/Supervised/LogisticRegression/logisticregression.py`

### 3. Naive Bayes
- Goal: Learn probabilistic classification, class priors, Gaussian likelihoods, and why log-probabilities matter.
- Files:
  - `learn/Supervised/NaiveBayes/naive_bayes.py`
  - `learn/Supervised/NaiveBayes/README.md`
- Run:
  - `python learn/Supervised/NaiveBayes/naive_bayes.py`

### 4. K-Nearest Neighbors
- Goal: Understand lazy learning, distance-based reasoning, and why no real parameter fitting happens.
- Files:
  - `learn/Supervised/knn/knn.py`
  - `learn/Supervised/knn/README.md`
- Run:
  - `python learn/Supervised/knn/knn.py`

### 5. Decision Trees
- Goal: Learn recursive partitioning, entropy, information gain, and stopping conditions.
- Files:
  - `learn/Supervised/DecisionTree/decisiontree.py`
  - `learn/Supervised/DecisionTree/README.md`
- Run:
  - `python learn/Supervised/DecisionTree/decisiontree.py`

### 6. Support Vector Machines
- Goal: Understand margins, hinge loss, and the tradeoff between fit and regularization.
- Files:
  - `learn/Supervised/svm/svm.py`
  - `learn/Supervised/svm/README.md`
- Run:
  - `python learn/Supervised/svm/svm.py`

### 7. Ensemble Methods
- Goal: Learn how stronger models are built from simpler trees.
- Files:
  - `learn/Supervised/ensemble/random_forest.py`
  - `learn/Supervised/ensemble/gbm.py`
  - `learn/Supervised/ensemble/test_gbm.py`
- Run:
  - `python learn/Supervised/ensemble/test_gbm.py`

### 8. K-Means
- Goal: Learn centroid-based clustering, inertia, and the effect of initialization.
- Files:
  - `learn/Unsupervised/kmeans/kmeans.py`
  - `learn/Unsupervised/kmeans/README.md`
- Run:
  - `python learn/Unsupervised/kmeans/kmeans.py`

### 9. PCA
- Goal: Learn linear dimensionality reduction through covariance, eigenvectors, and reconstruction.
- Files:
  - `learn/Unsupervised/pca/pca.py`
  - `learn/Unsupervised/pca/README.md`
- Run:
  - `python learn/Unsupervised/pca/pca.py`

### 10. Q-Learning And SARSA
- Goal: Learn the Bellman update, bootstrapping, and the difference between off-policy and on-policy control.
- Files:
  - `learn/Reinforcement_Learning/envs/grid_env.py`
  - `learn/Reinforcement_Learning/TD/qlearning.py`
  - `learn/Reinforcement_Learning/TD/sarsa.py`
  - `learn/Reinforcement_Learning/TD/README.md`
- Run:
  - `python learn/Reinforcement_Learning/TD/qlearning.py`
  - `python learn/Reinforcement_Learning/TD/sarsa.py`

### 11. REINFORCE And A2C
- Goal: Move from value-based RL to policy gradients and actor-critic updates.
- Files:
  - `learn/Reinforcement_Learning/policygrad/reinforce.py`
  - `learn/Reinforcement_Learning/policygrad/actorcritic/a2c.py`
  - `learn/Reinforcement_Learning/policygrad/README.md`
- Run:
  - `python learn/Reinforcement_Learning/policygrad/reinforce.py`
  - `python learn/Reinforcement_Learning/policygrad/actorcritic/a2c.py`

### 12. Move From Algorithms To Applied Examples
- Goal: See how the ideas above show up in more realistic workflows without jumping straight into cloud notebooks.
- Files:
  - `Use_Cases/README.md`
  - `Use_Cases/learning_with_less/README.md`
  - `Use_Cases/SparkImageProcessing/README.md`
  - `Use_Cases/RealTimeDataProcessing/README.md`
- Run:
  - `python Use_Cases/learning_with_less/run.py --dataset_source digits --mode supervised --epochs 2`
  - `python Use_Cases/SparkImageProcessing/convert_to_gray_opencv.py`

## What To Ignore At First

- Large notebooks in `learn/NLP/` and cloud-specific notebook folders under `Use_Cases/`
- Infrastructure-heavy workflows you cannot run locally yet
- Kafka, Spark, SageMaker, and external-API demos until you finish the beginner path
- Agent demos and DSPy scripts that require external keys

Those are useful later, but they are not the right starting point if your goal is intuition. When you are ready for applied examples, start with `Use_Cases/learning_with_less/` and `Use_Cases/SparkImageProcessing/` before the multi-service workflows.

## What To Ask Yourself While Reading Code

- What quantity is being optimized?
- Where is the model storing what it learned?
- Which assumptions does the model make about the data?
- What happens if the assumptions are violated?
- Which part of the code is math, and which part is engineering scaffolding?

## Good First Extensions

- Add train/test splits to the foundational scripts.
- Print a loss curve for logistic regression.
- Add Gini impurity as an alternative split criterion for the tree.
- Compare Naive Bayes and logistic regression on the same dataset.
- Add feature scaling before running KNN or SVM and compare the boundary.

## Practical Note

This repository mixes polished learning examples with exploratory work and production-oriented material. The beginner path above is the safest place to start if your goal is understanding before scale, and the curated `Use_Cases/README.md` is the next step after that.
