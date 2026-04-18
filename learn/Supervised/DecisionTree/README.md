# Decision Trees From First Principles

This folder contains a simple decision tree classifier built without external tree libraries. It is one of the best places in the repository to study how a model can learn non-linear decision boundaries with a sequence of simple rules.

## Core Idea

A decision tree repeatedly asks questions of the form:

- Is feature `j` less than or equal to threshold `t`?

Each split tries to make the child nodes more pure than the parent node. In this implementation, purity is measured with entropy and the split score is information gain.

## What The Code Is Teaching

- How recursion builds a tree
- How entropy measures class uncertainty
- How information gain chooses a split
- Why stopping rules are necessary
- How prediction becomes a tree traversal problem

## Read The Code In This Order

1. `Node`
2. `_entropy()`
3. `_information_gain()`
4. `_best_split()`
5. `_grow_tree()`
6. `predict()`

## Run It

```bash
python learn/Supervised/DecisionTree/decisiontree.py
```

This will save:

- `learn/Supervised/DecisionTree/results/data.png`
- `learn/Supervised/DecisionTree/results/decisiontree.png`

## Questions To Answer Yourself

- Why can trees model non-linear boundaries even when each split is simple?
- What happens if the tree keeps growing without a depth limit?
- Why do empty splits need to be rejected?
- How would the behavior change if you used Gini impurity instead of entropy?

## Good Next Steps

- Implement Gini impurity.
- Add a minimum leaf size hyperparameter.
- Compare a shallow tree against a deep tree on noisy data.
