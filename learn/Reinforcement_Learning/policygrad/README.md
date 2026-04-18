# Policy Gradient Notes

This folder starts with REINFORCE, the simplest policy-gradient method in the repository.

## Why REINFORCE Waits Until The End

REINFORCE is a Monte Carlo method. It needs the full return:

`G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...`

That means the update for a time step is only available after the episode finishes.

## Core Update

The policy parameters `theta` are updated in the direction:

`theta <- theta + alpha * G_t * grad(log pi(a_t | s_t, theta))`

In this repository's Gridworld example, the policy is represented with a softmax over actions for each state.

## REINFORCE vs Q-Learning

- REINFORCE directly updates a policy.
- Q-learning learns action values and then acts greedily with respect to them.
- REINFORCE is on-policy and high variance.
- Q-learning is off-policy and bootstraps from the current value estimate.

## Verified Starter Commands

- `python learn/Reinforcement_Learning/policygrad/reinforce.py`
- `python learn/Reinforcement_Learning/policygrad/actorcritic/a2c.py`

## What To Watch For

- Reward variance across episodes
- The difference between sampling actions and choosing greedy actions
- Why actor-critic methods can learn faster than pure REINFORCE
