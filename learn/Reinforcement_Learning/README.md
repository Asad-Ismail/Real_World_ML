# Reinforcement Learning Study Path

This section is best approached in a narrow order. Start with the tiny Gridworld environment and move from value-based methods to policy-based methods.

## Recommended Order

1. `envs/grid_env.py`
2. `TD/qlearning.py`
3. `TD/sarsa.py`
4. `policygrad/reinforce.py`
5. `policygrad/actorcritic/a2c.py`

## What Each Step Teaches

- `grid_env.py`: The state, action, transition, and reward loop.
- `qlearning.py`: Off-policy value learning with a greedy bootstrap target.
- `sarsa.py`: On-policy value learning with the next sampled action.
- `reinforce.py`: Monte Carlo policy gradients.
- `a2c.py`: Actor-critic learning with a learned baseline.

## Verified Starter Commands

- `python learn/Reinforcement_Learning/TD/qlearning.py`
- `python learn/Reinforcement_Learning/TD/sarsa.py`
- `python learn/Reinforcement_Learning/policygrad/reinforce.py`
- `python learn/Reinforcement_Learning/policygrad/actorcritic/a2c.py`

## Key Contrast: SARSA vs Q-Learning

- SARSA updates with the next action actually chosen by the current policy.
- Q-learning updates with the best next action under the current value estimate.
- That makes SARSA on-policy and Q-learning off-policy.

## Study Questions

- What is the Bellman target in each algorithm?
- Where does exploration enter the loop?
- Which method is updating a value function, and which is directly updating a policy?
- Why does REINFORCE wait until the episode ends?
