# Temporal Difference Learning

This folder contains the cleanest value-based RL examples in the repository.

## Files

- `qlearning.py`: Off-policy control
- `sarsa.py`: On-policy control

## Run Them

```bash
python learn/Reinforcement_Learning/TD/qlearning.py
python learn/Reinforcement_Learning/TD/sarsa.py
```

Each script trains on the small Gridworld environment and saves a reward curve in `learn/Reinforcement_Learning/TD/results/`.

## Study Questions

- What is the target in SARSA?
- What is the target in Q-learning?
- How does epsilon-greedy exploration affect both methods?
