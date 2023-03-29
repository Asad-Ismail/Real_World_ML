"""
SARSA is an on-policy algorithm, meaning it learns the action-value function Q(s, a) with respect to the policy that it is currently following. This means that the policy used for action selection during training is the same policy being improved upon.

The main components of the SARSA algorithm are:

Action-value function Q(s, a): The function that estimates the expected cumulative reward when taking action a in state s and following the current policy thereafter.
Policy: The strategy used by the agent to select actions based on the current state. In SARSA, the policy is usually derived from the action-value function Q(s, a) using methods like epsilon-greedy exploration.
TD error: The difference between the estimated action-value for the current state-action pair (Q(s, a)) and the updated estimate based on the observed reward and next state-action pair (r + γ * Q(s', a')).
Learning rate (α): Controls the step size of the Q-value updates. A smaller learning rate makes the updates more conservative, while a larger learning rate makes the updates more aggressive.
Discount factor (γ): Determines the importance of future rewards compared to immediate rewards. A value closer to 0 makes the agent focus on immediate rewards, while a value closer to 1 makes the agent consider long-term rewards.
"""

import numpy as np

class Gridworld:
    def __init__(self, size=4, start=(0, 0), goal=(3, 3)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start

    def step(self, action):
        x, y = self.state

        if action == 0:  # up
            y = max(y - 1, 0)
        elif action == 1:  # right
            x = min(x + 1, self.size - 1)
        elif action == 2:  # down
            y = min(y + 1, self.size - 1)
        elif action == 3:  # left
            x = max(x - 1, 0)

        self.state = (x, y)

        reward = -1
        if self.state == self.goal:
            reward = 0

        return self.state, reward

    def reset(self):
        self.state = self.start
        return self.state


