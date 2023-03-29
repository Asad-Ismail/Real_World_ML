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

def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[state])

def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.size, env.size, 4))

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        while state != env.goal:
            next_state, reward = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            
            state = next_state
            action = next_action

    return Q

if __name__ == "__main__":
    env = Gridworld()
    episodes = 5000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    Q = sarsa(env, episodes, alpha, gamma, epsilon)
    
    # Print the learned Q-values
    print("Learned Q-values:")
    print(Q)

    # Test the learned policy
    test_episodes = 100
    total_reward = 0

    for _ in range(test_episodes):
        state = env.reset()
        episode_reward = 0
        while state != env.goal:
            action = np.argmax(Q[state])
            state, reward = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward

    average_reward = total_reward / test_episodes
    print(f"Average reward during testing: {average_reward}")


