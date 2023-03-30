
from grid_env import Gridworld
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[state]) 

def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.size, env.size, 4))

    for episode in range(episodes):
        state = env.reset()

        while state != env.goal:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = env.step(action)
            next_action = np.argmax(Q[next_state])

            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state

    return Q

if __name__ == "__main__":
    env = Gridworld()
    episodes = 5000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    Q = q_learning(env, episodes, alpha, gamma, epsilon)

    # Print the learned Q-values
    print("Learned Q-values:")
    #print(Q)

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
