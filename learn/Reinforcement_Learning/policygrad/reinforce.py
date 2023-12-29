import sys
sys.path.append("../")
from envs.grid_env import Gridworld
import numpy as np



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def state_to_index(state, grid_shape):
    x, y = state
    return x * grid_shape[1] + y

def state_to_one_hot(state, grid_shape=(4,4)):
    index = state_to_index(state, grid_shape)
    one_hot = np.zeros(np.prod(grid_shape), dtype=int)
    one_hot[index] = 1
    return one_hot

def policy(state, theta):
    return softmax(np.dot(state, theta))

def reinforce(env, episodes, alpha, gamma):
    n_actions = 4
    theta = np.zeros((env.size * env.size, n_actions))

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []

        while state != env.goal:
            one_hot_state = state_to_one_hot(state)
            probs = policy(one_hot_state, theta)
            action = np.random.choice(n_actions, p=probs)
            next_state, reward = env.step(action)
            states.append(one_hot_state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        print(f"Reward of episode is {sum(rewards)}")
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            one_hot_state = states[t]
            action = actions[t]
            theta[:, action] += alpha * G * (one_hot_state - policy(one_hot_state, theta)[action] * one_hot_state)

    return theta

if __name__ == "__main__":
    env = Gridworld()
    episodes = 5000
    alpha = 0.001
    gamma = 0.99

    theta = reinforce(env, episodes, alpha, gamma)

    # Print the learned policy parameters
    print("Learned policy parameters:")
    print(theta)

    # Test the learned policy
    test_episodes = 100
    total_reward = 0

    for _ in range(test_episodes):
        state = env.reset()
        episode_reward = 0
        while state != env.goal:
            one_hot_state = state_to_one_hot(state)
            action_probs = policy(one_hot_state, theta)
            action = np.argmax(action_probs)
            state, reward = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    average_reward = total_reward / test_episodes
    print(f"Average reward during testing: {average_reward}")
