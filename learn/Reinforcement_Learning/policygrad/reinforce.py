from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
RL_DIR = CURRENT_DIR.parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from envs.grid_env import Gridworld

RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def softmax(x):
    shifted = x - np.max(x)
    e_x = np.exp(shifted)
    return e_x / np.sum(e_x)

def state_to_index(state, grid_shape):
    x, y = state
    return x * grid_shape[1] + y

def state_to_one_hot(state, grid_shape):
    index = state_to_index(state, grid_shape)
    one_hot = np.zeros(np.prod(grid_shape), dtype=float)
    one_hot[index] = 1
    return one_hot

def action_to_one_hot(action, n_actions):
    one_hot = np.zeros(n_actions, dtype=float)
    one_hot[action] = 1.0
    return one_hot

def policy(state, theta):
    return softmax(state @ theta)

def reinforce(env, episodes, alpha, gamma, seed=42, max_steps_per_episode=None, log_interval=100):
    rng = np.random.default_rng(seed)
    n_actions = env.action_space
    grid_shape = (env.size, env.size)
    theta = np.zeros((env.size * env.size, n_actions), dtype=float)
    reward_history = []
    max_steps_per_episode = max_steps_per_episode or env.size * env.size * 4

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []

        for _ in range(max_steps_per_episode):
            one_hot_state = state_to_one_hot(state, grid_shape)
            probs = policy(one_hot_state, theta)
            action = int(rng.choice(n_actions, p=probs))
            next_state, reward = env.step(action)
            states.append(one_hot_state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if state == env.goal:
                break

        episode_reward = float(sum(rewards))
        reward_history.append(episode_reward)

        returns = np.zeros(len(rewards), dtype=float)
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G

        for one_hot_state, action, G in zip(states, actions, returns):
            probs = policy(one_hot_state, theta)
            grad_log_policy = np.outer(one_hot_state, action_to_one_hot(action, n_actions) - probs)
            theta += alpha * G * grad_log_policy

        if (episode + 1) % log_interval == 0 or episode == 0:
            print(f"Episode {episode + 1}/{episodes} reward: {episode_reward}")

    return theta, reward_history


def evaluate_policy(env, theta, episodes=100, max_steps_per_episode=None):
    grid_shape = (env.size, env.size)
    max_steps_per_episode = max_steps_per_episode or env.size * env.size * 4
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(max_steps_per_episode):
            one_hot_state = state_to_one_hot(state, grid_shape)
            action_probs = policy(one_hot_state, theta)
            action = int(np.argmax(action_probs))
            state, reward = env.step(action)
            episode_reward += reward
            if state == env.goal:
                break
        total_reward += episode_reward

    return total_reward / episodes


def plot_rewards(reward_history, save_path):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(reward_history) + 1), reward_history)
    ax.set_title("REINFORCE Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ensure_results_dir()
    env = Gridworld()
    episodes = 2000
    alpha = 0.01
    gamma = 0.99

    theta, reward_history = reinforce(env, episodes, alpha, gamma)
    average_reward = evaluate_policy(env, theta)
    plot_rewards(reward_history, RESULTS_DIR / "reinforce_rewards.png")
    print("Learned policy parameters:")
    print(theta)
    print(f"Average reward during testing: {average_reward}")
