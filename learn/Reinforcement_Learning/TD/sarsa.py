"""
SARSA is an on-policy algorithm, meaning it learns the action-value function Q(s, a) with respect to the policy that it is currently following. This means that the policy used for action selection during training is the same policy being improved upon.

The main components of the SARSA algorithm are:

Action-value function Q(s, a): The function that estimates the expected cumulative reward when taking action a in state s and following the current policy thereafter.
Policy: The strategy used by the agent to select actions based on the current state. In SARSA, the policy is usually derived from the action-value function Q(s, a) using methods like epsilon-greedy exploration.
TD error: The difference between the estimated action-value for the current state-action pair (Q(s, a)) and the updated estimate based on the observed reward and next state-action pair (r + γ * Q(s', a')).
Learning rate (α): Controls the step size of the Q-value updates. A smaller learning rate makes the updates more conservative, while a larger learning rate makes the updates more aggressive.
Discount factor (γ): Determines the importance of future rewards compared to immediate rewards. A value closer to 0 makes the agent focus on immediate rewards, while a value closer to 1 makes the agent consider long-term rewards.
"""

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


def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[-1]))
    return int(np.argmax(Q[state]))


def sarsa(env, episodes, alpha, gamma, epsilon, seed=42, max_steps_per_episode=None, log_interval=100):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.size, env.size, env.action_space), dtype=float)
    reward_history = []
    max_steps_per_episode = max_steps_per_episode or env.size * env.size * 4

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, rng)
        episode_reward = 0

        for _ in range(max_steps_per_episode):
            next_state, reward = env.step(action)
            episode_reward += reward
            next_action = epsilon_greedy(Q, next_state, epsilon, rng)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            if state == env.goal:
                break

        reward_history.append(episode_reward)
        if (episode + 1) % log_interval == 0 or episode == 0:
            print(f"Episode {episode + 1}/{episodes} reward: {episode_reward}")
    return Q, reward_history


def evaluate_q_policy(env, Q, episodes=100, max_steps_per_episode=None):
    max_steps_per_episode = max_steps_per_episode or env.size * env.size * 4
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(max_steps_per_episode):
            action = int(np.argmax(Q[state]))
            state, reward = env.step(action)
            episode_reward += reward
            if state == env.goal:
                break
        total_reward += episode_reward

    return total_reward / episodes


def plot_rewards(reward_history, save_path):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(reward_history) + 1), reward_history)
    ax.set_title("SARSA Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ensure_results_dir()
    env = Gridworld()
    episodes = 1000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    Q, reward_history = sarsa(env, episodes, alpha, gamma, epsilon)
    average_reward = evaluate_q_policy(env, Q)
    plot_rewards(reward_history, RESULTS_DIR / "sarsa_rewards.png")
    print(f"Average reward during testing: {average_reward}")

