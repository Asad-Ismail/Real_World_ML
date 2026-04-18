from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
RL_DIR = CURRENT_DIR.parent.parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from envs.grid_env import Gridworld

RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def softmax(x):
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


class ActorCritic:
    def __init__(self, env, gamma=0.99, actor_lr=0.01, critic_lr=0.1):
        self.env = env
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = np.zeros((env.size, env.size, env.action_space), dtype=float)
        self.critic = np.zeros((env.size, env.size), dtype=float)

    def get_action_distribution(self, state):
        x, y = state
        return softmax(self.actor[x, y])

    def get_action(self, state, rng):
        action_probs = self.get_action_distribution(state)
        action = int(rng.choice(self.env.action_space, p=action_probs))
        return action, action_probs

    def update(self, state, action, action_probs, reward, next_state, done):
        x, y = state
        nx, ny = next_state

        value = self.critic[x, y]
        next_value = 0.0 if done else self.critic[nx, ny]
        td_error = reward + self.gamma * next_value - value

        self.critic[x, y] += self.critic_lr * td_error

        action_one_hot = np.zeros(self.env.action_space, dtype=float)
        action_one_hot[action] = 1.0
        self.actor[x, y] += self.actor_lr * td_error * (action_one_hot - action_probs)

    def train(self, episodes=1000, seed=42, max_steps_per_episode=None, log_interval=100):
        rng = np.random.default_rng(seed)
        reward_history = []
        max_steps_per_episode = max_steps_per_episode or self.env.size * self.env.size * 4

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for _ in range(max_steps_per_episode):
                action, action_probs = self.get_action(state, rng)
                next_state, reward = self.env.step(action)
                done = next_state == self.env.goal

                self.update(state, action, action_probs, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if done:
                    break

            reward_history.append(episode_reward)
            if (episode + 1) % log_interval == 0 or episode == 0:
                print(f"Episode {episode + 1}/{episodes} reward: {episode_reward}")
        return reward_history

    def predict(self, state):
        x, y = state
        return int(np.argmax(self.actor[x, y]))


def evaluate_policy(env, model, episodes=100, max_steps_per_episode=None):
    max_steps_per_episode = max_steps_per_episode or env.size * env.size * 4
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(max_steps_per_episode):
            action = model.predict(state)
            state, reward = env.step(action)
            episode_reward += reward
            if state == env.goal:
                break
        total_reward += episode_reward

    return total_reward / episodes


def plot_rewards(reward_history, save_path):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(reward_history) + 1), reward_history)
    ax.set_title("A2C Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_results_dir()
    env = Gridworld()
    episodes = 2000
    actor_lr = 0.01
    critic_lr = 0.1
    gamma = 0.99

    model = ActorCritic(env, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr)
    reward_history = model.train(episodes)
    average_reward = evaluate_policy(env, model)
    plot_rewards(reward_history, RESULTS_DIR / "a2c_rewards.png")
    print(f"Average reward during testing: {average_reward}")
