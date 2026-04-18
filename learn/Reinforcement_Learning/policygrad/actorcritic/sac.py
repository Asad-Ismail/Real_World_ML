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


def softmax(logits):
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)


class SoftActorCritic:
    """
    Simplified discrete SAC-style learner for small environments.

    This is not the full continuous-control SAC algorithm used in modern RL
    benchmarks. It is a compact, first-principles version that keeps the key
    ideas:
    - entropy-regularized policy improvement
    - double Q estimates
    - stochastic policy updates
    """

    def __init__(self, env, gamma=0.99, alpha=0.3, q_lr=0.1, policy_lr=0.05):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.q_lr = q_lr
        self.policy_lr = policy_lr

        self.q1 = np.zeros((env.size, env.size, env.action_space), dtype=float)
        self.q2 = np.zeros((env.size, env.size, env.action_space), dtype=float)
        self.policy_logits = np.zeros((env.size, env.size, env.action_space), dtype=float)

    def policy(self, state):
        x, y = state
        return softmax(self.policy_logits[x, y])

    def soft_value(self, state):
        probs = self.policy(state)
        q_values = np.minimum(self.q1[state], self.q2[state])
        entropy_bonus = -self.alpha * np.log(probs + 1e-8)
        return float(np.sum(probs * (q_values + entropy_bonus)))

    def sample_action(self, state, rng):
        action_probs = self.policy(state)
        action = int(rng.choice(self.env.action_space, p=action_probs))
        return action, action_probs

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.soft_value(next_state)

        self.q1[state][action] += self.q_lr * (target - self.q1[state][action])
        self.q2[state][action] += self.q_lr * (target - self.q2[state][action])

        q_values = np.minimum(self.q1[state], self.q2[state])
        target_policy = softmax(q_values / max(self.alpha, 1e-6))
        current_policy = self.policy(state)
        self.policy_logits[state] += self.policy_lr * (target_policy - current_policy)

    def train(self, episodes=1500, seed=42, max_steps_per_episode=None, log_interval=100):
        rng = np.random.default_rng(seed)
        reward_history = []
        max_steps_per_episode = max_steps_per_episode or self.env.size * self.env.size * 4

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for _ in range(max_steps_per_episode):
                action, _ = self.sample_action(state, rng)
                next_state, reward = self.env.step(action)
                done = next_state == self.env.goal

                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if done:
                    break

            reward_history.append(episode_reward)
            if (episode + 1) % log_interval == 0 or episode == 0:
                print(f"Episode {episode + 1}/{episodes} reward: {episode_reward}")
        return reward_history

    def predict(self, state):
        return int(np.argmax(self.policy(state)))


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
    ax.set_title("Simplified SAC Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_results_dir()
    env = Gridworld()

    model = SoftActorCritic(env, gamma=0.99, alpha=0.3, q_lr=0.1, policy_lr=0.05)
    reward_history = model.train(episodes=1500)
    average_reward = evaluate_policy(env, model)
    plot_rewards(reward_history, RESULTS_DIR / "sac_rewards.png")

    print(f"Average reward during testing: {average_reward}")
