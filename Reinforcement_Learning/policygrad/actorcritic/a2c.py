import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("../")
from envs.grid_env import Gridworld

class ActorCritic:
    def __init__(self, env, gamma=0.99, lr=0.01):
        self.env = env
        self.gamma = gamma

        self.actor = torch.zeros(env.size, env.size, 4)
        self.critic = torch.zeros(env.size, env.size)
        
        self.actor_optimizer = optim.Adam([self.actor], lr=lr)
        self.critic_optimizer = optim.Adam([self.critic], lr=lr)
        
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        x, y = state
        action_probs = nn.Softmax(dim=0)(self.actor[x, y])
        action = torch.multinomial(action_probs, 1)
        return action.item()

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        target_value = reward + self.gamma * self.critic[nx, ny]
        predicted_value = self.critic[x, y]

        critic_loss = self.criterion(predicted_value, target_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        advantage = (reward + self.gamma * self.critic[nx, ny] - self.critic[x, y]).detach()

        self.actor_optimizer.zero_grad()
        self.actor[x, y, action].backward(-advantage)
        self.actor_optimizer.step()

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward = self.env.step(action)
                done = next_state == self.env.goal

                self.update(state, action, reward, next_state)
                state = next_state

    def predict(self, state):
        x, y = state
        action_probs = nn.Softmax(dim=0)(self.actor[x, y])
        action = torch.argmax(action_probs).item()
        return action

if __name__ == "__main__":
    env = Gridworld()
    episodes = 5000
    alpha = 0.001
    gamma = 0.99

    model = ActorCritic(env, gamma, alpha)
    model.train(episodes)

    # Test the learned policy
    test_episodes = 100
    total_reward = 0

    for _ in range(test_episodes):
        state = env.reset()
        episode_reward = 0
        while state != env.goal:
            action = model.predict(state)
            state, reward = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    average_reward = total_reward / test_episodes
    print(f"Average reward during testing: {average_reward}")
