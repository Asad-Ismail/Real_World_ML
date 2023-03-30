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