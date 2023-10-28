import numpy as np

# Define the states
states = ["Sunny", "Rainy"]

# Transition matrix
# M[i][j] is the probability of transitioning from state i to state j
M = np.array([[0.8, 0.2],  # from Sunny to [Sunny, Rainy]
              [0.4, 0.6]]) # from Rainy to [Sunny, Rainy]

def next_state(current_state):
    """Return the next state based on the current state and the transition matrix."""
    return np.random.choice(states, p=M[states.index(current_state)])

# Starting state is Sunny
current_state = "Sunny"

# Simulate for 10 days
for _ in range(10):
    print(current_state)
    current_state = next_state(current_state)
