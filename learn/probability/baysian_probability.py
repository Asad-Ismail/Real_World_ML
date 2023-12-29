# Define the world, robot's true position, and initial belief
world = ['empty', 'wall', 'empty', 'door', 'empty']
robot_position = 2  # The robot starts at position 2
belief = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform belief about robot's location

# The robot senses 'wall'. Update belief using Bayesian rule.
def sense(belief, world, measurement):
    q = []
    for i in range(len(world)):
        hit = (measurement == world[i])  # Check if the measurement matches the landmark
        q.append(belief[i] * (hit * 0.6 + (1-hit) * 0.2))  # Assume P(hit)=0.6, P(miss)=0.2
    # Normalize the belief so it sums to 1
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

belief = sense(belief, world, 'wall')

# The robot tries to move one step to the right. But it's not always sure it moves correctly.
def move(belief, step):
    q = []
    for i in range(len(belief)):
        q.append(belief[(i-step) % len(belief)] * 0.8 + belief[i] * 0.2)  # Assume 80% chance it moves correctly
    return q

belief = move(belief, 1)

