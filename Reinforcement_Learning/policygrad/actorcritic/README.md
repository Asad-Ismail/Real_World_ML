
# A2C
A2C, or Advantage Actor-Critic, is a type of actor-critic method used in reinforcement learning. The A2C method aims to address the problem of determining the best action to take in a given state, as per a given policy.

Here's a basic breakdown of the different components involved in A2C:

Actor: The actor is the component that decides which action to take, based on the current policy. The actor's policy is often represented by a neural network that takes a state as input and outputs a probability distribution over the possible actions.

Critic: The critic evaluates the action taken by the actor and gives feedback. It estimates the value function (expected return) of being in a state and taking an action according to the current policy.

Advantage: The advantage function measures how much better an action is compared to the average action in that state, as estimated by the critic. It's calculated as the difference between the estimated value of taking an action in a state and the average value of that state.

In A2C, the actor uses feedback from the critic to update its policy. In particular, it tries to increase the probability of actions that have a higher advantage and decrease the probability of actions with a lower advantage. The critic, on the other hand, uses the Temporal Difference (TD) error (the difference between the estimated and actual return) to update its value function.

A2C is an on-policy method, which means that it learns the value of the policy that is currently being used to make decisions. This is in contrast to off-policy methods, which learn about the optimal policy independently of the policy that is currently being used to make decisions.

# A3C

Asynchronous Advantage Actor-Critic (A3C) is a reinforcement learning algorithm that is an extension of the standard Advantage Actor-Critic (A2C) method. A3C was proposed by Volodymyr Mnih et al. in their paper "Asynchronous Methods for Deep Reinforcement Learning" in 2016. The main idea behind A3C is to use multiple parallel actor-critic agents, each with their own copy of the model parameters, to interact with their own environment. This allows the algorithm to learn more efficiently and explore more diverse experiences.

The key components of A3C are:

Parallelism: Multiple agents run in parallel, each interacting with their own copy of the environment. This parallelism helps with exploration and provides diverse experiences, which can lead to better and more stable learning.

Asynchronous updates: Each agent updates the global model parameters independently and asynchronously. This prevents any one agent from dominating the learning process and allows the algorithm to make use of the most recent experiences from all agents.

Advantage estimation: Similar to A2C, A3C uses advantage estimation to update the policy. The advantage is the difference between the estimated value of taking an action in a state and the average value of that state. It measures how much better an action is compared to the average action in that state.

The learning process in A3C consists of the following steps:

Each agent initializes its local model parameters with the global model parameters.

The agent interacts with the environment, collects a sequence of experiences (states, actions, rewards), and calculates the advantage using these experiences.

The agent updates its local model parameters using the calculated advantages and the gradient of the policy and value functions.

The agent asynchronously updates the global model parameters using its local model parameters.

The agent then synchronizes its local model parameters with the updated global model parameters and continues the process.

A3C has been shown to achieve better performance and faster convergence compared to other methods like Deep Q-Network (DQN) and vanilla A2C. However, it should be noted that more recent algorithms like Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) have demonstrated even better performance and stability in various tasks so we will not implement A3C