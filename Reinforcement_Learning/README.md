## Quitesential Implementation of RL algorithms from scratch

We will learn and implement these concepts/methods

1. Fundamental Concepts:

    Markov Decision Processes (MDPs)
    Value functions (state-value and action-value functions)
    Bellman equations
    Exploration vs. Exploitation trade-off

2. Dynamic Programming (DP) methods:

    Policy iteration
    Value iteration

3. Model-free methods:

    a. Temporal-Difference (TD) Learning:
    
        - SARSA (on-policy)
        - Q-learning (off-policy)

    b. Policy Gradient methods:

        REINFORCE
        Actor-Critic algorithms:
        Advantage Actor-Critic (A2C)
        Asynchronous Advantage Actor-Critic (A3C)
        Soft Actor-Critic (SAC)

    c. Deep Reinforcement Learning:

        Deep Q-Networks (DQN)
        Double Deep Q-Networks (DDQN)
        Dueling Deep Q-Networks (Dueling DQN)
        Proximal Policy Optimization (PPO)

4. Model-based methods:

    Monte Carlo Tree Search (MCTS)
    Dyna-Q
    Model-based planning with learned models

5.Inverse Reinforcement Learning (IRL):

    Maximum Entropy IRL
    Apprenticeship Learning
    Bayesian IRL
    Generative Adversarial Imitation Learning (GAIL)


Multi-Agent Reinforcement Learning:

    Independent Q-Learning (IQL)
    Coordinated Reinforcement Learning
    Centralized Training with Decentralized Execution (CTDE)
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

Exploration Techniques:

    Epsilon-greedy exploration
    Boltzmann exploration (Softmax)
    Upper Confidence Bound (UCB)
    Thompson Sampling
    Intrinsic motivation (Curiosity-driven exploration)

Hierarchical Reinforcement Learning:

    Options framework
    Hierarchical Abstract Machines (HAMs)
    MAXQ framework
    Feudal Networks

Transfer Learning and Domain Adaptation:

    Progressive Networks
    Distillation methods
    Meta-learning techniques



## Exaplantation 

Here are examples of on-policy and off-policy reinforcement learning algorithms, illustrated using the classic Gridworld problem.

Gridworld is a simple environment where an agent navigates a grid to reach a goal state while avoiding obstacles. The agent receives a reward for each action taken, with positive rewards for reaching the goal and negative rewards for hitting obstacles or taking too long to reach the goal.

On-policy example: SARSA
Suppose we have a 4x4 grid with a starting point, a goal point, and some obstacles. The agent follows an epsilon-greedy policy, meaning it chooses the action with the highest estimated value (Q-value) most of the time and occasionally selects a random action for exploration. SARSA is an on-policy algorithm, so the agent learns while following the same epsilon-greedy policy.

During training, the agent updates its Q-value estimates using the SARSA update rule:

Q(s, a) ← Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a))

where s and a are the current state and action, r is the immediate reward, s' and a' are the next state and action, α is the learning rate, and γ is the discount factor. Since SARSA is on-policy, both the current action (a) and next action (a') are selected using the epsilon-greedy policy.

Off-policy example: Q-learning:

In the same Gridworld problem, an off-policy algorithm like Q-learning can be used. In Q-learning, the agent still follows an epsilon-greedy policy for action selection and exploration, but it learns a separate policy based on the maximum Q-value for the next state.

During training, the agent updates its Q-value estimates using the Q-learning update rule:

Q(s, a) ← Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))

where s and a are the current state and action, r is the immediate reward, s' is the next state, α is the learning rate, and γ is the discount factor. In Q-learning, the next action a' is not explicitly selected; instead, the maximum Q-value for the next state (max_a' Q(s', a')) is used.

Since Q-learning is an off-policy algorithm, the agent learns an optimal policy independent of the behavior policy (epsilon-greedy in this case). This decoupling allows Q-learning to learn the optimal policy even when the behavior policy is more exploratory or suboptimal.

In summary, SARSA is an example of an on-policy algorithm where the agent learns while following the same policy used for action selection, whereas Q-learning is an example of an off-policy algorithm where the agent learns an optimal policy separate from the policy used for action selection and exploration.