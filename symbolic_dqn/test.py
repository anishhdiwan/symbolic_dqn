import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import psutil
import gym

import model # Import the classes and functions defined in model.py
from actions import *
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 32
GAMMA = 0.99
EPS = 0.01
TAU = 0.005
LR = 1e-4
num_episodes = 10
num_steps = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
lander_env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = model.Environment(lander_env)


# Number of observation features in the state vector of each tree. The state is the (current pre-order traversal + empty space for a complete binary tree) X dimensionality of each token's vector
n_observation_feats = (2**env.tree_depth - 1) * node_vector_dim
n_actions = len(node_vectors)

# Defining one Q networks per lunar lander env action
policy_nets = []
target_nets = []
for _ in range(lander_env.action_space.n):
    policy_net = model.DQN(n_observation_feats, n_actions, BATCH_SIZE).to(device)
    policy_net = policy_net.float()
    target_net = model.DQN(n_observation_feats, n_actions, BATCH_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    policy_nets.append(policy_net)
    target_nets.append(target_net)


states = env.reset()
print(states[0].shape)
temp = torch.reshape(states[0], (1,-1))
print(temp.shape)

actions = model.select_action(states, EPS, policy_nets)
print(actions)
