import torch
import torch.optim as optim
import cv2
import numpy as np
from tqdm import tqdm
import psutil
import gym

import model # Import the classes and functions defined in model.py
from actions import actions as action_list
from torch.utils.tensorboard import SummaryWriter

# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# Defining model hyper-parameters
# BATCH_SIZE is the number of transitions sampled from the replay memory. A batch of inputs is sampled and fed through the optimizer when training the policy network
# GAMMA is the discount factor
# EPS is the epsilon greedy exploration probability
# TAU is the update rate of the target network
# LR is the learning rate of the optimizer
BATCH_SIZE = 32
GAMMA = 0.99
EPS = 0.05
TAU = 0.005
LR = 1e-4
num_episodes = 20
num_steps = 1500
save_checkpoint = 500 # save the model after these many steps
RUN_NAME = "HP_combo_1"
logdir = f"runs/frame_stack:{FRAME_STACK}_|batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}"
save_path = f"saved_models/frame_stack:{FRAME_STACK}_|batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}.pt"

# Setting up the tensorboard summary writer
writer = SummaryWriter(log_dir=logdir)

# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
env = gym.make('MineRLTreechop-v0')
print("Gym.make done")

replay_memory = model.ReplayMemory(5000)
print("Replay memory & demo replay memory initialized")


# Defining the simple model Q networks
policy_net = model.DQfD
(n_observation_feats, n_actions, BATCH_SIZE).to(device)
policy_net = policy_net.float()
target_net = model.DQfD(n_observation_feats, n_actions, BATCH_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Defining the loss function and optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-5) # Weight decay is L2 regularization
dqfd_loss = model.DQfD_Loss()

