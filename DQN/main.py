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
env = model.Environment()

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
dqn_loss = model.DQN_Loss()




# Main function
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    # Metrics
    episode_return = 0
    episode_steps = 0
    loop = tqdm(range(num_steps))
    for t in loop:
        loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")   
        action = model.select_action(torch.reshape(torch.tensor(state, dtype=torch.float32), (1,-1)), EPS, policy_net)

        
        next_state, reward, done = env.step(action)
        replay_memory.append(state, action, reward, next_state)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = model.optimize_model(optimizer, policy_net, target_net, replay_memory, dqn_loss, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA)       
        
        # Logging step level metrics
        episode_return += reward
        episode_steps = t
        writer.add_scalar("Loss vs Total Steps (all episodes)", loss, total_steps)
        total_steps += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        # print("Completed one step of soft update")
        
        if (total_steps % save_checkpoint) == 0:
            torch.save(policy_net.state_dict(), save_path)

        if done:
            break
        # print("--------------")

    # Logging episode level metrics
    writer.add_scalar("Num Steps vs Episode", episode_steps, i_episode)
    writer.add_scalar("Total Episode Return vs Episode", episode_return, i_episode)

writer.close()
torch.save(policy_net.state_dict(), save_path)

print('Complete')