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
EPS = 0.01
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
lander_env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = model.Environment(lander_env)


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

# Defining the loss function and one optimizer per policy_net
optimizers = []
for policy_net in policy_nets:
    optimizer = optim.Adam(policy_net.parameters(), lr=LR) # Weight decay is L2 regularization
    optimizers.append(optimizer)

dqn_loss = model.DQN_Loss()

# Creating one replay memory per policy_net
replay_memories = []
for _ in range(len(policy_nets)):
    replay_memory = model.ReplayMemory()
    replay_memories.append(replay_memory)



# Main function
for i_episode in range(num_episodes):
    # Initialize the environment and get a list of the states of each tree in the multitree.
    states = env.reset()
    # Metrics
    episode_return = 0
    episode_steps = 0
    loop = tqdm(range(num_steps))
    for t in loop:
        loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")   
        actions = model.select_action(states, EPS, policy_nets)

        # env.step() returns all updated state vectors and the reward and done status upon applying the actions from all individual trees
        # to the current state. We store all of these transitions in the replay buffer but only progress to the next state in the second MDP
        # as per the action picked by softmax action selection
        next_state, reward, done = env.step(actions)
        for i in range(len(replay_memories)):
            replay_memory.append(states[i], actions[i], reward[i], next_state[i])

        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the policy network)
        losses = model.optimize_model(optimizers, policy_nets, target_nets, replay_memories, dqn_loss, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA)       
        
        # Logging step level metrics
        episode_return += reward
        episode_steps = t
        for i in range(len(losses)):
            writer.add_scalar(f"Loss: Policy Net {i} vs Total Steps (across all episodes)", loss[i], total_steps)
        total_steps += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for i in range(len(policy_nets)):
            policy_net = policy_nets[i]
            target_net = target_nets[i]
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            # print("Completed one step of soft update")
        
            if (total_steps % save_checkpoint) == 0:
                torch.save(policy_net.state_dict(), save_path + f"_NET{i}")

        if done:
            break
        # print("--------------")

    # Logging episode level metrics
    writer.add_scalar("Num Steps vs Episode", episode_steps, i_episode)
    writer.add_scalar("Total Episode Return vs Episode", episode_return, i_episode)

writer.close()
torch.save(policy_net.state_dict(), save_path)

print('Complete')