import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import psutil
import gymnasium as gym

import model # Import the classes and functions defined in model.py
from actions import node_vectors, node_instances, node_indices, node_vector_dim, add_feature_nodes
main_env = gym.make("LunarLander-v2", render_mode="rgb_array")
node_vectors, node_instances, node_indices = add_feature_nodes(node_vectors, node_instances, node_indices, main_env)
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
num_episodes = 10
num_steps = 500
save_checkpoint = 100 # save the model after these many steps
RUN_NAME = "HP_combo_1"
#logdir = f"runs/batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}"
logdir = f"runs/testing"
save_path = f"saved_models/batch_size:{BATCH_SIZE}_|gamma:{GAMMA}_|eps:{EPS}_|tau:{TAU}_|lr:{LR}_|episodes:{num_episodes}_|steps:{num_steps}_|run:{RUN_NAME}.pt"

# Setting up the tensorboard summary writer
writer = SummaryWriter(log_dir=logdir)

# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
lander_env = gym.make("LunarLander-v2", render_mode="rgb_array")

node_vectors, node_instances, node_indices = add_feature_nodes(node_vectors, node_instances, node_indices, lander_env)
#print("node indices",node_indices)
env = model.Environment(lander_env, node_vectors, node_instances, node_vector_dim)


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

# Defining the loss function and one optimizer per policy_net
optimizers = []
for policy_net in policy_nets:
    optimizer = optim.Adam(policy_net.parameters(), lr=LR) # Weight decay is L2 regularization
    optimizers.append(optimizer)

dqn_loss = model.DQN_Loss()

# Creating one replay memory per policy_net
replay_memories = []
for _ in range(len(policy_nets)):
    replay_memory = model.ReplayMemory(10000) #added capacity
    replay_memories.append(replay_memory)



# Main function
for i_episode in range(num_episodes):
    # Initialize the environment and get a list of the states of each tree in the multitree.
    states = env.reset()
    # Metrics
    episode_return = np.array([0,0,0,0])
    episode_steps = 0
    loop = tqdm(range(num_steps))
    for t in loop:
        loop.set_description(f"Episode {i_episode} Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")   
        actions = model.select_action(states, EPS, policy_nets, node_instances)

        # env.step() returns all updated state vectors and the reward and done status upon applying the actions from all individual trees
        # to the current state. We store all of these transitions in the replay buffer but only progress to the next state in the second MDP
        # as per the action picked by softmax action selection. done is a boolean indicating if the first environment was completed. 
        # Individual replay buffers are filled with the state, action (operation addition), reward (from deep copies), and next state if those trees were not already full
        # The env is done if no more state additions are possible, or if the main_env is done. 
        next_states, rewards, done, tree_full = env.step(actions)
        #print("next states",next_states)
        #print("rewards",rewards)
        #print("done",done)
        #print("tree full",tree_full)

        #print("memories length:", len(replay_memories))
        for i in range(len(replay_memories)):
            if not tree_full[i]:
                for _ in range(BATCH_SIZE-len(replay_memories[i])): #set up temporary fix to get past empty memory
                #replay_memories[i].append(states[i], actions[i], rewards[i], next_states[i]) #specified which memory in replay memories
                    replay_memories[i].push(states[i], actions[i], rewards[i], next_states[i]) #should be the correct order to push info
        #for memory in replay_memories:
            #print("len replay mem", len(memory))
        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the policy network)
        losses = model.optimize_model(optimizers, policy_nets, target_nets, replay_memories, dqn_loss, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA)       
        
        # Logging step level metrics
        print("rewards:",rewards)
        for _ in rewards:
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
    for i in range(len(episode_return)):
        writer.add_scalar(f"Total Episode Return {i} vs Episode", episode_return[i], i_episode)

writer.close()
torch.save(policy_net.state_dict(), save_path)

print('Complete')