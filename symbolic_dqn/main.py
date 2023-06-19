import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import psutil
import gymnasium as gym

import model # Import the classes and functions defined in model.py
from actions import node_vectors, node_instances, node_indices, node_vector_dim, add_feature_nodes
from torch.utils.tensorboard import SummaryWriter
import configparser
from utils import early_stopping, ewma

# Reading the configuration file to get training params
config = configparser.ConfigParser()
config.read('GP_symbolic_DQN_config.ini')
params = {}
for each_section in config.sections():
    for (each_key, each_val) in config.items(each_section):
        params[each_key] = each_val


# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining model hyper-parameters
# BATCH_SIZE is the number of transitions sampled from the replay memory. A batch of inputs is sampled and fed through the optimizer when training the policy network
# GAMMA is the discount factor
# EPS is the epsilon greedy exploration probability
# TAU is the update rate of the target network
# LR is the learning rate of the optimizer
BATCH_SIZE = int(params['batch_size'])
GAMMA = float(params['gamma'])
EPS = float(params['eps'])
TAU = float(params['tau'])
LR = float(params['lr'])
num_episodes = int(params['num_episodes'])
num_steps = int(params['num_steps'])
save_checkpoint = int(params['save_checkpoint']) # save the model after these many steps
main_env_steps_per_first_env_step = int(params['main_env_steps_per_first_env_step'])

RUN_NAME = params['run_name']
logdir = f"runs/batch_size_{BATCH_SIZE}_gamma_{GAMMA}_eps_{EPS}_tau_{TAU}_lr_{LR}_episodes_{num_episodes}_steps_{num_steps}_run_{RUN_NAME}"
save_path = f"saved_models/batch_size_{BATCH_SIZE}_gamma_{GAMMA}_eps_{EPS}_tau_{TAU}_lr_{LR}_episodes_{num_episodes}_steps_{num_steps}_run_{RUN_NAME}"
# config.set('GP and Inference', 'model_path', save_path)

# Setting up the tensorboard summary writer
writer = SummaryWriter(log_dir=logdir)

# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
lander_env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Adding the main env's features to node vectors, instances, and indices
node_vectors, node_instances, node_indices = add_feature_nodes(node_vectors, node_instances, node_indices, lander_env)

# Creating the MDP1 env
env = model.Environment(lander_env, node_vectors, node_instances, node_vector_dim, main_env_steps_per_first_env_step=main_env_steps_per_first_env_step)


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

# Defining the loss
dqn_loss = model.DQN_Loss()

# Creating one replay memory per policy_net
replay_memories = []
for _ in range(len(policy_nets)):
    replay_memory = model.ReplayMemory(10000) #added capacity
    replay_memories.append(replay_memory)


total_steps = 0
early_stop = False
episode_return_ewma = 0.0
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

        '''
        - env.step() returns all updated state vectors and the reward and done status upon applying the actions from all individual trees
        to the current state.
        - done is a boolean indicating if the first environment was completed. 
        - Individual replay buffers are filled with the state, action (operation addition), reward (from deep copies), and next state if those trees were not already full
        '''
        next_states, rewards, done, tree_full_before_step = env.step(actions)
        rewards += t # Adding a small reward if the tree progresses further in the episode

        for i in range(len(replay_memories)):
            if not tree_full_before_step[i]:
                replay_memories[i].push(states[i], actions[i], rewards[i], next_states[i]) #should be the correct order to push info
        
        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the policy network)
        losses = model.optimize_model(optimizers, policy_nets, target_nets, replay_memories, dqn_loss, node_indices, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA)       
        
        # Logging step level metrics
        episode_return += rewards
        
        episode_steps = t
        for i in range(len(losses)):
           writer.add_scalar(f"Loss: Policy Net {i} vs Total Steps (across all episodes)", losses[i], total_steps)
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
                torch.save(policy_net.state_dict(), save_path + f"_NET{i}" + ".pt")

        if done:
            # print("--------------")
            break
        # print("--------------")

    # Logging episode level metrics
    writer.add_scalar("Num Steps vs Episode", episode_steps, i_episode)
    writer.add_scalar("Sum Episode Return vs Episode", episode_return.sum(), i_episode)
    for i in range(len(episode_return)):
        writer.add_scalar(f"Episode Return {i} vs Episode", episode_return[i], i_episode)

    # Easrly stopping 
    episode_return_ewma = ewma(episode_return.sum(), episode_return_ewma, i_episode)
    if i_episode > 10:
        if early_stopping(episode_return_ewma, threshold=0, tolerance = 1):
            print(f"Rewards seem to be consistently above the threshold. Episode return EWMA: {episode_return_ewma}. Stopping now at episode {i_episode}")
            early_stop = True
            for i in range(len(episode_return)):
                writer.add_scalar(f"Total Episode Return {i} vs Episode", episode_return[i], i_episode)

            break

writer.close()

# If no early stopping was done, save model
if not early_stop:  
    for i in range(len(policy_nets)):
        torch.save(policy_net.state_dict(), save_path + f"_NET{i}" + ".pt")

print('Complete')