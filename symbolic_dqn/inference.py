import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import psutil
import gymnasium as gym

import model # Import the classes and functions defined in model.py
from actions import node_vectors, node_instances, node_indices, node_vector_dim, add_feature_nodes

# Setting up a device
print(f"Is GPU available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# device = "cpu"


# Defining model hyper-parameters
# BATCH_SIZE is not used during inference. But it is kept here as it is an argument to the policy network in some cases (does not affect inference)
# GAMMA is the discount factor
# EPS is the epsilon greedy exploration probability
# TAU is the update rate of the target network
# LR is the learning rate of the optimizer
BATCH_SIZE = 32
GAMMA = 0.9
EPS = 0.05
num_steps = 100
save_path = f"saved_models/batch_size_32_gamma_0.9_eps_0.05_tau_0.005_lr_1e-05_episodes_200_steps_100_run_HP_combo_1"



# Creating the environment (this may take a few minutes) and setting up the data sampling iterator
lander_env = gym.make("LunarLander-v2", render_mode="rgb_array")

node_vectors, node_instances, node_indices = add_feature_nodes(node_vectors, node_instances, node_indices, lander_env)
env = model.Environment(lander_env, node_vectors, node_instances, node_vector_dim, main_env_steps_per_first_env_step=1)


# Number of observation features in the state vector of each tree. The state is the (current pre-order traversal + empty space for a complete binary tree) X dimensionality of each token's vector
n_observation_feats = (2**env.tree_depth - 1) * node_vector_dim
n_actions = len(node_vectors)

# Defining one Q networks per lunar lander env action
policy_nets = []
for i in range(lander_env.action_space.n):
	policy_net = model.DQN(n_observation_feats, n_actions, BATCH_SIZE).to(device)
	# policy_net = policy_net.float()
	policy_net.load_state_dict(torch.load(save_path + f"_NET{i}" + ".pt"))

	policy_nets.append(policy_net)


# dqn_loss = model.DQN_Loss()


def run_policy_inference():
	'''
	Use the learnt policies from save_path to generate a multitree. Returns the multi-tree and its preorder traversal
	'''
	# Initialize the environment and get a list of the states of each tree in the multitree.
	states = env.reset()
	# Metrics
	# episode_return = np.array([0,0,0,0])
	# episode_steps = 0
	loop = tqdm(range(num_steps))
	for t in loop:
		loop.set_description(f"Episode Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")   
		actions = model.select_action(states, EPS, policy_nets, node_instances)

		'''
		- env.step() returns all updated state vectors and the reward and done status upon applying the actions from all individual trees
		to the current state. We store all of these transitions in the replay buffer but only progress to the next state in the second MDP
		as per the action picked by softmax action selection. done is a boolean indicating if the first environment was completed. 
		- Individual replay buffers are filled with the state, action (operation addition), reward (from deep copies), and next state if those trees were not already full
		- The env is done if no more state additions are possible, or if the main_env is done.
		'''
		next_states, rewards, done, tree_full_before_step = env.step(actions)
		# rewards += t # Adding a small reward if the tree progresses further in the episode

		# Move to the next state
		states = next_states

		if done:
			# print("--------------")
			break
		# print("--------------")

	for trav in env.state.multitree_preorder_travs:
		print(trav)

	return env.state.multitree



multitree = run_policy_inference()