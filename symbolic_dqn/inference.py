import sys
sys.path.append("../")
import torch
import numpy as np
from tqdm import tqdm
import psutil
import gymnasium as gym
import pickle
import model # Import the classes and functions defined in model.py
import configparser


config = configparser.ConfigParser()
config.read('GP_symbolic_DQN_config.ini')
params = {}
for each_section in config.sections():
	for (each_key, each_val) in config.items(each_section):
		params[each_key] = each_val



def neural_guided_multitrees(population_size, save_path, EPS=0.2, num_steps=100, print_preorder_trav=False):
	'''
	Generator function to use the learnt policies from save_path to generate a multitree. Returns the multi-tree and prints its preorder traversal
	population_size: int - number of multitrees to return
	save_path: string - path to the saved model.pt file
	'''

	# Importing within the function to avoid passing a long list of arguments. Python caches imported modules so this does not slow things down.
	# This also allows any other script to import this script and use these imported modules as long as its in the Python path
	from actions import node_vectors, node_instances, node_indices, node_vector_dim, add_feature_nodes

	# Setting up a device
	# print(f"Is GPU available: {torch.cuda.is_available()}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# BATCH_SIZE is not used during inference. But it is kept here as it is an argument to the policy network in some cases (does not affect inference)
	# EPS is the epsilon greedy exploration probability. NOTE: Epsilon can be used to vary the diversity of the generated population. High EPS corresponds to more diversity
	# num_steps is the number of steps to run during inference. Larger num_steps corresponds to larger trees

	# Best to set these hyperparam to the values used while training the model being used for inference. Except for EPS
	BATCH_SIZE = 32
	EPS = EPS
	num_steps = num_steps

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

	for _ in range(population_size):
		# Initialize the environment and get a list of the states of each tree in the multitree.
		states = env.reset()
		# Metrics
		# episode_return = np.array([0,0,0,0])
		# episode_steps = 0
		loop = tqdm(range(num_steps))
		for t in loop:
			loop.set_description(f"Episode Steps | CPU {psutil.cpu_percent()} | RAM {psutil.virtual_memory().percent}")   
			actions = model.select_action(states, EPS, policy_nets, node_instances)

			next_states, rewards, done, tree_full_before_step = env.step(actions)

			# Move to the next state
			states = next_states

			if done:
				# print("--------------")
				break
			# print("--------------")

		if print_preorder_trav:
			for trav in env.state.multitree_preorder_travs:
				print(trav)
		print("-----")

		yield env.state.multitree


if __name__ == "__main__":
	population = []
	for multitree in neural_guided_multitrees(int(params['pop_size']), params['model_path'], EPS=0.05, num_steps=int(params['num_steps']), print_preorder_trav=False):
		population.append(multitree)

	with open(params['pop_save_path'], 'wb') as handle:
		pickle.dump(population, handle)